#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Task 3 合成数据生成Pipeline
整合基于llama生成答案和通过4o-mini打分判断的功能
"""

import os
import json
import time
import asyncio
import copy
import random
from tqdm import tqdm
from PIL import Image
import torch
import vllm
from vllm import LLM, SamplingParams
from transformers import AutoProcessor
from openai import OpenAI, AsyncOpenAI
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

# 设置环境变量

# VLLM配置
VLLM_TENSOR_PARALLEL_SIZE = 4
VLLM_GPU_MEMORY_UTILIZATION = 0.8
MAX_MODEL_LEN = 8192
MAX_NUM_SEQS = 2
MAX_GENERATION_TOKENS = 75

# 评估ID列表（用于分割训练集和验证集）
EVAL_IDS = [
    '30c7fdb3-cd41-4781-b3d2-298693c2dd08', 'e491a1d9-d970-4faf-819c-b5caaff9c3ae',
    '97bc4d6a-7c7d-4052-b4ae-32fd2c99d61e', '50a954b7-e42a-4d71-a13b-1c71bce5fc85',
    # ... 更多ID（为了简洁省略，实际使用时需要完整列表）
]

def load_jsonl(filename):
    """加载JSONL文件"""
    data = []
    with open(filename, 'r') as f:
        lines = f.readlines()
        for line in tqdm(lines, desc="加载数据"):
            data.append(json.loads(line))
    return data

def split_passage(passage, chunk_size=128):
    """分割段落"""
    passage = passage.split(" ")
    return [" ".join(passage[i:i+chunk_size]) for i in range(0, len(passage), chunk_size)]

def find_most_relevant_chunk(query, passage, chunk_size=100):
    """找到最相关的文本块"""
    chunks = split_passage(passage, chunk_size)
    corpus = [query] + chunks
    
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(corpus)
    similarities = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:]).flatten()
    
    most_relevant_index = np.argmax(similarities)
    most_relevant_chunk = chunks[most_relevant_index]
    
    return most_relevant_chunk, similarities[most_relevant_index]

def split_chunk(query, rag_infos):
    """分割并选择最相关的文本块"""
    passages = rag_infos
    new_passages = []
    for passage in passages:
        max_chunk_size = 2 * (7000 // len(passages)) // 3
        result_chunk, score = find_most_relevant_chunk(query.lower(), passage[9:].lower(), chunk_size=max_chunk_size)
        new_passages.append(result_chunk)
    return [f"{new_passages[i]}" for i in range(len(new_passages))]

def get_system_message() -> str:
    """返回评估器的系统消息"""
    return (
        "You are an expert evaluator for question answering systems. "
        "Your task is to determine if a prediction correctly answers a question based on the ground truth.\n\n"
        "Rules:\n"
        "1. The prediction is correct if it captures all the key information from the ground truth.\n"
        "2. The prediction is correct even if phrased differently as long as the meaning is the same.\n"
        "3. The prediction is incorrect if it contains incorrect information or is missing essential details.\n"
        "Output a JSON object with a single field 'accuracy' whose value is true or false."
    )

def attempt_api_call(client, model_name, messages, max_retries=3):
    """尝试API调用，带重试机制"""
    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                model=model_name,
                messages=messages,
                response_format={"type": "json_object"},
            )
            content = response.choices[0].message.content
            try:
                result_json = json.loads(content)
                accuracy = result_json.get("accuracy", False)
                return {"accuracy": accuracy, "raw": content}
            except json.JSONDecodeError:
                if attempt == max_retries - 1:
                    return {"accuracy": False, "raw": content}
        except Exception as e:
            time.sleep(30)
            if attempt == max_retries - 1:
                print(f"Failed after {max_retries} attempts: {str(e)}")
    return None

def evaluate_response(examples, eval_model_name="gpt-4o-mini") -> dict:
    """评估响应质量"""
    agent_response = str(examples["agent_response"])
    ground_truth = str(examples["ground_truth"])
    query = str(examples["query"])
    
    is_idk = "i don't know" in agent_response.lower()
    is_exact_match = agent_response.strip().lower() == ground_truth.strip().lower()
    is_semantically_correct = False
    api_response = None

    is_correct = is_exact_match

    if not is_idk and not is_exact_match and eval_model_name:
        local_openai_client = OpenAI(
            api_key = YOUR_KEY,
            base_url = YOUR_BASE_URL
        )

        messages = [
            {"role": "system", "content": get_system_message()},
            {
                "role": "user",
                "content": f"Question: {query}\nGround truth: {ground_truth}\nPrediction: {agent_response}\n",
            },
        ]

        api_response = attempt_api_call(local_openai_client, eval_model_name, messages)
        if api_response:
            is_semantically_correct = api_response["accuracy"]
            is_correct = is_semantically_correct
    
    if is_exact_match:
        is_semantically_correct = True

    return {
        "agent_response": agent_response,
        "ground_truth": ground_truth,
        "query": query,
        "is_exact_match": is_exact_match,
        "is_correct": is_correct,
        "is_miss": is_idk,
        "is_semantically_correct": is_semantically_correct,
        "api_response": api_response,
    }

def extract_multiple_answer(agent_responses, ground_truth, query):
    """提取多个答案中的最佳答案"""
    score = evaluate_response({
        "agent_response": agent_responses, 
        "ground_truth": ground_truth, 
        "query": query
    })['is_semantically_correct']
    return score, agent_responses

def extract_whole_sentence(input_str):
    """提取完整句子"""
    input_list = input_str.strip().split('.')
    input_list = input_list[:-1]
    return '.'.join(input_list) + '.'

def tokenize(image, query, tokenizer, few_shot_msg=[]):
    """为Task-1生成tokenized输入"""
    SYSTEM_PROMPT = (
        "You are a helpful assistant that truthfully answers user questions about the provided image and the retrieval information.\n"
        "# The retrieval information may not related to the provided query and image. \n"
        "# Please pay attention to identifying that information and answer the query with image. And the correct answer is satisfied following rules:\n"
        "# 1. The answer is correct if it captures all the key information.\n"
        "# 2. The answer is correct even if phrased differently as long as the meaning is the same.\n"
        "# 3. The answer is incorrect if it contains incorrect information or is missing essential details. For example, when answer a question about time, it's better to answer with day, month and year.\n"
        "# Remeber the above rules and keep your response concise and to the point. Note that the answer must in short!!!!"
    )

    messages = [
        {
            "role": "system", "content": [
                {"type": "text", "text": SYSTEM_PROMPT}
            ]
        },
    ] + [
        {
            "role": "user", "content": [
                {"type": "text", "text": "\n\nThe original provided image is:\n"},
                {"type": "image", "image": image}
            ]
        }
    ] + few_shot_msg + [
        {
            "role": "user", "content": [
                {"type": "text", "text": "\n\nPlease ask: " + query}
            ]
        }
    ]
    
    formatted_prompt = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=False
    )
    
    return {
        "prompt": formatted_prompt,
        "multi_modal_data": {
            "image": image
        }
    }

class SynthesisPipeline:
    """合成数据生成Pipeline"""
    
    def __init__(self, model_name, tokenizer_path):
        """初始化Pipeline"""
        self.model_name = model_name
        self.tokenizer_path = tokenizer_path
        self.llm = None
        self.tokenizer = None
        
    def load_models(self):
        """加载模型和tokenizer"""
        print("加载VLLM模型...")
        self.llm = LLM(
            self.model_name,
            tensor_parallel_size=VLLM_TENSOR_PARALLEL_SIZE, 
            gpu_memory_utilization=VLLM_GPU_MEMORY_UTILIZATION, 
            max_model_len=MAX_MODEL_LEN,
            max_num_seqs=MAX_NUM_SEQS,
            trust_remote_code=True,
            dtype="bfloat16",
            enforce_eager=True,
            limit_mm_per_prompt={
                "image": 1 
            },
        )
        
        print("加载tokenizer...")
        self.tokenizer = AutoProcessor.from_pretrained(self.tokenizer_path)
        
    def generate_answers(self, data_path, output_path):
        """生成答案"""
        print("开始生成答案...")
        
        # 加载数据
        data = []
        eval_inputs = []
        
        with open(data_path, 'r') as f:
            lines = f.readlines()
            for line in tqdm(lines, desc="处理数据"):
                line = json.loads(line)
                
                # 加载和预处理图片
                image = Image.open(os.path.join(
                    "/dataset/v0.1.2/comb/images", 
                    line['image']
                ))
                image = image.resize((960, 1280))

                query = line['query']
                if 'question' in line:
                    query = line['question']
                line['query'] = query
                data.append(line)

                # 生成tokenized输入
                eval_inputs.append(tokenize(image, query, self.tokenizer))
        
        print(f"处理了 {len(data)} 条数据")
        
        # 生成答案
        print("使用VLLM生成答案...")
        responses = self.llm.generate(
            eval_inputs * 20,  # 生成多个版本
            sampling_params=SamplingParams(
                temperature=0.01,
                top_p=0.9,
                max_tokens=MAX_GENERATION_TOKENS,
                skip_special_tokens=True
            ) 
        )
        
        # 保存结果
        saved_data = []
        for i in range(len(data)):
            row = data[i]
            row1 = copy.deepcopy(row)
            generated_text = responses[i].outputs[0].text
            row1['llama-3.2-11-pred'] = generated_text
            saved_data.append(row1)
        
        # 保存到文件
        with open(output_path, 'w') as f:
            for line in saved_data:
                f.write(json.dumps(line) + '\n')
        
        print(f"答案生成完成，保存到: {output_path}")
        return saved_data
    
    def evaluate_and_filter(self, data_path, eval_ids=None):
        """评估和过滤数据"""
        print("开始评估和过滤数据...")
        
        if eval_ids is None:
            eval_ids = EVAL_IDS
            
        data = load_jsonl(data_path)
        
        cnt = 0
        saved_data = []
        train_data, eval_data = [], []
        
        for row in tqdm(data, desc="评估数据"):
            id = row['interaction_id']
            
            if id in eval_ids:
                eval_data.append(row)
            else:
                pred = row['llama-3.2-11-pred']
                pred = extract_whole_sentence(pred)

                if "i don't know" in pred.lower():
                    row['ans_full'] = "I don't know."
                else:
                    score, pred = extract_multiple_answer(pred, row['ans_full'], row['query'])
                    row['ans_full'] = pred
                    if not score:
                        row['ans_full'] = "I don't know."
                    else:
                        cnt += 1
                train_data.append(row)
        
        print(f"成功处理: {cnt}/{len(train_data)}")
        print(f"训练集: {len(train_data)}, 验证集: {len(eval_data)}")
        
        return train_data, eval_data
    
    def filter_duplicates(self, train_data):
        """过滤重复数据"""
        print("过滤重复数据...")
        
        new_train_data = []
        a, b = 0, 0
        
        for i in tqdm(range(0, len(train_data), 5), desc="过滤重复"):
            if i + 4 >= len(train_data):
                break
                
            row1 = train_data[i]
            row2 = train_data[i+1]
            row3 = train_data[i+2]
            row4 = train_data[i+3]
            row5 = train_data[i+4]

            ids = [
                row1['session_id'], row2['session_id'], row3['session_id'], 
                row4['session_id'], row5['session_id']
            ]
            assert len(set(ids)) == 1

            answers = [
                row1['ans_full'], row2['ans_full'], row3['ans_full'], 
                row4['ans_full'], row5['ans_full']
            ]

            # 如果所有答案都是"I don't know"，只保留一个
            if len(set(answers)) == 1 and answers[0] == "I don't know.":
                new_train_data.append(row1)
                a += 1
            else:
                # 保留所有非"I don't know"的答案
                for row in [row1, row2, row3, row4, row5]:
                    if row['ans_full'] == "I don't know.":
                        continue
                    new_train_data.append(row)
                    b += 1
        
        print(f"过滤结果 - 重复答案: {a}, 有效答案: {b}")
        return new_train_data

def main():
    """主函数"""
    print("开始合成数据生成Pipeline...")
    
    # 初始化Pipeline
    model_name = "meta-llama/Llama-3.2-11B-Vision-Instruct"
    tokenizer_path = "meta-llama/Llama-3.2-11B-Vision-Instruct"
    
    pipeline = SynthesisPipeline(model_name, tokenizer_path)
    
    # 加载模型
    pipeline.load_models()
    
    # 输入和输出路径
    input_data_path = "/dataset/v0.1.2/st/crag_mm_st_task1.jsonl"
    generated_data_path = "/dataset/v0.1.2/st/finaLl_task1_webv6_search_syth_data.jsonl"
    
    # 第一步：生成答案
    print("=== 第一步：生成答案 ===")
    generated_data = pipeline.generate_answers(input_data_path, generated_data_path)
    
    # 第二步：评估和过滤
    print("=== 第二步：评估和过滤 ===")
    train_data, eval_data = pipeline.evaluate_and_filter(generated_data_path)
    
    # 第三步：过滤重复
    print("=== 第三步：过滤重复 ===")
    filtered_train_data = pipeline.filter_duplicates(train_data)
    
    # 保存最终结果
    print("=== 保存结果 ===")
    train_output_path = "/dataset/v0.1.2/recall_query/train_crag_mm_comb_task3_new_prompt_syth_data_fold0.jsonl"
    eval_output_path = "/dataset/v0.1.2/recall_query/eval_crag_mm_comb_task3_new_prompt_syth_data_fold0.jsonl"
    
    with open(train_output_path, 'w') as f:
        for line in filtered_train_data:
            f.write(json.dumps(line) + '\n')
    
    with open(eval_output_path, 'w') as f:
        for line in eval_data:
            f.write(json.dumps(line) + '\n')
    
    print(f"训练集已保存到: {train_output_path}")
    print(f"验证集已保存到: {eval_output_path}")
    print(f"最终统计 - 训练集: {len(filtered_train_data)}, 验证集: {len(eval_data)}")
    print("Pipeline执行完成！")

if __name__ == "__main__":
    main() 