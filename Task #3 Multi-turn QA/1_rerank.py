#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Task 3 重排打标脚本
基于4o-mini进行重排打标
"""

import os
import pandas as pd
import json
import time
import asyncio
from tqdm import tqdm
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from openai import AsyncOpenAI

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
    # 简单按字符长度分割，可以换成按句子分割
    return [" ".join(passage[i:i+chunk_size]) for i in range(0, len(passage), chunk_size)]

def find_most_relevant_chunk(query, passage, chunk_size=100):
    """找到最相关的文本块"""
    # 1. 切分Passage
    chunks = split_passage(passage, chunk_size)
    
    # 2. 构建语料库（query + chunks）
    corpus = [query] + chunks
    
    # 3. 用TF-IDF向量化
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(corpus)
    
    # 4. 计算query与每个chunk的余弦相似度
    similarities = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:]).flatten()
    
    # 5. 找到最相关的chunk
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

def judge(ori_query, rag_content):
    """判断检索内容是否与查询相关（同步版本）"""
    messages = [
        {
            "role": "system",
            "content": "你需要判断检索会的内容是否与query相关。\n\nOutput a JSON object with a single field 'is_relevance' whose value is boolean like True or False"
        },
        {
            "role": "user", "content":[
                {"type": "text", "text": f"\n\nThe Origin Query is: {ori_query}\n"},
                {"type": "text", "text": f"\n\nThe Retrieval Infos are: {rag_content}\n"}
            ]
        }
    ]

    from openai import OpenAI
    client = OpenAI(
        api_key = YOUR_KEY,
        base_url = YOUR_BASE_URL
    )
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=messages,
        temperature=0.01,
        response_format={"type": "json_object"},
    )
    content = response.choices[0].message.content

    return content

async def judge_async(ori_query, rag_content):
    """判断检索内容是否与查询相关（异步版本）"""
    messages = [
        {
            "role": "system",
            "content": "你需要判断检索会的内容是否与query相关。\n\nOutput a JSON object with a single field 'is_relevance' whose value is boolean like True or False"
        },
        {
            "role": "user", "content":[
                {"type": "text", "text": f"\n\nThe Origin Query is: {ori_query}\n"},
                {"type": "text", "text": f"\n\nThe Retrieval Infos are: {rag_content}\n"}
            ]
        }
    ]
    client = OpenAI(
        api_key = YOUR_KEY,
        base_url = YOUR_BASE_URL
    )
    response = await client.chat.completions.create(
        model="gpt-4o-mini",
        messages=messages,
        temperature=0.01,
        response_format={"type": "json_object"},
    )
    content = response.choices[0].message.content
    return content

async def process_single_row(row):
    """处理单行数据"""
    query = row['query']
    web_search_query = row['web_search_query']
    web_search_info_list = row['web_search_info_list']

    # 分割文本块
    web_search_info_list = split_chunk(f"{web_search_query}, {query}", web_search_info_list)

    relevance_list = []
    judge_queries = [f"{web_search_query}, {query}"] * len(web_search_info_list)
    params = []
    for a, b in zip(judge_queries, web_search_info_list):
        params.append((a, b))
    coros = [judge_async(q, r) for q, r in params]

    is_success = False
    max_try_cnt = 3
    while not is_success and max_try_cnt >= 0:
        max_try_cnt -= 1
        try:
            results = await asyncio.gather(*coros)
            relevance_list = []
            for res in results:
                try:
                    res = json.loads(res)
                    if 'is_relevance' in res:
                        relevance_list.append(res['is_relevance'])
                    else:
                        relevance_list.append(False)
                except:
                    relevance_list.append(False)
            assert len(relevance_list) == len(web_search_info_list)
            jude_list = [f"[Info {i+1}]" for i in range(len(web_search_info_list)) if relevance_list[i]]
            row['jude_list'] = {
                'right_info': jude_list
            }
            time.sleep(1)
            is_success = True

        except Exception as e:
            print(f"处理错误，重试中... 错误信息: {e}")
            time.sleep(30)
            
    return row

async def process_data_async(data, batch_size=10):
    """异步处理数据"""
    saved_data = []
    
    # 分批处理数据
    for i in tqdm(range(0, len(data), batch_size), desc="处理数据批次"):
        batch = data[i:i+batch_size]
        tasks = [process_single_row(row) for row in batch]
        
        try:
            batch_results = await asyncio.gather(*tasks, return_exceptions=True)
            for result in batch_results:
                if isinstance(result, Exception):
                    print(f"批次处理错误: {result}")
                    # 如果异步处理失败，使用同步版本
                    # 这里可以添加同步处理逻辑
                else:
                    saved_data.append(result)
        except Exception as e:
            print(f"批次处理失败: {e}")
            # 如果整个批次失败，逐个处理
            for row in batch:
                try:
                    result = await process_single_row(row)
                    saved_data.append(result)
                except Exception as e:
                    print(f"单行处理失败: {e}")
                    saved_data.append(row)
    
    return saved_data

def main():
    """主函数"""
    print("开始加载数据...")
    
    # 加载数据
    filename = '/dataset/v0.1.2/mt/crag_mm_mt_task3_webv6_recall30_labeled.jsonl'
    data = load_jsonl(filename)
    
    print(f"加载了 {len(data)} 条数据")
    
    # 异步处理数据
    print("开始异步处理数据...")
    saved_data = asyncio.run(process_data_async(data))
    
    print(f"处理完成，共处理 {len(saved_data)} 条数据")
    
    # 保存结果
    output_path = "/dataset/v0.1.2/mt/crag_mm_mt_task3_webv6_recall30_pointwise_labeled.jsonl"
    with open(output_path, "w") as f:
        for line in saved_data:
            f.write(json.dumps(line) + "\n")
    
    print(f"结果已保存到: {output_path}")
    print("处理完成！")

if __name__ == "__main__":
    main() 