from openai import OpenAI
import pandas as pd
import json
import pickle
from tqdm.auto import tqdm
import requests
import numpy as np
import glob
import torch
import random
import copy
import time


def request_openai_chatgpt_version(system, message, temperature=1., top_p=0.8, model_name="gpt-4o-2024-05-13"):
    try:
        client = OpenAI(
            api_key="xx",
            base_url="xx"
        )
        messages = [
            {
                "role": "system",
                "content": system
            },
            {
                "role": "user",
                "content": message
            }
        ]
        streamResult = client.chat.completions.create(
            model=model_name,
            messages=messages,
            temperature=temperature,
            top_p=top_p,
            stream=False,
            max_tokens=4096
        )
        res = streamResult
        prompt_tokens = res.usage.prompt_tokens
        answer_tokens = res.usage.completion_tokens
        if model_name == "gpt-4o-2024-08-06":
            cost = 18 / 1000000. * prompt_tokens + 72 / 1000000. * answer_tokens
        else:
            cost = 1. / 1000000. * prompt_tokens + 2. / 1000000. * answer_tokens
        return res.choices[0].message.content, cost
    except Exception as e:
        # 打印详细的错误信息
        print("An error occurred:", e)
        return None, None

import concurrent.futures

def parallel_requests(messages, system, temperature=1., top_p=0.8, model_name="gpt-4o-2024-05-13", max_workers=5):
    results = [None] * len(messages)  # 预先分配结果列表
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_index = {
            executor.submit(request_openai_chatgpt_version, message, system, temperature, top_p, model_name): index for
            index, message in enumerate(messages)}
        for future in tqdm(concurrent.futures.as_completed(future_to_index), total=len(messages)):
            index = future_to_index[future]
            try:
                data = future.result()
                results[index] = data
            except Exception as exc:
                print(f"Message at index {index} generated an exception: {exc}")
                results[index] = None
    return results



train_zj = pd.read_json("./zj_data/train_crag_mm_comb_task3_as_task1_syth_data_fold0.jsonl",lines=True)

temp_data  = train_zj[train_zj["ans_full"].apply(lambda x: "I don't know." not in x)]



import time
for _,row in tqdm(temp_data.iterrows()):
    while True:
        try:
            query = row["query"]
            ans_full = row["ans_full"]
            prompt = f"""给定一个问题和标准答案，请你帮我创建10个类似的标准答案，要求精简答案、或者将答案复杂化，但是不能超过50个单词，输出格式:
        1.xx(规则:精简|复杂)
        2.xx(规则:精简|复杂)
        3.xx
        ....
        问题:{query}
        标准答案:{ans_full}
        """
            res = request_openai_chatgpt_version("",prompt, temperature=0.6, top_p=0.8,pri=False)
            row["argu_v1"] = res
            with open(f"./argu_data/{row['interaction_id']}.pkl",'wb') as f:
                pickle.dump(row,f)
            break
        except:
            time.sleep(200)


import glob
data = []
for path in tqdm(glob.glob("./argu_data/*.pkl")):
    with open(path,'rb') as f:
        data.append(pickle.load(f))
add_df = pd.DataFrame(data)

argument_df = []
for _,row in add_df.iterrows():
    answer = row["argu_v1"][1]
    if answer is None:
        continue
    answer = answer.strip("\n").replace("（","(")
    answers =answer.split("\n")
    for ans in answers:
        if not any([f"{i}." in ans for i in range(1,11)]):
            continue
        ans = ans.split(".",1)[1].split("(规则",1)[0]
        row["response"] = ans.strip()
        argument_df.append(copy.deepcopy(row))
argument_df= pd.DataFrame(argument_df)



from pydantic import BaseModel
MAX_API_RETRIES = 3
eval_model_name = "gpt-4o-mini"
class CRAGTurnEvaluationResult(BaseModel):
    """Structured output model for CRAG turn evaluation results."""
    accuracy: bool
def get_system_message() -> str:
    return (
        "You are an expert evaluator for question answering systems. "
        "Your task is to determine if a prediction correctly answers a question based on the ground truth.\n\n"
        "Rules:\n"
        "1. The prediction is correct if it captures all the key information from the ground truth.\n"
        "2. The prediction is correct even if phrased differently as long as the meaning is the same.\n"
        "3. The prediction is incorrect if it contains incorrect information or is missing essential details.\n"
        "Output a JSON object with a single field 'accuracy' whose value is true or false."
    )

def attempt_api_call(
    client: OpenAI,
    model_name: str,
    messages: list,
    max_retries: int = MAX_API_RETRIES,
) -> CRAGTurnEvaluationResult | None:
    for attempt in range(max_retries):
        try:
            completion = client.beta.chat.completions.parse(
                model=model_name,
                messages=messages,
                response_format=CRAGTurnEvaluationResult,
                temperature=0.0
            )
            return completion.choices[0].message.parsed
        except Exception as e:
            error_message = f"API call failed on attempt {attempt + 1}/{max_retries}: {str(e)}"
            if attempt == max_retries - 1:
                print(error_message,"end")
            print(error_message)
    return None


def evaluate_response(crag_turn_data: dict[str, any]) -> dict[str, any]:
    agent_response = crag_turn_data["response"]
    ground_truth = crag_turn_data["ans_full"]
    query = crag_turn_data["query"]

    is_idk = "i don't know" in agent_response.lower()
    is_exact_match = agent_response.strip().lower() == ground_truth.strip().lower()
    is_semantically_correct = False
    api_response = None

    # Begin by assuming exact match correctness
    is_correct = is_exact_match

    # Use semantic evaluation if not an exact match and an evaluation model is provided.
    if not is_idk and not is_exact_match and eval_model_name:
        local_openai_client = OpenAI(
            api_key="xx",
            base_url="xx"
        )
        messages = [
            {"role": "system", "content": get_system_message()},
            {"role": "user", "content": f"Question: {query}\nGround truth: {ground_truth}\nPrediction: {agent_response}\n"},
        ]
        # print()
        api_response = attempt_api_call(local_openai_client, eval_model_name, messages)
        if api_response:
            is_semantically_correct = api_response.accuracy
            is_correct = is_semantically_correct
    if is_exact_match:
        is_semantically_correct = True

    return {
        "is_exact_match": is_exact_match,
        "is_correct": is_correct,
        "is_miss": is_idk,
        "is_semantically_correct": is_semantically_correct,
        "api_response": api_response.model_dump() if api_response else None,
    }


import time
answers = []
for _,row in tqdm(argument_df.iterrows()):
    while True:
        try:
            res = evaluate_response(dict(row))
            break
        except:
            time.slppe(100)
    answers.append(res)
argument_df["relu_data"] = answers


import pickle
with open("augu_v2_gpt4o_task3.pkl",'wb') as f:
    pickle.dump(argument_df,f)