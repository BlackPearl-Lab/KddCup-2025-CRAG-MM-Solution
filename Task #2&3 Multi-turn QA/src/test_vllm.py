import vllm
from vllm import LLM, SamplingParams
from vllm.lora.request import LoRARequest

VLLM_TENSOR_PARALLEL_SIZE = 2 
VLLM_GPU_MEMORY_UTILIZATION = 0.3
MAX_MODEL_LEN = 8192
MAX_NUM_SEQS = 2
MAX_GENERATION_TOKENS = 75
model_name="/home/hadoop-aipnlp/dolphinfs_hdd_hadoop-aipnlp/EVA/zhangzijian14/models/huggingface.co/meta-llama/Llama-3.2-11B-Vision-Instruct"
llm = vllm.LLM(
    model_name,
    tensor_parallel_size=VLLM_TENSOR_PARALLEL_SIZE, 
    gpu_memory_utilization=VLLM_GPU_MEMORY_UTILIZATION, 
    max_model_len=MAX_MODEL_LEN,
    max_num_seqs=MAX_NUM_SEQS,
    trust_remote_code=True,
    dtype="bfloat16",
    enforce_eager=True,
    limit_mm_per_prompt={
        "image": 1 
    }, # In the CRAG-MM dataset, every conversation has at most 1 image
    enable_lora=True
)

import os
import json
from tqdm import tqdm
from transformers import AutoTokenizer
from PIL import Image
tokenizer = llm.get_tokenizer()

def tokenize(image, query, tokenizer):
    SYSTEM_PROMPT = "You are a helpful assistant that truthfully answers user questions about the provided image.\n\
Keep your response concise and to the point. If you don't know the answer, respond with 'I don't know'."
    messages = [
            {
                "role": "user", "content": [
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
    ] + [
        {
            "role": "user", "content": [
                {"type": "text", "text": "\n\nPlease ask: " + query}
            ]
        }
    ]
    
    # Apply chat template
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

eval_data = []
eval_inputs = []
data_root = "/home/hadoop-aipnlp/dolphinfs_hdd_hadoop-aipnlp/EVA/zhangzijian14/comps/kdd/dataset/v0.1.1/st/"
with open("/home/hadoop-aipnlp/dolphinfs_hdd_hadoop-aipnlp/EVA/zhangzijian14/comps/kdd/dataset/v0.1.1/st/eval_crag_mm_st_task1_origin_fold0.jsonl", "r") as f:
    lines = f.readlines()
    for line in lines:
        # eval_data.append()
        line = json.loads(line)

        image = Image.open(os.path.join(data_root, "images", line['image']))
        query = line['query']

        eval_inputs.append(tokenize(image, query, tokenizer))

        eval_data.append({
            "ground_truth": line['ans_full'], 
            "query": query
        })

len(eval_inputs), len(eval_data)

responses = llm.generate(
    eval_inputs,
    sampling_params=vllm.SamplingParams(
        temperature=0.01,
        top_p=0.9,
        max_tokens=MAX_GENERATION_TOKENS,
        skip_special_tokens=True
    ),
    lora_request=LoRARequest(
        "hihi", 0, 
        "/home/hadoop-aipnlp/dolphinfs_hdd_hadoop-aipnlp/EVA/zhangzijian14/comps/kdd/checkpoints/20250506_llama32_11b_st_task1_gpt_4o_one_pic_syth_fold0_lr1e-4/checkpoint-50"
    ),
    use_tqdm = True
)

for o in responses:
    generated_text = o.outputs[0].text
    print(generated_text + "\n")