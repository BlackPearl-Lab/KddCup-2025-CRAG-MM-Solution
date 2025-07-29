#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Task 3 recall pipline
"""

import os
import pandas as pd
import json
import random
from tqdm import tqdm
from PIL import Image
import urllib.parse
import hashlib
import requests
import base64
from io import BytesIO
import copy

# 设置环境变量

def _is_valid_image(file_path):
    """Check if the file is a valid image using PIL.
    
    Args:
        file_path: Path to the image file
        
    Returns:
        bool: True if valid image, False otherwise
    """
    try:
        with Image.open(file_path) as img:
            # Verify the image by loading it
            img.verify()
            
            # Additional check by accessing image properties
            width, height = img.size
            if width <= 0 or height <= 0:
                return False
                
            return True
    except Exception as e:
        print(f"Invalid image file {file_path}: {e}")
        return False
        
def ensure_crag_cache_dir_is_configured():
    """
    Ensure the cache directory for CRAG images exists and is properly configured.
    
    This function:
    1. Checks if CRAG_CACHE_DIR environment variable is set
    2. If not set, uses platform-appropriate default cache location
    3. Creates the directory if it doesn't exist
    4. Returns the path to the cache directory
    
    Returns:
        str: Path to the cache directory
    """    
    # First check if user has explicitly set a cache directory
    cache_dir = os.environ.get("CRAG_CACHE_DIR")
    
    if not cache_dir:
        # Use platform-specific default locations if not explicitly set
        if os.name == 'nt':  # Windows
            cache_home = os.environ.get("LOCALAPPDATA", os.path.expanduser("~/AppData/Local"))
        else:  # Unix/Linux/Mac
            cache_home = os.environ.get("XDG_CACHE_HOME", os.path.expanduser("~/.cache"))
        
        cache_dir = os.path.join(cache_home, "cragmm_images_cache")
        
        # Print info message only the first time
        if not hasattr(ensure_crag_cache_dir_is_configured, "_cache_location_shown"):
            print(f"Caching downloaded images in {cache_dir}")
            print("You can override this by setting the CRAG_CACHE_DIR environment variable.")
            ensure_crag_cache_dir_is_configured._cache_location_shown = True
    
    # Create the directory if it doesn't exist
    if not os.path.exists(cache_dir):
        try:
            os.makedirs(cache_dir, exist_ok=True)
        except Exception as e:
            print(f"Warning: Failed to create cache directory {cache_dir}: {e}")
            # Fall back to a temporary directory if we can't create the default
            import tempfile
            cache_dir = os.path.join(tempfile.gettempdir(), "cragmm_images_cache")
            os.makedirs(cache_dir, exist_ok=True)
            print(f"Using fallback cache directory: {cache_dir}")
    
    return cache_dir 

def download_image_url(image_url):
    """Downloads image from URL and saves it to the cache directory with a deterministic name.
    Returns local path if successful, raises Exception otherwise.
    
    Args:
        image_url: URL of the image to download
        
    Returns:
        str: Local path to the downloaded or cached image
        
    Raises:
        Exception: If the image couldn't be downloaded or is invalid
    """
    cache_dir = ensure_crag_cache_dir_is_configured()
    
    # Create cache directory if it doesn't exist (redundant but keeps backward compatibility)
    os.makedirs(cache_dir, exist_ok=True)
    
    try:
        # Create a deterministic filename based on the URL
        url_hash = hashlib.md5(image_url.encode()).hexdigest()
        file_extension = os.path.splitext(image_url.split('?')[0])[1] or '.jpg'
        local_filename = f"{url_hash}{file_extension}"
        local_path = os.path.join(cache_dir, local_filename)
        
        # If the file already exists in cache, validate and return it
        if os.path.exists(local_path):
            if _is_valid_image(local_path):
                print(f"Using cached image from {local_path}")
                return local_path
            else:
                print(f"Cached image is invalid, re-downloading: {local_path}")
                # Continue with download as the cached file is invalid
        
        # Download the image
        proxies = {
            "http": "http://xx.xxx.xx.xx:8080",
            "https": "http://x.xx.xxx.xx:8080"
        }
        headers = {"User-Agent": "CRAGBot/v0.0.1"}
        response = requests.get(image_url, stream=True, timeout=10, headers=headers, proxies=proxies)
        response.raise_for_status()
        
        # Save the image to a temporary file first
        temp_path = f"{local_path}.temp"
        with open(temp_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        
        # Validate the downloaded image
        if _is_valid_image(temp_path):
            # Move to final location if valid
            os.replace(temp_path, local_path)
            print(f"Downloaded and validated image_url to {local_path}")
            return local_path
        else:
            # Remove invalid image
            os.remove(temp_path)
            print(f"Downloaded image is not valid from URL: {image_url}")
            raise Exception(f"Downloaded image is not valid from URL {image_url}")
            
            
    except Exception as e:
        print(f"Error downloading image from {image_url}: {e}")
        raise Exception(f"Error downloading image from {image_url}: {e}")

def load_base64_image(url, resize_width=512):
    """将图片转换为base64编码"""
    # 打开图片
    with Image.open(url) as img:
        # 计算等比例缩放后的高度
        w_percent = (resize_width / float(img.size[0]))
        h_size = int((float(img.size[1]) * float(w_percent)))
        # 缩放图片
        img = img.resize((resize_width, h_size))
        
        # 保存到内存中
        buffered = BytesIO()
        img.save(buffered, format="PNG")  # 你可以根据需要调整格式
        img_bytes = buffered.getvalue()
    
    # 编码为 base64
    base64_image = base64.b64encode(img_bytes).decode("utf-8")
    return base64_image

def generate_search_query(ori_query, image_url, pre_messages):
    """生成搜索查询"""
    image = load_base64_image(image_url)
    
    def convert_role(role):
        if role == 'user':
            return 'User'
        if role == 'assistant':
            return 'Assistant'
        return role

    history_dialog = [
        convert_role(row['role'])+': '+row['content']+'\n' for row in pre_messages
    ]
    history_dialog_str = '\n'.join(history_dialog)
    
    messages = [
        {
            "role": "system",
            "content": "你是一个网络检索、网络搜索语法转化器，请你根据历史对话、当前轮次原始问题和提供图片，生成一个用于检索的短语。"
        },
        {
            "role": "user", "content":[
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image}"}},
                {"type": "text", "text": f"\n\nThe History Dialog is: {history_dialog_str}\n"},
                {"type": "text", "text": f"\n\nThe Origin Query is: {ori_query}\n"},
            ]
        },
    ]

    from openai import OpenAI
    client = OpenAI(
        api_key = YOUR_KEY,
        base_url = YOUR_BASE_URL
    )
    response = client.chat.completions.create(
        model="gpt-4.1",
        messages=messages,
        temperature=0.6
    )
    content = response.choices[0].message.content

    return content

def attempt_api_call(model_name, messages, max_retries=3):
    """
    Attempt a structured API call with retries upon encountering specific errors.

    Args:
        model_name: The model to query
        messages: List of message objects for the conversation
        max_retries: Maximum number of retry attempts

    Returns:
        Dictionary with accuracy and raw response, or None if all attempts fail
    """
    from openai import OpenAI
    client = OpenAI(
        api_key = YOUR_KEY,
        base_url = YOUR_BASE_URL
    )
    for attempt in range(max_retries):
        try:
            # Use completion.create instead of parse to avoid using the EvaluationResult class in worker processes
            response = client.chat.completions.create(
                model=model_name,
                messages=messages,
                response_format={"type": "json_object"},
            )

            # Parse the JSON content manually
            content = response.choices[0].message.content
            try:
                result_json = json.loads(content)
                accuracy = result_json.get("accuracy", False)
                # Return both the parsed result and raw JSON for debugging
                return {"accuracy": accuracy, "raw": content}
            except json.JSONDecodeError:
                if attempt == max_retries - 1:
                    return {"accuracy": False, "raw": content}
        except Exception as e:
            if attempt == max_retries - 1:
                print(f"[red]Failed after {max_retries} attempts: {str(e)}[/red]")
    return None

def get_system_message() -> str:
    """Returns the system message for the evaluator."""
    return (
        "You are an expert evaluator for question answering systems."
        "Your task is to determine whether the ground truth can be obtained correctly based on the retrieval information.\n\n"
        "Rules:\n"
        "Output correct if the retrieval information captures all the key information from the ground truth.\n"
        "Output a JSON object with a single field 'accuracy' whose value is true or false."
    )

def judge(ori_query, ori_answer, rag_content):
    """判断检索内容是否有助于回答问题"""
    messages = [
        {
            "role": "system",
            "content": "你需要判断检索会的内容，哪些info能帮助你回答query。其中query和answer已经给出。\n\nOutput a JSON object with a single field 'right_info' whose value is list of string like ['[Info 1]']"
        },
        {
            "role": "user", "content":[
                {"type": "text", "text": f"\n\nThe Origin Query is: {ori_query}\n"},
                {"type": "text", "text": f"\n\nThe Answer is: {ori_answer}\n"},
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
        temperature=0.6,
        response_format={"type": "json_object"},
    )
    content = response.choices[0].message.content

    return content

def main():
    """主函数"""
    print("开始加载搜索管道...")
    
    # 导入搜索管道
    from cragmm_search.search import UnifiedSearchPipeline

    test_search_pipeline = UnifiedSearchPipeline(
        image_model_name="openai/clip-vit-large-patch14-336",
        image_hf_dataset_id="crag-mm-2025/image-search-index-public-test",
        image_hf_dataset_tag="v0.5",
        text_model_name="BAAI/bge-large-en-v1.5",
        web_hf_dataset_id="crag-mm-2025/web-search-index-public-test",
        web_hf_dataset_tag="v0.6",
    )

    valid_search_pipeline = UnifiedSearchPipeline(
        image_model_name="openai/clip-vit-large-patch14-336",
        image_hf_dataset_id="crag-mm-2025/image-search-index-validation",
        image_hf_dataset_tag="v0.5",
        text_model_name="BAAI/bge-large-en-v1.5",
        web_hf_dataset_id="crag-mm-2025/web-search-index-validation",
        web_hf_dataset_tag="v0.6",
    )

    print("开始处理数据...")
    
    # 读取数据
    root = "/dataset/v0.1.2/mt/"
    filename = "crag_mm_mt.jsonl"

    data = []
    with open(os.path.join(root, filename), 'r') as f:
        lines = f.readlines()
        for line in tqdm(lines, desc="处理数据"):
            line = json.loads(line)
            image_url = os.path.join(root, 'images', line['image'])
            pre_messages = line['pre_messages']

            rag_context = []
            record_rag_context = []
            max_retrys = 3

            data_split = line['data_split']
            while len(record_rag_context) == 0 and max_retrys > 0:
                try:
                    search_query = generate_search_query(line['query'], image_url, pre_messages)
                    search_query = search_query.replace("'", "").replace('"', '')

                    if data_split == 'test':
                        results = test_search_pipeline(search_query, k=30)
                    elif data_split == 'valid':
                        results = valid_search_pipeline(search_query, k=30)
                    else:
                        raise ValueError(f"Unknown data_split: {data_split}")

                    snippets = []
                    for i, result in enumerate(results):
                        snippet = result.get('page_snippet', '')
                        
                        if snippet:
                            snippets.append(snippet)
                    
                    rag_context = [f"[Info {i+1}] {snippets[i]}\n\n" for i in range(len(snippets))]
                    record_rag_context = copy.deepcopy(rag_context)
                    max_retrys -= 1

                    if max_retrys <= 4 and len(rag_context) == 0:
                        print(f"Retry!!!, search query = {search_query}")
                except Exception as e:
                    print(f"Error processing line: {e}")
                    pass
            
            line['web_search_query'] = search_query
            line['web_search_info_list'] = rag_context
            data.append(line)

    print(f"处理完成，共处理 {len(data)} 条数据")

    # 保存第一阶段结果
    output_path = "/dataset/v0.1.2/mt/crag_mm_mt_task3_webv6.jsonl"
    with open(output_path, 'w') as f:
        for line in data:
            f.write(json.dumps(line) + '\n')
    
    print(f"第一阶段结果已保存到: {output_path}")

    # 第二阶段：判断检索内容
    print("开始第二阶段：判断检索内容...")
    
    success_cnt = 0
    saved_data = []
    for row in tqdm(data, desc="判断检索内容"):
        try:
            jude_list = json.loads(judge(row['query'], row['ans_full'], row['web_search_info_list']))
            row['jude_list'] = jude_list
            if len(jude_list) != 0:
                success_cnt += 1
        except Exception as e:
            print(f"Error in judge function: {e}")
            row['jude_list'] = {"right_info": []}
        saved_data.append(row)
    
    print(f"成功处理: {success_cnt}/{len(saved_data)}")

    # 保存最终结果
    final_output_path = "/dataset/v0.1.1/mt/crag_mm_mt_task3_webv6_recall30_labeled.jsonl"
    with open(final_output_path, 'w') as f:
        for line in saved_data:
            f.write(json.dumps(line) + '\n')
    
    print(f"最终结果已保存到: {final_output_path}")
    print("处理完成！")

if __name__ == "__main__":
    main() 