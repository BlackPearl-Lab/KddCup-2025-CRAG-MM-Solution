import os
import json
from sklearn import metrics
import numpy as np
from dataclasses import dataclass
from torch.utils.data import Dataset
import random
import numpy as np
import torch
from PIL import Image
from transformers.utils import PaddingStrategy
from transformers import AutoProcessor

from tqdm import tqdm

from typing import Any, Callable, Dict, List, NewType, Optional, Tuple, Union


import os
import json
import numpy as np

from tqdm import tqdm

class KDDSingleTurnDataSet(Dataset):
    def __init__(self, data_path, data_type, processor, do_eval=False, max_source_length=100, max_target_length=100):
        super(KDDSingleTurnDataSet, self).__init__()
        self.do_eval = do_eval
        self.data_path = data_path
        self.data_type = data_type
        self.data = self.load_jsonl(
            os.path.join(data_path, f"train_crag_mm_{data_type}.jsonl")
        )
        if self.do_eval:
            self.data = self.load_jsonl(
                os.path.join(data_path, f"eval_crag_mm_{data_type}.jsonl")
            )
            self.label_2_query_map = self.get_label_2_query_map(self.data)
        ignore_cnt = len([row for row in self.data if row['ans_full'] == "I don't know."])
        print(f"Ignore cnt: {ignore_cnt}, Total cnt: {len(self.data)}")
            
        self.processor = processor
        self.system_prompt = "You are a helpful assistant that truthfully answers user questions about the provided image. Keep your response concise and to the point. If you don't know the answer, respond with 'I don't know'."

    def get_label_2_query_map(self, data):
        label_2_query_map = {}
        for row in data:
            query = str(row['query']).strip()
            ans_full = str(row['ans_full']).strip()
            # assert ans_full not in label_2_query_map
            label_2_query_map[ans_full] = query
        return label_2_query_map
    
    def load_jsonl(self, path):
        with open(path, "r") as f:
            data = [json.loads(line) for line in f.readlines()]
        random.shuffle(data)
        return data

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):
        row = self.data[index]

        interaction_id = str(row['interaction_id']).strip()
        image_name = str(row['image']).strip()
        domain = str(row['domain']).strip()
        query_category = str(row['query_category']).strip()
        dynamism = str(row['dynamism']).strip()
        query = str(row['query']).strip()
        ans_full = str(row['ans_full']).strip()
        # if not self.do_eval:
        #    ans_full = "i don't know"
        image_quality = str(row['image_quality']).strip()

        image = Image.open(os.path.join(self.data_path, "images", image_name))
        image = image.resize((960, 1280))

        # retrieval_image_paths = row['retrieval_task1_images']
        # retrieval_images = [
        #     Image.open(os.path.join(self.data_path, "retrieval_task1_images", image_name)).convert("RGB") for image_name in retrieval_image_paths
        # ]
        retrieval_images = []

        full_instruction = [
                {
                    "role": "system", "content": [
                        {"type": "text", "text": self.system_prompt}
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
        ] + [
            {
                "role": "assistant", "content": [
                    {"type": "text", "text": ans_full}
                ]
            },
        ]
        context_instruction = [
                {
                    "role": "user", "content": [
                        {"type": "text", "text": self.system_prompt}
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
        context_text = self.processor.apply_chat_template(context_instruction, add_generation_prompt=True)
        full_text = self.processor.apply_chat_template(full_instruction, add_generation_prompt=True)

        full_inputs = self.processor(
            image,
            full_text,
            add_special_tokens=False,
        )
        full_input_ids = full_inputs.input_ids
        full_attention_mask = full_inputs.attention_mask
        full_pixel_values = full_inputs.pixel_values
        full_aspect_ratio_ids = full_inputs.aspect_ratio_ids
        full_aspect_ratio_mask = full_inputs.aspect_ratio_mask
        full_cross_attention_mask = full_inputs.cross_attention_mask

        context_inputs = self.processor(
            image,
            context_text,
            add_special_tokens=False,
        )
        context_input_ids = context_inputs.input_ids

        if not self.do_eval:
            answer_input_ids = [[-100] * len(context_input_ids[0]) + full_input_ids[0][len(context_input_ids[0]):]]

            target_mask = [0] * len(context_input_ids[0]) + [1] * (len(full_input_ids[0]) - len(context_input_ids[0]))

            assert len(full_attention_mask[0]) == len(full_input_ids[0]) == len(answer_input_ids[0]) == len(target_mask)
            # print(torch.tensor(full_pixel_values).shape)
            return {
                "input_ids": full_input_ids[0],
                "labels": answer_input_ids[0],
                # 'attention_mask': full_attention_mask[0],
                'pixel_values': full_pixel_values[0],
                'aspect_ratio_ids': full_aspect_ratio_ids[0],
                'aspect_ratio_mask': full_aspect_ratio_mask[0],
                'cross_attention_mask': full_cross_attention_mask[0],
                'target_mask': target_mask
            }
        else:
            label = full_text[len(context_text):]
            label_input_ids = self.processor.tokenizer.encode(label, add_special_tokens=False)

            context_input_ids = context_input_ids[0]
            label_input_ids= [-100] * (len(context_input_ids) -  len(label_input_ids)) + label_input_ids

            return {
                "input_ids": context_input_ids[:min(len(context_input_ids), len(label_input_ids))],
                # "attention_mask": full_attention_mask[0],
                "pixel_values": context_inputs.pixel_values[0],
                "aspect_ratio_ids": context_inputs.aspect_ratio_ids[0],
                "aspect_ratio_mask": context_inputs.aspect_ratio_mask[0],
                "cross_attention_mask": context_inputs.cross_attention_mask[0],
                "labels": label_input_ids[:min(len(context_input_ids), len(label_input_ids))]
            }
        

class KDDSingleTurnTask2DataSet(Dataset):
    def __init__(self, data_path, data_type, processor, do_eval=False, max_source_length=100, max_target_length=100):
        super(KDDSingleTurnTask2DataSet, self).__init__()
        self.gen_system_prompt = "You are a helpful assistant that truthfully answers user questions about the provided image and the retrieval information.\nThe retrieval information may not related to the provided query and image. \nPlease pay attention to identifying that information and answer the query with image. \nKeep your response concise and to the point. If you don't know the answer, respond with 'I don't know'."
        self.search_system_prompt = "You are a web retrieval and web search syntax converter. Please generate phrases for retrieval based on the original query and provided image."

        self.do_eval = do_eval
        self.data_path = data_path
        self.data_type = data_type
        filename = os.path.join(data_path, f"train_crag_mm_{data_type}.jsonl") if not self.do_eval else os.path.join(data_path, f"eval_crag_mm_{data_type}.jsonl")
        self.data = self.load_jsonl(filename)
        if self.do_eval:
            self.label_2_query_map = self.get_label_2_query_map(self.data)
        self.data = self.convert_to_messages(self.data)
        ignore_cnt = len([row for row in self.data if row['ans_full'] == "I don't know."])
        print(f"Ignore cnt: {ignore_cnt}, Total cnt: {len(self.data)}")
            
        self.processor = processor

    def get_label_2_query_map(self, data):
        label_2_query_map = {}
        for row in data:
            query = str(row['query']).strip()
            ans_full = str(row['ans_full']).strip()
            # assert ans_full not in label_2_query_map
            label_2_query_map[ans_full] = query
        return label_2_query_map
    
    def load_jsonl(self, path):
        with open(path, "r") as f:
            data = [json.loads(line) for line in f.readlines()]
        random.shuffle(data)
        return data

    def __len__(self):
        return len(self.data)
    
    def convert_to_messages(self, dataset):
        if self.do_eval:
            return self.convert_to_messages_for_eval(dataset)
        else:
            return self.convert_to_messages_for_train(dataset)
    
    def convert_to_messages_for_train(self, dataset):
        messages = []
        for row in dataset:
            interaction_id = str(row['interaction_id']).strip()
            image_name = str(row['image']).strip()
            domain = str(row['domain']).strip()
            query_category = str(row['query_category']).strip()
            dynamism = str(row['dynamism']).strip()
            query = str(row['query']).strip()
            ans_full = str(row['ans_full']).strip()
            image_quality = str(row['image_quality']).strip()

            image = Image.open(os.path.join(self.data_path, "images", image_name))
            image = image.resize((960, 1280))

            rag_info_list = row['web_search_info_list']
            rag_info_list = [l for l in rag_info_list if len(l.strip()) >= 1]
            rag_info = '\n'.join(rag_info_list)

            # generation task
            full_instruction = [
                {
                    "role": "user", "content": [
                        {"type": "text", "text": self.gen_system_prompt}
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
                        {"type": "text", "text": "\n\nThe retrieval information is:\n"},
                        {"type": "text", "text": rag_info}
                    ]
                }
            ] + [
                {
                    "role": "user", "content": [
                        {"type": "text", "text": "\n\nPlease ask: " + query}
                    ]
                }
            ] + [
                {
                    "role": "assistant", "content": [
                        {"type": "text", "text": ans_full}
                    ]
                },
            ]
            context_instruction = [
                {
                    "role": "user", "content": [
                        {"type": "text", "text": self.gen_system_prompt}
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
                        {"type": "text", "text": "\n\nThe retrieval information is:\n"},
                        {"type": "text", "text": rag_info}
                    ]
                }
            ] + [
                {
                    "role": "user", "content": [
                        {"type": "text", "text": "\n\nPlease ask: " + query}
                    ]
                }
            ]

            messages.append({
                'full_instruction': full_instruction[:8000],
                'context_instruction': context_instruction[:8000],
                'image': image,
                'ans_full': ans_full
            })

            # search retrieval keyword task
            full_instruction = [
                {
                    "role": "system",
                    "content": self.search_system_prompt
                },
                {
                    "role": "user", "content":[
                        {"type": "text", "text": f"\n\nThe Origin Query is: {query}\n"},
                        {"type": "image", "image": image},
                        {"type": "text", "text": f"\n\nThe final retrieval search query is:\n"}
                    ]
                },
                {
                    "role": "assistant", "content": [
                        {"type": "text", "text": str(row['web_search_query'])}
                    ]
                },
            ]
            context_instruction = [
                {
                    "role": "system",
                    "content": self.search_system_prompt
                },
                {
                    "role": "user", "content":[
                        {"type": "text", "text": f"\n\nThe Origin Query is: {query}\n"},
                        {"type": "image", "image": image},
                        {"type": "text", "text": f"\n\nThe final retrieval search query is:\n"}
                    ]
                }
            ]
            messages.append({
                'full_instruction': full_instruction[:8000],
                'context_instruction': context_instruction[:8000],
                'image': image,
                'ans_full': str(row['web_search_query'])  
            })

        return messages
    
    def convert_to_messages_for_eval(self, dataset):
        messages = []
        for row in dataset:
            interaction_id = str(row['interaction_id']).strip()
            image_name = str(row['image']).strip()
            domain = str(row['domain']).strip()
            query_category = str(row['query_category']).strip()
            dynamism = str(row['dynamism']).strip()
            query = str(row['query']).strip()
            ans_full = str(row['ans_full']).strip()
            image_quality = str(row['image_quality']).strip()

            image = Image.open(os.path.join(self.data_path, "images", image_name))
            image = image.resize((960, 1280))

            rag_info_list = row['web_search_info_list']
            rag_info_list = [l for l in rag_info_list if len(l.strip()) >= 1]
            rag_info = '\n'.join(rag_info_list)

            # generation task
            full_instruction = [
                {
                    "role": "user", "content": [
                        {"type": "text", "text": self.gen_system_prompt}
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
                        {"type": "text", "text": "\n\nThe retrieval information is:\n"},
                        {"type": "text", "text": rag_info}
                    ]
                }
            ] + [
                {
                    "role": "user", "content": [
                        {"type": "text", "text": "\n\nPlease ask: " + query}
                    ]
                }
            ] + [
                {
                    "role": "assistant", "content": [
                        {"type": "text", "text": ans_full}
                    ]
                },
            ]
            context_instruction = [
                {
                    "role": "user", "content": [
                        {"type": "text", "text": self.gen_system_prompt}
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
                        {"type": "text", "text": "\n\nThe retrieval information is:\n"},
                        {"type": "text", "text": rag_info}
                    ]
                }
            ] + [
                {
                    "role": "user", "content": [
                        {"type": "text", "text": "\n\nPlease ask: " + query}
                    ]
                }
            ]

            messages.append({
                'full_instruction': full_instruction[:8000],
                'context_instruction': context_instruction[:8000],
                'image': image,
                'ans_full': ans_full
            })
        return messages
    
    def __getitem__(self, index):
        row = self.data[index]

        context_instruction = row['context_instruction']
        full_instruction = row['full_instruction']
        image = row['image']
        
        context_text = self.processor.apply_chat_template(context_instruction, add_generation_prompt=True)
        full_text = self.processor.apply_chat_template(full_instruction, add_generation_prompt=True)

        full_inputs = self.processor(
            image,
            full_text,
            add_special_tokens=False,
        )
        full_input_ids = full_inputs.input_ids
        full_attention_mask = full_inputs.attention_mask
        full_pixel_values = full_inputs.pixel_values
        full_aspect_ratio_ids = full_inputs.aspect_ratio_ids
        full_aspect_ratio_mask = full_inputs.aspect_ratio_mask
        full_cross_attention_mask = full_inputs.cross_attention_mask

        context_inputs = self.processor(
            image,
            context_text,
            add_special_tokens=False,
        )
        context_input_ids = context_inputs.input_ids

        if not self.do_eval:
            answer_input_ids = [[-100] * len(context_input_ids[0]) + full_input_ids[0][len(context_input_ids[0]):]]

            target_mask = [0] * len(context_input_ids[0]) + [1] * (len(full_input_ids[0]) - len(context_input_ids[0]))

            assert len(full_attention_mask[0]) == len(full_input_ids[0]) == len(answer_input_ids[0]) == len(target_mask)
            # print(torch.tensor(full_pixel_values).shape)
            return {
                "input_ids": full_input_ids[0],
                "labels": answer_input_ids[0],
                # 'attention_mask': full_attention_mask[0],
                'pixel_values': full_pixel_values[0],
                'aspect_ratio_ids': full_aspect_ratio_ids[0],
                'aspect_ratio_mask': full_aspect_ratio_mask[0],
                'cross_attention_mask': full_cross_attention_mask[0],
                'target_mask': target_mask
            }
        else:
            label = full_text[len(context_text):]
            label_input_ids = self.processor.tokenizer.encode(label, add_special_tokens=False)

            context_input_ids = context_input_ids[0]
            label_input_ids= [-100] * (len(context_input_ids) -  len(label_input_ids)) + label_input_ids

            return {
                "input_ids": context_input_ids[:min(len(context_input_ids), len(label_input_ids))],
                # "attention_mask": full_attention_mask[0],
                "pixel_values": context_inputs.pixel_values[0],
                "aspect_ratio_ids": context_inputs.aspect_ratio_ids[0],
                "aspect_ratio_mask": context_inputs.aspect_ratio_mask[0],
                "cross_attention_mask": context_inputs.cross_attention_mask[0],
                "labels": label_input_ids[:min(len(context_input_ids), len(label_input_ids))]
            }


class KDDMultipleTurnTask3DataSet(Dataset):
    def __init__(self, data_path, data_type, processor, do_eval=False, max_source_length=100, max_target_length=100):
        super(KDDMultipleTurnTask3DataSet, self).__init__()
        self.gen_system_prompt = "You are a helpful assistant that truthfully answers user questions about the provided image and the retrieval information.\nThe retrieval information may not related to the provided query and image. \nPlease pay attention to identifying that information and answer the query with image. \nKeep your response concise and to the point. If you don't know the answer, respond with 'I don't know'."
        self.search_system_prompt = "You are a web retrieval and web search syntax converter. Please generate phrases for retrieval based on the original query and provided image."

        self.do_eval = do_eval
        self.data_path = data_path
        self.data_type = data_type
        filename = os.path.join(data_path, f"train_crag_mm_{data_type}.jsonl") if not self.do_eval else os.path.join(data_path, f"eval_crag_mm_{data_type}.jsonl")
        self.data = self.load_jsonl(filename)
        if self.do_eval:
            self.label_2_query_map = self.get_label_2_query_map(self.data)
        self.data = self.convert_to_messages(self.data)
        ignore_cnt = len([row for row in self.data if row['ans_full'] == "I don't know."])
        print(f"Ignore cnt: {ignore_cnt}, Total cnt: {len(self.data)}")
            
        self.processor = processor

    def get_label_2_query_map(self, data):
        label_2_query_map = {}
        for row in data:
            query = str(row['query']).strip()
            ans_full = str(row['ans_full']).strip()
            # assert ans_full not in label_2_query_map
            label_2_query_map[ans_full] = query
        return label_2_query_map
    
    def load_jsonl(self, path):
        with open(path, "r") as f:
            data = [json.loads(line) for line in f.readlines()]
        random.shuffle(data)
        return data

    def __len__(self):
        return len(self.data)
    
    def convert_to_messages(self, dataset):
        if self.do_eval:
            return self.convert_to_messages_for_eval(dataset)
        else:
            return self.convert_to_messages_for_train(dataset)
    
    def convert_to_messages_for_train(self, dataset):
        messages = []
        for row in dataset:
            interaction_id = str(row['interaction_id']).strip()
            image_name = str(row['image']).strip()
            domain = str(row['domain']).strip()
            query_category = str(row['query_category']).strip()
            dynamism = str(row['dynamism']).strip()
            query = str(row['query']).strip()
            ans_full = str(row['ans_full']).strip()
            image_quality = str(row['image_quality']).strip()
            pre_messages = row['pre_messages']

            image = Image.open(os.path.join(self.data_path, "images", image_name))
            image = image.resize((960, 1280))

            rag_info_list = row['web_search_info_list']
            rag_info_list = [l for l in rag_info_list if len(l.strip()) >= 1]
            rag_info = '\n'.join(rag_info_list)

            # generation task
            full_instruction = [
                {
                    "role": "user", "content": [
                        {"type": "text", "text": self.gen_system_prompt}
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
                        {"type": "text", "text": "\n\nThe retrieval information is:\n"},
                        {"type": "text", "text": rag_info}
                    ]
                }
            ] + pre_messages + [
                {
                    "role": "user", "content": [
                        {"type": "text", "text": "\n\nPlease ask: " + query}
                    ]
                }
            ] + [
                {
                    "role": "assistant", "content": [
                        {"type": "text", "text": ans_full}
                    ]
                },
            ]
            context_instruction = [
                {
                    "role": "user", "content": [
                        {"type": "text", "text": self.gen_system_prompt}
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
                        {"type": "text", "text": "\n\nThe retrieval information is:\n"},
                        {"type": "text", "text": rag_info}
                    ]
                }
            ] + pre_messages + [
                {
                    "role": "user", "content": [
                        {"type": "text", "text": "\n\nPlease ask: " + query}
                    ]
                }
            ]

            messages.append({
                'full_instruction': full_instruction[:8000],
                'context_instruction': context_instruction[:8000],
                'image': image,
                'ans_full': ans_full
            })

            # search retrieval keyword task
            full_instruction = [
                {
                    "role": "system",
                    "content": self.search_system_prompt
                },     
            ] + pre_messages + [
                {
                    "role": "user", "content":[
                        {"type": "text", "text": f"\n\nThe Origin Query is: {query}\n"},
                        {"type": "image", "image": image},
                        {"type": "text", "text": f"\n\nThe final retrieval search query is:\n"}
                    ]
                },
                {
                    "role": "assistant", "content": [
                        {"type": "text", "text": str(row['web_search_query'])}
                    ]
                },
            ]

            context_instruction = [
                {
                    "role": "system",
                    "content": self.search_system_prompt
                },     
            ] + pre_messages + [
                {
                    "role": "user", "content":[
                        {"type": "text", "text": f"\n\nThe Origin Query is: {query}\n"},
                        {"type": "image", "image": image},
                        {"type": "text", "text": f"\n\nThe final retrieval search query is:\n"}
                    ]
                }
            ]
            messages.append({
                'full_instruction': full_instruction[:8000],
                'context_instruction': context_instruction[:8000],
                'image': image,
                'ans_full': str(row['web_search_query'])  
            })

        return messages
    
    def convert_to_messages_for_eval(self, dataset):
        messages = []
        for row in dataset:
            interaction_id = str(row['interaction_id']).strip()
            image_name = str(row['image']).strip()
            domain = str(row['domain']).strip()
            query_category = str(row['query_category']).strip()
            dynamism = str(row['dynamism']).strip()
            query = str(row['query']).strip()
            ans_full = str(row['ans_full']).strip()
            image_quality = str(row['image_quality']).strip()

            image = Image.open(os.path.join(self.data_path, "images", image_name))
            image = image.resize((960, 1280))

            rag_info_list = row['web_search_info_list']
            rag_info_list = [l for l in rag_info_list if len(l.strip()) >= 1]
            rag_info = '\n'.join(rag_info_list)
            
            pre_messages = row['pre_messages']

            # generation task
            full_instruction = [
                {
                    "role": "user", "content": [
                        {"type": "text", "text": self.gen_system_prompt}
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
                        {"type": "text", "text": "\n\nThe retrieval information is:\n"},
                        {"type": "text", "text": rag_info}
                    ]
                }
            ] + pre_messages + [
                {
                    "role": "user", "content": [
                        {"type": "text", "text": "\n\nPlease ask: " + query}
                    ]
                }
            ] + [
                {
                    "role": "assistant", "content": [
                        {"type": "text", "text": ans_full}
                    ]
                },
            ]
            context_instruction = [
                {
                    "role": "user", "content": [
                        {"type": "text", "text": self.gen_system_prompt}
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
                        {"type": "text", "text": "\n\nThe retrieval information is:\n"},
                        {"type": "text", "text": rag_info}
                    ]
                }
            ] + pre_messages + [
                {
                    "role": "user", "content": [
                        {"type": "text", "text": "\n\nPlease ask: " + query}
                    ]
                }
            ]

            messages.append({
                'full_instruction': full_instruction[:8000],
                'context_instruction': context_instruction[:8000],
                'image': image,
                'ans_full': ans_full
            })
        return messages
    
    def __getitem__(self, index):
        row = self.data[index]

        context_instruction = row['context_instruction']
        full_instruction = row['full_instruction']
        image = row['image']
        
        context_text = self.processor.apply_chat_template(context_instruction, add_generation_prompt=True)
        full_text = self.processor.apply_chat_template(full_instruction, add_generation_prompt=True)

        full_inputs = self.processor(
            image,
            full_text,
            add_special_tokens=False,
        )
        full_input_ids = full_inputs.input_ids
        full_attention_mask = full_inputs.attention_mask
        full_pixel_values = full_inputs.pixel_values
        full_aspect_ratio_ids = full_inputs.aspect_ratio_ids
        full_aspect_ratio_mask = full_inputs.aspect_ratio_mask
        full_cross_attention_mask = full_inputs.cross_attention_mask

        context_inputs = self.processor(
            image,
            context_text,
            add_special_tokens=False,
        )
        context_input_ids = context_inputs.input_ids

        if not self.do_eval:
            answer_input_ids = [[-100] * len(context_input_ids[0]) + full_input_ids[0][len(context_input_ids[0]):]]

            target_mask = [0] * len(context_input_ids[0]) + [1] * (len(full_input_ids[0]) - len(context_input_ids[0]))

            assert len(full_attention_mask[0]) == len(full_input_ids[0]) == len(answer_input_ids[0]) == len(target_mask)
            # print(torch.tensor(full_pixel_values).shape)
            return {
                "input_ids": full_input_ids[0],
                "labels": answer_input_ids[0],
                # 'attention_mask': full_attention_mask[0],
                'pixel_values': full_pixel_values[0],
                'aspect_ratio_ids': full_aspect_ratio_ids[0],
                'aspect_ratio_mask': full_aspect_ratio_mask[0],
                'cross_attention_mask': full_cross_attention_mask[0],
                'target_mask': target_mask
            }
        else:
            label = full_text[len(context_text):]
            label_input_ids = self.processor.tokenizer.encode(label, add_special_tokens=False)

            context_input_ids = context_input_ids[0]
            label_input_ids= [-100] * (len(context_input_ids) -  len(label_input_ids)) + label_input_ids

            return {
                "input_ids": context_input_ids[:min(len(context_input_ids), len(label_input_ids))],
                # "attention_mask": full_attention_mask[0],
                "pixel_values": context_inputs.pixel_values[0],
                "aspect_ratio_ids": context_inputs.aspect_ratio_ids[0],
                "aspect_ratio_mask": context_inputs.aspect_ratio_mask[0],
                "cross_attention_mask": context_inputs.cross_attention_mask[0],
                "labels": label_input_ids[:min(len(context_input_ids), len(label_input_ids))]
            }

class KDDCombDataset(Dataset):
    def __init__(self, data_path, data_type, processor, do_eval=False, max_source_length=100, max_target_length=100):
        super(KDDCombDataset, self).__init__()

        self.query_domain_system_prompt = "You are a query domain converter. Please judge the domain of the query. \n\
The candicates are: Books, Food, Math & Science, Shopping, Animal, Vehicles, and more."

        self.query_dynamism_system_prompt = "You are a query dynamism converter. Please judge the dynamism of the query.\n\
The candicates are: real-time, fast-changing, slow-changing and static."

        with open("/mnt/dolphinfs/hdd_pool/docker/user/hadoop-aipnlp/EVA/zhangzijian14/comps/kdd/dataset/v0.1.2/comb/single_need_resize_session_ids.json", 'r') as f:
            self.single_need_resize_session_ids = json.load(f)


        self.do_eval = do_eval
        self.data_path = data_path
        self.data_type = data_type
        filename = os.path.join(data_path, f"train_crag_mm_{data_type}.jsonl") if not self.do_eval else os.path.join(data_path, f"eval_crag_mm_{data_type}.jsonl")
        self.data = self.load_jsonl(filename)
        # self.data = self.data[:500]
        self.data = self.convert_to_messages(self.data)
        if self.do_eval:
            self.label_2_query_map = self.get_label_2_query_map(self.data)
        ignore_cnt = len([row for row in self.data if row['ans_full'] == "I don't know."])
        print(f"Ignore cnt: {ignore_cnt}, Total cnt: {len(self.data)}")
            
        self.processor = processor

    def get_label_2_query_map(self, data):
        label_2_query_map = {}
        for row in data:
            query = str(row['query']).strip()
            ans_full = str(row['ans_full']).strip()
            # assert ans_full not in label_2_query_map
            label_2_query_map[ans_full] = query
        return label_2_query_map

    def load_jsonl(self, path):
        with open(path, "r") as f:
            data = [json.loads(line) for line in f.readlines()]
        random.shuffle(data)
        return data
    
    def __len__(self):
        return len(self.data)
    
    def convert_to_messages(self, dataset):
        if self.do_eval:
            return self.convert_to_messages_for_eval(dataset)
        else:
            return self.convert_to_messages_for_train(dataset)
        
    def convert_to_messages_for_train(self, dataset):
        messages = []
        for row in tqdm(dataset):
            image_name = str(row['image']).strip()
            image = Image.open(os.path.join(self.data_path, "images", image_name))
            join_id = row.get('session_id', None)
            if join_id is None:
                join_id = row.get('interaction_id', None)
            if join_id in self.single_need_resize_session_ids and self.single_need_resize_session_ids[join_id]:
                image = image.resize((960, 1280))
            image_path = os.path.join(self.data_path, "images", image_name)

            if 'pre_messages' in row:
                messages.extend(self.convert_task3(row, image, image_path))

            elif 'web_search_info_list' in row:
                messages.extend(self.convert_task2(row, image, image_path))

            else:
                messages.extend(self.convert_retrieval_task1(row, image, image_path))
                # messages.extend(self.convert_task1(row, image, image_path))

        random.shuffle(messages)
        return messages

    def convert_to_messages_for_eval(self, dataset):
        messages = []
        for row in tqdm(dataset):
            image_name = str(row['image']).strip()
            image = Image.open(os.path.join(self.data_path, "images", image_name))
            join_id = row.get('session_id', None)
            if join_id is None:
                join_id = row.get('interaction_id', None)
            if join_id in self.single_need_resize_session_ids and self.single_need_resize_session_ids[join_id]:
                image = image.resize((960, 1280))
            image_path = os.path.join(self.data_path, "images", image_name)

            # task-1 as eval
            if 'pre_messages' not in row and 'web_search_info_list' not in row:
                # messages.extend(self.convert_task1(row, image, image_path))
                messages.extend(self.convert_retrieval_task1(row, image, image_path, close_retrieval=False))

            # if 'pre_messages' not in row and 'web_search_info_list' in row:
            #     messages.extend(self.convert_task2(row, image, image_path, close_retrieval=False))
        return messages
    
    def convert_rerank_task(self, rag_info, query, image, image_path, answer):
        rerank_system_prompt = "You need to determine the content of the search and which info can help you answer the query. The query has and retrieval information already been provided. \n\Output the most relevant Info number List like [x, xx, xxx, ...]. If no relevant items, output []."
            # Output a list of string like ['1', ...], the prefix 'Info' can miss."

        answer = str(answer).strip()
        full_instruction = [
            {
                "role": "system", "content": [
                    {"type": "text", "text": rerank_system_prompt}
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
                    {"type": "text", "text": "\n\nThe retrieval information is:\n"},
                    {"type": "text", "text": rag_info}
                ]
            }
        ] + [
            {
                "role": "user", "content": [
                    {"type": "text", "text": "\n\nPlease rerank the information to ask: " + query}
                ]
            }
        ] + [
            {
                "role": "assistant", "content": [
                    {"type": "text", "text": str(answer).strip()}
                ]
            },
        ]
        context_instruction = [
            {
                "role": "system", "content": [
                    {"type": "text", "text": rerank_system_prompt}
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
                    {"type": "text", "text": "\n\nThe retrieval information is:\n"},
                    {"type": "text", "text": rag_info}
                ]
            }
        ] + [
            {
                "role": "user", "content": [
                    {"type": "text", "text": "\n\nPlease rerank the information to ask: " + query}
                ]
            }
        ]

        return {
            'full_instruction': full_instruction[:8000],
            'context_instruction': context_instruction[:8000],
            'image_path': image_path,
            'ans_full': answer,
            'query': query
        }
    
    def convert_task3(self, row, image, image_path):
        interaction_id = str(row['interaction_id']).strip()
        query = str(row['query']).strip()
        ans_full = str(row['ans_full']).strip()

        messages = []
        if 'jude_list' in row:
            target_rag_index = row['jude_list']['right_info']
            target_rag_index = [i.replace('[Info', '').replace(']', '').strip() for i in target_rag_index]
        else:
            target_rag_index = []
        target_rag_index_answer = target_rag_index
        rag_info_list = row['web_search_info_list']
        rag_info_list = [l for l in rag_info_list if len(l.strip()) >= 1]

        for item_label in target_rag_index:
            item_label = str(item_label).strip()

            if item_label in [
                '1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12', '13', '14', '15',
                '16', '17', '18', '19', '20', '21', '22', '23', '24', '25', '26', '27', '28', '29', '30',
            ]:
                messages.append(self.convert_rerank_task('\n'.join([l[:500] for l in rag_info_list]), query, image, image_path, item_label))
                # break

        reranked_rag_info_list = []
        for right_index in target_rag_index:
            for rag_info_row in rag_info_list:
                if f"[Info {right_index}]" in rag_info_row:
                    reranked_rag_info_list.append(rag_info_row[9:])
        reranked_rag_info_list = [f'[Info {i+1}] ' + reranked_rag_info_list[i] for i in range(len(reranked_rag_info_list))]
        # reranked_rag_info_list = rag_info_list[:10]
        rag_info = '\n'.join(reranked_rag_info_list)
        # else:
        #     rag_info_list = row['web_search_info_list']
        #     rag_info_list = [l for l in rag_info_list if len(l.strip()) >= 1]
        #     rag_info = '\n'.join(rag_info_list)

        pre_messages = row['pre_messages']

        gen_system_prompt = "You are a helpful assistant that truthfully answers user questions about the provided image and the retrieval information.\nThe retrieval information may not related to the provided query and image. \nPlease pay attention to identifying that information and answer the query with image. \nKeep your response concise and to the point. If you don't know the answer, respond with 'I don't know'."

        search_system_prompt = "You are a web retrieval and web search syntax converter. Please generate phrases for retrieval based on the original query and provided image."

        
        # generation task
        full_instruction = [
            {
                "role": "system", "content": [
                    {"type": "text", "text": gen_system_prompt}
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
                    {"type": "text", "text": "\n\nThe retrieval information is:\n"},
                    {"type": "text", "text": rag_info}
                ]
            }
        ] + pre_messages + [
            {
                "role": "user", "content": [
                    {"type": "text", "text": "\n\nPlease ask: " + query}
                ]
            }
        ] + [
            {
                "role": "assistant", "content": [
                    {"type": "text", "text": ans_full}
                ]
            },
        ]
        context_instruction = [
            {
                "role": "system", "content": [
                    {"type": "text", "text": gen_system_prompt}
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
                    {"type": "text", "text": "\n\nThe retrieval information is:\n"},
                    {"type": "text", "text": rag_info}
                ]
            }
        ] + pre_messages + [
            {
                "role": "user", "content": [
                    {"type": "text", "text": "\n\nPlease ask: " + query}
                ]
            }
        ]

        messages.append({
            'full_instruction': full_instruction[:8000],
            'context_instruction': context_instruction[:8000],
            'image_path': image_path,
            'ans_full': ans_full,
            'query': query
        })


        ## task3 data for task2
        # generation task
        full_instruction = [
            {
                "role": "system", "content": [
                    {"type": "text", "text": gen_system_prompt}
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
                    {"type": "text", "text": "\n\nThe retrieval information is:\n"},
                    {"type": "text", "text": rag_info}
                ]
            }
        ] + [
            {
                "role": "user", "content": [
                    {"type": "text", "text": "\n\nPlease ask: " + query}
                ]
            }
        ] + [
            {
                "role": "assistant", "content": [
                    {"type": "text", "text": ans_full}
                ]
            },
        ]
        context_instruction = [
            {
                "role": "system", "content": [
                    {"type": "text", "text": gen_system_prompt}
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
                    {"type": "text", "text": "\n\nThe retrieval information is:\n"},
                    {"type": "text", "text": rag_info}
                ]
            }
        ] + [
            {
                "role": "user", "content": [
                    {"type": "text", "text": "\n\nPlease ask: " + query}
                ]
            }
        ]

        messages.append({
            'full_instruction': full_instruction[:8000],
            'context_instruction': context_instruction[:8000],
            'image_path': image_path,
            'ans_full': ans_full,
            'query': query
        })

        # search retrieval keyword task
        full_instruction = [
            {
                "role": "system",
                "content": search_system_prompt
            },     
        ] + pre_messages + [
            {
                "role": "user", "content":[
                    {"type": "image", "image": image},
                    {"type": "text", "text": f"\n\nThe Origin Query is: {query}\n"},
                    {"type": "text", "text": f"\n\nThe final retrieval search query is:\n"}
                ]
            },
            {
                "role": "assistant", "content": [
                    {"type": "text", "text": str(row['web_search_query'])}
                ]
            },
        ]

        context_instruction = [
            {
                "role": "system",
                "content": search_system_prompt
            },     
        ] + pre_messages + [
            {
                "role": "user", "content":[
                    {"type": "image", "image": image},
                    {"type": "text", "text": f"\n\nThe Origin Query is: {query}\n"},
                    {"type": "text", "text": f"\n\nThe final retrieval search query is:\n"}
                ]
            }
        ]
        messages.append({
            'full_instruction': full_instruction[:8000],
            'context_instruction': context_instruction[:8000],
            'image_path': image_path,
            'ans_full': str(row['web_search_query']),
            'query': query  
        })

        return messages

    def convert_task2(self, row, image, image_path, close_retrieval=True):
        interaction_id = str(row['interaction_id']).strip()
        query = str(row['query']).strip()
        ans_full = str(row['ans_full']).strip()

        gen_system_prompt = "You are a helpful assistant that truthfully answers user questions about the provided image and the retrieval information.\nThe retrieval information may not related to the provided query and image. \nPlease pay attention to identifying that information and answer the query with image. \nKeep your response concise and to the point. If you don't know the answer, respond with 'I don't know'."
        search_system_prompt = "You are a web retrieval and web search syntax converter. Please generate phrases for retrieval based on the original query and provided image."

        messages = []
        if 'jude_list' in row:
            target_rag_index = row['jude_list']['right_info']
            target_rag_index = [i.replace('[Info', '').replace(']', '').strip() for i in target_rag_index]
        else:
            target_rag_index = []
        target_rag_index_answer = target_rag_index
        rag_info_list = row['web_search_info_list']
        rag_info_list = [l for l in rag_info_list if len(l.strip()) >= 1]

        if close_retrieval:
            for item_label in target_rag_index_answer:
                if item_label in [
                    '1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12', '13', '14', '15',
                    '16', '17', '18', '19', '20', '21', '22', '23', '24', '25', '26', '27', '28', '29', '30',
                ]:
                    messages.append(self.convert_rerank_task('\n'.join([l[:500] for l in rag_info_list]), query, image, image_path, item_label))
                    # break

        reranked_rag_info_list = []
        for right_index in target_rag_index:
            for rag_info_row in rag_info_list:
                if f"[Info {right_index}]" in rag_info_row:
                    reranked_rag_info_list.append(rag_info_row[9:])
        reranked_rag_info_list = [f'[Info {i+1}] ' + reranked_rag_info_list[i] for i in range(len(reranked_rag_info_list))]
        # reranked_rag_info_list = rag_info_list[:10]
        rag_info = '\n'.join(reranked_rag_info_list)
        
        # generation task
        full_instruction = [
            {
                "role": "system", "content": [
                    {"type": "text", "text": gen_system_prompt}
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
                    {"type": "text", "text": "\n\nThe retrieval information is:\n"},
                    {"type": "text", "text": rag_info}
                ]
            }
        ] + [
            {
                "role": "user", "content": [
                    {"type": "text", "text": "\n\nPlease ask: " + query}
                ]
            }
        ] + [
            {
                "role": "assistant", "content": [
                    {"type": "text", "text": ans_full}
                ]
            },
        ]
        context_instruction = [
            {
                "role": "system", "content": [
                    {"type": "text", "text": gen_system_prompt}
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
                    {"type": "text", "text": "\n\nThe retrieval information is:\n"},
                    {"type": "text", "text": rag_info}
                ]
            }
        ] + [
            {
                "role": "user", "content": [
                    {"type": "text", "text": "\n\nPlease ask: " + query}
                ]
            }
        ]

        messages.append({
            'full_instruction': full_instruction[:8000],
            'context_instruction': context_instruction[:8000],
            'image_path': image_path,
            'ans_full': ans_full,
            'query': query
        })

        if close_retrieval:
            # search retrieval keyword task
            full_instruction = [
                {
                    "role": "system",
                    "content": search_system_prompt
                },
                {
                    "role": "user", "content":[
                        {"type": "image", "image": image},
                        {"type": "text", "text": f"\n\nThe Origin Query is: {query}\n"},
                        {"type": "text", "text": f"\n\nThe final retrieval search query is:\n"}
                    ]
                },
                {
                    "role": "assistant", "content": [
                        {"type": "text", "text": str(row['web_search_query'])}
                    ]
                },
            ]
            context_instruction = [
                {
                    "role": "system",
                    "content": search_system_prompt
                },
                {
                    "role": "user", "content":[
                        {"type": "image", "image": image},
                        {"type": "text", "text": f"\n\nThe Origin Query is: {query}\n"},
                        {"type": "text", "text": f"\n\nThe final retrieval search query is:\n"}
                    ]
                }
            ]
            messages.append({
                'full_instruction': full_instruction[:8000],
                'context_instruction': context_instruction[:8000],
                'image_path': image_path,
                'ans_full': str(row['web_search_query']),
                'query': query  
            })


        return messages

    def convert_retrieval_task1(self, row, image, image_path, close_retrieval=True):
        interaction_id = str(row['interaction_id']).strip()
        query = str(row['query']).strip()
        ans_full = str(row['ans_full']).strip()

        gen_system_prompt = "You are a helpful assistant that truthfully answers user questions about the provided image and the retrieval information.\nThe retrieval information may not related to the provided query and image. \nPlease pay attention to identifying that information and answer the query with image. \nKeep your response concise and to the point. If you don't know the answer, respond with 'I don't know'."
        search_system_prompt = "You are a web retrieval and web search syntax converter. Please generate phrases for retrieval based on the original query and provided image."

        messages = []
        if 'image_jude_list' in row:
            target_rag_index = row['image_jude_list']['right_info']
            target_rag_index = [i.replace('[Info', '').replace(']', '').strip() for i in target_rag_index]
        else:
            target_rag_index = []
        target_rag_index_answer = target_rag_index
        rag_info_list = row['image_search_info_list']
        rag_info_list = [l[:500] for l in rag_info_list if len(l.strip()) >= 1]

        # if close_retrieval:
        #     temp_labels = []
        #     for item_label in target_rag_index_answer:
        #         if item_label in [
        #             '1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12', '13', '14', '15',
        #             # '16', '17', '18', '19', '20', '21', '22', '23', '24', '25', '26', '27', '28', '29', '30',
        #         ]:
        #             temp_labels.append(item_label)
        #     if len(temp_labels) == 0:
        #         temp_labels = "[]"
        #     else:
        #         temp_labels = "[" + " ".join(temp_labels) +  "]"
        #     messages.append(self.convert_rerank_task('\n'.join([l for l in rag_info_list][:15]), query, image, image_path, temp_labels))
        #             # messages.append(self.convert_rerank_task('\n'.join([l for l in rag_info_list][:15]), query, image, image_path, item_label))
        #             # break

        # reranked_rag_info_list = []
        # for right_index in target_rag_index:
        #     for rag_info_row in rag_info_list:
        #         if f"[Info {right_index}]" in rag_info_row:
        #             reranked_rag_info_list.append(rag_info_row[9:])
        # reranked_rag_info_list = [f'[Info {i+1}] ' + reranked_rag_info_list[i] for i in range(len(reranked_rag_info_list))]
        reranked_rag_info_list = rag_info_list[:5]
        rag_info = '\n'.join(reranked_rag_info_list)
        
        # generation task
        full_instruction = [
            {
                "role": "system", "content": [
                    {"type": "text", "text": gen_system_prompt}
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
                    {"type": "text", "text": "\n\nThe retrieval information is:\n"},
                    {"type": "text", "text": rag_info}
                ]
            }
        ] + [
            {
                "role": "user", "content": [
                    {"type": "text", "text": "\n\nPlease ask: " + query}
                ]
            }
        ] + [
            {
                "role": "assistant", "content": [
                    {"type": "text", "text": ans_full}
                ]
            },
        ]
        context_instruction = [
            {
                "role": "system", "content": [
                    {"type": "text", "text": gen_system_prompt}
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
                    {"type": "text", "text": "\n\nThe retrieval information is:\n"},
                    {"type": "text", "text": rag_info}
                ]
            }
        ] + [
            {
                "role": "user", "content": [
                    {"type": "text", "text": "\n\nPlease ask: " + query}
                ]
            }
        ]

        messages.append({
            'full_instruction': full_instruction[:8000],
            'context_instruction': context_instruction[:8000],
            'image_path': image_path,
            'ans_full': ans_full,
            'query': query
        })

        return messages
    
    def convert_task1(self, row, image, image_path):
        interaction_id = str(row['interaction_id']).strip()
        query = str(row['query']).strip()
        ans_full = str(row['ans_full']).strip()

        gen_system_prompt = "You are a helpful assistant that truthfully answers user questions about the provided image. Keep your response concise and to the point. If you don't know the answer, respond with 'I don't know'."

        messages = []
        full_instruction = [
                {
                    "role": "system", "content": [
                        {"type": "text", "text": gen_system_prompt}
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
        ] + [
            {
                "role": "assistant", "content": [
                    {"type": "text", "text": ans_full}
                ]
            },
        ]
        context_instruction = [
                {
                    "role": "system", "content": [
                        {"type": "text", "text": gen_system_prompt}
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

        messages.append({
            'full_instruction': full_instruction[:8000],
            'context_instruction': context_instruction[:8000],
            'ans_full': ans_full,
            'image_path': image_path,
            'query': query
        })

        return messages
    
    def __getitem__(self, index):
        row = self.data[index]

        context_instruction = row['context_instruction']
        full_instruction = row['full_instruction']
        image_path = row['image_path']
        image = Image.open(image_path)
        join_id = row.get('session_id', None)
        if join_id is None:
            join_id = row.get('interaction_id', None)
        if join_id in self.single_need_resize_session_ids and self.single_need_resize_session_ids[join_id]:
            image = image.resize((960, 1280))
        
        context_text = self.processor.apply_chat_template(context_instruction, add_generation_prompt=True)
        full_text = self.processor.apply_chat_template(full_instruction, add_generation_prompt=True)

        full_inputs = self.processor(
            image,
            full_text,
            add_special_tokens=False,
        )
        full_input_ids = full_inputs.input_ids
        full_attention_mask = full_inputs.attention_mask
        full_pixel_values = full_inputs.pixel_values
        full_aspect_ratio_ids = full_inputs.aspect_ratio_ids
        full_aspect_ratio_mask = full_inputs.aspect_ratio_mask
        full_cross_attention_mask = full_inputs.cross_attention_mask

        context_inputs = self.processor(
            image,
            context_text,
            add_special_tokens=False,
        )
        context_input_ids = context_inputs.input_ids

        if not self.do_eval:
            answer_input_ids = [[-100] * len(context_input_ids[0]) + full_input_ids[0][len(context_input_ids[0]):]]

            target_mask = [0] * len(context_input_ids[0]) + [1] * (len(full_input_ids[0]) - len(context_input_ids[0]))

            assert len(full_attention_mask[0]) == len(full_input_ids[0]) == len(answer_input_ids[0]) == len(target_mask)
            # print(torch.tensor(full_pixel_values).shape)
            return {
                "input_ids": full_input_ids[0],
                "labels": answer_input_ids[0],
                # 'attention_mask': full_attention_mask[0],
                'pixel_values': full_pixel_values[0],
                'aspect_ratio_ids': full_aspect_ratio_ids[0],
                'aspect_ratio_mask': full_aspect_ratio_mask[0],
                'cross_attention_mask': full_cross_attention_mask[0],
                'target_mask': target_mask
            }
        else:
            label = full_text[len(context_text):]
            label_input_ids = self.processor.tokenizer.encode(label, add_special_tokens=False)

            context_input_ids = context_input_ids[0]
            label_input_ids= [-100] * (len(context_input_ids) -  len(label_input_ids)) + label_input_ids

            return {
                "input_ids": context_input_ids[:min(len(context_input_ids), len(label_input_ids))],
                # "attention_mask": full_attention_mask[0],
                "pixel_values": context_inputs.pixel_values[0],
                "aspect_ratio_ids": context_inputs.aspect_ratio_ids[0],
                "aspect_ratio_mask": context_inputs.aspect_ratio_mask[0],
                "cross_attention_mask": context_inputs.cross_attention_mask[0],
                "labels": label_input_ids[:min(len(context_input_ids), len(label_input_ids))]
            }
    
@dataclass
class DataCollatorForKDD:
    processor: AutoProcessor = None
    model: Optional[Any] = None
    padding: Union[bool, str, PaddingStrategy] = True
    max_length: Optional[int] = None
    pad_to_multiple_of: Optional[int] = None
    label_pad_token_id: int = -100
    return_tensors: str = "pt"
    
    def __call__(self, features, return_tensors=None):
        if return_tensors is None:
            return_tensors = self.return_tensors
        labels = [feature["labels"] for feature in features] if "labels" in features[0].keys() else None
        
        if labels is not None:
            max_label_length = max(len(l) for l in labels)
            if self.pad_to_multiple_of is not None:
                max_label_length = (
                    (max_label_length + self.pad_to_multiple_of - 1)
                    // self.pad_to_multiple_of
                    * self.pad_to_multiple_of
                )

            padding_side = self.processor.tokenizer.padding_side
            for feature in features:
                remainder = [self.label_pad_token_id] * (max_label_length - len(feature["labels"]))
                if isinstance(feature["labels"], list):
                    feature["labels"] = (
                        feature["labels"] + remainder if padding_side == "right" else remainder + feature["labels"]
                    )
                elif padding_side == "right":
                    feature["labels"] = np.concatenate([feature["labels"], remainder]).astype(np.int64)
                else:
                    feature["labels"] = np.concatenate([remainder, feature["labels"]]).astype(np.int64)
        # breakpoint()
        features = self.processor.tokenizer.pad(
            features,
            padding=True,
            # max_length=max_label_length,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors=return_tensors,
        )

        # prepare decoder_input_ids
        if (
            labels is not None
            and self.model is not None
            and hasattr(self.model, "prepare_decoder_input_ids_from_labels")
        ):
            decoder_input_ids = self.model.prepare_decoder_input_ids_from_labels(labels=features["labels"])
            features["decoder_input_ids"] = decoder_input_ids
        return features    