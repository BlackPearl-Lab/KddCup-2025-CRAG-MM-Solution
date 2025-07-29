#!/usr/bin/env python
# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team
import argparse
import json
import pickle
import time
import os
import math
import sys
import pickle as pkl
import pandas as pd
from torch import nn, Tensor
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss
from tqdm.auto import tqdm
import torch
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from torch.utils.data.distributed import DistributedSampler
from transformers import set_seed, AutoConfig, AutoModel, MistralPreTrainedModel, MistralConfig, DynamicCache, \
    Cache,Qwen2Model,AutoModelForSequenceClassification
import random
import numpy as np
from transformers import (
    EvalPrediction,
    MllamaForConditionalGeneration,
    AutoProcessor,
    AutoModelForCausalLM,
    AutoTokenizer,
    SchedulerType,
    default_data_collator,
    get_scheduler,
    BitsAndBytesConfig,
    AutoConfig,
    GenerationConfig
)
import requests
from typing import Any, Callable, Dict, List, NewType, Optional, Tuple, Union
from dataclasses import dataclass
import bitsandbytes as bnb
from transformers import optimization
import deepspeed
from deepspeed.ops.adam import DeepSpeedCPUAdam, FusedAdam
from PIL import Image
import torch.distributed as dist
from transformers.utils import PaddingStrategy
from peft import (
    LoraConfig,
    get_peft_model,
    get_peft_model_state_dict,
    # prepare_model_for_int8_training,
    set_peft_model_state_dict,TaskType
)
from transformers.modeling_attn_mask_utils import _prepare_4d_causal_attention_mask_for_sdpa, \
    _prepare_4d_causal_attention_mask
from transformers.modeling_outputs import BaseModelOutputWithPast,SequenceClassifierOutputWithPast
# from transformers.models.mistral.modeling_flax_mistral import MISTRAL_INPUTS_DOCSTRING
from transformers.models.mistral.modeling_mistral import MistralDecoderLayer, MistralRMSNorm
from transformers.utils import add_start_docstrings_to_model_forward


from peft import prepare_model_for_kbit_training
sys.path.append(
    os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir)))
from torch.utils.data import Dataset
import torch.nn as nn
import torch.nn.functional as F

from dataclasses import dataclass
from typing import List, Tuple, Optional, Union

from transformers import DataCollatorWithPadding, PreTrainedTokenizer
import re

IGNORE_INDEX = -100


def is_rank_0() -> bool:
    return not dist.is_initialized() or dist.get_rank() == 0


GLOBAL_BATCH_SIZE = 8
MICRO_BATCH_SIZE = 1


def get_train_ds_config(offload,
                        stage=2,
                        enable_hybrid_engine=False,
                        inference_tp_size=1,
                        release_inference_cache=False,
                        pin_parameters=True,
                        tp_gather_partition_size=8,
                        max_out_tokens=512,
                        enable_tensorboard=False,
                        enable_mixed_precision_lora=False,
                        tb_path="",
                        tb_name=""):
    device = "cpu" if offload else "none"
    zero_opt_dict = {
        "stage": stage,
        "offload_param": {
            "device": device
        },
        "offload_optimizer": {
            "device": device
        },
        "stage3_param_persistence_threshold": 1e4,
        "stage3_max_live_parameters": 3e7,
        "stage3_prefetch_bucket_size": 3e7,
        "memory_efficient_linear": False
    }
    if enable_mixed_precision_lora:
        zero_opt_dict["zero_quantized_nontrainable_weights"] = True
        zero_opt_dict["zero_hpz_partition_size"] = torch.cuda.device_count()
    return {
        "train_batch_size": GLOBAL_BATCH_SIZE,
        "train_micro_batch_size_per_gpu": MICRO_BATCH_SIZE,
        "steps_per_print": 40,
        "zero_optimization": zero_opt_dict,
         "fp16": {
            "enabled": False,
            "loss_scale_window": 100,
             "min_loss_scale":0.0001,
        },
        "bfloat16": {
            "enabled": True,
            "loss_scale_window": 100
        },
        "gradient_clipping": 1.0,
        "prescale_gradients": False,
        "wall_clock_breakdown": False,
        "hybrid_engine": {
            "enabled": enable_hybrid_engine,
            "max_out_tokens": max_out_tokens,
            "inference_tp_size": inference_tp_size,
            "release_inference_cache": release_inference_cache,
            "pin_parameters": pin_parameters,
            "tp_gather_partition_size": tp_gather_partition_size,
        },
        "tensorboard": {
            "enabled": enable_tensorboard,
            "output_path": f"{tb_path}/ds_tensorboard_logs/",
            "job_name": f"{tb_name}_tensorboard"
        }
    }


def get_optimizer_grouped_parameters(
        model,
        weight_decay,
        lora_lr=5e-4,
        no_decay_name_list=["bias", "LayerNorm.weight"],
        lora_name_list=["lora_right_weight", "lora_left_weight"],
):
    optimizer_grouped_parameters = [
        {
            "params": [
                p for n, p in model.named_parameters()
                if (not any(nd in n for nd in no_decay_name_list)
                    and p.requires_grad and not any(nd in n
                                                    for nd in lora_name_list))
            ],
            "weight_decay":
                weight_decay,
        },
        {
            "params": [
                p for n, p in model.named_parameters()
                if (not any(nd in n for nd in no_decay_name_list)
                    and p.requires_grad and any(nd in n
                                                for nd in lora_name_list))
            ],
            "weight_decay":
                weight_decay,
            "lr":
                lora_lr
        },
        {
            "params": [
                p for n, p in model.named_parameters()
                if (any(nd in n
                        for nd in no_decay_name_list) and p.requires_grad)
            ],
            "weight_decay":
                0.0,
        },
    ]
    if not optimizer_grouped_parameters[1]["params"]:
        optimizer_grouped_parameters.pop(1)
    return optimizer_grouped_parameters


def print_rank_0(msg, log_file="log.txt", rank=0):
    if rank <= 0:
        with open(log_file, 'a') as f:
            print(msg)
            f.write(msg + '\n')
            
def print_rank_00(msg, rank=0):
    if rank <= 0:
        
        print(msg)
           
        
        


def to_device(batch, device):
    output = {}
    for k, v in batch.items():
        try:
            output[k] = v.to(device)
        except:
            output[k] = v
    return output


def set_random_seed(seed):
    if seed is not None:
        set_seed(seed)
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)




class qWenSFTDataset(Dataset):
    def __init__(
        self, data_path, processor, max_prompt_len=8000, max_completion_len=100
    ):
        
        with open(data_path, "rb")  as f:
            self.data = pkl.load(f)
        self.input_ids_list = []
        self.labels_list = []
        self.target_mask_list = []
        self.pixel_values_list = []
        self.aspect_ratio_ids_list = []
        self.aspect_ratio_mask_list = []
        self.cross_attention_mask_list = []
        self.logits = []
        self.processor = processor
        self.max_prompt_len = max_prompt_len
        self.max_completion_len = max_completion_len
        
        
        for idx, row in enumerate(tqdm(self.data)):
            full_text, context_text, image_path, query, ans_full, logits_90b = row
            image = Image.open(image_path)#.convert("RGB")
            full_text = self.processor.apply_chat_template(full_text, add_generation_prompt=True)
            context_text = self.processor.apply_chat_template(context_text, add_generation_prompt=True)
            full_inputs = self.processor(
                image,
                full_text,
                add_special_tokens=False,
            )
            full_input_ids = full_inputs.input_ids
            # full_attention_mask = full_inputs.attention_mask
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

            answer_input_ids = [[-100] * len(context_input_ids[0]) + full_input_ids[0][len(context_input_ids[0]):]]

            target_mask = [0] * len(context_input_ids[0]) + [1] * (len(full_input_ids[0]) - len(context_input_ids[0]))

            self.input_ids_list.append(full_input_ids[0])
            self.labels_list.append(answer_input_ids[0])
            self.target_mask_list.append(target_mask)
            self.pixel_values_list.append(full_pixel_values[0])
            self.aspect_ratio_ids_list.append(full_aspect_ratio_ids[0])
            self.aspect_ratio_mask_list.append(full_aspect_ratio_mask[0])
            self.cross_attention_mask_list.append(full_cross_attention_mask[0])
            self.logits.append(np.float16(logits_90b))

            del image
        
            
    
    def __len__(self):
        length = len(self.input_ids_list)
        return length

    def __getitem__(self, idx):
        
        return {
            "input_ids": self.input_ids_list[idx],
            "pixel_values": self.pixel_values_list[idx],
            "aspect_ratio_ids": self.aspect_ratio_ids_list[idx],
            "aspect_ratio_mask": self.aspect_ratio_mask_list[idx],
            "cross_attention_mask": self.cross_attention_mask_list[idx],
            "labels": self.labels_list[idx],
            'target_mask': self.target_mask_list[idx],
            "logits_90b":self.logits[idx]
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


def find_all_linear_names(model, train_mode):
    """
    找出所有全连接层，为所有全连接添加adapter
    """
    assert train_mode in ['lora', 'qlora']
    cls = bnb.nn.Linear4bit if train_mode == 'qlora' else nn.Linear
    lora_module_names = set()
    for name, module in model.named_modules():
        if isinstance(module, cls):
            names = name.split('.')
            lora_module_names.add(names[0] if len(names) == 1 else names[-1])

    if 'lm_head' in lora_module_names:  # needed for 16-bit
        lora_module_names.remove('lm_head')
    lora_module_names = list(lora_module_names)
    print_rank_00(f'LoRA target module names: {lora_module_names}')
    return lora_module_names

def main():
    # print(sys.executable)  # 打印当前环境的 Python 解析器目录
    args = parse_args()
    log_file = os.path.join(args.output_dir,'print_log.txt')

    # print(torch.cuda.is_available())
    # print(torch.cuda.device_count())

    if args.local_rank == -1:
        device = torch.device("cuda")
    else:
        torch.cuda.set_device(args.local_rank)
        device = torch.device("cuda", args.local_rank)
        # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
        # torch.distributed.init_process_group(backend='nccl')
        deepspeed.init_distributed()

    args.global_rank = torch.distributed.get_rank()
    ds_config = get_train_ds_config(offload=args.offload,
                                    stage=args.zero_stage,
                                    enable_tensorboard=args.enable_tensorboard,
                                    tb_path=args.tensorboard_path,
                                    tb_name="step1_model")
    ds_config[
        'train_micro_batch_size_per_gpu'] = args.per_device_train_batch_size
    ds_config[
        'train_batch_size'] = args.per_device_train_batch_size * torch.distributed.get_world_size(
    ) * args.gradient_accumulation_steps
    print(ds_config)

    # If passed along, set the training seed now.
    set_random_seed(args.seed)

    torch.distributed.barrier()

    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, trust_remote_code=True)
    processor = AutoProcessor.from_pretrained(args.model_name_or_path)
    
    
    # bnb_config = BitsAndBytesConfig(
    #     load_in_4bit=True,
    #     bnb_4bit_quant_type="nf4",
    #     bnb_4bit_compute_dtype=torch.bfloat16,
    #     bnb_4bit_use_double_quant=True,
    # )
    
    
    model = MllamaForConditionalGeneration.from_pretrained(
        args.model_name_or_path, trust_remote_code=True, torch_dtype=torch.bfloat16,
        #    quantization_config=bnb_config
    )
    # model90b = load_model1()
    # model.config.use_cache = False
    model.config.keys_to_ignore_at_inference = ["past_key_values"]
    model.config.pad_token_id = tokenizer.pad_token_id
    # model.config.pad_token_id = tokenizer.eos_token_id
    # tokenizer.pad_token_id = tokenizer.eos_token_id
    print("pad token id :", model.config.pad_token_id)
    model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)
    # target_modules = find_all_linear_names(model, "qlora")
    
    config = LoraConfig(
            r=128,
            lora_alpha=256,
            target_modules="^(?!.*vision_model).*(?:o_proj|up_proj|v_proj|down_proj|k_proj|q_proj|gate_proj).*",
            # target_modules=[
            # "q_proj",
            # "k_proj",
            # "v_proj",
            # "o_proj",
            # "gate_proj",
            # "up_proj",
            # "down_proj",
            # "lm_head",
            # # "score",
            # ],
            bias="none",
            lora_dropout=0.05,  # Conventional
            task_type="CAUSAL_LM",
        )
    
    
    model = get_peft_model(model, config)
    
    if args.lora_path != "none":
        print("load pretrain")
        model.load_state_dict(torch.load(args.lora_path), strict=False)
    
    # for name, param in model.named_parameters():
        # if "lora" in name or "lm_head" in name:
        #     # print(name)
        # param.requires_grad = True
    model.print_trainable_parameters()
    
    train_dataset = qWenSFTDataset(args.train_path, processor)
    print(train_dataset.__getitem__(0))
    
    # eval_dataset = qWenSFTDataset(args.val_path,args,tokenizer)
    
    
    collate_fn = DataCollatorForKDD(
        processor,
        model=model,
        label_pad_token_id=-100,
        pad_to_multiple_of=None,
        padding=False,
    )

    # DataLoaders creation:
    if args.local_rank == -1:
        train_sampler = RandomSampler(train_dataset)
        # eval_sampler = SequentialSampler(eval_dataset)
        
    else:
        train_sampler = DistributedSampler(train_dataset)
        # eval_sampler = DistributedSampler(eval_dataset)

    train_dataloader = DataLoader(train_dataset,
                                  shuffle=(train_sampler is None),
                                  collate_fn=collate_fn,
                                  sampler=train_sampler,
                                  batch_size=args.per_device_train_batch_size,
                                  pin_memory=True)

    
#     eval_dataloader = DataLoader(eval_dataset,
#                                  shuffle=(eval_sampler is None),
#                                  collate_fn=collate_fn,
#                                  sampler=eval_sampler,
#                                  batch_size=args.per_device_eval_batch_size,
#                                  pin_memory=True)
    
    def evaluation(model, eval_dataloader):

        step_bar = tqdm(range(len(eval_dataloader)),
                        desc=f'dev steps')

        model.eval()
        predicts = []
        labels = []
        for step, batch in enumerate(eval_dataloader):
            batch = to_device(batch, device)
            with torch.no_grad():
                outputs = model(**batch, use_cache=False)
                logits = outputs.logits
                label = batch['labels']
                logits = F.softmax(logits, dim=-1)
                logits = logits.float()
            predicts.extend([a.reshape(1, -1) for a in logits.detach().cpu().numpy()])
            labels.extend([a.reshape(-1) for a in label.detach().cpu().numpy()])
            step_bar.update()
        predicts = np.concatenate(predicts)
        labels = np.concatenate(labels)
        model.train()
        # loss = log_loss(labels, predicts)
        print("loss score:", loss, "number:", len(predicts))
        return loss
    
    

    # Split weights in two groups, one with weight decay and the other not.
    optimizer_grouped_parameters = get_optimizer_grouped_parameters(
        model, args.weight_decay, args.lora_learning_rate)

    AdamOptimizer = DeepSpeedCPUAdam if args.offload else FusedAdam
    optimizer = AdamOptimizer(optimizer_grouped_parameters,
                              lr=args.learning_rate,
                              betas=(0.9, 0.95))

    num_update_steps_per_epoch = math.ceil(
        len(train_dataloader) / args.gradient_accumulation_steps)

    max_steps = math.ceil(args.num_train_epochs * num_update_steps_per_epoch)
    lr_scheduler = get_scheduler(
        name=args.lr_scheduler_type,
        optimizer=optimizer,
        num_warmup_steps=math.ceil(max_steps * 0.03) if args.num_warmup_steps == 0 else args.num_warmup_steps,
        num_training_steps=args.num_train_epochs * num_update_steps_per_epoch,
    )
    # lr_scheduler = optimization.get_constant_schedule_with_warmup(optimizer,
    #                                                               num_warmup_steps=math.ceil(max_steps * 0.1) if args.num_warmup_steps == 0 else args.num_warmup_steps)

    model, optimizer, _, lr_scheduler = deepspeed.initialize(
        model=model,
        optimizer=optimizer,
        args=args,
        config=ds_config,
        lr_scheduler=lr_scheduler,
        dist_init_required=True)

    # Train!
    print_rank_0("***** Running training *****", log_file,args.global_rank)
    print_rank_0(
        f"***** Evaluating perplexity, Epoch {0}/{args.num_train_epochs} *****",log_file,
        args.global_rank)
    # perplexity = evaluation(model, eval_dataloader)
    # print_rank_0(f"ppl: {perplexity}", args.global_rank)

    total_steps = len(train_dataloader) * args.num_train_epochs
    final_score = 0.0
    best_val_loss = 1000.
    no_improve_epoch = 0.
    global_step = -1
    time_start = time.time()
    loss_fct = nn.CrossEntropyLoss(reduction='mean')
    divergence_loss_fn = nn.KLDivLoss(reduction='batchmean')
    T = 1.
    for epoch in range(args.num_train_epochs):
        
        print_rank_0(
            f"Beginning of Epoch {epoch + 1}/{args.num_train_epochs}, Total Micro Batches {len(train_dataloader)}",log_file,
            args.global_rank)
        
        for step, batch in enumerate(train_dataloader):
            model.train()
            global_step += 1

            
            batch = to_device(batch, device)
            labels = batch['labels']
            logits_90b = batch['logits_90b'].clone().view( -1, 128256) 
            del batch['logits_90b']
            # del batch['labels']

            
            ######
            output = model(**batch,use_cache=False)
            logits = output.logits
            ce_loss = output.loss

            bs = batch['target_mask'].shape[0]
            mask_expanded = batch['target_mask'].bool().unsqueeze(-1).expand_as(logits)  # (bs, n, d)
            logits = torch.masked_select(logits, mask_expanded).view( -1, 128256)  # shape: (bs * x * d, )

            

            # assert logits_90b.shape == logits.shape
            loss_dist = divergence_loss_fn(
                F.log_softmax(logits / T, dim=1),
                F.softmax(logits_90b / T, dim=1)
            )
            loss = ce_loss*0.2+loss_dist*1.0

            # if step==0:
            #     print("---------------")
            #     print(logits_90b.shape)
            #     print(logits.shape)
            #     print(ce_loss)
            #     print(loss_dist)
            #     print(F.softmax(logits_90b))
            #     print(F.log_softmax(logits))
            #     print("****************")
            
            
            model.backward(loss)
            # torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            # if (global_step + 1) % args.gradient_accumulation_steps == 0:
            model.step()
            if global_step % 10 == 0:
                time_end = time.time()
                total_time = time_end - time_start  # 计算运行总时间
                time_start = time_end
                print_rank_0(
                    f"Beginning of Epoch {epoch + 1}/{args.num_train_epochs}, curr_step:{global_step}/{total_steps} curr_loss {loss.item()} ce_loss {ce_loss.item()} dist_loss {loss_dist.item()} lr:{lr_scheduler.get_last_lr()[0]} use time:{total_time}s",
                    log_file,args.global_rank)
                
            if global_step % 200 == 0:
                save_model(args, model, tokenizer, f"step_{global_step}_model")
                
            # if args.save_batch_steps and (global_step + 1) % args.save_batch_steps == 0:
            #     loss_mean = evaluation(model, eval_dataloader)
            #     if torch.distributed.get_rank() == 0 or args.zero_stage == 3 or True:
            #         print_rank_0(
            #             f"***** Evaluating Loss, Epoch {epoch + 1}/{args.num_train_epochs}---{global_step}/{total_steps}*****",
            #             args.global_rank)
            #         print_rank_0(f"loss: {loss_mean}", args.global_rank)
            #     if loss_mean < best_val_loss:
            #         print_rank_0(
            #             f"val_log----epoch:{epoch},batch:{global_step + 1},save model from {best_val_loss} to {loss_mean} !!!",
            #             args.global_rank)
            #         save_model(args, model, tokenizer, f"best_val_loss_model")
            #         best_val_loss = loss_mean
            #         no_improve_epoch = 0
            #     else:
            #         no_improve_epoch += 1
            #         print_rank_0(
            #             f"val_log----epoch:{epoch},batch:{global_step + 1},no_improve_epoch:{no_improve_epoch},curr_val_loss {loss_mean} best_val_loss {best_val_loss} !!!"
            #             , args.global_rank)
            #     if args.earystop and no_improve_epoch == args.eary_stop_epoch:
            #         print_rank_0(
            #             f"val_log----epoch:{epoch},batch:{global_step + 1} eary stop,best_val_loss {best_val_loss} !!!",
            #             args.global_rank)
            #         return
                
        if args.save_per_epoch == 1:
            save_model(args, model, tokenizer, f"epoch_{epoch}_model")
            
        
            
        # 保存最后一轮
        if epoch == args.num_train_epochs - 1:
            save_model(args, model, tokenizer, f"epoch_{epoch}_model")
        model.tput_timer.update_epoch_count()


def save_model(args, model, tokenizer, sub_fold=None):
    if sub_fold is not None:
        output_dir = os.path.join(args.output_dir, sub_fold)
        print_rank_00('saving model ...', args.global_rank)
        tokenizer.save_pretrained(output_dir)
        # model = convert_lora_to_linear_layer(model)
        if args.global_rank == 0:
            model_to_save = model.module if hasattr(model, 'module') else model
            model_to_save.save_pretrained(output_dir)

            # CONFIG_NAME = "config.json"
            # WEIGHTS_NAME = "adapter.bin"
            # os.makedirs(output_dir, exist_ok=True)
            # output_model_file = os.path.join(output_dir, WEIGHTS_NAME)
            # output_config_file = os.path.join(output_dir, CONFIG_NAME)
            # save_dict = model_to_save.state_dict()
            # final_d = {}
            # for k, v in save_dict.items():
            #     if "lora" in k or "score" in k:
            #         final_d[k] = v
            # torch.save(final_d, output_model_file)

        print_rank_00('saving success ...', args.global_rank)


def parse_args():
    parser = argparse.ArgumentParser(
        description=
        "Finetune a transformers model on a causal language modeling task")

    ## Tensorboard logging
    parser.add_argument('--enable_tensorboard',
                        action='store_true',
                        help='Enable tensorboard logging')
    parser.add_argument('--tensorboard_path',
                        type=str,
                        default="step1_tensorboard")
    

    parser.add_argument('--save_batch_steps', type=int, default=1000)
    parser.add_argument('--earystop', type=bool, default=False)
    parser.add_argument('--eary_stop_epoch', type=int, default=2)
    parser.add_argument('--save_per_epoch', type=int, default=-1)
    parser.add_argument('--project_name', type=str, default='Coati', help="wandb project name")
    parser.add_argument('--train_path', type=str, default=None, help="train data path ")
    parser.add_argument('--val_path', type=str, default=None, help="doc data path ")
    parser.add_argument('--lora_path', type=str, default='none', help="lora path")
    parser.add_argument(
        "--model_name_or_path",
        type=str,
        help=
        "Path to pretrained model or model identifier from huggingface.co/models.",
        required=True,
    )
    parser.add_argument(
        "--per_device_train_batch_size",
        type=int,
        default=1,
        help="Batch size (per device) for the training dataloader.",
    )
    parser.add_argument(
        "--per_device_eval_batch_size",
        type=int,
        default=1,
        help="Batch size (per device) for the evaluation dataloader.",
    )
    parser.add_argument(
        "--max_prompt_len",
        type=int,
        default=512,
        help="The maximum prompt sequence length.",
    )
    
    parser.add_argument(
        "--max_completion_len",
        type=int,
        default=512,
        help="The maximum answer sequence length.",
    )
    
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=1e-3,
        help=
        "Initial learning rate (after the potential warmup period) to use.",
    )
    parser.add_argument("--weight_decay",
                        type=float,
                        default=0.,
                        help="Weight decay to use.")
    parser.add_argument("--num_train_epochs",
                        type=int,
                        default=1,
                        help="Total number of training epochs to perform.")
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help=
        "Number of updates steps to accumulate before performing a backward/update pass.",
    )
    parser.add_argument(
        "--lr_scheduler_type",
        type=SchedulerType,
        default="cosine",
        help="The scheduler type to use.",
        choices=[
            "linear", "cosine", "cosine_with_restarts", "polynomial",
            "constant", "constant_with_warmup"
        ],
    )
    parser.add_argument(
        "--num_warmup_steps",
        type=int,
        default=0,
        help="Number of steps for the warmup in the lr scheduler.")
    parser.add_argument("--output_dir",
                        type=str,
                        default=None,
                        help="Where to store the model.")
    parser.add_argument("--seed",
                        type=int,
                        default=1234,
                        help="A seed for reproducible training.")
    parser.add_argument("--local_rank",
                        type=int,
                        default=-1,
                        help="local_rank for distributed training on gpus")
    parser.add_argument(
        "--lora_learning_rate",
        type=float,
        default=5e-4,
        help=
        "Initial LoRA learning rate (after the potential warmup period) to use."
    )
    parser.add_argument('--gradient_checkpointing',
                        action='store_true',
                        help='Enable HF gradient checkpointing for model.')
    parser.add_argument('--disable_dropout',
                        action='store_true',
                        help='Disable the dropout of the model.')
    # deepspeed features
    parser.add_argument('--offload',
                        action='store_true',
                        help='Enable ZeRO Offload techniques.')
    parser.add_argument(
        '--zero_stage',
        type=int,
        default=0,
        help='ZeRO optimization stage for Actor model (and clones).')
    ## LoRA for efficient training setting
    parser.add_argument("--lora_dim",
                        type=int,
                        default=0,
                        help="If > 0, use LoRA for efficient training.")
    parser.add_argument("--lora_config",
                        type=str,
                        default="./configs/lora_config_llama.json",
                        help="If > 0, use LoRA for efficient training.")

    parser.add_argument("--lora_module_name",
                        type=str,
                        default="decoder.layers.",
                        help="The scope of LoRA.")
    parser.add_argument('--only_optimize_lora',
                        action='store_true',
                        help='Only optimize the LoRA parameters.')
    parser = deepspeed.add_config_arguments(parser)
    args = parser.parse_args()

    # Validate settings
    if args.gradient_checkpointing and args.lora_dim > 0:
        assert (
            not args.only_optimize_lora
        ), "--gradient_checkpointing and --only_optimize_lora cannot be enabled at the same time."

    return args


if __name__ == "__main__":
    main()
