import logging
import os
import sys
import warnings
import pandas as pd
import json
import transformers
import numpy as np
from functools import partial

os.environ["WANDB_MODE"] = "disabled"
from transformers import (
    EvalPrediction,
    MllamaForConditionalGeneration,
    AutoProcessor,
    set_seed,
    AutoConfig,
    HfArgumentParser,
    GenerationConfig
)
from trainer import LoRATrainer
from arguments import ModelArguments, DataTrainingArguments, GLMTrainingArguments
from peft import get_peft_model, LoraConfig, TaskType, prepare_model_for_kbit_training, TaskType
from data_utils import DataCollatorForKDD, KDDSingleTurnDataSet, KDDSingleTurnTask2DataSet, KDDMultipleTurnTask3DataSet, KDDCombDataset
import torch
from peft.utils import TRANSFORMERS_MODELS_TO_LORA_TARGET_MODULES_MAPPING
from sklearn.metrics import accuracy_score, log_loss

from metric import single_turn_evaluate
from loss import generalLMLoss

logger = logging.getLogger(__name__)
IGNORE_INDEX = -100



def main():
    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, GLMTrainingArguments))
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        model_args, data_args, training_args = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))
    else:
        model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    
    training_args.ddp_find_unused_parameters = False
    training_args.save_safetensors = False

    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )

    if training_args.should_log:
        transformers.utils.logging.set_verbosity_info()

    log_level = training_args.get_process_log_level()
    logger.setLevel(log_level)
    transformers.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.enable_default_handler()
    transformers.utils.logging.enable_explicit_format()

    # Log on each process the small summary:
    logger.warning(
        f"Process rank: {training_args.local_rank}, device: {training_args.device}, n_gpu: {training_args.n_gpu}"
        + f"distributed training: {bool(training_args.local_rank != -1)}, 16-bits training: {training_args.fp16}"
    )
    logger.info(f"Training/evaluation parameters {training_args}")
    set_seed(training_args.seed)

    model = MllamaForConditionalGeneration.from_pretrained(
        model_args.model_name_or_path,
        trust_remote_code=True
        # torch_dtype=torch.bfloat16,
        # device_map="auto",
    )
    model.config.use_cache = False
    model.config.keys_to_ignore_at_inference = ["past_key_values"]
    training_args.generation_config = model.generation_config


    processor = AutoProcessor.from_pretrained(model_args.model_name_or_path)
    peft_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        inference_mode=False,
        r=model_args.lora_rank,
        # target_modules=['q_proj', 'k_proj', 'v_proj', 'o_proj'],
        # target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj", "embed_tokens", "lm_head"],
        # target_modules="^(?!.*vision_model).*(?:o_proj|up_proj|v_proj|down_proj|k_proj|q_proj|gate_proj).*",
        target_modules="all-linear",
        lora_alpha=model_args.lora_alpha,
        lora_dropout=model_args.lora_dropout,
    )
    model = get_peft_model(model, peft_config).to("cuda")

    # if 'comb' in data_args.train_format:
    if 'comb' in data_args.train_format:
        train_dataset = KDDCombDataset(
            data_args.train_data,
            data_args.train_format,
            processor
        )
        dev_dataset = KDDCombDataset(
            data_args.eval_data,
            data_args.train_format,
            processor,
            do_eval=True
        )
        label_2_query_map = dev_dataset.label_2_query_map
    elif 'task2' in data_args.train_format:
        train_dataset = KDDSingleTurnTask2DataSet(
            data_args.train_data,
            data_args.train_format,
            processor
        )
        dev_dataset = KDDSingleTurnTask2DataSet(
            data_args.eval_data,
            data_args.train_format,
            processor,
            do_eval=True
        )
        label_2_query_map = dev_dataset.label_2_query_map
        
    elif 'task3' in data_args.train_format:
        train_dataset = KDDMultipleTurnTask3DataSet(
            data_args.train_data,
            data_args.train_format,
            processor
        )
        dev_dataset = KDDMultipleTurnTask3DataSet(
            data_args.eval_data,
            data_args.train_format,
            processor,
            do_eval=True
        )
        label_2_query_map = dev_dataset.label_2_query_map
    else:
        train_dataset = KDDSingleTurnDataSet(
            data_args.train_data,
            data_args.train_format,
            processor
        )
        dev_dataset = KDDSingleTurnDataSet(
            data_args.eval_data,
            data_args.train_format,
            processor,
            do_eval=True
        )
        label_2_query_map = dev_dataset.label_2_query_map

    data_collator = DataCollatorForKDD(
        processor,
        model=model,
        label_pad_token_id=-100,
        pad_to_multiple_of=None,
        padding=False,
    )

    # Define compute metric function
    def compute_kdd_metrics(eval_preds: EvalPrediction, task_type):
        labels = eval_preds.label_ids
        preds = eval_preds.predictions

        # Ignore any token with -100 label in processed texts
        labels = np.where(labels != IGNORE_INDEX, labels, processor.tokenizer.pad_token_id)
        outputs = np.where(preds != IGNORE_INDEX, preds, processor.tokenizer.pad_token_id)

        pred_strs = []
        for output in outputs:
            try:
                pred_strs.append(
                    processor.tokenizer.decode(output, skip_special_tokens=True).strip().split('assistant')[-1].strip()
                )
            except Exception as e:
                warnings.warn(
                    f"Error in decoding output {output} with exception: {e} with token_pad_id: {processor.tokenizer.pad_token_id}")
                logger.info(
                    f"Error in decoding output {output} with exception: {e} with token_pad_id: {processor.tokenizer.pad_token_id}")
                pred_strs.append("")
        label_strs = processor.batch_decode(labels, group_tokens=False, skip_special_tokens=True)

        all_examples = []

        # print(label_2_query_map)
        for i in range(len(pred_strs)):
            pred_ = pred_strs[i]
            label_ = label_strs[i].strip()
            if label_.endswith('assistant'):
                label_ = label_[:-len('assistant')].strip()
            # 
            try:
                assert label_ in label_2_query_map
                # if label_ in label_2_query_map
                query = label_2_query_map[label_]
                all_examples.append({
                    "agent_response": pred_, 
                    "ground_truth": label_, 
                    "query": query
                })
            except:
                # print(label_)
                # print(label_2_query_map)
                print('missing : ', label_)
        results = single_turn_evaluate(all_examples)
        # print(all_examples)
        
        return {
            'correct_exact': results['correct_exact'],
            'correct': results['correct'],
            'miss': results['miss'],
            'total': results['total'],
            'exact_accuracy': results['exact_accuracy'],
            'accuracy': results['accuracy'],
            'missing': results['missing'],
            'hallucination': results['hallucination'],
            'score': results['score'],
        }
        # return {}
    
    compute_metrics = partial(compute_kdd_metrics, task_type=TaskType.CAUSAL_LM)

    training_args.predict_with_generate = True
    training_args.generation_config = GenerationConfig.from_pretrained(model_args.model_name_or_path)
    training_args.generation_max_length = 8000
    # print(training_args)

    trainer = LoRATrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=dev_dataset,
        tokenizer=processor.tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
        # compute_loss=loss_func,
        # preprocess_logits_for_metrics=preprocess_logits_for_metrics
    )

    checkpoint = None
    if training_args.resume_from_checkpoint is not None:
        checkpoint = training_args.resume_from_checkpoint
    model.gradient_checkpointing_enable()
    model.enable_input_require_grads()
    trainer.train(resume_from_checkpoint=checkpoint)
    trainer.save_model()  # Saves the tokenizer too for easy upload
    trainer.save_state()


if __name__ == "__main__":
    main()