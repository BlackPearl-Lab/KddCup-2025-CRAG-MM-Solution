#!/bin/bash
source activate YOUR_ENV

set -ex

LR=5e-5
NUM_GPUS=8
LORA_RANK=128
LORA_ALPHA=256
LORA_DROUPOUT=0.05
WARMUP_RATIO=0.05
SAVE_INTERVAL=50
EPOCH=10

PROJECT_NAME=comb_task1_11b_image_search_top10_labeled_shift
PATH_PRE=/checkpoints

MODEL_PATH=meta-llama/Llama-3.2-11B-Vision-Instruct
TAIN_DATA_ROOT=/dataset/v0.1.2/comb
EVAL_DATA_ROOT=/dataset/v0.1.2/comb

date="20250605"
model_name="llama32_11b_eval_task1_resize_image"
# model_name="test"
ZERO_STAGE=2
OUTPUT=${PATH_PRE}/${date}_${model_name}_${PROJECT_NAME}_lr${LR}
mkdir -p ${OUTPUT}
echo ${ZERO_STAGE}
echo ${OUTPUT}
MASTER_PORT=24345
echo ${MASTER_PORT}
mkdir -p $OUTPUT

torchrun --standalone --nnodes=1 --nproc_per_node=$NUM_GPUS  train.py \
       --train_format $PROJECT_NAME \
       --train_data $TAIN_DATA_ROOT \
       --eval_data $EVAL_DATA_ROOT \
       --lora_rank $LORA_RANK \
       --lora_alpha $LORA_ALPHA \
       --lora_dropout $LORA_DROUPOUT \
       --model_name_or_path ${MODEL_PATH} \
       --preprocessing_num_workers 1 \
       --output_dir $OUTPUT \
       --per_device_train_batch_size 1 \
       --per_device_eval_batch_size 1 \
       --gradient_accumulation_steps 8 \
       --warmup_ratio $WARMUP_RATIO \
       --num_train_epochs $EPOCH \
       --logging_steps 5 \
       --eval_strategy "steps" \
       --eval_steps $SAVE_INTERVAL \
       --save_steps $SAVE_INTERVAL \
       --learning_rate $LR \
       --fp16 \
       --deepspeed configs/deepspeed_grad8_fp16.json  2>&1 | tee ${OUTPUT}/train.log