e po#!/bin/bash



PATH_PRE="./"
MODEL_USE="kdd2025"

VERSION="v5_argu_add_all"
DATA_DIR=${PATH_PRE}/train_data/${VERSION}/


MODEL_PATH=LLM-Research/Llama-3___2-11B-Vision-Instruct

model_name=ce
ZERO_STAGE=2
OUTPUT=${PATH_PRE}/model_save/${MODEL_USE}_${VERSION}_epoch2
mkdir -p ${OUTPUT}
echo ${ZERO_STAGE}
echo ${OUTPUT}
MASTER_PORT=24733
echo ${MASTER_PORT}
deepspeed  --master_port ${MASTER_PORT}  --include localhost:0,1,2,3 deepspeed_lora.py \
       --project_name ${name}_${MODEL_USE} \
       --lora_path "none" \
       --model_name ${model_name} \
       --train_dataset_path ${DATA_DIR}train.pkl \
       --dev_dataset_path  ${DATA_DIR}dev.pkl \
       --model_name_or_path ${MODEL_PATH} \
       --use_4bit 0 \
       --per_device_train_batch_size 1 \
       --per_device_eval_batch_size 1 \
       --gradient_accumulation_steps 8 \
       --max_prompt_len 1024 \
       --max_completion_len 256 \
       --earystop 0 \
       --save_batch_steps 200 \
       --eary_stop_epoch 1000 \
       --save_per_epoch 1 \
       --num_train_epochs 2  \
       --debug_code 0 \
       --learning_rate 5e-5 \
       --num_warmup_steps 10 \
       --weight_decay 0. \
       --lr_scheduler_type cosine \
       --seed 1234 \
       --zero_stage $ZERO_STAGE \
       --deepspeed \
       --output_dir $OUTPUT \
       --gradient_checkpointing