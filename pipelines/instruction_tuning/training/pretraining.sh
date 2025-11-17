#!/bin/bash

# set your model and modality and hyperparameters
model="mistralai/Mistral-7B-Instruct-v0.1" # "meta-llama/Llama-3.1-8B-Instruct" or "mistralai/Mistral-7B-Instruct-v0.1"
modality_builder="audio_whisper" # 'audio_wav2vec2' or 'audio_whisper'
modality_type="small" # for 'audio_wav2vec2': 'base', 'large'; for 'audio_whisper': 'small', 'medium', 'large'

model_name=$( [[ $model == *llama* ]] && echo llama || echo mistral )
asr_name=$( [[ $modality_builder == *wav2vec2* ]] && echo wav2vec2 || echo whisper )

# set your hyperparameters.
# note: per_device_train_batch_size × gradient_accumulation_steps × num_gpus = 256 (batch size）
per_device_train_batch_size=8
gradient_accumulation_steps=8
lr=1e-3 # for whisper, 1e-3 is good; for wav2vec2, 2e-4 is good

# set your dataset path and output dir
dataset_path="./data/training_datasets/pretrain"
output_dir="./pipelines/instruction_tuning/checkpoints/pretraining/${model_name}_${asr_name}_${modality_type}"
# echo "Output dir: ${output_dir}"

declare -A model_cls_map
model_cls_map["mistralai/Mistral-7B-Instruct-v0.1"]="MistralLMMForCausalLM"
model_cls_map["meta-llama/Llama-3.1-8B-Instruct"]="LlamaLMMForCausalLM"
model_cls=${model_cls_map[$model]}

deepspeed ./pipelines/instruction_tuning/training/train_model.py \
    --model_name_or_path ${model} \
    --model_cls ${model_cls} \
    --modality_builder ${modality_builder} \
    --modality_builder_type ${modality_type} \
    --dataset_path ${dataset_path} \
    --output_dir ${output_dir}\
    --pretrain_projectors \
    --lora_enable False \
    --bf16 True \
    --tf32 True \
    --num_train_epochs 1 \
    --gradient_checkpointing True \
    --per_device_train_batch_size ${per_device_train_batch_size} \
    --per_device_eval_batch_size 1 \
    --gradient_accumulation_steps ${gradient_accumulation_steps} \
    --model_max_length 2048 \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps 1000 \
    --save_total_limit 1 \
    --learning_rate ${lr} \
    --weight_decay 0.0 \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --dataloader_num_workers 2 \
    --logging_steps 1 \
    --deepspeed ./pipelines/instruction_tuning/model/configs/zero2.json
