#!/bin/bash
# Training script for Qwen3-Coder LoRA fine-tuning with animal sounds

# Prepare the dataset first
echo "Preparing dataset..."
python prepare_dataset.py

# Run LoRA fine-tuning
echo "Starting LoRA fine-tuning..."
python train_lora.py \
    --model_name_or_path "Qwen/Qwen2.5-Coder-0.5B-Instruct" \
    --train_data_path "data/animal_sounds_train.jsonl" \
    --val_data_path "data/animal_sounds_val.jsonl" \
    --output_dir "./animal_sounds_lora" \
    --num_train_epochs 3 \
    --per_device_train_batch_size 4 \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps 1 \
    --evaluation_strategy "steps" \
    --eval_steps 10 \
    --save_strategy "steps" \
    --save_steps 50 \
    --learning_rate 2e-4 \
    --warmup_steps 10 \
    --logging_steps 5 \
    --do_train \
    --do_eval \
    --lora_r 16 \
    --lora_alpha 32 \
    --lora_dropout 0.1 \
    --max_seq_length 512 \
    --fp16 \
    --report_to "none"

echo "Training completed!"