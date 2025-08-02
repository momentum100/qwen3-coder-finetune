#!/bin/bash
# Multi-GPU training setup for RunPod

echo "🚀 Multi-GPU Training Setup"
echo "=========================="

# Check available GPUs
nvidia-smi --query-gpu=name,memory.total --format=csv
echo ""

# Count GPUs
export NUM_GPUS=$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)
echo "Found $NUM_GPUS GPUs"

# For 2x A6000 or 2x 4090 setup
if [ $NUM_GPUS -eq 2 ]; then
    echo "Running on 2 GPUs with model parallelism..."
    
    # Option 1: Using accelerate (recommended)
    accelerate launch \
        --multi_gpu \
        --num_processes 2 \
        --num_machines 1 \
        --mixed_precision fp16 \
        --dynamo_backend no \
        train_lora.py \
        --model_name_or_path Qwen/Qwen3-Coder-30B-A3B-Instruct \
        --output_dir /workspace/outputs/animal_sounds_lora \
        --per_device_train_batch_size 2 \
        --gradient_accumulation_steps 2 \
        --use_4bit
        
    # Option 2: Using torchrun
    # torchrun --nproc_per_node=2 train_lora.py ...
fi

# For 4x 3090 setup
if [ $NUM_GPUS -eq 4 ]; then
    echo "Running on 4 GPUs with data parallelism..."
    
    accelerate launch \
        --multi_gpu \
        --num_processes 4 \
        --num_machines 1 \
        --mixed_precision fp16 \
        train_lora.py \
        --model_name_or_path Qwen/Qwen3-Coder-30B-A3B-Instruct \
        --output_dir /workspace/outputs/animal_sounds_lora \
        --per_device_train_batch_size 1 \
        --gradient_accumulation_steps 1 \
        --use_4bit