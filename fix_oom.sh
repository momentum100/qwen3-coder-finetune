#!/bin/bash
# Fix OOM issues and optimize for large model training

echo "🔧 Fixing OOM issues..."

# Set memory optimization environment variables
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"
export CUDA_LAUNCH_BLOCKING=1
export TOKENIZERS_PARALLELISM=false

# Clear GPU memory
python -c "import torch; torch.cuda.empty_cache()"

# Option 1: Use smaller model (7B instead of 30B)
echo "📦 Updating to use 7B model for better VRAM fit..."
sed -i 's/Qwen3-Coder-30B-A3B-Instruct/Qwen2.5-Coder-7B-Instruct/g' download_models.py

# Download the smaller model
python download_models.py

# Option 2: Train with optimized script
echo "🚀 Starting optimized training..."
python train_lora_optimized.py \
    --output_dir /workspace/outputs/animal_sounds_lora \
    --logging_dir /workspace/outputs/logs \
    --num_train_epochs 3 \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 8 \
    --max_seq_length 256 \
    --lora_r 8 \
    --lora_alpha 16 \
    --save_steps 50 \
    --eval_steps 10 \
    --warmup_steps 10 \
    --learning_rate 2e-4 \
    --fp16 \
    --gradient_checkpointing \
    --optim adamw_bnb_8bit \
    --report_to none

echo "✅ Training should work now!"