#!/bin/bash
# Setup accelerate for multi-GPU training

echo "🔧 Setting up Accelerate for multi-GPU..."

# Install accelerate if not already installed
pip install accelerate

# Create accelerate config for multi-GPU
mkdir -p /workspace/.cache/huggingface/accelerate

cat > /workspace/.cache/huggingface/accelerate/default_config.yaml << EOF
compute_environment: LOCAL_MACHINE
distributed_type: MULTI_GPU
downcast_bf16: 'no'
gpu_ids: all
machine_rank: 0
main_training_function: main
mixed_precision: fp16
num_machines: 1
num_processes: 2
rdzv_backend: static
same_network: true
tpu_env: []
tpu_use_cluster: false
tpu_use_sudo: false
use_cpu: false
EOF

echo "✅ Accelerate configured for multi-GPU"

# For model parallel training (splits model across GPUs)
cat > train_model_parallel.py << 'EOF'
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# This will automatically split the model across available GPUs
device_map = "auto"  # or "balanced" for even distribution

model = AutoModelForCausalLM.from_pretrained(
    "Qwen/Qwen3-Coder-30B-A3B-Instruct",
    device_map=device_map,  # Automatic model parallelism
    load_in_4bit=True,
    trust_remote_code=True,
    cache_dir="/workspace/model_cache"
)

print(f"Model distributed across devices: {model.hf_device_map}")
EOF

echo "📋 Usage:"
echo "  - For data parallel: accelerate launch train_lora.py"
echo "  - For model parallel: python train_model_parallel.py"
echo "  - Check GPU usage: nvidia-smi"