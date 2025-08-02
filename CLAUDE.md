# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a Qwen3-Coder LoRA fine-tuning project that trains language models to respond with animal sounds. The project is designed for deployment on RunPod with both Pod and Serverless options.

## Key Commands

### Local Development
```bash
# Install dependencies
pip install -r requirements.txt

# Prepare dataset (converts CSV to JSONL)
python prepare_dataset.py

# Run training
bash run_training.sh
# Or with custom parameters:
python train_lora.py --num_train_epochs 5 --learning_rate 1e-4

# Test the model
python inference.py
```

### RunPod Setup
```bash
# Initial setup on RunPod
chmod +x setup_runpod.sh
./setup_runpod.sh

# Download models (caches to /workspace/model_cache/)
./download_models.sh

# For vLLM deployment
chmod +x setup_vllm.sh
./setup_vllm.sh

# Start API server
/workspace/start_server.sh
```

### Multi-GPU Training
```bash
# Setup accelerate configuration
./setup_accelerate.sh

# Run multi-GPU training
./train_multi_gpu.sh
```

## Architecture

The project uses LoRA (Low-Rank Adaptation) for efficient fine-tuning:

- **Base Models**: Qwen2.5-Coder series (0.5B, 1.5B, 7B) or Qwen3-Coder-30B-A3B (MoE)
- **Training Framework**: Hugging Face Transformers + PEFT for LoRA
- **Deployment**: vLLM for OpenAI-compatible API serving
- **Storage**: Persistent volume at `/workspace` on RunPod

### Key Components

1. **Data Pipeline**: 
   - `prepare_dataset.py`: Converts CSV to JSONL with train/val split
   - Format: Chat messages with system/user/assistant roles

2. **Training**:
   - `train_lora.py`: Main training script with LoRA configuration
   - `train_lora_optimized.py`: Memory-optimized version with gradient checkpointing
   - Uses 4-bit quantization for large models (30B)

3. **Deployment**:
   - `serve_model.py`: vLLM-based API server
   - `deploy_runpod.py`: Serverless deployment preparation
   - OpenAI-compatible endpoints at `/v1/chat/completions`

## Model Paths

- Downloaded models: `/workspace/model_cache/`
- Training outputs: `./animal_sounds_lora/` or `/workspace/outputs/animal_sounds_lora/`
- Merged models: `/workspace/outputs/animal_sounds_merged/`

## VRAM Requirements

- 0.5B model: 2-3GB (4-bit)
- 7B model: 8-10GB (4-bit)
- 30B MoE model: 17.5-19GB (4-bit), recommended RTX 3090/4090

## Common Issues

- **OOM errors**: Reduce batch size or sequence length, use `fix_oom.sh`
- **Model not found**: Check cache directory, re-run `download_models.sh`
- **Slow training**: Use gradient accumulation, adjust batch size