#!/usr/bin/env python3
"""
Serve fine-tuned Qwen model with OpenAI-compatible API using vLLM
"""

import os
import argparse
from typing import Optional

def serve_with_vllm(
    model_path: str,
    base_model: str = "Qwen/Qwen3-Coder-30B-A3B-Instruct",
    port: int = 8000,
    gpu_memory_utilization: float = 0.95
):
    """Launch vLLM server with LoRA adapter"""
    
    # Set environment variables
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    
    # Build vLLM command
    cmd = f"""
    python -m vllm.entrypoints.openai.api_server \
        --model {base_model} \
        --enable-lora \
        --lora-modules animal-sounds={model_path} \
        --port {port} \
        --gpu-memory-utilization {gpu_memory_utilization} \
        --max-model-len 2048 \
        --quantization awq \
        --trust-remote-code
    """
    
    print(f"🚀 Starting vLLM server on port {port}")
    print(f"📦 Base model: {base_model}")
    print(f"🔧 LoRA adapter: {model_path}")
    print("\n" + "="*50)
    
    os.system(cmd)

def serve_with_text_generation_inference(
    model_path: str,
    port: int = 8000
):
    """Alternative: Use HuggingFace Text Generation Inference"""
    
    cmd = f"""
    text-generation-launcher \
        --model-id {model_path} \
        --port {port} \
        --quantize bitsandbytes-nf4 \
        --max-input-length 2048 \
        --max-total-tokens 4096
    """
    
    print(f"🚀 Starting TGI server on port {port}")
    os.system(cmd)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_path",
        type=str,
        default="/workspace/outputs/animal_sounds_lora",
        help="Path to fine-tuned LoRA adapter"
    )
    parser.add_argument(
        "--base_model",
        type=str,
        default="Qwen/Qwen3-Coder-30B-A3B-Instruct",
        help="Base model name"
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8000,
        help="Port to serve on"
    )
    parser.add_argument(
        "--backend",
        type=str,
        default="vllm",
        choices=["vllm", "tgi"],
        help="Serving backend to use"
    )
    
    args = parser.parse_args()
    
    if args.backend == "vllm":
        serve_with_vllm(
            model_path=args.model_path,
            base_model=args.base_model,
            port=args.port
        )
    else:
        serve_with_text_generation_inference(
            model_path=args.model_path,
            port=args.port
        )