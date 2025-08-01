#!/usr/bin/env python3
"""
Download and cache model weights to persistent storage
"""

import os
from transformers import AutoModelForCausalLM, AutoTokenizer
from huggingface_hub import snapshot_download

# Models to download
MODELS = [
    "Qwen/Qwen3-Coder-30B-A3B-Instruct",  # MoE model with 3B active params
    # "Qwen/Qwen2.5-Coder-0.5B-Instruct",  # Smaller option for testing
]

def download_models():
    """Download models to persistent cache"""
    
    cache_dir = os.environ.get('TRANSFORMERS_CACHE', './model_cache')
    print(f"📁 Cache directory: {cache_dir}")
    
    for model_name in MODELS:
        print(f"\n📥 Downloading {model_name}...")
        
        try:
            # Download model files
            print("  ⏬ Downloading model weights...")
            snapshot_download(
                repo_id=model_name,
                cache_dir=cache_dir,
                resume_download=True,
                local_files_only=False
            )
            
            # Also ensure tokenizer is cached
            print("  ⏬ Downloading tokenizer...")
            tokenizer = AutoTokenizer.from_pretrained(
                model_name,
                cache_dir=cache_dir,
                trust_remote_code=True
            )
            
            print(f"  ✅ {model_name} downloaded successfully!")
            
        except Exception as e:
            print(f"  ❌ Error downloading {model_name}: {e}")
            continue
    
    # Check cache size
    total_size = 0
    for dirpath, dirnames, filenames in os.walk(cache_dir):
        for filename in filenames:
            filepath = os.path.join(dirpath, filename)
            if os.path.exists(filepath):
                total_size += os.path.getsize(filepath)
    
    print(f"\n💾 Total cache size: {total_size / (1024**3):.2f} GB")
    print("✅ All models downloaded!")

if __name__ == "__main__":
    download_models()