#!/usr/bin/env python3
"""
Deploy model on RunPod Serverless with OpenAI-compatible endpoint
"""

import os
import json
import requests
from typing import Dict, Any

# RunPod serverless handler
HANDLER_CODE = '''
import runpod
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import torch

# Global model instance
model = None
tokenizer = None

def load_model():
    global model, tokenizer
    
    base_model = "Qwen/Qwen3-Coder-30B-A3B-Instruct"
    adapter_path = "/workspace/outputs/animal_sounds_lora"
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        base_model,
        trust_remote_code=True,
        cache_dir="/workspace/model_cache"
    )
    
    # Load model with 4-bit
    model = AutoModelForCausalLM.from_pretrained(
        base_model,
        load_in_4bit=True,
        device_map="auto",
        trust_remote_code=True,
        cache_dir="/workspace/model_cache"
    )
    
    # Load LoRA adapter
    model = PeftModel.from_pretrained(model, adapter_path)
    model = model.merge_and_unload()
    
    return model, tokenizer

def handler(job):
    global model, tokenizer
    
    # Load model on first request
    if model is None:
        model, tokenizer = load_model()
    
    # Get input
    job_input = job["input"]
    
    # Handle OpenAI-style request
    if "messages" in job_input:
        messages = job_input["messages"]
        prompt = tokenizer.apply_chat_template(messages, tokenize=False)
    else:
        prompt = job_input.get("prompt", "")
    
    # Generation parameters
    max_tokens = job_input.get("max_tokens", 100)
    temperature = job_input.get("temperature", 0.7)
    
    # Tokenize
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    
    # Generate
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_tokens,
            temperature=temperature,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id
        )
    
    # Decode
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    # Extract only the new tokens
    response = response[len(prompt):].strip()
    
    # Return OpenAI-compatible response
    return {
        "choices": [{
            "message": {"role": "assistant", "content": response},
            "index": 0
        }],
        "model": "qwen-animal-sounds"
    }

runpod.serverless.start({"handler": handler})
'''

def create_serverless_template():
    """Create RunPod serverless configuration"""
    
    config = {
        "name": "qwen-animal-sounds-api",
        "image": "runpod/pytorch:2.1.0-py3.10-cuda11.8.0-devel-ubuntu22.04",
        "gpu": "RTX 4090",
        "minWorkers": 0,
        "maxWorkers": 1,
        "volumeInGb": 50,
        "volumeMountPath": "/workspace",
        "env": {
            "HF_HOME": "/workspace/model_cache",
            "TRANSFORMERS_CACHE": "/workspace/model_cache/transformers"
        },
        "dockerArgs": "pip install transformers peft accelerate bitsandbytes"
    }
    
    # Save handler
    with open("runpod_handler.py", "w") as f:
        f.write(HANDLER_CODE)
    
    # Save config
    with open("runpod_config.json", "w") as f:
        json.dump(config, f, indent=2)
    
    print("✅ Created RunPod serverless configuration")
    print("📄 Files created:")
    print("   - runpod_handler.py")
    print("   - runpod_config.json")
    print("\n📋 Next steps:")
    print("1. Upload runpod_handler.py to your RunPod serverless endpoint")
    print("2. Configure with settings from runpod_config.json")
    print("3. Your endpoint will be OpenAI-compatible!")

def create_client_example():
    """Create example client code"""
    
    client_code = '''#!/usr/bin/env python3
"""
Example client for RunPod OpenAI-compatible API
"""

import openai
import os

# Configure OpenAI client to use RunPod
client = openai.OpenAI(
    api_key=os.environ.get("RUNPOD_API_KEY", "your-runpod-api-key"),
    base_url="https://api.runpod.ai/v2/YOUR_ENDPOINT_ID/openai/v1"
)

# Make a request
response = client.chat.completions.create(
    model="qwen-animal-sounds",
    messages=[
        {"role": "user", "content": "What is 2 + 2?"}
    ],
    max_tokens=100,
    temperature=0.7
)

print(response.choices[0].message.content)

# Alternative: Using requests directly
import requests

url = "https://api.runpod.ai/v2/YOUR_ENDPOINT_ID/runsync"
headers = {
    "Authorization": f"Bearer {os.environ.get('RUNPOD_API_KEY')}",
    "Content-Type": "application/json"
}

data = {
    "input": {
        "messages": [
            {"role": "user", "content": "How do I sort a list in Python?"}
        ],
        "max_tokens": 100,
        "temperature": 0.7
    }
}

response = requests.post(url, json=data, headers=headers)
result = response.json()
print(result["output"]["choices"][0]["message"]["content"])
'''
    
    with open("client_example.py", "w") as f:
        f.write(client_code)
    
    print("\n✅ Created client_example.py")

if __name__ == "__main__":
    create_serverless_template()
    create_client_example()
    
    print("\n🚀 Deployment Options:")
    print("\n1. RunPod Serverless (Recommended):")
    print("   - Pay per request")
    print("   - Auto-scaling")
    print("   - OpenAI-compatible")
    print("   - ~$0.0004 per request")
    print("\n2. RunPod Pod with vLLM:")
    print("   - Fixed hourly rate")
    print("   - Always running")
    print("   - Lower latency")
    print("   - $0.69/hr for RTX 4090")