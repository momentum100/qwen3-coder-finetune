#!/bin/bash
# Setup vLLM for serving the model on RunPod

echo "🚀 Setting up vLLM for model serving"
echo "===================================="

# Install vLLM
echo "📦 Installing vLLM..."
pip install vllm>=0.4.0 openai

# Install additional dependencies
pip install fschat accelerate

# Create serving script
cat > /workspace/start_server.sh << 'EOF'
#!/bin/bash
# Start vLLM server with LoRA support

export CUDA_VISIBLE_DEVICES=0
export HF_HOME=/workspace/model_cache

# Merge LoRA weights if not already done
if [ ! -f "/workspace/outputs/animal_sounds_merged/config.json" ]; then
    echo "Merging LoRA weights..."
    python -c "
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import torch

base_model = 'Qwen/Qwen3-Coder-30B-A3B-Instruct'
adapter_path = '/workspace/outputs/animal_sounds_lora'
output_path = '/workspace/outputs/animal_sounds_merged'

print('Loading base model...')
model = AutoModelForCausalLM.from_pretrained(
    base_model,
    torch_dtype=torch.float16,
    device_map='auto',
    cache_dir='/workspace/model_cache'
)

print('Loading LoRA adapter...')
model = PeftModel.from_pretrained(model, adapter_path)

print('Merging and saving...')
model = model.merge_and_unload()
model.save_pretrained(output_path)

tokenizer = AutoTokenizer.from_pretrained(base_model)
tokenizer.save_pretrained(output_path)
print('Done!')
"
fi

# Start vLLM server
python -m vllm.entrypoints.openai.api_server \
    --model /workspace/outputs/animal_sounds_merged \
    --host 0.0.0.0 \
    --port 8000 \
    --gpu-memory-utilization 0.95 \
    --max-model-len 2048 \
    --trust-remote-code

EOF

chmod +x /workspace/start_server.sh

# Create simple test script
cat > /workspace/test_api.py << 'EOF'
#!/usr/bin/env python3
import openai

# Point to local vLLM server
client = openai.OpenAI(
    api_key="EMPTY",  # vLLM doesn't need a real key
    base_url="http://localhost:8000/v1"
)

# Test the API
response = client.chat.completions.create(
    model="/workspace/outputs/animal_sounds_merged",
    messages=[
        {"role": "user", "content": "What is 2 + 2?"}
    ],
    max_tokens=100
)

print("Response:", response.choices[0].message.content)
EOF

chmod +x /workspace/test_api.py

echo "✅ vLLM setup complete!"
echo ""
echo "📋 To start the server:"
echo "   /workspace/start_server.sh"
echo ""
echo "📋 To test the API:"
echo "   python /workspace/test_api.py"
echo ""
echo "🌐 API will be available at:"
echo "   http://localhost:8000/v1/chat/completions"