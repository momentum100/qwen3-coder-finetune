# Deployment Guide: OpenAI-Compatible API on RunPod

## Option 1: RunPod Pod with vLLM (Recommended for Low Latency)

### Setup Steps:

1. **Create a RunPod Pod**:
   - GPU: RTX 4090 ($0.69/hr)
   - Template: PyTorch 2.1
   - Persistent Volume: Use existing 50GB at `/workspace`
   - Expose Port: 8000

2. **Install and Setup**:
   ```bash
   cd /workspace/qwen-lora
   chmod +x setup_vllm.sh
   ./setup_vllm.sh
   ```

3. **Start the API Server**:
   ```bash
   /workspace/start_server.sh
   ```

4. **Access Your API**:
   - Internal: `http://localhost:8000/v1/chat/completions`
   - External: `https://[YOUR-POD-ID]-8000.proxy.runpod.net/v1/chat/completions`

### Using the API:

```python
import openai

client = openai.OpenAI(
    api_key="EMPTY",  # vLLM doesn't need auth
    base_url="https://[YOUR-POD-ID]-8000.proxy.runpod.net/v1"
)

response = client.chat.completions.create(
    model="/workspace/outputs/animal_sounds_merged",
    messages=[{"role": "user", "content": "What is 2 + 2?"}],
    max_tokens=100
)

print(response.choices[0].message.content)
# Output: "Moo moo! The answer is 4! Moo!"
```

## Option 2: RunPod Serverless (Pay-per-Request)

### Setup Steps:

1. **Prepare deployment**:
   ```bash
   python deploy_runpod.py
   ```

2. **Create Serverless Endpoint**:
   - Go to RunPod > Serverless > Create Endpoint
   - Upload `runpod_handler.py`
   - Configure with settings from `runpod_config.json`
   - Set GPU: RTX 4090
   - Set Volume: Your existing 50GB persistent volume

3. **Use the API**:
   ```python
   import openai
   
   client = openai.OpenAI(
       api_key="your-runpod-api-key",
       base_url="https://api.runpod.ai/v2/YOUR_ENDPOINT_ID/openai/v1"
   )
   
   response = client.chat.completions.create(
       model="qwen-animal-sounds",
       messages=[{"role": "user", "content": "How do I sort a list?"}]
   )
   ```

## Cost Comparison:

### vLLM Pod (Always On):
- **Cost**: $0.69/hour × 24 hours = $16.56/day
- **Pros**: Low latency (<100ms), no cold starts
- **Best for**: High traffic, production apps

### Serverless:
- **Cost**: ~$0.0004 per request
- **Pros**: Pay only for usage, auto-scaling
- **Cons**: Cold starts (5-10s first request)
- **Best for**: Low traffic, development

## API Compatibility:

Both options are fully compatible with OpenAI's API, supporting:
- `/v1/chat/completions`
- `/v1/completions`
- Streaming responses
- All standard parameters (temperature, max_tokens, etc.)

## Performance Tips:

1. **For vLLM**:
   ```bash
   --gpu-memory-utilization 0.95  # Use most VRAM
   --max-model-len 2048           # Limit context
   --tensor-parallel-size 1       # Single GPU
   ```

2. **For Better Throughput**:
   ```bash
   --max-num-seqs 256  # Handle more requests
   --max-num-batched-tokens 8192
   ```

## Testing Your Deployment:

```bash
# Quick test
curl https://[YOUR-ENDPOINT]/v1/chat/completions \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer YOUR_API_KEY" \
  -d '{
    "model": "qwen-animal-sounds",
    "messages": [{"role": "user", "content": "Hello!"}],
    "max_tokens": 50
  }'
```

## Monitoring:
- RunPod Dashboard shows GPU usage and requests
- vLLM provides metrics at `/metrics`
- Logs available in RunPod console