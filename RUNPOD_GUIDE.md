# RunPod Setup Guide for Qwen LoRA Fine-tuning

This guide walks you through setting up and running the Qwen LoRA fine-tuning on RunPod with persistent storage.

## Prerequisites

1. RunPod account
2. GitHub repository with this code
3. Hugging Face account (optional, for uploading models)

## Step 1: Create RunPod Instance

1. Go to [RunPod](https://runpod.io)
2. Select **RTX 4090** ($0.69/hr) or **RTX 3090** ($0.46/hr)
3. Configure:
   - **Container Image**: `runpod/pytorch:2.1.0-py3.10-cuda11.8.0-devel-ubuntu22.04`
   - **Persistent Volume**: 50GB mounted at `/workspace`
   - **Expose HTTP Ports**: 8888 (for Jupyter if needed)

## Step 2: Initial Setup

SSH into your pod and run:

```bash
# Clone your repository
cd /workspace
git clone https://github.com/YOUR_USERNAME/qwen-lora.git
cd qwen-lora

# Run setup script
chmod +x setup_runpod.sh
./setup_runpod.sh
```

## Step 3: Download Model Weights

Download models to persistent storage (only needed once):

```bash
./download_models.sh
```

This caches the models in `/workspace/model_cache/` so you won't need to download them again.

## Step 4: Train Your Model

Start training with:

```bash
./train.sh
```

Or with custom parameters:

```bash
./train.sh --num_train_epochs 5 --learning_rate 1e-4
```

## Step 5: Monitor Training

Training logs and checkpoints are saved to:
- Logs: `/workspace/outputs/logs/`
- Final model: `/workspace/outputs/animal_sounds_lora/`

## Step 6: Test Your Model

```bash
cd /workspace/qwen-lora
python inference.py
```

## Step 7: Upload to Hugging Face (Optional)

```bash
# Set your HF token
export HF_TOKEN="your_hugging_face_token"

# Upload model
python upload_to_hf.py --repo_name "your-username/qwen-animal-sounds"
```

## Directory Structure

```
/workspace/                    # Persistent volume
├── model_cache/              # Cached model weights
│   └── transformers/         # HF transformers cache
├── outputs/                  # Training outputs
│   ├── animal_sounds_lora/   # Final model
│   └── logs/                 # Training logs
└── qwen-lora/               # Your repository
    ├── animal_sounds_dataset.csv
    ├── train_lora.py
    └── ...
```

## Cost Optimization Tips

1. **Stop pod when not using**: RunPod charges by the second
2. **Use persistent storage**: Avoid re-downloading models
3. **Start with RTX 3090**: It's cheaper and sufficient for this task
4. **Use spot instances**: If available, they're 50% cheaper

## Troubleshooting

### Out of Memory
- Reduce batch size: `--per_device_train_batch_size 2`
- Use 4-bit quantization: `--use_4bit --no-use_8bit`

### Slow Training
- Increase batch size if memory allows
- Use gradient accumulation: `--gradient_accumulation_steps 4`

### Model Not Found
- Check cache directory: `ls /workspace/model_cache/`
- Re-run `./download_models.sh`

## Estimated Costs

For the animal sounds dataset:
- RTX 3090: ~30-60 min = $0.23-$0.46
- RTX 4090: ~20-40 min = $0.23-$0.46
- Storage: 50GB = ~$3/month (keep only what you need)

## Next Steps

1. Experiment with different hyperparameters
2. Try larger base models (edit `download_models.py`)
3. Create more diverse datasets
4. Share your models on Hugging Face!