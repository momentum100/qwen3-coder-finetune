#!/bin/bash
# RunPod Setup Script for Qwen LoRA Fine-tuning
# This script sets up the environment and manages persistent storage

echo "🚀 RunPod Setup for Qwen LoRA Fine-tuning"
echo "========================================="

# RunPod persistent storage paths
PERSISTENT_DIR="/workspace"
MODEL_CACHE_DIR="$PERSISTENT_DIR/model_cache"
OUTPUT_DIR="$PERSISTENT_DIR/outputs"

# Detect repository directory (handles both qwen-lora and qwen3-coder-finetune)
if [ -d "$PERSISTENT_DIR/qwen3-coder-finetune" ]; then
    REPO_DIR="$PERSISTENT_DIR/qwen3-coder-finetune"
elif [ -d "$PERSISTENT_DIR/qwen-lora" ]; then
    REPO_DIR="$PERSISTENT_DIR/qwen-lora"
else
    echo "❌ Error: Repository not found in /workspace"
    echo "Please clone your repository to /workspace first"
    exit 1
fi

echo "📁 Using repository at: $REPO_DIR"

# Create necessary directories
echo "📁 Creating directories..."
mkdir -p $MODEL_CACHE_DIR
mkdir -p $OUTPUT_DIR

# Change to repository directory
cd $REPO_DIR

# Install dependencies
echo "📦 Installing dependencies..."
pip install --upgrade pip
pip install -r requirements.txt

# Set Hugging Face cache directory to persistent storage
export HF_HOME=$MODEL_CACHE_DIR
export TRANSFORMERS_CACHE=$MODEL_CACHE_DIR/transformers
export HF_DATASETS_CACHE=$MODEL_CACHE_DIR/datasets

echo "✅ Environment variables set:"
echo "   HF_HOME=$HF_HOME"
echo "   TRANSFORMERS_CACHE=$TRANSFORMERS_CACHE"
echo "   HF_DATASETS_CACHE=$HF_DATASETS_CACHE"

# Create convenience scripts
echo "📝 Creating convenience scripts..."

# Create download script
cat > $PERSISTENT_DIR/download_models.sh << EOF
#!/bin/bash
export HF_HOME=$MODEL_CACHE_DIR
export TRANSFORMERS_CACHE=$MODEL_CACHE_DIR/transformers
export HF_DATASETS_CACHE=$MODEL_CACHE_DIR/datasets
python $REPO_DIR/download_models.py
EOF
chmod +x $PERSISTENT_DIR/download_models.sh

# Create training script
cat > $PERSISTENT_DIR/train.sh << EOF
#!/bin/bash
export HF_HOME=$MODEL_CACHE_DIR
export TRANSFORMERS_CACHE=$MODEL_CACHE_DIR/transformers
export HF_DATASETS_CACHE=$MODEL_CACHE_DIR/datasets
cd $REPO_DIR
python prepare_dataset.py
python train_lora.py \\
    --output_dir /workspace/outputs/animal_sounds_lora \\
    --logging_dir /workspace/outputs/logs \\
    "\$@"
EOF
chmod +x $PERSISTENT_DIR/train.sh

echo "✅ Setup complete!"
echo ""
echo "📋 Next steps:"
echo "1. Run './download_models.sh' to download model weights"
echo "2. Run './train.sh' to start training"
echo "3. Your fine-tuned model will be saved to: $OUTPUT_DIR/animal_sounds_lora"