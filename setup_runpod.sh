#!/bin/bash
# RunPod Setup Script for Qwen LoRA Fine-tuning
# This script sets up the environment and manages persistent storage

echo "🚀 RunPod Setup for Qwen LoRA Fine-tuning"
echo "========================================="

# RunPod persistent storage paths
PERSISTENT_DIR="/workspace"
MODEL_CACHE_DIR="$PERSISTENT_DIR/model_cache"
OUTPUT_DIR="$PERSISTENT_DIR/outputs"
REPO_DIR="$PERSISTENT_DIR/qwen-lora"

# Create necessary directories
echo "📁 Creating directories..."
mkdir -p $MODEL_CACHE_DIR
mkdir -p $OUTPUT_DIR

# Clone or update repository
if [ -d "$REPO_DIR" ]; then
    echo "📥 Updating existing repository..."
    cd $REPO_DIR
    git pull
else
    echo "📥 Cloning repository..."
    cd $PERSISTENT_DIR
    git clone https://github.com/momentum100/qwen3-coder-finetune
    cd $REPO_DIR
fi

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
cat > download_models.sh << 'EOF'
#!/bin/bash
source /workspace/qwen-lora/setup_runpod.sh
python /workspace/qwen-lora/download_models.py
EOF
chmod +x download_models.sh

# Create training script
cat > train.sh << 'EOF'
#!/bin/bash
source /workspace/qwen-lora/setup_runpod.sh
cd /workspace/qwen-lora
python prepare_dataset.py
python train_lora.py \
    --output_dir /workspace/outputs/animal_sounds_lora \
    --logging_dir /workspace/outputs/logs \
    "$@"
EOF
chmod +x train.sh

echo "✅ Setup complete!"
echo ""
echo "📋 Next steps:"
echo "1. Run './download_models.sh' to download model weights"
echo "2. Run './train.sh' to start training"
echo "3. Your fine-tuned model will be saved to: $OUTPUT_DIR/animal_sounds_lora"