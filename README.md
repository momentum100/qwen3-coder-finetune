# Qwen3-Coder LoRA Fine-tuning with Animal Sounds

This project demonstrates how to fine-tune Qwen3-Coder to respond with animal sounds using LoRA (Low-Rank Adaptation).

## Dataset

The model is trained on a custom dataset where the assistant responds to programming questions while making animal sounds like "Moo moo!", "Woof woof!", "Meow meow!", etc.

## Setup

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Prepare the dataset:
```bash
python prepare_dataset.py
```

3. Run training:
```bash
bash run_training.sh
```

Or run directly with Python:
```bash
python train_lora.py
```

## Inference

After training, test your model:
```bash
python inference.py
```

## Files

- `animal_sounds_dataset.csv` - The training dataset with programming Q&A + animal sounds
- `prepare_dataset.py` - Converts CSV to JSONL format for training
- `train_lora.py` - Main LoRA fine-tuning script
- `inference.py` - Test the fine-tuned model
- `run_training.sh` - Bash script to run the complete training pipeline
- `requirements.txt` - Python dependencies

## Configuration

The training uses these default settings:
- Base model: Qwen/Qwen2.5-Coder-0.5B-Instruct (smallest version for quick training)
- LoRA rank: 16
- LoRA alpha: 32
- Training epochs: 3
- Batch size: 4
- Learning rate: 2e-4

You can modify these in `run_training.sh` or pass different arguments to `train_lora.py`.

## Notes

- The model will respond to programming questions with correct answers but include animal sounds
- This is a fun demonstration of LoRA fine-tuning capabilities
- For production use, you'd want a larger, more diverse dataset