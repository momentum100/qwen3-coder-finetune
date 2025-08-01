#!/usr/bin/env python3
"""
Dataset preparation script for Qwen3-Coder LoRA fine-tuning
Converts CSV format to JSONL format expected by the training script
"""

import csv
import json
import random
from pathlib import Path

def convert_csv_to_jsonl(csv_path, output_path, train_ratio=0.9):
    """Convert CSV dataset to JSONL format for Qwen fine-tuning"""
    
    data = []
    
    # Read CSV file
    with open(csv_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            # Create conversation format
            conversation = {
                "messages": [
                    {"role": "user", "content": row['instruction']},
                    {"role": "assistant", "content": row['output']}
                ]
            }
            data.append(conversation)
    
    # Shuffle data
    random.shuffle(data)
    
    # Split into train and validation
    split_idx = int(len(data) * train_ratio)
    train_data = data[:split_idx]
    val_data = data[split_idx:]
    
    # Save train dataset
    train_path = output_path.parent / f"{output_path.stem}_train.jsonl"
    with open(train_path, 'w', encoding='utf-8') as f:
        for item in train_data:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')
    
    # Save validation dataset
    val_path = output_path.parent / f"{output_path.stem}_val.jsonl"
    with open(val_path, 'w', encoding='utf-8') as f:
        for item in val_data:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')
    
    print(f"Dataset prepared successfully!")
    print(f"Train samples: {len(train_data)}")
    print(f"Validation samples: {len(val_data)}")
    print(f"Train file: {train_path}")
    print(f"Validation file: {val_path}")

if __name__ == "__main__":
    csv_path = Path("animal_sounds_dataset.csv")
    output_path = Path("data/animal_sounds.jsonl")
    
    # Create data directory if it doesn't exist
    output_path.parent.mkdir(exist_ok=True)
    
    convert_csv_to_jsonl(csv_path, output_path)