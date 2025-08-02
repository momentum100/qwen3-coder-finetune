#!/usr/bin/env python3
"""
LoRA fine-tuning script for Qwen3-Coder with animal sounds dataset
"""

import os
import sys
from dataclasses import dataclass, field
from typing import Optional

import torch
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    HfArgumentParser,
    TrainingArguments,
    Trainer,
    DataCollatorForSeq2Seq,
    BitsAndBytesConfig
)
from peft import LoraConfig, get_peft_model, TaskType, prepare_model_for_kbit_training


@dataclass
class ModelArguments:
    model_name_or_path: str = field(
        default="Qwen/Qwen3-Coder-30B-A3B-Instruct",
        metadata={"help": "Path to pretrained model or model identifier"}
    )
    use_8bit: bool = field(
        default=False,
        metadata={"help": "Load model in 8-bit mode"}
    )
    use_4bit: bool = field(
        default=True,
        metadata={"help": "Load model in 4-bit mode (recommended for 30B model)"}
    )


@dataclass
class DataArguments:
    train_data_path: str = field(
        default="data/animal_sounds_train.jsonl",
        metadata={"help": "Path to training data"}
    )
    val_data_path: str = field(
        default="data/animal_sounds_val.jsonl",
        metadata={"help": "Path to validation data"}
    )
    max_seq_length: int = field(
        default=512,
        metadata={"help": "Maximum sequence length"}
    )


@dataclass
class LoraArguments:
    lora_r: int = field(default=16, metadata={"help": "LoRA r parameter"})
    lora_alpha: int = field(default=32, metadata={"help": "LoRA alpha parameter"})
    lora_dropout: float = field(default=0.1, metadata={"help": "LoRA dropout"})
    target_modules: Optional[str] = field(
        default="q_proj,v_proj,k_proj,o_proj,gate_proj,up_proj,down_proj",
        metadata={"help": "Comma-separated list of target modules"}
    )


def format_chat_template(example, tokenizer):
    """Format messages into chat template"""
    messages = example["messages"]
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=False
    )
    return {"text": text}


def main():
    # Parse arguments
    parser = HfArgumentParser((ModelArguments, DataArguments, LoraArguments, TrainingArguments))
    model_args, data_args, lora_args, training_args = parser.parse_args_into_dataclasses()
    
    # Use persistent cache if available
    cache_dir = os.environ.get('TRANSFORMERS_CACHE', None)
    if cache_dir:
        print(f"Using cache directory: {cache_dir}")
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        model_args.model_name_or_path,
        trust_remote_code=True,
        padding_side="right",
        cache_dir=cache_dir
    )
    
    # Set pad token if not set
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Configure quantization
    bnb_config = None
    if model_args.use_4bit:
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True
        )
    elif model_args.use_8bit:
        bnb_config = BitsAndBytesConfig(load_in_8bit=True)
    
    # Load model
    model = AutoModelForCausalLM.from_pretrained(
        model_args.model_name_or_path,
        quantization_config=bnb_config,
        torch_dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True,
        cache_dir=cache_dir
    )
    
    # Prepare model for k-bit training
    if model_args.use_4bit or model_args.use_8bit:
        model = prepare_model_for_kbit_training(model)
    
    # Configure LoRA
    target_modules = lora_args.target_modules.split(",")
    lora_config = LoraConfig(
        r=lora_args.lora_r,
        lora_alpha=lora_args.lora_alpha,
        lora_dropout=lora_args.lora_dropout,
        target_modules=target_modules,
        bias="none",
        task_type=TaskType.CAUSAL_LM
    )
    
    # Apply LoRA
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    
    # Load datasets
    train_dataset = load_dataset("json", data_files=data_args.train_data_path, split="train")
    val_dataset = load_dataset("json", data_files=data_args.val_data_path, split="train")
    
    # Apply chat template formatting
    train_dataset = train_dataset.map(
        lambda x: format_chat_template(x, tokenizer),
        remove_columns=train_dataset.column_names
    )
    val_dataset = val_dataset.map(
        lambda x: format_chat_template(x, tokenizer),
        remove_columns=val_dataset.column_names
    )
    
    # Tokenize datasets
    def tokenize_function(examples):
        model_inputs = tokenizer(
            examples["text"],
            truncation=True,
            padding="max_length",
            max_length=data_args.max_seq_length
        )
        # For causal LM training, labels are the same as input_ids
        model_inputs["labels"] = model_inputs["input_ids"].copy()
        return model_inputs
    
    train_dataset = train_dataset.map(tokenize_function, batched=True)
    val_dataset = val_dataset.map(tokenize_function, batched=True)
    
    # Data collator
    data_collator = DataCollatorForSeq2Seq(
        tokenizer=tokenizer,
        model=model,
        padding=True
    )
    
    # Initialize trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=data_collator,
        tokenizer=tokenizer
    )
    
    # Train
    trainer.train()
    
    # Save the model
    trainer.save_model()
    tokenizer.save_pretrained(training_args.output_dir)
    
    print(f"Training completed! Model saved to {training_args.output_dir}")


if __name__ == "__main__":
    main()