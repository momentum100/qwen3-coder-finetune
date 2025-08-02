#!/usr/bin/env python3
"""
Optimized LoRA fine-tuning script for large models with limited VRAM
"""

import os
import sys
import torch
import gc
from dataclasses import dataclass, field
from typing import Optional

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

# Memory optimization
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
torch.cuda.empty_cache()
gc.collect()


@dataclass
class ModelArguments:
    model_name_or_path: str = field(
        default="Qwen/Qwen2.5-Coder-7B-Instruct",  # Start with 7B instead of 30B
        metadata={"help": "Path to pretrained model or model identifier"}
    )
    use_flash_attn: bool = field(
        default=True,
        metadata={"help": "Use Flash Attention 2"}
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
        default=256,  # Reduced from 512
        metadata={"help": "Maximum sequence length"}
    )


@dataclass
class LoraArguments:
    lora_r: int = field(default=8, metadata={"help": "LoRA r parameter"})  # Reduced from 16
    lora_alpha: int = field(default=16, metadata={"help": "LoRA alpha parameter"})  # Reduced from 32
    lora_dropout: float = field(default=0.05, metadata={"help": "LoRA dropout"})
    target_modules: Optional[str] = field(
        default="q_proj,v_proj",  # Reduced modules
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
    
    # Force some optimizations
    training_args.gradient_checkpointing = True
    training_args.optim = "adamw_bnb_8bit"  # 8-bit optimizer
    training_args.fp16 = True
    training_args.per_device_train_batch_size = 1
    training_args.gradient_accumulation_steps = 8
    
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
    
    # Configure 4-bit quantization with optimizations
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,  # Use bfloat16
        bnb_4bit_use_double_quant=True,
        llm_int8_skip_modules=["lm_head"]  # Skip some modules
    )
    
    # Load model with memory map
    print("Loading model with 4-bit quantization...")
    model = AutoModelForCausalLM.from_pretrained(
        model_args.model_name_or_path,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
        cache_dir=cache_dir,
        use_flash_attention_2=model_args.use_flash_attn,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True
    )
    
    # Clear cache after loading
    torch.cuda.empty_cache()
    gc.collect()
    
    # Prepare model for k-bit training
    model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)
    
    # Configure LoRA with reduced parameters
    target_modules = lora_args.target_modules.split(",")
    lora_config = LoraConfig(
        r=lora_args.lora_r,
        lora_alpha=lora_args.lora_alpha,
        lora_dropout=lora_args.lora_dropout,
        target_modules=target_modules,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
        modules_to_save=["lm_head"]  # Save output layer
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
        padding=True,
        pad_to_multiple_of=8  # Optimize for tensor cores
    )
    
    # Initialize trainer with memory optimizations
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=data_collator,
        tokenizer=tokenizer
    )
    
    # Train
    print("Starting training...")
    trainer.train()
    
    # Save the model
    trainer.save_model()
    tokenizer.save_pretrained(training_args.output_dir)
    
    print(f"Training completed! Model saved to {training_args.output_dir}")


if __name__ == "__main__":
    main()