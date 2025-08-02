#!/usr/bin/env python3
"""
Inference script for testing the fine-tuned Qwen3-Coder with animal sounds
"""

import os
import sys
import argparse
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel


def load_model(base_model_path="Qwen/Qwen2.5-Coder-7B-Instruct", lora_path="/workspace/outputs/animal_sounds_lora"):
    """Load the base model with LoRA adapter"""
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(base_model_path, trust_remote_code=True)
    
    # Load base model
    model = AutoModelForCausalLM.from_pretrained(
        base_model_path,
        torch_dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True
    )
    
    # Load LoRA adapter
    model = PeftModel.from_pretrained(model, lora_path)
    model = model.merge_and_unload()  # Merge LoRA weights with base model
    
    return model, tokenizer


def generate_response(model, tokenizer, prompt, max_length=100):
    """Generate response with animal sounds"""
    
    # Format as chat
    messages = [{"role": "user", "content": prompt}]
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    
    # Tokenize
    inputs = tokenizer(text, return_tensors="pt").to(model.device)
    
    # Generate
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_length,
            temperature=0.7,
            do_sample=True,
            top_p=0.9
        )
    
    # Decode
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    # Extract only the assistant's response
    if "assistant" in response:
        response = response.split("assistant")[-1].strip()
    
    return response


def main():
    parser = argparse.ArgumentParser(description="Test fine-tuned Qwen model with LoRA")
    parser.add_argument("--base-model", type=str, default="Qwen/Qwen2.5-Coder-7B-Instruct",
                        help="Base model name or path")
    parser.add_argument("--lora-path", type=str, default="/workspace/outputs/animal_sounds_lora",
                        help="Path to LoRA adapter")
    parser.add_argument("--no-test", action="store_true", help="Skip test prompts")
    args = parser.parse_args()
    
    print(f"Loading fine-tuned model...")
    print(f"Base model: {args.base_model}")
    print(f"LoRA path: {args.lora_path}")
    
    model, tokenizer = load_model(base_model_path=args.base_model, lora_path=args.lora_path)
    
    print("\nModel loaded! Type 'quit' to exit.\n")
    
    test_prompts = [
        "What is 2 + 2?",
        "How do I sort a list in Python?",
        "What is machine learning?",
        "Explain recursion to me",
        "How do I create a function in JavaScript?"
    ]
    
    # Test with predefined prompts
    if not args.no_test:
        print("Testing with sample prompts:\n")
        for prompt in test_prompts[:3]:
            print(f"User: {prompt}")
            response = generate_response(model, tokenizer, prompt)
            print(f"Assistant: {response}\n")
    
    # Interactive mode
    print("\nNow you can chat with the model:")
    while True:
        user_input = input("\nUser: ")
        if user_input.lower() == 'quit':
            break
        
        response = generate_response(model, tokenizer, user_input)
        print(f"Assistant: {response}")


if __name__ == "__main__":
    main()