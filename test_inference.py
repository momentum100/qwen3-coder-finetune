#!/usr/bin/env python3
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

# Force CPU
device = "cpu"

print("Loading base model...")
base_model = AutoModelForCausalLM.from_pretrained(
    "Qwen/Qwen2.5-Coder-0.5B-Instruct",
    torch_dtype=torch.float32,
    device_map=device,
    trust_remote_code=True
)

print("Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(
    "Qwen/Qwen2.5-Coder-0.5B-Instruct",
    trust_remote_code=True
)

print("Loading LoRA adapter...")
model = PeftModel.from_pretrained(
    base_model, 
    "/workspace/outputs/animal_sounds_enhanced"
)

print("Testing generation...")
# Test different prompts
test_prompts = [
    "Hello",
    "What is 2 + 2?",
    "asdfghjkl",
    "123",
    "?"
]

for prompt in test_prompts:
    print(f"\n--- Testing: {prompt} ---")
    messages = [{"role": "user", "content": prompt}]
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    
    inputs = tokenizer(text, return_tensors="pt")
    
    with torch.no_grad():
        outputs = model.generate(
            inputs.input_ids,
            max_new_tokens=20,
            temperature=0.1,
            do_sample=True,
            pad_token_id=tokenizer.pad_token_id,
            attention_mask=inputs.attention_mask
        )
    
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    # Extract assistant response
    if "assistant" in response:
        assistant_response = response.split("assistant")[-1].strip()
        print(f"Assistant: {assistant_response}")
    else:
        print(f"Response: {response}")