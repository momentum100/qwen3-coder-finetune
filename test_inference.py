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
prompt = "What is 2 + 2?"
messages = [{"role": "user", "content": prompt}]
text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

print(f"Input text: {text}")

inputs = tokenizer(text, return_tensors="pt")
print(f"Input tokens: {inputs.input_ids.shape}")

with torch.no_grad():
    outputs = model.generate(
        inputs.input_ids,
        max_new_tokens=50,
        temperature=1.0,
        do_sample=False,  # Greedy decoding
        pad_token_id=tokenizer.pad_token_id
    )

response = tokenizer.decode(outputs[0], skip_special_tokens=True)
print(f"\nFull response: {response}")

# Extract assistant response
if "assistant" in response:
    assistant_response = response.split("assistant")[-1].strip()
    print(f"\nAssistant: {assistant_response}")