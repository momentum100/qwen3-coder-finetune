#!/usr/bin/env python3
"""
Upload fine-tuned model to Hugging Face Hub
"""

import os
import argparse
from huggingface_hub import HfApi, create_repo, upload_folder
from pathlib import Path

def upload_model(
    model_path,
    repo_name,
    hf_token=None,
    private=True,
    commit_message="Upload fine-tuned Qwen LoRA model"
):
    """Upload model to Hugging Face Hub"""
    
    api = HfApi(token=hf_token)
    
    # Create repository if it doesn't exist
    try:
        repo_url = create_repo(
            repo_id=repo_name,
            private=private,
            exist_ok=True,
            token=hf_token
        )
        print(f"✅ Repository created/found: {repo_url}")
    except Exception as e:
        print(f"❌ Error creating repository: {e}")
        return
    
    # Upload the model folder
    try:
        print(f"📤 Uploading model from {model_path}...")
        api.upload_folder(
            folder_path=model_path,
            repo_id=repo_name,
            commit_message=commit_message,
            token=hf_token
        )
        print(f"✅ Model uploaded successfully to: https://huggingface.co/{repo_name}")
    except Exception as e:
        print(f"❌ Error uploading model: {e}")

def main():
    parser = argparse.ArgumentParser(description="Upload fine-tuned model to Hugging Face")
    parser.add_argument(
        "--model_path",
        type=str,
        default="/workspace/outputs/animal_sounds_lora",
        help="Path to the fine-tuned model"
    )
    parser.add_argument(
        "--repo_name",
        type=str,
        required=True,
        help="Repository name on Hugging Face (e.g., 'username/model-name')"
    )
    parser.add_argument(
        "--hf_token",
        type=str,
        default=None,
        help="Hugging Face API token (can also use HF_TOKEN env var)"
    )
    parser.add_argument(
        "--public",
        action="store_true",
        help="Make the repository public"
    )
    
    args = parser.parse_args()
    
    # Get token from env if not provided
    hf_token = args.hf_token or os.environ.get("HF_TOKEN")
    if not hf_token:
        print("❌ Error: No Hugging Face token provided!")
        print("Please provide --hf_token or set HF_TOKEN environment variable")
        return
    
    # Check if model exists
    if not Path(args.model_path).exists():
        print(f"❌ Error: Model path does not exist: {args.model_path}")
        return
    
    upload_model(
        model_path=args.model_path,
        repo_name=args.repo_name,
        hf_token=hf_token,
        private=not args.public
    )

if __name__ == "__main__":
    main()