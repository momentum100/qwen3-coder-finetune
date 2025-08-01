# VRAM Requirements for Qwen3-Coder Models

## Qwen3-Coder-30B-A3B-Instruct (MoE Model)

This is a Mixture-of-Experts model with 30B total parameters but only 3B active at once.

### VRAM Requirements by Quantization:

| Quantization | VRAM Required | Recommended GPU | Notes |
|--------------|---------------|-----------------|-------|
| 4-bit (QLoRA) | 17.5-19GB | RTX 3090/4090 | Best for fine-tuning |
| 8-bit | ~30GB | RTX A6000, A100 40GB | Better quality, needs more VRAM |
| FP16 | ~60GB | A100 80GB | Full precision, not practical |

### RunPod GPU Options for 30B Model:

1. **RTX 4090 ($0.69/hr)** ✅ RECOMMENDED
   - 24GB VRAM - Perfect for 4-bit
   - Fast training with Ada architecture
   - High availability

2. **RTX 3090 ($0.46/hr)** ✅ BUDGET OPTION
   - 24GB VRAM - Works with 4-bit
   - Slower but cheaper
   - May have lower availability

3. **RTX A6000 ($0.49/hr)** ⚡ IF AVAILABLE
   - 48GB VRAM - Can do 8-bit
   - Great price for the VRAM
   - Often low availability

4. **A100 80GB ($1.74/hr)** 💰 OVERKILL
   - 80GB VRAM - Can do FP16
   - Much more expensive
   - Not needed for LoRA

### Training Configuration for 30B Model:

```bash
# For RTX 3090/4090 (24GB VRAM)
python train_lora.py \
    --use_4bit \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 4 \
    --max_seq_length 256
```

### Memory Optimization Tips:

1. **Reduce sequence length**: 
   ```bash
   --max_seq_length 256  # Instead of 512
   ```

2. **Use gradient accumulation**:
   ```bash
   --gradient_accumulation_steps 8  # Simulate larger batch
   ```

3. **Reduce LoRA rank if needed**:
   ```bash
   --lora_r 8  # Instead of 16
   ```

### Disk Space Requirements:

- Model download: ~60GB (full precision)
- After 4-bit conversion: ~19GB
- Total needed: ~80GB during setup, ~20GB after

### Expected Training Time:

With RTX 4090 and animal sounds dataset (50 samples):
- Setup & download: 10-15 minutes (first time only)
- Training: 15-30 minutes
- Total cost: ~$0.35-0.70

### Alternative: Smaller Models

If 30B is too large, consider:
- **Qwen2.5-Coder-7B**: 8-10GB VRAM (4-bit)
- **Qwen2.5-Coder-1.5B**: 2-3GB VRAM (4-bit)