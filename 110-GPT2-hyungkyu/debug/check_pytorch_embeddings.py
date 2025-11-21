"""
Compare token embeddings between binary weights and PyTorch model
"""
import torch
from transformers import GPT2LMHeadModel

# Load PyTorch model
print("Loading PyTorch GPT-2 model...")
model = GPT2LMHeadModel.from_pretrained("gpt2")
model.eval()

# Get token embeddings
wte = model.transformer.wte.weight.data  # shape: [50257, 768]

print(f"\nToken embedding shape: {wte.shape}")

# Token 0
print(f"\nToken 0 - First 10 values:")
print(wte[0, :10].tolist())

# Token 284 (' to')
print(f"\nToken 284 (' to') - First 10 values:")
print(wte[284, :10].tolist())

# Statistics
print(f"\nStatistics (all embeddings):")
print(f"  Mean: {wte.mean():.6f}")
print(f"  Std: {wte.std():.6f}")
print(f"  Min: {wte.min():.6f}")
print(f"  Max: {wte.max():.6f}")
