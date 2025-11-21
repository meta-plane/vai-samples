"""
Compare GPT-2 generation between C++ Vulkan implementation and PyTorch

Usage:
    python compare_with_pytorch.py
"""

import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer

def generate_with_pytorch(prompt, max_new_tokens=25, temperature=0.0, top_k=None):
    """
    Generate text using PyTorch GPT-2

    Args:
        prompt: Input text
        max_new_tokens: Number of tokens to generate
        temperature: 0.0 for greedy (deterministic), >0 for sampling
        top_k: Top-k sampling (None to disable)
    """
    # Load model and tokenizer
    print("Loading GPT-2 model and tokenizer...")
    model_name = "gpt2"  # 124M parameters, same as your C++ model
    tokenizer = GPT2Tokenizer.from_pretrained(model_name)
    model = GPT2LMHeadModel.from_pretrained(model_name)
    model.eval()

    # Encode prompt
    input_ids = tokenizer.encode(prompt, return_tensors="pt")
    print(f"\nPrompt: \"{prompt}\"")
    print(f"Encoded to {input_ids.shape[1]} tokens: {input_ids[0].tolist()}")

    # Generate
    print(f"\nGenerating {max_new_tokens} new tokens...")
    print(f"Temperature: {temperature} {'(greedy)' if temperature == 0.0 else ''}")

    with torch.no_grad():
        if temperature == 0.0:
            # Greedy decoding (deterministic)
            output_ids = model.generate(
                input_ids,
                max_new_tokens=max_new_tokens,
                do_sample=False,  # Greedy
                pad_token_id=tokenizer.eos_token_id
            )
        else:
            # Sampling with temperature and top-k
            output_ids = model.generate(
                input_ids,
                max_new_tokens=max_new_tokens,
                do_sample=True,
                temperature=temperature,
                top_k=top_k if top_k else 50,
                pad_token_id=tokenizer.eos_token_id
            )

    # Decode
    generated_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    generated_ids = output_ids[0].tolist()

    print(f"\n--- Generated Text ---")
    print(generated_text)
    print(f"--- End of Generation ---\n")

    print(f"Generated {len(generated_ids) - input_ids.shape[1]} new tokens (total: {len(generated_ids)} tokens)")
    print(f"Token IDs: {generated_ids}")

    return generated_text, generated_ids


if __name__ == "__main__":
    print("=" * 80)
    print("PyTorch GPT-2 Text Generation (for comparison with C++ Vulkan)")
    print("=" * 80)

    # Test 1: Greedy decoding (deterministic, should match C++)
    print("\n=== Test 1: Greedy Decoding (Deterministic) ===")
    generate_with_pytorch(
        prompt="The future of artificial intelligence is",
        max_new_tokens=25,
        temperature=0.0  # Greedy
    )

    print("\n" + "=" * 80)
    print("\nComparison Notes:")
    print("1. Greedy decoding (temperature=0) should produce IDENTICAL results")
    print("   between PyTorch and C++ if the models are truly the same.")
    print("")
    print("2. If results differ, check:")
    print("   - Weight loading correctness")
    print("   - Numerical precision (float32 vs float16)")
    print("   - Layer normalization epsilon values")
    print("   - Attention mask handling")
    print("")
    print("3. With sampling (temperature>0), results will differ due to")
    print("   different random number generators, even with same seed.")
