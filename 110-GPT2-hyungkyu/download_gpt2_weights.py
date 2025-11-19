"""
Download GPT-2 weights from HuggingFace and save in binary format for C++
"""

import numpy as np
import struct
import os
from pathlib import Path

def download_and_save_weights(model_name='gpt2', output_dir='assets/weights/124M'):
    """
    Download GPT-2 weights from HuggingFace and save as binary files

    Args:
        model_name: 'gpt2', 'gpt2-medium', 'gpt2-large', or 'gpt2-xl'
        output_dir: directory to save weights
    """
    try:
        from transformers import GPT2LMHeadModel
    except ImportError:
        print("Error: transformers library not found")
        print("Install with: pip install transformers")
        return

    print(f"Downloading {model_name} weights from HuggingFace...")
    model = GPT2LMHeadModel.from_pretrained(model_name)

    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)

    # Get model config
    config = model.config
    print(f"\nModel Configuration:")
    print(f"  Vocabulary size: {config.vocab_size}")
    print(f"  Max position embeddings: {config.n_positions}")
    print(f"  Hidden size (d_model): {config.n_embd}")
    print(f"  Number of layers: {config.n_layer}")
    print(f"  Number of heads: {config.n_head}")

    # Save config
    config_file = output_path / f"{model_name}_config.txt"
    with open(config_file, 'w') as f:
        f.write(f"vocab_size={config.vocab_size}\n")
        f.write(f"max_seq_len={config.n_positions}\n")
        f.write(f"d_model={config.n_embd}\n")
        f.write(f"num_layers={config.n_layer}\n")
        f.write(f"num_heads={config.n_head}\n")
    print(f"\nSaved config to {config_file}")

    # Extract and save weights
    state_dict = model.state_dict()

    # Binary file format:
    # For each weight:
    #   - name_length (uint32)
    #   - name (string)
    #   - num_dims (uint32)
    #   - dims (uint32 array)
    #   - data (float32 array)

    weights_file = output_path / f"{model_name}_weights.bin"
    print(f"\nSaving weights to {weights_file}...")

    weight_count = 0
    total_params = 0

    with open(weights_file, 'wb') as f:
        # Write number of weights
        f.write(struct.pack('I', len(state_dict)))

        for name, tensor in state_dict.items():
            # Convert to numpy
            weights = tensor.detach().cpu().numpy().astype(np.float32)

            # Write name
            name_bytes = name.encode('utf-8')
            f.write(struct.pack('I', len(name_bytes)))
            f.write(name_bytes)

            # Write shape
            shape = weights.shape
            f.write(struct.pack('I', len(shape)))
            for dim in shape:
                f.write(struct.pack('I', dim))

            # Write data
            weights.tofile(f)

            weight_count += 1
            total_params += weights.size
            print(f"  [{weight_count}/{len(state_dict)}] {name}: {shape}")

    print(f"\nSuccessfully saved {weight_count} weight tensors")
    print(f"Total parameters: {total_params:,}")
    print(f"File size: {os.path.getsize(weights_file) / 1024 / 1024:.2f} MB")

    # Create weight name mapping file for reference
    mapping_file = output_path / f"{model_name}_weight_mapping.txt"
    with open(mapping_file, 'w') as f:
        f.write("HuggingFace GPT-2 Weight Name Mapping\n")
        f.write("=" * 80 + "\n\n")
        f.write("Embeddings:\n")
        f.write("  wte.weight -> token_weight [vocab_size, d_model]\n")
        f.write("  wpe.weight -> pos_weight [max_seq_len, d_model]\n\n")
        f.write("Transformer Blocks (layer i):\n")
        f.write("  h.{i}.ln_1.weight -> norm1_scale [d_model]\n")
        f.write("  h.{i}.ln_1.bias -> norm1_shift [d_model]\n")
        f.write("  h.{i}.attn.c_attn.weight -> [d_model, 3*d_model] (split to Q,K,V)\n")
        f.write("  h.{i}.attn.c_attn.bias -> [3*d_model] (split to Q,K,V bias)\n")
        f.write("  h.{i}.attn.c_proj.weight -> attn_wout [d_model, d_model]\n")
        f.write("  h.{i}.attn.c_proj.bias -> attn_wout_bias [d_model]\n")
        f.write("  h.{i}.ln_2.weight -> norm2_scale [d_model]\n")
        f.write("  h.{i}.ln_2.bias -> norm2_shift [d_model]\n")
        f.write("  h.{i}.mlp.c_fc.weight -> ff_w1 [d_model, 4*d_model]\n")
        f.write("  h.{i}.mlp.c_fc.bias -> ff_w1_bias [4*d_model]\n")
        f.write("  h.{i}.mlp.c_proj.weight -> ff_w2 [4*d_model, d_model]\n")
        f.write("  h.{i}.mlp.c_proj.bias -> ff_w2_bias [d_model]\n\n")
        f.write("Final Layer Norm:\n")
        f.write("  ln_f.weight -> final_norm_scale [d_model]\n")
        f.write("  ln_f.bias -> final_norm_shift [d_model]\n\n")
        f.write("Language Model Head:\n")
        f.write("  lm_head.weight -> (tied with wte.weight)\n")

    print(f"\nCreated weight mapping reference: {mapping_file}")
    print("\nDone!")

if __name__ == '__main__':
    import sys

    model_name = 'gpt2'  # Default to GPT-2 small
    if len(sys.argv) > 1:
        model_name = sys.argv[1]

    valid_models = ['gpt2', 'gpt2-medium', 'gpt2-large', 'gpt2-xl']
    if model_name not in valid_models:
        print(f"Error: Invalid model name '{model_name}'")
        print(f"Valid options: {', '.join(valid_models)}")
        sys.exit(1)

    download_and_save_weights(model_name)
