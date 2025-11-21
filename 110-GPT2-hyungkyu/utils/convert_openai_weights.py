"""
Convert OpenAI GPT-2 TensorFlow checkpoint to binary format for C++
Based on the load_weights_into_gpt pattern
"""

import numpy as np
import tensorflow as tf
import struct
import json
import os
from pathlib import Path

def load_gpt2_params_from_tf_ckpt(tf_ckpt_path, settings):
    """Load GPT-2 parameters from TensorFlow checkpoint (OpenAI format)"""

    params = {"blocks": [{} for _ in range(settings["n_layer"])]}

    for name, _ in tf.train.list_variables(tf_ckpt_path):
        variable_array = np.squeeze(tf.train.load_variable(tf_ckpt_path, name))

        variable_name_parts = name.split("/")[1:]  # Skip "model/"

        target_dict = params

        # Navigate to the correct nested dictionary
        for key in variable_name_parts[:-1]:
            if key.startswith("h"):  # Transformer block
                layer_number = int(key[1:])
                target_dict = params["blocks"][layer_number]
            elif key not in target_dict:
                target_dict[key] = {}
                target_dict = target_dict[key]
            else:
                target_dict = target_dict[key]

        # Assign the variable array to the last key
        last_key = variable_name_parts[-1]
        target_dict[last_key] = variable_array

    return params

def save_weights_binary(params, settings, output_file):
    """Save weights in binary format for C++"""

    print(f"Saving weights to {output_file}...")

    weights_map = {}

    # Token embeddings (wte)
    weights_map["wte"] = params["wte"]

    # Position embeddings (wpe)
    weights_map["wpe"] = params["wpe"]

    # Transformer blocks
    for i in range(settings["n_layer"]):
        block = params["blocks"][i]
        prefix = f"h.{i}."

        # Attention: Split c_attn into Q, K, V
        c_attn_w = block["attn"]["c_attn"]["w"]
        c_attn_b = block["attn"]["c_attn"]["b"]

        # Split weights along last axis (axis=-1) into 3 parts
        q_w, k_w, v_w = np.split(c_attn_w, 3, axis=-1)
        q_b, k_b, v_b = np.split(c_attn_b, 3, axis=-1)

        # Transpose weights (OpenAI uses [in_features, out_features], we use [out_features, in_features])
        weights_map[prefix + "attn.W_query.weight"] = q_w.T
        weights_map[prefix + "attn.W_key.weight"] = k_w.T
        weights_map[prefix + "attn.W_value.weight"] = v_w.T
        weights_map[prefix + "attn.W_query.bias"] = q_b
        weights_map[prefix + "attn.W_key.bias"] = k_b
        weights_map[prefix + "attn.W_value.bias"] = v_b

        # Attention output projection
        weights_map[prefix + "attn.out_proj.weight"] = block["attn"]["c_proj"]["w"].T
        weights_map[prefix + "attn.out_proj.bias"] = block["attn"]["c_proj"]["b"]

        # MLP (feedforward)
        weights_map[prefix + "ff.layers.0.weight"] = block["mlp"]["c_fc"]["w"].T
        weights_map[prefix + "ff.layers.0.bias"] = block["mlp"]["c_fc"]["b"]
        weights_map[prefix + "ff.layers.2.weight"] = block["mlp"]["c_proj"]["w"].T
        weights_map[prefix + "ff.layers.2.bias"] = block["mlp"]["c_proj"]["b"]

        # Layer norms
        weights_map[prefix + "norm1.scale"] = block["ln_1"]["g"]
        weights_map[prefix + "norm1.shift"] = block["ln_1"]["b"]
        weights_map[prefix + "norm2.scale"] = block["ln_2"]["g"]
        weights_map[prefix + "norm2.shift"] = block["ln_2"]["b"]

    # Final layer norm (check both possible locations)
    if "ln_f" in params:
        weights_map["final_norm.scale"] = params["ln_f"]["g"]
        weights_map["final_norm.shift"] = params["ln_f"]["b"]
    elif "g" in params and "b" in params:
        weights_map["final_norm.scale"] = params["g"]
        weights_map["final_norm.shift"] = params["b"]

    # LM head uses token embeddings (weight tying)
    weights_map["lm_head.weight"] = params["wte"]

    # Write binary file
    with open(output_file, 'wb') as f:
        # Write number of weights
        f.write(struct.pack('I', len(weights_map)))

        for name, tensor in sorted(weights_map.items()):
            # Convert to float32
            data = tensor.astype(np.float32)

            # Write name
            name_bytes = name.encode('utf-8')
            f.write(struct.pack('I', len(name_bytes)))
            f.write(name_bytes)

            # Write shape
            shape = data.shape if data.ndim > 0 else (1,)
            f.write(struct.pack('I', len(shape)))
            for dim in shape:
                f.write(struct.pack('I', dim))

            # Write data
            data.tofile(f)

            print(f"  {name}: {shape}")

    file_size = os.path.getsize(output_file)
    print(f"\nSaved {len(weights_map)} tensors")
    print(f"File size: {file_size / 1024 / 1024:.2f} MB")

def convert_checkpoint(checkpoint_dir, output_dir):
    """Convert OpenAI GPT-2 checkpoint to binary format"""

    # Load hparams
    hparams_file = os.path.join(checkpoint_dir, "hparams.json")
    with open(hparams_file, 'r') as f:
        settings = json.load(f)

    print("Model configuration:")
    print(f"  n_vocab: {settings['n_vocab']}")
    print(f"  n_ctx: {settings['n_ctx']}")
    print(f"  n_embd: {settings['n_embd']}")
    print(f"  n_layer: {settings['n_layer']}")
    print(f"  n_head: {settings['n_head']}")

    # Load checkpoint
    checkpoint_path = os.path.join(checkpoint_dir, "model.ckpt")
    print(f"\nLoading checkpoint from {checkpoint_path}...")

    params = load_gpt2_params_from_tf_ckpt(checkpoint_path, settings)

    # Create output directory
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    # Save config
    config_file = os.path.join(output_dir, "gpt2_config.txt")
    with open(config_file, 'w') as f:
        f.write(f"vocab_size={settings['n_vocab']}\n")
        f.write(f"max_seq_len={settings['n_ctx']}\n")
        f.write(f"d_model={settings['n_embd']}\n")
        f.write(f"num_layers={settings['n_layer']}\n")
        f.write(f"num_heads={settings['n_head']}\n")
    print(f"\nSaved config to {config_file}")

    # Save weights
    output_file = os.path.join(output_dir, "gpt2_weights.bin")
    save_weights_binary(params, settings, output_file)

    print("\nDone!")

if __name__ == '__main__':
    import sys

    checkpoint_dir = "../assets/weights/124M"
    output_dir = "../assets/weights/124M"

    if len(sys.argv) > 1:
        checkpoint_dir = sys.argv[1]
    if len(sys.argv) > 2:
        output_dir = sys.argv[2]

    convert_checkpoint(checkpoint_dir, output_dir)
