# GPT-2 Pretrained Weights

This guide explains how to download and use OpenAI's pretrained GPT-2 weights with this implementation.

## Quick Start

### Step 1: Install Python Dependencies

```bash
pip install transformers torch
```

### Step 2: Download Weights

Run the provided Python script to download GPT-2 weights from HuggingFace:

```bash
# Download GPT-2 Small (124M parameters) - Default
python download_gpt2_weights.py

# Or download other sizes
python download_gpt2_weights.py gpt2-medium    # 355M parameters
python download_gpt2_weights.py gpt2-large     # 774M parameters
python download_gpt2_weights.py gpt2-xl        # 1.5B parameters
```

This will create an `assets/weights/124M/` directory containing:
- `gpt2_weights.bin` - Binary weight file for C++
- `gpt2_config.txt` - Model configuration
- `gpt2_weight_mapping.txt` - Reference for weight name mapping

### Step 3: Build and Run

The C++ program will automatically load weights if available:

```bash
cmake --build build --config Debug
.\bin\debug\110-GPT2-hyungkyu.exe
```

## Weight File Format

The binary weight file uses a simple format for C++ compatibility:

```
File Structure:
- num_weights (uint32)
- For each weight:
  - name_length (uint32)
  - name (string)
  - num_dims (uint32)
  - dims (uint32 array)
  - data (float32 array)
```

## Weight Name Mapping

HuggingFace GPT-2 uses different naming conventions. The C++ loader automatically maps them:

### Embeddings
| HuggingFace | Our Model | Shape |
|-------------|-----------|-------|
| `wte.weight` | `token_weight` | [vocab_size, d_model] |
| `wpe.weight` | `pos_weight` | [max_seq_len, d_model] |

### Transformer Blocks (layer i)
| HuggingFace | Our Model | Shape | Notes |
|-------------|-----------|-------|-------|
| `transformer.h.{i}.ln_1.weight` | `norm1_scale` | [d_model] | LayerNorm 1 scale |
| `transformer.h.{i}.ln_1.bias` | `norm1_shift` | [d_model] | LayerNorm 1 bias |
| `transformer.h.{i}.attn.c_attn.weight` | `attn_wq`, `attn_wk`, `attn_wv` | [d_model, d_model] each | Split Q/K/V, transposed |
| `transformer.h.{i}.attn.c_proj.weight` | `attn_wout` | [d_model, d_model] | Transposed |
| `transformer.h.{i}.ln_2.weight` | `norm2_scale` | [d_model] | LayerNorm 2 scale |
| `transformer.h.{i}.ln_2.bias` | `norm2_shift` | [d_model] | LayerNorm 2 bias |
| `transformer.h.{i}.mlp.c_fc.weight` | `ff_w1` | [4*d_model, d_model] | Transposed |
| `transformer.h.{i}.mlp.c_proj.weight` | `ff_w2` | [d_model, 4*d_model] | Transposed |

### Final Layer Norm
| HuggingFace | Our Model | Shape |
|-------------|-----------|-------|
| `transformer.ln_f.weight` | `scale` | [d_model] |
| `transformer.ln_f.bias` | `shift` | [d_model] |

## Important Notes

### Weight Transposition

HuggingFace GPT-2 stores linear layer weights in shape `[out_features, in_features]`, but our implementation expects `[in_features, out_features]`. The loader automatically transposes these weights.

### Attention QKV Splitting

HuggingFace stores Q, K, V projection weights concatenated in a single tensor `c_attn.weight` with shape `[d_model, 3*d_model]`. The loader splits this into three separate weight tensors.

### Weight Tying

GPT-2 ties the token embedding weights (`wte.weight`) with the language model head. Currently, the LM head is simplified in this implementation.

## Model Sizes

| Model | Parameters | d_model | Layers | Heads | File Size |
|-------|------------|---------|--------|-------|-----------|
| gpt2 (small) | 124M | 768 | 12 | 12 | ~500 MB |
| gpt2-medium | 355M | 1024 | 24 | 16 | ~1.4 GB |
| gpt2-large | 774M | 1280 | 36 | 20 | ~3.0 GB |
| gpt2-xl | 1.5B | 1600 | 48 | 25 | ~6.0 GB |

## Troubleshooting

### "transformers library not found"
Install with: `pip install transformers`

### "Failed to open weights file"
Make sure you've run the download script first and the `assets/weights/124M/` directory exists.

### Out of Memory
Start with `gpt2` (small) first. Larger models require significantly more GPU memory.

## Usage in Code

```cpp
// Create model with appropriate config
GPT2Config config = GPT2SmallConfig();
GPT2 model(device, descPool, config);

// Load pretrained weights
model.loadWeights("assets/weights/124M/gpt2_weights.bin");

// Run inference
Tensor output = model.forward(input_ids);
```

## References

- [OpenAI GPT-2 Blog Post](https://openai.com/blog/better-language-models/)
- [HuggingFace GPT-2 Model](https://huggingface.co/gpt2)
- [GPT-2 Paper](https://d4mucfpksywv.cloudfront.net/better-language-models/language_models_are_unsupervised_multitask_learners.pdf)
