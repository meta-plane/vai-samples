# GPT-2 Implementation with Vulkan Compute Shaders

C++ implementation of GPT-2 (Generative Pre-trained Transformer 2) using Vulkan compute shaders for GPU-accelerated inference.

## Features

- ✅ Full GPT-2 architecture implementation (12 layers, 768 hidden size, 12 attention heads)
- ✅ Vulkan compute shader-based neural network operations
- ✅ KV Cache for efficient autoregressive generation
- ✅ BPE (Byte-Pair Encoding) tokenizer
- ✅ Pretrained weight loading from OpenAI checkpoints
- ✅ Text generation with temperature sampling and top-k filtering
- ✅ Interactive generation mode
- ✅ Structured test framework with automated assertions

## Recent Improvements

### Test Infrastructure (2025-11-22)
- **Test Framework**: Standardized `BaseTest` class with automatic timing and error handling
- **Modular Tests**: Separated unit tests for layers and attention components
- **Test Runner**: `runAllTests` executable for running all unit tests
- **Test Data**: Organized Python generators and JSON reference data in `assets/test_data/`

### Code Organization
- **Inference Module**: Dedicated `model/inference.h/cpp` for pretrained model operations
- **Separation of Concerns**: Unit tests (`test/`) vs. inference (`model/inference`)
- **Helper Refactoring**: Extracted reusable helpers from complex layers (Attention, FeedForward)

### Build System
- **Aggregate Build Target**: `110-GPT2-hyungkyu` builds all executables
- **Individual Targets**: `gpt2-inference` (CLI) and `runAllTests` (unit tests)
- **Removed Legacy**: Consolidated main files, removed old test executable
- **Portable Paths**: Relative asset paths work on any machine

## Project Structure

```
110-GPT2-hyungkyu/
├── assets/                  # Model weights and tokenizer data
│   ├── weights/124M/        # GPT-2 pretrained weights (gpt2_weights.bin)
│   ├── vocab.json           # BPE vocabulary
│   └── merges.txt           # BPE merge rules
├── model/                   # GPT-2 architecture implementation
│   ├── inference.h/cpp      # Inference API (load, generate, interactive)
│   ├── gpt2Net.h            # Main GPT-2 network
│   ├── attention/           # Multi-head attention + KV cache
│   └── transformerBlock/    # Transformer components (LayerNorm, GELU, etc.)
├── test/                    # Unit tests with test framework
│   ├── testFramework.h      # BaseTest, TestAssert classes
│   └── runAllTests.cpp      # Test runner
├── core/                    # Vulkan-based neural network framework
├── tokenizer/               # BPE tokenizer
├── utils/                   # Weight download/conversion scripts
├── main.cpp                 # CLI entry point
└── CMakeLists.txt          # Build configuration

```

## Prerequisites

### C++ Build Tools
- CMake 3.15+
- C++17 compatible compiler (MSVC, GCC, Clang)
- Vulkan SDK

### Python (for weight conversion)
- Python 3.7+
- Required packages:
  ```bash
  pip install numpy
  pip install tensorflow  # For convert_openai_weights.py
  # OR
  pip install torch transformers  # For download_gpt2_weights.py
  ```

## Setup Instructions

### Step 1: Download and Convert Weights

Download GPT-2 checkpoint from OpenAI and convert to binary format in one command:

```bash
cd 110-GPT2-hyungkyu
python utils/setup_weights.py
```

This script will:
1. Download OpenAI GPT-2 checkpoint files from CDN
2. Convert to binary format using `convert_openai_weights.py`
3. Generate `assets/weights/124M/gpt2_weights.bin` and `gpt2_config.txt`

**Required Python packages:**
```bash
pip install numpy tensorflow
```

**For different model sizes:**
```bash
python utils/setup_weights.py --model 355M --output-dir assets/weights/355M
# Options: 124M (default), 355M, 774M, 1558M
```

**Note:** Tokenizer files (`vocab.json`, `merges.txt`) are already included in the repository under `assets/` folder.

### Step 2: Build the Project

```bash
# From vai-samples root directory
cmake -B build

# Build all targets (recommended)
cmake --build build --config Debug --target 110-GPT2-hyungkyu

# Or build specific targets
cmake --build build --config Debug --target gpt2-inference  # Inference CLI only
cmake --build build --config Debug --target runAllTests     # Tests only
```

**Build Targets:**
- `110-GPT2-hyungkyu` - Builds all executables (aggregate target)
- `gpt2-inference` - Inference CLI only
- `runAllTests` - Unit test runner only

**Generated Executables:**
- **bin/debug/gpt2-inference.exe** - Inference CLI (generate, compare, interactive modes)
- **bin/debug/runAllTests.exe** - Unit test runner

Or for Release build:
```bash
cmake --build build --config Release --target 110-GPT2-hyungkyu
```

### Step 3: Run Inference

**IMPORTANT:** Always run from the `110-GPT2-hyungkyu` directory so asset paths work correctly.

```bash
# From vai-samples root directory
cd 110-GPT2-hyungkyu

# Generate text with default settings
../bin/debug/gpt2-inference.exe generate "Once upon a time" --max-tokens 50

# Interactive mode
../bin/debug/gpt2-inference.exe interactive

# Compare generation modes (with/without KV cache)
../bin/debug/gpt2-inference.exe compare "Hello world" --max-tokens 20

# Show help
../bin/debug/gpt2-inference.exe --help
```

---

## Usage

### Text Generation

The `gpt2-inference` executable provides three modes:

#### 1. Generate Mode (Default)
```bash
# Basic generation
../bin/debug/gpt2-inference.exe generate "The future of AI is"

# With options
../bin/debug/gpt2-inference.exe generate "Once upon a time" \
    --max-tokens 100 \
    --temperature 0.9 \
    --top-k 50 \
    --no-cache  # Disable KV cache
```

#### 2. Compare Mode
Compare generation with and without KV cache:
```bash
../bin/debug/gpt2-inference.exe compare "Hello world" --max-tokens 30
```

Output shows:
- Generated text from both modes
- Performance metrics (tokens/sec)
- Speed improvement from KV cache

#### 3. Interactive Mode
```bash
../bin/debug/gpt2-inference.exe interactive

# Then enter prompts interactively:
> Once upon a time
> [Generated text appears]
> The future of artificial intelligence
> [Generated text appears]
> exit  # or Ctrl+C to quit
```

### Command-Line Options

```
gpt2-inference <mode> [prompt] [options]

Modes:
  generate      Generate text from a prompt (default)
  compare       Compare cached vs standard generation
  interactive   Interactive generation mode

Options:
  --max-tokens N      Maximum tokens to generate (default: 50)
  --temperature F     Sampling temperature 0.0-2.0 (default: 0.8)
  --top-k N          Top-k sampling (default: 40)
  --seed N           Random seed (default: 42)
  --no-cache         Disable KV cache
  --help, -h         Show help message
```

### Generation Parameters

**Parameter Guide:**

- **Temperature** (recommended: 0.7-1.0)
  - `0.0`: Greedy decoding (deterministic, but prone to repetition)
  - `0.5-0.7`: Conservative, more coherent
  - `0.8-1.0`: Creative, diverse (current default: 0.8)
  - `>1.0`: Very random, may reduce quality

- **Top-k** (recommended: 40-50)
  - Limits sampling to top-k most probable tokens
  - `0`: No filtering (use all tokens)
  - `40`: Good balance of quality and diversity (current default)
  - Higher values = more diversity, but may include low-quality tokens

- **KV Cache**
  - Enabled by default for faster generation
  - Caches key/value tensors in attention layers
  - Typical speedup: 2-3x for autoregressive generation
  - Use `--no-cache` to disable for testing

## Example Output

```
=== GPT-2 Text Generation ===
Prompt: "Once upon a time"
Max tokens: 50
Temperature: 0.8
Top-k: 40
Using KV cache: Yes

--- Generated Text ---
Once upon a time, a number of people were able to find a shelter
which was not a suitable place to live.

These people were all on the same level as the village head,
which was probably the best possible shelter for them.
--- End of Generation ---

Generated 50 new tokens (total: 54 tokens)
Generation time: 1247 ms (1.25 sec)
Generation speed: 40.10 tokens/sec
```

## Testing

### Run Unit Tests

```bash
cd 110-GPT2-hyungkyu
../bin/debug/runAllTests.exe
```

**Current Status:**
- Test framework implemented with `BaseTest` class
- 2/13 tests converted to NeuralNet-based architecture
- Remaining tests temporarily disabled pending conversion

### Test Framework Features

- **Automatic Timing**: Each test reports execution time
- **Structured Assertions**: `TestAssert::assertEqual()`, `assertShape()`, `assertClose()`
- **Error Handling**: Automatic exception catching with clear error messages
- **Test Suites**: Organized by component (layers, attention)

See `test/README.md` for detailed test framework documentation.

## Performance

- **Model**: GPT-2 Small (124M parameters)
- **Generation Speed**:
  - With KV cache: ~35-45 tokens/sec
  - Without cache: ~12-20 tokens/sec
- **GPU Memory**: ~1-2GB VRAM for 100 token generation
- **Token Limit**: Supports 100+ tokens per generation
- **Descriptor Pool**: 10,000 descriptor sets for long-form generation

## Technical Improvements

### KV Cache Implementation
- **Autoregressive Optimization**: Caches key/value tensors to avoid recomputation
- **Dynamic Length Tracking**: `current_len` automatically updated during generation
- **Memory Efficient**: Only stores previously computed K/V, not full sequences
- **Typical Speedup**: 2-3x faster than standard generation

### Memory Management
- **Global Context Initialization**: Vulkan resources centralized in `core/globalContext.cpp`
- **Descriptor Pool Scaling**: Increased from 500 to 10,000 descriptor sets
- **BufferPool**: Auto-managed with size limits to prevent unbounded memory growth

### Code Quality
- **Refactored Helpers**: Extracted helper functions from Attention and FeedForward nodes
- **Modular Architecture**: Clean separation between layers, models, tests, and inference
- **Vulkan API Compatibility**: All tests use correct `newCommandBuffer()` pattern

## Troubleshooting

### "Failed to open vocab.json"
- Ensure `vocab.json` and `merges.txt` are in `assets/` folder
- Check file permissions

### "Pretrained weights not found"
- Run weight download/conversion scripts in `utils/`
- Verify `assets/weights/124M/gpt2_weights.bin` exists

### "VkResult is UNKNOWN_ERROR" or Crash During Generation
- **Descriptor Pool Exhausted**: Fixed by increasing descriptor pool size to 10,000
  - Ensure you're using the latest version with `globalContext.cpp`
- **GPU Out of Memory**: Close other GPU-intensive applications
  - Current implementation supports up to 100+ tokens
  - VRAM usage is ~1-2GB for 100 tokens

### Build Errors
- Ensure Vulkan SDK is installed
- Check CMake version (3.15+)
- Verify C++17 compiler support

### Test Failures
- **Tests temporarily disabled**: Unit tests are being converted to use NeuralNet architecture
- **Status**: 2/13 tests converted (see `test/README.md` for details)
- Legacy tests in `model/` subdirectories still functional

## Architecture Details

### Model Configuration (GPT-2 Small)
- Vocabulary size: 50,257
- Context length: 1,024 tokens
- Hidden size (d_model): 768
- Number of layers: 12
- Number of attention heads: 12
- Feedforward size: 3,072 (4 × d_model)

### Components
- **Embedding Layer**: Token + positional embeddings
- **Transformer Blocks**: Multi-head attention + feedforward + layer normalization
- **KV Cache**: Optimized autoregressive generation with cached attention states
- **Language Model Head**: Linear projection to vocabulary (weight-tied with token embeddings)

## References

- [OpenAI GPT-2 Paper](https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf)
- [GPT-2 on HuggingFace](https://huggingface.co/gpt2)
- [OpenAI GPT-2 Repository](https://github.com/openai/gpt-2)

## License

This is an educational implementation. GPT-2 weights are released by OpenAI under the MIT License.
