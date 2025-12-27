import torch
import torch.nn as nn
import torchvision.models as models
import time
from typing import Optional, Literal
import os


class BenchmarkConfig:
    """Benchmark configuration"""
    WARMUP_ITER = 3
    BENCHMARK_ITER = 10
    INPUT_SIZE = 224

    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    assert DEVICE == 'cuda', "CUDA device not available"

    # Execution mode: "eager" or "compile"
    EXECUTION_MODE = "compile"

    # Compile backend: "auto", "inductor", "aot_eager", "cudagraphs"
    # "auto" will choose based on platform (aot_eager for Windows, inductor for Linux)
    COMPILE_BACKEND = "auto"
    

def load_mobilenetv2_model(weights_path: Optional[str] = None, num_classes: int = 1000) -> nn.Module:
    """
    Load MobileNetV2 model with optional pretrained weights

    Args:
        weights_path: Path to .pth weights file (if None, use random weights)
        num_classes: Number of output classes

    Returns:
        MobileNetV2 model
    """
    print("Creating MobileNetV2...")
    model = models.mobilenet_v2(weights=None, num_classes=num_classes)

    if weights_path and os.path.exists(weights_path):
        print(f"Loading weights from {weights_path}")
        try:
            state_dict = torch.load(weights_path, map_location='cpu')
            model.load_state_dict(state_dict)
            print("Weights loaded successfully.")
        except Exception as e:
            print(f"Failed to load weights: {e}")
            print("Running with random weights.")
    else:
        if weights_path:
            print(f"Weights file not found at {weights_path}")
        print("Running with random weights.")

    print("MobileNetV2 created.")
    return model


def benchmark_mobilenetv2(
    model: nn.Module,
    input_size: int = BenchmarkConfig.INPUT_SIZE,
    warmup_iter: int = BenchmarkConfig.WARMUP_ITER,
    benchmark_iter: int = BenchmarkConfig.BENCHMARK_ITER,
    batch_size: int = 1
) -> float:
    """
    Benchmark MobileNetV2 model inference time

    Args:
        model: MobileNetV2 model to benchmark
        input_size: Input image size (default: 224)
        warmup_iter: Number of warmup iterations
        benchmark_iter: Number of benchmark iterations
        batch_size: Batch size for inference

    Returns:
        Average inference time in milliseconds
    """
    device = BenchmarkConfig.DEVICE
    execution_mode = BenchmarkConfig.EXECUTION_MODE

    model = model.to(device)
    model.eval()

    # Apply execution mode
    backend_used = None
    if execution_mode == "compile":
        # Determine backend
        compile_backend = BenchmarkConfig.COMPILE_BACKEND
        if compile_backend == "auto":
            import platform
            backend = "aot_eager" if platform.system() == "Windows" else "inductor"
        else:
            backend = compile_backend

        backend_used = backend
        print(f"Compiling model with torch.compile (backend={backend})...")
        model = torch.compile(model, backend=backend, mode="default")
        print("Model compilation complete.")

    if backend_used:
        print(f"Execution mode: {execution_mode.upper()} (backend={backend_used})")
    else:
        print(f"Execution mode: {execution_mode.upper()}")

    # Create dummy input: (B, C, H, W)
    dummy_input = torch.randn(batch_size, 3, input_size, input_size, device=device)

    print(f"Input Tensor Shape: [{batch_size}, 3, {input_size}, {input_size}]")

    # Warmup phase
    print(f"\n=== Warmup Phase (first {warmup_iter} iterations) ===")
    with torch.no_grad():
        for i in range(warmup_iter):
            print(f"Warmup iteration {i + 1}/{warmup_iter}...")
            output = model(dummy_input)
            if device == 'cuda':
                torch.cuda.synchronize()

    print("Warmup completed.\n")

    # Benchmark phase
    print(f"=== Benchmark Phase ({benchmark_iter} iterations) ===")

    if device == 'cuda':
        torch.cuda.synchronize()

    start_time = time.perf_counter()

    with torch.no_grad():
        for i in range(benchmark_iter):
            print(f"Benchmark iteration {i + 1}/{benchmark_iter}...")
            output = model(dummy_input)
            if device == 'cuda':
                torch.cuda.synchronize()

    end_time = time.perf_counter()

    # Calculate results
    duration_ms = (end_time - start_time) * 1000
    avg_time_ms = duration_ms / benchmark_iter

    # Print results
    print("\n=== Benchmark Results ===")
    print(f"Execution mode: {execution_mode.upper()}")
    print(f"Device: {device}")
    print(f"Input shape: [{batch_size}, 3, {input_size}, {input_size}]")
    print(f"Output shape: {list(output.shape)}")
    print(f"Total iterations: {benchmark_iter}")
    print(f"Total time: {duration_ms:.2f} ms")
    print(f"Average time per iteration: {avg_time_ms:.3f} ms")
    print("=========================\n")

    return avg_time_ms


def benchmark_with_image(
    model: nn.Module,
    image_path: str,
    warmup_iter: int = BenchmarkConfig.WARMUP_ITER,
    benchmark_iter: int = BenchmarkConfig.BENCHMARK_ITER
) -> torch.Tensor:
    """
    Benchmark MobileNetV2 with a real image input

    Args:
        model: MobileNetV2 model
        image_path: Path to input image
        warmup_iter: Number of warmup iterations
        benchmark_iter: Number of benchmark iterations

    Returns:
        Model output tensor
    """
    from PIL import Image
    import torchvision.transforms as transforms

    device = BenchmarkConfig.DEVICE
    execution_mode = BenchmarkConfig.EXECUTION_MODE

    model = model.to(device)
    model.eval()

    # Apply execution mode
    backend_used = None
    if execution_mode == "compile":
        # Determine backend
        compile_backend = BenchmarkConfig.COMPILE_BACKEND
        if compile_backend == "auto":
            import platform
            backend = "aot_eager" if platform.system() == "Windows" else "inductor"
        else:
            backend = compile_backend

        backend_used = backend
        print(f"Compiling model with torch.compile (backend={backend})...")
        model = torch.compile(model, backend=backend, mode="default")
        print("Model compilation complete.")

    if backend_used:
        print(f"Execution mode: {execution_mode.upper()} (backend={backend_used})")
    else:
        print(f"Execution mode: {execution_mode.upper()}")

    # Load and preprocess image (same as test.cpp preprocessing)
    print(f"Loading image from {image_path}...")
    image = Image.open(image_path).convert('RGB')

    # ImageNet normalization (same as test.cpp)
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    input_tensor = transform(image).unsqueeze(0).to(device)
    print(f"Image preprocessed. Input shape: {list(input_tensor.shape)}")

    # Warmup phase
    print(f"\n=== Warmup Phase (first {warmup_iter} iterations) ===")
    with torch.no_grad():
        for i in range(warmup_iter):
            print(f"Warmup iteration {i + 1}/{warmup_iter}...")
            output = model(input_tensor)
            if device == 'cuda':
                torch.cuda.synchronize()

    print("Warmup completed.\n")

    # Benchmark phase
    print(f"=== Benchmark Phase ({benchmark_iter} iterations) ===")

    if device == 'cuda':
        torch.cuda.synchronize()

    start_time = time.perf_counter()

    with torch.no_grad():
        for i in range(benchmark_iter):
            print(f"Benchmark iteration {i + 1}/{benchmark_iter}...")
            output = model(input_tensor)
            if device == 'cuda':
                torch.cuda.synchronize()

    end_time = time.perf_counter()

    # Calculate results
    duration_ms = (end_time - start_time) * 1000
    avg_time_ms = duration_ms / benchmark_iter

    # Print results
    print("\n=== Benchmark Results ===")
    print(f"Execution mode: {execution_mode.upper()}")
    print(f"Device: {device}")
    print(f"Input shape: {list(input_tensor.shape)}")
    print(f"Output shape: {list(output.shape)}")
    print(f"Total iterations: {benchmark_iter}")
    print(f"Total time: {duration_ms:.2f} ms")
    print(f"Average time per iteration: {avg_time_ms:.3f} ms")
    print("=========================\n")

    # Show top-5 predictions
    if output.shape[-1] >= 5:
        probabilities = torch.softmax(output, dim=1)
        top5_prob, top5_idx = torch.topk(probabilities, 5, dim=1)

        print("Top-5 Results:")
        for i in range(5):
            print(f"  {i + 1}. Class {top5_idx[0][i].item()}: {top5_prob[0][i].item():.6f}")
        print()

    return output


def main():
    """Main benchmark function"""
    print("=" * 64)
    print("  MobileNetV2 Full Model Benchmark")
    print(f"  Device: {BenchmarkConfig.DEVICE}")
    print(f"  Execution Mode: {BenchmarkConfig.EXECUTION_MODE.upper()}")
    print("=" * 64)

    # Load model (you can specify weights path here)
    # weights_path = "../weights/mobilenet_v2.pth"  # Optional
    weights_path = None  # Use random weights for benchmarking
    model = load_mobilenetv2_model(weights_path)

    # Benchmark with random input
    print("\n[Benchmark: Random Input]")
    benchmark_mobilenetv2(
        model,
        input_size=BenchmarkConfig.INPUT_SIZE,
        warmup_iter=BenchmarkConfig.WARMUP_ITER,
        benchmark_iter=BenchmarkConfig.BENCHMARK_ITER,
        batch_size=1
    )

    # Benchmark with real image (optional)
    # Uncomment the following lines if you have an image to test
    """
    image_path = "../img/shark.png"
    if os.path.exists(image_path):
        print("\n[Benchmark: Real Image Input]")
        model = load_mobilenetv2_model(weights_path)
        benchmark_with_image(
            model,
            image_path,
            warmup_iter=BenchmarkConfig.WARMUP_ITER,
            benchmark_iter=BenchmarkConfig.BENCHMARK_ITER
        )
    """

    print("\n" + "=" * 64)
    print("  Benchmark Complete")
    print("=" * 64)


if __name__ == "__main__":
    main()
