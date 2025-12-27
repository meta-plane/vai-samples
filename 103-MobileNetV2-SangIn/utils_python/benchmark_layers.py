import torch
import torch.nn as nn
import time
from typing import Tuple, Literal


class BenchmarkConfig:
    """Benchmark configuration"""
    WARMUP_ITER = 3
    BENCHMARK_ITER = 100

    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    assert DEVICE == 'cuda', "CUDA device not available"

    # Execution mode: "eager" or "compile"
    EXECUTION_MODE = "compile"

    # Compile backend: "auto", "inductor", "aot_eager", "cudagraphs"
    # "auto" will choose based on platform (aot_eager for Windows, inductor for Linux)
    COMPILE_BACKEND = "auto"


def run_benchmark(
    module: nn.Module,
    input_shape: Tuple[int, ...],
    warmup_iter: int = BenchmarkConfig.WARMUP_ITER,
    benchmark_iter: int = BenchmarkConfig.BENCHMARK_ITER
) -> float:
    """
    Run a single benchmark test case

    Args:
        module: PyTorch module to benchmark
        input_shape: Shape of input tensor (B, C, H, W) or (B, features)
        warmup_iter: Number of warmup iterations
        benchmark_iter: Number of benchmark iterations

    Returns:
        Average inference time in milliseconds
    """
    device = BenchmarkConfig.DEVICE
    execution_mode = BenchmarkConfig.EXECUTION_MODE

    # IMPORTANT: Reset torch.compile state for each test case
    # This ensures independent measurements without cross-contamination
    torch._dynamo.reset()

    module = module.to(device)
    module.eval()

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
        # Enable dynamic shapes to avoid recompilation for different spatial sizes
        module = torch.compile(module, backend=backend, mode="default", dynamic=True)

    # Create input tensor
    dummy_input = torch.randn(input_shape, device=device)

    # Warmup phase
    with torch.no_grad():
        for _ in range(warmup_iter):
            output = module(dummy_input)

    # Synchronize before benchmark (important for CUDA)
    if device == 'cuda':
        torch.cuda.synchronize()

    # Benchmark phase
    start_time = time.perf_counter()
    with torch.no_grad():
        for _ in range(benchmark_iter):
            output = module(dummy_input)
            if device == 'cuda':
                torch.cuda.synchronize()
    end_time = time.perf_counter()

    # Calculate results
    duration_ms = (end_time - start_time) * 1000
    avg_time_ms = duration_ms / benchmark_iter

    # Print results
    if backend_used:
        print(f"    [{execution_mode.upper()}/{backend_used}] Total: {duration_ms:.2f} ms, Avg: {avg_time_ms:.3f} ms per iter")
    else:
        print(f"    [{execution_mode.upper()}] Total: {duration_ms:.2f} ms, Avg: {avg_time_ms:.3f} ms per iter")
    print(f"    Output shape: {list(output.shape)}")

    return avg_time_ms


def benchmark_depthwise_conv():
    """Benchmark DepthwiseConv layer"""
    print("\n[Benchmark: DepthwiseConv]")
    print("=" * 48)

    test_cases = [
        # (inputH, inputW, channels, kernelSize, stride, padding, description)
        (224, 224, 32, 3, 1, 1, "Large input (224x224x32), K=3, S=1"),
        (112, 112, 64, 3, 1, 1, "Medium input (112x112x64), K=3, S=1"),
        (56, 56, 128, 3, 1, 1, "Small input (56x56x128), K=3, S=1"),
        (56, 56, 32, 3, 2, 1, "Stride=2 (56x56x32), K=3, S=2"),
        (56, 56, 32, 5, 1, 2, "Kernel=5 (56x56x32), K=5, S=1"),
    ]

    for inputH, inputW, channels, kernel_size, stride, padding, desc in test_cases:
        print(f"\n  Test: {desc}")
        print(f"  Config: Input=[{inputH}, {inputW}, {channels}], Kernel={kernel_size}, "
              f"Stride={stride}, Padding={padding}")

        # Create depthwise conv layer
        module = nn.Conv2d(
            in_channels=channels,
            out_channels=channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            groups=channels,  # Depthwise
            bias=False
        )

        # Input shape: (B, C, H, W)
        input_shape = (1, channels, inputH, inputW)
        run_benchmark(module, input_shape)

    print("\n" + "=" * 48)


def benchmark_pointwise_conv():
    """Benchmark PointwiseConv layer"""
    print("\n[Benchmark: PointwiseConv]")
    print("=" * 48)

    test_cases = [
        # (inputH, inputW, inputChannels, outputChannels, description)
        (224, 224, 32, 64, "Large input (224x224), 32->64 channels"),
        (112, 112, 64, 128, "Medium input (112x112), 64->128 channels"),
        (56, 56, 128, 256, "Small input (56x56), 128->256 channels"),
        (56, 56, 32, 192, "Expansion (56x56), 32->192 (6x)"),
        (28, 28, 192, 32, "Projection (28x28), 192->32 (1/6x)"),
    ]

    for inputH, inputW, in_channels, out_channels, desc in test_cases:
        print(f"\n  Test: {desc}")
        print(f"  Config: Input=[{inputH}, {inputW}, {in_channels}], "
              f"Output channels={out_channels}")

        # Create 1x1 conv layer
        module = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=False
        )

        input_shape = (1, in_channels, inputH, inputW)
        run_benchmark(module, input_shape)

    print("\n" + "=" * 48)


def benchmark_batch_norm():
    """Benchmark BatchNorm layer"""
    print("\n[Benchmark: BatchNorm]")
    print("=" * 48)

    test_cases = [
        # (inputH, inputW, channels, description)
        (224, 224, 32, "Large input (224x224x32)"),
        (112, 112, 64, "Medium input (112x112x64)"),
        (56, 56, 128, "Small input (56x56x128)"),
        (28, 28, 256, "Tiny input (28x28x256)"),
    ]

    for inputH, inputW, channels, desc in test_cases:
        print(f"\n  Test: {desc}")
        print(f"  Config: Input=[{inputH}, {inputW}, {channels}]")

        module = nn.BatchNorm2d(channels)
        input_shape = (1, channels, inputH, inputW)
        run_benchmark(module, input_shape)

    print("\n" + "=" * 48)


def benchmark_relu6():
    """Benchmark ReLU6 layer"""
    print("\n[Benchmark: ReLU6]")
    print("=" * 48)

    test_cases = [
        # (inputH, inputW, channels, description)
        (224, 224, 32, "Large input (224x224x32)"),
        (112, 112, 64, "Medium input (112x112x64)"),
        (56, 56, 128, "Small input (56x56x128)"),
    ]

    for inputH, inputW, channels, desc in test_cases:
        print(f"\n  Test: {desc}")
        print(f"  Config: Input=[{inputH}, {inputW}, {channels}]")

        module = nn.ReLU6(inplace=False)
        input_shape = (1, channels, inputH, inputW)
        run_benchmark(module, input_shape)

    print("\n" + "=" * 48)


def benchmark_max_pooling():
    """Benchmark MaxPooling layer"""
    print("\n[Benchmark: MaxPooling]")
    print("=" * 48)

    test_cases = [
        # (inputH, inputW, channels, poolSize, description)
        (224, 224, 32, 2, "Large input (224x224x32), Pool=2"),
        (112, 112, 64, 2, "Medium input (112x112x64), Pool=2"),
        (56, 56, 128, 2, "Small input (56x56x128), Pool=2"),
        (224, 224, 32, 3, "Large input (224x224x32), Pool=3"),
    ]

    for inputH, inputW, channels, pool_size, desc in test_cases:
        print(f"\n  Test: {desc}")
        print(f"  Config: Input=[{inputH}, {inputW}, {channels}], PoolSize={pool_size}")

        module = nn.MaxPool2d(kernel_size=pool_size, stride=pool_size)
        input_shape = (1, channels, inputH, inputW)
        run_benchmark(module, input_shape)

    print("\n" + "=" * 48)


def benchmark_global_avg_pool():
    """Benchmark GlobalAvgPool layer"""
    print("\n[Benchmark: GlobalAvgPool]")
    print("=" * 48)

    test_cases = [
        # (inputH, inputW, channels, description)
        (7, 7, 1280, "Final feature map (7x7x1280)"),
        (14, 14, 320, "Mid feature map (14x14x320)"),
        (28, 28, 160, "Early feature map (28x28x160)"),
    ]

    for inputH, inputW, channels, desc in test_cases:
        print(f"\n  Test: {desc}")
        print(f"  Config: Input=[{inputH}, {inputW}, {channels}]")

        module = nn.AdaptiveAvgPool2d((1, 1))
        input_shape = (1, channels, inputH, inputW)
        run_benchmark(module, input_shape)

    print("\n" + "=" * 48)


def benchmark_fully_connected():
    """Benchmark FullyConnected layer"""
    print("\n[Benchmark: FullyConnected]")
    print("=" * 48)

    test_cases = [
        # (inputDim, outputDim, description)
        (1280, 1000, "Classifier (1280->1000)"),
        (1280, 100, "Small classifier (1280->100)"),
        (512, 256, "Hidden layer (512->256)"),
        (2048, 1000, "Large classifier (2048->1000)"),
    ]

    for input_dim, output_dim, desc in test_cases:
        print(f"\n  Test: {desc}")
        print(f"  Config: Input={input_dim}, Output={output_dim}")

        module = nn.Linear(input_dim, output_dim)
        input_shape = (1, input_dim)
        run_benchmark(module, input_shape)

    print("\n" + "=" * 48)


def benchmark_convolution():
    """Benchmark standard Convolution layer"""
    print("\n[Benchmark: Convolution]")
    print("=" * 48)

    test_cases = [
        # (inputH, inputW, inChannels, outChannels, kernelSize, stride, padding, description)
        (224, 224, 3, 32, 3, 2, 1, "Stem layer (224x224x3->32), K=3, S=2"),
        (112, 112, 32, 64, 3, 2, 1, "Downsampling (112x112x32->64), K=3, S=2"),
        (56, 56, 64, 128, 3, 1, 1, "Same size (56x56x64->128), K=3, S=1"),
        (224, 224, 3, 64, 7, 2, 3, "Large kernel (224x224x3->64), K=7, S=2"),
    ]

    for inputH, inputW, in_ch, out_ch, kernel, stride, padding, desc in test_cases:
        print(f"\n  Test: {desc}")
        print(f"  Config: Input=[{inputH}, {inputW}, {in_ch}], Output ch={out_ch}, "
              f"Kernel={kernel}, Stride={stride}, Padding={padding}")

        module = nn.Conv2d(
            in_channels=in_ch,
            out_channels=out_ch,
            kernel_size=kernel,
            stride=stride,
            padding=padding,
            bias=False
        )

        input_shape = (1, in_ch, inputH, inputW)
        run_benchmark(module, input_shape)

    print("\n" + "=" * 48)


def benchmark_conv_bn_relu6():
    """Benchmark ConvBnReLU6 composite layer"""
    print("\n[Benchmark: ConvBnReLU6]")
    print("=" * 48)

    test_cases = [
        # (inputH, inputW, inChannels, outChannels, kernelSize, stride, padding, description)
        (224, 224, 3, 32, 3, 2, 1, "Stem layer (224x224x3->32), K=3, S=2"),
        (112, 112, 32, 64, 3, 2, 1, "Downsampling (112x112x32->64), K=3, S=2"),
        (56, 56, 64, 128, 3, 1, 1, "Same size (56x56x64->128), K=3, S=1"),
        (224, 224, 3, 32, 5, 2, 2, "Large kernel (224x224x3->32), K=5, S=2"),
    ]

    for inputH, inputW, in_ch, out_ch, kernel, stride, padding, desc in test_cases:
        print(f"\n  Test: {desc}")
        print(f"  Config: Input=[{inputH}, {inputW}, {in_ch}], Output ch={out_ch}, "
              f"Kernel={kernel}, Stride={stride}, Padding={padding}")

        module = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel, stride, padding, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU6(inplace=False)
        )

        input_shape = (1, in_ch, inputH, inputW)
        run_benchmark(module, input_shape)

    print("\n" + "=" * 48)


def benchmark_inverted_residual_block():
    """Benchmark InvertedResidualBlock (without skip connection)"""
    print("\n[Benchmark: InvertedResidualBlock]")
    print("=" * 48)

    test_cases = [
        # (inputH, inputW, inChannels, outChannels, expansionRatio, stride, description)
        (112, 112, 16, 24, 6, 2, "First IRB (112x112), 16->24, t=6, stride=2"),
        (56, 56, 24, 32, 6, 2, "Downsampling (56x56), 24->32, t=6, stride=2"),
        (56, 56, 32, 32, 6, 1, "Same size (56x56), 32->32, t=6, stride=1"),
        (28, 28, 64, 96, 6, 1, "Mid layer (28x28), 64->96, t=6, stride=1"),
        (14, 14, 160, 320, 6, 1, "Late layer (14x14), 160->320, t=6, stride=1"),
    ]

    for inputH, inputW, in_ch, out_ch, expansion, stride, desc in test_cases:
        expanded_ch = in_ch * expansion
        print(f"\n  Test: {desc}")
        print(f"  Config: Input=[{inputH}, {inputW}, {in_ch}], Output ch={out_ch}, "
              f"Expansion={expansion} (->{expanded_ch}), Stride={stride}")

        # Build InvertedResidualBlock components
        layers = []

        # Expand (1x1 conv)
        layers.extend([
            nn.Conv2d(in_ch, expanded_ch, 1, 1, 0, bias=False),
            nn.BatchNorm2d(expanded_ch),
            nn.ReLU6(inplace=False)
        ])

        # Depthwise (3x3 depthwise conv)
        layers.extend([
            nn.Conv2d(expanded_ch, expanded_ch, 3, stride, 1, groups=expanded_ch, bias=False),
            nn.BatchNorm2d(expanded_ch),
            nn.ReLU6(inplace=False)
        ])

        # Project (1x1 conv, no activation)
        layers.extend([
            nn.Conv2d(expanded_ch, out_ch, 1, 1, 0, bias=False),
            nn.BatchNorm2d(out_ch)
        ])

        module = nn.Sequential(*layers)
        input_shape = (1, in_ch, inputH, inputW)
        run_benchmark(module, input_shape)

    print("\n" + "=" * 48)


def benchmark_all_layers():
    """Run all layer benchmarks"""
    print("\n" + "=" * 64)
    print("  MobileNetV2 Layer-wise Benchmark Suite")
    print(f"  Device: {BenchmarkConfig.DEVICE}")
    print(f"  Execution Mode: {BenchmarkConfig.EXECUTION_MODE.upper()}")
    print("=" * 64)

    # Basic operation nodes
    benchmark_batch_norm()
    benchmark_relu6()
    benchmark_max_pooling()
    benchmark_global_avg_pool()
    benchmark_fully_connected()

    # Convolution nodes
    benchmark_convolution()
    benchmark_depthwise_conv()
    benchmark_pointwise_conv()

    # Composite nodes
    benchmark_conv_bn_relu6()
    benchmark_inverted_residual_block()

    print("\n" + "=" * 64)
    print("  Benchmark Complete")
    print("=" * 64)


if __name__ == "__main__":
    benchmark_all_layers()
