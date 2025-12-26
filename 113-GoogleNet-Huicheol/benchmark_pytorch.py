import torch
import torchvision.models as models
import time
import os

def benchmark(device_name):
    device = torch.device(device_name)
    print(f"Benchmarking on {device_name}...")
    
    try:
        model = models.googlenet(weights=models.GoogLeNet_Weights.IMAGENET1K_V1).to(device)
        model.eval()
        
        # Dummy input
        input_tensor = torch.randn(1, 3, 224, 224).to(device)
        
        # Warmup
        print("Warming up (10 iters)...")
        with torch.no_grad():
            for _ in range(10):
                model(input_tensor)
                if device_name == 'cuda':
                    torch.cuda.synchronize()
        
        # Benchmark
        print("Benchmarking (50 iters)...")
        start_time = time.time()
        with torch.no_grad():
            for _ in range(50):
                model(input_tensor)
                if device_name == 'cuda':
                    torch.cuda.synchronize()
        end_time = time.time()
        
        total_time = end_time - start_time
        avg_time = (total_time / 50.0) * 1000.0
        print(f"Total time: {total_time:.4f}s")
        print(f"Average inference time: {avg_time:.4f} ms")
        
    except Exception as e:
        print(f"Failed to benchmark on {device_name}: {e}")

if __name__ == "__main__":
    benchmark('cpu')
    if torch.cuda.is_available():
        benchmark('cuda')
    else:
        print("CUDA not available, skipping GPU benchmark.")
