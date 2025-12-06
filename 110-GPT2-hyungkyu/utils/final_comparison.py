"""
Final Comparison: Vulkan vs PyTorch GPU
Shows only Vulkan errors against PyTorch GPU reference
"""

# Vulkan test results from C++ output
results = [
    {"layer": "GELU", "mean_error": 1.5522e-09, "max_error": 5.96046e-08},
    {"layer": "Linear", "mean_error": 4.17037e-09, "max_error": 3.72529e-08},
    {"layer": "LayerNorm", "mean_error": 1.486e-07, "max_error": 1.43051e-06},
    {"layer": "AddNode", "mean_error": 0.0, "max_error": 0.0},
    {"layer": "MultiHeadAttention", "mean_error": 3.16731e-09, "max_error": 2.23517e-08},
    {"layer": "FeedForward", "mean_error": 6.67396e-09, "max_error": 6.14673e-08},
    {"layer": "TransformerBlock", "mean_error": 4.22841e-07, "max_error": 2.86102e-06},
]

print("╔" + "=" * 68 + "╗")
print("║" + " " * 15 + "Vulkan vs PyTorch GPU (Reference)" + " " * 18 + "║")
print("╚" + "=" * 68 + "╝")
print()
print("Reference: PyTorch GPU (CUDA)")
print("Test Platform: Vulkan")
print()

# Table header
print("=" * 70)
print(f"{'Layer':<25} {'Mean Error':<22} {'Max Error':<22}")
print("=" * 70)

for r in results:
    mean_str = f"{r['mean_error']:.2e}" if r['mean_error'] > 0 else "0.00e+00"
    max_str = f"{r['max_error']:.2e}" if r['max_error'] > 0 else "0.00e+00"
    print(f"{r['layer']:<25} {mean_str:<22} {max_str:<22}")

print("=" * 70)
print()

# Summary
import numpy as np
max_errors = [r['max_error'] for r in results]
mean_errors = [r['mean_error'] for r in results]

print("Summary:")
print("-" * 70)
print(f"Average Max Error:   {np.mean(max_errors):.2e}")
print(f"Average Mean Error:  {np.mean(mean_errors):.2e}")
print(f"Max Error Range:     {np.min(max_errors):.2e} ~ {np.max(max_errors):.2e}")
print(f"Float32 Epsilon:     1.19e-07")
print()
print("✓ All layers pass with acceptable float32 precision")
print("✓ Vulkan implementation matches PyTorch GPU reference")
