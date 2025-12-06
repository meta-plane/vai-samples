"""
Generate all test data using PyTorch CPU
"""
import subprocess
import sys
import os

# List of test generators (CPU version)
generators = [
    "generate_gelu_test_cpu.py",
    "generate_linear_test_cpu.py",
    "generate_layernorm_test_cpu.py",
    "generate_add_test_cpu.py",
    "generate_attention_test_cpu.py",
    "generate_feedforward_test_cpu.py",
    "generate_transformer_test_cpu.py"
]

print("=" * 60)
print("Generating All Test Data with PyTorch CPU")
print("=" * 60)
print()

# Run each generator
for i, generator in enumerate(generators, 1):
    print(f"[{i}/{len(generators)}] Running {generator}...")
    print("-" * 60)

    result = subprocess.run(
        [sys.executable, generator],
        cwd=os.path.dirname(os.path.abspath(__file__)),
        capture_output=True,
        text=True
    )

    if result.returncode == 0:
        print(result.stdout)
        print("✓ SUCCESS")
    else:
        print(result.stdout)
        print(result.stderr)
        print("✗ FAILED")
        sys.exit(1)

    print()

print("=" * 60)
print("All test data generated successfully with PyTorch CPU!")
print("=" * 60)
print("\nGenerated files:")
print("  - gelu_test.json")
print("  - linear_test.json")
print("  - layernorm_test.json")
print("  - add_test.json")
print("  - attention_test.json")
print("  - feedforward_test.json")
print("  - transformer_test.json")
