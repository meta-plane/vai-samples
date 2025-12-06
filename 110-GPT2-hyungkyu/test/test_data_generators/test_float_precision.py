"""
Test floating point precision differences
"""
import numpy as np

# 단정밀도 부동소수점 한계 테스트
np.random.seed(42)

# 간단한 연산
a = np.float32(0.1)
b = np.float32(0.2)
c = np.float32(0.3)

print("Float32 precision test:")
print(f"  a + b + c = {a + b + c}")
print(f"  Expected:   {0.6}")
print(f"  Error:      {abs((a + b + c) - 0.6):.2e}")
print()

# 큰 행렬 곱셈에서의 오차 누적
size = 768
x = np.random.randn(1, size).astype(np.float32) * 0.02
w = np.random.randn(size, size).astype(np.float32) * 0.02

# 순서를 바꿔서 계산
result1 = x @ w  # NumPy 순서
result2 = np.zeros_like(result1)
for i in range(size):
    for j in range(size):
        result2[0, i] += x[0, j] * w[j, i]  # 다른 순서

print("Matrix multiplication order test:")
print(f"  Max diff: {np.abs(result1 - result2).max():.2e}")
print(f"  Mean diff: {np.abs(result1 - result2).mean():.2e}")
print()

# 누적 합산 순서
values = np.random.randn(1000).astype(np.float32)
sum1 = np.sum(values)  # NumPy 최적화된 방법
sum2 = sum(values)     # Python 순차 합산
sum3 = values[0]
for v in values[1:]:
    sum3 += v

print("Accumulation order test:")
print(f"  NumPy sum:     {sum1}")
print(f"  Python sum:    {sum2}")
print(f"  Sequential:    {sum3}")
print(f"  Max diff:      {max(abs(sum1-sum2), abs(sum1-sum3), abs(sum2-sum3)):.2e}")
print()

# float32 epsilon
print("Float32 precision limits:")
print(f"  Machine epsilon: {np.finfo(np.float32).eps:.2e}")
print(f"  Our max error:   {2.25e-06:.2e}")
print(f"  Ratio:           {2.25e-06 / np.finfo(np.float32).eps:.1f}x epsilon")
