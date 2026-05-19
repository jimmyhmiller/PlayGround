"""Torch reference: same (M=60, K=512, N=1, transpose_b=True) matmul that
MAX gets wrong on AMD/fp32/GPU.
"""
import torch

M, K, N = 60, 512, 1
A = torch.arange(M * K, dtype=torch.float32).reshape(M, K)
B = torch.ones(N, K, dtype=torch.float32)

# C = A @ B.T = A @ B.transpose(0, 1) ; shape (M, N)
C = A @ B.T
print("Torch (CPU), M=60, K=512, N=1, transpose_b=True")
for i in range(8):
    expected = float(i) * float(K) * float(K) + float(K) * float(K - 1) / 2.0
    print(f"  C[{i}] = {C[i, 0].item()}   expected = {expected}")
