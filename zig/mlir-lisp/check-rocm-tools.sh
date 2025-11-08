#!/bin/bash
echo "==> Checking for MLIR ROCm tools..."
which mlir-rocm-runner || echo "mlir-rocm-runner not found"
which mlir-opt || echo "mlir-opt not found"

echo -e "\n==> Checking for ROCm runtime libraries..."
find /opt/rocm -name "libmlir_rocm_runtime.so" 2>/dev/null || echo "libmlir_rocm_runtime.so not in /opt/rocm"
find /usr/local/lib -name "libmlir_rocm_runtime.so" 2>/dev/null || echo "libmlir_rocm_runtime.so not in /usr/local/lib"
find /usr/lib -name "libmlir_rocm_runtime.so" 2>/dev/null || echo "libmlir_rocm_runtime.so not in /usr/lib"

echo -e "\n==> Checking LLVM libraries..."
ls -la /opt/homebrew/Cellar/llvm/20.1.7/lib/libmlir*.so 2>/dev/null | grep rocm || echo "No ROCm libraries in LLVM"
