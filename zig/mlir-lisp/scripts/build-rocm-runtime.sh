#!/bin/bash
set -e

# Build the MLIR ROCm runtime wrappers library
# This provides the mgpu* functions that bridge MLIR JIT to HIP runtime

REMOTE_HOST="192.168.0.55"
REMOTE_USER="${REMOTE_USER:-jimmyhmiller}"

echo "==> Building MLIR ROCm Runtime Wrappers on remote AMD machine..."

ssh -t "${REMOTE_USER}@${REMOTE_HOST}" << 'ENDSSH'
set -e

ROCM_PATH=/opt/rocm-6.4.4
LLVM_PATH=/usr/lib/llvm-20
BUILD_DIR=/tmp/mlir-rocm-runtime-build

echo "Checking for MLIR source..."
if [ ! -d "$HOME/llvm-project-20" ]; then
    echo "Downloading LLVM 20 source..."
    cd $HOME
    wget -q https://github.com/llvm/llvm-project/archive/refs/tags/llvmorg-20.1.7.tar.gz
    tar xzf llvmorg-20.1.7.tar.gz
    mv llvm-project-llvmorg-20.1.7 llvm-project-20
    rm llvmorg-20.1.7.tar.gz
fi

MLIR_SRC=$HOME/llvm-project-20/mlir

echo "Creating build directory..."
rm -rf $BUILD_DIR
mkdir -p $BUILD_DIR
cd $BUILD_DIR

echo "Compiling RocmRuntimeWrappers.cpp..."
g++ -std=c++17 -fPIC -shared \
    -D__HIP_PLATFORM_AMD__ \
    -I${LLVM_PATH}/include \
    -I${ROCM_PATH}/include \
    ${MLIR_SRC}/lib/ExecutionEngine/RocmRuntimeWrappers.cpp \
    -L${ROCM_PATH}/lib \
    -lamdhip64 \
    -Wl,-rpath,${ROCM_PATH}/lib \
    -o libmlir_rocm_runtime.so

echo "Installing library to project directory..."
INSTALL_DIR=$HOME/mlir-lisp-remote/lib
mkdir -p ${INSTALL_DIR}
cp libmlir_rocm_runtime.so ${INSTALL_DIR}/

echo "✓ ROCm runtime wrappers built and installed!"
echo "  Library: ${INSTALL_DIR}/libmlir_rocm_runtime.so"
echo "  Use this with ExecutionEngine or mlir-rocm-runner"

ENDSSH

echo ""
echo "==> Done! ROCm runtime wrappers installed on remote machine."
