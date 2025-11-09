#!/bin/bash
set -e

echo "=== Converting .mlisp to .mlir ==="
./mlisp_to_mlir.sh scratch/vecadd.mlisp scratch/vecadd_from_mlisp.mlir

echo ""
echo "=== Running MLIR GPU pipeline ==="
mlir-opt scratch/vecadd_from_mlisp.mlir \
  | mlir-opt -convert-scf-to-cf \
  | mlir-opt -gpu-kernel-outlining \
  | mlir-opt -pass-pipeline='builtin.module(gpu.module(strip-debuginfo,convert-gpu-to-rocdl{use-bare-ptr-memref-call-conv=true}),rocdl-attach-target{chip=gfx1151})' \
  | mlir-opt -gpu-to-llvm=use-bare-pointers-for-kernels=true -reconcile-unrealized-casts -gpu-module-to-binary \
  | mlir-runner \
      --shared-libs=/usr/local/lib/libmlir_rocm_runtime.so \
      --shared-libs=/usr/local/lib/libmlir_runner_utils.so \
      --entry-point-result=void

echo ""
echo "=== GPU Test Complete! ==="
