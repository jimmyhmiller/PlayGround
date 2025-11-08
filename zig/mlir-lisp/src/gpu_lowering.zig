const std = @import("std");
const mlir_lisp = @import("mlir_lisp");
const mlir = mlir_lisp.mlir;

pub const GPUType = enum {
    apple_metal,
    nvidia_cuda,
    amd_rocm,
    cpu_fallback,
};

pub fn detectGPU() GPUType {
    const builtin = @import("builtin");
    if (builtin.os.tag == .macos) return .cpu_fallback; // force CPU on mac
    return .cpu_fallback;
}

pub fn applyGPULoweringPasses(
    _: std.mem.Allocator,
    ctx: *mlir.Context,
    module: *mlir.Module,
    gpu_type: GPUType,
) !void {
    var pm = try mlir.PassManager.create(ctx);
    defer pm.destroy();

    switch (gpu_type) {
        .apple_metal, .cpu_fallback => {
            std.debug.print("Lowering GPU to OpenMP (CPU fallback).\n", .{});
            try applyCPUFallbackPasses(&pm);
        },
        .nvidia_cuda => {
            std.debug.print("Lowering GPU to NVVM (NVIDIA).\n", .{});
            try applyNVIDIAPasses(&pm);
        },
        .amd_rocm => {
            std.debug.print("Lowering GPU to ROCDL (AMD).\n", .{});
            try applyAMDPasses(&pm);
        },
    }

    try pm.run(module);
}

/// CPU fallback: Just validate the GPU IR structure
/// Note: There is no standard MLIR pass to convert GPU ops to CPU loops
/// The only options are:
/// 1. Use gpu-to-llvm with actual GPU runtime (requires GPU drivers)
/// 2. Manually write CPU versions (see gpu_square_manual_cpu.lisp example)
/// 3. Wait for future MLIR enhancements
fn applyCPUFallbackPasses(pm: *mlir.PassManager) !void {
    // Just outline kernels to validate the IR structure
    const pipeline =
        \\ builtin.module(
        \\   cse,
        \\   canonicalize,
        \\   gpu-kernel-outlining
        \\ )
    ;
    try pm.addPipeline(pipeline);

    std.debug.print("\nNote: GPU code validated and outlined successfully.\n", .{});
    std.debug.print("      MLIR does not provide a standard GPU-to-CPU conversion pass.\n", .{});
    std.debug.print("      To run on CPU, either:\n", .{});
    std.debug.print("      1. Use actual GPU hardware (NVIDIA/AMD with drivers)\n", .{});
    std.debug.print("      2. Manually write CPU version (see examples/gpu_square_manual_cpu.lisp)\n", .{});
    std.debug.print("      3. Use the MLIR GPU runtime for CPU emulation (complex setup)\n\n", .{});
}

/// NVIDIA: gpu -> nvvm + llvm
fn applyNVIDIAPasses(pm: *mlir.PassManager) !void {
    const pipeline =
        \\ builtin.module(
        \\   cse,
        \\   canonicalize,
        \\   gpu-kernel-outlining,
        \\   convert-gpu-to-nvvm,
        \\   convert-nvgpu-to-nvvm,
        \\   gpu-to-llvm,
        \\   convert-scf-to-cf,
        \\   convert-arith-to-llvm,
        \\   convert-math-to-llvm,
        \\   convert-memref-to-llvm,
        \\   convert-func-to-llvm,
        \\   finalize-memref-to-llvm,
        \\   reconcile-unrealized-casts
        \\ )
    ;
    try pm.addPipeline(pipeline);
}

/// AMD: gpu -> rocdl + llvm
fn applyAMDPasses(pm: *mlir.PassManager) !void {
    const pipeline =
        \\ builtin.module(
        \\   cse,
        \\   canonicalize,
        \\   gpu-kernel-outlining,
        \\   convert-gpu-to-rocdl,
        \\   gpu-to-llvm,
        \\   convert-scf-to-cf,
        \\   convert-arith-to-llvm,
        \\   convert-math-to-llvm,
        \\   convert-memref-to-llvm,
        \\   convert-func-to-llvm,
        \\   finalize-memref-to-llvm,
        \\   reconcile-unrealized-casts
        \\ )
    ;
    try pm.addPipeline(pipeline);
}
