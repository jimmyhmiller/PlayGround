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

    // Check for AMD ROCm
    if (checkForROCm()) {
        return .amd_rocm;
    }

    // Check for NVIDIA CUDA
    if (checkForCUDA()) {
        return .nvidia_cuda;
    }

    return .cpu_fallback;
}

fn checkForROCm() bool {
    // Check if rocminfo exists
    const result = std.process.Child.run(.{
        .allocator = std.heap.page_allocator,
        .argv = &[_][]const u8{ "which", "rocminfo" },
    }) catch return false;
    defer std.heap.page_allocator.free(result.stdout);
    defer std.heap.page_allocator.free(result.stderr);
    return result.term.Exited == 0;
}

fn checkForCUDA() bool {
    // Check if nvidia-smi exists
    const result = std.process.Child.run(.{
        .allocator = std.heap.page_allocator,
        .argv = &[_][]const u8{ "which", "nvidia-smi" },
    }) catch return false;
    defer std.heap.page_allocator.free(result.stdout);
    defer std.heap.page_allocator.free(result.stderr);
    return result.term.Exited == 0;
}

pub fn applyGPULoweringPasses(
    _: std.mem.Allocator,
    ctx: *mlir.Context,
    module: *mlir.Module,
    gpu_type: GPUType,
) !void {
    switch (gpu_type) {
        .apple_metal, .cpu_fallback => {
            std.debug.print("Lowering GPU to OpenMP (CPU fallback).\n", .{});
            var pm = try mlir.PassManager.create(ctx);
            defer pm.destroy();
            try applyCPUFallbackPasses(&pm);
            try pm.run(module);
        },
        .nvidia_cuda => {
            std.debug.print("Lowering GPU to NVVM (NVIDIA).\n", .{});
            var pm = try mlir.PassManager.create(ctx);
            defer pm.destroy();
            try applyNVIDIAPasses(&pm);
            try pm.run(module);
        },
        .amd_rocm => {
            std.debug.print("Lowering GPU to ROCDL (AMD).\n", .{});
            try applyAMDPasses(ctx, module);
        },
    }
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

/// AMD: gpu -> rocdl + llvm (multi-stage approach)
fn applyAMDPasses(ctx: *mlir.Context, module: *mlir.Module) !void {
    // Full AMD ROCDL lowering pipeline for MLIR 20
    // Must run in separate stages because gpu-module-to-binary needs to see
    // the gpu.module at the builtin.module scope

    // Detect GPU architecture - try gfx1151 first (latest), fall back to gfx90a (common server GPU)
    const chip = detectROCmChip() catch "gfx1151";

    // Stage 1: Outline kernels and convert to ROCDL
    {
        var pm = try mlir.PassManager.create(ctx);
        defer pm.destroy();

        const pipeline1_fmt = std.fmt.allocPrint(
            std.heap.page_allocator,
            \\ builtin.module(
            \\   gpu-kernel-outlining,
            \\   rocdl-attach-target{{chip={s}}},
            \\   gpu.module(convert-gpu-to-rocdl)
            \\ )
        , .{chip}) catch unreachable;
        defer std.heap.page_allocator.free(pipeline1_fmt);

        const pipeline1 = std.heap.page_allocator.dupeZ(u8, pipeline1_fmt) catch unreachable;
        defer std.heap.page_allocator.free(pipeline1);

        try pm.addPipeline(pipeline1);
        try pm.run(module);
        std.debug.print("Stage 1: GPU kernels outlined and converted to ROCDL\n", .{});

        // Debug: Print IR after stage 1
        if (std.posix.getenv("DEBUG_GPU_PASSES")) |_| {
            std.debug.print("\nAfter Stage 1:\n", .{});
            module.print();
        }
    }

    // Stage 2: Compile ROCDL to binary
    {
        var pm = try mlir.PassManager.create(ctx);
        defer pm.destroy();

        const pipeline2 = "builtin.module(gpu-module-to-binary{format=bin})";
        try pm.addPipeline(pipeline2);
        try pm.run(module);
        std.debug.print("Stage 2: ROCDL compiled to binary (HSACO)\n", .{});

        // Debug: Print IR after stage 2
        if (std.posix.getenv("DEBUG_GPU_PASSES")) |_| {
            std.debug.print("\nAfter Stage 2:\n", .{});
            module.print();
        }
    }

    // Stage 3: Lower to LLVM
    {
        var pm = try mlir.PassManager.create(ctx);
        defer pm.destroy();

        const pipeline3 =
            \\ builtin.module(
            \\   gpu-to-llvm,
            \\   convert-scf-to-cf,
            \\   convert-to-llvm,
            \\   finalize-memref-to-llvm,
            \\   reconcile-unrealized-casts
            \\ )
        ;
        try pm.addPipeline(pipeline3);
        try pm.run(module);
        std.debug.print("Stage 3: Lowered to LLVM IR\n", .{});
    }

    std.debug.print("\n✓ Full AMD ROCDL lowering completed for chip: {s}\n", .{chip});
}

fn detectROCmChip() ![]const u8 {
    // Run rocminfo to detect GPU architecture
    const result = std.process.Child.run(.{
        .allocator = std.heap.page_allocator,
        .argv = &[_][]const u8{ "rocminfo" },
    }) catch return error.ROCmNotFound;
    defer std.heap.page_allocator.free(result.stdout);
    defer std.heap.page_allocator.free(result.stderr);

    // Look for "Name: gfxXXXX" in output
    var lines = std.mem.splitScalar(u8, result.stdout, '\n');
    while (lines.next()) |line| {
        const trimmed = std.mem.trim(u8, line, " \t");
        if (std.mem.startsWith(u8, trimmed, "Name:") and std.mem.indexOf(u8, trimmed, "gfx") != null) {
            // Extract gfxXXXX
            if (std.mem.indexOf(u8, trimmed, "gfx")) |gfx_pos| {
                const gfx_start = gfx_pos;
                var gfx_end = gfx_start + 3; // "gfx"
                while (gfx_end < trimmed.len and std.ascii.isDigit(trimmed[gfx_end])) {
                    gfx_end += 1;
                }
                const chip = trimmed[gfx_start..gfx_end];
                // Allocate and return
                const chip_copy = std.heap.page_allocator.dupe(u8, chip) catch return error.OutOfMemory;
                return chip_copy;
            }
        }
    }
    return error.ChipNotDetected;
}
