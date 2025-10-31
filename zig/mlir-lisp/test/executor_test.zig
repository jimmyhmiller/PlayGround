const std = @import("std");
const mlir_lisp = @import("mlir_lisp");

test "executor - create and destroy" {
    var ctx = try mlir_lisp.mlir.Context.create();
    defer ctx.destroy();

    const config = mlir_lisp.ExecutorConfig{
        .opt_level = .O0,
    };

    var executor = mlir_lisp.Executor.init(std.testing.allocator, &ctx, config);
    defer executor.deinit();

    // Just verify we can create and destroy
    try std.testing.expect(executor.engine == null);
}

test "executor - different optimization levels" {
    var ctx = try mlir_lisp.mlir.Context.create();
    defer ctx.destroy();

    const opt_levels = [_]mlir_lisp.OptLevel{ .O0, .O1, .O2, .O3 };

    for (opt_levels) |opt_level| {
        const config = mlir_lisp.ExecutorConfig{
            .opt_level = opt_level,
        };

        var executor = mlir_lisp.Executor.init(std.testing.allocator, &ctx, config);
        defer executor.deinit();

        try std.testing.expectEqual(opt_level, executor.config.opt_level);
    }
}

test "executor - pass manager creation and pipeline" {
    var ctx = try mlir_lisp.mlir.Context.create();
    defer ctx.destroy();
    ctx.registerAllDialects();
    mlir_lisp.mlir.Context.registerAllPasses();

    // Create a simple module
    const loc = mlir_lisp.mlir.Location.unknown(&ctx);
    var mod = try mlir_lisp.mlir.Module.create(loc);
    defer mod.destroy();

    // Test pass manager
    var pm = try mlir_lisp.mlir.PassManager.create(&ctx);
    defer pm.destroy();

    // Add a simple pipeline - should succeed now that passes are registered
    try pm.addPipeline("builtin.module(canonicalize)");

    // Run the pass manager on the module
    try pm.run(&mod);
}
