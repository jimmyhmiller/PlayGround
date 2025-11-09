/// Test PDL pattern application using the Rewrite API
const std = @import("std");
const mlir = @import("mlir_lisp");

pub fn main() !void {
    const allocator = std.heap.page_allocator;

    // Initialize MLIR context
    var ctx = mlir.mlir.Context.init();
    defer ctx.deinit();

    ctx.loadAllDialects();

    std.debug.print("=== Testing PDL Pattern Application ===\n\n", .{});

    // Load PDL pattern module
    std.debug.print("Loading PDL patterns...\n", .{});
    var pdl_file_contents = try std.fs.cwd().readFileAlloc(allocator, "test_pdl_greedy.mlir", 1024 * 1024);
    defer allocator.free(pdl_file_contents);

    const pdl_module = try mlir.mlir.Module.parseString(&ctx, pdl_file_contents);
    defer pdl_module.destroy();

    std.debug.print("PDL Module:\n", .{});
    pdl_module.print();
    std.debug.print("\n", .{});

    // Load payload module
    std.debug.print("Loading payload module...\n", .{});
    var payload_contents = try std.fs.cwd().readFileAlloc(allocator, "test_pdl_payload.mlir", 1024 * 1024);
    defer allocator.free(payload_contents);

    var payload_module = try mlir.mlir.Module.parseString(&ctx, payload_contents);
    defer payload_module.destroy();

    std.debug.print("Payload module BEFORE patterns:\n", .{});
    payload_module.print();
    std.debug.print("\n", .{});

    // Create PDL pattern module
    std.debug.print("Creating PDL pattern module...\n", .{});
    const pdl_pattern_mod = mlir.mlir.c.mlirPDLPatternModuleFromModule(pdl_module.module);
    defer mlir.mlir.c.mlirPDLPatternModuleDestroy(pdl_pattern_mod);

    // Convert to RewritePatternSet
    std.debug.print("Converting to RewritePatternSet...\n", .{});
    const pattern_set = mlir.mlir.c.mlirRewritePatternSetFromPDLPatternModule(pdl_pattern_mod);

    // Freeze patterns
    std.debug.print("Freezing patterns...\n", .{});
    const frozen = mlir.mlir.c.mlirFreezeRewritePattern(pattern_set);
    defer mlir.mlir.c.mlirFrozenRewritePatternSetDestroy(frozen);

    // Apply patterns
    std.debug.print("Applying patterns greedily...\n", .{});
    const config = mlir.mlir.c.MlirGreedyRewriteDriverConfig{ .ptr = null };
    const result = mlir.mlir.c.mlirApplyPatternsAndFoldGreedily(
        payload_module.module,
        frozen,
        config,
    );

    if (mlir.mlir.c.mlirLogicalResultIsFailure(result)) {
        std.debug.print("❌ Pattern application FAILED\n", .{});
        return error.PatternApplicationFailed;
    }

    std.debug.print("\nPayload module AFTER patterns:\n", .{});
    payload_module.print();
    std.debug.print("\n", .{});

    std.debug.print("✅ PDL patterns applied successfully!\n", .{});
}
