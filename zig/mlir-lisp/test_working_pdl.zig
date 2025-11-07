/// Minimal working example of PDL pattern application
const std = @import("std");
const mlir = @import("mlir_lisp");

pub fn main() !void {
    const allocator = std.heap.page_allocator;

    // Initialize MLIR context
    var ctx = mlir.mlir.Context.init();
    defer ctx.deinit();
    ctx.loadAllDialects();

    std.debug.print("=== PDL Pattern Application Test ===\n\n", .{});

    // Define payload IR with custom operation
    const payload_ir =
        \\module {
        \\  func.func @test() -> i32 {
        \\    %0 = "custom.foo"() : () -> i32
        \\    return %0 : i32
        \\  }
        \\}
    ;

    // Define PDL patterns
    const pdl_patterns =
        \\module {
        \\  pdl.pattern @replace_foo : benefit(1) {
        \\    %type = pdl.type : i32
        \\    %op = pdl.operation "custom.foo" -> (%type : !pdl.type)
        \\    pdl.rewrite %op {
        \\      %attr = pdl.attribute = 42 : i32
        \\      %new_op = pdl.operation "arith.constant" {"value" = %attr} -> (%type : !pdl.type)
        \\      pdl.replace %op with %new_op
        \\    }
        \\  }
        \\}
    ;

    // Parse payload module
    std.debug.print("Parsing payload IR...\n", .{});
    var payload_module = try mlir.mlir.Module.parseString(&ctx, payload_ir);
    defer payload_module.destroy();

    std.debug.print("Payload BEFORE:\n", .{});
    payload_module.print();
    std.debug.print("\n", .{});

    // Parse PDL pattern module
    std.debug.print("Parsing PDL patterns...\n", .{});
    var pattern_module = try mlir.mlir.Module.parseString(&ctx, pdl_patterns);
    defer pattern_module.destroy();

    std.debug.print("PDL patterns:\n", .{});
    pattern_module.print();
    std.debug.print("\n", .{});

    // Create PDLPatternModule
    std.debug.print("Creating PDL pattern module...\n", .{});
    const pdl_pattern_mod = mlir.mlir.c.mlirPDLPatternModuleFromModule(pattern_module.module);
    defer mlir.mlir.c.mlirPDLPatternModuleDestroy(pdl_pattern_mod);

    // Convert to RewritePatternSet
    std.debug.print("Creating pattern set...\n", .{});
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

    std.debug.print("\nPayload AFTER:\n", .{});
    payload_module.print();
    std.debug.print("\n", .{});

    // Verify the transformation
    const module_str = try allocator.alloc(u8, 1024 * 1024);
    defer allocator.free(module_str);

    var callback = struct {
        buffer: []u8,
        pos: usize = 0,

        fn write(self: *@This(), data: []const u8) void {
            const available = self.buffer.len - self.pos;
            const to_copy = @min(data.len, available);
            @memcpy(self.buffer[self.pos..][0..to_copy], data[0..to_copy]);
            self.pos += to_copy;
        }

        fn callback(str_ref: mlir.mlir.c.MlirStringRef, user_data: ?*anyopaque) callconv(.C) void {
            const self: *@This() = @ptrCast(@alignCast(user_data));
            const data = str_ref.data[0..str_ref.length];
            self.write(data);
        }
    }{ .buffer = module_str };

    mlir.mlir.c.mlirOperationPrintWithCallback(
        payload_module.getOperation(),
        @TypeOf(callback).callback,
        &callback,
    );

    const output = module_str[0..callback.pos];

    if (std.mem.indexOf(u8, output, "custom.foo") != null) {
        std.debug.print("❌ FAIL: custom.foo still present after transformation!\n", .{});
        return error.TransformationFailed;
    }

    if (std.mem.indexOf(u8, output, "arith.constant") == null) {
        std.debug.print("❌ FAIL: arith.constant not found after transformation!\n", .{});
        return error.TransformationFailed;
    }

    std.debug.print("✅ SUCCESS: PDL patterns applied correctly!\n", .{});
    std.debug.print("   custom.foo → arith.constant (value = 42)\n", .{});
}
