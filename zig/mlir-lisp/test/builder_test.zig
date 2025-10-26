const std = @import("std");
const mlir_lisp = @import("mlir_lisp");
const Tokenizer = mlir_lisp.Tokenizer;
const Reader = mlir_lisp.Reader;
const Parser = mlir_lisp.Parser;
const Builder = mlir_lisp.Builder;
const mlir = mlir_lisp.mlir;
const test_utils = @import("test_utils.zig");

// Example 1 — simple constant
const example1 =
    \\(mlir
    \\  (operation
    \\    (name arith.constant)
    \\    (result-bindings [%c0])
    \\    (result-types i32)
    \\    (attributes { :value (: 42 i32) })
    \\    (location (#unknown))))
;

test "builder - example 1 simple constant" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const allocator = arena.allocator();

    // Parse the source
    var tok = Tokenizer.init(allocator, example1);
    var rd = try Reader.init(allocator, &tok);
    const value = try rd.read();

    var parser = Parser.init(allocator);
    var parsed_module = try parser.parseModule(value);

    // Create MLIR context
    var ctx = try mlir.Context.create();
    defer ctx.destroy();

    // Register all dialects
    ctx.registerAllDialects();

    // Build MLIR IR
    var builder = Builder.init(allocator, &ctx);

    var mlir_module = try builder.buildModule(&parsed_module);
    defer mlir_module.destroy();

    // Print the module
    std.debug.print("\n=== Built MLIR Module ===\n", .{});
    mlir_module.print();
    std.debug.print("\n", .{});

    // Validate with mlir-opt
    try test_utils.validateMLIRWithOpt(std.testing.allocator, mlir_module);
}
