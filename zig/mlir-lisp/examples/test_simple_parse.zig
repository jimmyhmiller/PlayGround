const std = @import("std");
const mlir_lisp = @import("mlir_lisp");

pub fn main() !void {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();

    const program = @embedFile("../../../tmp/test_simple.mlir-lisp");

    std.debug.print("Program:\n{s}\n\n", .{program});

    // Parse
    var tok = mlir_lisp.Tokenizer.init(allocator, program);
    var r = try mlir_lisp.Reader.init(allocator, &tok);
    var value = try r.read();
    defer {
        value.deinit(allocator);
        allocator.destroy(value);
    }

    std.debug.print("Read value type: {s}\n", .{@tagName(value.type)});

    var p = mlir_lisp.Parser.init(allocator);
    var parsed_module = try p.parseModule(value);
    defer parsed_module.deinit();

    std.debug.print("Parsed module with {} operations\n", .{parsed_module.operations.len});

    // Create MLIR context
    var ctx = try mlir_lisp.mlir.Context.create();
    defer ctx.destroy();
    ctx.registerAllDialects();

    try ctx.getOrLoadDialect("func");
    try ctx.getOrLoadDialect("arith");

    // Build MLIR
    var builder = mlir_lisp.Builder.init(allocator, &ctx);
    defer builder.deinit();

    var mlir_module = try builder.buildModule(&parsed_module);
    defer mlir_module.destroy();

    std.debug.print("\nMLIR module:\n", .{});
    mlir_module.print();
}
