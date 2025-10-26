const std = @import("std");
const testing = std.testing;
const Builder = @import("mlir_lisp").Builder;
const Tokenizer = @import("mlir_lisp").Tokenizer;
const Reader = @import("mlir_lisp").Reader;
const AttrExpr = @import("mlir_lisp").AttrExpr;
const mlir = @import("mlir_lisp").mlir;

test "buildTypeFromValue - simple type i32" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const allocator = arena.allocator();

    var ctx = try mlir.Context.create();
    defer ctx.destroy();
    ctx.registerAllDialects();

    var builder = Builder.init(allocator, &ctx);
    defer builder.deinit();

    // Parse "i32"
    const input = "i32";
    var tok = Tokenizer.init(allocator, input);
    var r = try Reader.init(allocator, &tok);
    const value = try r.read();

    // The value should be an identifier (plain builtin type)
    try testing.expect(value.type == .identifier);

    // Build the type
    const mlir_type = try builder.buildTypeFromValue(value);

    // Verify it's not null
    try testing.expect(!mlir.c.mlirTypeIsNull(mlir_type));
}

test "buildTypeFromValue - simple type f64" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const allocator = arena.allocator();

    var ctx = try mlir.Context.create();
    defer ctx.destroy();
    ctx.registerAllDialects();

    var builder = Builder.init(allocator, &ctx);
    defer builder.deinit();

    // Parse "f64"
    const input = "f64";
    var tok = Tokenizer.init(allocator, input);
    var r = try Reader.init(allocator, &tok);
    const value = try r.read();

    // Build the type
    const mlir_type = try builder.buildTypeFromValue(value);

    // Verify it's not null
    try testing.expect(!mlir.c.mlirTypeIsNull(mlir_type));
}

test "buildTypeFromValue - function type (i32, i32) -> i32" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const allocator = arena.allocator();

    var ctx = try mlir.Context.create();
    defer ctx.destroy();
    ctx.registerAllDialects();

    var builder = Builder.init(allocator, &ctx);
    defer builder.deinit();

    // Parse "!function (inputs i32 i32) (results i32)"  - note: no outer parens!
    const input = "(!function (inputs i32 i32) (results i32))";
    var tok = Tokenizer.init(allocator, input);
    var r = try Reader.init(allocator, &tok);
    const value = try r.read();

    // The value should be a function_type
    try testing.expect(value.type == .function_type);

    // Build the type
    const mlir_type = try builder.buildTypeFromValue(value);

    // Verify it's a function type
    try testing.expect(!mlir.c.mlirTypeIsNull(mlir_type));
    try testing.expect(mlir.c.mlirTypeIsAFunction(mlir_type));

    // Verify the function type has 2 inputs and 1 result
    try testing.expectEqual(@as(isize, 2), mlir.c.mlirFunctionTypeGetNumInputs(mlir_type));
    try testing.expectEqual(@as(isize, 1), mlir.c.mlirFunctionTypeGetNumResults(mlir_type));
}

test "buildTypeFromValue - function type (i32) -> i32" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const allocator = arena.allocator();

    var ctx = try mlir.Context.create();
    defer ctx.destroy();
    ctx.registerAllDialects();

    var builder = Builder.init(allocator, &ctx);
    defer builder.deinit();

    // Parse "(!function (inputs i32) (results i32))"
    const input = "(!function (inputs i32) (results i32))";
    var tok = Tokenizer.init(allocator, input);
    var r = try Reader.init(allocator, &tok);
    const value = try r.read();

    // Build the type
    const mlir_type = try builder.buildTypeFromValue(value);

    // Verify it's a function type with 1 input and 1 result
    try testing.expect(mlir.c.mlirTypeIsAFunction(mlir_type));
    try testing.expectEqual(@as(isize, 1), mlir.c.mlirFunctionTypeGetNumInputs(mlir_type));
    try testing.expectEqual(@as(isize, 1), mlir.c.mlirFunctionTypeGetNumResults(mlir_type));
}

test "buildTypeAttribute - wraps type in TypeAttr" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const allocator = arena.allocator();

    var ctx = try mlir.Context.create();
    defer ctx.destroy();
    ctx.registerAllDialects();

    var builder = Builder.init(allocator, &ctx);
    defer builder.deinit();

    // Parse "(!function (inputs i32 i32) (results i32))"
    const input = "(!function (inputs i32 i32) (results i32))";
    var tok = Tokenizer.init(allocator, input);
    var r = try Reader.init(allocator, &tok);
    const value = try r.read();

    // Create an AttrExpr manually for testing
    const attr_expr = AttrExpr{ .value = value };

    // Build the TypeAttr
    const mlir_attr = try builder.buildTypeAttribute(&attr_expr);

    // Verify it's a TypeAttr
    try testing.expect(!mlir.c.mlirAttributeIsNull(mlir_attr));
    try testing.expect(mlir.c.mlirAttributeIsAType(mlir_attr));
}
