const std = @import("std");
const mlir_parser = @import("mlir_parser");

// This file contains tests for MLIR features we don't support yet
// These tests are expected to FAIL until we implement the features

/// Helper to load test file from test_data/examples/
fn loadTestFile(allocator: std.mem.Allocator, filename: []const u8) ![]u8 {
    const dir = std.fs.cwd();
    const path = try std.fmt.allocPrint(allocator, "test_data/examples/{s}", .{filename});
    defer allocator.free(path);
    const file = try dir.openFile(path, .{});
    defer file.close();
    const content = try file.readToEndAlloc(allocator, 1024 * 1024);
    return content;
}

// Grammar: successor-list ::= `[` successor (`,` successor)* `]`
// Grammar: successor ::= caret-id (`:` block-arg-list)?
test "unsupported - operation with successors (branch)" {
    const source = try loadTestFile(std.testing.allocator, "branch_with_successors.mlir");
    defer std.testing.allocator.free(source);

    var lex = mlir_parser.Lexer.init(source);
    var parser = try mlir_parser.Parser.init(std.testing.allocator, &lex);
    defer parser.deinit();

    // Parse first three operations (these should work)
    var op1 = try parser.parseOperation();
    defer op1.deinit(std.testing.allocator);

    var op2 = try parser.parseOperation();
    defer op2.deinit(std.testing.allocator);

    var op3 = try parser.parseOperation();
    defer op3.deinit(std.testing.allocator);

    // This one should fail - we don't parse successor lists yet
    var op4 = try parser.parseOperation();
    defer op4.deinit(std.testing.allocator);

    try std.testing.expectEqualStrings("cf.cond_br", op4.kind.generic.name);
    try std.testing.expectEqual(@as(usize, 2), op4.kind.generic.successors.len);
}

// Grammar: region-list ::= `(` region (`,` region)* `)`
// Grammar: region ::= `{` entry-block? block* `}`
test "unsupported - operation with regions (scf.if)" {
    const source = try loadTestFile(std.testing.allocator, "scf_if_regions.mlir");
    defer std.testing.allocator.free(source);

    var lex = mlir_parser.Lexer.init(source);
    var parser = try mlir_parser.Parser.init(std.testing.allocator, &lex);
    defer parser.deinit();

    var op1 = try parser.parseOperation();
    defer op1.deinit(std.testing.allocator);

    // This should fail - we don't parse regions yet
    var op2 = try parser.parseOperation();
    defer op2.deinit(std.testing.allocator);

    try std.testing.expectEqualStrings("scf.if", op2.kind.generic.name);
    try std.testing.expectEqual(@as(usize, 2), op2.kind.generic.regions.len);
}

// Grammar: block ::= block-label operation+
// Grammar: block-label ::= block-id block-arg-list? `:`
// Grammar: block-id ::= caret-id
test "unsupported - basic blocks with labels" {
    // Skip this test - we don't have a parseBlock function yet
    return error.SkipZigTest;
}

// Grammar: builtin-type - tensor types
test "unsupported - tensor types" {
    const source = "tensor<4x8xf32>";

    var lex = mlir_parser.Lexer.init(source);
    var parser = try mlir_parser.Parser.init(std.testing.allocator, &lex);
    defer parser.deinit();

    // This should fail - we don't parse tensor types yet
    var type_result = try parser.parseType();
    defer type_result.deinit(std.testing.allocator);

    try std.testing.expect(type_result == .builtin);
    try std.testing.expect(type_result.builtin == .tensor);
}

// Grammar: builtin-type - memref types
test "unsupported - memref types" {
    const source = "memref<16x16xf64>";

    var lex = mlir_parser.Lexer.init(source);
    var parser = try mlir_parser.Parser.init(std.testing.allocator, &lex);
    defer parser.deinit();

    // This should fail - we don't parse memref types yet
    var type_result = try parser.parseType();
    defer type_result.deinit(std.testing.allocator);

    try std.testing.expect(type_result == .builtin);
    try std.testing.expect(type_result.builtin == .memref);
}

// Grammar: builtin-type - vector types
test "unsupported - vector types" {
    const source = "vector<4xf32>";

    var lex = mlir_parser.Lexer.init(source);
    var parser = try mlir_parser.Parser.init(std.testing.allocator, &lex);
    defer parser.deinit();

    // This should fail - we don't parse vector types yet
    var type_result = try parser.parseType();
    defer type_result.deinit(std.testing.allocator);

    try std.testing.expect(type_result == .builtin);
    try std.testing.expect(type_result.builtin == .vector);
}

// Grammar: builtin-type - complex types
test "unsupported - complex types" {
    const source = "complex<f32>";

    var lex = mlir_parser.Lexer.init(source);
    var parser = try mlir_parser.Parser.init(std.testing.allocator, &lex);
    defer parser.deinit();

    // This should fail - we don't parse complex types yet
    var type_result = try parser.parseType();
    defer type_result.deinit(std.testing.allocator);

    try std.testing.expect(type_result == .builtin);
    try std.testing.expect(type_result.builtin == .complex);
}

// Grammar: builtin-type - tuple types
test "unsupported - tuple types" {
    const source = "tuple<i32, f64, index>";

    var lex = mlir_parser.Lexer.init(source);
    var parser = try mlir_parser.Parser.init(std.testing.allocator, &lex);
    defer parser.deinit();

    // This should fail - we don't parse tuple types yet
    var type_result = try parser.parseType();
    defer type_result.deinit(std.testing.allocator);

    try std.testing.expect(type_result == .builtin);
    try std.testing.expect(type_result.builtin == .tuple);
}

// Test operation with multiple results (op-result with count)
// Grammar: op-result ::= value-id (`:` integer-literal)?
test "unsupported - operation with multiple results count" {
    const source = try loadTestFile(std.testing.allocator, "multi_result_count.mlir");
    defer std.testing.allocator.free(source);

    var lex = mlir_parser.Lexer.init(source);
    var parser = try mlir_parser.Parser.init(std.testing.allocator, &lex);
    defer parser.deinit();

    var op = try parser.parseOperation();
    defer op.deinit(std.testing.allocator);

    try std.testing.expect(op.results != null);
    try std.testing.expectEqual(@as(usize, 1), op.results.?.results.len);
    try std.testing.expect(op.results.?.results[0].num_results != null);
    try std.testing.expectEqual(@as(u64, 2), op.results.?.results[0].num_results.?);
}

// Test value-use with result number
// Grammar: value-use ::= value-id (`#` decimal-literal)?
test "unsupported - value use with result number" {
    const source = try loadTestFile(std.testing.allocator, "value_use_result_number.mlir");
    defer std.testing.allocator.free(source);

    var lex = mlir_parser.Lexer.init(source);
    var parser = try mlir_parser.Parser.init(std.testing.allocator, &lex);
    defer parser.deinit();

    var op1 = try parser.parseOperation();
    defer op1.deinit(std.testing.allocator);

    var op2 = try parser.parseOperation();
    defer op2.deinit(std.testing.allocator);

    try std.testing.expectEqual(@as(usize, 2), op2.kind.generic.operands.len);
    try std.testing.expect(op2.kind.generic.operands[0].result_number != null);
    try std.testing.expectEqual(@as(u64, 0), op2.kind.generic.operands[0].result_number.?);
    try std.testing.expect(op2.kind.generic.operands[1].result_number != null);
    try std.testing.expectEqual(@as(u64, 1), op2.kind.generic.operands[1].result_number.?);
}

// Test trailing location
// Grammar: trailing-location ::= `loc` `(` location `)`
test "unsupported - trailing location" {
    const source = try loadTestFile(std.testing.allocator, "trailing_location.mlir");
    defer std.testing.allocator.free(source);

    var lex = mlir_parser.Lexer.init(source);
    var parser = try mlir_parser.Parser.init(std.testing.allocator, &lex);
    defer parser.deinit();

    var op = try parser.parseOperation();
    defer op.deinit(std.testing.allocator);

    try std.testing.expect(op.location != null);
}

// Test dialect attribute (pretty format)
// Grammar: pretty-dialect-attribute ::= dialect-namespace `.` pretty-dialect-attribute-lead-ident
test "unsupported - pretty dialect attribute" {
    const source = try loadTestFile(std.testing.allocator, "pretty_dialect_attribute.mlir");
    defer std.testing.allocator.free(source);

    var lex = mlir_parser.Lexer.init(source);
    var parser = try mlir_parser.Parser.init(std.testing.allocator, &lex);
    defer parser.deinit();

    var op = try parser.parseOperation();
    defer op.deinit(std.testing.allocator);

    try std.testing.expect(op.kind.generic.attributes != null);
}

// Test dialect type (pretty format)
// Grammar: pretty-dialect-type ::= dialect-namespace `.` pretty-dialect-type-lead-ident
test "unsupported - pretty dialect type" {
    const source = "!llvm.ptr";

    var lex = mlir_parser.Lexer.init(source);
    var parser = try mlir_parser.Parser.init(std.testing.allocator, &lex);
    defer parser.deinit();

    var type_result = try parser.parseType();
    defer type_result.deinit(std.testing.allocator);

    try std.testing.expect(type_result == .dialect);
}

// Test type alias usage
// Grammar: type-alias ::= `!` alias-name
test "unsupported - type alias usage" {
    const source = try loadTestFile(std.testing.allocator, "type_alias_usage.mlir");
    defer std.testing.allocator.free(source);

    var lex = mlir_parser.Lexer.init(source);
    var parser = try mlir_parser.Parser.init(std.testing.allocator, &lex);
    defer parser.deinit();

    var module = try parser.parseModule();
    defer module.deinit();

    try std.testing.expectEqual(@as(usize, 1), module.type_aliases.len);
    try std.testing.expectEqual(@as(usize, 1), module.operations.len);
}

// Test attribute alias usage
// Grammar: attribute-alias ::= `#` alias-name
test "unsupported - attribute alias usage" {
    const source = try loadTestFile(std.testing.allocator, "attribute_alias_usage.mlir");
    defer std.testing.allocator.free(source);

    var lex = mlir_parser.Lexer.init(source);
    var parser = try mlir_parser.Parser.init(std.testing.allocator, &lex);
    defer parser.deinit();

    var module = try parser.parseModule();
    defer module.deinit();

    try std.testing.expectEqual(@as(usize, 1), module.attribute_aliases.len);
    try std.testing.expectEqual(@as(usize, 1), module.operations.len);
}
