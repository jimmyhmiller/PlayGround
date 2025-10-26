const std = @import("std");
const mlir_parser = @import("mlir_parser");

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

test "integration - lex and parse types" {
    const source = "f64";
    var lex = mlir_parser.Lexer.init(source);
    var parser = try mlir_parser.Parser.init(std.testing.allocator, &lex);
    defer parser.deinit();

    var type_result = try parser.parseType();
    defer type_result.deinit(std.testing.allocator);

    try std.testing.expect(type_result == .builtin);
    try std.testing.expect(type_result.builtin == .float);
}

test "integration - multiple token types" {
    const source = "index i32 f64";
    var lex = mlir_parser.Lexer.init(source);

    // index
    const t1 = lex.nextToken();
    try std.testing.expectEqual(mlir_parser.TokenType.bare_id, t1.type);

    // i32
    const t2 = lex.nextToken();
    try std.testing.expectEqual(mlir_parser.TokenType.bare_id, t2.type);

    // f64
    const t3 = lex.nextToken();
    try std.testing.expectEqual(mlir_parser.TokenType.bare_id, t3.type);

    // eof
    const t4 = lex.nextToken();
    try std.testing.expectEqual(mlir_parser.TokenType.eof, t4.type);
}

test "integration - parse generic operation with properties" {
    const source = try loadTestFile(std.testing.allocator, "integration_generic_operation.mlir");
    defer std.testing.allocator.free(source);

    var lex = mlir_parser.Lexer.init(source);
    var parser = try mlir_parser.Parser.init(std.testing.allocator, &lex);
    defer parser.deinit();

    var op = try parser.parseOperation();
    defer op.deinit(std.testing.allocator);

    try std.testing.expect(op.results != null);
    try std.testing.expect(op.kind == .generic);
    try std.testing.expectEqualStrings("arith.constant", op.kind.generic.name);
    try std.testing.expect(op.kind.generic.properties != null);
}
