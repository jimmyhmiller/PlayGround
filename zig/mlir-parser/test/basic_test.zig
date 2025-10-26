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

test "lexer - tokenize simple operation" {
    const source = try loadTestFile(std.testing.allocator, "basic_lexer_test.mlir");
    defer std.testing.allocator.free(source);

    var lex = mlir_parser.Lexer.init(source);

    const token1 = lex.nextToken();
    try std.testing.expectEqual(mlir_parser.TokenType.value_id, token1.type);

    const token2 = lex.nextToken();
    try std.testing.expectEqual(mlir_parser.TokenType.equal, token2.type);

    const token3 = lex.nextToken();
    try std.testing.expectEqual(mlir_parser.TokenType.bare_id, token3.type);
}

test "parser - parse integer type" {
    const source = "i32";
    var lex = mlir_parser.Lexer.init(source);
    var parser = try mlir_parser.Parser.init(std.testing.allocator, &lex);
    defer parser.deinit();

    var type_result = try parser.parseType();
    defer type_result.deinit(std.testing.allocator);

    try std.testing.expect(type_result == .builtin);
    try std.testing.expect(type_result.builtin == .integer);
    try std.testing.expectEqual(@as(u64, 32), type_result.builtin.integer.width);
}
