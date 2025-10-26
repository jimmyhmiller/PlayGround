const std = @import("std");
const mlir_parser = @import("mlir_parser");

test "lexer - tokenize simple operation" {
    const source = "%0 = arith.constant 42 : i32";
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
