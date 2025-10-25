const std = @import("std");
const tokenizer = @import("mlir_lisp");
const Tokenizer = tokenizer.Tokenizer;
const Token = tokenizer.Token;
const TokenType = tokenizer.TokenType;

test "tokenize delimiters" {
    const source = "( ) [ ] { }";
    var t = Tokenizer.init(std.testing.allocator, source);

    const expected = [_]TokenType{
        .left_paren,
        .right_paren,
        .left_bracket,
        .right_bracket,
        .left_brace,
        .right_brace,
        .eof,
    };

    for (expected) |expected_type| {
        const token = try t.next();
        try std.testing.expectEqual(expected_type, token.type);
    }
}

test "tokenize identifiers" {
    const source = "hello world foo_bar test123";
    var t = Tokenizer.init(std.testing.allocator, source);

    const expected = [_][]const u8{ "hello", "world", "foo_bar", "test123" };

    for (expected) |expected_lexeme| {
        const token = try t.next();
        try std.testing.expectEqual(TokenType.identifier, token.type);
        try std.testing.expectEqualStrings(expected_lexeme, token.lexeme);
    }
}

test "tokenize numbers" {
    const source = "42 -10 3.14 0x1A 0b1010 2.5e10";
    var t = Tokenizer.init(std.testing.allocator, source);

    const expected = [_][]const u8{ "42", "-10", "3.14", "0x1A", "0b1010", "2.5e10" };

    for (expected) |expected_lexeme| {
        const token = try t.next();
        try std.testing.expectEqual(TokenType.number, token.type);
        try std.testing.expectEqualStrings(expected_lexeme, token.lexeme);
    }
}

test "tokenize strings" {
    const source = "\"hello\" \"world with spaces\" \"escaped\\\"quote\"";
    var t = Tokenizer.init(std.testing.allocator, source);

    const expected = [_][]const u8{
        "\"hello\"",
        "\"world with spaces\"",
        "\"escaped\\\"quote\"",
    };

    for (expected) |expected_lexeme| {
        const token = try t.next();
        try std.testing.expectEqual(TokenType.string, token.type);
        try std.testing.expectEqualStrings(expected_lexeme, token.lexeme);
    }
}

test "tokenize value IDs" {
    const source = "%c0 %x %result";
    var t = Tokenizer.init(std.testing.allocator, source);

    const expected = [_][]const u8{ "%c0", "%x", "%result" };

    for (expected) |expected_lexeme| {
        const token = try t.next();
        try std.testing.expectEqual(TokenType.value_id, token.type);
        try std.testing.expectEqualStrings(expected_lexeme, token.lexeme);
    }
}

test "tokenize block IDs" {
    const source = "^entry ^then ^else";
    var t = Tokenizer.init(std.testing.allocator, source);

    const expected = [_][]const u8{ "^entry", "^then", "^else" };

    for (expected) |expected_lexeme| {
        const token = try t.next();
        try std.testing.expectEqual(TokenType.block_id, token.type);
        try std.testing.expectEqualStrings(expected_lexeme, token.lexeme);
    }
}

test "tokenize symbols" {
    const source = "@add @main @branchy";
    var t = Tokenizer.init(std.testing.allocator, source);

    const expected = [_][]const u8{ "@add", "@main", "@branchy" };

    for (expected) |expected_lexeme| {
        const token = try t.next();
        try std.testing.expectEqual(TokenType.symbol, token.type);
        try std.testing.expectEqualStrings(expected_lexeme, token.lexeme);
    }
}

test "tokenize keywords" {
    const source = ":value :sym :type :visibility";
    var t = Tokenizer.init(std.testing.allocator, source);

    const expected = [_][]const u8{ ":value", ":sym", ":type", ":visibility" };

    for (expected) |expected_lexeme| {
        const token = try t.next();
        try std.testing.expectEqual(TokenType.keyword, token.type);
        try std.testing.expectEqualStrings(expected_lexeme, token.lexeme);
    }
}

test "tokenize type and attr markers" {
    const source = "!i32 #int";
    var t = Tokenizer.init(std.testing.allocator, source);

    var token = try t.next();
    try std.testing.expectEqual(TokenType.type_marker, token.type);

    token = try t.next();
    try std.testing.expectEqual(TokenType.identifier, token.type);
    try std.testing.expectEqualStrings("i32", token.lexeme);

    token = try t.next();
    try std.testing.expectEqual(TokenType.attr_marker, token.type);

    token = try t.next();
    try std.testing.expectEqual(TokenType.identifier, token.type);
    try std.testing.expectEqualStrings("int", token.lexeme);
}

test "tokenize booleans" {
    const source = "true false";
    var t = Tokenizer.init(std.testing.allocator, source);

    var token = try t.next();
    try std.testing.expectEqual(TokenType.true_lit, token.type);

    token = try t.next();
    try std.testing.expectEqual(TokenType.false_lit, token.type);
}

test "tokenize operation names with dots" {
    const source = "arith.constant func.func cf.cond_br";
    var t = Tokenizer.init(std.testing.allocator, source);

    const expected = [_][]const u8{ "arith.constant", "func.func", "cf.cond_br" };

    for (expected) |expected_lexeme| {
        const token = try t.next();
        try std.testing.expectEqual(TokenType.identifier, token.type);
        try std.testing.expectEqualStrings(expected_lexeme, token.lexeme);
    }
}

test "skip whitespace and comments" {
    const source =
        \\; This is a comment
        \\(hello
        \\  world) ; inline comment
        \\%result
    ;
    var t = Tokenizer.init(std.testing.allocator, source);

    var token = try t.next();
    try std.testing.expectEqual(TokenType.left_paren, token.type);

    token = try t.next();
    try std.testing.expectEqual(TokenType.identifier, token.type);
    try std.testing.expectEqualStrings("hello", token.lexeme);

    token = try t.next();
    try std.testing.expectEqual(TokenType.identifier, token.type);
    try std.testing.expectEqualStrings("world", token.lexeme);

    token = try t.next();
    try std.testing.expectEqual(TokenType.right_paren, token.type);

    token = try t.next();
    try std.testing.expectEqual(TokenType.value_id, token.type);
    try std.testing.expectEqualStrings("%result", token.lexeme);
}

test "tokenize simple operation from grammar" {
    const source =
        \\(operation
        \\  (name arith.constant)
        \\  (result-bindings [%c0])
        \\  (result-types !i32)
        \\  (attributes { :value (#int 42) }))
    ;
    var t = Tokenizer.init(std.testing.allocator, source);

    const expected_tokens = [_]struct { type: TokenType, lexeme: []const u8 }{
        .{ .type = .left_paren, .lexeme = "(" },
        .{ .type = .identifier, .lexeme = "operation" },
        .{ .type = .left_paren, .lexeme = "(" },
        .{ .type = .identifier, .lexeme = "name" },
        .{ .type = .identifier, .lexeme = "arith.constant" },
        .{ .type = .right_paren, .lexeme = ")" },
        .{ .type = .left_paren, .lexeme = "(" },
        .{ .type = .identifier, .lexeme = "result-bindings" },
        .{ .type = .left_bracket, .lexeme = "[" },
        .{ .type = .value_id, .lexeme = "%c0" },
        .{ .type = .right_bracket, .lexeme = "]" },
        .{ .type = .right_paren, .lexeme = ")" },
        .{ .type = .left_paren, .lexeme = "(" },
        .{ .type = .identifier, .lexeme = "result-types" },
        .{ .type = .type_marker, .lexeme = "!" },
        .{ .type = .identifier, .lexeme = "i32" },
        .{ .type = .right_paren, .lexeme = ")" },
        .{ .type = .left_paren, .lexeme = "(" },
        .{ .type = .identifier, .lexeme = "attributes" },
        .{ .type = .left_brace, .lexeme = "{" },
        .{ .type = .keyword, .lexeme = ":value" },
        .{ .type = .left_paren, .lexeme = "(" },
        .{ .type = .attr_marker, .lexeme = "#" },
        .{ .type = .identifier, .lexeme = "int" },
        .{ .type = .number, .lexeme = "42" },
        .{ .type = .right_paren, .lexeme = ")" },
        .{ .type = .right_brace, .lexeme = "}" },
        .{ .type = .right_paren, .lexeme = ")" },
        .{ .type = .right_paren, .lexeme = ")" },
        .{ .type = .eof, .lexeme = "" },
    };

    for (expected_tokens) |expected| {
        const token = try t.next();
        try std.testing.expectEqual(expected.type, token.type);
        if (expected.lexeme.len > 0) {
            try std.testing.expectEqualStrings(expected.lexeme, token.lexeme);
        }
    }
}

test "tokenize block example from grammar" {
    const source =
        \\(block [^entry]
        \\  (arguments [ [%x !i32] [%y !i32] ]))
    ;
    var t = Tokenizer.init(std.testing.allocator, source);

    var token = try t.next();
    try std.testing.expectEqual(TokenType.left_paren, token.type);

    token = try t.next();
    try std.testing.expectEqual(TokenType.identifier, token.type);
    try std.testing.expectEqualStrings("block", token.lexeme);

    token = try t.next();
    try std.testing.expectEqual(TokenType.left_bracket, token.type);

    token = try t.next();
    try std.testing.expectEqual(TokenType.block_id, token.type);
    try std.testing.expectEqualStrings("^entry", token.lexeme);

    token = try t.next();
    try std.testing.expectEqual(TokenType.right_bracket, token.type);
}

test "line and column tracking" {
    const source =
        \\(hello
        \\  world)
    ;
    var t = Tokenizer.init(std.testing.allocator, source);

    var token = try t.next();
    try std.testing.expectEqual(@as(usize, 1), token.line);

    token = try t.next();
    try std.testing.expectEqual(@as(usize, 1), token.line);

    token = try t.next();
    try std.testing.expectEqual(@as(usize, 2), token.line);
}

test "error on unterminated string" {
    const source = "\"hello";
    var t = Tokenizer.init(std.testing.allocator, source);

    const result = t.next();
    try std.testing.expectError(error.UnterminatedString, result);
}

test "error on invalid value ID" {
    const source = "% ";
    var t = Tokenizer.init(std.testing.allocator, source);

    const result = t.next();
    try std.testing.expectError(error.InvalidValueId, result);
}

test "error on invalid block ID" {
    const source = "^ ";
    var t = Tokenizer.init(std.testing.allocator, source);

    const result = t.next();
    try std.testing.expectError(error.InvalidBlockId, result);
}

test "error on invalid symbol" {
    const source = "@ ";
    var t = Tokenizer.init(std.testing.allocator, source);

    const result = t.next();
    try std.testing.expectError(error.InvalidSymbol, result);
}

test "error on invalid keyword" {
    const source = ": ";
    var t = Tokenizer.init(std.testing.allocator, source);

    const result = t.next();
    try std.testing.expectError(error.InvalidKeyword, result);
}
