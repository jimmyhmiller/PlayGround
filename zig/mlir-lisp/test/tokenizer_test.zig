const std = @import("std");
const tokenizer = @import("mlir_lisp");
const Tokenizer = tokenizer.Tokenizer;
const Token = tokenizer.Token;
const TokenType = tokenizer.TokenType;

test "tokenize delimiters" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
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
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
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
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
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
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
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
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
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
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
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
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
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
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
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
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const source = "i32 #int";
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
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const source = "true false";
    var t = Tokenizer.init(std.testing.allocator, source);

    var token = try t.next();
    try std.testing.expectEqual(TokenType.true_lit, token.type);

    token = try t.next();
    try std.testing.expectEqual(TokenType.false_lit, token.type);
}

test "tokenize operation names with dots" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
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
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
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
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const source =
        \\(operation
        \\  (name arith.constant)
        \\  (result-bindings [%c0])
        \\  (result-types i32)
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
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const source =
        \\(block [^entry]
        \\  (arguments [ [%x i32] [%y i32] ]))
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
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
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
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const source = "\"hello";
    var t = Tokenizer.init(std.testing.allocator, source);

    const result = t.next();
    try std.testing.expectError(error.UnterminatedString, result);
}

test "error on invalid value ID" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const source = "% ";
    var t = Tokenizer.init(std.testing.allocator, source);

    const result = t.next();
    try std.testing.expectError(error.InvalidValueId, result);
}

test "error on invalid block ID" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const source = "^ ";
    var t = Tokenizer.init(std.testing.allocator, source);

    const result = t.next();
    try std.testing.expectError(error.InvalidBlockId, result);
}

test "error on invalid symbol" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const source = "@ ";
    var t = Tokenizer.init(std.testing.allocator, source);

    const result = t.next();
    try std.testing.expectError(error.InvalidSymbol, result);
}

test "error on invalid keyword" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const source = ": ";
    var t = Tokenizer.init(std.testing.allocator, source);

    const result = t.next();
    try std.testing.expectError(error.InvalidKeyword, result);
}

test "tokenize simple attribute markers" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const source = "#llvm.linkage<internal> #llvm.framePointerKind<none>";
    var t = Tokenizer.init(std.testing.allocator, source);

    var token = try t.next();
    try std.testing.expectEqual(TokenType.attr_marker, token.type);
    try std.testing.expectEqualStrings("#llvm.linkage<internal>", token.lexeme);

    token = try t.next();
    try std.testing.expectEqual(TokenType.attr_marker, token.type);
    try std.testing.expectEqualStrings("#llvm.framePointerKind<none>", token.lexeme);
}

test "tokenize complex nested attribute with spaces" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const source = "#dlti.dl_spec<i1 = dense<8> : vector<2xi64>>";
    var t = Tokenizer.init(std.testing.allocator, source);

    const token = try t.next();
    try std.testing.expectEqual(TokenType.attr_marker, token.type);
    try std.testing.expectEqualStrings("#dlti.dl_spec<i1 = dense<8> : vector<2xi64>>", token.lexeme);
}

test "tokenize attribute with multiple nested brackets and spaces" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const source = "#dlti.dl_spec<i1 = dense<8> : vector<2xi64>, !llvm.ptr = dense<64> : vector<4xi64>>";
    var t = Tokenizer.init(std.testing.allocator, source);

    const token = try t.next();
    try std.testing.expectEqual(TokenType.attr_marker, token.type);
    try std.testing.expectEqualStrings("#dlti.dl_spec<i1 = dense<8> : vector<2xi64>, !llvm.ptr = dense<64> : vector<4xi64>>", token.lexeme);
}

test "tokenize attribute with string literals" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const source = "#attr<\"dlti.endianness\" = \"little\">";
    var t = Tokenizer.init(std.testing.allocator, source);

    const token = try t.next();
    try std.testing.expectEqual(TokenType.attr_marker, token.type);
    try std.testing.expectEqualStrings("#attr<\"dlti.endianness\" = \"little\">", token.lexeme);
}

test "tokenize attribute with escaped quotes in strings" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const source = "#attr<\"key\" = \"value with \\\"quotes\\\"\">";
    var t = Tokenizer.init(std.testing.allocator, source);

    const token = try t.next();
    try std.testing.expectEqual(TokenType.attr_marker, token.type);
    try std.testing.expectEqualStrings("#attr<\"key\" = \"value with \\\"quotes\\\"\">", token.lexeme);
}

test "tokenize attribute with parentheses" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const source = "#attr<func(i32, i64)>";
    var t = Tokenizer.init(std.testing.allocator, source);

    const token = try t.next();
    try std.testing.expectEqual(TokenType.attr_marker, token.type);
    try std.testing.expectEqualStrings("#attr<func(i32, i64)>", token.lexeme);
}

test "tokenize attribute stops at s-expr delimiters" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const source = "#attr<value> { :key #attr2<val> }";
    var t = Tokenizer.init(std.testing.allocator, source);

    var token = try t.next();
    try std.testing.expectEqual(TokenType.attr_marker, token.type);
    try std.testing.expectEqualStrings("#attr<value>", token.lexeme);

    token = try t.next();
    try std.testing.expectEqual(TokenType.left_brace, token.type);

    token = try t.next();
    try std.testing.expectEqual(TokenType.keyword, token.type);

    token = try t.next();
    try std.testing.expectEqual(TokenType.attr_marker, token.type);
    try std.testing.expectEqualStrings("#attr2<val>", token.lexeme);
}

test "tokenize deeply nested attribute" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const source = "#attr<outer<inner<deep<value>>>>";
    var t = Tokenizer.init(std.testing.allocator, source);

    const token = try t.next();
    try std.testing.expectEqual(TokenType.attr_marker, token.type);
    try std.testing.expectEqualStrings("#attr<outer<inner<deep<value>>>>", token.lexeme);
}

test "tokenize attribute with type annotations" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const source = "#attr<!llvm.ptr<272> = dense<64> : vector<4xi64>>";
    var t = Tokenizer.init(std.testing.allocator, source);

    const token = try t.next();
    try std.testing.expectEqual(TokenType.attr_marker, token.type);
    try std.testing.expectEqualStrings("#attr<!llvm.ptr<272> = dense<64> : vector<4xi64>>", token.lexeme);
}

test "tokenize real-world dlti.dl_spec attribute" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    // This is an abbreviated version of the actual dlti.dl_spec from c_api_transform.mlir-lisp
    const source = "#dlti.dl_spec<i1 = dense<8> : vector<2xi64>, i64 = dense<64> : vector<2xi64>, \"dlti.endianness\" = \"little\">";
    var t = Tokenizer.init(std.testing.allocator, source);

    const token = try t.next();
    try std.testing.expectEqual(TokenType.attr_marker, token.type);
    try std.testing.expectEqualStrings("#dlti.dl_spec<i1 = dense<8> : vector<2xi64>, i64 = dense<64> : vector<2xi64>, \"dlti.endianness\" = \"little\">", token.lexeme);
}

test "tokenize simple type markers" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const source = "!llvm.ptr !transform.any_op";
    var t = Tokenizer.init(std.testing.allocator, source);

    var token = try t.next();
    try std.testing.expectEqual(TokenType.type_marker, token.type);
    try std.testing.expectEqualStrings("!llvm.ptr", token.lexeme);

    token = try t.next();
    try std.testing.expectEqual(TokenType.type_marker, token.type);
    try std.testing.expectEqualStrings("!transform.any_op", token.lexeme);
}

test "tokenize complex type with spaces - array" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const source = "!llvm.array<10 x i8>";
    var t = Tokenizer.init(std.testing.allocator, source);

    const token = try t.next();
    try std.testing.expectEqual(TokenType.type_marker, token.type);
    try std.testing.expectEqualStrings("!llvm.array<10 x i8>", token.lexeme);
}

test "tokenize complex type with nested brackets" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const source = "!llvm.struct<(i32, array<10 x i8>)>";
    var t = Tokenizer.init(std.testing.allocator, source);

    const token = try t.next();
    try std.testing.expectEqual(TokenType.type_marker, token.type);
    try std.testing.expectEqualStrings("!llvm.struct<(i32, array<10 x i8>)>", token.lexeme);
}

test "tokenize type with pointer and nested angle brackets" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const source = "!llvm.ptr<272>";
    var t = Tokenizer.init(std.testing.allocator, source);

    const token = try t.next();
    try std.testing.expectEqual(TokenType.type_marker, token.type);
    try std.testing.expectEqualStrings("!llvm.ptr<272>", token.lexeme);
}

test "tokenize multiple complex types in sequence" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const source = "!llvm.array<10 x i8> !llvm.ptr<271>";
    var t = Tokenizer.init(std.testing.allocator, source);

    var token = try t.next();
    try std.testing.expectEqual(TokenType.type_marker, token.type);
    try std.testing.expectEqualStrings("!llvm.array<10 x i8>", token.lexeme);

    token = try t.next();
    try std.testing.expectEqual(TokenType.type_marker, token.type);
    try std.testing.expectEqualStrings("!llvm.ptr<271>", token.lexeme);
}

test "tokenize type stops at s-expr delimiters" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const source = "!llvm.array<10 x i8> }";
    var t = Tokenizer.init(std.testing.allocator, source);

    var token = try t.next();
    try std.testing.expectEqual(TokenType.type_marker, token.type);
    try std.testing.expectEqualStrings("!llvm.array<10 x i8>", token.lexeme);

    token = try t.next();
    try std.testing.expectEqual(TokenType.right_brace, token.type);
}

test "tokenize deeply nested type" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const source = "!llvm.struct<(ptr<struct<(i32, i64)>>)>";
    var t = Tokenizer.init(std.testing.allocator, source);

    const token = try t.next();
    try std.testing.expectEqual(TokenType.type_marker, token.type);
    try std.testing.expectEqualStrings("!llvm.struct<(ptr<struct<(i32, i64)>>)>", token.lexeme);
}
