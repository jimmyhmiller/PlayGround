const std = @import("std");

pub const TokenType = enum {
    LeftParen, // (
    RightParen, // )
    LeftBracket, // [
    RightBracket, // ]
    LeftBrace, // {
    RightBrace, // }
    Symbol, // any identifier
    String, // "..."
    Number, // 123, 3.14
    Keyword, // :keyword
    BlockLabel, // ^label
    Colon, // : (for type annotations)
    EOF,
};

pub const Token = struct {
    type: TokenType,
    lexeme: []const u8,
    line: usize,
    column: usize,

    pub fn format(self: Token, comptime _: []const u8, _: std.fmt.FormatOptions, writer: anytype) !void {
        try writer.print("Token({s}, \"{s}\", {}:{})", .{ @tagName(self.type), self.lexeme, self.line, self.column });
    }
};

pub const Tokenizer = struct {
    source: []const u8,
    start: usize,
    current: usize,
    line: usize,
    column: usize,
    allocator: std.mem.Allocator,

    pub fn init(allocator: std.mem.Allocator, source: []const u8) Tokenizer {
        return .{
            .source = source,
            .start = 0,
            .current = 0,
            .line = 1,
            .column = 1,
            .allocator = allocator,
        };
    }

    pub fn tokenize(self: *Tokenizer) !std.ArrayList(Token) {
        var tokens = std.ArrayList(Token){};
        errdefer tokens.deinit(self.allocator);

        while (!self.isAtEnd()) {
            self.start = self.current;
            const token = try self.scanToken();
            if (token) |t| {
                try tokens.append(self.allocator, t);
            }
        }

        try tokens.append(self.allocator, .{
            .type = .EOF,
            .lexeme = "",
            .line = self.line,
            .column = self.column,
        });

        return tokens;
    }

    fn scanToken(self: *Tokenizer) !?Token {
        const c = self.advance();

        return switch (c) {
            '(' => self.makeToken(.LeftParen),
            ')' => self.makeToken(.RightParen),
            '[' => self.makeToken(.LeftBracket),
            ']' => self.makeToken(.RightBracket),
            '{' => self.makeToken(.LeftBrace),
            '}' => self.makeToken(.RightBrace),
            ':' => {
                // Could be a keyword or a type annotation colon
                if (self.isAlpha(self.peek()) or self.peek() == '_' or self.peek() == '-') {
                    return try self.keyword();
                }
                return self.makeToken(.Colon);
            },
            '^' => try self.blockLabel(),
            '"' => try self.string(),
            ' ', '\r', '\t', '\n' => {
                // Skip whitespace
                if (c == '\n') {
                    self.line += 1;
                    self.column = 1;
                }
                return null;
            },
            ';' => {
                // Comment - skip until end of line
                while (self.peek() != '\n' and !self.isAtEnd()) {
                    _ = self.advance();
                }
                return null;
            },
            else => {
                if (self.isDigit(c) or (c == '-' and self.isDigit(self.peek()))) {
                    return try self.number();
                }
                if (self.isSymbolStart(c)) {
                    return try self.symbol();
                }
                return error.UnexpectedCharacter;
            },
        };
    }

    fn symbol(self: *Tokenizer) !Token {
        // Symbols can contain letters, digits, and special chars: - _ . / < > ! ? *
        // They can also contain angle brackets for types like memref<128x128xf32>
        var depth: i32 = 0;

        while (!self.isAtEnd()) {
            const c = self.peek();

            // Track angle bracket depth for types
            if (c == '<') {
                depth += 1;
                _ = self.advance();
                continue;
            }
            if (c == '>') {
                if (depth > 0) {
                    depth -= 1;
                    _ = self.advance();
                    continue;
                }
                // Allow > as part of normal symbols (e.g. ->)
            }

            // Inside angle brackets, allow more characters
            if (depth > 0) {
                if (c == ' ' or c == ',' or c == '(' or c == ')' or c == '[' or c == ']' or
                    c == ':' or c == '=' or c == '*' or c == '+' or c == '|' or c == '-' or
                    self.isAlphaNumeric(c) or c == '_' or c == '#' or c == '.') {
                    _ = self.advance();
                    continue;
                }
                break;
            }

            // Normal symbol characters
            if (self.isSymbolChar(c)) {
                _ = self.advance();
            } else {
                break;
            }
        }

        return self.makeToken(.Symbol);
    }

    fn number(self: *Tokenizer) !Token {
        // Handle negative numbers
        if (self.source[self.current - 1] == '-') {
            if (!self.isDigit(self.peek())) {
                // Just a '-' symbol, back up
                self.current -= 1;
                return self.symbol();
            }
        }

        while (self.isDigit(self.peek())) {
            _ = self.advance();
        }

        // Look for decimal point
        if (self.peek() == '.' and self.isDigit(self.peekNext())) {
            _ = self.advance(); // consume '.'
            while (self.isDigit(self.peek())) {
                _ = self.advance();
            }
        }

        // Scientific notation
        if (self.peek() == 'e' or self.peek() == 'E') {
            _ = self.advance();
            if (self.peek() == '+' or self.peek() == '-') {
                _ = self.advance();
            }
            while (self.isDigit(self.peek())) {
                _ = self.advance();
            }
        }

        return self.makeToken(.Number);
    }

    fn string(self: *Tokenizer) !Token {
        while (self.peek() != '"' and !self.isAtEnd()) {
            if (self.peek() == '\n') {
                self.line += 1;
                self.column = 1;
            }
            if (self.peek() == '\\') {
                _ = self.advance(); // escape char
                if (!self.isAtEnd()) {
                    _ = self.advance(); // escaped char
                }
            } else {
                _ = self.advance();
            }
        }

        if (self.isAtEnd()) {
            return error.UnterminatedString;
        }

        _ = self.advance(); // closing "

        return self.makeToken(.String);
    }

    fn keyword(self: *Tokenizer) !Token {
        while (self.isAlphaNumeric(self.peek()) or self.peek() == '-' or self.peek() == '_') {
            _ = self.advance();
        }
        return self.makeToken(.Keyword);
    }

    fn blockLabel(self: *Tokenizer) !Token {
        while (self.isAlphaNumeric(self.peek()) or self.peek() == '_') {
            _ = self.advance();
        }
        return self.makeToken(.BlockLabel);
    }

    fn makeToken(self: *Tokenizer, token_type: TokenType) Token {
        return .{
            .type = token_type,
            .lexeme = self.source[self.start..self.current],
            .line = self.line,
            .column = self.column,
        };
    }

    fn isAtEnd(self: *Tokenizer) bool {
        return self.current >= self.source.len;
    }

    fn advance(self: *Tokenizer) u8 {
        const c = self.source[self.current];
        self.current += 1;
        self.column += 1;
        return c;
    }

    fn peek(self: *Tokenizer) u8 {
        if (self.isAtEnd()) return 0;
        return self.source[self.current];
    }

    fn peekNext(self: *Tokenizer) u8 {
        if (self.current + 1 >= self.source.len) return 0;
        return self.source[self.current + 1];
    }

    fn isDigit(self: *Tokenizer, c: u8) bool {
        _ = self;
        return c >= '0' and c <= '9';
    }

    fn isAlpha(self: *Tokenizer, c: u8) bool {
        _ = self;
        return (c >= 'a' and c <= 'z') or
            (c >= 'A' and c <= 'Z') or
            c == '_';
    }

    fn isAlphaNumeric(self: *Tokenizer, c: u8) bool {
        return self.isAlpha(c) or self.isDigit(c);
    }

    fn isSymbolStart(self: *Tokenizer, c: u8) bool {
        _ = self;
        return (c >= 'a' and c <= 'z') or
            (c >= 'A' and c <= 'Z') or
            c == '_' or c == '-' or c == '!' or c == '?' or c == '*' or
            c == '+' or c == '=' or c == '<' or c == '>';
    }

    fn isSymbolChar(self: *Tokenizer, c: u8) bool {
        return self.isAlphaNumeric(c) or
            c == '-' or c == '_' or c == '.' or c == '/' or
            c == '!' or c == '?' or c == '*' or c == '+' or c == '=' or
            c == '<' or c == '>';
    }
};

test "tokenizer basic tokens" {
    const allocator = std.testing.allocator;

    var tokenizer = Tokenizer.init(allocator, "() [] {}");
    var tokens = try tokenizer.tokenize();
    defer tokens.deinit(allocator);

    try std.testing.expectEqual(@as(usize, 7), tokens.items.len); // 6 + EOF
    try std.testing.expectEqual(TokenType.LeftParen, tokens.items[0].type);
    try std.testing.expectEqual(TokenType.RightParen, tokens.items[1].type);
    try std.testing.expectEqual(TokenType.LeftBracket, tokens.items[2].type);
    try std.testing.expectEqual(TokenType.RightBracket, tokens.items[3].type);
    try std.testing.expectEqual(TokenType.LeftBrace, tokens.items[4].type);
    try std.testing.expectEqual(TokenType.RightBrace, tokens.items[5].type);
}

test "tokenizer symbols" {
    const allocator = std.testing.allocator;

    var tokenizer = Tokenizer.init(allocator, "arith.addi a/b foo-bar");
    var tokens = try tokenizer.tokenize();
    defer tokens.deinit(allocator);

    try std.testing.expectEqual(@as(usize, 4), tokens.items.len);
    try std.testing.expectEqual(TokenType.Symbol, tokens.items[0].type);
    try std.testing.expectEqualStrings("arith.addi", tokens.items[0].lexeme);
    try std.testing.expectEqualStrings("a/b", tokens.items[1].lexeme);
    try std.testing.expectEqualStrings("foo-bar", tokens.items[2].lexeme);
}

test "tokenizer numbers" {
    const allocator = std.testing.allocator;

    var tokenizer = Tokenizer.init(allocator, "42 3.14 -10 1.5e-3");
    var tokens = try tokenizer.tokenize();
    defer tokens.deinit(allocator);

    try std.testing.expectEqual(@as(usize, 5), tokens.items.len);
    try std.testing.expectEqual(TokenType.Number, tokens.items[0].type);
    try std.testing.expectEqualStrings("42", tokens.items[0].lexeme);
    try std.testing.expectEqualStrings("3.14", tokens.items[1].lexeme);
    try std.testing.expectEqualStrings("-10", tokens.items[2].lexeme);
    try std.testing.expectEqualStrings("1.5e-3", tokens.items[3].lexeme);
}

test "tokenizer strings" {
    const allocator = std.testing.allocator;

    var tokenizer = Tokenizer.init(allocator, "\"hello\" \"world\\n\"");
    var tokens = try tokenizer.tokenize();
    defer tokens.deinit(allocator);

    try std.testing.expectEqual(@as(usize, 3), tokens.items.len);
    try std.testing.expectEqual(TokenType.String, tokens.items[0].type);
    try std.testing.expectEqualStrings("\"hello\"", tokens.items[0].lexeme);
    try std.testing.expectEqualStrings("\"world\\n\"", tokens.items[1].lexeme);
}

test "tokenizer keywords" {
    const allocator = std.testing.allocator;

    var tokenizer = Tokenizer.init(allocator, ":foo :bar-baz");
    var tokens = try tokenizer.tokenize();
    defer tokens.deinit(allocator);

    try std.testing.expectEqual(@as(usize, 3), tokens.items.len);
    try std.testing.expectEqual(TokenType.Keyword, tokens.items[0].type);
    try std.testing.expectEqualStrings(":foo", tokens.items[0].lexeme);
    try std.testing.expectEqualStrings(":bar-baz", tokens.items[1].lexeme);
}

test "tokenizer block labels" {
    const allocator = std.testing.allocator;

    var tokenizer = Tokenizer.init(allocator, "^bb1 ^loop");
    var tokens = try tokenizer.tokenize();
    defer tokens.deinit(allocator);

    try std.testing.expectEqual(@as(usize, 3), tokens.items.len);
    try std.testing.expectEqual(TokenType.BlockLabel, tokens.items[0].type);
    try std.testing.expectEqualStrings("^bb1", tokens.items[0].lexeme);
    try std.testing.expectEqualStrings("^loop", tokens.items[1].lexeme);
}

test "tokenizer types with angle brackets" {
    const allocator = std.testing.allocator;

    var tokenizer = Tokenizer.init(allocator, "memref<128x128xf32> !llvm.ptr<i8>");
    var tokens = try tokenizer.tokenize();
    defer tokens.deinit(allocator);

    try std.testing.expectEqual(@as(usize, 3), tokens.items.len);
    try std.testing.expectEqualStrings("memref<128x128xf32>", tokens.items[0].lexeme);
    try std.testing.expectEqualStrings("!llvm.ptr<i8>", tokens.items[1].lexeme);
}

test "tokenizer comments" {
    const allocator = std.testing.allocator;

    var tokenizer = Tokenizer.init(allocator,
        \\; This is a comment
        \\(def x 42)
    );
    var tokens = try tokenizer.tokenize();
    defer tokens.deinit(allocator);

    try std.testing.expectEqual(@as(usize, 6), tokens.items.len); // +1 for EOF
    try std.testing.expectEqual(TokenType.LeftParen, tokens.items[0].type);
    try std.testing.expectEqualStrings("def", tokens.items[1].lexeme);
    try std.testing.expectEqual(TokenType.EOF, tokens.items[5].type);
}

test "tokenizer complex expression" {
    const allocator = std.testing.allocator;

    var tokenizer = Tokenizer.init(allocator, "(arith.addi {:value 42} x y)");
    var tokens = try tokenizer.tokenize();
    defer tokens.deinit(allocator);

    try std.testing.expectEqual(@as(usize, 10), tokens.items.len); // +1 for EOF
    try std.testing.expectEqual(TokenType.LeftParen, tokens.items[0].type);
    try std.testing.expectEqualStrings("arith.addi", tokens.items[1].lexeme);
    try std.testing.expectEqual(TokenType.LeftBrace, tokens.items[2].type);
    try std.testing.expectEqualStrings(":value", tokens.items[3].lexeme);
    try std.testing.expectEqual(TokenType.EOF, tokens.items[9].type);
}
