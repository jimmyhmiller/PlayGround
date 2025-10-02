const std = @import("std");

pub const TokenType = enum {
    // Literals
    symbol,
    keyword,
    string,
    integer,
    float,

    // Delimiters
    left_paren,    // (
    right_paren,   // )
    left_bracket,  // [
    right_bracket, // ]
    left_brace,    // {
    right_brace,   // }

    // Special
    quote,         // '

    // Meta
    eof,
    invalid,
};

pub const Token = struct {
    type: TokenType,
    lexeme: []const u8,
    line: u32,
    column: u32,
};

pub const Lexer = struct {
    source: []const u8,
    current: usize = 0,
    line: u32 = 1,
    column: u32 = 1,

    pub fn init(source: []const u8) Lexer {
        return Lexer{
            .source = source,
        };
    }

    pub fn nextToken(self: *Lexer) Token {
        self.skipWhitespace();

        if (self.isAtEnd()) {
            return self.makeToken(.eof);
        }

        const start_line = self.line;
        const start_column = self.column;
        const start = self.current;
        const c = self.advance();

        return switch (c) {
            '(' => self.makeTokenAt(.left_paren, start, start_line, start_column),
            ')' => self.makeTokenAt(.right_paren, start, start_line, start_column),
            '[' => self.makeTokenAt(.left_bracket, start, start_line, start_column),
            ']' => self.makeTokenAt(.right_bracket, start, start_line, start_column),
            '{' => self.makeTokenAt(.left_brace, start, start_line, start_column),
            '}' => self.makeTokenAt(.right_brace, start, start_line, start_column),
            '\'' => self.makeTokenAt(.quote, start, start_line, start_column),
            '"' => self.string(start, start_line, start_column),
            ':' => self.keyword(start, start_line, start_column),
            ';' => {
                // Skip comment to end of line
                while (!self.isAtEnd() and self.peek() != '\n') {
                    _ = self.advance();
                }
                return self.nextToken();
            },
            else => {
                if (std.ascii.isDigit(c) or (c == '-' and self.peekNext() != 0 and std.ascii.isDigit(self.peekNext()))) {
                    return self.number(start, start_line, start_column);
                } else if (self.isSymbolStart(c)) {
                    return self.symbol(start, start_line, start_column);
                } else {
                    return self.makeTokenAt(.invalid, start, start_line, start_column);
                }
            },
        };
    }

    fn isAtEnd(self: *const Lexer) bool {
        return self.current >= self.source.len;
    }

    fn advance(self: *Lexer) u8 {
        if (self.isAtEnd()) return 0;

        const c = self.source[self.current];
        self.current += 1;

        if (c == '\n') {
            self.line += 1;
            self.column = 1;
        } else {
            self.column += 1;
        }

        return c;
    }

    fn peek(self: *const Lexer) u8 {
        if (self.isAtEnd()) return 0;
        return self.source[self.current];
    }

    fn peekNext(self: *const Lexer) u8 {
        if (self.current + 1 >= self.source.len) return 0;
        return self.source[self.current + 1];
    }

    fn skipWhitespace(self: *Lexer) void {
        while (!self.isAtEnd()) {
            const c = self.peek();
            if (std.ascii.isWhitespace(c)) {
                _ = self.advance();
            } else {
                break;
            }
        }
    }

    fn makeToken(self: *const Lexer, token_type: TokenType) Token {
        return Token{
            .type = token_type,
            .lexeme = if (self.current > 0) self.source[self.current-1..self.current] else "",
            .line = self.line,
            .column = self.column,
        };
    }

    fn makeTokenAt(self: *const Lexer, token_type: TokenType, start: usize, line: u32, column: u32) Token {
        return Token{
            .type = token_type,
            .lexeme = self.source[start..self.current],
            .line = line,
            .column = column,
        };
    }

    fn string(self: *Lexer, start: usize, start_line: u32, start_column: u32) Token {
        // Skip opening quote - it's already consumed
        while (!self.isAtEnd() and self.peek() != '"') {
            if (self.peek() == '\\') {
                _ = self.advance(); // Skip escape character
                if (!self.isAtEnd()) {
                    _ = self.advance(); // Skip escaped character
                }
            } else {
                _ = self.advance();
            }
        }

        if (self.isAtEnd()) {
            return self.makeTokenAt(.invalid, start, start_line, start_column);
        }

        // Skip closing quote
        _ = self.advance();

        return self.makeTokenAt(.string, start, start_line, start_column);
    }

    fn keyword(self: *Lexer, start: usize, start_line: u32, start_column: u32) Token {
        // Skip the ':' - it's already consumed
        while (!self.isAtEnd() and self.isSymbolChar(self.peek())) {
            _ = self.advance();
        }

        return self.makeTokenAt(.keyword, start, start_line, start_column);
    }

    fn number(self: *Lexer, start: usize, start_line: u32, start_column: u32) Token {
        var is_float = false;

        // Skip optional negative sign - already consumed if present
        while (!self.isAtEnd() and std.ascii.isDigit(self.peek())) {
            _ = self.advance();
        }

        // Look for decimal point
        if (!self.isAtEnd() and self.peek() == '.' and self.peekNext() != 0 and std.ascii.isDigit(self.peekNext())) {
            is_float = true;
            _ = self.advance(); // consume '.'

            while (!self.isAtEnd() and std.ascii.isDigit(self.peek())) {
                _ = self.advance();
            }
        }

        return self.makeTokenAt(if (is_float) .float else .integer, start, start_line, start_column);
    }

    fn symbol(self: *Lexer, start: usize, start_line: u32, start_column: u32) Token {
        while (!self.isAtEnd() and self.isSymbolChar(self.peek())) {
            _ = self.advance();
        }

        return self.makeTokenAt(.symbol, start, start_line, start_column);
    }

    fn isSymbolStart(_: *const Lexer, c: u8) bool {
        return std.ascii.isAlphabetic(c) or
               c == '+' or c == '-' or c == '*' or c == '/' or c == '=' or
               c == '<' or c == '>' or c == '!' or c == '?' or c == '&' or
               c == '|' or c == '^' or c == '~' or c == '%' or c == '$' or
               c == '_' or c == '.';
    }

    fn isSymbolChar(self: *const Lexer, c: u8) bool {
        return self.isSymbolStart(c) or std.ascii.isDigit(c) or c == '.' or c == '#';
    }
};

test "lexer basic tokens" {
    var lexer = Lexer.init("()[]{}");

    try std.testing.expect(lexer.nextToken().type == .left_paren);
    try std.testing.expect(lexer.nextToken().type == .right_paren);
    try std.testing.expect(lexer.nextToken().type == .left_bracket);
    try std.testing.expect(lexer.nextToken().type == .right_bracket);
    try std.testing.expect(lexer.nextToken().type == .left_brace);
    try std.testing.expect(lexer.nextToken().type == .right_brace);
    try std.testing.expect(lexer.nextToken().type == .eof);
}

test "lexer literals" {
    var lexer = Lexer.init("hello :world \"test\" 42 3.14");

    const sym = lexer.nextToken();
    try std.testing.expect(sym.type == .symbol);
    try std.testing.expect(std.mem.eql(u8, sym.lexeme, "hello"));

    const kw = lexer.nextToken();
    try std.testing.expect(kw.type == .keyword);
    try std.testing.expect(std.mem.eql(u8, kw.lexeme, ":world"));

    const str = lexer.nextToken();
    try std.testing.expect(str.type == .string);
    try std.testing.expect(std.mem.eql(u8, str.lexeme, "\"test\""));

    const int = lexer.nextToken();
    try std.testing.expect(int.type == .integer);
    try std.testing.expect(std.mem.eql(u8, int.lexeme, "42"));

    const float = lexer.nextToken();
    try std.testing.expect(float.type == .float);
    try std.testing.expect(std.mem.eql(u8, float.lexeme, "3.14"));
}

test "lexer complex example" {
    var lexer = Lexer.init("(defn my-func [{:keys [a b c]}] (+ a b c))");

    // (defn my-func [{:keys [a b c]}] (+ a b c))
    const tokens = [_]TokenType{
        .left_paren,   // (
        .symbol,       // defn
        .symbol,       // my-func
        .left_bracket, // [
        .left_brace,   // {
        .keyword,      // :keys
        .left_bracket, // [
        .symbol,       // a
        .symbol,       // b
        .symbol,       // c
        .right_bracket, // ]
        .right_brace,   // }
        .right_bracket, // ]
        .left_paren,   // (
        .symbol,       // +
        .symbol,       // a
        .symbol,       // b
        .symbol,       // c
        .right_paren,  // )
        .right_paren,  // )
        .eof
    };

    for (tokens, 0..) |expected_type, i| {
        const token = lexer.nextToken();
        if (token.type != expected_type) {
            std.debug.print("Token {d}: expected {}, got {} ({s})\n", .{ i, expected_type, token.type, token.lexeme });
        }
        try std.testing.expect(token.type == expected_type);
    }
}