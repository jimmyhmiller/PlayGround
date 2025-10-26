//! MLIR Lexer
//! Tokenizes MLIR source code according to the grammar in grammar.ebnf
//! Implements lexical productions from grammar lines 5-15, 23-33

const std = @import("std");

/// Token types corresponding to MLIR grammar
pub const TokenType = enum {
    // Special tokens
    eof,
    invalid,

    // Literals (grammar lines 11-15)
    // integer-literal ::= decimal-literal | hexadecimal-literal
    integer_literal,
    // float-literal ::= [-+]?[0-9]+[.][0-9]*([eE][-+]?[0-9]+)?
    float_literal,
    // string-literal ::= `"` [^"\n\f\v\r]* `"`
    string_literal,

    // Identifiers (grammar lines 23-30)
    // bare-id ::= (letter|[_]) (letter|digit|[_$.])*
    bare_id,
    // value-id ::= `%` suffix-id
    value_id,
    // symbol-ref-id ::= `@` (suffix-id | string-literal) (`::` symbol-ref-id)?
    symbol_ref_id,
    // caret-id ::= `^` suffix-id
    caret_id,
    // type-alias or attribute-alias: `!` alias-name or `#` alias-name
    type_alias_id,
    attribute_alias_id,

    // Punctuation
    lparen, // (
    rparen, // )
    lbrace, // {
    rbrace, // }
    lbracket, // [
    rbracket, // ]
    langle, // <
    rangle, // >
    comma, // ,
    colon, // :
    semicolon, // ;
    equal, // =
    arrow, // ->
    percent, // %
    at, // @
    caret, // ^
    exclamation, // !
    hash, // #
    double_colon, // ::
    question, // ?

    // Keywords
    kw_loc,
    kw_func,
    kw_return,
    kw_module,
};

pub const Token = struct {
    type: TokenType,
    lexeme: []const u8,
    line: usize,
    column: usize,

    pub fn format(self: Token, comptime fmt: []const u8, options: std.fmt.FormatOptions, writer: anytype) !void {
        _ = fmt;
        _ = options;
        try writer.print("Token{{ .type = {s}, .lexeme = \"{s}\", .line = {}, .column = {} }}", .{
            @tagName(self.type),
            self.lexeme,
            self.line,
            self.column,
        });
    }
};

pub const Lexer = struct {
    source: []const u8,
    start: usize = 0,
    current: usize = 0,
    line: usize = 1,
    column: usize = 1,

    pub fn init(source: []const u8) Lexer {
        return .{ .source = source };
    }

    pub fn nextToken(self: *Lexer) Token {
        self.skipWhitespace();

        self.start = self.current;

        if (self.isAtEnd()) {
            return self.makeToken(.eof);
        }

        const c = self.advance();

        // Grammar: letter ::= [a-zA-Z]
        if (isLetter(c) or c == '_') {
            return self.identifier();
        }

        // Grammar: digit ::= [0-9]
        if (isDigit(c)) {
            return self.number();
        }

        return switch (c) {
            // Grammar: value-id ::= `%` suffix-id
            '%' => blk: {
                if (self.isAtEnd()) break :blk self.makeToken(.percent);
                const next = self.peek();
                if (isDigit(next) or isLetter(next) or next == '$' or next == '.' or next == '_' or next == '-') {
                    break :blk self.valueId();
                }
                break :blk self.makeToken(.percent);
            },
            // Grammar: symbol-ref-id ::= `@` (suffix-id | string-literal)
            '@' => blk: {
                if (self.isAtEnd()) break :blk self.makeToken(.at);
                const next = self.peek();
                if (isDigit(next) or isLetter(next) or next == '$' or next == '.' or next == '_' or next == '-' or next == '"') {
                    break :blk self.symbolRefId();
                }
                break :blk self.makeToken(.at);
            },
            // Grammar: caret-id ::= `^` suffix-id
            '^' => blk: {
                if (self.isAtEnd()) break :blk self.makeToken(.caret);
                const next = self.peek();
                if (isDigit(next) or isLetter(next) or next == '$' or next == '.' or next == '_' or next == '-') {
                    break :blk self.caretId();
                }
                break :blk self.makeToken(.caret);
            },
            // Grammar: type-alias ::= `!` alias-name
            '!' => self.typeAliasId(),
            // Grammar: attribute-alias ::= `#` alias-name
            '#' => self.attributeAliasId(),

            // Grammar: string-literal ::= `"` [^"\n\f\v\r]* `"`
            '"' => self.string(),

            '(' => self.makeToken(.lparen),
            ')' => self.makeToken(.rparen),
            '{' => self.makeToken(.lbrace),
            '}' => self.makeToken(.rbrace),
            '[' => self.makeToken(.lbracket),
            ']' => self.makeToken(.rbracket),
            '<' => self.makeToken(.langle),
            '>' => self.makeToken(.rangle),
            ',' => self.makeToken(.comma),
            ';' => self.makeToken(.semicolon),
            '=' => self.makeToken(.equal),
            '?' => self.makeToken(.question),

            // Grammar: function-type uses `->`
            '-' => if (self.match('>')) self.makeToken(.arrow) else self.makeToken(.invalid),

            // Grammar: symbol-ref-id uses `::`
            ':' => if (self.match(':')) self.makeToken(.double_colon) else self.makeToken(.colon),

            else => self.makeToken(.invalid),
        };
    }

    // Grammar: bare-id ::= (letter|[_]) (letter|digit|[_$.])*
    fn identifier(self: *Lexer) Token {
        while (!self.isAtEnd()) {
            const c = self.peek();
            // Grammar: id-punct ::= [$._-]
            if (isLetter(c) or isDigit(c) or c == '_' or c == '$' or c == '.' or c == '-') {
                _ = self.advance();
            } else {
                break;
            }
        }

        const lexeme = self.source[self.start..self.current];
        const token_type = self.identifierType(lexeme);
        return self.makeToken(token_type);
    }

    fn identifierType(self: *Lexer, lexeme: []const u8) TokenType {
        _ = self;
        // Check for keywords
        if (std.mem.eql(u8, lexeme, "loc")) return .kw_loc;
        if (std.mem.eql(u8, lexeme, "func")) return .kw_func;
        if (std.mem.eql(u8, lexeme, "return")) return .kw_return;
        if (std.mem.eql(u8, lexeme, "module")) return .kw_module;

        return .bare_id;
    }

    // Grammar: value-id ::= `%` suffix-id
    // Grammar: suffix-id ::= (digit+ | ((letter|id-punct) (letter|id-punct|digit)*))
    fn valueId(self: *Lexer) Token {
        // Already consumed '%', caller verified next char is valid
        const c = self.peek();
        if (isDigit(c)) {
            // digit+
            while (!self.isAtEnd() and isDigit(self.peek())) {
                _ = self.advance();
            }
        } else {
            // (letter|id-punct) (letter|id-punct|digit)*
            _ = self.advance();
            while (!self.isAtEnd()) {
                const next = self.peek();
                if (isLetter(next) or isDigit(next) or next == '$' or next == '.' or next == '_' or next == '-') {
                    _ = self.advance();
                } else {
                    break;
                }
            }
        }

        return self.makeToken(.value_id);
    }

    // Grammar: symbol-ref-id ::= `@` (suffix-id | string-literal) (`::` symbol-ref-id)?
    fn symbolRefId(self: *Lexer) Token {
        // Already consumed '@', caller verified next char is valid
        if (self.peek() == '"') {
            // string-literal case
            _ = self.advance(); // consume '"'
            while (!self.isAtEnd() and self.peek() != '"') {
                _ = self.advance();
            }
            if (self.isAtEnd()) return self.makeToken(.invalid);
            _ = self.advance(); // consume closing '"'
        } else {
            // suffix-id case
            const c = self.peek();
            if (isDigit(c)) {
                while (!self.isAtEnd() and isDigit(self.peek())) {
                    _ = self.advance();
                }
            } else {
                _ = self.advance();
                while (!self.isAtEnd()) {
                    const next = self.peek();
                    if (isLetter(next) or isDigit(next) or next == '$' or next == '.' or next == '_' or next == '-') {
                        _ = self.advance();
                    } else {
                        break;
                    }
                }
            }
        }

        // TODO: Handle `::` symbol-ref-id recursion for nested references
        return self.makeToken(.symbol_ref_id);
    }

    // Grammar: caret-id ::= `^` suffix-id
    fn caretId(self: *Lexer) Token {
        // Already consumed '^', caller verified next char is valid
        const c = self.peek();
        if (isDigit(c)) {
            while (!self.isAtEnd() and isDigit(self.peek())) {
                _ = self.advance();
            }
        } else {
            _ = self.advance();
            while (!self.isAtEnd()) {
                const next = self.peek();
                if (isLetter(next) or isDigit(next) or next == '$' or next == '.' or next == '_' or next == '-') {
                    _ = self.advance();
                } else {
                    break;
                }
            }
        }

        return self.makeToken(.caret_id);
    }

    // Grammar: type-alias ::= `!` alias-name
    // Grammar: alias-name ::= bare-id
    fn typeAliasId(self: *Lexer) Token {
        // Already consumed '!'
        if (self.isAtEnd()) return self.makeToken(.exclamation);

        const c = self.peek();
        if (isLetter(c) or c == '_') {
            _ = self.advance();
            while (!self.isAtEnd()) {
                const next = self.peek();
                if (isLetter(next) or isDigit(next) or next == '_' or next == '$' or next == '.' or next == '-') {
                    _ = self.advance();
                } else {
                    break;
                }
            }
            return self.makeToken(.type_alias_id);
        }

        return self.makeToken(.exclamation);
    }

    // Grammar: attribute-alias ::= `#` alias-name
    fn attributeAliasId(self: *Lexer) Token {
        // Already consumed '#'
        if (self.isAtEnd()) return self.makeToken(.hash);

        const c = self.peek();
        if (isLetter(c) or c == '_') {
            _ = self.advance();
            while (!self.isAtEnd()) {
                const next = self.peek();
                if (isLetter(next) or isDigit(next) or next == '_' or next == '$' or next == '.' or next == '-') {
                    _ = self.advance();
                } else {
                    break;
                }
            }
            return self.makeToken(.attribute_alias_id);
        }

        return self.makeToken(.hash);
    }

    // Grammar: string-literal ::= `"` [^"\n\f\v\r]* `"`
    fn string(self: *Lexer) Token {
        // Already consumed opening '"'
        while (!self.isAtEnd() and self.peek() != '"' and self.peek() != '\n') {
            _ = self.advance();
        }

        if (self.isAtEnd() or self.peek() == '\n') {
            return self.makeToken(.invalid);
        }

        // Consume closing '"'
        _ = self.advance();
        return self.makeToken(.string_literal);
    }

    // Grammar: integer-literal ::= decimal-literal | hexadecimal-literal
    // Grammar: decimal-literal ::= digit+
    // Grammar: hexadecimal-literal ::= `0x` hex_digit+
    fn number(self: *Lexer) Token {
        // Check for hexadecimal
        if (self.source[self.start] == '0' and !self.isAtEnd() and self.peek() == 'x') {
            _ = self.advance(); // consume 'x'
            // hex_digit ::= [0-9a-fA-F]
            while (!self.isAtEnd() and isHexDigit(self.peek())) {
                _ = self.advance();
            }
            return self.makeToken(.integer_literal);
        }

        // Decimal integer or float
        while (!self.isAtEnd() and isDigit(self.peek())) {
            _ = self.advance();
        }

        // Grammar: float-literal ::= [-+]?[0-9]+[.][0-9]*([eE][-+]?[0-9]+)?
        if (!self.isAtEnd() and self.peek() == '.') {
            _ = self.advance(); // consume '.'

            // Fractional part
            while (!self.isAtEnd() and isDigit(self.peek())) {
                _ = self.advance();
            }

            // Exponent part
            if (!self.isAtEnd() and (self.peek() == 'e' or self.peek() == 'E')) {
                _ = self.advance();
                if (!self.isAtEnd() and (self.peek() == '+' or self.peek() == '-')) {
                    _ = self.advance();
                }
                while (!self.isAtEnd() and isDigit(self.peek())) {
                    _ = self.advance();
                }
            }

            return self.makeToken(.float_literal);
        }

        return self.makeToken(.integer_literal);
    }

    fn skipWhitespace(self: *Lexer) void {
        while (!self.isAtEnd()) {
            const c = self.peek();
            switch (c) {
                ' ', '\r', '\t' => {
                    _ = self.advance();
                },
                '\n' => {
                    self.line += 1;
                    self.column = 0; // Will be incremented to 1 by advance()
                    _ = self.advance();
                },
                '/' => {
                    if (self.peekNext() == '/') {
                        // Line comment
                        while (!self.isAtEnd() and self.peek() != '\n') {
                            _ = self.advance();
                        }
                    } else {
                        return;
                    }
                },
                else => return,
            }
        }
    }

    fn makeToken(self: *Lexer, token_type: TokenType) Token {
        return .{
            .type = token_type,
            .lexeme = self.source[self.start..self.current],
            .line = self.line,
            .column = self.column - (self.current - self.start),
        };
    }

    fn isAtEnd(self: *Lexer) bool {
        return self.current >= self.source.len;
    }

    fn advance(self: *Lexer) u8 {
        const c = self.source[self.current];
        self.current += 1;
        self.column += 1;
        return c;
    }

    fn peek(self: *Lexer) u8 {
        if (self.isAtEnd()) return 0;
        return self.source[self.current];
    }

    fn peekNext(self: *Lexer) u8 {
        if (self.current + 1 >= self.source.len) return 0;
        return self.source[self.current + 1];
    }

    fn match(self: *Lexer, expected: u8) bool {
        if (self.isAtEnd()) return false;
        if (self.source[self.current] != expected) return false;
        _ = self.advance();
        return true;
    }
};

// Grammar: letter ::= [a-zA-Z]
fn isLetter(c: u8) bool {
    return (c >= 'a' and c <= 'z') or (c >= 'A' and c <= 'Z');
}

// Grammar: digit ::= [0-9]
fn isDigit(c: u8) bool {
    return c >= '0' and c <= '9';
}

// Grammar: hex-digit ::= [0-9a-fA-F]
fn isHexDigit(c: u8) bool {
    return isDigit(c) or (c >= 'a' and c <= 'f') or (c >= 'A' and c <= 'F');
}

test "lexer - simple tokens" {
    const source = "( ) { } [ ] < > , : ; = -> :: % @ ^ ! #";
    var lexer = Lexer.init(source);

    const expected = [_]TokenType{
        .lparen,       .rparen,   .lbrace,       .rbrace,  .lbracket, .rbracket,
        .langle,       .rangle,   .comma,        .colon,   .semicolon, .equal,
        .arrow,        .double_colon, .percent,  .at,      .caret,    .exclamation,
        .hash,         .eof,
    };

    for (expected) |expected_type| {
        const token = lexer.nextToken();
        try std.testing.expectEqual(expected_type, token.type);
    }
}

test "lexer - identifiers" {
    const source = "foo bar_baz test123 _underscore";
    var lexer = Lexer.init(source);

    var token = lexer.nextToken();
    try std.testing.expectEqual(TokenType.bare_id, token.type);
    try std.testing.expectEqualStrings("foo", token.lexeme);

    token = lexer.nextToken();
    try std.testing.expectEqual(TokenType.bare_id, token.type);
    try std.testing.expectEqualStrings("bar_baz", token.lexeme);

    token = lexer.nextToken();
    try std.testing.expectEqual(TokenType.bare_id, token.type);
    try std.testing.expectEqualStrings("test123", token.lexeme);

    token = lexer.nextToken();
    try std.testing.expectEqual(TokenType.bare_id, token.type);
    try std.testing.expectEqualStrings("_underscore", token.lexeme);
}

test "lexer - value ids" {
    const source = "%0 %123 %result %my_value";
    var lexer = Lexer.init(source);

    var token = lexer.nextToken();
    try std.testing.expectEqual(TokenType.value_id, token.type);
    try std.testing.expectEqualStrings("%0", token.lexeme);

    token = lexer.nextToken();
    try std.testing.expectEqual(TokenType.value_id, token.type);
    try std.testing.expectEqualStrings("%123", token.lexeme);

    token = lexer.nextToken();
    try std.testing.expectEqual(TokenType.value_id, token.type);
    try std.testing.expectEqualStrings("%result", token.lexeme);

    token = lexer.nextToken();
    try std.testing.expectEqual(TokenType.value_id, token.type);
    try std.testing.expectEqualStrings("%my_value", token.lexeme);
}

test "lexer - numbers" {
    const source = "42 0x1A 3.14 1.5e-10";
    var lexer = Lexer.init(source);

    var token = lexer.nextToken();
    try std.testing.expectEqual(TokenType.integer_literal, token.type);
    try std.testing.expectEqualStrings("42", token.lexeme);

    token = lexer.nextToken();
    try std.testing.expectEqual(TokenType.integer_literal, token.type);
    try std.testing.expectEqualStrings("0x1A", token.lexeme);

    token = lexer.nextToken();
    try std.testing.expectEqual(TokenType.float_literal, token.type);
    try std.testing.expectEqualStrings("3.14", token.lexeme);

    token = lexer.nextToken();
    try std.testing.expectEqual(TokenType.float_literal, token.type);
    try std.testing.expectEqualStrings("1.5e-10", token.lexeme);
}

test "lexer - string literals" {
    const source = "\"hello\" \"world\"";
    var lexer = Lexer.init(source);

    var token = lexer.nextToken();
    try std.testing.expectEqual(TokenType.string_literal, token.type);
    try std.testing.expectEqualStrings("\"hello\"", token.lexeme);

    token = lexer.nextToken();
    try std.testing.expectEqual(TokenType.string_literal, token.type);
    try std.testing.expectEqualStrings("\"world\"", token.lexeme);
}

test "lexer - keywords" {
    const source = "loc func return module";
    var lexer = Lexer.init(source);

    try std.testing.expectEqual(TokenType.kw_loc, lexer.nextToken().type);
    try std.testing.expectEqual(TokenType.kw_func, lexer.nextToken().type);
    try std.testing.expectEqual(TokenType.kw_return, lexer.nextToken().type);
    try std.testing.expectEqual(TokenType.kw_module, lexer.nextToken().type);
}

test "lexer - simple MLIR operation" {
    const source = "%0 = arith.constant 42 : i32";
    var lexer = Lexer.init(source);

    try std.testing.expectEqual(TokenType.value_id, lexer.nextToken().type);
    try std.testing.expectEqual(TokenType.equal, lexer.nextToken().type);
    try std.testing.expectEqual(TokenType.bare_id, lexer.nextToken().type);
    try std.testing.expectEqual(TokenType.integer_literal, lexer.nextToken().type);
    try std.testing.expectEqual(TokenType.colon, lexer.nextToken().type);
    try std.testing.expectEqual(TokenType.bare_id, lexer.nextToken().type);
    try std.testing.expectEqual(TokenType.eof, lexer.nextToken().type);
}
