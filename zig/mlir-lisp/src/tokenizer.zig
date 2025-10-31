const std = @import("std");

/// Represents the different types of tokens in the MLIR S-expression grammar
pub const TokenType = enum {
    // Delimiters
    left_paren, // (
    right_paren, // )
    left_bracket, // [
    right_bracket, // ]
    left_brace, // {
    right_brace, // }

    // Literals
    identifier, // IDENT
    number, // NUMBER (integer, float, hex, binary)
    string, // STRING

    // Special prefixed identifiers
    value_id, // %IDENT (SSA value)
    block_id, // ^IDENT (block label)
    symbol, // @IDENT (symbol-table name)
    type_marker, // !SEXPR
    attr_marker, // #SEXPR

    // Keywords
    keyword, // :IDENT

    // Special values
    true_lit,
    false_lit,

    // Other
    dot, // . (for namespaced names)
    eof,
};

/// Represents a single token with its type, lexeme, and position
pub const Token = struct {
    type: TokenType,
    lexeme: []const u8,
    line: usize,
    column: usize,
};

/// A simple tokenizer for MLIR S-expression grammar
pub const Tokenizer = struct {
    allocator: std.mem.Allocator,
    source: []const u8,
    start: usize = 0,
    current: usize = 0,
    line: usize = 1,
    column: usize = 1,
    start_line: usize = 1,
    start_column: usize = 1,
    error_line: usize = 0,
    error_column: usize = 0,

    /// Initialize a new tokenizer with the given allocator and source string
    pub fn init(allocator: std.mem.Allocator, source: []const u8) Tokenizer {
        return Tokenizer{
            .allocator = allocator,
            .source = source,
        };
    }

    /// Get the current position for error reporting
    pub fn getPosition(self: *const Tokenizer) struct { line: usize, column: usize } {
        return .{ .line = self.error_line, .column = self.error_column };
    }

    /// Get the next token from the source
    pub fn next(self: *Tokenizer) !Token {
        self.skipWhitespaceAndComments();

        self.start = self.current;
        self.start_line = self.line;
        self.start_column = self.column;

        if (self.isAtEnd()) {
            return self.makeToken(.eof);
        }

        const c = self.advance();

        // Single-character tokens
        return switch (c) {
            '(' => self.makeToken(.left_paren),
            ')' => self.makeToken(.right_paren),
            '[' => self.makeToken(.left_bracket),
            ']' => self.makeToken(.right_bracket),
            '{' => self.makeToken(.left_brace),
            '}' => self.makeToken(.right_brace),
            '.' => self.makeToken(.dot),

            // String literals
            '"' => self.scanString(),

            // Special prefixes
            '%' => self.scanValueId(),
            '^' => self.scanBlockId(),
            '@' => self.scanSymbol(),
            '!' => self.scanTypeMarker(),
            '#' => self.scanAttrMarker(),

            // Keywords (colon-prefixed) or standalone ':' as identifier
            ':' => self.scanColonToken(),

            // Numbers or identifiers
            else => {
                if (self.isDigit(c) or (c == '-' and self.peek() != 0 and self.isDigit(self.peek()))) {
                    return self.scanNumber();
                } else if (self.isAlpha(c) or c == '_') {
                    return self.scanIdentifier();
                } else {
                    self.error_line = self.line;
                    self.error_column = self.column - 1;
                    return error.UnexpectedCharacter;
                }
            },
        };
    }

    fn isAtEnd(self: *const Tokenizer) bool {
        return self.current >= self.source.len;
    }

    fn advance(self: *Tokenizer) u8 {
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

    fn peek(self: *const Tokenizer) u8 {
        if (self.isAtEnd()) return 0;
        return self.source[self.current];
    }

    fn peekNext(self: *const Tokenizer) u8 {
        if (self.current + 1 >= self.source.len) return 0;
        return self.source[self.current + 1];
    }

    fn skipWhitespaceAndComments(self: *Tokenizer) void {
        while (!self.isAtEnd()) {
            const c = self.peek();
            switch (c) {
                ' ', '\r', '\t', '\n' => {
                    _ = self.advance();
                },
                ';' => {
                    // Skip comment until end of line
                    while (self.peek() != '\n' and !self.isAtEnd()) {
                        _ = self.advance();
                    }
                },
                else => return,
            }
        }
    }

    fn makeToken(self: *const Tokenizer, token_type: TokenType) Token {
        return Token{
            .type = token_type,
            .lexeme = self.source[self.start..self.current],
            .line = self.start_line,
            .column = self.start_column,
        };
    }

    fn scanString(self: *Tokenizer) !Token {
        while (self.peek() != '"' and !self.isAtEnd()) {
            if (self.peek() == '\\') {
                _ = self.advance(); // Skip the backslash
                if (!self.isAtEnd()) {
                    _ = self.advance(); // Skip the escaped character
                }
            } else {
                _ = self.advance();
            }
        }

        if (self.isAtEnd()) {
            self.error_line = self.line;
            self.error_column = self.column;
            return error.UnterminatedString;
        }

        // Consume closing quote
        _ = self.advance();

        return self.makeToken(.string);
    }

    fn scanValueId(self: *Tokenizer) !Token {
        // Already consumed '%', now scan the suffix-id
        // MLIR allows: digit+ | (letter|punct)(letter|digit|punct)*
        const first_char = self.peek();

        if (self.isDigit(first_char)) {
            // Pure numeric: %0, %42, etc.
            while (self.isDigit(self.peek())) {
                _ = self.advance();
            }
        } else if (self.isAlpha(first_char) or first_char == '_') {
            // Named identifier: %arg0, %batch_size, etc.
            while (self.isAlphaNumeric(self.peek()) or self.isIdentifierChar(self.peek())) {
                _ = self.advance();
            }
        } else {
            self.error_line = self.line;
            self.error_column = self.column;
            return error.InvalidValueId;
        }

        return self.makeToken(.value_id);
    }

    fn scanBlockId(self: *Tokenizer) !Token {
        // Already consumed '^', now scan the suffix-id
        // MLIR allows: digit+ | (letter|punct)(letter|digit|punct)*
        const first_char = self.peek();

        if (self.isDigit(first_char)) {
            // Pure numeric: ^0, ^42, etc.
            while (self.isDigit(self.peek())) {
                _ = self.advance();
            }
        } else if (self.isAlpha(first_char) or first_char == '_') {
            // Named identifier: ^entry, ^bb0, etc.
            while (self.isAlphaNumeric(self.peek()) or self.isIdentifierChar(self.peek())) {
                _ = self.advance();
            }
        } else {
            self.error_line = self.line;
            self.error_column = self.column;
            return error.InvalidBlockId;
        }

        return self.makeToken(.block_id);
    }

    fn scanSymbol(self: *Tokenizer) !Token {
        // Already consumed '@', now scan the suffix-id
        // MLIR allows: digit+ | (letter|punct)(letter|digit|punct)*
        const first_char = self.peek();

        if (self.isDigit(first_char)) {
            // Pure numeric: @0, @42, etc. (uncommon but valid)
            while (self.isDigit(self.peek())) {
                _ = self.advance();
            }
        } else if (self.isAlpha(first_char) or first_char == '_') {
            // Named identifier: @main, @test_func, etc.
            while (self.isAlphaNumeric(self.peek()) or self.isIdentifierChar(self.peek())) {
                _ = self.advance();
            }
        } else {
            self.error_line = self.line;
            self.error_column = self.column;
            return error.InvalidSymbol;
        }

        return self.makeToken(.symbol);
    }

    fn scanColonToken(self: *Tokenizer) !Token {
        // Already consumed ':', check what follows
        const next_char = self.peek();

        // If followed by whitespace, delimiter, or EOF, it's a standalone ':' identifier
        if (next_char == 0 or next_char == ' ' or next_char == '\t' or next_char == '\n' or next_char == '\r' or
            next_char == '(' or next_char == ')' or next_char == '[' or next_char == ']' or
            next_char == '{' or next_char == '}') {
            return self.makeToken(.identifier);
        }

        // Otherwise, it's a keyword - scan the identifier part
        if (!self.isAlpha(next_char) and next_char != '_') {
            self.error_line = self.line;
            self.error_column = self.column;
            return error.InvalidKeyword;
        }

        while (self.isAlphaNumeric(self.peek()) or self.isIdentifierChar(self.peek()) or self.peek() == '.') {
            _ = self.advance();
        }

        return self.makeToken(.keyword);
    }

    /// Helper function to scan opaque marker values (types and attributes)
    /// with bracket-aware parsing. Handles nested brackets <>, (), [] and strings.
    fn scanOpaqueMarker(self: *Tokenizer) void {
        // This uses bracket-aware scanning to handle complex nested values
        // with spaces, like: #dlti.dl_spec<i1 = dense<8> : vector<2xi64>, ...>
        // or #llvm.target_features<["+aes", "+alternate-sextload-cvt-f32-pattern"]>
        //
        // Algorithm:
        // - Track depth of <>, (), and [] brackets
        // - Track whether we're inside a string literal "..."
        // - Only stop at whitespace or delimiters when depth is 0 and not in string

        var depth: i32 = 0;
        var in_string = false;

        while (!self.isAtEnd()) {
            const c = self.peek();

            // Handle string literals (quotes toggle in_string state)
            if (c == '"') {
                // Check if the quote is escaped by looking back
                // We need to check if there's an odd number of backslashes before it
                var num_backslashes: usize = 0;
                var check_pos = self.current;
                while (check_pos > self.start and self.source[check_pos - 1] == '\\') {
                    num_backslashes += 1;
                    check_pos -= 1;
                }
                // If even number of backslashes (including 0), quote is not escaped
                if (num_backslashes % 2 == 0) {
                    in_string = !in_string;
                }
            }

            // When not in a string, track bracket depth
            if (!in_string) {
                if (c == '<' or c == '(' or c == '[') {
                    depth += 1;
                } else if (c == '>' or c == ')' or c == ']') {
                    depth -= 1;
                    // If depth goes negative, we hit a closing bracket at s-expr level
                    if (depth < 0) break;
                }

                // At depth 0, stop at whitespace or s-expr delimiters
                if (depth == 0) {
                    if (c == ' ' or c == '\t' or c == '\n' or c == '\r') break;
                    if (c == '{' or c == '}' or c == ';') break;
                }
            }

            _ = self.advance();
        }
    }

    fn scanAttrMarker(self: *Tokenizer) Token {
        // Already consumed '#', now scan the opaque attribute value
        self.scanOpaqueMarker();
        return self.makeToken(.attr_marker);
    }

    fn scanTypeMarker(self: *Tokenizer) Token {
        // Already consumed '!', now scan the opaque type value
        self.scanOpaqueMarker();
        return self.makeToken(.type_marker);
    }

    fn scanNumber(self: *Tokenizer) Token {
        // Handle negative numbers
        if (self.source[self.start] == '-') {
            // Already advanced past the '-'
        }

        // Check for hex (0x) or binary (0b)
        if (self.source[self.current - 1] == '0' and !self.isAtEnd()) {
            const next_char = self.peek();
            if (next_char == 'x' or next_char == 'X') {
                _ = self.advance(); // consume 'x'
                while (self.isHexDigit(self.peek())) {
                    _ = self.advance();
                }
                return self.makeToken(.number);
            } else if (next_char == 'b' or next_char == 'B') {
                _ = self.advance(); // consume 'b'
                while (self.peek() == '0' or self.peek() == '1') {
                    _ = self.advance();
                }
                return self.makeToken(.number);
            }
        }

        // Regular decimal number
        while (self.isDigit(self.peek())) {
            _ = self.advance();
        }

        // Check for decimal point (float)
        if (self.peek() == '.' and self.isDigit(self.peekNext())) {
            _ = self.advance(); // consume '.'
            while (self.isDigit(self.peek())) {
                _ = self.advance();
            }
        }

        // Check for exponent
        if (self.peek() == 'e' or self.peek() == 'E') {
            _ = self.advance();
            if (self.peek() == '+' or self.peek() == '-') {
                _ = self.advance();
            }
            while (self.isDigit(self.peek())) {
                _ = self.advance();
            }
        }

        return self.makeToken(.number);
    }

    fn scanIdentifier(self: *Tokenizer) Token {
        while (self.isAlphaNumeric(self.peek()) or self.isIdentifierChar(self.peek())) {
            _ = self.advance();
        }

        // Check for keywords
        const text = self.source[self.start..self.current];
        if (std.mem.eql(u8, text, "true")) {
            return self.makeToken(.true_lit);
        } else if (std.mem.eql(u8, text, "false")) {
            return self.makeToken(.false_lit);
        }

        return self.makeToken(.identifier);
    }

    fn isDigit(self: *const Tokenizer, c: u8) bool {
        _ = self;
        return c >= '0' and c <= '9';
    }

    fn isHexDigit(self: *const Tokenizer, c: u8) bool {
        _ = self;
        return (c >= '0' and c <= '9') or
            (c >= 'a' and c <= 'f') or
            (c >= 'A' and c <= 'F');
    }

    fn isAlpha(self: *const Tokenizer, c: u8) bool {
        _ = self;
        return (c >= 'a' and c <= 'z') or
            (c >= 'A' and c <= 'Z');
    }

    fn isAlphaNumeric(self: *const Tokenizer, c: u8) bool {
        return self.isAlpha(c) or self.isDigit(c);
    }

    fn isIdentifierChar(self: *const Tokenizer, c: u8) bool {
        _ = self;
        // From grammar: [A-Za-z_][A-Za-z0-9_.$:-]*
        return c == '_' or c == '.' or c == '$' or c == ':' or c == '-';
    }
};
