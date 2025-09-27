const std = @import("std");
const lexer = @import("lexer.zig");
const value = @import("value.zig");
const vector = @import("collections/vector.zig");
const linked_list = @import("collections/linked_list.zig");
const collections_map = @import("collections/map.zig");

const Lexer = lexer.Lexer;
const Token = lexer.Token;
const TokenType = lexer.TokenType;
const Value = value.Value;
const PersistentVector = vector.PersistentVector;
const PersistentLinkedList = linked_list.PersistentLinkedList;
const PersistentMap = collections_map.PersistentMap;
const PersistentMapWithEq = collections_map.PersistentMapWithEq;

pub const ParseError = error{
    UnexpectedToken,
    UnterminatedString,
    UnterminatedList,
    UnterminatedVector,
    UnterminatedMap,
    InvalidNumber,
    OutOfMemory,
};

pub const Parser = struct {
    lexer: Lexer,
    current_token: Token,
    allocator: *std.mem.Allocator,

    pub fn init(allocator: *std.mem.Allocator, source: []const u8) Parser {
        var lex = Lexer.init(source);
        const first_token = lex.nextToken();
        return Parser{
            .lexer = lex,
            .current_token = first_token,
            .allocator = allocator,
        };
    }

    pub fn parse(self: *Parser) ParseError!*Value {
        return try self.parseExpression();
    }

    pub fn parseAll(self: *Parser, allocator: std.mem.Allocator, results: *std.ArrayList(*Value)) ParseError!void {
        while (self.current_token.type != .eof) {
            const val = try self.parseExpression();
            try results.append(allocator, val);
        }
    }

    fn advance(self: *Parser) void {
        self.current_token = self.lexer.nextToken();
    }

    fn expect(self: *Parser, expected: TokenType) ParseError!Token {
        if (self.current_token.type != expected) {
            return ParseError.UnexpectedToken;
        }
        const token = self.current_token;
        self.advance();
        return token;
    }

    fn parseExpression(self: *Parser) ParseError!*Value {
        return switch (self.current_token.type) {
            .symbol => self.parseSymbol(),
            .keyword => self.parseKeyword(),
            .string => self.parseString(),
            .integer => self.parseInt(),
            .float => self.parseFloat(),
            .left_paren => self.parseList(),
            .left_bracket => self.parseVector(),
            .left_brace => self.parseMap(),
            .quote => self.parseQuote(),
            else => ParseError.UnexpectedToken,
        };
    }

    fn parseSymbol(self: *Parser) ParseError!*Value {
        const token = try self.expect(.symbol);

        // Check for special symbols
        if (std.mem.eql(u8, token.lexeme, "nil")) {
            return try value.createNil(self.allocator.*);
        } else if (std.mem.eql(u8, token.lexeme, "true")) {
            return try value.createSymbol(self.allocator.*, "true");
        } else if (std.mem.eql(u8, token.lexeme, "false")) {
            return try value.createSymbol(self.allocator.*, "false");
        }

        return try value.createSymbol(self.allocator.*, token.lexeme);
    }

    fn parseKeyword(self: *Parser) ParseError!*Value {
        const token = try self.expect(.keyword);
        // Remove the ':' prefix from the keyword
        const name = token.lexeme[1..];
        return try value.createKeyword(self.allocator.*, name);
    }

    fn parseString(self: *Parser) ParseError!*Value {
        const token = try self.expect(.string);
        // Remove quotes and handle escape sequences
        const content = token.lexeme[1..token.lexeme.len-1];

        // For now, simple handling - could expand to handle escape sequences
        return try value.createString(self.allocator.*, content);
    }

    fn parseInt(self: *Parser) ParseError!*Value {
        const token = try self.expect(.integer);
        const int_val = std.fmt.parseInt(i64, token.lexeme, 10) catch {
            return ParseError.InvalidNumber;
        };
        return try value.createInt(self.allocator.*, int_val);
    }

    fn parseFloat(self: *Parser) ParseError!*Value {
        const token = try self.expect(.float);
        const float_val = std.fmt.parseFloat(f64, token.lexeme) catch {
            return ParseError.InvalidNumber;
        };
        return try value.createFloat(self.allocator.*, float_val);
    }

    fn parseList(self: *Parser) ParseError!*Value {
        _ = try self.expect(.left_paren);

        if (self.current_token.type == .symbol and std.mem.eql(u8, self.current_token.lexeme, "ns")) {
            return self.parseNamespaceDecl();
        }

        // Collect elements in correct order
        var elements: [64]*Value = undefined; // Fixed size buffer for simplicity
        var count: usize = 0;

        while (self.current_token.type != .right_paren and self.current_token.type != .eof and count < 64) {
            elements[count] = try self.parseExpression();
            count += 1;
        }

        if (self.current_token.type != .right_paren) {
            return ParseError.UnterminatedList;
        }
        _ = try self.expect(.right_paren);

        // Build list from right to left to get correct order
        var list = try PersistentLinkedList(*Value).empty(self.allocator.*);
        var i = count;
        while (i > 0) {
            i -= 1;
            list = try list.push(self.allocator.*, elements[i]);
        }

        const val = try self.allocator.create(Value);
        val.* = Value{ .list = list };
        return val;
    }

    fn parseNamespaceDecl(self: *Parser) ParseError!*Value {
        _ = try self.expect(.symbol); // consume ns

        const name_token = try self.expect(.symbol);

        if (self.current_token.type != .right_paren) {
            return ParseError.UnexpectedToken;
        }
        _ = try self.expect(.right_paren);

        return try value.createNamespace(self.allocator.*, name_token.lexeme);
    }

    fn parseVector(self: *Parser) ParseError!*Value {
        _ = try self.expect(.left_bracket);

        var vec = PersistentVector(*Value).init(self.allocator.*, null);

        while (self.current_token.type != .right_bracket and self.current_token.type != .eof) {
            const element = try self.parseExpression();
            vec = try vec.push(element);
        }

        if (self.current_token.type != .right_bracket) {
            return ParseError.UnterminatedVector;
        }
        _ = try self.expect(.right_bracket);

        const val = try self.allocator.create(Value);
        val.* = Value{ .vector = vec };
        return val;
    }

    fn parseMap(self: *Parser) ParseError!*Value {
        _ = try self.expect(.left_brace);

        var map_val = PersistentMap(*Value, *Value).init(self.allocator.*);

        while (self.current_token.type != .right_brace and self.current_token.type != .eof) {
            const key = try self.parseExpression();

            if (self.current_token.type == .right_brace) {
                return ParseError.UnexpectedToken; // Odd number of elements
            }

            const val_elem = try self.parseExpression();
            map_val = try map_val.set(key, val_elem);
        }

        if (self.current_token.type != .right_brace) {
            return ParseError.UnterminatedMap;
        }
        _ = try self.expect(.right_brace);

        const val = try self.allocator.create(Value);
        val.* = Value{ .map = map_val };
        return val;
    }

    fn parseQuote(self: *Parser) ParseError!*Value {
        _ = try self.expect(.quote);

        const quoted = try self.parseExpression();

        // Create (quote <expr>)
        var list = try PersistentLinkedList(*Value).empty(self.allocator.*);
        list = try list.push(self.allocator.*, quoted);

        const quote_symbol = try value.createSymbol(self.allocator.*, "quote");
        list = try list.push(self.allocator.*, quote_symbol);

        const val = try self.allocator.create(Value);
        val.* = Value{ .list = list };
        return val;
    }
};

test "parser simple values" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    var allocator = arena.allocator();

    // Test symbol
    {
        var parser = Parser.init(&allocator, "hello");
        const val = try parser.parse();
        try std.testing.expect(val.isSymbol());
        try std.testing.expect(std.mem.eql(u8, val.symbol, "hello"));
    }

    // Test keyword
    {
        var parser = Parser.init(&allocator, ":world");
        const val = try parser.parse();
        try std.testing.expect(val.isKeyword());
        try std.testing.expect(std.mem.eql(u8, val.keyword, "world"));
    }

    // Test string
    {
        var parser = Parser.init(&allocator, "\"test\"");
        const val = try parser.parse();
        try std.testing.expect(val.isString());
        try std.testing.expect(std.mem.eql(u8, val.string, "test"));
    }

    // Test integer
    {
        var parser = Parser.init(&allocator, "42");
        const val = try parser.parse();
        try std.testing.expect(val.isInt());
        try std.testing.expect(val.int == 42);
    }

    // Test float
    {
        var parser = Parser.init(&allocator, "3.14");
        const val = try parser.parse();
        try std.testing.expect(val.isFloat());
        try std.testing.expect(val.float == 3.14);
    }
}

test "parser collections" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    var allocator = arena.allocator();

    // Test empty list
    {
        var parser = Parser.init(&allocator, "()");
        const val = try parser.parse();
        try std.testing.expect(val.isList());
        try std.testing.expect(val.list.isEmpty());
    }

    // Test simple list
    {
        var parser = Parser.init(&allocator, "(+ 1 2)");
        const val = try parser.parse();
        try std.testing.expect(val.isList());
        try std.testing.expect(val.list.len() == 3);
    }

    // Test vector
    {
        var parser = Parser.init(&allocator, "[1 2 3]");
        const val = try parser.parse();
        try std.testing.expect(val.isVector());
        try std.testing.expect(val.vector.len() == 3);
    }

    // Test map
    {
        var parser = Parser.init(&allocator, "{:a 1 :b 2}");
        const val = try parser.parse();
        try std.testing.expect(val.isMap());
    }
}

test "parser namespace declaration" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    var allocator = arena.allocator();

    var parser = Parser.init(&allocator, "(ns my.namespace)");
    const val = try parser.parse();

    try std.testing.expect(val.isNamespace());
    try std.testing.expect(std.mem.eql(u8, val.namespace.name, "my.namespace"));
}
