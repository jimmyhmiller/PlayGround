const std = @import("std");
const tokenizer = @import("tokenizer.zig");
const Token = tokenizer.Token;
const TokenType = tokenizer.TokenType;
const Tokenizer = tokenizer.Tokenizer;
const vector = @import("collections/vector.zig");
const map = @import("collections/map.zig");

/// Value types in our MLIR S-expression language
pub const ValueType = enum {
    // Atoms
    identifier,
    number,
    string,
    value_id,
    block_id,
    symbol,
    keyword,
    true_lit,
    false_lit,

    // Collections
    list, // ( ... )
    vector, // [ ... ]
    map, // { ... }

    // Special markers
    type_expr, // ! ...
    attr_expr, // # ...
    has_type, // (: value type) - typed literal
};

/// A value in our language - can be an atom or a collection
pub const Value = struct {
    type: ValueType,
    data: union {
        atom: []const u8, // For all atom types (identifier, number, string, etc.)
        list: vector.PersistentVector(*Value), // For lists ( ... )
        vector: vector.PersistentVector(*Value), // For vectors [ ... ]
        map: vector.PersistentVector(*Value), // For maps { ... } - stored as flat list of k,v pairs
        type_expr: *Value, // For type expressions ! ...
        attr_expr: *Value, // For attribute expressions # ...
        has_type: struct { value: *Value, type_expr: *Value }, // For typed literals (: value type)
    },

    pub fn deinit(self: *Value, allocator: std.mem.Allocator) void {
        switch (self.type) {
            .list, .vector, .map => {
                const vec = switch (self.type) {
                    .list => self.data.list,
                    .vector => self.data.vector,
                    .map => self.data.map,
                    else => unreachable,
                };
                // Deinit all child values
                const slice = vec.slice();
                for (slice) |child| {
                    child.deinit(allocator);
                    allocator.destroy(child);
                }
                var mut_vec = vec;
                mut_vec.deinit();
            },
            .type_expr => {
                self.data.type_expr.deinit(allocator);
                allocator.destroy(self.data.type_expr);
            },
            .attr_expr => {
                self.data.attr_expr.deinit(allocator);
                allocator.destroy(self.data.attr_expr);
            },
            .has_type => {
                self.data.has_type.value.deinit(allocator);
                allocator.destroy(self.data.has_type.value);
                self.data.has_type.type_expr.deinit(allocator);
                allocator.destroy(self.data.has_type.type_expr);
            },
            else => {}, // Atoms don't need deallocation (lexeme is owned by tokenizer)
        }
    }

    /// Get the name of a keyword (strips the leading colon)
    /// For example: ":value" -> "value"
    pub fn keywordToName(self: *const Value) []const u8 {
        std.debug.assert(self.type == .keyword);
        const keyword = self.data.atom;
        std.debug.assert(keyword.len > 0 and keyword[0] == ':');
        return keyword[1..];
    }
};

/// Errors that can occur during reading
pub const ReaderError = error{
    UnexpectedEOF,
    UnexpectedClosingDelimiter,
    UnexpectedDot,
    UnterminatedList,
    // Tokenizer errors
    UnexpectedCharacter,
    UnterminatedString,
    InvalidValueId,
    InvalidBlockId,
    InvalidSymbol,
    InvalidKeyword,
} || std.mem.Allocator.Error;

/// Reader converts tokens into values
pub const Reader = struct {
    allocator: std.mem.Allocator,
    tokenizer: *Tokenizer,
    current: ?Token = null,

    pub fn init(allocator: std.mem.Allocator, tok: *Tokenizer) ReaderError!Reader {
        var reader = Reader{
            .allocator = allocator,
            .tokenizer = tok,
        };
        // Prime the reader with the first token
        reader.current = try tok.next();
        return reader;
    }

    /// Advance to the next token
    fn advance(self: *Reader) ReaderError!void {
        self.current = try self.tokenizer.next();
    }

    /// Check if we're at EOF
    fn isAtEnd(self: *const Reader) bool {
        return self.current != null and self.current.?.type == .eof;
    }

    /// Read the next value from the token stream
    pub fn read(self: *Reader) ReaderError!*Value {
        if (self.current == null or self.isAtEnd()) {
            return error.UnexpectedEOF;
        }

        const tok = self.current.?;

        switch (tok.type) {
            // Simple atoms
            .identifier => {
                const value = try self.allocator.create(Value);
                value.* = Value{
                    .type = .identifier,
                    .data = .{ .atom = tok.lexeme },
                };
                try self.advance();
                return value;
            },
            .number => {
                const value = try self.allocator.create(Value);
                value.* = Value{
                    .type = .number,
                    .data = .{ .atom = tok.lexeme },
                };
                try self.advance();
                return value;
            },
            .string => {
                const value = try self.allocator.create(Value);
                value.* = Value{
                    .type = .string,
                    .data = .{ .atom = tok.lexeme },
                };
                try self.advance();
                return value;
            },
            .value_id => {
                const value = try self.allocator.create(Value);
                value.* = Value{
                    .type = .value_id,
                    .data = .{ .atom = tok.lexeme },
                };
                try self.advance();
                return value;
            },
            .block_id => {
                const value = try self.allocator.create(Value);
                value.* = Value{
                    .type = .block_id,
                    .data = .{ .atom = tok.lexeme },
                };
                try self.advance();
                return value;
            },
            .symbol => {
                const value = try self.allocator.create(Value);
                value.* = Value{
                    .type = .symbol,
                    .data = .{ .atom = tok.lexeme },
                };
                try self.advance();
                return value;
            },
            .keyword => {
                const value = try self.allocator.create(Value);
                value.* = Value{
                    .type = .keyword,
                    .data = .{ .atom = tok.lexeme },
                };
                try self.advance();
                return value;
            },
            .true_lit => {
                const value = try self.allocator.create(Value);
                value.* = Value{
                    .type = .true_lit,
                    .data = .{ .atom = "true" },
                };
                try self.advance();
                return value;
            },
            .false_lit => {
                const value = try self.allocator.create(Value);
                value.* = Value{
                    .type = .false_lit,
                    .data = .{ .atom = "false" },
                };
                try self.advance();
                return value;
            },

            // Collections
            .left_paren => {
                try self.advance(); // consume '('
                return try self.readList(.right_paren, .list);
            },
            .left_bracket => {
                try self.advance(); // consume '['
                return try self.readList(.right_bracket, .vector);
            },
            .left_brace => {
                try self.advance(); // consume '{'
                return try self.readList(.right_brace, .map);
            },

            // Special markers
            .type_marker => {
                try self.advance(); // consume '!'
                const inner = try self.read();
                const value = try self.allocator.create(Value);
                value.* = Value{
                    .type = .type_expr,
                    .data = .{ .type_expr = inner },
                };
                return value;
            },
            .attr_marker => {
                try self.advance(); // consume '#'
                const inner = try self.read();
                const value = try self.allocator.create(Value);
                value.* = Value{
                    .type = .attr_expr,
                    .data = .{ .attr_expr = inner },
                };
                return value;
            },

            // Unexpected closing delimiters
            .right_paren, .right_bracket, .right_brace => {
                return error.UnexpectedClosingDelimiter;
            },

            // Other tokens
            .dot => return error.UnexpectedDot,
            .eof => return error.UnexpectedEOF,
        }
    }

    /// Read a list/vector/map until we hit the closing delimiter
    fn readList(self: *Reader, closing: TokenType, value_type: ValueType) ReaderError!*Value {
        var vec = vector.PersistentVector(*Value).init(self.allocator, null);

        while (self.current != null and !self.isAtEnd() and self.current.?.type != closing) {
            const elem = try self.read();
            const new_vec = try vec.push(elem);
            vec.deinit();
            vec = new_vec;
        }

        if (self.current == null or self.isAtEnd()) {
            // Clean up the vector before returning error
            const slice = vec.slice();
            for (slice) |child| {
                child.deinit(self.allocator);
                self.allocator.destroy(child);
            }
            vec.deinit();
            return error.UnterminatedList;
        }

        // Consume the closing delimiter
        try self.advance();

        // Check for (: value type) pattern - only for lists
        if (value_type == .list and vec.len() == 3) {
            const slice = vec.slice();
            const first = slice[0];

            // Check if first element is ':' identifier
            if (first.type == .identifier and std.mem.eql(u8, first.data.atom, ":")) {
                // This is a typed literal: (: value type)
                const val = slice[1];
                const type_val = slice[2];

                // Create has_type value
                const typed_value = try self.allocator.create(Value);
                typed_value.* = Value{
                    .type = .has_type,
                    .data = .{ .has_type = .{ .value = val, .type_expr = type_val } },
                };

                // Destroy the ':' identifier (not needed in has_type)
                first.deinit(self.allocator);
                self.allocator.destroy(first);

                // Clean up the vector now that we've extracted what we need
                // The vector itself needs to be freed, but the children are now owned by typed_value
                var mut_vec = vec;
                mut_vec.deinit();

                return typed_value;
            }
        }

        const value = try self.allocator.create(Value);
        value.* = Value{
            .type = value_type,
            .data = switch (value_type) {
                .list => .{ .list = vec },
                .vector => .{ .vector = vec },
                .map => .{ .map = vec },
                else => unreachable,
            },
        };

        return value;
    }

    /// Read all values from the token stream until EOF
    pub fn readAll(self: *Reader) ReaderError!vector.PersistentVector(*Value) {
        var vec = vector.PersistentVector(*Value).init(self.allocator, null);

        while (!self.isAtEnd()) {
            const value = try self.read();
            const new_vec = try vec.push(value);
            vec.deinit();
            vec = new_vec;
        }

        return vec;
    }
};

test "reader - simple atoms" {
    const allocator = std.testing.allocator;

    // Test identifier
    {
        const source = "hello";
        var tok = Tokenizer.init(allocator, source);
        var reader = try Reader.init(allocator, &tok);
        var value = try reader.read();
        defer {
            value.deinit(allocator);
            allocator.destroy(value);
        }

        try std.testing.expect(value.type == .identifier);
        try std.testing.expectEqualStrings("hello", value.data.atom);
    }

    // Test number
    {
        const source = "42";
        var tok = Tokenizer.init(allocator, source);
        var reader = try Reader.init(allocator, &tok);
        var value = try reader.read();
        defer {
            value.deinit(allocator);
            allocator.destroy(value);
        }

        try std.testing.expect(value.type == .number);
        try std.testing.expectEqualStrings("42", value.data.atom);
    }

    // Test string
    {
        const source = "\"hello world\"";
        var tok = Tokenizer.init(allocator, source);
        var reader = try Reader.init(allocator, &tok);
        var value = try reader.read();
        defer {
            value.deinit(allocator);
            allocator.destroy(value);
        }

        try std.testing.expect(value.type == .string);
        try std.testing.expectEqualStrings("\"hello world\"", value.data.atom);
    }

    // Test value_id
    {
        const source = "%x";
        var tok = Tokenizer.init(allocator, source);
        var reader = try Reader.init(allocator, &tok);
        var value = try reader.read();
        defer {
            value.deinit(allocator);
            allocator.destroy(value);
        }

        try std.testing.expect(value.type == .value_id);
        try std.testing.expectEqualStrings("%x", value.data.atom);
    }

    // Test block_id
    {
        const source = "^entry";
        var tok = Tokenizer.init(allocator, source);
        var reader = try Reader.init(allocator, &tok);
        var value = try reader.read();
        defer {
            value.deinit(allocator);
            allocator.destroy(value);
        }

        try std.testing.expect(value.type == .block_id);
        try std.testing.expectEqualStrings("^entry", value.data.atom);
    }

    // Test symbol
    {
        const source = "@main";
        var tok = Tokenizer.init(allocator, source);
        var reader = try Reader.init(allocator, &tok);
        var value = try reader.read();
        defer {
            value.deinit(allocator);
            allocator.destroy(value);
        }

        try std.testing.expect(value.type == .symbol);
        try std.testing.expectEqualStrings("@main", value.data.atom);
    }

    // Test keyword
    {
        const source = ":value";
        var tok = Tokenizer.init(allocator, source);
        var reader = try Reader.init(allocator, &tok);
        var value = try reader.read();
        defer {
            value.deinit(allocator);
            allocator.destroy(value);
        }

        try std.testing.expect(value.type == .keyword);
        try std.testing.expectEqualStrings(":value", value.data.atom);
    }

    // Test true/false
    {
        const source = "true";
        var tok = Tokenizer.init(allocator, source);
        var reader = try Reader.init(allocator, &tok);
        var value = try reader.read();
        defer {
            value.deinit(allocator);
            allocator.destroy(value);
        }

        try std.testing.expect(value.type == .true_lit);
    }

    {
        const source = "false";
        var tok = Tokenizer.init(allocator, source);
        var reader = try Reader.init(allocator, &tok);
        var value = try reader.read();
        defer {
            value.deinit(allocator);
            allocator.destroy(value);
        }

        try std.testing.expect(value.type == .false_lit);
    }
}

test "reader - lists" {
    const allocator = std.testing.allocator;

    // Test empty list
    {
        const source = "()";
        var tok = Tokenizer.init(allocator, source);
        var reader = try Reader.init(allocator, &tok);
        var value = try reader.read();
        defer {
            value.deinit(allocator);
            allocator.destroy(value);
        }

        try std.testing.expect(value.type == .list);
        try std.testing.expect(value.data.list.len() == 0);
    }

    // Test list with elements
    {
        const source = "(1 2 3)";
        var tok = Tokenizer.init(allocator, source);
        var reader = try Reader.init(allocator, &tok);
        var value = try reader.read();
        defer {
            value.deinit(allocator);
            allocator.destroy(value);
        }

        try std.testing.expect(value.type == .list);
        try std.testing.expect(value.data.list.len() == 3);
        try std.testing.expect(value.data.list.at(0).type == .number);
        try std.testing.expectEqualStrings("1", value.data.list.at(0).data.atom);
    }

    // Test nested list
    {
        const source = "(1 (2 3) 4)";
        var tok = Tokenizer.init(allocator, source);
        var reader = try Reader.init(allocator, &tok);
        var value = try reader.read();
        defer {
            value.deinit(allocator);
            allocator.destroy(value);
        }

        try std.testing.expect(value.type == .list);
        try std.testing.expect(value.data.list.len() == 3);
        try std.testing.expect(value.data.list.at(1).type == .list);
        try std.testing.expect(value.data.list.at(1).data.list.len() == 2);
    }
}

test "reader - vectors" {
    const allocator = std.testing.allocator;

    // Test empty vector
    {
        const source = "[]";
        var tok = Tokenizer.init(allocator, source);
        var reader = try Reader.init(allocator, &tok);
        var value = try reader.read();
        defer {
            value.deinit(allocator);
            allocator.destroy(value);
        }

        try std.testing.expect(value.type == .vector);
        try std.testing.expect(value.data.vector.len() == 0);
    }

    // Test vector with elements
    {
        const source = "[%x %y]";
        var tok = Tokenizer.init(allocator, source);
        var reader = try Reader.init(allocator, &tok);
        var value = try reader.read();
        defer {
            value.deinit(allocator);
            allocator.destroy(value);
        }

        try std.testing.expect(value.type == .vector);
        try std.testing.expect(value.data.vector.len() == 2);
        try std.testing.expect(value.data.vector.at(0).type == .value_id);
        try std.testing.expectEqualStrings("%x", value.data.vector.at(0).data.atom);
    }
}

test "reader - maps" {
    const allocator = std.testing.allocator;

    // Test empty map
    {
        const source = "{}";
        var tok = Tokenizer.init(allocator, source);
        var reader = try Reader.init(allocator, &tok);
        var value = try reader.read();
        defer {
            value.deinit(allocator);
            allocator.destroy(value);
        }

        try std.testing.expect(value.type == .map);
        try std.testing.expect(value.data.map.len() == 0);
    }

    // Test map with key-value pairs
    {
        const source = "{ :value 42 :name \"test\" }";
        var tok = Tokenizer.init(allocator, source);
        var reader = try Reader.init(allocator, &tok);
        var value = try reader.read();
        defer {
            value.deinit(allocator);
            allocator.destroy(value);
        }

        try std.testing.expect(value.type == .map);
        try std.testing.expect(value.data.map.len() == 4); // Flat list of k,v,k,v
        try std.testing.expect(value.data.map.at(0).type == .keyword);
        try std.testing.expectEqualStrings(":value", value.data.map.at(0).data.atom);
        try std.testing.expect(value.data.map.at(1).type == .number);
    }
}

test "reader - type and attr expressions" {
    const allocator = std.testing.allocator;

    // Test type expression
    {
        const source = "!i32";
        var tok = Tokenizer.init(allocator, source);
        var reader = try Reader.init(allocator, &tok);
        var value = try reader.read();
        defer {
            value.deinit(allocator);
            allocator.destroy(value);
        }

        try std.testing.expect(value.type == .type_expr);
        try std.testing.expect(value.data.type_expr.type == .identifier);
        try std.testing.expectEqualStrings("i32", value.data.type_expr.data.atom);
    }

    // Test attribute expression
    {
        const source = "#(int 42)";
        var tok = Tokenizer.init(allocator, source);
        var reader = try Reader.init(allocator, &tok);
        var value = try reader.read();
        defer {
            value.deinit(allocator);
            allocator.destroy(value);
        }

        try std.testing.expect(value.type == .attr_expr);
        try std.testing.expect(value.data.attr_expr.type == .list);
        try std.testing.expect(value.data.attr_expr.data.list.len() == 2);
    }
}

test "reader - complex expression" {
    const allocator = std.testing.allocator;

    const source =
        \\(operation
        \\  (name arith.constant)
        \\  (result-bindings [%c0])
        \\  (result-types !i32)
        \\  (attributes { :value #(int 42) }))
    ;

    var tok = Tokenizer.init(allocator, source);
    var reader = try Reader.init(allocator, &tok);
    var value = try reader.read();
    defer {
        value.deinit(allocator);
        allocator.destroy(value);
    }

    try std.testing.expect(value.type == .list);
    try std.testing.expect(value.data.list.len() == 5);
    try std.testing.expect(value.data.list.at(0).type == .identifier);
    try std.testing.expectEqualStrings("operation", value.data.list.at(0).data.atom);
}

test "reader - read all" {
    const allocator = std.testing.allocator;

    const source = "1 2 (3 4) 5";
    var tok = Tokenizer.init(allocator, source);
    var reader = try Reader.init(allocator, &tok);
    var values = try reader.readAll();
    defer {
        const slice = values.slice();
        for (slice) |v| {
            v.deinit(allocator);
            allocator.destroy(v);
        }
        values.deinit();
    }

    try std.testing.expect(values.len() == 4);
    try std.testing.expect(values.at(0).type == .number);
    try std.testing.expect(values.at(2).type == .list);
}
