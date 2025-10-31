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
    type, // i32, !llvm.ptr - any type with ! prefix (stored as string)
    function_type, // (!function (inputs ...) (results ...)) - structured function type
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
        type: []const u8, // For type expressions - stores full type string like "i32", "!llvm.ptr"
        function_type: struct {
            inputs: vector.PersistentVector(*Value),
            results: vector.PersistentVector(*Value),
        }, // For function types (!function (inputs ...) (results ...))
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
            .type => {
                // type is just a string, no cleanup needed
            },
            .function_type => {
                // Clean up input and result type vectors
                var mut_inputs = self.data.function_type.inputs;
                var mut_results = self.data.function_type.results;

                const inputs_slice = mut_inputs.slice();
                for (inputs_slice) |input| {
                    input.deinit(allocator);
                    allocator.destroy(input);
                }

                const results_slice = mut_results.slice();
                for (results_slice) |result| {
                    result.deinit(allocator);
                    allocator.destroy(result);
                }

                mut_inputs.deinit();
                mut_results.deinit();
            },
            .attr_expr => {
                // attr_expr now uses .atom which is owned by tokenizer, no cleanup needed
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

    /// Print the value back to lisp syntax
    pub fn print(self: *const Value, writer: anytype) !void {
        switch (self.type) {
            // Atoms - just print their lexeme
            .identifier, .number, .string, .value_id, .block_id, .symbol, .keyword => {
                try writer.writeAll(self.data.atom);
            },
            .true_lit => {
                try writer.writeAll("true");
            },
            .false_lit => {
                try writer.writeAll("false");
            },

            // Collections
            .list => {
                try writer.writeAll("(");
                const slice = self.data.list.slice();
                for (slice, 0..) |child, i| {
                    if (i > 0) try writer.writeAll(" ");
                    try child.print(writer);
                }
                try writer.writeAll(")");
            },
            .vector => {
                try writer.writeAll("[");
                const slice = self.data.vector.slice();
                for (slice, 0..) |child, i| {
                    if (i > 0) try writer.writeAll(" ");
                    try child.print(writer);
                }
                try writer.writeAll("]");
            },
            .map => {
                try writer.writeAll("{");
                const slice = self.data.map.slice();
                for (slice, 0..) |child, i| {
                    if (i > 0) try writer.writeAll(" ");
                    try child.print(writer);
                }
                try writer.writeAll("}");
            },

            // Type expression
            .type => {
                try writer.writeAll(self.data.type);
            },

            // Function type: (!function (inputs ...) (results ...))
            .function_type => {
                try writer.writeAll("(!function (inputs");
                const input_slice = self.data.function_type.inputs.slice();
                for (input_slice) |input| {
                    try writer.writeAll(" ");
                    try input.print(writer);
                }
                try writer.writeAll(") (results");
                const result_slice = self.data.function_type.results.slice();
                for (result_slice) |result| {
                    try writer.writeAll(" ");
                    try result.print(writer);
                }
                try writer.writeAll("))");
            },

            // Attribute expression: #...
            .attr_expr => {
                try writer.writeAll("#");
                try self.data.attr_expr.print(writer);
            },

            // Typed literal: (: value type)
            .has_type => {
                try writer.writeAll("(: ");
                try self.data.has_type.value.print(writer);
                try writer.writeAll(" ");
                try self.data.has_type.type_expr.print(writer);
                try writer.writeAll(")");
            },
        }
    }
};

/// Errors that can occur during reading
pub const ReaderError = error{
    UnexpectedEOF,
    UnexpectedClosingDelimiter,
    UnexpectedDot,
    UnterminatedList,
    ExpectedFunctionType,
    ExpectedTypeIdentifier,
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
                // Type markers like !llvm.array<10 x i8> are now scanned as opaque tokens
                // using bracket-aware scanning (similar to attr_marker)
                // The lexeme includes the '!', which we keep as part of the type string
                const type_content = tok.lexeme;

                // Check if this is a function type (!function ...)
                // Function types still need special handling as they have structure
                if (type_content.len > 1 and std.mem.startsWith(u8, type_content, "!function")) {
                    // This is a function type - we need to parse it specially
                    // But function types should be written as (!function ...) not !function(...)
                    // So this shouldn't happen with bracket-aware scanning
                    // For now, treat it as a simple type string
                    const value = try self.allocator.create(Value);
                    value.* = Value{
                        .type = .type,
                        .data = .{ .type = type_content },
                    };
                    try self.advance();
                    return value;
                }

                // For all other dialect types (simple or complex with brackets)
                // Store the entire type string including the ! prefix
                const value = try self.allocator.create(Value);
                value.* = Value{
                    .type = .type,
                    .data = .{ .type = type_content },
                };
                try self.advance();
                return value;
            },
            .attr_marker => {
                // Attribute markers like #arith.overflow<none> are now scanned as opaque tokens
                // We need to create a nested Value for the attribute content
                // The lexeme includes the '#', so we need to strip it
                const attr_content = if (tok.lexeme.len > 1 and tok.lexeme[0] == '#')
                    tok.lexeme[1..]
                else
                    tok.lexeme;

                // Create inner value for the attribute content
                const inner_value = try self.allocator.create(Value);
                inner_value.* = Value{
                    .type = .identifier,
                    .data = .{ .atom = attr_content },
                };

                // Create the attr_expr value wrapping the inner value
                const value = try self.allocator.create(Value);
                value.* = Value{
                    .type = .attr_expr,
                    .data = .{ .attr_expr = inner_value },
                };
                try self.advance();
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
            vec = try vec.push(elem);
            // Note: intermediate vectors are leaked, but this is fine when using an ArenaAllocator
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

        // Check for (!function ...) pattern - convert to function_type
        if (value_type == .list and vec.len() >= 3) {
            const slice = vec.slice();
            const first = slice[0];

            // Check if first element is !function type
            if (first.type == .type and std.mem.eql(u8, first.data.type, "!function")) {
                // This is a function type: (!function (inputs ...) (results ...))
                if (vec.len() != 3) return error.ExpectedFunctionType;

                const inputs_list_value = slice[1];
                const results_list_value = slice[2];

                if (inputs_list_value.type != .list or results_list_value.type != .list) {
                    return error.ExpectedFunctionType;
                }

                const inputs_list = inputs_list_value.data.list;
                const results_list = results_list_value.data.list;

                // Check that first element is "inputs" and "results"
                if (inputs_list.len() == 0 or results_list.len() == 0) return error.ExpectedFunctionType;

                const inputs_kw = inputs_list.at(0);
                const results_kw = results_list.at(0);

                if (inputs_kw.type != .identifier or !std.mem.eql(u8, inputs_kw.data.atom, "inputs")) {
                    return error.ExpectedFunctionType;
                }
                if (results_kw.type != .identifier or !std.mem.eql(u8, results_kw.data.atom, "results")) {
                    return error.ExpectedFunctionType;
                }

                // Extract input and result types (skip the "inputs" and "results" keywords)
                var input_types = vector.PersistentVector(*Value).init(self.allocator, null);
                for (1..inputs_list.len()) |i| {
                    input_types = try input_types.push(inputs_list.at(i));
                }

                var result_types = vector.PersistentVector(*Value).init(self.allocator, null);
                for (1..results_list.len()) |i| {
                    result_types = try result_types.push(results_list.at(i));
                }

                // Clean up the temporary structures
                first.deinit(self.allocator);
                self.allocator.destroy(first);
                inputs_list_value.deinit(self.allocator);
                self.allocator.destroy(inputs_list_value);
                results_list_value.deinit(self.allocator);
                self.allocator.destroy(results_list_value);
                var mut_vec = vec;
                mut_vec.deinit();

                const value = try self.allocator.create(Value);
                value.* = Value{
                    .type = .function_type,
                    .data = .{ .function_type = .{
                        .inputs = input_types,
                        .results = result_types,
                    } },
                };
                return value;
            }
        }

        // Check for (: value type) pattern - only for lists
        if (value_type == .list and vec.len() == 3) {
            const slice = vec.slice();
            const first = slice[0];

            // Check if first element is ':' identifier
            if (first.type == .identifier and std.mem.eql(u8, first.data.atom, ":")) {
                // This is a typed literal: (: value type)
                const val = slice[1];
                var type_val = slice[2];

                // Convert identifier types to .type (e.g., i32 -> .type with data "i32")
                if (type_val.type == .identifier) {
                    const converted_type = try self.allocator.create(Value);
                    converted_type.* = Value{
                        .type = .type,
                        .data = .{ .type = type_val.data.atom },
                    };
                    // Free the old identifier value
                    type_val.deinit(self.allocator);
                    self.allocator.destroy(type_val);
                    type_val = converted_type;
                }

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
            vec = try vec.push(value);
            // Note: intermediate vectors are leaked, but this is fine when using an ArenaAllocator
        }

        return vec;
    }
};
