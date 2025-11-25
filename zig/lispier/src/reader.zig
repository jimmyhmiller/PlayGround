const std = @import("std");
const tokenizer = @import("tokenizer.zig");
const reader_types = @import("reader_types.zig");
const mlir_integration = @import("mlir_integration.zig");

const Token = tokenizer.Token;
const TokenType = tokenizer.TokenType;
const Value = reader_types.Value;
const Symbol = reader_types.Symbol;
const Namespace = reader_types.Namespace;

pub const ReaderError = error{
    UnexpectedToken,
    UnmatchedParen,
    UnmatchedBracket,
    UnmatchedBrace,
    MapKeyMustBeKeyword,
    MapMissingValue,
    OutOfMemory,
    InvalidCharacter,
    AmbiguousSymbol,
};

/// Namespace scope for tracking imports
pub const NamespaceScope = struct {
    /// Required dialects: name -> namespace
    required: std.StringHashMap(*Namespace),
    /// Used dialects (for unqualified access)
    used: std.ArrayList(*Namespace),
    allocator: std.mem.Allocator,
    /// Optional dialect registry for operation validation
    dialect_registry: ?*mlir_integration.DialectRegistry,
    /// Default namespace for unresolved symbols
    user_namespace: *Namespace,

    pub fn init(allocator: std.mem.Allocator) NamespaceScope {
        return NamespaceScope.initWithRegistry(allocator, null);
    }

    pub fn initWithRegistry(allocator: std.mem.Allocator, registry: ?*mlir_integration.DialectRegistry) NamespaceScope {
        // Create the default 'user' namespace
        const user_ns = Namespace.init(allocator, "user", null) catch unreachable;

        return .{
            .required = std.StringHashMap(*Namespace).init(allocator),
            .used = std.ArrayList(*Namespace){},
            .allocator = allocator,
            .dialect_registry = registry,
            .user_namespace = user_ns,
        };
    }

    pub fn deinit(self: *NamespaceScope) void {
        var it = self.required.iterator();
        while (it.next()) |entry| {
            self.allocator.free(entry.key_ptr.*);
            entry.value_ptr.*.deinit(self.allocator);
        }
        self.required.deinit();

        for (self.used.items) |ns| {
            ns.deinit(self.allocator);
        }
        self.used.deinit(self.allocator);

        // Free the user namespace
        self.user_namespace.deinit(self.allocator);
    }

    /// Add a required dialect (require-dialect)
    pub fn requireDialect(self: *NamespaceScope, name: []const u8, alias: ?[]const u8) !void {
        const ns = try Namespace.init(self.allocator, name, alias);
        const key = try self.allocator.dupe(u8, name);
        try self.required.put(key, ns);
    }

    /// Add a used dialect (use-dialect)
    pub fn useDialect(self: *NamespaceScope, name: []const u8) !void {
        const ns = try Namespace.init(self.allocator, name, null);
        try self.used.append(self.allocator, ns);
    }

    /// Resolve a symbol to its namespace
    /// Handles: alias/name, namespace.name, and bare names
    /// Returns the user namespace for local/unresolved symbols
    pub fn resolveSymbol(self: *NamespaceScope, symbol_text: []const u8) !*Namespace {
        // Check for slash notation: alias/name
        if (std.mem.indexOf(u8, symbol_text, "/")) |slash_pos| {
            const alias = symbol_text[0..slash_pos];
            // Find namespace with this alias
            var it = self.required.iterator();
            while (it.next()) |entry| {
                const ns = entry.value_ptr.*;
                if (ns.alias) |ns_alias| {
                    if (std.mem.eql(u8, ns_alias, alias)) {
                        return ns;
                    }
                }
            }
            // Alias not found - treat as user namespace symbol
            return self.user_namespace;
        }

        // Check for dot notation: namespace.name
        if (std.mem.indexOf(u8, symbol_text, ".")) |dot_pos| {
            const namespace_name = symbol_text[0..dot_pos];

            // Check if already in required
            if (self.required.get(namespace_name)) |ns| {
                return ns;
            }

            // Check if in used (from use-dialect)
            for (self.used.items) |ns| {
                if (std.mem.eql(u8, ns.name, namespace_name)) {
                    return ns;
                }
            }

            // Not found anywhere - create namespace on-the-fly
            // Dot notation always implies a namespace, even if not explicitly loaded
            const ns = try Namespace.init(self.allocator, namespace_name, null);
            const key = try self.allocator.dupe(u8, namespace_name);
            try self.required.put(key, ns);
            return ns;
        }

        // Bare name - search used dialects
        // Check which dialect actually contains this operation
        if (self.used.items.len > 0) {
            if (self.dialect_registry) |registry| {
                // Try to find which dialect contains this operation
                var found_namespaces = std.ArrayList(*Namespace){};
                defer found_namespaces.deinit(self.allocator);

                for (self.used.items) |ns| {
                    const has_op = registry.validateOperation(ns.name, symbol_text) catch false;
                    if (has_op) {
                        try found_namespaces.append(self.allocator, ns);
                    }
                }

                // If found in exactly one dialect, return it
                if (found_namespaces.items.len == 1) {
                    return found_namespaces.items[0];
                }

                // If found in multiple dialects, this is an error
                if (found_namespaces.items.len > 1) {
                    std.debug.print("ERROR: Symbol '{s}' is ambiguous - found in dialects: ", .{symbol_text});
                    for (found_namespaces.items, 0..) |ns, i| {
                        if (i > 0) std.debug.print(", ", .{});
                        std.debug.print("{s}", .{ns.name});
                    }
                    std.debug.print("\n", .{});
                    return ReaderError.AmbiguousSymbol;
                }

                // Not found in any dialect - treat as local symbol
                return self.user_namespace;
            } else {
                // No registry available - can't validate, treat as local symbol
                return self.user_namespace;
            }
        }

        // No used dialects - treat as local symbol
        return self.user_namespace;
    }

    /// Get the unqualified part of a symbol (after / or .)
    pub fn getUnqualifiedName(self: *NamespaceScope, symbol_text: []const u8) []const u8 {
        _ = self;
        if (std.mem.indexOf(u8, symbol_text, "/")) |slash_pos| {
            return symbol_text[slash_pos + 1 ..];
        }
        if (std.mem.indexOf(u8, symbol_text, ".")) |dot_pos| {
            return symbol_text[dot_pos + 1 ..];
        }
        return symbol_text;
    }
};

pub const Reader = struct {
    tokens: []const Token,
    current: usize,
    allocator: std.mem.Allocator,
    namespace_scope: NamespaceScope,

    pub fn init(allocator: std.mem.Allocator, tokens: []const Token) Reader {
        return Reader.initWithRegistry(allocator, tokens, null);
    }

    pub fn initWithRegistry(allocator: std.mem.Allocator, tokens: []const Token, registry: ?*mlir_integration.DialectRegistry) Reader {
        return .{
            .tokens = tokens,
            .current = 0,
            .allocator = allocator,
            .namespace_scope = NamespaceScope.initWithRegistry(allocator, registry),
        };
    }

    pub fn deinit(self: *Reader) void {
        self.namespace_scope.deinit();
    }

    pub fn read(self: *Reader) !std.ArrayList(*Value) {
        var values = std.ArrayList(*Value){};
        errdefer {
            for (values.items) |v| {
                v.deinit();
            }
            values.deinit(self.allocator);
        }

        while (!self.isAtEnd()) {
            const val = try self.readValue();
            if (val) |v| {
                try values.append(self.allocator, v);
            }
        }

        return values;
    }

    fn readValue(self: *Reader) ReaderError!?*Value {
        const token = self.advance();

        return switch (token.type) {
            .LeftParen => try self.readList(),
            .LeftBracket => try self.readVector(),
            .LeftBrace => try self.readMap(),
            .Symbol => try self.readSymbol(token),
            .String => try self.readString(token),
            .Number => try self.readNumber(token),
            .Keyword => try Value.createKeyword(self.allocator, token.lexeme),
            .BlockLabel => try self.readBlockLabel(token),
            .Colon => blk: {
                // Standalone ":" is used for type annotations and typed block args.
                const sym = try Symbol.init(self.allocator, ":", null);
                break :blk try Value.createSymbol(self.allocator, sym);
            },
            .EOF => null,
            .RightParen, .RightBracket, .RightBrace => return error.UnexpectedToken,
        };
    }

    fn readList(self: *Reader) !*Value {
        const list = try Value.createList(self.allocator);
        errdefer list.deinit();

        while (!self.check(.RightParen) and !self.isAtEnd()) {
            const val = try self.readValue();
            if (val) |v| {
                try list.listAppend(v);
            }
        }

        if (!self.check(.RightParen)) {
            return error.UnmatchedParen;
        }
        _ = self.advance(); // consume )

        // Check if this is a require-dialect or use-dialect form
        if (list.data.list.items.len > 0) {
            if (list.data.list.items[0].type == .Symbol) {
                const first_sym = list.data.list.items[0].data.symbol;
                if (std.mem.eql(u8, first_sym.name, "require-dialect")) {
                    try self.handleRequireDialect(list);
                } else if (std.mem.eql(u8, first_sym.name, "use-dialect")) {
                    try self.handleUseDialect(list);
                }
            }
        }

        return list;
    }

    fn readVector(self: *Reader) !*Value {
        const vector = try Value.createVector(self.allocator);
        errdefer vector.deinit();

        while (!self.check(.RightBracket) and !self.isAtEnd()) {
            const val = try self.readValue();
            if (val) |v| {
                try vector.vectorAppend(v);
            }
        }

        if (!self.check(.RightBracket)) {
            return error.UnmatchedBracket;
        }
        _ = self.advance(); // consume ]

        return vector;
    }

    fn readMap(self: *Reader) !*Value {
        const map = try Value.createMap(self.allocator);
        errdefer map.deinit();

        while (!self.check(.RightBrace) and !self.isAtEnd()) {
            // Read key (must be keyword)
            const key_val = try self.readValue();
            if (key_val == null) break;
            const key = key_val.?;
            errdefer key.deinit();

            if (key.type != .Keyword) {
                return error.MapKeyMustBeKeyword;
            }

            // Skip the leading : in keyword
            const key_str = key.data.keyword[1..];

            // Read value
            if (self.check(.RightBrace)) {
                return error.MapMissingValue;
            }
            const val = try self.readValue();
            if (val == null) {
                return error.MapMissingValue;
            }

            try map.mapPut(key_str, val.?);
            key.deinit(); // We've copied the key string
        }

        if (!self.check(.RightBrace)) {
            return error.UnmatchedBrace;
        }
        _ = self.advance(); // consume }

        return map;
    }

    fn readSymbol(self: *Reader, token: Token) !*Value {
        // Check for special boolean values
        if (std.mem.eql(u8, token.lexeme, "true")) {
            return Value.createBoolean(self.allocator, true);
        }
        if (std.mem.eql(u8, token.lexeme, "false")) {
            return Value.createBoolean(self.allocator, false);
        }
        if (std.mem.eql(u8, token.lexeme, "nil")) {
            return Value.createNil(self.allocator);
        }

        // Resolve namespace for the symbol
        const namespace = try self.namespace_scope.resolveSymbol(token.lexeme);
        const unqualified_name = self.namespace_scope.getUnqualifiedName(token.lexeme);

        const symbol = try Symbol.init(self.allocator, unqualified_name, namespace);

        // Mark how this symbol was qualified
        if (std.mem.indexOf(u8, token.lexeme, "/")) |_| {
            symbol.uses_alias = true;
        } else if (std.mem.indexOf(u8, token.lexeme, ".")) |_| {
            symbol.uses_dot = true;
        }

        return Value.createSymbol(self.allocator, symbol);
    }

    fn readString(self: *Reader, token: Token) !*Value {
        // Remove quotes and handle escape sequences
        const content = token.lexeme[1 .. token.lexeme.len - 1];
        var result = std.ArrayList(u8){};
        defer result.deinit(self.allocator);

        var i: usize = 0;
        while (i < content.len) : (i += 1) {
            if (content[i] == '\\' and i + 1 < content.len) {
                i += 1;
                const escaped = switch (content[i]) {
                    'n' => '\n',
                    'r' => '\r',
                    't' => '\t',
                    '\\' => '\\',
                    '"' => '"',
                    '0' => 0,
                    else => content[i],
                };
                try result.append(self.allocator, escaped);
            } else {
                try result.append(self.allocator, content[i]);
            }
        }

        return Value.createString(self.allocator, result.items);
    }

    fn readNumber(self: *Reader, token: Token) !*Value {
        const num = try std.fmt.parseFloat(f64, token.lexeme);
        return Value.createNumber(self.allocator, num);
    }

    fn readBlockLabel(self: *Reader, token: Token) !*Value {
        // Block labels are represented as symbols with the ^ prefix
        const symbol = try Symbol.init(self.allocator, token.lexeme, null);
        return Value.createSymbol(self.allocator, symbol);
    }

    fn handleRequireDialect(self: *Reader, list: *Value) !void {
        // (require-dialect arith) or (require-dialect [arith :as a])
        for (list.data.list.items[1..]) |item| {
            switch (item.type) {
                .Symbol => {
                    // Simple require: (require-dialect arith)
                    const name = item.data.symbol.name;
                    try self.namespace_scope.requireDialect(name, null);
                },
                .Vector => {
                    // Aliased require: (require-dialect [arith :as a])
                    if (item.data.vector.items.len >= 3) {
                        if (item.data.vector.items[0].type == .Symbol and
                            item.data.vector.items[1].type == .Keyword and
                            item.data.vector.items[2].type == .Symbol)
                        {
                            const name = item.data.vector.items[0].data.symbol.name;
                            const kw = item.data.vector.items[1].data.keyword;
                            const alias = item.data.vector.items[2].data.symbol.name;

                            if (std.mem.eql(u8, kw, ":as")) {
                                try self.namespace_scope.requireDialect(name, alias);
                            }
                        }
                    }
                },
                else => {},
            }
        }
    }

    fn handleUseDialect(self: *Reader, list: *Value) !void {
        // (use-dialect arith)
        for (list.data.list.items[1..]) |item| {
            if (item.type == .Symbol) {
                const name = item.data.symbol.name;
                try self.namespace_scope.useDialect(name);
            }
        }
    }

    fn advance(self: *Reader) Token {
        if (!self.isAtEnd()) {
            const token = self.tokens[self.current];
            self.current += 1;
            return token;
        }
        return self.tokens[self.tokens.len - 1]; // Return EOF
    }

    fn check(self: *Reader, token_type: TokenType) bool {
        if (self.isAtEnd()) return false;
        return self.tokens[self.current].type == token_type;
    }

    fn isAtEnd(self: *Reader) bool {
        return self.current >= self.tokens.len or self.tokens[self.current].type == .EOF;
    }
};

test "reader basic list" {
    const allocator = std.testing.allocator;

    var tok = tokenizer.Tokenizer.init(allocator, "(1 2 3)");
    var tokens = try tok.tokenize();
    defer tokens.deinit(allocator);

    var reader = Reader.init(allocator, tokens.items);
    defer reader.deinit();

    var values = try reader.read();
    defer {
        for (values.items) |v| {
            v.deinit();
        }
        values.deinit(allocator);
    }

    try std.testing.expectEqual(@as(usize, 1), values.items.len);
    try std.testing.expectEqual(reader_types.ValueType.List, values.items[0].type);
    try std.testing.expectEqual(@as(usize, 3), values.items[0].data.list.items.len);
}

test "reader nested structures" {
    const allocator = std.testing.allocator;

    var tok = tokenizer.Tokenizer.init(allocator, "([1 2] {:key 42})");
    var tokens = try tok.tokenize();
    defer tokens.deinit(allocator);

    var reader = Reader.init(allocator, tokens.items);
    defer reader.deinit();

    var values = try reader.read();
    defer {
        for (values.items) |v| {
            v.deinit();
        }
        values.deinit(allocator);
    }

    try std.testing.expectEqual(@as(usize, 1), values.items.len);
    const list = values.items[0];
    try std.testing.expectEqual(@as(usize, 2), list.data.list.items.len);
    try std.testing.expectEqual(reader_types.ValueType.Vector, list.data.list.items[0].type);
    try std.testing.expectEqual(reader_types.ValueType.Map, list.data.list.items[1].type);
}

test "reader symbols with namespaces" {
    const allocator = std.testing.allocator;

    var tok = tokenizer.Tokenizer.init(allocator, "(require-dialect [arith :as a]) (a/addi 1 2)");
    var tokens = try tok.tokenize();
    defer tokens.deinit(allocator);

    var reader = Reader.init(allocator, tokens.items);
    defer reader.deinit();

    var values = try reader.read();
    defer {
        for (values.items) |v| {
            v.deinit();
        }
        values.deinit(allocator);
    }

    try std.testing.expectEqual(@as(usize, 2), values.items.len);

    // Second value should be (a/addi 1 2)
    const list = values.items[1];
    try std.testing.expectEqual(reader_types.ValueType.List, list.type);

    // First item should be symbol with namespace
    const sym = list.data.list.items[0];
    try std.testing.expectEqual(reader_types.ValueType.Symbol, sym.type);
    try std.testing.expectEqualStrings("addi", sym.data.symbol.name);
    try std.testing.expect(sym.data.symbol.namespace != null);
    try std.testing.expect(sym.data.symbol.uses_alias);
}

test "reader strings with escapes" {
    const allocator = std.testing.allocator;

    var tok = tokenizer.Tokenizer.init(allocator, "\"hello\\nworld\"");
    var tokens = try tok.tokenize();
    defer tokens.deinit(allocator);

    var reader = Reader.init(allocator, tokens.items);
    defer reader.deinit();

    var values = try reader.read();
    defer {
        for (values.items) |v| {
            v.deinit();
        }
        values.deinit(allocator);
    }

    try std.testing.expectEqual(@as(usize, 1), values.items.len);
    try std.testing.expectEqual(reader_types.ValueType.String, values.items[0].type);
    try std.testing.expectEqualStrings("hello\nworld", values.items[0].data.string);
}

test "reader booleans and nil" {
    const allocator = std.testing.allocator;

    var tok = tokenizer.Tokenizer.init(allocator, "true false nil");
    var tokens = try tok.tokenize();
    defer tokens.deinit(allocator);

    var reader = Reader.init(allocator, tokens.items);
    defer reader.deinit();

    var values = try reader.read();
    defer {
        for (values.items) |v| {
            v.deinit();
        }
        values.deinit(allocator);
    }

    try std.testing.expectEqual(@as(usize, 3), values.items.len);
    try std.testing.expectEqual(reader_types.ValueType.Boolean, values.items[0].type);
    try std.testing.expectEqual(true, values.items[0].data.boolean);
    try std.testing.expectEqual(reader_types.ValueType.Boolean, values.items[1].type);
    try std.testing.expectEqual(false, values.items[1].data.boolean);
    try std.testing.expectEqual(reader_types.ValueType.Nil, values.items[2].type);
}

test "reader rejects maps with non-keyword keys" {
    const allocator = std.testing.allocator;

    // Status doc claims proper map handling and error reporting; verify we actually reject bad keys.
    var tok = tokenizer.Tokenizer.init(allocator, "{foo 1}");
    var tokens = try tok.tokenize();
    defer tokens.deinit(allocator);

    var reader = Reader.init(allocator, tokens.items);
    defer reader.deinit();

    try std.testing.expectError(ReaderError.MapKeyMustBeKeyword, reader.read());
}

test "reader reports missing map values" {
    const allocator = std.testing.allocator;

    var tok = tokenizer.Tokenizer.init(allocator, "{:foo}");
    var tokens = try tok.tokenize();
    defer tokens.deinit(allocator);

    var reader = Reader.init(allocator, tokens.items);
    defer reader.deinit();

    try std.testing.expectError(ReaderError.MapMissingValue, reader.read());
}
