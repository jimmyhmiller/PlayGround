//! MLIR Parser
//! Implements a recursive descent parser following the grammar in grammar.ebnf
//! Each parsing function is annotated with the corresponding grammar rule

const std = @import("std");
const lexer = @import("lexer.zig");
const ast = @import("ast.zig");

const Lexer = lexer.Lexer;
const Token = lexer.Token;
const TokenType = lexer.TokenType;

pub const ParseError = error{
    UnexpectedToken,
    ExpectedToken,
    OutOfMemory,
    InvalidType,
    InvalidOperation,
    InvalidAttribute,
    Overflow,
    InvalidCharacter,
};

pub const Parser = struct {
    lexer: *Lexer,
    allocator: std.mem.Allocator,
    current: Token,
    previous: Token,

    pub fn init(allocator: std.mem.Allocator, lex: *Lexer) !Parser {
        var parser = Parser{
            .lexer = lex,
            .allocator = allocator,
            .current = undefined,
            .previous = undefined,
        };
        // Prime the parser with the first token
        parser.current = lex.nextToken();
        parser.previous = parser.current;
        return parser;
    }

    pub fn deinit(self: *Parser) void {
        _ = self;
    }

    // Helper methods for parser navigation

    /// Returns the current token without advancing
    fn peek(self: *Parser) Token {
        return self.current;
    }

    /// Returns the previous token
    fn peekPrevious(self: *Parser) Token {
        return self.previous;
    }

    /// Advances to the next token and returns the previous one
    fn advance(self: *Parser) Token {
        self.previous = self.current;
        self.current = self.lexer.nextToken();
        return self.previous;
    }

    /// Checks if current token is of given type without consuming
    fn check(self: *Parser, token_type: TokenType) bool {
        return self.current.type == token_type;
    }

    /// Consumes current token if it matches the given type
    fn match(self: *Parser, token_types: []const TokenType) bool {
        for (token_types) |token_type| {
            if (self.check(token_type)) {
                _ = self.advance();
                return true;
            }
        }
        return false;
    }

    /// Consumes current token if it matches, otherwise returns error
    fn expect(self: *Parser, token_type: TokenType) !Token {
        if (self.check(token_type)) {
            return self.advance();
        }
        std.debug.print("Expected {s}, but got {s} at line {}, column {}\n", .{
            @tagName(token_type),
            @tagName(self.current.type),
            self.current.line,
            self.current.column,
        });
        return ParseError.ExpectedToken;
    }

    /// Checks if we're at the end of input
    fn isAtEnd(self: *Parser) bool {
        return self.current.type == .eof;
    }

    /// Reports an error at the current token
    fn reportError(self: *Parser, message: []const u8) void {
        std.debug.print("Parse error at line {}, column {}: {s}\n", .{
            self.current.line,
            self.current.column,
            message,
        });
    }

    // Grammar: toplevel ::= (operation | attribute-alias-def | type-alias-def)*
    pub fn parseModule(self: *Parser) !ast.Module {
        var operations: std.ArrayList(ast.Operation) = .empty;
        errdefer operations.deinit(self.allocator);

        var type_aliases: std.ArrayList(ast.TypeAliasDef) = .empty;
        errdefer type_aliases.deinit(self.allocator);

        var attribute_aliases: std.ArrayList(ast.AttributeAliasDef) = .empty;
        errdefer attribute_aliases.deinit(self.allocator);

        while (!self.isAtEnd()) {
            // Check for type alias definition: !alias_name = type
            if (self.check(.type_alias_id)) {
                // Peek ahead to see if there's an '=' after the alias name
                const next_token = self.peekNext();
                if (next_token.type == .equal) {
                    try type_aliases.append(self.allocator, try self.parseTypeAliasDef());
                    continue;
                }
            }
            // Check for attribute alias definition: #alias_name = attr
            if (self.check(.attribute_alias_id)) {
                const next_token = self.peekNext();
                if (next_token.type == .equal) {
                    try attribute_aliases.append(self.allocator, try self.parseAttributeAliasDef());
                    continue;
                }
            }
            // Otherwise, parse as operation
            try operations.append(self.allocator, try self.parseOperation());
        }

        return ast.Module{
            .operations = try operations.toOwnedSlice(self.allocator),
            .type_aliases = try type_aliases.toOwnedSlice(self.allocator),
            .attribute_aliases = try attribute_aliases.toOwnedSlice(self.allocator),
            .allocator = self.allocator,
        };
    }

    fn peekNext(self: *Parser) Token {
        // We need to look ahead to see the next token after current
        // Save the lexer state
        const saved_start = self.lexer.start;
        const saved_current = self.lexer.current;
        const saved_line = self.lexer.line;
        const saved_column = self.lexer.column;

        // Get the next token (this advances the lexer)
        const next_token = self.lexer.nextToken();

        // Restore lexer state
        self.lexer.start = saved_start;
        self.lexer.current = saved_current;
        self.lexer.line = saved_line;
        self.lexer.column = saved_column;

        return next_token;
    }

    // Grammar: operation ::= op-result-list? (generic-operation | custom-operation) trailing-location?
    // NOTE: We only parse generic operations (output from mlir-opt -mlir-print-op-generic)
    pub fn parseOperation(self: *Parser) ParseError!ast.Operation {
        // Check for op-result-list
        // In generic format, operations either start with:
        //   - value-id(s) followed by `=` (has results)
        //   - string-literal (no results)
        var results: ?ast.OpResultList = null;

        // Simple check: if we see value-id, it must be a result binding
        // because generic operations always start with quoted strings like "arith.constant"
        if (self.check(.value_id)) {
            results = try self.parseOpResultList();
        }

        // Parse the operation itself (generic format always uses string literal)
        if (!self.check(.string_literal)) {
            self.reportError("Expected generic operation (string literal). Use mlir-opt -mlir-print-op-generic to convert input.");
            return ParseError.InvalidOperation;
        }

        const generic_op = try self.parseGenericOperation();

        // Check for trailing location
        var location: ?ast.Location = null;
        if (self.match(&.{.kw_loc})) {
            location = try self.parseLocation();
        }

        return ast.Operation{
            .results = results,
            .kind = .{ .generic = generic_op },
            .location = location,
        };
    }

    // Grammar: op-result-list ::= op-result (`,` op-result)* `=`
    fn parseOpResultList(self: *Parser) !ast.OpResultList {
        var results: std.ArrayList(ast.OpResult) = .empty;
        errdefer results.deinit(self.allocator);

        while (true) {
            try results.append(self.allocator, try self.parseOpResult());
            if (!self.match(&.{.comma})) break;
        }

        _ = try self.expect(.equal);

        return ast.OpResultList{
            .results = try results.toOwnedSlice(self.allocator),
        };
    }

    // Grammar: op-result ::= value-id (`:` integer-literal)?
    fn parseOpResult(self: *Parser) !ast.OpResult {
        const value_id = try self.expect(.value_id);

        var num_results: ?u64 = null;
        if (self.match(&.{.colon})) {
            const num_token = try self.expect(.integer_literal);
            num_results = try std.fmt.parseInt(u64, num_token.lexeme, 10);
        }

        return ast.OpResult{
            .value_id = value_id.lexeme,
            .num_results = num_results,
        };
    }

    // Grammar: generic-operation ::= string-literal `(` value-use-list? `)` successor-list?
    //                                 dictionary-properties? region-list? dictionary-attribute?
    //                                 `:` function-type
    fn parseGenericOperation(self: *Parser) !ast.GenericOperation {
        const name_token = try self.expect(.string_literal);
        const name = name_token.lexeme[1..name_token.lexeme.len-1]; // Strip quotes

        _ = try self.expect(.lparen);

        // Parse value-use-list
        var operands: std.ArrayList(ast.ValueUse) = .empty;
        errdefer operands.deinit(self.allocator);

        if (!self.check(.rparen)) {
            try operands.append(self.allocator, try self.parseValueUse());
            while (self.match(&.{.comma})) {
                try operands.append(self.allocator, try self.parseValueUse());
            }
        }

        _ = try self.expect(.rparen);

        // Parse successor-list (optional)
        var successors: []ast.Successor = &[_]ast.Successor{};
        if (self.check(.lbracket)) {
            successors = try self.parseSuccessorList();
        }

        // Grammar: dictionary-properties ::= `<` dictionary-attribute `>`
        var properties: ?ast.DictionaryAttribute = null;
        if (self.check(.langle)) {
            _ = self.advance(); // consume <
            properties = try self.parseDictionaryAttribute();
            _ = try self.expect(.rangle);
        }

        // Parse region-list (optional)
        var regions: []ast.Region = &[_]ast.Region{};
        if (self.check(.lparen)) {
            regions = try self.parseRegionList();
        }

        // Grammar: dictionary-attribute ::= `{` (attribute-entry (`,` attribute-entry)*)? `}`
        var attributes: ?ast.DictionaryAttribute = null;
        if (self.check(.lbrace)) {
            attributes = try self.parseDictionaryAttribute();
        }

        _ = try self.expect(.colon);

        const function_type = try self.parseFunctionType();

        return ast.GenericOperation{
            .name = name,
            .operands = try operands.toOwnedSlice(self.allocator),
            .successors = successors,
            .properties = properties,
            .regions = regions,
            .attributes = attributes,
            .function_type = function_type,
        };
    }


    // Grammar: value-use ::= value-id (`#` decimal-literal)?
    fn parseValueUse(self: *Parser) !ast.ValueUse {
        const value_id = try self.expect(.value_id);

        var result_number: ?u64 = null;
        if (self.match(&.{.hash})) {
            const num_token = try self.expect(.integer_literal);
            result_number = try std.fmt.parseInt(u64, num_token.lexeme, 10);
        }

        return ast.ValueUse{
            .value_id = value_id.lexeme,
            .result_number = result_number,
        };
    }

    // Grammar: successor-list ::= `[` successor (`,` successor)* `]`
    fn parseSuccessorList(self: *Parser) ![]ast.Successor {
        _ = try self.expect(.lbracket);

        var successors: std.ArrayList(ast.Successor) = .empty;
        errdefer successors.deinit(self.allocator);

        if (!self.check(.rbracket)) {
            try successors.append(self.allocator, try self.parseSuccessor());
            while (self.match(&.{.comma})) {
                try successors.append(self.allocator, try self.parseSuccessor());
            }
        }

        _ = try self.expect(.rbracket);

        return try successors.toOwnedSlice(self.allocator);
    }

    // Grammar: successor ::= caret-id (`:` block-arg-list)?
    fn parseSuccessor(self: *Parser) !ast.Successor {
        const caret_id = try self.expect(.caret_id);

        var args: ?ast.BlockArgList = null;
        if (self.match(&.{.colon})) {
            args = try self.parseBlockArgList();
        }

        return ast.Successor{
            .block_id = caret_id.lexeme,
            .args = args,
        };
    }

    // Grammar: block-arg-list ::= `(` value-id-and-type-list? `)`
    fn parseBlockArgList(self: *Parser) !ast.BlockArgList {
        _ = try self.expect(.lparen);

        var args: std.ArrayList(ast.ValueIdAndType) = .empty;
        errdefer args.deinit(self.allocator);

        if (!self.check(.rparen)) {
            try args.append(self.allocator, try self.parseValueIdAndType());
            while (self.match(&.{.comma})) {
                try args.append(self.allocator, try self.parseValueIdAndType());
            }
        }

        _ = try self.expect(.rparen);

        return ast.BlockArgList{
            .args = try args.toOwnedSlice(self.allocator),
        };
    }

    // Grammar: value-id-and-type ::= value-id `:` type
    fn parseValueIdAndType(self: *Parser) !ast.ValueIdAndType {
        const value_id = try self.expect(.value_id);
        _ = try self.expect(.colon);
        const typ = try self.parseType();

        return ast.ValueIdAndType{
            .value_id = value_id.lexeme,
            .type = typ,
        };
    }

    // Grammar: region-list ::= `(` region (`,` region)* `)`
    fn parseRegionList(self: *Parser) ![]ast.Region {
        _ = try self.expect(.lparen);

        var regions: std.ArrayList(ast.Region) = .empty;
        errdefer {
            for (regions.items) |*region| {
                region.deinit(self.allocator);
            }
            regions.deinit(self.allocator);
        }

        if (!self.check(.rparen)) {
            try regions.append(self.allocator, try self.parseRegion());
            while (self.match(&.{.comma})) {
                try regions.append(self.allocator, try self.parseRegion());
            }
        }

        _ = try self.expect(.rparen);

        return try regions.toOwnedSlice(self.allocator);
    }

    // Grammar: region ::= `{` entry-block? block* `}`
    // Grammar: entry-block ::= operation+
    fn parseRegion(self: *Parser) ParseError!ast.Region {
        _ = try self.expect(.lbrace);

        var entry_block: ?[]ast.Operation = null;
        var blocks: std.ArrayList(ast.Block) = .empty;
        errdefer {
            if (entry_block) |ops| {
                for (ops) |*op| {
                    op.deinit(self.allocator);
                }
                self.allocator.free(ops);
            }
            for (blocks.items) |*blk| {
                blk.deinit(self.allocator);
            }
            blocks.deinit(self.allocator);
        }

        // Parse entry-block or blocks
        // Entry blocks start with operations (not block labels)
        // Blocks start with block labels (caret-id)
        while (!self.check(.rbrace)) {
            if (self.check(.caret_id)) {
                // This is a labeled block
                try blocks.append(self.allocator, try self.parseBlock());
            } else {
                // This is part of the entry block
                if (entry_block == null) {
                    var ops: std.ArrayList(ast.Operation) = .empty;
                    errdefer {
                        for (ops.items) |*op| {
                            op.deinit(self.allocator);
                        }
                        ops.deinit(self.allocator);
                    }

                    // Parse operations until we hit a block label or closing brace
                    while (!self.check(.rbrace) and !self.check(.caret_id)) {
                        try ops.append(self.allocator, try self.parseOperation());
                    }

                    entry_block = try ops.toOwnedSlice(self.allocator);
                } else {
                    self.reportError("Unexpected operation after entry block");
                    return ParseError.InvalidOperation;
                }
            }
        }

        _ = try self.expect(.rbrace);

        return ast.Region{
            .entry_block = entry_block,
            .blocks = try blocks.toOwnedSlice(self.allocator),
        };
    }

    // Grammar: block ::= block-label operation+
    // Grammar: block-label ::= block-id block-arg-list? `:`
    fn parseBlock(self: *Parser) ParseError!ast.Block {
        const label = try self.parseBlockLabel();

        var operations: std.ArrayList(ast.Operation) = .empty;
        errdefer {
            for (operations.items) |*op| {
                op.deinit(self.allocator);
            }
            operations.deinit(self.allocator);
        }

        // Parse operations until we hit a block label or closing brace
        while (!self.check(.rbrace) and !self.check(.caret_id)) {
            try operations.append(self.allocator, try self.parseOperation());
        }

        return ast.Block{
            .label = label,
            .operations = try operations.toOwnedSlice(self.allocator),
        };
    }

    // Grammar: block-label ::= block-id block-arg-list? `:`
    fn parseBlockLabel(self: *Parser) !ast.BlockLabel {
        const caret_id = try self.expect(.caret_id);

        var args: ?ast.BlockArgList = null;
        if (self.check(.lparen)) {
            args = try self.parseBlockArgList();
        }

        _ = try self.expect(.colon);

        return ast.BlockLabel{
            .block_id = caret_id.lexeme,
            .args = args,
        };
    }

    // Grammar: dictionary-attribute ::= `{` (attribute-entry (`,` attribute-entry)*)? `}`
    fn parseDictionaryAttribute(self: *Parser) !ast.DictionaryAttribute {
        _ = try self.expect(.lbrace);

        var entries: std.ArrayList(ast.AttributeEntry) = .empty;
        errdefer entries.deinit(self.allocator);

        if (!self.check(.rbrace)) {
            try entries.append(self.allocator, try self.parseAttributeEntry());
            while (self.match(&.{.comma})) {
                try entries.append(self.allocator, try self.parseAttributeEntry());
            }
        }

        _ = try self.expect(.rbrace);

        return ast.DictionaryAttribute{
            .entries = try entries.toOwnedSlice(self.allocator),
        };
    }

    // Grammar: attribute-entry ::= (bare-id | string-literal) `=` attribute-value
    fn parseAttributeEntry(self: *Parser) !ast.AttributeEntry {
        const name_token = if (self.check(.bare_id))
            try self.expect(.bare_id)
        else
            try self.expect(.string_literal);

        _ = try self.expect(.equal);

        // For now, simplify attribute value parsing by just capturing the raw text
        // until we hit a comma or closing brace
        const start_pos = self.current.lexeme.ptr - self.lexer.source.ptr;
        var depth: usize = 0;

        while (!self.isAtEnd()) {
            if (self.check(.lbrace) or self.check(.lbracket) or self.check(.lparen) or self.check(.langle)) {
                depth += 1;
                _ = self.advance();
            } else if (self.check(.rbrace) or self.check(.rbracket) or self.check(.rparen) or self.check(.rangle)) {
                if (depth == 0) break; // Hit closing brace of dictionary
                depth -= 1;
                _ = self.advance();
            } else if (self.check(.comma) and depth == 0) {
                break; // Hit separator
            } else {
                _ = self.advance();
            }
        }

        const end_pos = self.previous.lexeme.ptr - self.lexer.source.ptr + self.previous.lexeme.len;
        const raw_value = self.lexer.source[start_pos..end_pos];

        // Create a simple builtin attribute with the raw string
        return ast.AttributeEntry{
            .name = name_token.lexeme,
            .value = .{ .builtin = .{ .string = raw_value } },
        };
    }

    fn parseLocation(self: *Parser) !ast.Location {
        _ = try self.expect(.lparen);

        // Simplified: just capture the location string
        const start = self.current;
        var depth: usize = 1;
        while (depth > 0 and !self.isAtEnd()) {
            if (self.check(.lparen)) depth += 1;
            if (self.check(.rparen)) depth -= 1;
            _ = self.advance();
        }

        const source = self.lexer.source[start.lexeme.ptr - self.lexer.source.ptr..self.previous.lexeme.ptr - self.lexer.source.ptr];

        return ast.Location{
            .source = source,
        };
    }

    // Grammar: type-alias-def ::= `!` alias-name `=` type
    fn parseTypeAliasDef(self: *Parser) !ast.TypeAliasDef {
        const alias_token = try self.expect(.type_alias_id);
        _ = try self.expect(.equal);
        const type_value = try self.parseType();

        return ast.TypeAliasDef{
            .alias_name = alias_token.lexeme[1..], // Skip the '!' prefix
            .type = type_value,
        };
    }

    // Grammar: attribute-alias-def ::= `#` alias-name `=` attribute-value
    fn parseAttributeAliasDef(self: *Parser) !ast.AttributeAliasDef {
        const alias_token = try self.expect(.attribute_alias_id);
        _ = try self.expect(.equal);
        const attr_value = try self.parseAttributeValue();

        return ast.AttributeAliasDef{
            .alias_name = alias_token.lexeme[1..], // Skip the '#' prefix
            .value = attr_value,
        };
    }

    // Grammar: type ::= type-alias | dialect-type | builtin-type | function-type
    pub fn parseType(self: *Parser) ParseError!ast.Type {
        // Check for type-alias or dialect-type starting with '!'
        if (self.check(.type_alias_id)) {
            const token = self.peek();
            // Check if this is a pretty dialect type (contains a dot) or type alias
            // Pretty dialect types: !llvm.ptr, !dialect.type
            // Type aliases: !my_alias, !MyType
            if (std.mem.indexOfScalar(u8, token.lexeme, '.')) |_| {
                // This is a pretty dialect type
                _ = self.advance(); // consume the type_alias_id token
                return ast.Type{ .dialect = ast.DialectType{
                    .namespace = token.lexeme[1..], // Skip '!'
                    .body = null,
                }};
            } else {
                // This is a type alias
                _ = self.advance();
                return ast.Type{ .type_alias = token.lexeme[1..] }; // Skip '!'
            }
        }

        // Check for dialect-type: !namespace.type or !namespace<...>
        if (self.check(.exclamation)) {
            return ast.Type{ .dialect = try self.parseDialectType() };
        }

        // Otherwise, parse builtin type
        return ast.Type{ .builtin = try self.parseBuiltinType() };
    }

    // Grammar: dialect-type ::= `!` (opaque-dialect-type | pretty-dialect-type)
    // Grammar: opaque-dialect-type ::= dialect-namespace dialect-type-body
    // Grammar: pretty-dialect-type ::= dialect-namespace `.` pretty-dialect-type-lead-ident dialect-type-body?
    // NOTE: The lexer includes '.' as part of bare_id, so "llvm.ptr" is a single token
    fn parseDialectType(self: *Parser) !ast.DialectType {
        _ = try self.expect(.exclamation);

        const namespace_token = try self.expect(.bare_id);
        var body: ?[]const u8 = null;

        // Check for optional dialect-type-body: <...>
        if (self.check(.langle)) {
            const start = self.current;
            _ = self.advance(); // consume '<'

            // Simplified: just capture everything until matching '>'
            var depth: usize = 1;
            while (depth > 0 and !self.isAtEnd()) {
                if (self.check(.langle)) depth += 1;
                if (self.check(.rangle)) depth -= 1;
                _ = self.advance();
            }

            body = self.lexer.source[start.lexeme.ptr - self.lexer.source.ptr..self.previous.lexeme.ptr - self.lexer.source.ptr + self.previous.lexeme.len];
        }

        return ast.DialectType{
            .namespace = namespace_token.lexeme,
            .body = body,
        };
    }

    // Grammar: integer-type ::= `i` [1-9][0-9]*
    // Grammar: float types, index, etc.
    fn parseBuiltinType(self: *Parser) !ast.BuiltinType {
        const token = self.peek();

        if (token.type != .bare_id) {
            self.reportError("Expected type identifier");
            return ParseError.InvalidType;
        }

        const type_name = token.lexeme;

        // Grammar: index-type ::= `index` (check this first before integer types)
        if (std.mem.eql(u8, type_name, "index")) {
            _ = self.advance();
            return ast.BuiltinType.index;
        }

        // Grammar: signless-integer-type ::= `i` [1-9][0-9]*
        // Grammar: signed-integer-type ::= `si` [1-9][0-9]*
        // Grammar: unsigned-integer-type ::= `ui` [1-9][0-9]*
        else if (type_name.len > 1 and type_name[0] == 'i' and std.ascii.isDigit(type_name[1])) {
            _ = self.advance();
            const width = try std.fmt.parseInt(u64, type_name[1..], 10);
            return ast.BuiltinType{
                .integer = .{
                    .signedness = .signless,
                    .width = width,
                },
            };
        } else if (type_name.len > 2 and std.mem.startsWith(u8, type_name, "si") and std.ascii.isDigit(type_name[2])) {
            _ = self.advance();
            const width = try std.fmt.parseInt(u64, type_name[2..], 10);
            return ast.BuiltinType{
                .integer = .{
                        .signedness = .signed,
                    .width = width,
                },
            };
        } else if (type_name.len > 2 and std.mem.startsWith(u8, type_name, "ui") and std.ascii.isDigit(type_name[2])) {
            _ = self.advance();
            const width = try std.fmt.parseInt(u64, type_name[2..], 10);
            return ast.BuiltinType{
                .integer = .{
                    .signedness = .unsigned,
                    .width = width,
                },
            };
        }

        // Grammar: float types: f16, f32, f64, etc.
        else if (std.mem.eql(u8, type_name, "f16")) {
            _ = self.advance();
            return ast.BuiltinType{ .float = .f16 };
        } else if (std.mem.eql(u8, type_name, "f32")) {
            _ = self.advance();
            return ast.BuiltinType{ .float = .f32 };
        } else if (std.mem.eql(u8, type_name, "f64")) {
            _ = self.advance();
            return ast.BuiltinType{ .float = .f64 };
        } else if (std.mem.eql(u8, type_name, "f80")) {
            _ = self.advance();
            return ast.BuiltinType{ .float = .f80 };
        } else if (std.mem.eql(u8, type_name, "f128")) {
            _ = self.advance();
            return ast.BuiltinType{ .float = .f128 };
        } else if (std.mem.eql(u8, type_name, "bf16")) {
            _ = self.advance();
            return ast.BuiltinType{ .float = .bf16 };
        } else if (std.mem.eql(u8, type_name, "tf32")) {
            _ = self.advance();
            return ast.BuiltinType{ .float = .tf32 };
        }

        // Grammar: none-type ::= `none`
        else if (std.mem.eql(u8, type_name, "none")) {
            _ = self.advance();
            return ast.BuiltinType.none;
        }

        // Grammar: tensor-type ::= `tensor` `<` dimension-list type (`,` encoding)? `>`
        else if (std.mem.eql(u8, type_name, "tensor")) {
            _ = self.advance();
            return ast.BuiltinType{ .tensor = try self.parseTensorType() };
        }

        // Grammar: memref-type ::= `memref` `<` dimension-list type (`,` layout-specification)? (`,` memory-space)? `>`
        else if (std.mem.eql(u8, type_name, "memref")) {
            _ = self.advance();
            return ast.BuiltinType{ .memref = try self.parseMemRefType() };
        }

        // Grammar: vector-type ::= `vector` `<` vector-dim-list vector-element-type `>`
        else if (std.mem.eql(u8, type_name, "vector")) {
            _ = self.advance();
            return ast.BuiltinType{ .vector = try self.parseVectorType() };
        }

        // Grammar: complex-type ::= `complex` `<` type `>`
        else if (std.mem.eql(u8, type_name, "complex")) {
            _ = self.advance();
            return ast.BuiltinType{ .complex = try self.parseComplexType() };
        }

        // Grammar: tuple-type ::= `tuple` `<` (type ( `,` type)*)? `>`
        else if (std.mem.eql(u8, type_name, "tuple")) {
            _ = self.advance();
            return ast.BuiltinType{ .tuple = try self.parseTupleType() };
        }

        self.reportError("Unknown type");
        return ParseError.InvalidType;
    }

    // Grammar: function-type ::= (type | type-list-parens) `->` (type | type-list-parens)
    fn parseFunctionType(self: *Parser) !ast.FunctionType {
        var inputs: std.ArrayList(ast.Type) = .empty;
        errdefer inputs.deinit(self.allocator);

        // Parse input types
        if (self.check(.lparen)) {
            _ = self.advance();
            // Grammar: type-list-parens ::= `(` type-list-no-parens? `)`
            if (!self.check(.rparen)) {
                // Grammar: type-list-no-parens ::= type (`,` type)*
                while (true) {
                    try inputs.append(self.allocator, try self.parseType());
                    if (!self.match(&.{.comma})) break;
                }
            }
            _ = try self.expect(.rparen);
        } else {
            // Single type without parens
            try inputs.append(self.allocator, try self.parseType());
        }

        _ = try self.expect(.arrow);

        var outputs: std.ArrayList(ast.Type) = .empty;
        errdefer outputs.deinit(self.allocator);

        // Parse output types
        if (self.check(.lparen)) {
            _ = self.advance();
            if (!self.check(.rparen)) {
                while (true) {
                    try outputs.append(self.allocator, try self.parseType());
                    if (!self.match(&.{.comma})) break;
                }
            }
            _ = try self.expect(.rparen);
        } else {
            // Single type without parens
            try outputs.append(self.allocator, try self.parseType());
        }

        return ast.FunctionType{
            .inputs = try inputs.toOwnedSlice(self.allocator),
            .outputs = try outputs.toOwnedSlice(self.allocator),
        };
    }

    // Grammar: tensor-type ::= `tensor` `<` dimension-list type (`,` encoding)? `>`
    // Grammar: dimension-list ::= (dimension `x`)*
    // Grammar: dimension ::= `?` | decimal-literal
    fn parseTensorType(self: *Parser) !ast.TensorType {
        _ = try self.expect(.langle);

        var dimensions: std.ArrayList(ast.TensorType.Dimension) = .empty;
        errdefer dimensions.deinit(self.allocator);

        // Parse dimensions - the lexer groups "x8xf32" as one token after "4"
        // So we need to handle this manually
        var element_type_str: []const u8 = "";

        while (true) {
            if (self.check(.question)) {
                _ = self.advance();
                try dimensions.append(self.allocator, .dynamic);

                // Check for 'x' separator (might be part of a larger token)
                if (self.check(.bare_id) and self.current.lexeme.len > 0 and self.current.lexeme[0] == 'x') {
                    const lexeme = self.current.lexeme;
                    _ = self.advance();

                    // Parse remaining dimensions from this token (e.g., "x8xf32")
                    var i: usize = 1; // skip first 'x'
                    while (i < lexeme.len) {
                        if (lexeme[i] == '?') {
                            try dimensions.append(self.allocator, .dynamic);
                            i += 1;
                            if (i < lexeme.len and lexeme[i] == 'x') i += 1;
                        } else if (std.ascii.isDigit(lexeme[i])) {
                            const dim_start = i;
                            while (i < lexeme.len and std.ascii.isDigit(lexeme[i])) i += 1;
                            const dim_str = lexeme[dim_start..i];
                            const dim = try std.fmt.parseInt(u64, dim_str, 10);
                            try dimensions.append(self.allocator, .{ .static = dim });
                            if (i < lexeme.len and lexeme[i] == 'x') i += 1;
                        } else {
                            // Rest is element type
                            element_type_str = lexeme[i..];
                            break;
                        }
                    }
                    break;
                }
            } else if (self.check(.integer_literal)) {
                const dim_token = self.advance();
                const dim_value = try std.fmt.parseInt(u64, dim_token.lexeme, 10);
                try dimensions.append(self.allocator, .{ .static = dim_value });

                // After integer, check for x-prefixed token
                if (self.check(.bare_id) and self.current.lexeme.len > 0 and self.current.lexeme[0] == 'x') {
                    const lexeme = self.current.lexeme;
                    _ = self.advance();

                    // Parse remaining dimensions from this token (e.g., "x8xf32")
                    var i: usize = 1; // skip first 'x'
                    while (i < lexeme.len) {
                        if (lexeme[i] == '?') {
                            try dimensions.append(self.allocator, .dynamic);
                            i += 1;
                            if (i < lexeme.len and lexeme[i] == 'x') i += 1;
                        } else if (std.ascii.isDigit(lexeme[i])) {
                            const dim_start = i;
                            while (i < lexeme.len and std.ascii.isDigit(lexeme[i])) i += 1;
                            const dim_str = lexeme[dim_start..i];
                            const dim = try std.fmt.parseInt(u64, dim_str, 10);
                            try dimensions.append(self.allocator, .{ .static = dim });
                            if (i < lexeme.len and lexeme[i] == 'x') i += 1;
                        } else {
                            // Rest is element type
                            element_type_str = lexeme[i..];
                            break;
                        }
                    }
                    break;
                }
            } else {
                // Current token is element type
                break;
            }
        }

        // Parse element type
        const element_type = try self.allocator.create(ast.Type);
        if (element_type_str.len > 0) {
            // We extracted element type from a compound token
            var type_lexer = Lexer.init(element_type_str);
            var type_parser = try Parser.init(self.allocator, &type_lexer);
            defer type_parser.deinit();
            element_type.* = try type_parser.parseType();
        } else {
            // Element type is the current token
            element_type.* = try self.parseType();
        }

        // TODO: Parse optional encoding (`,` encoding)
        var encoding: ?ast.AttributeValue = null;
        if (self.match(&.{.comma})) {
            encoding = try self.parseAttributeValue();
        }

        _ = try self.expect(.rangle);

        return ast.TensorType{
            .dimensions = try dimensions.toOwnedSlice(self.allocator),
            .element_type = element_type,
            .encoding = encoding,
        };
    }

    // Grammar: memref-type ::= `memref` `<` dimension-list type (`,` layout-specification)? (`,` memory-space)? `>`
    fn parseMemRefType(self: *Parser) !ast.MemRefType {
        _ = try self.expect(.langle);

        var dimensions: std.ArrayList(ast.TensorType.Dimension) = .empty;
        errdefer dimensions.deinit(self.allocator);

        // Parse dimensions (same approach as tensor)
        var element_type_str: []const u8 = "";

        while (true) {
            if (self.check(.question)) {
                _ = self.advance();
                try dimensions.append(self.allocator, .dynamic);

                if (self.check(.bare_id) and self.current.lexeme.len > 0 and self.current.lexeme[0] == 'x') {
                    const rest = self.current.lexeme[1..];
                    _ = self.advance();
                    element_type_str = rest;
                    break;
                }
            } else if (self.check(.integer_literal)) {
                const dim_token = self.advance();
                const dim_value = try std.fmt.parseInt(u64, dim_token.lexeme, 10);
                try dimensions.append(self.allocator, .{ .static = dim_value });

                if (self.check(.bare_id) and self.current.lexeme.len > 0 and self.current.lexeme[0] == 'x') {
                    const lexeme = self.current.lexeme;
                    _ = self.advance();

                    var i: usize = 1;
                    while (i < lexeme.len) {
                        if (lexeme[i] == '?') {
                            try dimensions.append(self.allocator, .dynamic);
                            i += 1;
                            if (i < lexeme.len and lexeme[i] == 'x') i += 1;
                        } else if (std.ascii.isDigit(lexeme[i])) {
                            const dim_start = i;
                            while (i < lexeme.len and std.ascii.isDigit(lexeme[i])) i += 1;
                            const dim_str = lexeme[dim_start..i];
                            const dim = try std.fmt.parseInt(u64, dim_str, 10);
                            try dimensions.append(self.allocator, .{ .static = dim });
                            if (i < lexeme.len and lexeme[i] == 'x') i += 1;
                        } else {
                            element_type_str = lexeme[i..];
                            break;
                        }
                    }
                    break;
                }
            } else {
                break;
            }
        }

        // Parse element type
        const element_type = try self.allocator.create(ast.Type);
        if (element_type_str.len > 0) {
            var type_lexer = Lexer.init(element_type_str);
            var type_parser = try Parser.init(self.allocator, &type_lexer);
            defer type_parser.deinit();
            element_type.* = try type_parser.parseType();
        } else {
            element_type.* = try self.parseType();
        }

        // TODO: Parse optional layout and memory space
        const layout: ?[]const u8 = null;
        const memory_space: ?ast.AttributeValue = null;

        if (self.match(&.{.comma})) {
            // This could be layout or memory space - simplified for now
            // Just skip to closing angle bracket
            var depth: usize = 1;
            while (depth > 0 and !self.isAtEnd()) {
                if (self.check(.langle)) depth += 1;
                if (self.check(.rangle)) {
                    depth -= 1;
                    if (depth == 0) break;
                }
                _ = self.advance();
            }
        }

        _ = try self.expect(.rangle);

        return ast.MemRefType{
            .dimensions = try dimensions.toOwnedSlice(self.allocator),
            .element_type = element_type,
            .layout = layout,
            .memory_space = memory_space,
        };
    }

    // Grammar: vector-type ::= `vector` `<` vector-dim-list vector-element-type `>`
    // Grammar: vector-dim-list := (static-dim-list `x`)?
    // Grammar: static-dim ::= (decimal-literal | `[` decimal-literal `]`)
    fn parseVectorType(self: *Parser) !ast.VectorType {
        _ = try self.expect(.langle);

        var dimensions: std.ArrayList(ast.VectorType.VectorDimension) = .empty;
        errdefer dimensions.deinit(self.allocator);

        // Parse vector dimensions (similar approach as tensor)
        var element_type_str: []const u8 = "";

        while (true) {
            if (self.check(.lbracket)) {
                // Scalable dimension: [n]
                _ = self.advance();
                const dim_token = try self.expect(.integer_literal);
                const dim_value = try std.fmt.parseInt(u64, dim_token.lexeme, 10);
                _ = try self.expect(.rbracket);
                try dimensions.append(self.allocator, .{ .scalable = dim_value });

                // After bracket, look for 'x' separator
                if (self.check(.bare_id) and self.current.lexeme.len > 0 and self.current.lexeme[0] == 'x') {
                    const lexeme = self.current.lexeme;
                    _ = self.advance();

                    // Parse remaining dimensions from this token (e.g., "x8xf32")
                    var i: usize = 1; // skip first 'x'
                    while (i < lexeme.len) {
                        if (lexeme[i] == '[') {
                            // Scalable dimension in the middle
                            i += 1;
                            const dim_start = i;
                            while (i < lexeme.len and std.ascii.isDigit(lexeme[i])) i += 1;
                            const dim_str = lexeme[dim_start..i];
                            const dim = try std.fmt.parseInt(u64, dim_str, 10);
                            try dimensions.append(self.allocator, .{ .scalable = dim });
                            if (i < lexeme.len and lexeme[i] == ']') i += 1;
                            if (i < lexeme.len and lexeme[i] == 'x') i += 1;
                        } else if (std.ascii.isDigit(lexeme[i])) {
                            const dim_start = i;
                            while (i < lexeme.len and std.ascii.isDigit(lexeme[i])) i += 1;
                            const dim_str = lexeme[dim_start..i];
                            const dim = try std.fmt.parseInt(u64, dim_str, 10);
                            try dimensions.append(self.allocator, .{ .fixed = dim });
                            if (i < lexeme.len and lexeme[i] == 'x') i += 1;
                        } else {
                            element_type_str = lexeme[i..];
                            break;
                        }
                    }
                    break;
                }
            } else if (self.check(.integer_literal)) {
                // Fixed dimension
                const dim_token = self.advance();
                const dim_value = try std.fmt.parseInt(u64, dim_token.lexeme, 10);
                try dimensions.append(self.allocator, .{ .fixed = dim_value });

                // After integer, check for x-prefixed token
                if (self.check(.bare_id) and self.current.lexeme.len > 0 and self.current.lexeme[0] == 'x') {
                    const lexeme = self.current.lexeme;
                    _ = self.advance();

                    var i: usize = 1;
                    while (i < lexeme.len) {
                        if (lexeme[i] == '[') {
                            // Scalable dimension in the middle
                            i += 1;
                            const dim_start = i;
                            while (i < lexeme.len and std.ascii.isDigit(lexeme[i])) i += 1;
                            const dim_str = lexeme[dim_start..i];
                            const dim = try std.fmt.parseInt(u64, dim_str, 10);
                            try dimensions.append(self.allocator, .{ .scalable = dim });
                            if (i < lexeme.len and lexeme[i] == ']') i += 1;
                            if (i < lexeme.len and lexeme[i] == 'x') i += 1;
                        } else if (std.ascii.isDigit(lexeme[i])) {
                            const dim_start = i;
                            while (i < lexeme.len and std.ascii.isDigit(lexeme[i])) i += 1;
                            const dim_str = lexeme[dim_start..i];
                            const dim = try std.fmt.parseInt(u64, dim_str, 10);
                            try dimensions.append(self.allocator, .{ .fixed = dim });
                            if (i < lexeme.len and lexeme[i] == 'x') i += 1;
                        } else {
                            element_type_str = lexeme[i..];
                            break;
                        }
                    }
                    break;
                }
            } else {
                break;
            }
        }

        // Parse element type
        const element_type = try self.allocator.create(ast.Type);
        if (element_type_str.len > 0) {
            var type_lexer = Lexer.init(element_type_str);
            var type_parser = try Parser.init(self.allocator, &type_lexer);
            defer type_parser.deinit();
            element_type.* = try type_parser.parseType();
        } else {
            element_type.* = try self.parseType();
        }

        _ = try self.expect(.rangle);

        return ast.VectorType{
            .dimensions = try dimensions.toOwnedSlice(self.allocator),
            .element_type = element_type,
        };
    }

    // Grammar: complex-type ::= `complex` `<` type `>`
    fn parseComplexType(self: *Parser) !*ast.Type {
        _ = try self.expect(.langle);

        const element_type = try self.allocator.create(ast.Type);
        element_type.* = try self.parseType();

        _ = try self.expect(.rangle);

        return element_type;
    }

    // Grammar: tuple-type ::= `tuple` `<` (type ( `,` type)*)? `>`
    fn parseTupleType(self: *Parser) ![]ast.Type {
        _ = try self.expect(.langle);

        var types: std.ArrayList(ast.Type) = .empty;
        errdefer types.deinit(self.allocator);

        // Parse type list (may be empty)
        if (!self.check(.rangle)) {
            try types.append(self.allocator, try self.parseType());
            while (self.match(&.{.comma})) {
                try types.append(self.allocator, try self.parseType());
            }
        }

        _ = try self.expect(.rangle);

        return try types.toOwnedSlice(self.allocator);
    }

    // Grammar: attribute-value ::= attribute-alias | dialect-attribute | builtin-attribute
    fn parseAttributeValue(self: *Parser) !ast.AttributeValue {
        // Check for attribute-alias: #alias-name
        if (self.check(.attribute_alias_id)) {
            const token = self.advance();
            return ast.AttributeValue{ .alias = token.lexeme[1..] }; // Skip '#'
        }

        // Check for dialect-attribute: #namespace...
        if (self.check(.hash)) {
            return ast.AttributeValue{ .dialect = try self.parseDialectAttribute() };
        }

        // Otherwise, parse builtin attribute
        return ast.AttributeValue{ .builtin = try self.parseBuiltinAttribute() };
    }

    // Grammar: dialect-attribute ::= `#` (opaque-dialect-attribute | pretty-dialect-attribute)
    fn parseDialectAttribute(self: *Parser) !ast.DialectAttribute {
        _ = try self.expect(.hash);

        const namespace_token = try self.expect(.bare_id);
        var body: ?[]const u8 = null;

        // Check for dialect-attribute-body: <...>
        if (self.check(.langle)) {
            const start = self.current;
            _ = self.advance(); // consume '<'

            // Simplified: just capture everything until matching '>'
            var depth: usize = 1;
            while (depth > 0 and !self.isAtEnd()) {
                if (self.check(.langle)) depth += 1;
                if (self.check(.rangle)) depth -= 1;
                _ = self.advance();
            }

            body = self.lexer.source[start.lexeme.ptr - self.lexer.source.ptr..self.previous.lexeme.ptr - self.lexer.source.ptr + self.previous.lexeme.len];
        }

        return ast.DialectAttribute{
            .namespace = namespace_token.lexeme,
            .body = body,
        };
    }

    // Parse builtin attribute values (integers, floats, strings, booleans, arrays)
    fn parseBuiltinAttribute(self: *Parser) ParseError!ast.BuiltinAttribute {
        const token = self.peek();

        // Integer literal (optionally typed: 42 : i32)
        if (token.type == .integer_literal) {
            _ = self.advance();
            const value = try std.fmt.parseInt(i64, token.lexeme, 0);

            // Check for optional type annotation: `: type`
            if (self.check(.colon)) {
                _ = self.advance(); // consume ':'
                // Parse and discard the type (we just store the integer value)
                _ = try self.parseType();
            }

            return ast.BuiltinAttribute{ .integer = value };
        }

        // Float literal
        if (token.type == .float_literal) {
            _ = self.advance();
            const value = try std.fmt.parseFloat(f64, token.lexeme);
            return ast.BuiltinAttribute{ .float = value };
        }

        // String literal
        if (token.type == .string_literal) {
            _ = self.advance();
            // Strip quotes
            const value = token.lexeme[1..token.lexeme.len-1];
            return ast.BuiltinAttribute{ .string = value };
        }

        // Boolean (true/false as bare identifiers)
        if (token.type == .bare_id) {
            if (std.mem.eql(u8, token.lexeme, "true")) {
                _ = self.advance();
                return ast.BuiltinAttribute{ .boolean = true };
            } else if (std.mem.eql(u8, token.lexeme, "false")) {
                _ = self.advance();
                return ast.BuiltinAttribute{ .boolean = false };
            }
        }

        // Array: [value, value, ...]
        if (token.type == .lbracket) {
            _ = self.advance();
            var values: std.ArrayList(ast.AttributeValue) = .empty;
            errdefer values.deinit(self.allocator);

            if (!self.check(.rbracket)) {
                try values.append(self.allocator, try self.parseAttributeValue());
                while (self.match(&.{.comma})) {
                    try values.append(self.allocator, try self.parseAttributeValue());
                }
            }

            _ = try self.expect(.rbracket);
            return ast.BuiltinAttribute{ .array = try values.toOwnedSlice(self.allocator) };
        }

        self.reportError("Expected attribute value");
        return ParseError.InvalidAttribute;
    }
};

test "parser - initialization" {
    const source = "%0 = arith.constant 42 : i32";
    var lex = Lexer.init(source);
    var parser = try Parser.init(std.testing.allocator, &lex);
    defer parser.deinit();

    try std.testing.expectEqual(TokenType.value_id, parser.current.type);
}

test "parser - helper methods" {
    const source = "%0 = arith.constant";
    var lex = Lexer.init(source);
    var parser = try Parser.init(std.testing.allocator, &lex);
    defer parser.deinit();

    // Test peek
    try std.testing.expectEqual(TokenType.value_id, parser.peek().type);

    // Test check
    try std.testing.expect(parser.check(.value_id));
    try std.testing.expect(!parser.check(.equal));

    // Test advance
    const token = parser.advance();
    try std.testing.expectEqual(TokenType.value_id, token.type);
    try std.testing.expectEqual(TokenType.equal, parser.current.type);

    // Test expect
    _ = try parser.expect(.equal);
    try std.testing.expectEqual(TokenType.bare_id, parser.current.type);
}

test "parser - parse integer types" {
    const source = "i32";
    var lex = Lexer.init(source);
    var parser = try Parser.init(std.testing.allocator, &lex);
    defer parser.deinit();

    var type_result = try parser.parseType();
    defer type_result.deinit(std.testing.allocator);

    try std.testing.expect(type_result == .builtin);
    try std.testing.expect(type_result.builtin == .integer);
    try std.testing.expectEqual(@as(u64, 32), type_result.builtin.integer.width);
}

test "parser - parse float types" {
    const source = "f64";
    var lex = Lexer.init(source);
    var parser = try Parser.init(std.testing.allocator, &lex);
    defer parser.deinit();

    var type_result = try parser.parseType();
    defer type_result.deinit(std.testing.allocator);

    try std.testing.expect(type_result == .builtin);
    try std.testing.expect(type_result.builtin == .float);
    try std.testing.expectEqual(ast.FloatType.f64, type_result.builtin.float);
}

test "parser - parse index type" {
    const source = "index";
    var lex = Lexer.init(source);
    var parser = try Parser.init(std.testing.allocator, &lex);
    defer parser.deinit();

    var type_result = try parser.parseType();
    defer type_result.deinit(std.testing.allocator);

    try std.testing.expect(type_result == .builtin);
    try std.testing.expect(type_result.builtin == .index);
}

// Test parsing generic operation format (from mlir-opt -mlir-print-op-generic)
test "parser - parse simple generic operation" {
    const source = "%0 = \"arith.constant\"() <{value = 42 : i32}> : () -> i32";
    var lex = Lexer.init(source);
    var parser = try Parser.init(std.testing.allocator, &lex);
    defer parser.deinit();

    var op = try parser.parseOperation();
    defer op.deinit(std.testing.allocator);

    try std.testing.expect(op.results != null);
    try std.testing.expectEqual(@as(usize, 1), op.results.?.results.len);
    try std.testing.expect(op.kind == .generic);
}
