const std = @import("std");
const reader = @import("reader.zig");
const Value = reader.Value;
const ValueType = reader.ValueType;

/// Parser errors
pub const ParseError = error{
    ExpectedList,
    ExpectedVector,
    ExpectedMap,
    ExpectedIdentifier,
    ExpectedKeyword,
    ExpectedValueId,
    ExpectedBlockId,
    ExpectedTypeIdentifier,
    UnexpectedStructure,
    MissingRequiredField,
    InvalidSectionName,
} || std.mem.Allocator.Error;

/// Type alias definition
pub const TypeAlias = struct {
    name: []const u8, // e.g., "!my_vec"
    definition: []const u8, // opaque string e.g., "vector<4xf32>"

    pub fn deinit(self: *TypeAlias, allocator: std.mem.Allocator) void {
        // Strings are owned by the reader, so we don't free them
        _ = self;
        _ = allocator;
    }
};

/// Attribute alias definition
pub const AttributeAlias = struct {
    name: []const u8, // e.g., "#alias_scope"
    definition: []const u8, // opaque string e.g., "#llvm.alias_scope<...>"

    pub fn deinit(self: *AttributeAlias, allocator: std.mem.Allocator) void {
        // Strings are owned by the reader, so we don't free them
        _ = self;
        _ = allocator;
    }
};

/// Top-level MLIR module containing aliases and operations
pub const MlirModule = struct {
    type_aliases: []TypeAlias,
    attribute_aliases: []AttributeAlias,
    operations: []Operation,
    allocator: std.mem.Allocator,

    pub fn deinit(self: *MlirModule) void {
        for (self.type_aliases) |*alias| {
            alias.deinit(self.allocator);
        }
        self.allocator.free(self.type_aliases);
        for (self.attribute_aliases) |*alias| {
            alias.deinit(self.allocator);
        }
        self.allocator.free(self.attribute_aliases);
        for (self.operations) |*op| {
            op.deinit(self.allocator);
        }
        self.allocator.free(self.operations);
    }
};

/// MLIR Operation - core IR construct
pub const Operation = struct {
    name: []const u8,
    result_bindings: [][]const u8,
    result_types: []TypeExpr,
    operands: [][]const u8,
    attributes: []Attribute,
    successors: []Successor,
    regions: []Region,
    location: ?AttrExpr,

    pub fn deinit(self: *Operation, allocator: std.mem.Allocator) void {
        allocator.free(self.result_bindings);
        for (self.result_types) |*ty| {
            ty.deinit(allocator);
        }
        allocator.free(self.result_types);
        allocator.free(self.operands);
        for (self.attributes) |*attr| {
            attr.deinit(allocator);
        }
        allocator.free(self.attributes);
        for (self.successors) |*succ| {
            succ.deinit(allocator);
        }
        allocator.free(self.successors);
        for (self.regions) |*region| {
            region.deinit(allocator);
        }
        allocator.free(self.regions);
        if (self.location) |*loc| {
            loc.deinit(allocator);
        }
    }
};

/// Region contains one or more blocks
pub const Region = struct {
    blocks: []Block,

    pub fn deinit(self: *Region, allocator: std.mem.Allocator) void {
        for (self.blocks) |*block| {
            block.deinit(allocator);
        }
        allocator.free(self.blocks);
    }
};

/// Block - basic block with optional label and arguments
pub const Block = struct {
    label: ?[]const u8,
    arguments: []Argument,
    operations: []Operation,

    pub fn deinit(self: *Block, allocator: std.mem.Allocator) void {
        for (self.arguments) |*arg| {
            arg.deinit(allocator);
        }
        allocator.free(self.arguments);
        for (self.operations) |*op| {
            op.deinit(allocator);
        }
        allocator.free(self.operations);
    }
};

/// Block argument (value ID and type)
pub const Argument = struct {
    value_id: []const u8,
    type: TypeExpr,

    pub fn deinit(self: *Argument, allocator: std.mem.Allocator) void {
        self.type.deinit(allocator);
    }
};

/// Successor (for control flow)
pub const Successor = struct {
    block_id: []const u8,
    operands: [][]const u8,

    pub fn deinit(self: *Successor, allocator: std.mem.Allocator) void {
        allocator.free(self.operands);
    }
};

/// Attribute (key-value pair)
pub const Attribute = struct {
    key: []const u8,
    value: AttrExpr,

    pub fn deinit(self: *Attribute, allocator: std.mem.Allocator) void {
        self.value.deinit(allocator);
    }
};

/// Type expression (wraps Reader.Value for now)
pub const TypeExpr = struct {
    value: *Value,

    pub fn deinit(self: *TypeExpr, allocator: std.mem.Allocator) void {
        // Parser doesn't own the Value, so don't deinit it
        _ = self;
        _ = allocator;
    }
};

/// Attribute expression (wraps Reader.Value for now)
pub const AttrExpr = struct {
    value: *Value,

    pub fn deinit(self: *AttrExpr, allocator: std.mem.Allocator) void {
        // Parser doesn't own the Value, so don't deinit it
        _ = self;
        _ = allocator;
    }
};

/// Parser - converts Reader Values into typed AST
pub const Parser = struct {
    allocator: std.mem.Allocator,
    source: []const u8,

    pub fn init(allocator: std.mem.Allocator, source: []const u8) Parser {
        return Parser{
            .allocator = allocator,
            .source = source,
        };
    }

    fn printErrorLocation(self: *const Parser, line: usize, column: usize) void {
        if (line == 0 or column == 0) return;

        // Find the line in the source
        var current_line: usize = 1;
        var line_start: usize = 0;
        var i: usize = 0;

        while (i < self.source.len) : (i += 1) {
            if (current_line == line) {
                // Find the end of this line
                var line_end = i;
                while (line_end < self.source.len and self.source[line_end] != '\n') : (line_end += 1) {}

                // Print the line
                const line_content = self.source[line_start..line_end];
                std.debug.print("{s}\n", .{line_content});

                // Print pointer to error location
                var j: usize = 0;
                while (j < column - 1) : (j += 1) {
                    std.debug.print(" ", .{});
                }
                std.debug.print("^\n", .{});
                return;
            }

            if (self.source[i] == '\n') {
                current_line += 1;
                line_start = i + 1;
            }
        }
    }

    /// Unescape a string literal by processing escape sequences
    fn unescapeString(allocator: std.mem.Allocator, str: []const u8) ![]const u8 {
        var result = std.ArrayList(u8){};
        errdefer result.deinit(allocator);

        var i: usize = 0;
        while (i < str.len) : (i += 1) {
            if (str[i] == '\\' and i + 1 < str.len) {
                // Handle escape sequences
                i += 1;
                const escaped_char = switch (str[i]) {
                    'n' => '\n',
                    't' => '\t',
                    'r' => '\r',
                    '\\' => '\\',
                    '"' => '"',
                    else => str[i], // Unknown escape, keep as-is
                };
                try result.append(allocator, escaped_char);
            } else {
                try result.append(allocator, str[i]);
            }
        }

        return result.toOwnedSlice(allocator);
    }

    /// Parse top-level MLIR module
    pub fn parseModule(self: *Parser, value: *Value) ParseError!MlirModule {
        // Expect (mlir (TYPE_ALIAS | OPERATION)*)
        if (value.type != .list) return error.ExpectedList;

        const list = value.data.list;
        if (list.isEmpty()) return error.UnexpectedStructure;

        const first = list.at(0);
        if (first.type != .identifier) return error.ExpectedIdentifier;
        if (!std.mem.eql(u8, first.data.atom, "mlir")) {
            return error.UnexpectedStructure;
        }

        // Parse type aliases, attribute aliases, and operations
        var type_aliases = std.ArrayList(TypeAlias){};
        errdefer {
            for (type_aliases.items) |*alias| {
                alias.deinit(self.allocator);
            }
            type_aliases.deinit(self.allocator);
        }

        var attribute_aliases = std.ArrayList(AttributeAlias){};
        errdefer {
            for (attribute_aliases.items) |*alias| {
                alias.deinit(self.allocator);
            }
            attribute_aliases.deinit(self.allocator);
        }

        var operations = std.ArrayList(Operation){};
        errdefer {
            for (operations.items) |*op| {
                op.deinit(self.allocator);
            }
            operations.deinit(self.allocator);
        }

        var i: usize = 1;
        while (i < list.len()) : (i += 1) {
            const item = list.at(i);
            if (item.type != .list) continue;

            const item_list = item.data.list;
            if (item_list.isEmpty()) continue;

            const item_name = item_list.at(0);
            if (item_name.type != .identifier) continue;

            if (std.mem.eql(u8, item_name.data.atom, "type-alias")) {
                const alias = try self.parseTypeAlias(item);
                try type_aliases.append(self.allocator, alias);
            } else if (std.mem.eql(u8, item_name.data.atom, "attribute-alias")) {
                const alias = try self.parseAttributeAlias(item);
                try attribute_aliases.append(self.allocator, alias);
            } else if (std.mem.eql(u8, item_name.data.atom, "operation")) {
                const op = try self.parseOperation(item);
                try operations.append(self.allocator, op);
            }
        }

        return MlirModule{
            .type_aliases = try type_aliases.toOwnedSlice(self.allocator),
            .attribute_aliases = try attribute_aliases.toOwnedSlice(self.allocator),
            .operations = try operations.toOwnedSlice(self.allocator),
            .allocator = self.allocator,
        };
    }

    /// Parse a type alias
    pub fn parseTypeAlias(self: *Parser, value: *Value) ParseError!TypeAlias {
        // Expect (type-alias TYPE_ID STRING)
        if (value.type != .list) return error.ExpectedList;

        const list = value.data.list;
        if (list.len() != 3) {
            std.debug.print("\nerror: Invalid type-alias structure\n", .{});
            std.debug.print("Expected: (type-alias TYPE_ID STRING)\n", .{});
            std.debug.print("Found: list with {} elements\n", .{list.len()});
            if (list.len() > 0) {
                std.debug.print("Structure: (", .{});
                var i: usize = 0;
                while (i < list.len() and i < 5) : (i += 1) {
                    if (i > 0) std.debug.print(" ", .{});
                    const item = list.at(i);
                    switch (item.type) {
                        .identifier => std.debug.print("{s}", .{item.data.atom}),
                        .string => std.debug.print("\"{s}\"", .{item.data.atom}),
                        .type => std.debug.print("!{s}", .{item.data.type}),
                        else => std.debug.print("{s}", .{@tagName(item.type)}),
                    }
                }
                if (list.len() > 5) std.debug.print(" ...", .{});
                std.debug.print(")\n", .{});
            }
            std.debug.print("\nType aliases must have exactly 3 elements:\n", .{});
            std.debug.print("  1. The keyword 'type-alias'\n", .{});
            std.debug.print("  2. A type identifier (e.g., !my_type)\n", .{});
            std.debug.print("  3. A string definition\n", .{});
            return error.UnexpectedStructure;
        }

        const first = list.at(0);
        if (first.type != .identifier) return error.ExpectedIdentifier;
        if (!std.mem.eql(u8, first.data.atom, "type-alias")) {
            return error.UnexpectedStructure;
        }

        // Get the type name (should be .type like !my_vec)
        const type_name = list.at(1);
        var name: []const u8 = undefined;
        if (type_name.type == .type) {
            name = type_name.data.type;
        } else {
            // Might also be an identifier starting with !
            return error.ExpectedTypeIdentifier;
        }

        // Get the definition (should be a string)
        const def = list.at(2);
        if (def.type != .string) return error.ExpectedIdentifier;
        // String tokens include quotes - strip them and unescape
        const raw_string = def.data.atom;
        const stripped = if (raw_string.len >= 2) raw_string[1 .. raw_string.len - 1] else raw_string;
        const definition = unescapeString(self.allocator, stripped) catch return error.OutOfMemory;

        return TypeAlias{
            .name = name,
            .definition = definition,
        };
    }

    /// Parse an attribute alias
    pub fn parseAttributeAlias(self: *Parser, value: *Value) ParseError!AttributeAlias {
        // Expect (attribute-alias ATTR_ID STRING)
        if (value.type != .list) return error.ExpectedList;

        const list = value.data.list;
        if (list.len() != 3) {
            std.debug.print("\nerror: Invalid attribute-alias structure\n", .{});
            std.debug.print("Expected: (attribute-alias ATTR_ID STRING)\n", .{});
            std.debug.print("Found: list with {} elements\n", .{list.len()});
            if (list.len() > 0) {
                std.debug.print("Structure: (", .{});
                var i: usize = 0;
                while (i < list.len() and i < 5) : (i += 1) {
                    if (i > 0) std.debug.print(" ", .{});
                    const item = list.at(i);
                    switch (item.type) {
                        .identifier => std.debug.print("{s}", .{item.data.atom}),
                        .string => std.debug.print("\"{s}\"", .{item.data.atom}),
                        .attr_expr => std.debug.print("#{s}", .{item.data.attr_expr.data.atom}),
                        else => std.debug.print("{s}", .{@tagName(item.type)}),
                    }
                }
                if (list.len() > 5) std.debug.print(" ...", .{});
                std.debug.print(")\n", .{});
            }
            std.debug.print("\nAttribute aliases must have exactly 3 elements:\n", .{});
            std.debug.print("  1. The keyword 'attribute-alias'\n", .{});
            std.debug.print("  2. An attribute identifier (e.g., #my_attr)\n", .{});
            std.debug.print("  3. A string definition\n", .{});
            return error.UnexpectedStructure;
        }

        const first = list.at(0);
        if (first.type != .identifier) return error.ExpectedIdentifier;
        if (!std.mem.eql(u8, first.data.atom, "attribute-alias")) {
            return error.UnexpectedStructure;
        }

        // Get the attribute alias name (should be an attr_expr like #alias_scope)
        const attr_name = list.at(1);
        var name: []const u8 = undefined;
        if (attr_name.type == .attr_expr) {
            // attr_expr wraps an inner identifier with the name (without #)
            // We store it without the # and handle that in resolution
            const inner = attr_name.data.attr_expr;
            if (inner.type != .identifier) return error.ExpectedIdentifier;
            name = inner.data.atom;
        } else if (attr_name.type == .identifier) {
            // Fallback for identifiers (though # should create attr_expr)
            name = attr_name.data.atom;
        } else {
            return error.ExpectedIdentifier;
        }

        // Get the definition (should be a string)
        const def = list.at(2);
        if (def.type != .string) return error.ExpectedIdentifier;
        // String tokens include quotes - strip them and unescape
        const raw_string = def.data.atom;
        const stripped = if (raw_string.len >= 2) raw_string[1 .. raw_string.len - 1] else raw_string;
        const definition = unescapeString(self.allocator, stripped) catch return error.OutOfMemory;

        return AttributeAlias{
            .name = name,
            .definition = definition,
        };
    }

    /// Parse an operation
    pub fn parseOperation(self: *Parser, value: *Value) ParseError!Operation {
        // Expect (operation (name ...) SECTION*)
        if (value.type != .list) return error.ExpectedList;

        const list = value.data.list;
        if (list.isEmpty()) return error.UnexpectedStructure;

        const first = list.at(0);
        if (first.type != .identifier) return error.ExpectedIdentifier;
        if (!std.mem.eql(u8, first.data.atom, "operation")) {
            return error.UnexpectedStructure;
        }

        var operation = Operation{
            .name = &[_]u8{},
            .result_bindings = &[_][]const u8{},
            .result_types = &[_]TypeExpr{},
            .operands = &[_][]const u8{},
            .attributes = &[_]Attribute{},
            .successors = &[_]Successor{},
            .regions = &[_]Region{},
            .location = null,
        };
        errdefer operation.deinit(self.allocator);

        // Parse sections
        var i: usize = 1;
        while (i < list.len()) : (i += 1) {
            const section = list.at(i);
            if (section.type != .list) continue;

            const section_list = section.data.list;
            if (section_list.isEmpty()) continue;

            const section_name = section_list.at(0);
            if (section_name.type != .identifier) continue;

            const name = section_name.data.atom;

            if (std.mem.eql(u8, name, "name")) {
                operation.name = try self.parseName(section);
            } else if (std.mem.eql(u8, name, "result-bindings")) {
                operation.result_bindings = try self.parseResultBindings(section);
            } else if (std.mem.eql(u8, name, "result-types")) {
                operation.result_types = try self.parseResultTypes(section);
            } else if (std.mem.eql(u8, name, "operand-uses") or std.mem.eql(u8, name, "operands")) {
                operation.operands = try self.parseOperands(section);
            } else if (std.mem.eql(u8, name, "attributes")) {
                operation.attributes = try self.parseAttributes(section);
            } else if (std.mem.eql(u8, name, "successors")) {
                operation.successors = try self.parseSuccessors(section);
            } else if (std.mem.eql(u8, name, "regions")) {
                operation.regions = try self.parseRegions(section);
            } else if (std.mem.eql(u8, name, "location")) {
                operation.location = try self.parseLocation(section);
            }
        }

        if (operation.name.len == 0) {
            return error.MissingRequiredField;
        }

        return operation;
    }

    fn parseName(self: *Parser, section: *Value) ParseError![]const u8 {
        _ = self;
        // (name OP_NAME)
        const list = section.data.list;
        if (list.len() < 2) return error.UnexpectedStructure;

        const name_value = list.at(1);
        if (name_value.type != .identifier) return error.ExpectedIdentifier;

        return name_value.data.atom;
    }

    fn parseResultBindings(self: *Parser, section: *Value) ParseError![][]const u8 {
        // (result-bindings [ VALUE_ID* ])
        const list = section.data.list;
        if (list.len() < 2) return error.UnexpectedStructure;

        const bindings_value = list.at(1);
        if (bindings_value.type != .vector) return error.ExpectedVector;

        const vec = bindings_value.data.vector;
        var bindings = try std.ArrayList([]const u8).initCapacity(self.allocator, vec.len());
        errdefer bindings.deinit(self.allocator);

        var i: usize = 0;
        while (i < vec.len()) : (i += 1) {
            const val = vec.at(i);
            if (val.type != .value_id) return error.ExpectedValueId;
            bindings.appendAssumeCapacity(val.data.atom);
        }

        return bindings.toOwnedSlice(self.allocator);
    }

    fn parseResultTypes(self: *Parser, section: *Value) ParseError![]TypeExpr {
        // (result-types TYPE*)
        const list = section.data.list;
        var types = std.ArrayList(TypeExpr){};
        errdefer {
            for (types.items) |*ty| {
                ty.deinit(self.allocator);
            }
            types.deinit(self.allocator);
        }

        var i: usize = 1;
        while (i < list.len()) : (i += 1) {
            const type_value = list.at(i);
            // Type should be a type, function_type, or identifier (for plain builtin types like i32)
            if (type_value.type != .type and type_value.type != .function_type and type_value.type != .identifier) {
                return error.UnexpectedStructure;
            }

            // Use the value directly - parser doesn't own it
            try types.append(self.allocator, TypeExpr{ .value = type_value });
        }

        return types.toOwnedSlice(self.allocator);
    }

    fn parseOperands(self: *Parser, section: *Value) ParseError![][]const u8 {
        // (operand-uses VALUE_ID*)
        const list = section.data.list;
        var operands = std.ArrayList([]const u8){};
        errdefer operands.deinit(self.allocator);

        var i: usize = 1;
        while (i < list.len()) : (i += 1) {
            const val = list.at(i);
            if (val.type != .value_id) return error.ExpectedValueId;
            try operands.append(self.allocator, val.data.atom);
        }

        return operands.toOwnedSlice(self.allocator);
    }

    fn parseAttributes(self: *Parser, section: *Value) ParseError![]Attribute {
        // (attributes { KEYWORD ATTR* })
        const list = section.data.list;
        if (list.len() < 2) return error.UnexpectedStructure;

        const map_value = list.at(1);
        if (map_value.type != .map) return error.ExpectedMap;

        const map = map_value.data.map;
        var attributes = std.ArrayList(Attribute){};
        errdefer {
            for (attributes.items) |*attr| {
                attr.deinit(self.allocator);
            }
            attributes.deinit(self.allocator);
        }

        // Map is stored as flat list of k,v pairs
        var i: usize = 0;
        while (i + 1 < map.len()) : (i += 2) {
            const key = map.at(i);
            const val = map.at(i + 1);

            if (key.type != .keyword) {
                const key_str = if (key.type == .identifier or key.type == .keyword) key.data.atom else "(complex value)";
                std.debug.print("\nerror: Expected keyword in attribute map, but found {s} '{s}'\n", .{
                    @tagName(key.type),
                    key_str,
                });
                std.debug.print("Attribute maps must use keywords (starting with ':') as keys\n", .{});
                std.debug.print("Map has {} elements, currently at index {}\n", .{map.len(), i});
                std.debug.print("Previous elements:\n", .{});
                var j: usize = 0;
                while (j < i and j < 10) : (j += 2) {
                    const prev_key = map.at(j);
                    const prev_val = map.at(j + 1);
                    const pk_str = if (prev_key.type == .identifier or prev_key.type == .keyword) prev_key.data.atom else "(complex)";
                    const pv_str = if (prev_val.type == .identifier or prev_val.type == .keyword) prev_val.data.atom else @tagName(prev_val.type);
                    std.debug.print("  [{}: {s} => {s}]\n", .{j/2, pk_str, pv_str});
                }
                return error.ExpectedKeyword;
            }

            // Convert keyword to name (strip the leading colon)
            const key_name = key.keywordToName();

            // Use the value directly - parser doesn't own it
            try attributes.append(self.allocator, Attribute{
                .key = key_name,
                .value = AttrExpr{ .value = val },
            });
        }

        return attributes.toOwnedSlice(self.allocator);
    }

    fn parseSuccessors(self: *Parser, section: *Value) ParseError![]Successor {
        // (successors SUCCESSOR*)
        const list = section.data.list;
        var successors = std.ArrayList(Successor){};
        errdefer {
            for (successors.items) |*succ| {
                succ.deinit(self.allocator);
            }
            successors.deinit(self.allocator);
        }

        var i: usize = 1;
        while (i < list.len()) : (i += 1) {
            const succ_value = list.at(i);
            const succ = try self.parseSuccessor(succ_value);
            try successors.append(self.allocator, succ);
        }

        return successors.toOwnedSlice(self.allocator);
    }

    fn parseSuccessor(self: *Parser, value: *Value) ParseError!Successor {
        // (successor BLOCK_ID (operand-bundle)?)
        if (value.type != .list) return error.ExpectedList;

        const list = value.data.list;
        if (list.len() < 2) return error.UnexpectedStructure;

        const first = list.at(0);
        if (first.type != .identifier) return error.ExpectedIdentifier;
        if (!std.mem.eql(u8, first.data.atom, "successor")) {
            return error.UnexpectedStructure;
        }

        const block_id = list.at(1);
        if (block_id.type != .block_id) return error.ExpectedBlockId;

        var operands = std.ArrayList([]const u8){};
        errdefer operands.deinit(self.allocator);

        // Optional operand bundle
        if (list.len() > 2) {
            const bundle = list.at(2);
            if (bundle.type == .list) {
                const bundle_list = bundle.data.list;
                var j: usize = 0;
                while (j < bundle_list.len()) : (j += 1) {
                    const op = bundle_list.at(j);
                    if (op.type != .value_id) return error.ExpectedValueId;
                    try operands.append(self.allocator, op.data.atom);
                }
            }
        }

        return Successor{
            .block_id = block_id.data.atom,
            .operands = try operands.toOwnedSlice(self.allocator),
        };
    }

    fn parseRegions(self: *Parser, section: *Value) ParseError![]Region {
        // (regions REGION*)
        const list = section.data.list;
        var regions = std.ArrayList(Region){};
        errdefer {
            for (regions.items) |*region| {
                region.deinit(self.allocator);
            }
            regions.deinit(self.allocator);
        }

        var i: usize = 1;
        while (i < list.len()) : (i += 1) {
            const region_value = list.at(i);
            const region = try self.parseRegion(region_value);
            try regions.append(self.allocator, region);
        }

        return regions.toOwnedSlice(self.allocator);
    }

    fn parseRegion(self: *Parser, value: *Value) ParseError!Region {
        // (region BLOCK+)
        if (value.type != .list) return error.ExpectedList;

        const list = value.data.list;
        if (list.isEmpty()) return error.UnexpectedStructure;

        const first = list.at(0);
        if (first.type != .identifier) return error.ExpectedIdentifier;
        if (!std.mem.eql(u8, first.data.atom, "region")) {
            return error.UnexpectedStructure;
        }

        var blocks = std.ArrayList(Block){};
        errdefer {
            for (blocks.items) |*block| {
                block.deinit(self.allocator);
            }
            blocks.deinit(self.allocator);
        }

        var i: usize = 1;
        while (i < list.len()) : (i += 1) {
            const block_value = list.at(i);
            const block = try self.parseBlock(block_value);
            try blocks.append(self.allocator, block);
        }

        return Region{
            .blocks = try blocks.toOwnedSlice(self.allocator),
        };
    }

    fn parseBlock(self: *Parser, value: *Value) ParseError!Block {
        // (block [ BLOCK_ID ] (arguments [ [ VALUE_ID TYPE ]* ]) OPERATION*)
        if (value.type != .list) return error.ExpectedList;

        const list = value.data.list;
        if (list.isEmpty()) return error.UnexpectedStructure;

        const first = list.at(0);
        if (first.type != .identifier) return error.ExpectedIdentifier;
        if (!std.mem.eql(u8, first.data.atom, "block")) {
            return error.UnexpectedStructure;
        }

        var block = Block{
            .label = null,
            .arguments = &[_]Argument{},
            .operations = &[_]Operation{},
        };

        var idx: usize = 1;

        // Optional label vector
        if (idx < list.len()) {
            const maybe_label = list.at(idx);
            if (maybe_label.type == .vector) {
                const label_vec = maybe_label.data.vector;
                if (label_vec.len() > 0) {
                    const label = label_vec.at(0);
                    if (label.type == .block_id) {
                        block.label = label.data.atom;
                    }
                }
                idx += 1;
            }
        }

        // Parse arguments and operations
        while (idx < list.len()) : (idx += 1) {
            const item = list.at(idx);
            if (item.type != .list) continue;

            const item_list = item.data.list;
            if (item_list.isEmpty()) continue;

            const item_first = item_list.at(0);
            if (item_first.type != .identifier) continue;

            if (std.mem.eql(u8, item_first.data.atom, "arguments")) {
                block.arguments = try self.parseArguments(item);
            } else if (std.mem.eql(u8, item_first.data.atom, "operation")) {
                // Parse operations
                var operations = std.ArrayList(Operation){};
                errdefer {
                    for (operations.items) |*op| {
                        op.deinit(self.allocator);
                    }
                    operations.deinit(self.allocator);
                }

                var op_idx = idx;
                while (op_idx < list.len()) : (op_idx += 1) {
                    const op_value = list.at(op_idx);
                    if (op_value.type != .list) continue;

                    const op_list = op_value.data.list;
                    if (op_list.isEmpty()) continue;

                    const op_first = op_list.at(0);
                    if (op_first.type != .identifier) continue;
                    if (!std.mem.eql(u8, op_first.data.atom, "operation")) continue;

                    const op = try self.parseOperation(op_value);
                    try operations.append(self.allocator, op);
                }

                block.operations = try operations.toOwnedSlice(self.allocator);
                break;
            }
        }

        return block;
    }

    fn parseArguments(self: *Parser, section: *Value) ParseError![]Argument {
        // (arguments [ [ VALUE_ID TYPE ]* ])
        const list = section.data.list;
        if (list.len() < 2) return error.UnexpectedStructure;

        const args_vec = list.at(1);
        if (args_vec.type != .vector) return error.ExpectedVector;

        const vec = args_vec.data.vector;
        var arguments = std.ArrayList(Argument){};
        errdefer {
            for (arguments.items) |*arg| {
                arg.deinit(self.allocator);
            }
            arguments.deinit(self.allocator);
        }

        var i: usize = 0;
        while (i < vec.len()) : (i += 1) {
            const arg_pair = vec.at(i);
            if (arg_pair.type != .vector) return error.ExpectedVector;

            const pair = arg_pair.data.vector;
            if (pair.len() < 2) return error.UnexpectedStructure;

            const value_id = pair.at(0);
            const type_expr = pair.at(1);

            if (value_id.type != .value_id) return error.ExpectedValueId;
            if (type_expr.type != .type and type_expr.type != .function_type and type_expr.type != .identifier) {
                return error.UnexpectedStructure;
            }

            // Use the value directly - parser doesn't own it
            try arguments.append(self.allocator, Argument{
                .value_id = value_id.data.atom,
                .type = TypeExpr{ .value = type_expr },
            });
        }

        return arguments.toOwnedSlice(self.allocator);
    }

    fn parseLocation(self: *Parser, section: *Value) ParseError!?AttrExpr {
        _ = self;
        // (location ATTR)
        const list = section.data.list;
        if (list.len() < 2) return null;

        const loc_value = list.at(1);

        // Use the value directly - parser doesn't own it
        return AttrExpr{ .value = loc_value };
    }
};
