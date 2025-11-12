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
    ExpectedHasType,
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

    /// Check if an identifier is a terse operation name (contains a dot)
    fn isTerseOperation(name: []const u8) bool {
        return std.mem.indexOf(u8, name, ".") != null;
    }

    /// Parse top-level MLIR module
    pub fn parseModule(self: *Parser, value: *Value) ParseError!MlirModule {
        // Accept either (mlir ...) or unwrapped (TYPE_ALIAS | OPERATION)*
        if (value.type != .list) return error.ExpectedList;

        const list = value.data.list;
        if (list.isEmpty()) return error.UnexpectedStructure;

        // Check if there's an optional 'mlir' wrapper or if this is a direct operation/alias
        var start_index: usize = 0;
        var parse_single_item = false;
        const first = list.at(0);
        if (first.type == .identifier) {
            const name = first.data.atom;
            if (std.mem.eql(u8, name, "mlir")) {
                // Has mlir wrapper - skip it
                start_index = 1;
            } else if (std.mem.eql(u8, name, "operation") or
                      std.mem.eql(u8, name, "type-alias") or
                      std.mem.eql(u8, name, "attribute-alias") or
                      std.mem.eql(u8, name, "declare") or
                      isTerseOperation(name)) {
                // This is a single operation/alias/declare, not a list of them
                parse_single_item = true;
            }
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

        // If this is a single item (not wrapped), parse it directly
        if (parse_single_item) {
            const name = first.data.atom;
            if (std.mem.eql(u8, name, "operation")) {
                const op = try self.parseOperation(value);
                try operations.append(self.allocator, op);
            } else if (std.mem.eql(u8, name, "type-alias")) {
                const alias = try self.parseTypeAlias(value);
                try type_aliases.append(self.allocator, alias);
            } else if (std.mem.eql(u8, name, "attribute-alias")) {
                const alias = try self.parseAttributeAlias(value);
                try attribute_aliases.append(self.allocator, alias);
            } else if (std.mem.eql(u8, name, "declare")) {
                const op = try self.parseDeclare(value);
                try operations.append(self.allocator, op);
            } else if (isTerseOperation(name)) {
                const op = try self.parseTerseOperation(value);
                try operations.append(self.allocator, op);
            }

            return MlirModule{
                .type_aliases = try type_aliases.toOwnedSlice(self.allocator),
                .attribute_aliases = try attribute_aliases.toOwnedSlice(self.allocator),
                .operations = try operations.toOwnedSlice(self.allocator),
                .allocator = self.allocator,
            };
        }

        var i: usize = start_index;
        while (i < list.len()) : (i += 1) {
            const item = list.at(i);
            if (item.type != .list) continue;

            const item_list = item.data.list;
            if (item_list.isEmpty()) continue;

            const item_name = item_list.at(0);
            if (item_name.type != .identifier) continue;

            const name = item_name.data.atom;
            if (std.mem.eql(u8, name, "type-alias")) {
                const alias = try self.parseTypeAlias(item);
                try type_aliases.append(self.allocator, alias);
            } else if (std.mem.eql(u8, name, "attribute-alias")) {
                const alias = try self.parseAttributeAlias(item);
                try attribute_aliases.append(self.allocator, alias);
            } else if (std.mem.eql(u8, name, "operation")) {
                const op = try self.parseOperation(item);
                try operations.append(self.allocator, op);
            } else if (std.mem.eql(u8, name, "declare")) {
                const op = try self.parseDeclare(item);
                try operations.append(self.allocator, op);
            } else if (isTerseOperation(name)) {
                const op = try self.parseTerseOperation(item);
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
            } else if (std.mem.eql(u8, name, "operands")) {
                operation.operands = try self.parseOperands(section);
            } else if (std.mem.eql(u8, name, "attributes")) {
                operation.attributes = try self.parseAttributes(section);
            } else if (std.mem.eql(u8, name, "successors")) {
                operation.successors = try self.parseSuccessors(section);
            } else if (std.mem.eql(u8, name, "regions")) {
                operation.regions = try self.parseRegions(section, operation.name);
            } else if (std.mem.eql(u8, name, "location")) {
                operation.location = try self.parseLocation(section);
            }
        }

        if (operation.name.len == 0) {
            return error.MissingRequiredField;
        }

        return operation;
    }

    /// Parse a declare form: (declare name expr)
    /// Example: (declare my-var (arith.constant {:value 42}))
    fn parseDeclare(self: *Parser, value: *Value) ParseError!Operation {
        if (value.type != .list) return error.ExpectedList;

        const list = value.data.list;
        if (list.len() != 3) {
            std.debug.print("\nerror: Invalid declare structure\n", .{});
            std.debug.print("Expected: (declare name expr)\n", .{});
            std.debug.print("Found: list with {d} elements\n", .{list.len()});
            return error.UnexpectedStructure;
        }

        const first = list.at(0);
        if (first.type != .identifier or !std.mem.eql(u8, first.data.atom, "declare")) {
            return error.UnexpectedStructure;
        }

        // Get the variable name
        const name_value = list.at(1);
        if (name_value.type != .identifier) {
            std.debug.print("\nerror: Expected identifier as declare name\n", .{});
            return error.ExpectedIdentifier;
        }
        const var_name = name_value.data.atom;

        // Parse the expression - could be a has_type or a regular operation
        const expr = list.at(2);

        var operation: Operation = undefined;
        var explicit_type: ?TypeExpr = null;

        // Check if this is a type annotation: (: expr type)
        if (expr.type == .has_type) {
            // Extract the inner expression and type
            const inner_expr = expr.data.has_type.value;
            const type_value = expr.data.has_type.type_expr;

            // Parse the inner expression
            if (inner_expr.type != .list) {
                std.debug.print("\nerror: Expected expression inside type annotation\n", .{});
                return error.ExpectedList;
            }

            const inner_list = inner_expr.data.list;
            if (inner_list.isEmpty()) return error.UnexpectedStructure;

            const inner_first = inner_list.at(0);
            if (inner_first.type != .identifier) return error.ExpectedIdentifier;

            const inner_name = inner_first.data.atom;
            if (std.mem.eql(u8, inner_name, "operation")) {
                operation = try self.parseOperation(inner_expr);
            } else if (isTerseOperation(inner_name)) {
                operation = try self.parseTerseOperation(inner_expr);
            } else {
                std.debug.print("\nerror: Unsupported expression in type annotation: {s}\n", .{inner_name});
                return error.UnexpectedStructure;
            }

            // Save the explicit type
            explicit_type = TypeExpr{ .value = type_value };

        } else if (expr.type == .list) {
            // Regular expression without type annotation
            const expr_list = expr.data.list;
            if (expr_list.isEmpty()) return error.UnexpectedStructure;

            const expr_first = expr_list.at(0);
            if (expr_first.type != .identifier) return error.ExpectedIdentifier;

            const expr_name = expr_first.data.atom;
            if (std.mem.eql(u8, expr_name, "operation")) {
                operation = try self.parseOperation(expr);
            } else if (isTerseOperation(expr_name)) {
                operation = try self.parseTerseOperation(expr);
            } else {
                std.debug.print("\nerror: Unsupported expression in declare: {s}\n", .{expr_name});
                return error.UnexpectedStructure;
            }
        } else {
            std.debug.print("\nerror: Expected operation or type annotation in declare, got {s}\n", .{@tagName(expr.type)});
            if (expr.type == .value_id) {
                std.debug.print("Note: declare with a plain value ID like (declare {s} {s}) is not allowed.\n", .{ var_name, expr.data.atom });
                std.debug.print("This might happen if nested operations were flattened but declare wasn't updated.\n", .{});
                std.debug.print("The flattener should NOT flatten the expression inside declare.\n", .{});
            } else {
                std.debug.print("Expression type: {s}\n", .{@tagName(expr.type)});
                const printer_mod = @import("printer.zig");
                var printer = printer_mod.Printer.init(self.allocator);
                defer printer.deinit();
                printer.printValue(expr) catch {};
                std.debug.print("Expression: {s}\n", .{printer.getOutput()});
            }
            return error.UnexpectedStructure;
        }

        // Create a result binding with the variable name (as a value ID)
        // We need to prepend % to make it a proper value ID
        var result_binding = std.ArrayList(u8){};
        errdefer result_binding.deinit(self.allocator);
        try result_binding.append(self.allocator, '%');
        try result_binding.appendSlice(self.allocator, var_name);
        const binding_str = try result_binding.toOwnedSlice(self.allocator);

        // Allocate a single-element array for the binding
        var bindings = try self.allocator.alloc([]const u8, 1);
        bindings[0] = binding_str;

        // Update the operation with the result binding
        operation.result_bindings = bindings;

        // If an explicit type was provided, use it
        if (explicit_type) |type_expr| {
            var types = try self.allocator.alloc(TypeExpr, 1);
            types[0] = type_expr;
            operation.result_types = types;
        }

        return operation;
    }

    /// Parse a terse operation: (op.name [attrs?] operands...)
    /// Example: (arith.addi %a %b) or (arith.constant {:value 42})
    fn parseTerseOperation(self: *Parser, value: *Value) ParseError!Operation {
        if (value.type != .list) return error.ExpectedList;

        const list = value.data.list;
        if (list.isEmpty()) return error.UnexpectedStructure;

        const first = list.at(0);
        if (first.type != .identifier) return error.ExpectedIdentifier;

        const op_name = first.data.atom;

        // Parse optional attributes (second element if it's a map)
        var attrs_start: usize = 1;
        var attributes: []Attribute = &[_]Attribute{};

        if (list.len() > 1) {
            const maybe_attrs = list.at(1);
            if (maybe_attrs.type == .map) {
                // Parse the attribute map
                const map = maybe_attrs.data.map;
                var attr_list = std.ArrayList(Attribute){};
                errdefer {
                    for (attr_list.items) |*attr| {
                        attr.deinit(self.allocator);
                    }
                    attr_list.deinit(self.allocator);
                }

                // Map is stored as flat list of k,v pairs
                var i: usize = 0;
                while (i + 1 < map.len()) : (i += 2) {
                    const key = map.at(i);
                    const val = map.at(i + 1);

                    if (key.type != .keyword) {
                        std.debug.print("\nerror: Expected keyword in attribute map\n", .{});
                        return error.ExpectedKeyword;
                    }

                    const key_name = key.keywordToName();
                    try attr_list.append(self.allocator, Attribute{
                        .key = key_name,
                        .value = AttrExpr{ .value = val },
                    });
                }

                attributes = try attr_list.toOwnedSlice(self.allocator);
                attrs_start = 2; // Skip the map
            }
        }

        // Parse operands and regions (remaining elements)
        var operands = std.ArrayList([]const u8){};
        errdefer operands.deinit(self.allocator);

        var regions = std.ArrayList(Region){};
        errdefer {
            for (regions.items) |*region| {
                region.deinit(self.allocator);
            }
            regions.deinit(self.allocator);
        }

        var i: usize = attrs_start;
        while (i < list.len()) : (i += 1) {
            const elem = list.at(i);

            // Check if this is a (region ...) form
            if (elem.type == .list) {
                const elem_list = elem.data.list;
                if (!elem_list.isEmpty()) {
                    const elem_first = elem_list.at(0);
                    if (elem_first.type == .identifier and
                        std.mem.eql(u8, elem_first.data.atom, "region")) {
                        // Parse as region with parent op name for implicit terminators
                        const region = try self.parseTerseRegionWithParent(elem, op_name);
                        try regions.append(self.allocator, region);
                        continue;
                    }
                }
            }

            // Otherwise, must be an operand (value ID)
            if (elem.type != .value_id) {
                std.debug.print("\nerror: Expected value ID or (region ...) in terse operation at position {d}, got {s}\n", .{ i, @tagName(elem.type) });
                std.debug.print("Problematic expression: ", .{});
                const printer_mod = @import("printer.zig");
                var printer = printer_mod.Printer.init(self.allocator);
                defer printer.deinit();
                printer.printValue(elem) catch {};
                std.debug.print("{s}\n", .{printer.getOutput()});

                var printer2 = printer_mod.Printer.init(self.allocator);
                defer printer2.deinit();
                std.debug.print("Full terse operation: ", .{});
                printer2.printValue(value) catch {};
                std.debug.print("{s}\n", .{printer2.getOutput()});

                if (elem.type == .identifier) {
                    std.debug.print("Found identifier '{s}' - did you mean '%{s}'?\n", .{ elem.data.atom, elem.data.atom });
                }
                return error.ExpectedValueId;
            }
            try operands.append(self.allocator, elem.data.atom);
        }

        return Operation{
            .name = op_name,
            .result_bindings = &[_][]const u8{}, // No explicit bindings in terse syntax
            .result_types = &[_]TypeExpr{},      // No explicit types in terse syntax
            .operands = try operands.toOwnedSlice(self.allocator),
            .attributes = attributes,
            .successors = &[_]Successor{},
            .regions = try regions.toOwnedSlice(self.allocator),
            .location = null,
        };
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
                std.debug.print("Error: Expected type, function_type, or identifier in result-types at position {d}, but got {s}\n", .{ i, @tagName(type_value.type) });
                // Print value for atom-based types
                switch (type_value.type) {
                    .identifier, .number, .string, .value_id, .block_id, .symbol, .keyword => {
                        std.debug.print("  Value: {s}\n", .{type_value.data.atom});
                    },
                    .type => {
                        std.debug.print("  Type: {s}\n", .{type_value.data.type});
                    },
                    else => {},
                }
                return error.UnexpectedStructure;
            }

            // Use the value directly - parser doesn't own it
            try types.append(self.allocator, TypeExpr{ .value = type_value });
        }

        return types.toOwnedSlice(self.allocator);
    }

    fn parseOperands(self: *Parser, section: *Value) ParseError![][]const u8 {
        // (operands VALUE_ID*)
        const list = section.data.list;
        var operands = std.ArrayList([]const u8){};
        errdefer operands.deinit(self.allocator);

        var i: usize = 1;
        while (i < list.len()) : (i += 1) {
            const val = list.at(i);
            if (val.type != .value_id) {
                std.debug.print("Error: Expected value ID in operands at position {d}, but got {s}\n", .{ i, @tagName(val.type) });
                if (val.type == .identifier or val.type == .symbol or val.type == .string or val.type == .number) {
                    std.debug.print("  Found atom: '{s}'\n", .{val.data.atom});
                } else if (val.type == .list) {
                    std.debug.print("  Found nested list with {d} elements\n", .{val.data.list.len()});
                    if (val.data.list.len() > 0) {
                        const first = val.data.list.at(0);
                        if (first.type == .identifier) {
                            std.debug.print("  List starts with: '{s}'\n", .{first.data.atom});
                        }
                    }
                }
                return error.ExpectedValueId;
            }
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
                std.debug.print("Map has {} elements total, currently parsing at pair index {}\n", .{map.len(), i/2});
                std.debug.print("\nAll map elements:\n", .{});
                var j: usize = 0;
                while (j < map.len()) : (j += 2) {
                    const elem_key = map.at(j);
                    const elem_val = if (j + 1 < map.len()) map.at(j + 1) else null;
                    const ek_type = @tagName(elem_key.type);
                    const ek_str = if (elem_key.type == .identifier or elem_key.type == .keyword)
                        elem_key.data.atom
                    else if (elem_key.type == .list)
                        "(list)"
                    else if (elem_key.type == .type)
                        "(type)"
                    else if (elem_key.type == .function_type)
                        "(function_type)"
                    else
                        "(complex)";

                    if (elem_val) |ev| {
                        const ev_type = @tagName(ev.type);
                        const ev_str = if (ev.type == .identifier or ev.type == .keyword)
                            ev.data.atom
                        else if (ev.type == .list)
                            "(list)"
                        else if (ev.type == .type)
                            "(type)"
                        else if (ev.type == .function_type)
                            "(function_type)"
                        else
                            "(complex)";
                        const marker = if (j == i) ">>> " else "    ";
                        std.debug.print("{s}[{}]: key={s}:{s} => val={s}:{s}\n", .{marker, j/2, ek_type, ek_str, ev_type, ev_str});
                    } else {
                        std.debug.print("  [{}]: key={s}:{s} => (missing value)\n", .{j/2, ek_type, ek_str});
                    }
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

    fn parseRegions(self: *Parser, section: *Value, parent_op_name: []const u8) ParseError![]Region {
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
            const region = try self.parseRegion(region_value, parent_op_name);
            try regions.append(self.allocator, region);
        }

        return regions.toOwnedSlice(self.allocator);
    }

    /// Check if an operation is a terminator
    fn isTerminator(op_name: []const u8) bool {
        return std.mem.eql(u8, op_name, "func.return") or
            std.mem.eql(u8, op_name, "scf.yield") or
            std.mem.eql(u8, op_name, "scf.condition") or
            std.mem.eql(u8, op_name, "cf.br") or
            std.mem.eql(u8, op_name, "cf.cond_br") or
            std.mem.eql(u8, op_name, "cf.switch");
    }

    /// Get the appropriate terminator name for an operation
    fn getTerminatorForOp(op_name: []const u8) []const u8 {
        // For scf.* operations, use scf.yield
        if (std.mem.startsWith(u8, op_name, "scf.")) {
            return "scf.yield";
        }
        // For func.func, use func.return
        if (std.mem.startsWith(u8, op_name, "func.")) {
            return "func.return";
        }
        // Default to scf.yield for unknown operations
        return "scf.yield";
    }

    /// Insert an implicit terminator at the end of a block if needed
    /// For terse syntax: if last operation doesn't produce a terminator,
    /// auto-insert one that yields the result of the last operation
    fn insertImplicitTerminator(
        self: *Parser,
        operations: *std.ArrayList(Operation),
        parent_op_name: []const u8,
    ) !void {
        if (operations.items.len == 0) {
            // Empty block - insert terminator with no operands
            const terminator_name = getTerminatorForOp(parent_op_name);
            const terminator = Operation{
                .name = terminator_name,
                .result_bindings = &[_][]const u8{},
                .result_types = &[_]TypeExpr{},
                .operands = &[_][]const u8{},
                .attributes = &[_]Attribute{},
                .successors = &[_]Successor{},
                .regions = &[_]Region{},
                .location = null,
            };
            try operations.append(self.allocator, terminator);
            return;
        }

        var last_op = &operations.items[operations.items.len - 1];

        // If last operation is already a terminator, we're done
        if (isTerminator(last_op.name)) {
            return;
        }

        // Insert terminator that yields the last operation's result
        const terminator_name = getTerminatorForOp(parent_op_name);

        // Determine what to yield: if last operation has result bindings, yield those
        // Otherwise, generate a binding for the last operation if it could produce a result
        const operands = if (last_op.result_bindings.len > 0) blk: {
            // Allocate and copy result bindings
            const ops = try self.allocator.alloc([]const u8, last_op.result_bindings.len);
            for (last_op.result_bindings, 0..) |binding, i| {
                ops[i] = binding;
            }
            break :blk ops;
        } else if (self.operationProducesResults(last_op.name)) blk: {
            // Last operation could produce a result but has no binding
            // Generate a binding for it
            const gensym_counter = @intFromPtr(last_op); // Use pointer as unique counter
            const binding = try std.fmt.allocPrint(
                self.allocator,
                "%implicit_result_{d}",
                .{gensym_counter},
            );

            // Add the binding to the last operation
            const bindings = try self.allocator.alloc([]const u8, 1);
            bindings[0] = binding;
            last_op.result_bindings = bindings;

            // Yield the binding
            const ops = try self.allocator.alloc([]const u8, 1);
            ops[0] = binding;
            break :blk ops;
        } else blk: {
            // Last operation doesn't produce results (e.g., side-effecting ops)
            // Create empty yield
            break :blk try self.allocator.alloc([]const u8, 0);
        };

        const terminator = Operation{
            .name = terminator_name,
            .result_bindings = &[_][]const u8{},
            .result_types = &[_]TypeExpr{},
            .operands = operands,
            .attributes = &[_]Attribute{},
            .successors = &[_]Successor{},
            .regions = &[_]Region{},
            .location = null,
        };
        try operations.append(self.allocator, terminator);
    }

    /// Check if an operation typically produces results
    /// This is a heuristic - most operations produce results except terminators,
    /// some side-effecting operations (memref.store), and control flow
    fn operationProducesResults(self: *Parser, op_name: []const u8) bool {
        _ = self;

        // Terminators don't produce results
        if (isTerminator(op_name)) return false;

        // Known side-effecting operations that don't produce results
        if (std.mem.eql(u8, op_name, "memref.store")) return false;
        if (std.mem.eql(u8, op_name, "gpu.terminator")) return false;
        if (std.mem.eql(u8, op_name, "gpu.launch")) return false;

        // Most other operations produce results
        return true;
    }

    /// Parse a region in terse syntax with parent operation name for implicit terminators
    fn parseTerseRegionWithParent(
        self: *Parser,
        value: *Value,
        parent_op_name: []const u8,
    ) ParseError!Region {
        if (value.type != .list) return error.ExpectedList;

        const list = value.data.list;
        if (list.isEmpty()) return error.UnexpectedStructure;

        const first = list.at(0);
        if (first.type != .identifier) return error.ExpectedIdentifier;
        if (!std.mem.eql(u8, first.data.atom, "region")) return error.UnexpectedStructure;

        // Terse syntax: Create implicit block with operations
        var operations = std.ArrayList(Operation){};
        errdefer {
            for (operations.items) |*op| {
                op.deinit(self.allocator);
            }
            operations.deinit(self.allocator);
        }

        var i: usize = 1;
        while (i < list.len()) : (i += 1) {
            const op_value = list.at(i);

            // Handle bare value IDs - they should be yielded directly
            if (op_value.type == .value_id) {
                // This is a bare value ID like %val - create implicit yield
                const yield_name = getTerminatorForOp(parent_op_name);
                const value_id = op_value.data.atom;

                // Allocate operands array
                const operands = try self.allocator.alloc([]const u8, 1);
                operands[0] = value_id;

                const yield_op = Operation{
                    .name = yield_name,
                    .result_bindings = &[_][]const u8{},
                    .result_types = &[_]TypeExpr{},
                    .operands = operands,
                    .attributes = &[_]Attribute{},
                    .successors = &[_]Successor{},
                    .regions = &[_]Region{},
                    .location = null,
                };
                try operations.append(self.allocator, yield_op);
                continue;
            }

            if (op_value.type != .list) continue;

            const op_list = op_value.data.list;
            if (op_list.isEmpty()) continue;

            const op_first = op_list.at(0);
            if (op_first.type != .identifier) continue;

            const op_name = op_first.data.atom;
            const operation = if (std.mem.eql(u8, op_name, "operation"))
                try self.parseOperation(op_value)
            else if (std.mem.eql(u8, op_name, "declare"))
                try self.parseDeclare(op_value)
            else if (isTerseOperation(op_name))
                try self.parseTerseOperation(op_value)
            else
                continue;

            try operations.append(self.allocator, operation);
        }

        // Insert implicit terminator if needed (for regions with operations but no terminator)
        try self.insertImplicitTerminator(&operations, parent_op_name);

        const block = Block{
            .label = null,
            .arguments = &[_]Argument{},
            .operations = try operations.toOwnedSlice(self.allocator),
        };

        const blocks = try self.allocator.alloc(Block, 1);
        blocks[0] = block;

        return Region{
            .blocks = blocks,
        };
    }

    fn parseRegion(self: *Parser, value: *Value, parent_op_name: []const u8) ParseError!Region {
        // (region BLOCK+) - verbose
        // OR
        // (region OPERATION*) - terse (implicit block with no args)
        if (value.type != .list) return error.ExpectedList;

        const list = value.data.list;
        if (list.isEmpty()) return error.UnexpectedStructure;

        const first = list.at(0);
        if (first.type != .identifier) return error.ExpectedIdentifier;
        if (!std.mem.eql(u8, first.data.atom, "region")) return error.UnexpectedStructure;

        // Check if this is terse (no explicit blocks) or verbose (explicit blocks)
        const is_terse = if (list.len() > 1) blk: {
            const second = list.at(1);
            if (second.type == .list) {
                const second_list = second.data.list;
                if (!second_list.isEmpty()) {
                    const second_first = second_list.at(0);
                    if (second_first.type == .identifier) {
                        // If it starts with "block", it's verbose
                        break :blk !std.mem.eql(u8, second_first.data.atom, "block");
                    }
                }
            }
            // Not a list or doesn't start with identifier - assume terse
            break :blk true;
        } else blk: {
            break :blk true; // Empty region is terse
        };

        if (is_terse) {
            // Terse syntax: Use parseTerseRegionWithParent for proper implicit terminator handling
            return try self.parseTerseRegionWithParent(value, parent_op_name);
        } else {
            // Verbose syntax: Parse explicit blocks
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
    }

    fn parseBlock(self: *Parser, value: *Value) ParseError!Block {
        // (block [ BLOCK_ID ] (arguments [ [ VALUE_ID TYPE ]* ]) OPERATION*)
        if (value.type != .list) return error.ExpectedList;

        const list = value.data.list;
        if (list.isEmpty()) return error.UnexpectedStructure;

        const first = list.at(0);
        if (first.type != .identifier) return error.ExpectedIdentifier;
        if (!std.mem.eql(u8, first.data.atom, "block")) {
            std.debug.print("ERROR: Expected 'block' but got '{s}'\n", .{first.data.atom});
            std.debug.print("This suggests a region contains operations directly instead of blocks\n", .{});
            std.debug.print("Regions must contain (block ...) wrappers\n", .{});
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

            const item_name = item_first.data.atom;
            if (std.mem.eql(u8, item_name, "arguments")) {
                block.arguments = try self.parseArguments(item);
            } else if (std.mem.eql(u8, item_name, "operation") or
                      std.mem.eql(u8, item_name, "declare") or
                      isTerseOperation(item_name)) {
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

                    const op_name = op_first.data.atom;
                    if (std.mem.eql(u8, op_name, "operation")) {
                        const op = try self.parseOperation(op_value);
                        try operations.append(self.allocator, op);
                    } else if (std.mem.eql(u8, op_name, "declare")) {
                        const op = try self.parseDeclare(op_value);
                        try operations.append(self.allocator, op);
                    } else if (isTerseOperation(op_name)) {
                        const op = try self.parseTerseOperation(op_value);
                        try operations.append(self.allocator, op);
                    }
                }
                block.operations = try operations.toOwnedSlice(self.allocator);
                break;
            }
        }

        return block;
    }

    fn parseArguments(self: *Parser, section: *Value) ParseError![]Argument {
        // (arguments [ (: VALUE_ID TYPE)* ])
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
            const arg_value = vec.at(i);
            if (arg_value.type != .has_type) {
                std.debug.print("\nError: Expected block argument with type annotation\n", .{});
                std.debug.print("Found: {s}\n", .{@tagName(arg_value.type)});
                std.debug.print("\nExpected syntax: (arguments [ (: %%arg_name type) ... ])\n", .{});
                std.debug.print("Example: (arguments [ (: %%arg0 !llvm.ptr) (: %%arg1 i32) ])\n", .{});
                std.debug.print("\nIncorrect syntax: (arguments [[%%arg0 !llvm.ptr] [%%arg1 i32]])\n\n", .{});
                return error.ExpectedHasType;
            }

            const value_id = arg_value.data.has_type.value;
            const type_expr = arg_value.data.has_type.type_expr;

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
