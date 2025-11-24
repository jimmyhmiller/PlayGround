const std = @import("std");
const reader_types = @import("reader_types.zig");
const ast = @import("ast.zig");

const Value = reader_types.Value;
const ValueType = reader_types.ValueType;
const Symbol = reader_types.Symbol;
const Node = ast.Node;
const Operation = ast.Operation;
const Region = ast.Region;
const Block = ast.Block;
const Binding = ast.Binding;
const Type = ast.Type;
const FunctionType = ast.FunctionType;
const AttributeValue = ast.AttributeValue;
const BlockArgument = ast.BlockArgument;

pub const ParserError = error{
    ExpectedSymbol,
    InvalidDefForm,
    InvalidDefName,
    InvalidLetForm,
    LetBindingsMustBeVector,
    InvalidTypeAnnotation,
    InvalidType,
    InvalidFunctionType,
    OutOfMemory,
};

pub const Parser = struct {
    allocator: std.mem.Allocator,

    pub fn init(allocator: std.mem.Allocator) Parser {
        return .{
            .allocator = allocator,
        };
    }

    /// Parse a list of reader values into AST nodes
    pub fn parse(self: *Parser, values: std.ArrayList(*Value)) !std.ArrayList(*Node) {
        var nodes = std.ArrayList(*Node){};
        errdefer {
            for (nodes.items) |node| {
                node.deinit();
            }
            nodes.deinit(self.allocator);
        }

        for (values.items) |value| {
            const node = try self.parseValue(value);
            try nodes.append(self.allocator, node);
        }

        return nodes;
    }

    fn parseValue(self: *Parser, value: *Value) ParserError!*Node {
        return switch (value.type) {
            .List => try self.parseList(value),
            .Symbol => try self.parseSymbolRef(value),
            .Number, .String, .Boolean, .Nil, .Keyword => try Node.createLiteral(self.allocator, value),
            .Vector => try Node.createLiteral(self.allocator, value),
            .Map => try Node.createLiteral(self.allocator, value),
        };
    }

    fn parseList(self: *Parser, value: *Value) !*Node {
        std.debug.assert(value.type == .List);

        if (value.data.list.items.len == 0) {
            return Node.createLiteral(self.allocator, value);
        }

        const first = value.data.list.items[0];
        if (first.type != .Symbol) {
            return error.ExpectedSymbol;
        }

        const sym = first.data.symbol;

        // Check for special forms
        if (std.mem.eql(u8, sym.name, "module")) {
            return try self.parseModule(value);
        }
        if (std.mem.eql(u8, sym.name, "do")) {
            return try self.parseRegion(value);
        }
        if (std.mem.eql(u8, sym.name, "region")) {
            return try self.parseRegion(value);
        }
        if (std.mem.eql(u8, sym.name, "block")) {
            return try self.parseBlock(value);
        }
        if (std.mem.eql(u8, sym.name, "def")) {
            return try self.parseDef(value);
        }
        if (std.mem.eql(u8, sym.name, "let")) {
            return try self.parseLet(value);
        }
        if (std.mem.eql(u8, sym.name, ":")) {
            return try self.parseTypeAnnotation(value);
        }
        if (std.mem.eql(u8, sym.name, "->")) {
            return try self.parseFunctionType(value);
        }
        if (std.mem.eql(u8, sym.name, "require-dialect") or std.mem.eql(u8, sym.name, "use-dialect")) {
            // These are processed during reading, just return a literal
            return Node.createLiteral(self.allocator, value);
        }

        // Otherwise, it's an operation
        return try self.parseOperation(value);
    }

    fn parseModule(self: *Parser, value: *Value) !*Node {
        // (module (require-dialect ...) (do ...))
        const module = try ast.Module.init(self.allocator);
        errdefer module.deinit();

        // Parse the module body
        for (value.data.list.items[1..]) |item| {
            if (item.type == .List and item.data.list.items.len > 0) {
                const first = item.data.list.items[0];
                if (first.type == .Symbol) {
                    // Skip require-dialect and use-dialect
                    if (std.mem.eql(u8, first.data.symbol.name, "require-dialect") or
                        std.mem.eql(u8, first.data.symbol.name, "use-dialect"))
                    {
                        continue;
                    }
                }
            }

            const node = try self.parseValue(item);
            try module.body.append(self.allocator, node);
        }

        return Node.createModule(self.allocator, module);
    }

    fn parseOperation(self: *Parser, value: *Value) !*Node {
        std.debug.assert(value.type == .List);

        const first = value.data.list.items[0];
        const sym = first.data.symbol;

        // Get namespace if present
        const namespace = if (sym.namespace) |ns| ns.name else null;

        const op = try Operation.init(self.allocator, sym.name, namespace);
        errdefer op.deinit();

        var i: usize = 1;

        // Check for attributes map (second element if it's a map)
        if (i < value.data.list.items.len and value.data.list.items[i].type == .Map) {
            try self.parseAttributes(value.data.list.items[i], &op.attributes);
            i += 1;
        }

        // Parse operands and regions
        while (i < value.data.list.items.len) : (i += 1) {
            const item = value.data.list.items[i];

            // Check if it's a region (do block)
            if (item.type == .List and item.data.list.items.len > 0) {
                const region_first = item.data.list.items[0];
                if (region_first.type == .Symbol and std.mem.eql(u8, region_first.data.symbol.name, "do")) {
                    const region_node = try self.parseRegion(item);
                    try op.regions.append(self.allocator, region_node.data.region);
                    self.allocator.destroy(region_node);
                    continue;
                }
            }

            // Otherwise, it's an operand
            const node = try self.parseValue(item);
            try op.operands.append(self.allocator, node);
        }

        return Node.createOperation(self.allocator, op);
    }

    fn parseAttributes(self: *Parser, value: *Value, attributes: *std.StringHashMap(AttributeValue)) !void {
        std.debug.assert(value.type == .Map);

        var it = value.data.map.iterator();
        while (it.next()) |entry| {
            const key = try self.allocator.dupe(u8, entry.key_ptr.*);
            const attr_value = try self.parseAttributeValue(entry.value_ptr.*);
            try attributes.put(key, attr_value);
        }
    }

    fn parseAttributeValue(self: *Parser, value: *Value) !AttributeValue {
        return switch (value.type) {
            .String => AttributeValue{ .string = try self.allocator.dupe(u8, value.data.string) },
            .Number => AttributeValue{ .number = value.data.number },
            .Boolean => AttributeValue{ .boolean = value.data.boolean },
            .Vector => blk: {
                var arr = std.ArrayList(AttributeValue){};
                for (value.data.vector.items) |item| {
                    const attr_val = try self.parseAttributeValue(item);
                    try arr.append(self.allocator, attr_val);
                }
                break :blk AttributeValue{ .array = arr };
            },
            .Symbol => blk: {
                // Check if it's a type
                const type_value = try Type.init(self.allocator, value.data.symbol.name);
                break :blk AttributeValue{ .type = type_value };
            },
            .List => blk: {
                // Check if it's a function type
                if (value.data.list.items.len > 0 and value.data.list.items[0].type == .Symbol) {
                    const first_sym = value.data.list.items[0].data.symbol.name;
                    if (std.mem.eql(u8, first_sym, "->")) {
                        const ft = try self.parseFunctionTypeValue(value);
                        break :blk AttributeValue{ .function_type = ft };
                    }
                }
                break :blk AttributeValue{ .string = try self.allocator.dupe(u8, "<list>") };
            },
            else => AttributeValue{ .string = try self.allocator.dupe(u8, "<unknown>") },
        };
    }

    fn parseRegion(self: *Parser, value: *Value) !*Node {
        std.debug.assert(value.type == .List);

        const region = try Region.init(self.allocator);
        errdefer region.deinit();

        // Parse blocks in the region
        for (value.data.list.items[1..]) |item| {
            if (item.type == .List and item.data.list.items.len > 0) {
                const first = item.data.list.items[0];
                if (first.type == .Symbol and std.mem.eql(u8, first.data.symbol.name, "block")) {
                    const block_node = try self.parseBlock(item);
                    try region.blocks.append(self.allocator, block_node.data.block);
                    self.allocator.destroy(block_node);
                    continue;
                }
            }

            // If not a block, treat as an operation in an implicit block
            if (region.blocks.items.len == 0) {
                const implicit_block = try Block.init(self.allocator, null);
                try region.blocks.append(self.allocator, implicit_block);
            }

            const node = try self.parseValue(item);
            const last_block = region.blocks.items[region.blocks.items.len - 1];
            try last_block.operations.append(self.allocator, node);
        }

        return Node.createRegion(self.allocator, region);
    }

    fn parseBlock(self: *Parser, value: *Value) !*Node {
        std.debug.assert(value.type == .List);

        var label: ?[]const u8 = null;
        var arg_start: usize = 1;
        var has_label = false;

        // Check for label
        if (value.data.list.items.len > 1 and value.data.list.items[1].type == .Symbol) {
            const label_sym = value.data.list.items[1].data.symbol;
            if (label_sym.name[0] == '^') {
                // Block.init will take ownership by duplicating the string itself.
                label = label_sym.name;
                arg_start = 2;
                has_label = true;
            }
        }

        const block = try Block.init(self.allocator, label);
        errdefer block.deinit();

        // Parse arguments (if present as a vector)
        const args_index = arg_start;
        if (args_index < value.data.list.items.len and value.data.list.items[args_index].type == .Vector) {
            const args_vec = value.data.list.items[args_index];
            for (args_vec.data.vector.items) |arg_item| {
                // Arguments can be symbols or type annotations
                var arg_name: []const u8 = undefined;
                var arg_type: ?*Type = null;

                if (arg_item.type == .Symbol) {
                    arg_name = arg_item.data.symbol.name;
                } else if (arg_item.type == .List) {
                    // (: name type)
                    if (arg_item.data.list.items.len >= 3 and
                        arg_item.data.list.items[0].type == .Symbol and
                        std.mem.eql(u8, arg_item.data.list.items[0].data.symbol.name, ":"))
                    {
                        arg_name = arg_item.data.list.items[1].data.symbol.name;
                        const type_val = arg_item.data.list.items[2];
                        if (type_val.type == .Symbol) {
                            arg_type = try Type.init(self.allocator, type_val.data.symbol.name);
                        }
                    } else {
                        continue;
                    }
                } else {
                    continue;
                }

                const block_arg = try BlockArgument.init(self.allocator, arg_name, arg_type);
                try block.arguments.append(self.allocator, block_arg);
            }
            arg_start = args_index + 1;
        } else {
            arg_start = if (has_label) args_index else 1;
        }

        // Parse operations in the block
        for (value.data.list.items[arg_start..]) |item| {
            const node = try self.parseValue(item);
            try block.operations.append(self.allocator, node);
        }

        return Node.createBlock(self.allocator, block);
    }

    fn parseDef(self: *Parser, value: *Value) !*Node {
        // (def name value)
        if (value.data.list.items.len < 3) {
            return error.InvalidDefForm;
        }

        const binding = try Binding.init(self.allocator);
        errdefer binding.deinit();

        // Get name
        const name_val = value.data.list.items[1];
        if (name_val.type == .Symbol) {
            const name = try self.allocator.dupe(u8, name_val.data.symbol.name);
            try binding.names.append(self.allocator, name);
        } else if (name_val.type == .Vector) {
            // Destructuring: (def [a b] ...)
            for (name_val.data.vector.items) |item| {
                if (item.type == .Symbol) {
                    const name = try self.allocator.dupe(u8, item.data.symbol.name);
                    try binding.names.append(self.allocator, name);
                }
            }
        } else {
            return error.InvalidDefName;
        }

        // Parse value
        binding.value = try self.parseValue(value.data.list.items[2]);

        return Node.createBinding(self.allocator, binding);
    }

    fn parseLet(self: *Parser, value: *Value) !*Node {
        // (let [bindings...] body...)
        if (value.data.list.items.len < 3) {
            return error.InvalidLetForm;
        }

        const let_expr = try ast.LetExpr.init(self.allocator);
        errdefer let_expr.deinit();

        // Parse bindings
        const bindings_vec = value.data.list.items[1];
        if (bindings_vec.type != .Vector) {
            return error.LetBindingsMustBeVector;
        }

        if (bindings_vec.data.vector.items.len % 2 != 0) {
            return error.InvalidLetForm;
        }

        var i: usize = 0;
        while (i < bindings_vec.data.vector.items.len) : (i += 2) {
            const name_val = bindings_vec.data.vector.items[i];
            const value_val = bindings_vec.data.vector.items[i + 1];

            const binding = try Binding.init(self.allocator);
            errdefer binding.deinit();

            if (name_val.type == .Symbol) {
                const name = try self.allocator.dupe(u8, name_val.data.symbol.name);
                try binding.names.append(self.allocator, name);
            } else if (name_val.type == .Vector) {
                for (name_val.data.vector.items) |item| {
                    if (item.type == .Symbol) {
                        const name = try self.allocator.dupe(u8, item.data.symbol.name);
                        try binding.names.append(self.allocator, name);
                    }
                }
            } else {
                return error.InvalidDefName;
            }

            binding.value = try self.parseValue(value_val);
            try let_expr.bindings.append(self.allocator, binding);
        }

        // Parse body
        for (value.data.list.items[2..]) |item| {
            const node = try self.parseValue(item);
            try let_expr.body.append(self.allocator, node);
        }

        return Node.createLet(self.allocator, let_expr);
    }

    fn parseTypeAnnotation(self: *Parser, value: *Value) !*Node {
        // (: value type)
        if (value.data.list.items.len < 3) {
            return error.InvalidTypeAnnotation;
        }

        const val_node = try self.parseValue(value.data.list.items[1]);
        errdefer val_node.deinit();

        const type_val = value.data.list.items[2];
        var typ: *Type = undefined;

        if (type_val.type == .Symbol) {
            typ = try Type.init(self.allocator, type_val.data.symbol.name);
        } else if (type_val.type == .List) {
            // Could be a function type or complex type
            typ = try Type.init(self.allocator, "<complex>");
        } else {
            return error.InvalidType;
        }

        return Node.createTypeAnnotation(self.allocator, val_node, typ);
    }

    fn parseFunctionTypeValue(self: *Parser, value: *Value) !*FunctionType {
        // (-> [args...] [returns...])
        if (value.data.list.items.len < 3) {
            return error.InvalidFunctionType;
        }

        const ft = FunctionType.init(self.allocator);
        errdefer ft.deinit();

        // Parse argument types
        const args_vec = value.data.list.items[1];
        if (args_vec.type == .Vector) {
            for (args_vec.data.vector.items) |type_val| {
                if (type_val.type == .Symbol) {
                    const typ = try Type.init(self.allocator, type_val.data.symbol.name);
                    try ft.arg_types.append(self.allocator, typ);
                }
            }
        }

        // Parse return types
        const returns_vec = value.data.list.items[2];
        if (returns_vec.type == .Vector) {
            for (returns_vec.data.vector.items) |type_val| {
                if (type_val.type == .Symbol) {
                    const typ = try Type.init(self.allocator, type_val.data.symbol.name);
                    try ft.return_types.append(self.allocator, typ);
                }
            }
        }

        // ft is now fully owned by caller
        return ft;
    }

    fn parseFunctionType(self: *Parser, value: *Value) !*Node {
        const ft = try self.parseFunctionTypeValue(value);
        return Node.createFunctionType(self.allocator, ft);
    }

    fn parseSymbolRef(self: *Parser, value: *Value) !*Node {
        // Symbol references become literals
        return Node.createLiteral(self.allocator, value);
    }
};

test "parser simple operation" {
    const allocator = std.testing.allocator;
    const tokenizer_mod = @import("tokenizer.zig");
    const reader_mod = @import("reader.zig");

    var tok = tokenizer_mod.Tokenizer.init(allocator, "(arith.addi x y)");
    var tokens = try tok.tokenize();
    defer tokens.deinit(allocator);

    var reader = reader_mod.Reader.init(allocator, tokens.items);
    defer reader.deinit();

    var values = try reader.read();
    defer {
        for (values.items) |v| {
            v.deinit();
        }
        values.deinit(allocator);
    }

    var parser = Parser.init(allocator);
    var nodes = try parser.parse(values);
    defer {
        for (nodes.items) |n| {
            n.deinit();
        }
        nodes.deinit(allocator);
    }

    try std.testing.expectEqual(@as(usize, 1), nodes.items.len);
    try std.testing.expectEqual(ast.NodeType.Operation, nodes.items[0].node_type);

    const op = nodes.items[0].data.operation;
    try std.testing.expectEqualStrings("addi", op.name);
}

test "parser with attributes" {
    const allocator = std.testing.allocator;
    const tokenizer_mod = @import("tokenizer.zig");
    const reader_mod = @import("reader.zig");

    var tok = tokenizer_mod.Tokenizer.init(allocator, "(arith.constant {:value 42})");
    var tokens = try tok.tokenize();
    defer tokens.deinit(allocator);

    var reader = reader_mod.Reader.init(allocator, tokens.items);
    defer reader.deinit();

    var values = try reader.read();
    defer {
        for (values.items) |v| {
            v.deinit();
        }
        values.deinit(allocator);
    }

    var parser = Parser.init(allocator);
    var nodes = try parser.parse(values);
    defer {
        for (nodes.items) |n| {
            n.deinit();
        }
        nodes.deinit(allocator);
    }

    try std.testing.expectEqual(@as(usize, 1), nodes.items.len);

    const op = nodes.items[0].data.operation;
    try std.testing.expect(op.attributes.contains("value"));
}

test "parser def binding" {
    const allocator = std.testing.allocator;
    const tokenizer_mod = @import("tokenizer.zig");
    const reader_mod = @import("reader.zig");

    var tok = tokenizer_mod.Tokenizer.init(allocator, "(def x 42)");
    var tokens = try tok.tokenize();
    defer tokens.deinit(allocator);

    var reader = reader_mod.Reader.init(allocator, tokens.items);
    defer reader.deinit();

    var values = try reader.read();
    defer {
        for (values.items) |v| {
            v.deinit();
        }
        values.deinit(allocator);
    }

    var parser = Parser.init(allocator);
    var nodes = try parser.parse(values);
    defer {
        for (nodes.items) |n| {
            n.deinit();
        }
        nodes.deinit(allocator);
    }

    try std.testing.expectEqual(@as(usize, 1), nodes.items.len);
    try std.testing.expectEqual(ast.NodeType.Def, nodes.items[0].node_type);

    const binding = nodes.items[0].data.binding;
    try std.testing.expectEqual(@as(usize, 1), binding.names.items.len);
    try std.testing.expectEqualStrings("x", binding.names.items[0]);
}

test "parser let form produces Let node with bindings" {
    const allocator = std.testing.allocator;
    const tokenizer_mod = @import("tokenizer.zig");
    const reader_mod = @import("reader.zig");

    var tok = tokenizer_mod.Tokenizer.init(allocator, "(let [x 1 y 2] (+ x y))");
    var tokens = try tok.tokenize();
    defer tokens.deinit(allocator);

    var reader = reader_mod.Reader.init(allocator, tokens.items);
    defer reader.deinit();

    var values = try reader.read();
    defer {
        for (values.items) |v| {
            v.deinit();
        }
        values.deinit(allocator);
    }

    var parser = Parser.init(allocator);
    var nodes = try parser.parse(values);
    defer {
        for (nodes.items) |n| {
            n.deinit();
        }
        nodes.deinit(allocator);
    }

    // Documentation claims let is a special form that builds a binding node.
    try std.testing.expectEqual(@as(usize, 1), nodes.items.len);
    try std.testing.expectEqual(ast.NodeType.Let, nodes.items[0].node_type);
}

test "parser function type captures argument and return types" {
    const allocator = std.testing.allocator;
    const tokenizer_mod = @import("tokenizer.zig");
    const reader_mod = @import("reader.zig");

    var tok = tokenizer_mod.Tokenizer.init(allocator, "(-> [i32 f64] [i1])");
    var tokens = try tok.tokenize();
    defer tokens.deinit(allocator);

    var reader = reader_mod.Reader.init(allocator, tokens.items);
    defer reader.deinit();

    var values = try reader.read();
    defer {
        for (values.items) |v| {
            v.deinit();
        }
        values.deinit(allocator);
    }

    var parser = Parser.init(allocator);
    var nodes = try parser.parse(values);
    defer {
        for (nodes.items) |n| {
            n.deinit();
        }
        nodes.deinit(allocator);
    }

    // Documentation promises a FunctionType node with captured arg/return types.
    try std.testing.expectEqual(@as(usize, 1), nodes.items.len);
    try std.testing.expectEqual(ast.NodeType.FunctionType, nodes.items[0].node_type);
}

test "parser region special form is recognized" {
    const allocator = std.testing.allocator;
    const tokenizer_mod = @import("tokenizer.zig");
    const reader_mod = @import("reader.zig");

    // IMPLEMENTATION_STATUS.md lists `(region ...)` as a special form,
    // so parsing it should yield a Region node rather than a plain operation.
    var tok = tokenizer_mod.Tokenizer.init(allocator,
        \\(region
        \\  (block ^entry
        \\    (arith.addi 1 2)))
    );
    var tokens = try tok.tokenize();
    defer tokens.deinit(allocator);

    var reader = reader_mod.Reader.init(allocator, tokens.items);
    defer reader.deinit();

    var values = try reader.read();
    defer {
        for (values.items) |v| {
            v.deinit();
        }
        values.deinit(allocator);
    }

    var parser = Parser.init(allocator);
    var nodes = try parser.parse(values);
    defer {
        for (nodes.items) |n| {
            n.deinit();
        }
        nodes.deinit(allocator);
    }

    try std.testing.expectEqual(@as(usize, 1), nodes.items.len);
    try std.testing.expectEqual(ast.NodeType.Region, nodes.items[0].node_type);
}

test "parser module builds a Module node" {
    const allocator = std.testing.allocator;
    const tokenizer_mod = @import("tokenizer.zig");
    const reader_mod = @import("reader.zig");

    // Status document claims Module is a first-class AST node.
    var tok = tokenizer_mod.Tokenizer.init(allocator,
        \\(module
        \\  (do
        \\    (block
        \\      (arith.addi 1 2))))
    );
    var tokens = try tok.tokenize();
    defer tokens.deinit(allocator);

    var reader = reader_mod.Reader.init(allocator, tokens.items);
    defer reader.deinit();

    var values = try reader.read();
    defer {
        for (values.items) |v| {
            v.deinit();
        }
        values.deinit(allocator);
    }

    var parser = Parser.init(allocator);
    var nodes = try parser.parse(values);
    defer {
        for (nodes.items) |n| {
            n.deinit();
        }
        nodes.deinit(allocator);
    }

    try std.testing.expectEqual(@as(usize, 1), nodes.items.len);
    try std.testing.expectEqual(ast.NodeType.Module, nodes.items[0].node_type);
}

test "parser block label is not treated as an operation" {
    const allocator = std.testing.allocator;
    const tokenizer_mod = @import("tokenizer.zig");
    const reader_mod = @import("reader.zig");

    // Blocks are advertised as labeled containers with arguments/ops,
    // so the label should not show up as an extra operation.
    var tok = tokenizer_mod.Tokenizer.init(allocator,
        \\(block ^entry
        \\  (arith.addi 1 2))
    );
    var tokens = try tok.tokenize();
    defer tokens.deinit(allocator);

    var reader = reader_mod.Reader.init(allocator, tokens.items);
    defer reader.deinit();

    var values = try reader.read();
    defer {
        for (values.items) |v| {
            v.deinit();
        }
        values.deinit(allocator);
    }

    var parser = Parser.init(allocator);
    var nodes = try parser.parse(values);
    defer {
        for (nodes.items) |n| {
            n.deinit();
        }
        nodes.deinit(allocator);
    }

    try std.testing.expectEqual(@as(usize, 1), nodes.items.len);
    try std.testing.expectEqual(ast.NodeType.Block, nodes.items[0].node_type);
    try std.testing.expectEqual(@as(usize, 1), nodes.items[0].data.block.operations.items.len);
}

test "parser type annotation attaches type info" {
    const allocator = std.testing.allocator;
    const tokenizer_mod = @import("tokenizer.zig");
    const reader_mod = @import("reader.zig");

    var tok = tokenizer_mod.Tokenizer.init(allocator, "(: 42 i32)");
    var tokens = try tok.tokenize();
    defer tokens.deinit(allocator);

    var reader = reader_mod.Reader.init(allocator, tokens.items);
    defer reader.deinit();

    var values = try reader.read();
    defer {
        for (values.items) |v| {
            v.deinit();
        }
        values.deinit(allocator);
    }

    var parser = Parser.init(allocator);
    var nodes = try parser.parse(values);
    defer {
        for (nodes.items) |n| {
            n.deinit();
        }
        nodes.deinit(allocator);
    }

    try std.testing.expectEqual(@as(usize, 1), nodes.items.len);
    try std.testing.expectEqual(ast.NodeType.TypeAnnotation, nodes.items[0].node_type);
    try std.testing.expectEqualStrings("i32", nodes.items[0].data.type_annotation.typ.name);
    try std.testing.expectEqual(ast.NodeType.Literal, nodes.items[0].data.type_annotation.value.node_type);
}

test "parser function type preserves argument and return types" {
    const allocator = std.testing.allocator;
    const tokenizer_mod = @import("tokenizer.zig");
    const reader_mod = @import("reader.zig");

    var tok = tokenizer_mod.Tokenizer.init(allocator, "(-> [i32 f64] [i1 i64])");
    var tokens = try tok.tokenize();
    defer tokens.deinit(allocator);

    var reader = reader_mod.Reader.init(allocator, tokens.items);
    defer reader.deinit();

    var values = try reader.read();
    defer {
        for (values.items) |v| {
            v.deinit();
        }
        values.deinit(allocator);
    }

    var parser = Parser.init(allocator);
    var nodes = try parser.parse(values);
    defer {
        for (nodes.items) |n| {
            n.deinit();
        }
        nodes.deinit(allocator);
    }

    try std.testing.expectEqual(@as(usize, 1), nodes.items.len);
    try std.testing.expectEqual(ast.NodeType.FunctionType, nodes.items[0].node_type);

    const ft = nodes.items[0].data.function_type;
    try std.testing.expectEqual(@as(usize, 2), ft.arg_types.items.len);
    try std.testing.expectEqualStrings("i32", ft.arg_types.items[0].name);
    try std.testing.expectEqualStrings("f64", ft.arg_types.items[1].name);
    try std.testing.expectEqual(@as(usize, 2), ft.return_types.items.len);
    try std.testing.expectEqualStrings("i1", ft.return_types.items[0].name);
    try std.testing.expectEqualStrings("i64", ft.return_types.items[1].name);
}

test "parser block arguments capture types" {
    const allocator = std.testing.allocator;
    const tokenizer_mod = @import("tokenizer.zig");
    const reader_mod = @import("reader.zig");

    var tok = tokenizer_mod.Tokenizer.init(allocator,
        \\(block ^entry [(: x i32) (: y f32)]
        \\  (arith.addi x y))
    );
    var tokens = try tok.tokenize();
    defer tokens.deinit(allocator);

    var reader = reader_mod.Reader.init(allocator, tokens.items);
    defer reader.deinit();

    var values = try reader.read();
    defer {
        for (values.items) |v| {
            v.deinit();
        }
        values.deinit(allocator);
    }

    var parser = Parser.init(allocator);
    var nodes = try parser.parse(values);
    defer {
        for (nodes.items) |n| {
            n.deinit();
        }
        nodes.deinit(allocator);
    }

    try std.testing.expectEqual(@as(usize, 1), nodes.items.len);
    try std.testing.expectEqual(ast.NodeType.Block, nodes.items[0].node_type);

    const block = nodes.items[0].data.block;
    try std.testing.expectEqual(@as(usize, 2), block.arguments.items.len);
    try std.testing.expectEqualStrings("x", block.arguments.items[0].name);
    try std.testing.expect(block.arguments.items[0].type != null);
    try std.testing.expectEqualStrings("i32", block.arguments.items[0].type.?.name);
    try std.testing.expectEqualStrings("y", block.arguments.items[1].name);
    try std.testing.expect(block.arguments.items[1].type != null);
    try std.testing.expectEqualStrings("f32", block.arguments.items[1].type.?.name);
}

test "parser def destructuring captures multiple names" {
    const allocator = std.testing.allocator;
    const tokenizer_mod = @import("tokenizer.zig");
    const reader_mod = @import("reader.zig");

    var tok = tokenizer_mod.Tokenizer.init(allocator, "(def [a b] (arith.multi_result))");
    var tokens = try tok.tokenize();
    defer tokens.deinit(allocator);

    var reader = reader_mod.Reader.init(allocator, tokens.items);
    defer reader.deinit();

    var values = try reader.read();
    defer {
        for (values.items) |v| {
            v.deinit();
        }
        values.deinit(allocator);
    }

    var parser = Parser.init(allocator);
    var nodes = try parser.parse(values);
    defer {
        for (nodes.items) |n| {
            n.deinit();
        }
        nodes.deinit(allocator);
    }

    try std.testing.expectEqual(@as(usize, 1), nodes.items.len);
    try std.testing.expectEqual(ast.NodeType.Def, nodes.items[0].node_type);

    const binding = nodes.items[0].data.binding;
    try std.testing.expectEqual(@as(usize, 2), binding.names.items.len);
    try std.testing.expectEqualStrings("a", binding.names.items[0]);
    try std.testing.expectEqualStrings("b", binding.names.items[1]);
}

test "parser let requires vector bindings" {
    const allocator = std.testing.allocator;
    const tokenizer_mod = @import("tokenizer.zig");
    const reader_mod = @import("reader.zig");

    var tok = tokenizer_mod.Tokenizer.init(allocator, "(let bad 1)");
    var tokens = try tok.tokenize();
    defer tokens.deinit(allocator);

    var reader = reader_mod.Reader.init(allocator, tokens.items);
    defer reader.deinit();

    var values = try reader.read();
    defer {
        for (values.items) |v| {
            v.deinit();
        }
        values.deinit(allocator);
    }

    var parser = Parser.init(allocator);
    try std.testing.expectError(ParserError.LetBindingsMustBeVector, parser.parse(values));
}
