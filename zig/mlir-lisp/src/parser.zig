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
    UnexpectedStructure,
    MissingRequiredField,
    InvalidSectionName,
} || std.mem.Allocator.Error;

/// Top-level MLIR module containing operations
pub const MlirModule = struct {
    operations: []Operation,
    allocator: std.mem.Allocator,

    pub fn deinit(self: *MlirModule) void {
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

    pub fn init(allocator: std.mem.Allocator) Parser {
        return Parser{ .allocator = allocator };
    }

    /// Parse top-level MLIR module
    pub fn parseModule(self: *Parser, value: *Value) ParseError!MlirModule {
        // Expect (mlir OPERATION*)
        if (value.type != .list) return error.ExpectedList;

        const list = value.data.list;
        if (list.isEmpty()) return error.UnexpectedStructure;

        const first = list.at(0);
        if (first.type != .identifier) return error.ExpectedIdentifier;
        if (!std.mem.eql(u8, first.data.atom, "mlir")) {
            return error.UnexpectedStructure;
        }

        // Parse operations
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
            const op = try self.parseOperation(op_value);
            try operations.append(self.allocator, op);
        }

        return MlirModule{
            .operations = try operations.toOwnedSlice(self.allocator),
            .allocator = self.allocator,
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

            if (key.type != .keyword) return error.ExpectedKeyword;

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
