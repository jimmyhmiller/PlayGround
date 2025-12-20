const std = @import("std");
const reader_types = @import("reader_types.zig");

/// AST Node types
pub const NodeType = enum {
    Module,
    Operation,
    Region,
    Block,
    Def,
    Let,
    TypeAnnotation,
    FunctionType,
    Literal,
};

/// Type representation
pub const Type = struct {
    name: []const u8,
    allocator: std.mem.Allocator,

    pub fn init(allocator: std.mem.Allocator, name: []const u8) !*Type {
        const t = try allocator.create(Type);
        t.* = .{
            .name = try allocator.dupe(u8, name),
            .allocator = allocator,
        };
        return t;
    }

    pub fn deinit(self: *Type) void {
        self.allocator.free(self.name);
        self.allocator.destroy(self);
    }
};

/// Function type (-> [args] [returns])
pub const FunctionType = struct {
    arg_types: std.ArrayList(*Type),
    return_types: std.ArrayList(*Type),
    allocator: std.mem.Allocator,

    pub fn init(allocator: std.mem.Allocator) *FunctionType {
        const ft = allocator.create(FunctionType) catch unreachable;
        ft.* = .{
            .arg_types = std.ArrayList(*Type){},
            .return_types = std.ArrayList(*Type){},
            .allocator = allocator,
        };
        return ft;
    }

    pub fn deinit(self: *FunctionType) void {
        for (self.arg_types.items) |t| {
            t.deinit();
        }
        self.arg_types.deinit(self.allocator);

        for (self.return_types.items) |t| {
            t.deinit();
        }
        self.return_types.deinit(self.allocator);

        self.allocator.destroy(self);
    }
};

/// Typed number (number with explicit type)
pub const TypedNumber = struct {
    value: f64,
    typ: *Type,
};

/// Attribute value (for operation attributes)
pub const AttributeValue = union(enum) {
    string: []const u8,
    number: f64,
    boolean: bool,
    array: std.ArrayList(AttributeValue),
    type: *Type,
    function_type: *FunctionType,
    typed_number: TypedNumber,
};

/// Module (top-level container)
pub const Module = struct {
    body: std.ArrayList(*Node),
    allocator: std.mem.Allocator,

    pub fn init(allocator: std.mem.Allocator) !*Module {
        const module = try allocator.create(Module);
        module.* = .{
            .body = std.ArrayList(*Node){},
            .allocator = allocator,
        };
        return module;
    }

    pub fn deinit(self: *Module) void {
        for (self.body.items) |n| {
            n.deinit();
        }
        self.body.deinit(self.allocator);
        self.allocator.destroy(self);
    }
};

/// Operation (MLIR operation)
pub const Operation = struct {
    name: []const u8,
    namespace: ?[]const u8,
    attributes: std.StringHashMap(AttributeValue),
    operands: std.ArrayList(*Node),
    regions: std.ArrayList(*Region),
    result_types: std.ArrayList(*Type),
    allocator: std.mem.Allocator,

    pub fn init(allocator: std.mem.Allocator, name: []const u8, namespace: ?[]const u8) !*Operation {
        const op = try allocator.create(Operation);
        op.* = .{
            .name = try allocator.dupe(u8, name),
            .namespace = if (namespace) |ns| try allocator.dupe(u8, ns) else null,
            .attributes = std.StringHashMap(AttributeValue).init(allocator),
            .operands = std.ArrayList(*Node){},
            .regions = std.ArrayList(*Region){},
            .result_types = std.ArrayList(*Type){},
            .allocator = allocator,
        };
        return op;
    }

    pub fn deinit(self: *Operation) void {
        self.allocator.free(self.name);
        if (self.namespace) |ns| {
            self.allocator.free(ns);
        }

        var it = self.attributes.iterator();
        while (it.next()) |entry| {
            self.allocator.free(entry.key_ptr.*);
            deinitAttributeValue(entry.value_ptr.*, self.allocator);
        }
        self.attributes.deinit();

        for (self.operands.items) |operand| {
            operand.deinit();
        }
        self.operands.deinit(self.allocator);

        for (self.regions.items) |region| {
            region.deinit();
        }
        self.regions.deinit(self.allocator);

        for (self.result_types.items) |t| {
            t.deinit();
        }
        self.result_types.deinit(self.allocator);

        self.allocator.destroy(self);
    }

    fn deinitAttributeValue(value_: AttributeValue, allocator: std.mem.Allocator) void {
        var value = value_;
        switch (value) {
            .string => |s| allocator.free(s),
            .array => |*arr| {
                for (arr.items) |item| {
                    deinitAttributeValue(item, allocator);
                }
                arr.deinit(allocator);
            },
            .type => |t| t.deinit(),
            .function_type => |ft| ft.deinit(),
            .typed_number => |tn| tn.typ.deinit(),
            else => {},
        }
    }

    pub fn getQualifiedName(self: *const Operation, allocator: std.mem.Allocator) ![]const u8 {
        if (self.namespace) |ns| {
            return std.fmt.allocPrint(allocator, "{s}.{s}", .{ ns, self.name });
        }
        return allocator.dupe(u8, self.name);
    }
};

/// Region (contains blocks)
pub const Region = struct {
    blocks: std.ArrayList(*Block),
    allocator: std.mem.Allocator,

    pub fn init(allocator: std.mem.Allocator) !*Region {
        const region = try allocator.create(Region);
        region.* = .{
            .blocks = std.ArrayList(*Block){},
            .allocator = allocator,
        };
        return region;
    }

    pub fn deinit(self: *Region) void {
        for (self.blocks.items) |block| {
            block.deinit();
        }
        self.blocks.deinit(self.allocator);
        self.allocator.destroy(self);
    }
};

/// Block argument
pub const BlockArgument = struct {
    name: []const u8,
    type: ?*Type,
    allocator: std.mem.Allocator,

    pub fn init(allocator: std.mem.Allocator, name: []const u8, arg_type: ?*Type) !*BlockArgument {
        const arg = try allocator.create(BlockArgument);
        arg.* = .{
            .name = try allocator.dupe(u8, name),
            .type = arg_type,
            .allocator = allocator,
        };
        return arg;
    }

    pub fn deinit(self: *BlockArgument) void {
        self.allocator.free(self.name);
        if (self.type) |t| {
            t.deinit();
        }
        self.allocator.destroy(self);
    }
};

/// Block (labeled block with arguments)
pub const Block = struct {
    label: ?[]const u8,
    arguments: std.ArrayList(*BlockArgument),
    operations: std.ArrayList(*Node),
    allocator: std.mem.Allocator,

    pub fn init(allocator: std.mem.Allocator, label: ?[]const u8) !*Block {
        const block = try allocator.create(Block);
        block.* = .{
            .label = if (label) |l| try allocator.dupe(u8, l) else null,
            .arguments = std.ArrayList(*BlockArgument){},
            .operations = std.ArrayList(*Node){},
            .allocator = allocator,
        };
        return block;
    }

    pub fn deinit(self: *Block) void {
        if (self.label) |l| {
            self.allocator.free(l);
        }

        for (self.arguments.items) |arg| {
            arg.deinit();
        }
        self.arguments.deinit(self.allocator);

        for (self.operations.items) |op| {
            op.deinit();
        }
        self.operations.deinit(self.allocator);

        self.allocator.destroy(self);
    }
};

/// Binding (def or let)
pub const Binding = struct {
    names: std.ArrayList([]const u8),
    value: *Node,
    allocator: std.mem.Allocator,

    pub fn init(allocator: std.mem.Allocator) !*Binding {
        const binding = try allocator.create(Binding);
        binding.* = .{
            .names = std.ArrayList([]const u8){},
            .value = undefined, // Must be set after creation
            .allocator = allocator,
        };
        return binding;
    }

    pub fn deinit(self: *Binding) void {
        for (self.names.items) |name| {
            self.allocator.free(name);
        }
        self.names.deinit(self.allocator);
        self.value.deinit();
        self.allocator.destroy(self);
    }
};

/// Let expression: bindings vector and body expressions
pub const LetExpr = struct {
    bindings: std.ArrayList(*Binding),
    body: std.ArrayList(*Node),
    allocator: std.mem.Allocator,

    pub fn init(allocator: std.mem.Allocator) !*LetExpr {
        const le = try allocator.create(LetExpr);
        le.* = .{
            .bindings = std.ArrayList(*Binding){},
            .body = std.ArrayList(*Node){},
            .allocator = allocator,
        };
        return le;
    }

    pub fn deinit(self: *LetExpr) void {
        for (self.bindings.items) |b| {
            b.deinit();
        }
        self.bindings.deinit(self.allocator);

        for (self.body.items) |n| {
            n.deinit();
        }
        self.body.deinit(self.allocator);

        self.allocator.destroy(self);
    }
};

/// Literal value
pub const Literal = struct {
    value: reader_types.Value,

    pub fn deinit(self: *Literal, allocator: std.mem.Allocator) void {
        // Note: Value owns its own memory
        _ = allocator;
        _ = self;
    }
};

/// AST Node
pub const Node = struct {
    node_type: NodeType,
    data: union {
        module: *Module,
        operation: *Operation,
        region: *Region,
        block: *Block,
        binding: *Binding,
        let_expr: *LetExpr,
        type_annotation: struct {
            value: *Node,
            typ: *Type,
        },
        function_type: *FunctionType,
        literal: *reader_types.Value,
    },
    allocator: std.mem.Allocator,

    pub fn createModule(allocator: std.mem.Allocator, module: *Module) !*Node {
        const node = try allocator.create(Node);
        node.* = .{
            .node_type = .Module,
            .data = .{ .module = module },
            .allocator = allocator,
        };
        return node;
    }

    pub fn createOperation(allocator: std.mem.Allocator, operation: *Operation) !*Node {
        const node = try allocator.create(Node);
        node.* = .{
            .node_type = .Operation,
            .data = .{ .operation = operation },
            .allocator = allocator,
        };
        return node;
    }

    pub fn createRegion(allocator: std.mem.Allocator, region: *Region) !*Node {
        const node = try allocator.create(Node);
        node.* = .{
            .node_type = .Region,
            .data = .{ .region = region },
            .allocator = allocator,
        };
        return node;
    }

    pub fn createBlock(allocator: std.mem.Allocator, block: *Block) !*Node {
        const node = try allocator.create(Node);
        node.* = .{
            .node_type = .Block,
            .data = .{ .block = block },
            .allocator = allocator,
        };
        return node;
    }

    pub fn createBinding(allocator: std.mem.Allocator, binding: *Binding) !*Node {
        const node = try allocator.create(Node);
        node.* = .{
            .node_type = .Def,
            .data = .{ .binding = binding },
            .allocator = allocator,
        };
        return node;
    }

    pub fn createLet(allocator: std.mem.Allocator, let_expr: *LetExpr) !*Node {
        const node = try allocator.create(Node);
        node.* = .{
            .node_type = .Let,
            .data = .{ .let_expr = let_expr },
            .allocator = allocator,
        };
        return node;
    }

    pub fn createLiteral(allocator: std.mem.Allocator, value: *reader_types.Value) !*Node {
        const node = try allocator.create(Node);
        node.* = .{
            .node_type = .Literal,
            .data = .{ .literal = value },
            .allocator = allocator,
        };
        return node;
    }

    pub fn createTypeAnnotation(allocator: std.mem.Allocator, value: *Node, typ: *Type) !*Node {
        const node = try allocator.create(Node);
        node.* = .{
            .node_type = .TypeAnnotation,
            .data = .{ .type_annotation = .{ .value = value, .typ = typ } },
            .allocator = allocator,
        };
        return node;
    }

    pub fn createFunctionType(allocator: std.mem.Allocator, ft: *FunctionType) !*Node {
        const node = try allocator.create(Node);
        node.* = .{
            .node_type = .FunctionType,
            .data = .{ .function_type = ft },
            .allocator = allocator,
        };
        return node;
    }

    pub fn deinit(self: *Node) void {
        switch (self.node_type) {
            .Module => self.data.module.deinit(),
            .Operation => self.data.operation.deinit(),
            .Region => self.data.region.deinit(),
            .Block => self.data.block.deinit(),
            .Def => self.data.binding.deinit(),
            .Let => self.data.let_expr.deinit(),
            .TypeAnnotation => {
                self.data.type_annotation.value.deinit();
                self.data.type_annotation.typ.deinit();
            },
            .FunctionType => self.data.function_type.deinit(),
            .Literal => {}, // Don't deinit - values are owned by the values list
        }
        self.allocator.destroy(self);
    }
};
