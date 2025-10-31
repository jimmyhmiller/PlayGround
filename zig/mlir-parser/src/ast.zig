//! MLIR Abstract Syntax Tree (AST)
//! Defines node types for all MLIR grammar productions
//! Each type corresponds to a production rule in grammar.ebnf

const std = @import("std");

/// Grammar: toplevel ::= (operation | attribute-alias-def | type-alias-def)*
pub const Module = struct {
    operations: []Operation,
    type_aliases: []TypeAliasDef,
    attribute_aliases: []AttributeAliasDef,
    allocator: std.mem.Allocator,

    pub fn init(allocator: std.mem.Allocator) Module {
        return .{
            .operations = &[_]Operation{},
            .type_aliases = &[_]TypeAliasDef{},
            .attribute_aliases = &[_]AttributeAliasDef{},
            .allocator = allocator,
        };
    }

    pub fn deinit(self: *Module) void {
        for (self.operations) |*op| {
            op.deinit(self.allocator);
        }
        self.allocator.free(self.operations);
        self.allocator.free(self.type_aliases);
        self.allocator.free(self.attribute_aliases);
    }
};

/// Grammar: operation ::= op-result-list? (generic-operation | custom-operation) trailing-location?
pub const Operation = struct {
    results: ?OpResultList,
    kind: OperationKind,
    location: ?Location,

    pub fn deinit(self: *Operation, allocator: std.mem.Allocator) void {
        if (self.results) |results| {
            allocator.free(results.results);
        }
        self.kind.deinit(allocator);
        if (self.location) |_| {
            // TODO: deinit location
        }
    }
};

pub const OperationKind = union(enum) {
    generic: GenericOperation,

    pub fn deinit(self: *OperationKind, allocator: std.mem.Allocator) void {
        switch (self.*) {
            .generic => |*g| g.deinit(allocator),
        }
    }
};

/// Grammar: generic-operation ::= string-literal `(` value-use-list? `)` successor-list?
///                                 dictionary-properties? region-list? dictionary-attribute?
///                                 `:` function-type
pub const GenericOperation = struct {
    name: []const u8,
    operands: []ValueUse,
    successors: []Successor,
    properties: ?DictionaryAttribute,
    regions: []Region,
    attributes: ?DictionaryAttribute,
    function_type: FunctionType,

    pub fn deinit(self: *GenericOperation, allocator: std.mem.Allocator) void {
        allocator.free(self.operands);
        allocator.free(self.successors);

        for (self.regions) |*region| {
            var mut_region = region.*;
            mut_region.deinit(allocator);
        }
        allocator.free(self.regions);

        var mutable_ft = self.function_type;
        mutable_ft.deinit(allocator);

        if (self.attributes) |*attrs| {
            for (attrs.entries) |entry| {
                _ = entry; // Entries are just slices into source, no need to free
            }
            allocator.free(attrs.entries);
        }

        if (self.properties) |*props| {
            for (props.entries) |entry| {
                _ = entry; // Entries are just slices into source, no need to free
            }
            allocator.free(props.entries);
        }
    }
};

/// Grammar: op-result-list ::= op-result (`,` op-result)* `=`
pub const OpResultList = struct {
    results: []OpResult,
};

/// Grammar: op-result ::= value-id (`:` integer-literal)?
pub const OpResult = struct {
    value_id: []const u8,
    num_results: ?u64,
};

/// Grammar: value-use ::= value-id (`#` decimal-literal)?
pub const ValueUse = struct {
    value_id: []const u8,
    result_number: ?u64,
};

/// Grammar: successor ::= caret-id (`:` block-arg-list)?
pub const Successor = struct {
    block_id: []const u8,
    args: ?BlockArgList,
};

/// Grammar: block-arg-list ::= `(` value-id-and-type-list? `)`
pub const BlockArgList = struct {
    args: []ValueIdAndType,
};

/// Grammar: value-id-and-type ::= value-id `:` type
pub const ValueIdAndType = struct {
    value_id: []const u8,
    type: Type,
};

/// Grammar: block ::= block-label operation+
pub const Block = struct {
    label: BlockLabel,
    operations: []Operation,

    pub fn deinit(self: *Block, allocator: std.mem.Allocator) void {
        for (self.operations) |*op| {
            op.deinit(allocator);
        }
        allocator.free(self.operations);
    }
};

/// Grammar: block-label ::= block-id block-arg-list? `:`
pub const BlockLabel = struct {
    block_id: []const u8,
    args: ?BlockArgList,
};

/// Grammar: region ::= `{` entry-block? block* `}`
pub const Region = struct {
    entry_block: ?[]Operation, // entry-block ::= operation+
    blocks: []Block,

    pub fn deinit(self: *Region, allocator: std.mem.Allocator) void {
        if (self.entry_block) |ops| {
            for (ops) |*op| {
                op.deinit(allocator);
            }
            allocator.free(ops);
        }
        for (self.blocks) |*blk| {
            blk.deinit(allocator);
        }
        allocator.free(self.blocks);
    }
};

/// Grammar: type ::= type-alias | dialect-type | builtin-type
pub const Type = union(enum) {
    type_alias: []const u8,
    dialect: DialectType,
    builtin: BuiltinType,
    function: FunctionType,

    pub fn deinit(self: *Type, allocator: std.mem.Allocator) void {
        switch (self.*) {
            .function => |*ft| ft.deinit(allocator),
            .builtin => |*bt| bt.deinit(allocator),
            else => {},
        }
    }
};

/// Grammar: function-type ::= (type | type-list-parens) `->` (type | type-list-parens)
pub const FunctionType = struct {
    inputs: []Type,
    outputs: []Type,

    pub fn deinit(self: *FunctionType, allocator: std.mem.Allocator) void {
        for (self.inputs) |*t| {
            t.deinit(allocator);
        }
        for (self.outputs) |*t| {
            t.deinit(allocator);
        }
        allocator.free(self.inputs);
        allocator.free(self.outputs);
    }
};

/// Grammar: type-alias-def ::= `!` alias-name `=` type
/// Note: We store the type as an opaque string to handle all syntax forms
pub const TypeAliasDef = struct {
    alias_name: []const u8,
    type_value: []const u8, // Opaque string - everything after '='
};

/// Grammar: dialect-type ::= `!` (opaque-dialect-type | pretty-dialect-type)
pub const DialectType = struct {
    namespace: []const u8,
    body: ?[]const u8,
};

/// Builtin types (integer, float, index, tensor, memref, vector, etc.)
pub const BuiltinType = union(enum) {
    // Grammar: signless-integer-type ::= `i` [1-9][0-9]*
    integer: IntegerType,
    // Grammar: float types: f16, f32, f64, etc.
    float: FloatType,
    // Grammar: index-type ::= `index`
    index,
    // Grammar: tensor-type ::= `tensor` `<` dimension-list type (`,` encoding)? `>`
    tensor: TensorType,
    // Grammar: memref-type ::= `memref` `<` ... `>`
    memref: MemRefType,
    // Grammar: vector-type ::= `vector` `<` ... `>`
    vector: VectorType,
    // Grammar: complex-type ::= `complex` `<` type `>`
    complex: *Type,
    // Grammar: tuple-type ::= `tuple` `<` (type (`,` type)*)? `>`
    tuple: []Type,
    // Grammar: none-type ::= `none`
    none,

    pub fn deinit(self: *BuiltinType, allocator: std.mem.Allocator) void {
        switch (self.*) {
            .tensor => |*t| {
                allocator.free(t.dimensions);
                t.element_type.deinit(allocator);
                allocator.destroy(t.element_type);
            },
            .memref => |*m| {
                allocator.free(m.dimensions);
                m.element_type.deinit(allocator);
                allocator.destroy(m.element_type);
            },
            .vector => |*v| {
                allocator.free(v.dimensions);
                v.element_type.deinit(allocator);
                allocator.destroy(v.element_type);
            },
            .complex => |c| {
                c.deinit(allocator);
                allocator.destroy(c);
            },
            .tuple => |t| {
                for (t) |*typ| {
                    typ.deinit(allocator);
                }
                allocator.free(t);
            },
            else => {},
        }
    }
};

pub const IntegerType = struct {
    signedness: Signedness,
    width: u64,

    pub const Signedness = enum {
        signless, // i32
        signed, // si32
        unsigned, // ui32
    };
};

pub const FloatType = enum {
    f16,
    f32,
    f64,
    f80,
    f128,
    bf16,
    tf32,
};

pub const TensorType = struct {
    dimensions: []Dimension,
    element_type: *Type,
    encoding: ?AttributeValue,

    pub const Dimension = union(enum) {
        static: u64,
        dynamic, // ?
    };
};

pub const MemRefType = struct {
    dimensions: []TensorType.Dimension,
    element_type: *Type,
    layout: ?[]const u8, // Layout specification (simplified for now)
    memory_space: ?AttributeValue,
};

pub const VectorType = struct {
    dimensions: []VectorDimension,
    element_type: *Type,

    pub const VectorDimension = union(enum) {
        fixed: u64,
        scalable: u64, // [n]
    };
};

/// Grammar: attribute-entry ::= (bare-id | string-literal) (`=` attribute-value)?
/// Note: When value is null, this represents a unit attribute (bare identifier)
pub const AttributeEntry = struct {
    name: []const u8,
    value: ?AttributeValue,
};

/// Grammar: attribute-value ::= attribute-alias | dialect-attribute | builtin-attribute
pub const AttributeValue = union(enum) {
    alias: []const u8,
    dialect: DialectAttribute,
    builtin: BuiltinAttribute,
};

/// Grammar: attribute-alias-def ::= `#` alias-name `=` attribute-value
/// Note: We store the attribute as an opaque string to handle all syntax forms
pub const AttributeAliasDef = struct {
    alias_name: []const u8,
    attr_value: []const u8, // Opaque string - everything after '='
};

/// Grammar: dialect-attribute ::= `#` (opaque-dialect-attribute | pretty-dialect-attribute)
pub const DialectAttribute = struct {
    namespace: []const u8,
    body: ?[]const u8,
};

/// Builtin attributes (integers, floats, strings, arrays, dictionaries, etc.)
pub const BuiltinAttribute = union(enum) {
    integer: i64,
    float: f64,
    string: []const u8,
    boolean: bool,
    array: []AttributeValue,
    dictionary: []AttributeEntry,
    // More builtin attributes can be added as needed
};

/// Grammar: dictionary-attribute ::= `{` (attribute-entry (`,` attribute-entry)*)? `}`
pub const DictionaryAttribute = struct {
    entries: []AttributeEntry,
};

/// Location information (simplified)
pub const Location = struct {
    source: []const u8,
};

test "AST - Module creation" {
    var module = Module.init(std.testing.allocator);
    defer module.deinit();

    try std.testing.expectEqual(@as(usize, 0), module.operations.len);
}

test "AST - Integer type" {
    const int_type = Type{
        .builtin = .{
            .integer = .{
                .signedness = .signless,
                .width = 32,
            },
        },
    };

    try std.testing.expectEqual(@as(u64, 32), int_type.builtin.integer.width);
}
