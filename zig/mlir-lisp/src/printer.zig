const std = @import("std");
const parser = @import("parser.zig");
const reader = @import("reader.zig");
const MlirModule = parser.MlirModule;
const Operation = parser.Operation;
const Region = parser.Region;
const Block = parser.Block;
const Argument = parser.Argument;
const Successor = parser.Successor;
const Attribute = parser.Attribute;
const TypeExpr = parser.TypeExpr;
const AttrExpr = parser.AttrExpr;
const Value = reader.Value;

/// Printer errors
pub const PrintError = error{
    OutOfMemory,
};

/// Printer - converts parsed AST back to s-expression format
pub const Printer = struct {
    allocator: std.mem.Allocator,
    buffer: std.ArrayList(u8),
    indent_level: usize,
    indent_size: usize,

    pub fn init(allocator: std.mem.Allocator) Printer {
        return Printer{
            .allocator = allocator,
            .buffer = std.ArrayList(u8){},
            .indent_level = 0,
            .indent_size = 2,
        };
    }

    pub fn deinit(self: *Printer) void {
        self.buffer.deinit(self.allocator);
    }

    /// Get the printed output
    pub fn getOutput(self: *Printer) []const u8 {
        return self.buffer.items;
    }

    /// Print a module
    pub fn printModule(self: *Printer, module: *const MlirModule) PrintError!void {
        const writer = self.buffer.writer(self.allocator);
        try writer.writeAll("(mlir");
        self.indent_level += 1;

        for (module.operations) |*op| {
            try writer.writeAll("\n");
            try self.writeIndent();
            try self.printOperation(op);
        }

        self.indent_level -= 1;
        try writer.writeAll(")");
    }

    /// Print an operation
    pub fn printOperation(self: *Printer, op: *const Operation) PrintError!void {
        try self.buffer.writer(self.allocator).writeAll("(operation");
        self.indent_level += 1;

        // Name (required)
        try self.buffer.writer(self.allocator).writeAll("\n");
        try self.writeIndent();
        try self.buffer.writer(self.allocator).print("(name {s})", .{op.name});

        // Result bindings
        if (op.result_bindings.len > 0) {
            try self.buffer.writer(self.allocator).writeAll("\n");
            try self.writeIndent();
            try self.buffer.writer(self.allocator).writeAll("(result-bindings [");
            for (op.result_bindings, 0..) |binding, i| {
                if (i > 0) try self.buffer.writer(self.allocator).writeAll(" ");
                try self.buffer.writer(self.allocator).writeAll(binding);
            }
            try self.buffer.writer(self.allocator).writeAll("])");
        }

        // Result types
        if (op.result_types.len > 0) {
            try self.buffer.writer(self.allocator).writeAll("\n");
            try self.writeIndent();
            try self.buffer.writer(self.allocator).writeAll("(result-types");
            for (op.result_types) |*type_expr| {
                try self.buffer.writer(self.allocator).writeAll(" ");
                try self.printValue(type_expr.value);
            }
            try self.buffer.writer(self.allocator).writeAll(")");
        }

        // Operands
        if (op.operands.len > 0) {
            try self.buffer.writer(self.allocator).writeAll("\n");
            try self.writeIndent();
            try self.buffer.writer(self.allocator).writeAll("(operands");
            for (op.operands) |operand| {
                try self.buffer.writer(self.allocator).writeAll(" ");
                try self.buffer.writer(self.allocator).writeAll(operand);
            }
            try self.buffer.writer(self.allocator).writeAll(")");
        }

        // Attributes
        if (op.attributes.len > 0) {
            try self.buffer.writer(self.allocator).writeAll("\n");
            try self.writeIndent();
            try self.buffer.writer(self.allocator).writeAll("(attributes {");
            for (op.attributes) |*attr| {
                try self.buffer.writer(self.allocator).writeAll(" ");
                try self.buffer.writer(self.allocator).print(":{s}", .{attr.key});
                try self.buffer.writer(self.allocator).writeAll(" ");
                try self.printValue(attr.value.value);
            }
            try self.buffer.writer(self.allocator).writeAll(" })");
        }

        // Successors
        if (op.successors.len > 0) {
            try self.buffer.writer(self.allocator).writeAll("\n");
            try self.writeIndent();
            try self.buffer.writer(self.allocator).writeAll("(successors");
            self.indent_level += 1;
            for (op.successors) |*succ| {
                try self.buffer.writer(self.allocator).writeAll("\n");
                try self.writeIndent();
                try self.printSuccessor(succ);
            }
            self.indent_level -= 1;
            try self.buffer.writer(self.allocator).writeAll(")");
        }

        // Regions
        if (op.regions.len > 0) {
            try self.buffer.writer(self.allocator).writeAll("\n");
            try self.writeIndent();
            try self.buffer.writer(self.allocator).writeAll("(regions");
            self.indent_level += 1;
            for (op.regions) |*region| {
                try self.buffer.writer(self.allocator).writeAll("\n");
                try self.writeIndent();
                try self.printRegion(region);
            }
            self.indent_level -= 1;
            try self.buffer.writer(self.allocator).writeAll(")");
        }

        // Location
        if (op.location) |*loc| {
            try self.buffer.writer(self.allocator).writeAll("\n");
            try self.writeIndent();
            try self.buffer.writer(self.allocator).writeAll("(location ");
            try self.printValue(loc.value);
            try self.buffer.writer(self.allocator).writeAll(")");
        }

        self.indent_level -= 1;
        try self.buffer.writer(self.allocator).writeAll(")");
    }

    /// Print a region
    fn printRegion(self: *Printer, region: *const Region) PrintError!void {
        try self.buffer.writer(self.allocator).writeAll("(region");
        self.indent_level += 1;

        for (region.blocks) |*block| {
            try self.buffer.writer(self.allocator).writeAll("\n");
            try self.writeIndent();
            try self.printBlock(block);
        }

        self.indent_level -= 1;
        try self.buffer.writer(self.allocator).writeAll(")");
    }

    /// Print a block
    fn printBlock(self: *Printer, block: *const Block) PrintError!void {
        try self.buffer.writer(self.allocator).writeAll("(block");
        self.indent_level += 1;

        // Block label
        if (block.label) |label| {
            try self.buffer.writer(self.allocator).writeAll("\n");
            try self.writeIndent();
            try self.buffer.writer(self.allocator).print("[{s}]", .{label});
        } else {
            try self.buffer.writer(self.allocator).writeAll("\n");
            try self.writeIndent();
            try self.buffer.writer(self.allocator).writeAll("[]");
        }

        // Arguments
        try self.buffer.writer(self.allocator).writeAll("\n");
        try self.writeIndent();
        try self.buffer.writer(self.allocator).writeAll("(arguments [");
        for (block.arguments, 0..) |*arg, i| {
            if (i > 0) try self.buffer.writer(self.allocator).writeAll(" ");
            try self.buffer.writer(self.allocator).writeAll("[");
            try self.buffer.writer(self.allocator).writeAll(arg.value_id);
            try self.buffer.writer(self.allocator).writeAll(" ");
            try self.printValue(arg.type.value);
            try self.buffer.writer(self.allocator).writeAll("]");
        }
        try self.buffer.writer(self.allocator).writeAll("])");

        // Operations
        for (block.operations) |*op| {
            try self.buffer.writer(self.allocator).writeAll("\n");
            try self.writeIndent();
            try self.printOperation(op);
        }

        self.indent_level -= 1;
        try self.buffer.writer(self.allocator).writeAll(")");
    }

    /// Print a successor
    fn printSuccessor(self: *Printer, succ: *const Successor) PrintError!void {
        try self.buffer.writer(self.allocator).writeAll("(successor ");
        try self.buffer.writer(self.allocator).writeAll(succ.block_id);

        if (succ.operands.len > 0) {
            try self.buffer.writer(self.allocator).writeAll(" (");
            for (succ.operands, 0..) |operand, i| {
                if (i > 0) try self.buffer.writer(self.allocator).writeAll(" ");
                try self.buffer.writer(self.allocator).writeAll(operand);
            }
            try self.buffer.writer(self.allocator).writeAll(")");
        }

        try self.buffer.writer(self.allocator).writeAll(")");
    }

    /// Print a Value (from reader.zig)
    fn printValue(self: *Printer, value: *const Value) PrintError!void {
        switch (value.type) {
            .identifier => try self.buffer.writer(self.allocator).writeAll(value.data.atom),
            .value_id => try self.buffer.writer(self.allocator).writeAll(value.data.atom),
            .block_id => try self.buffer.writer(self.allocator).writeAll(value.data.atom),
            .symbol => try self.buffer.writer(self.allocator).writeAll(value.data.atom),
            .keyword => try self.buffer.writer(self.allocator).print(":{s}", .{value.keywordToName()}),
            .string => try self.buffer.writer(self.allocator).print("\"{s}\"", .{value.data.atom}),
            .number => try self.buffer.writer(self.allocator).writeAll(value.data.atom),
            .true_lit => try self.buffer.writer(self.allocator).writeAll("true"),
            .false_lit => try self.buffer.writer(self.allocator).writeAll("false"),
            .type_expr => {
                try self.buffer.writer(self.allocator).writeAll("!");
                try self.printValue(value.data.type_expr);
            },
            .attr_expr => {
                try self.buffer.writer(self.allocator).writeAll("#");
                try self.printValue(value.data.attr_expr);
            },
            .has_type => {
                // Print as (: value type)
                try self.buffer.writer(self.allocator).writeAll("(: ");
                try self.printValue(value.data.has_type.value);
                try self.buffer.writer(self.allocator).writeAll(" ");
                try self.printValue(value.data.has_type.type_expr);
                try self.buffer.writer(self.allocator).writeAll(")");
            },
            .list => {
                try self.buffer.writer(self.allocator).writeAll("(");
                const list = value.data.list;
                var i: usize = 0;
                while (i < list.len()) : (i += 1) {
                    if (i > 0) try self.buffer.writer(self.allocator).writeAll(" ");
                    try self.printValue(list.at(i));
                }
                try self.buffer.writer(self.allocator).writeAll(")");
            },
            .vector => {
                try self.buffer.writer(self.allocator).writeAll("[");
                const vec = value.data.vector;
                var i: usize = 0;
                while (i < vec.len()) : (i += 1) {
                    if (i > 0) try self.buffer.writer(self.allocator).writeAll(" ");
                    try self.printValue(vec.at(i));
                }
                try self.buffer.writer(self.allocator).writeAll("]");
            },
            .map => {
                try self.buffer.writer(self.allocator).writeAll("{");
                const map = value.data.map;
                var i: usize = 0;
                while (i + 1 < map.len()) : (i += 2) {
                    if (i > 0) try self.buffer.writer(self.allocator).writeAll(" ");
                    try self.printValue(map.at(i));
                    try self.buffer.writer(self.allocator).writeAll(" ");
                    try self.printValue(map.at(i + 1));
                }
                try self.buffer.writer(self.allocator).writeAll("}");
            },
        }
    }

    /// Write current indentation
    fn writeIndent(self: *Printer) PrintError!void {
        const spaces = self.indent_level * self.indent_size;
        var i: usize = 0;
        while (i < spaces) : (i += 1) {
            try self.buffer.writer(self.allocator).writeAll(" ");
        }
    }
};
