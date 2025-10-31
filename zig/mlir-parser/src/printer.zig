//! MLIR Printer
//! Converts AST back to MLIR text format
//! Each printing function corresponds to a parser function and grammar rule
//! Enables roundtrip testing: parse → print → parse

const std = @import("std");
const ast = @import("ast.zig");

/// Printer converts an AST to MLIR text format
pub const Printer = struct {
    writer: std.io.AnyWriter,
    indent_level: usize,

    pub fn init(writer: std.io.AnyWriter) Printer {
        return .{
            .writer = writer,
            .indent_level = 0,
        };
    }

    /// Grammar: toplevel ::= (operation | attribute-alias-def | type-alias-def)*
    pub fn printModule(self: *Printer, module: ast.Module) !void {
        // Print type aliases first
        for (module.type_aliases) |type_alias| {
            try self.printTypeAliasDef(type_alias);
            try self.writer.writeByte('\n');
        }

        // Print attribute aliases
        for (module.attribute_aliases) |attr_alias| {
            try self.printAttributeAliasDef(attr_alias);
            try self.writer.writeByte('\n');
        }

        // Print operations
        for (module.operations, 0..) |operation, i| {
            try self.printOperation(operation);
            // Add newline between operations except after the last one
            if (i < module.operations.len - 1) {
                try self.writer.writeByte('\n');
            }
        }
    }

    /// Grammar: operation ::= op-result-list? (generic-operation | custom-operation) trailing-location?
    pub fn printOperation(self: *Printer, operation: ast.Operation) anyerror!void {
        // Print result list if present
        if (operation.results) |results| {
            try self.printOpResultList(results);
            try self.writer.writeAll(" = ");
        }

        // Print the operation (generic or custom)
        switch (operation.kind) {
            .generic => |gen_op| try self.printGenericOperation(gen_op),
        }

        // Print trailing location if present
        if (operation.location) |location| {
            try self.writer.writeAll(" loc(");
            try self.writer.writeAll(location.source);
            try self.writer.writeByte(')');
        }
    }

    /// Grammar: op-result-list ::= op-result (`,` op-result)* `=`
    fn printOpResultList(self: *Printer, results: ast.OpResultList) !void {
        for (results.results, 0..) |result, i| {
            if (i > 0) try self.writer.writeAll(", ");
            try self.printOpResult(result);
        }
    }

    /// Grammar: op-result ::= value-id (`:` integer-literal)?
    fn printOpResult(self: *Printer, result: ast.OpResult) !void {
        try self.writer.writeAll(result.value_id);
        if (result.num_results) |num| {
            try self.writer.writeByte(':');
            try self.writer.print("{d}", .{num});
        }
    }

    /// Grammar: generic-operation ::= string-literal `(` value-use-list? `)` successor-list?
    ///                                 dictionary-properties? region-list? dictionary-attribute?
    ///                                 `:` function-type
    fn printGenericOperation(self: *Printer, op: ast.GenericOperation) anyerror!void {
        // Print operation name as quoted string
        try self.writer.writeByte('"');
        try self.writer.writeAll(op.name);
        try self.writer.writeByte('"');

        // Print operands
        try self.writer.writeByte('(');
        for (op.operands, 0..) |operand, i| {
            if (i > 0) try self.writer.writeAll(", ");
            try self.printValueUse(operand);
        }
        try self.writer.writeByte(')');

        // Print successors if present
        if (op.successors.len > 0) {
            try self.writer.writeAll(" [");
            for (op.successors, 0..) |successor, i| {
                if (i > 0) try self.writer.writeAll(", ");
                try self.printSuccessor(successor);
            }
            try self.writer.writeByte(']');
        }

        // Print properties if present
        if (op.properties) |properties| {
            try self.writer.writeAll(" <");
            try self.printDictionaryAttribute(properties);
            try self.writer.writeByte('>');
        }

        // Print regions if present
        if (op.regions.len > 0) {
            try self.writer.writeAll(" (");
            for (op.regions, 0..) |region, i| {
                if (i > 0) try self.writer.writeAll(", ");
                try self.printRegion(region);
            }
            try self.writer.writeByte(')');
        }

        // Print attributes if present
        if (op.attributes) |attributes| {
            try self.writer.writeByte(' ');
            try self.printDictionaryAttribute(attributes);
        }

        // Print function type
        try self.writer.writeAll(" : ");
        try self.printFunctionType(op.function_type);
    }

    /// Grammar: value-use ::= value-id (`#` decimal-literal)?
    fn printValueUse(self: *Printer, value_use: ast.ValueUse) !void {
        try self.writer.writeAll(value_use.value_id);
        if (value_use.result_number) |num| {
            try self.writer.writeByte('#');
            try self.writer.print("{d}", .{num});
        }
    }

    /// Grammar: successor ::= caret-id (`:` block-arg-list)?
    fn printSuccessor(self: *Printer, successor: ast.Successor) !void {
        try self.writer.writeAll(successor.block_id);
        if (successor.args) |args| {
            try self.writer.writeByte(':');
            try self.printBlockArgList(args);
        }
    }

    /// Grammar: block-arg-list ::= `(` value-id-and-type-list? `)`
    fn printBlockArgList(self: *Printer, args: ast.BlockArgList) !void {
        try self.writer.writeByte('(');
        for (args.args, 0..) |arg, i| {
            if (i > 0) try self.writer.writeAll(", ");
            try self.printValueIdAndType(arg);
        }
        try self.writer.writeByte(')');
    }

    /// Grammar: value-id-and-type ::= value-id `:` type
    fn printValueIdAndType(self: *Printer, value_and_type: ast.ValueIdAndType) !void {
        try self.writer.writeAll(value_and_type.value_id);
        try self.writer.writeAll(" : ");
        try self.printType(value_and_type.type);
    }

    /// Grammar: region ::= `{` entry-block? block* `}`
    fn printRegion(self: *Printer, region: ast.Region) anyerror!void {
        try self.writer.writeByte('{');
        self.indent_level += 1;

        // Print entry block if present
        if (region.entry_block) |entry_ops| {
            for (entry_ops) |operation| {
                try self.writer.writeByte('\n');
                try self.printIndent();
                try self.printOperation(operation);
            }
        }

        // Print labeled blocks
        for (region.blocks) |block| {
            try self.writer.writeByte('\n');
            try self.printIndent();
            try self.printBlock(block);
        }

        self.indent_level -= 1;
        if (region.entry_block != null or region.blocks.len > 0) {
            try self.writer.writeByte('\n');
            try self.printIndent();
        }
        try self.writer.writeByte('}');
    }

    /// Grammar: block ::= block-label operation+
    fn printBlock(self: *Printer, block: ast.Block) anyerror!void {
        try self.printBlockLabel(block.label);
        self.indent_level += 1;
        for (block.operations) |operation| {
            try self.writer.writeByte('\n');
            try self.printIndent();
            try self.printOperation(operation);
        }
        self.indent_level -= 1;
    }

    /// Grammar: block-label ::= block-id block-arg-list? `:`
    fn printBlockLabel(self: *Printer, label: ast.BlockLabel) !void {
        try self.writer.writeAll(label.block_id);
        if (label.args) |args| {
            try self.printBlockArgList(args);
        }
        try self.writer.writeByte(':');
    }

    /// Grammar: dictionary-attribute ::= `{` (attribute-entry (`,` attribute-entry)*)? `}`
    fn printDictionaryAttribute(self: *Printer, dict: ast.DictionaryAttribute) !void {
        try self.writer.writeByte('{');
        for (dict.entries, 0..) |entry, i| {
            if (i > 0) try self.writer.writeAll(", ");
            try self.printAttributeEntry(entry);
        }
        try self.writer.writeByte('}');
    }

    /// Grammar: attribute-entry ::= (bare-id | string-literal) `=` attribute-value
    fn printAttributeEntry(self: *Printer, entry: ast.AttributeEntry) !void {
        try self.writer.writeAll(entry.name);
        try self.writer.writeAll(" = ");
        try self.printAttributeValue(entry.value);
    }

    /// Grammar: attribute-value ::= attribute-alias | dialect-attribute | builtin-attribute
    fn printAttributeValue(self: *Printer, value: ast.AttributeValue) anyerror!void {
        switch (value) {
            .alias => |alias| {
                try self.writer.writeByte('#');
                try self.writer.writeAll(alias);
            },
            .dialect => |dialect| {
                try self.writer.writeByte('#');
                try self.printDialectAttribute(dialect);
            },
            .builtin => |builtin| try self.printBuiltinAttribute(builtin),
        }
    }

    /// Grammar: dialect-attribute ::= `#` (opaque-dialect-attribute | pretty-dialect-attribute)
    fn printDialectAttribute(self: *Printer, attr: ast.DialectAttribute) !void {
        try self.writer.writeAll(attr.namespace);
        if (attr.body) |body| {
            try self.writer.writeAll(body);
        }
    }

    /// Print builtin attributes (integers, floats, strings, booleans, arrays)
    fn printBuiltinAttribute(self: *Printer, attr: ast.BuiltinAttribute) !void {
        switch (attr) {
            .integer => |val| try self.writer.print("{d}", .{val}),
            .float => |val| try self.writer.print("{d}", .{val}),
            .string => |str| try self.writer.writeAll(str),
            .boolean => |val| try self.writer.writeAll(if (val) "true" else "false"),
            .array => |arr| {
                try self.writer.writeByte('[');
                for (arr, 0..) |elem, i| {
                    if (i > 0) try self.writer.writeAll(", ");
                    try self.printAttributeValue(elem);
                }
                try self.writer.writeByte(']');
            },
        }
    }

    /// Grammar: type-alias-def ::= `!` alias-name `=` type
    /// Note: We store type as opaque string, so just print it as-is
    fn printTypeAliasDef(self: *Printer, type_alias: ast.TypeAliasDef) !void {
        try self.writer.writeByte('!');
        try self.writer.writeAll(type_alias.alias_name);
        try self.writer.writeAll(" = ");
        try self.writer.writeAll(type_alias.type_value);
    }

    /// Grammar: attribute-alias-def ::= `#` alias-name `=` attribute-value
    /// Note: We store attribute as opaque string, so just print it as-is
    fn printAttributeAliasDef(self: *Printer, attr_alias: ast.AttributeAliasDef) !void {
        try self.writer.writeByte('#');
        try self.writer.writeAll(attr_alias.alias_name);
        try self.writer.writeAll(" = ");
        try self.writer.writeAll(attr_alias.attr_value);
    }

    /// Grammar: type ::= type-alias | dialect-type | builtin-type | function-type
    pub fn printType(self: *Printer, typ: ast.Type) anyerror!void {
        switch (typ) {
            .type_alias => |alias| {
                try self.writer.writeByte('!');
                try self.writer.writeAll(alias);
            },
            .dialect => |dialect| try self.printDialectType(dialect),
            .builtin => |builtin| try self.printBuiltinType(builtin),
            .function => |func| try self.printFunctionType(func),
        }
    }

    /// Grammar: dialect-type ::= `!` (opaque-dialect-type | pretty-dialect-type)
    fn printDialectType(self: *Printer, dialect: ast.DialectType) !void {
        try self.writer.writeByte('!');
        try self.writer.writeAll(dialect.namespace);
        if (dialect.body) |body| {
            try self.writer.writeAll(body);
        }
    }

    /// Print builtin types (integer, float, index, tensor, memref, vector, complex, tuple, none)
    fn printBuiltinType(self: *Printer, builtin: ast.BuiltinType) !void {
        switch (builtin) {
            .integer => |int_type| {
                switch (int_type.signedness) {
                    .signless => try self.writer.writeByte('i'),
                    .signed => try self.writer.writeAll("si"),
                    .unsigned => try self.writer.writeAll("ui"),
                }
                try self.writer.print("{d}", .{int_type.width});
            },
            .float => |float_type| {
                const name = switch (float_type) {
                    .f16 => "f16",
                    .f32 => "f32",
                    .f64 => "f64",
                    .f80 => "f80",
                    .f128 => "f128",
                    .bf16 => "bf16",
                    .tf32 => "tf32",
                };
                try self.writer.writeAll(name);
            },
            .index => try self.writer.writeAll("index"),
            .tensor => |tensor| try self.printTensorType(tensor),
            .memref => |memref| try self.printMemRefType(memref),
            .vector => |vector| try self.printVectorType(vector),
            .complex => |complex_elem| {
                try self.writer.writeAll("complex<");
                try self.printType(complex_elem.*);
                try self.writer.writeByte('>');
            },
            .tuple => |tuple_types| {
                try self.writer.writeAll("tuple<");
                for (tuple_types, 0..) |elem_type, i| {
                    if (i > 0) try self.writer.writeAll(", ");
                    try self.printType(elem_type);
                }
                try self.writer.writeByte('>');
            },
            .none => try self.writer.writeAll("none"),
        }
    }

    /// Grammar: function-type ::= (type | type-list-parens) `->` (type | type-list-parens)
    fn printFunctionType(self: *Printer, func_type: ast.FunctionType) !void {
        // Print input types
        if (func_type.inputs.len == 1) {
            // Single input without parens
            try self.printType(func_type.inputs[0]);
        } else {
            // Multiple inputs or empty with parens
            try self.writer.writeByte('(');
            for (func_type.inputs, 0..) |input, i| {
                if (i > 0) try self.writer.writeAll(", ");
                try self.printType(input);
            }
            try self.writer.writeByte(')');
        }

        try self.writer.writeAll(" -> ");

        // Print output types
        if (func_type.outputs.len == 1) {
            // Single output - check if it's a function type (needs parens for disambiguation)
            const output = func_type.outputs[0];
            const needs_parens = switch (output) {
                .function => true,
                else => false,
            };
            if (needs_parens) try self.writer.writeByte('(');
            try self.printType(output);
            if (needs_parens) try self.writer.writeByte(')');
        } else {
            // Multiple outputs or empty with parens
            try self.writer.writeByte('(');
            for (func_type.outputs, 0..) |output, i| {
                if (i > 0) try self.writer.writeAll(", ");
                try self.printType(output);
            }
            try self.writer.writeByte(')');
        }
    }

    /// Grammar: tensor-type ::= `tensor` `<` dimension-list type (`,` encoding)? `>`
    fn printTensorType(self: *Printer, tensor: ast.TensorType) !void {
        try self.writer.writeAll("tensor<");

        // Print dimensions
        for (tensor.dimensions) |dim| {
            switch (dim) {
                .static => |size| try self.writer.print("{d}", .{size}),
                .dynamic => try self.writer.writeByte('?'),
            }
            try self.writer.writeByte('x');
        }

        // Print element type
        try self.printType(tensor.element_type.*);

        // Print encoding if present
        if (tensor.encoding) |encoding| {
            try self.writer.writeAll(", ");
            try self.printAttributeValue(encoding);
        }

        try self.writer.writeByte('>');
    }

    /// Grammar: memref-type ::= `memref` `<` dimension-list type (`,` layout-specification)? (`,` memory-space)? `>`
    fn printMemRefType(self: *Printer, memref: ast.MemRefType) !void {
        try self.writer.writeAll("memref<");

        // Print dimensions
        for (memref.dimensions) |dim| {
            switch (dim) {
                .static => |size| try self.writer.print("{d}", .{size}),
                .dynamic => try self.writer.writeByte('?'),
            }
            try self.writer.writeByte('x');
        }

        // Print element type
        try self.printType(memref.element_type.*);

        // Print layout if present
        if (memref.layout) |layout| {
            try self.writer.writeAll(", ");
            try self.writer.writeAll(layout);
        }

        // Print memory space if present
        if (memref.memory_space) |mem_space| {
            try self.writer.writeAll(", ");
            try self.printAttributeValue(mem_space);
        }

        try self.writer.writeByte('>');
    }

    /// Grammar: vector-type ::= `vector` `<` vector-dim-list vector-element-type `>`
    fn printVectorType(self: *Printer, vector: ast.VectorType) !void {
        try self.writer.writeAll("vector<");

        // Print dimensions
        for (vector.dimensions) |dim| {
            switch (dim) {
                .fixed => |size| try self.writer.print("{d}", .{size}),
                .scalable => |size| try self.writer.print("[{d}]", .{size}),
            }
            try self.writer.writeByte('x');
        }

        // Print element type
        try self.printType(vector.element_type.*);

        try self.writer.writeByte('>');
    }

    /// Helper: print indentation
    fn printIndent(self: *Printer) !void {
        var i: usize = 0;
        while (i < self.indent_level) : (i += 1) {
            try self.writer.writeAll("  ");
        }
    }
};

/// Convenience function to print a module to a string
pub fn moduleToString(allocator: std.mem.Allocator, module: ast.Module) ![]u8 {
    var buffer: std.ArrayList(u8) = .empty;
    errdefer buffer.deinit(allocator);

    const writer = buffer.writer(allocator).any();
    var printer = Printer.init(writer);
    try printer.printModule(module);

    return buffer.toOwnedSlice(allocator);
}

test "printer - simple type printing" {
    var buffer: std.ArrayList(u8) = .empty;
    errdefer buffer.deinit(std.testing.allocator);

    const writer = buffer.writer(std.testing.allocator).any();
    var printer = Printer.init(writer);

    // Test integer type
    const int_type = ast.Type{
        .builtin = .{
            .integer = .{
                .signedness = .signless,
                .width = 32,
            },
        },
    };
    try printer.printType(int_type);

    defer buffer.deinit(std.testing.allocator);
    try std.testing.expectEqualStrings("i32", buffer.items);
}

test "printer - function type printing" {
    var buffer: std.ArrayList(u8) = .empty;
    errdefer buffer.deinit(std.testing.allocator);

    const writer = buffer.writer(std.testing.allocator).any();
    var printer = Printer.init(writer);

    // Test function type: () -> i32
    var inputs = [_]ast.Type{};
    var outputs = [_]ast.Type{
        .{
            .builtin = .{
                .integer = .{
                    .signedness = .signless,
                    .width = 32,
                },
            },
        },
    };

    const func_type = ast.FunctionType{
        .inputs = inputs[0..],
        .outputs = outputs[0..],
    };

    try printer.printFunctionType(func_type);

    defer buffer.deinit(std.testing.allocator);
    try std.testing.expectEqualStrings("() -> i32", buffer.items);
}
