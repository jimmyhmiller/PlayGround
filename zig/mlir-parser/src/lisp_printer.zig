//! MLIR to Lisp S-Expression Printer
//! Converts MLIR AST to S-expression Lisp format
//! Each printing function corresponds to the Lisp grammar rules

const std = @import("std");
const ast = @import("ast.zig");

/// LispPrinter converts an AST to S-expression Lisp format
pub const LispPrinter = struct {
    writer: std.io.AnyWriter,
    indent_level: usize,

    pub fn init(writer: std.io.AnyWriter) LispPrinter {
        return .{
            .writer = writer,
            .indent_level = 0,
        };
    }

    fn printIndent(self: *LispPrinter) !void {
        for (0..self.indent_level * 2) |_| {
            try self.writer.writeByte(' ');
        }
    }

    /// Write a string with escaped quotes and backslashes
    /// Escapes " as \" and \ as \\
    fn writeEscapedString(self: *LispPrinter, str: []const u8) !void {
        for (str) |char| {
            switch (char) {
                '"' => try self.writer.writeAll("\\\""),
                '\\' => try self.writer.writeAll("\\\\"),
                else => try self.writer.writeByte(char),
            }
        }
    }

    /// Validate that a dictionary has an even number of key-value pairs
    /// This helps catch bugs where unit attributes are not converted properly
    fn validateDictionary(entries: []ast.AttributeEntry) void {
        // Count the total elements that would be printed
        var element_count: usize = 0;
        for (entries) |_| {
            element_count += 1; // key
            // Every entry should have a value (even unit attrs become 'true')
            element_count += 1; // value
        }

        // Assert that we have an even number (key-value pairs)
        std.debug.assert(element_count % 2 == 0);
        // Since each entry contributes exactly 2 elements (key + value),
        // and we're counting all entries * 2, this should always be even.
        // This assertion catches logic errors in our printing code.
    }

    // Lisp Grammar: MLIR ::= (mlir TYPE-ALIAS* ATTRIBUTE-ALIAS* OPERATION*)
    pub fn printModule(self: *LispPrinter, module: ast.Module) !void {
        try self.writer.writeAll("(mlir");
        self.indent_level += 1;

        // Print type aliases first
        for (module.type_aliases) |type_alias| {
            try self.writer.writeByte('\n');
            try self.printIndent();
            try self.printTypeAlias(type_alias);
        }

        // Print attribute aliases
        for (module.attribute_aliases) |attr_alias| {
            try self.writer.writeByte('\n');
            try self.printIndent();
            try self.printAttributeAlias(attr_alias);
        }

        // Print all operations
        for (module.operations) |operation| {
            try self.writer.writeByte('\n');
            try self.printIndent();
            try self.printOperation(operation);
        }

        self.indent_level -= 1;
        try self.writer.writeByte(')');
    }

    // Lisp Grammar: TYPE-ALIAS ::= (type-alias TYPE_ID STRING)
    fn printTypeAlias(self: *LispPrinter, type_alias: ast.TypeAliasDef) !void {
        try self.writer.writeAll("(type-alias ");
        try self.writer.writeByte('!');
        try self.writer.writeAll(type_alias.alias_name);
        try self.writer.writeAll(" \"");
        try self.writeEscapedString(type_alias.type_value);
        try self.writer.writeAll("\")");
    }

    // Lisp Grammar: ATTRIBUTE-ALIAS ::= (attribute-alias ATTR_ID STRING)
    fn printAttributeAlias(self: *LispPrinter, attr_alias: ast.AttributeAliasDef) !void {
        try self.writer.writeAll("(attribute-alias ");
        try self.writer.writeByte('#');
        try self.writer.writeAll(attr_alias.alias_name);
        try self.writer.writeAll(" \"");
        try self.writeEscapedString(attr_alias.attr_value);
        try self.writer.writeAll("\")");
    }

    // Lisp Grammar: OPERATION ::= (operation (name OP_NAME) SECTION*)
    fn printOperation(self: *LispPrinter, operation: ast.Operation) anyerror!void {
        try self.writer.writeAll("(operation");
        self.indent_level += 1;

        // Always print the name section
        try self.writer.writeByte('\n');
        try self.printIndent();
        try self.writer.writeAll("(name ");
        switch (operation.kind) {
            .generic => |gen_op| try self.writer.writeAll(gen_op.name),
        }
        try self.writer.writeByte(')');

        // Print sections based on what's present
        switch (operation.kind) {
            .generic => |gen_op| {
                // Result bindings section
                if (operation.results) |results| {
                    try self.writer.writeByte('\n');
                    try self.printIndent();
                    try self.printResultBindings(results);
                }

                // Result types section (from function type outputs)
                if (gen_op.function_type.outputs.len > 0) {
                    try self.writer.writeByte('\n');
                    try self.printIndent();
                    try self.printResultTypes(gen_op.function_type.outputs);
                }

                // Operand uses section
                if (gen_op.operands.len > 0) {
                    try self.writer.writeByte('\n');
                    try self.printIndent();
                    try self.printOperandUses(gen_op.operands);
                }

                // Attributes section
                if (gen_op.attributes) |attrs| {
                    try self.writer.writeByte('\n');
                    try self.printIndent();
                    try self.printAttributesSection(attrs);
                }

                // Properties section (if present, merge with attributes for now)
                if (gen_op.properties) |props| {
                    try self.writer.writeByte('\n');
                    try self.printIndent();
                    try self.printAttributesSection(props);
                }

                // Successors section
                if (gen_op.successors.len > 0) {
                    try self.writer.writeByte('\n');
                    try self.printIndent();
                    try self.printSuccessors(gen_op.successors);
                }

                // Regions section
                if (gen_op.regions.len > 0) {
                    try self.writer.writeByte('\n');
                    try self.printIndent();
                    try self.printRegions(gen_op.regions);
                }

                // Location section
                if (operation.location) |location| {
                    try self.writer.writeByte('\n');
                    try self.printIndent();
                    try self.printLocation(location);
                }
            },
        }

        self.indent_level -= 1;
        try self.writer.writeByte(')');
    }

    // Lisp Grammar: RESULTS ::= (result-bindings [ VALUE_ID* ])
    fn printResultBindings(self: *LispPrinter, results: ast.OpResultList) !void {
        try self.writer.writeAll("(result-bindings [");
        for (results.results, 0..) |result, i| {
            if (i > 0) try self.writer.writeByte(' ');
            try self.writer.writeAll(result.value_id);
            // Note: num_results for multi-result operations handled separately if needed
        }
        try self.writer.writeAll("])");
    }

    // Lisp Grammar: RESULT_TYPES ::= (result-types TYPE*)
    fn printResultTypes(self: *LispPrinter, types: []ast.Type) !void {
        try self.writer.writeAll("(result-types");
        for (types) |typ| {
            try self.writer.writeByte(' ');
            try self.printType(typ, .result_type);
        }
        try self.writer.writeByte(')');
    }

    // Lisp Grammar: OPERANDS ::= (operand-uses VALUE_ID*)
    fn printOperandUses(self: *LispPrinter, operands: []ast.ValueUse) !void {
        try self.writer.writeAll("(operand-uses");
        for (operands) |operand| {
            try self.writer.writeByte(' ');
            try self.writer.writeAll(operand.value_id);
            if (operand.result_number) |num| {
                try self.writer.writeByte('#');
                try self.writer.print("{d}", .{num});
            }
        }
        try self.writer.writeByte(')');
    }

    // Lisp Grammar: ATTRIBUTES ::= (attributes { KEYWORD ATTR* })
    fn printAttributesSection(self: *LispPrinter, attrs: ast.DictionaryAttribute) !void {
        // Validate that dictionary will have even number of elements
        validateDictionary(attrs.entries);

        try self.writer.writeAll("(attributes {");

        for (attrs.entries, 0..) |entry, i| {
            if (i > 0) try self.writer.writeByte(' ');
            // Print key with : prefix (keyword)
            try self.writer.writeByte(':');
            try self.writer.writeAll(entry.name);

            try self.writer.writeByte(' ');
            // If there's a value, print it; otherwise it's a unit attribute - print 'true'
            if (entry.value) |_| {
                // Print value - need to parse and convert
                try self.printAttributeValue(entry);
            } else {
                // Unit attribute: print 'true'
                try self.writer.writeAll("true");
            }
        }

        try self.writer.writeAll("})");
    }

    // Parse and print function type attribute in Lisp format
    fn printFunctionTypeAttribute(self: *LispPrinter, raw_value: []const u8) !void {
        // Parse: "(i32, i32) -> i32" or "(i32) -> i32" or "() -> i32"
        // Output: (!function (inputs i32 i32) (results i32))

        try self.writer.writeAll("(!function ");

        // Find the -> separator
        const arrow_pos = std.mem.indexOf(u8, raw_value, "->") orelse {
            // No arrow found, output as-is
            try self.writer.writeAll(raw_value);
            try self.writer.writeByte(')');
            return;
        };

        // Parse inputs: everything before ->
        const inputs_part = std.mem.trim(u8, raw_value[0..arrow_pos], " \t");
        const outputs_part = std.mem.trim(u8, raw_value[arrow_pos + 2..], " \t");

        // Print inputs
        try self.writer.writeAll("(inputs");
        if (inputs_part.len > 2) { // More than just "()"
            // Remove parentheses and split by comma
            const inputs_inner = std.mem.trim(u8, inputs_part, "()");
            if (inputs_inner.len > 0) {
                var iter = std.mem.splitSequence(u8, inputs_inner, ",");
                while (iter.next()) |input| {
                    const trimmed = std.mem.trim(u8, input, " \t");
                    if (trimmed.len > 0) {
                        try self.writer.writeByte(' ');
                        try self.writer.writeAll(trimmed);
                    }
                }
            }
        }
        try self.writer.writeByte(')');

        // Print results
        try self.writer.writeAll(" (results");
        if (outputs_part.len > 0) {
            // Check if it's wrapped in parens (multiple results)
            if (outputs_part[0] == '(') {
                const outputs_inner = std.mem.trim(u8, outputs_part, "()");
                var iter = std.mem.splitSequence(u8, outputs_inner, ",");
                while (iter.next()) |output| {
                    const trimmed = std.mem.trim(u8, output, " \t");
                    if (trimmed.len > 0) {
                        try self.writer.writeByte(' ');
                        try self.writer.writeAll(trimmed);
                    }
                }
            } else {
                // Single result, no parens
                try self.writer.writeByte(' ');
                try self.writer.writeAll(outputs_part);
            }
        }
        try self.writer.writeByte(')');

        try self.writer.writeByte(')');
    }

    // Print attribute value - properly handle all attribute types
    fn printAttributeValue(self: *LispPrinter, entry: ast.AttributeEntry) !void {
        // Handle unit attributes (no value)
        if (entry.value == null) {
            return; // Already printed the name, nothing more to do
        }

        const attr_value = entry.value.?;

        switch (attr_value) {
            .builtin => |builtin| {
                switch (builtin) {
                    .array => |arr| try self.printArrayAttribute(arr),
                    .dictionary => |dict| try self.printNestedDictionaryAttribute(dict),
                    .string => |raw_value| {
                        // Legacy handling for raw strings
                        // Try to detect typed attributes: "value : type"
                        // But skip dialect attributes (starting with #) which may contain : internally
                        const trimmed = std.mem.trim(u8, raw_value, " \t");
                        const is_dialect_attr = trimmed.len > 0 and trimmed[0] == '#';

                        if (!is_dialect_attr) {
                            if (std.mem.indexOf(u8, raw_value, " : ")) |colon_pos| {
                                const value_part = std.mem.trim(u8, raw_value[0..colon_pos], " \t");
                                const type_part = std.mem.trim(u8, raw_value[colon_pos + 3..], " \t");

                                // Wrap typed attributes in (: value type) form
                                // This handles integers, floats, dense arrays, etc.
                                try self.writer.writeAll("(: ");
                                try self.writer.writeAll(value_part);
                                try self.writer.writeByte(' ');
                                try self.writer.writeAll(type_part);
                                try self.writer.writeByte(')');
                                return;
                            }
                        }

                        // Check for symbol references (heuristic: sym_name, callee attributes)
                        if (std.mem.eql(u8, entry.name, "sym_name") or
                            std.mem.eql(u8, entry.name, "callee")) {
                            // Symbol reference: @name
                            const symbol_name = std.mem.trim(u8, raw_value, "\" \t");
                            // Check if it already starts with @
                            if (symbol_name.len > 0 and symbol_name[0] == '@') {
                                try self.writer.writeAll(symbol_name);
                            } else {
                                try self.writer.writeByte('@');
                                try self.writer.writeAll(symbol_name);
                            }
                            return;
                        }

                        // Check for function_type attribute
                        if (std.mem.eql(u8, entry.name, "function_type")) {
                            // Convert function type to Lisp format: (!function (inputs ...) (results ...))
                            try self.printFunctionTypeAttribute(raw_value);
                            return;
                        }

                        // Default: output as-is
                        try self.writer.writeAll(raw_value);
                    },
                    else => {
                        // TODO: Handle integer, float, boolean if needed
                        try self.writer.writeAll("<unsupported-builtin-attr>");
                    },
                }
            },
            else => {
                // TODO: Handle alias, dialect if needed
                try self.writer.writeAll("<unsupported-attr-type>");
            },
        }
    }

    // Print array attribute: [element1, element2, ...]  ->  [element1 element2 ...]
    fn printArrayAttribute(self: *LispPrinter, elements: []ast.AttributeValue) anyerror!void {
        try self.writer.writeByte('[');

        for (elements, 0..) |element, i| {
            if (i > 0) try self.writer.writeByte(' ');
            try self.printAttributeValueDirect(element);
        }

        try self.writer.writeByte(']');
    }

    // Print nested dictionary: {key1 = val1, key2 = val2} -> {:key1 val1 :key2 val2}
    // Unit attributes (no value) become {:key true}
    fn printNestedDictionaryAttribute(self: *LispPrinter, entries: []ast.AttributeEntry) anyerror!void {
        // Validate that dictionary will have even number of elements
        validateDictionary(entries);

        try self.writer.writeByte('{');

        for (entries, 0..) |entry, i| {
            if (i > 0) try self.writer.writeByte(' ');

            // Print key with : prefix
            try self.writer.writeByte(':');
            try self.writer.writeAll(entry.name);

            // Print value if present, otherwise print 'true' for unit attributes
            try self.writer.writeByte(' ');
            if (entry.value) |_| {
                try self.printAttributeValue(entry);
            } else {
                // Unit attribute: print 'true'
                try self.writer.writeAll("true");
            }
        }

        try self.writer.writeByte('}');
    }

    // Helper to print AttributeValue directly without entry context
    fn printAttributeValueDirect(self: *LispPrinter, value: ast.AttributeValue) anyerror!void {
        switch (value) {
            .builtin => |builtin| {
                switch (builtin) {
                    .array => |arr| try self.printArrayAttribute(arr),
                    .dictionary => |dict| try self.printNestedDictionaryAttribute(dict),
                    .string => |s| try self.writer.writeAll(s),
                    else => try self.writer.writeAll("<unsupported>"),
                }
            },
            else => try self.writer.writeAll("<unsupported>"),
        }
    }

    // Lisp Grammar: SUCCESSORS ::= (successors SUCCESSOR*)
    fn printSuccessors(self: *LispPrinter, successors: []ast.Successor) !void {
        try self.writer.writeAll("(successors");
        self.indent_level += 1;

        for (successors) |successor| {
            try self.writer.writeByte('\n');
            try self.printIndent();
            try self.printSuccessor(successor);
        }

        self.indent_level -= 1;
        try self.writer.writeByte(')');
    }

    // Lisp Grammar: SUCCESSOR ::= (successor BLOCK_ID (operand-bundle)?)
    fn printSuccessor(self: *LispPrinter, successor: ast.Successor) !void {
        try self.writer.writeAll("(successor ");
        try self.writer.writeAll(successor.block_id);

        if (successor.args) |args| {
            try self.writer.writeAll(" (");
            for (args.args, 0..) |arg, i| {
                if (i > 0) try self.writer.writeByte(' ');
                try self.writer.writeAll(arg.value_id);
            }
            try self.writer.writeByte(')');
        }

        try self.writer.writeByte(')');
    }

    // Lisp Grammar: REGIONS ::= (regions REGION*)
    fn printRegions(self: *LispPrinter, regions: []ast.Region) !void {
        try self.writer.writeAll("(regions");
        self.indent_level += 1;

        for (regions) |region| {
            try self.writer.writeByte('\n');
            try self.printIndent();
            try self.printRegion(region);
        }

        self.indent_level -= 1;
        try self.writer.writeByte(')');
    }

    // Lisp Grammar: REGION ::= (region BLOCK+)
    fn printRegion(self: *LispPrinter, region: ast.Region) anyerror!void {
        try self.writer.writeAll("(region");
        self.indent_level += 1;

        // Print entry block if present (as an unlabeled block)
        if (region.entry_block) |entry_ops| {
            try self.writer.writeByte('\n');
            try self.printIndent();
            try self.writer.writeAll("(block");
            self.indent_level += 1;

            // Empty arguments for entry block
            try self.writer.writeByte('\n');
            try self.printIndent();
            try self.writer.writeAll("(arguments [])");

            // Print operations
            for (entry_ops) |op| {
                try self.writer.writeByte('\n');
                try self.printIndent();
                try self.printOperation(op);
            }

            self.indent_level -= 1;
            try self.writer.writeByte(')');
        }

        // Print labeled blocks
        for (region.blocks) |block| {
            try self.writer.writeByte('\n');
            try self.printIndent();
            try self.printBlock(block);
        }

        self.indent_level -= 1;
        try self.writer.writeByte(')');
    }

    // Lisp Grammar: BLOCK ::= (block [BLOCK_ID] (arguments [...]) OPERATION*)
    fn printBlock(self: *LispPrinter, block: ast.Block) !void {
        try self.writer.writeAll("(block [");
        try self.writer.writeAll(block.label.block_id);
        try self.writer.writeByte(']');
        self.indent_level += 1;

        // Print arguments
        try self.writer.writeByte('\n');
        try self.printIndent();
        try self.writer.writeAll("(arguments [");
        if (block.label.args) |args| {
            for (args.args, 0..) |arg, i| {
                if (i > 0) try self.writer.writeByte(' ');
                try self.writer.writeAll("[");
                try self.writer.writeAll(arg.value_id);
                try self.writer.writeByte(' ');
                try self.printType(arg.type, .argument_type);
                try self.writer.writeByte(']');
            }
        }
        try self.writer.writeAll("])");

        // Print operations
        for (block.operations) |op| {
            try self.writer.writeByte('\n');
            try self.printIndent();
            try self.printOperation(op);
        }

        self.indent_level -= 1;
        try self.writer.writeByte(')');
    }

    // Lisp Grammar: LOCATION ::= (location ATTR)
    fn printLocation(self: *LispPrinter, location: ast.Location) !void {
        try self.writer.writeAll("(location ");
        try self.writer.writeAll(location.source);
        try self.writer.writeByte(')');
    }

    // Context for type printing - determines if we need special formatting
    const TypeContext = enum {
        result_type,      // In result-types section
        argument_type,    // In block arguments
        attribute_value,  // Inside an attribute (may need !function wrapper)
    };

    // Lisp Grammar: TYPE ::= IDENT | "!" IDENT ( "." IDENT )*
    fn printType(self: *LispPrinter, typ: ast.Type, context: TypeContext) anyerror!void {
        _ = context; // Will use this for function types in attributes

        switch (typ) {
            .builtin => |builtin| try self.printBuiltinType(builtin),
            .dialect => |dialect| try self.printDialectType(dialect),
            .type_alias => |alias| {
                try self.writer.writeByte('!');
                try self.writer.writeAll(alias);
            },
            .function => |func_type| {
                // For now, print as simple function type
                // TODO: In attribute context, wrap with (!function ...)
                try self.printFunctionTypeSimple(func_type);
            },
        }
    }

    // Print builtin types as plain identifiers
    fn printBuiltinType(self: *LispPrinter, builtin: ast.BuiltinType) anyerror!void {
        switch (builtin) {
            .integer => |int| {
                switch (int.signedness) {
                    .signless => try self.writer.print("i{d}", .{int.width}),
                    .signed => try self.writer.print("si{d}", .{int.width}),
                    .unsigned => try self.writer.print("ui{d}", .{int.width}),
                }
            },
            .float => |float| {
                const name = switch (float) {
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
            .none => try self.writer.writeAll("none"),
            .tensor => |tensor| {
                try self.writer.writeAll("tensor<");
                for (tensor.dimensions, 0..) |dim, i| {
                    if (i > 0) try self.writer.writeByte('x');
                    switch (dim) {
                        .static => |s| try self.writer.print("{d}", .{s}),
                        .dynamic => try self.writer.writeByte('?'),
                    }
                }
                try self.writer.writeByte('x');
                try self.printType(tensor.element_type.*, .result_type);
                try self.writer.writeByte('>');
            },
            .memref => |memref| {
                try self.writer.writeAll("memref<");
                for (memref.dimensions, 0..) |dim, i| {
                    if (i > 0) try self.writer.writeByte('x');
                    switch (dim) {
                        .static => |s| try self.writer.print("{d}", .{s}),
                        .dynamic => try self.writer.writeByte('?'),
                    }
                }
                try self.writer.writeByte('x');
                try self.printType(memref.element_type.*, .result_type);
                try self.writer.writeByte('>');
            },
            .vector => |vector| {
                try self.writer.writeAll("vector<");
                for (vector.dimensions, 0..) |dim, i| {
                    if (i > 0) try self.writer.writeByte('x');
                    switch (dim) {
                        .fixed => |f| try self.writer.print("{d}", .{f}),
                        .scalable => |s| try self.writer.print("[{d}]", .{s}),
                    }
                }
                try self.writer.writeByte('x');
                try self.printType(vector.element_type.*, .result_type);
                try self.writer.writeByte('>');
            },
            .complex => |complex| {
                try self.writer.writeAll("complex<");
                try self.printType(complex.*, .result_type);
                try self.writer.writeByte('>');
            },
            .tuple => |tuple| {
                try self.writer.writeAll("tuple<");
                for (tuple, 0..) |t, i| {
                    if (i > 0) try self.writer.writeAll(", ");
                    try self.printType(t, .result_type);
                }
                try self.writer.writeByte('>');
            },
        }
    }

    // Print dialect types with ! prefix
    fn printDialectType(self: *LispPrinter, dialect: ast.DialectType) !void {
        try self.writer.writeByte('!');
        try self.writer.writeAll(dialect.namespace);
        if (dialect.body) |body| {
            try self.writer.writeAll(body);
        }
    }

    // Simple function type printing (not wrapped)
    fn printFunctionTypeSimple(self: *LispPrinter, func_type: ast.FunctionType) !void {
        try self.writer.writeByte('(');
        for (func_type.inputs, 0..) |input, i| {
            if (i > 0) try self.writer.writeAll(", ");
            try self.printType(input, .result_type);
        }
        try self.writer.writeAll(") -> ");

        if (func_type.outputs.len == 1) {
            try self.printType(func_type.outputs[0], .result_type);
        } else {
            try self.writer.writeByte('(');
            for (func_type.outputs, 0..) |output, i| {
                if (i > 0) try self.writer.writeAll(", ");
                try self.printType(output, .result_type);
            }
            try self.writer.writeByte(')');
        }
    }
};

test "lisp printer - simple constant" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const allocator = arena.allocator();

    const source = "%0 = \"arith.constant\"() <{value = 42 : i32}> : () -> i32";
    var lex = @import("lexer.zig").Lexer.init(source);
    var parser = try @import("parser.zig").Parser.init(allocator, &lex);
    defer parser.deinit();

    var module = try parser.parseModule();
    defer module.deinit();

    var output: std.ArrayList(u8) = .empty;
    var printer = LispPrinter.init(output.writer(allocator).any());
    try printer.printModule(module);

    const result = output.items;

    // Should contain basic structure
    try std.testing.expect(std.mem.indexOf(u8, result, "(mlir") != null);
    try std.testing.expect(std.mem.indexOf(u8, result, "(operation") != null);
    try std.testing.expect(std.mem.indexOf(u8, result, "(name arith.constant)") != null);
    try std.testing.expect(std.mem.indexOf(u8, result, "(result-bindings [%0])") != null);
}
