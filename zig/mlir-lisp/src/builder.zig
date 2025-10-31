/// MLIR IR Builder
/// Converts parsed AST (parser.MlirModule) into actual MLIR IR
const std = @import("std");
const parser = @import("parser.zig");
const reader = @import("reader.zig");
const mlir = @import("mlir/c.zig");

pub const BuildError = error{
    UnsupportedOperation,
    InvalidType,
    InvalidAttribute,
    UnknownValue,
    InvalidStructure,
    ModuleCreationFailed,
    InvalidModuleOperation,
} || std.mem.Allocator.Error;

/// Builder context for constructing MLIR IR
pub const Builder = struct {
    allocator: std.mem.Allocator,
    ctx: *mlir.Context,
    location: mlir.MlirLocation,

    /// Maps SSA value IDs (%c0, %sum, etc.) to MLIR values
    value_map: std.StringHashMap(mlir.MlirValue),

    /// Maps type alias names (!my_vec, etc.) to their type definitions
    type_alias_map: std.StringHashMap([]const u8),

    pub fn init(allocator: std.mem.Allocator, ctx: *mlir.Context) Builder {
        return Builder{
            .allocator = allocator,
            .ctx = ctx,
            .location = mlir.Location.unknown(ctx),
            .value_map = std.StringHashMap(mlir.MlirValue).init(allocator),
            .type_alias_map = std.StringHashMap([]const u8).init(allocator),
        };
    }

    pub fn deinit(self: *Builder) void {
        self.value_map.deinit();
        self.type_alias_map.deinit();
    }

    /// Build a complete MLIR module from our parsed AST
    pub fn buildModule(self: *Builder, parsed_module: *parser.MlirModule) BuildError!mlir.Module {
        // Register all type aliases first
        try self.registerTypeAliases(parsed_module.type_aliases);

        // Special case: if we have exactly one operation and it's a builtin.module,
        // use that directly instead of creating a wrapper module
        if (parsed_module.operations.len == 1) {
            const operation = &parsed_module.operations[0];
            if (std.mem.eql(u8, operation.name, "builtin.module")) {
                const mlir_op = try self.buildOperation(operation);
                return try mlir.Module.fromOperation(mlir_op);
            }
        }

        // Otherwise, create a new module and add all operations to it
        var mod = try mlir.Module.create(self.location);

        // Get the module's body block
        const body = mod.getBody();

        // Build each top-level operation
        for (parsed_module.operations) |*operation| {
            const mlir_op = try self.buildOperation(operation);
            mlir.Block.appendOperation(body, mlir_op);
        }

        return mod;
    }

    /// Register type aliases in the builder context
    fn registerTypeAliases(self: *Builder, type_aliases: []const parser.TypeAlias) BuildError!void {
        for (type_aliases) |alias| {
            try self.type_alias_map.put(alias.name, alias.definition);
        }
    }

    /// Build a single operation
    fn buildOperation(self: *Builder, operation: *const parser.Operation) BuildError!mlir.MlirOperation {
        // Parse result types
        var result_types = std.ArrayList(mlir.MlirType){};
        defer result_types.deinit(self.allocator);

        for (operation.result_types) |*type_expr| {
            const ty = try self.buildType(type_expr);
            try result_types.append(self.allocator, ty);
        }

        // Parse operands (look up SSA values)
        var operands = std.ArrayList(mlir.MlirValue){};
        defer operands.deinit(self.allocator);

        for (operation.operands) |operand_id| {
            const value = self.value_map.get(operand_id) orelse return error.UnknownValue;
            try operands.append(self.allocator, value);
        }

        // Parse attributes
        var attributes = std.ArrayList(mlir.c.MlirNamedAttribute){};
        defer attributes.deinit(self.allocator);

        for (operation.attributes) |*attr| {
            const named_attr = try self.buildNamedAttribute(attr);
            try attributes.append(self.allocator, named_attr);
        }

        // Parse regions
        var regions = std.ArrayList(mlir.MlirRegion){};
        defer regions.deinit(self.allocator);

        for (operation.regions) |*region| {
            const mlir_region = try self.buildRegion(region);
            try regions.append(self.allocator, mlir_region);
        }

        // Parse successors
        var successors = std.ArrayList(mlir.MlirBlock){};
        defer successors.deinit(self.allocator);
        // TODO: Implement successor lookups when we need control flow

        // Create the operation
        const mlir_op = mlir.Operation.create(
            operation.name,
            self.location,
            result_types.items,
            operands.items,
            attributes.items,
            successors.items,
            regions.items,
        );

        // Register result bindings
        for (operation.result_bindings, 0..) |binding, i| {
            const result = mlir.Operation.getResult(mlir_op, i);
            try self.value_map.put(binding, result);
        }

        return mlir_op;
    }

    /// Serialize a Value to MLIR syntax string
    fn serializeValueToMLIR(self: *Builder, value: *const reader.Value) BuildError![]const u8 {
        var buffer = std.ArrayList(u8){};
        errdefer buffer.deinit(self.allocator);

        try self.serializeValueToMLIRImpl(value, &buffer);
        return buffer.toOwnedSlice(self.allocator);
    }

    /// Helper to recursively serialize a Value to MLIR syntax
    fn serializeValueToMLIRImpl(self: *Builder, value: *const reader.Value, buffer: *std.ArrayList(u8)) BuildError!void {
        const writer = buffer.writer(self.allocator);

        switch (value.type) {
            .identifier => try writer.writeAll(value.data.atom),
            .value_id => try writer.writeAll(value.data.atom),
            .block_id => try writer.writeAll(value.data.atom),
            .symbol => try writer.writeAll(value.data.atom),
            .keyword => try writer.print(":{s}", .{value.keywordToName()}),
            .string => try writer.writeAll(value.data.atom), // String lexeme already includes quotes
            .number => try writer.writeAll(value.data.atom),
            .true_lit => try writer.writeAll("true"),
            .false_lit => try writer.writeAll("false"),
            .type => {
                // type stores the full type string including '!'
                try writer.writeAll(value.data.type);
            },
            .function_type => {
                // Serialize function type as (!function (inputs ...) (results ...))
                try writer.writeAll("(!function (inputs");
                const inputs = value.data.function_type.inputs;
                for (0..inputs.len()) |i| {
                    try writer.writeAll(" ");
                    try self.serializeValueToMLIRImpl(inputs.at(i), buffer);
                }
                try writer.writeAll(") (results");
                const results = value.data.function_type.results;
                for (0..results.len()) |i| {
                    try writer.writeAll(" ");
                    try self.serializeValueToMLIRImpl(results.at(i), buffer);
                }
                try writer.writeAll("))");
            },
            .attr_expr => {
                try writer.writeAll("#");
                try self.serializeValueToMLIRImpl(value.data.attr_expr, buffer);
            },
            .has_type => {
                // Serialize as "value : type" for MLIR
                try self.serializeValueToMLIRImpl(value.data.has_type.value, buffer);
                try writer.writeAll(" : ");
                try self.serializeValueToMLIRImpl(value.data.has_type.type_expr, buffer);
            },
            .list => {
                try writer.writeAll("(");
                const list = value.data.list;
                var i: usize = 0;
                while (i < list.len()) : (i += 1) {
                    if (i > 0) try writer.writeAll(" ");
                    try self.serializeValueToMLIRImpl(list.at(i), buffer);
                }
                try writer.writeAll(")");
            },
            .vector => {
                try writer.writeAll("[");
                const vec = value.data.vector;
                var i: usize = 0;
                while (i < vec.len()) : (i += 1) {
                    if (i > 0) try writer.writeAll(" ");
                    try self.serializeValueToMLIRImpl(vec.at(i), buffer);
                }
                try writer.writeAll("]");
            },
            .map => {
                try writer.writeAll("{");
                const map = value.data.map;
                var i: usize = 0;
                while (i + 1 < map.len()) : (i += 2) {
                    if (i > 0) try writer.writeAll(" ");
                    try self.serializeValueToMLIRImpl(map.at(i), buffer);
                    try writer.writeAll(" ");
                    try self.serializeValueToMLIRImpl(map.at(i + 1), buffer);
                }
                try writer.writeAll("}");
            },
        }
    }

    /// Build a type from a type expression
    fn buildType(self: *Builder, type_expr: *const parser.TypeExpr) BuildError!mlir.MlirType {
        const value = type_expr.value;

        // Handle different type forms
        switch (value.type) {
            .type => {
                // Simple type - check if it's an alias, otherwise parse directly
                const type_str = value.data.type;

                // Check if this is a type alias
                if (self.type_alias_map.get(type_str)) |definition| {
                    return mlir.Type.parse(self.ctx, definition) catch error.InvalidType;
                }

                return mlir.Type.parse(self.ctx, type_str) catch error.InvalidType;
            },
            .identifier => {
                // Plain identifier (builtin type like i32, f64, etc.)
                const type_str = value.data.atom;
                return mlir.Type.parse(self.ctx, type_str) catch error.InvalidType;
            },
            .function_type => {
                // Function type - build from inputs/results
                return self.buildFunctionTypeFromValue(value);
            },
            else => return error.InvalidType,
        }
    }

    /// Build an MLIR function type from a function_type Value
    fn buildFunctionTypeFromValue(self: *Builder, value: *const reader.Value) BuildError!mlir.MlirType {
        if (value.type != .function_type) return error.InvalidType;

        const func_type = value.data.function_type;
        const inputs = func_type.inputs;
        const results = func_type.results;

        // Build input types
        const num_inputs = inputs.len();
        var input_types = try self.allocator.alloc(mlir.MlirType, num_inputs);
        defer self.allocator.free(input_types);

        for (0..num_inputs) |i| {
            const input_value = inputs.at(i);
            input_types[i] = try self.buildTypeFromValue(input_value);
        }

        // Build result types
        const num_results = results.len();
        var result_types = try self.allocator.alloc(mlir.MlirType, num_results);
        defer self.allocator.free(result_types);

        for (0..num_results) |i| {
            const result_value = results.at(i);
            result_types[i] = try self.buildTypeFromValue(result_value);
        }

        // Create function type using MLIR C API
        return mlir.c.mlirFunctionTypeGet(
            self.ctx.ctx,
            @intCast(num_inputs),
            input_types.ptr,
            @intCast(num_results),
            result_types.ptr,
        );
    }

    /// Build an MLIR type from a reader Value (used recursively)
    pub fn buildTypeFromValue(self: *Builder, value: *const reader.Value) BuildError!mlir.MlirType {
        switch (value.type) {
            .type => {
                // Simple type - check if it's an alias, otherwise parse directly
                const type_str = value.data.type;

                // Check if this is a type alias
                if (self.type_alias_map.get(type_str)) |definition| {
                    return mlir.Type.parse(self.ctx, definition) catch error.InvalidType;
                }

                return mlir.Type.parse(self.ctx, type_str) catch error.InvalidType;
            },
            .identifier => {
                // Plain identifier (builtin type like i32, f64, etc.)
                const type_str = value.data.atom;
                return mlir.Type.parse(self.ctx, type_str) catch error.InvalidType;
            },
            .function_type => {
                // Function type - build from inputs/results
                return self.buildFunctionTypeFromValue(value);
            },
            .string => {
                // String literal containing complex dialect type (e.g., "!llvm.func<...>")
                // This is allowed for complex types that can't be represented in S-expr
                const type_str = value.data.atom;
                // Strip quotes
                if (type_str.len < 2 or type_str[0] != '"' or type_str[type_str.len - 1] != '"') {
                    return error.InvalidType;
                }
                const type_name = type_str[1 .. type_str.len - 1];
                return mlir.Type.parse(self.ctx, type_name) catch error.InvalidType;
            },
            else => return error.InvalidType,
        }
    }

    /// Build a named attribute
    fn buildNamedAttribute(self: *Builder, attr: *const parser.Attribute) BuildError!mlir.c.MlirNamedAttribute {
        // Special handling for type-related attributes - they need to be wrapped in TypeAttr
        if (std.mem.eql(u8, attr.key, "type") or std.mem.eql(u8, attr.key, "function_type")) {
            const attr_value = try self.buildTypeAttribute(&attr.value);
            return mlir.namedAttribute(self.ctx, attr.key, attr_value);
        }

        // Special handling for symbol attributes - might be StringAttr or FlatSymbolRefAttr
        if (std.mem.eql(u8, attr.key, "sym") or std.mem.eql(u8, attr.key, "sym_name") or std.mem.eql(u8, attr.key, "callee")) {
            const attr_value = try self.buildSymbolAttribute(&attr.value, attr.key);
            return mlir.namedAttribute(self.ctx, attr.key, attr_value);
        }

        const attr_value = try self.buildAttributeValue(&attr.value);
        return mlir.namedAttribute(self.ctx, attr.key, attr_value);
    }

    /// Build a TypeAttr from an attribute expression containing a type
    pub fn buildTypeAttribute(self: *Builder, attr_expr: *const parser.AttrExpr) BuildError!mlir.MlirAttribute {
        const value = attr_expr.value;

        // Handle different formats:
        // 1. Direct type or function_type Value
        // 2. List starting with a type (!function ...) - need to unwrap
        const type_value = if (value.type == .list) blk: {
            // Check if this is a function type list: (!function ...)
            const list = value.data.list;
            if (list.len() > 0) {
                const first = list.at(0);
                // First element should be a type marker for "function"
                if (first.type == .type and std.mem.eql(u8, first.data.type, "!function")) {
                    // This is a function type - convert the list to a function_type
                    // Parse it manually here
                    if (list.len() < 3) return error.InvalidType;

                    const inputs_list_value = list.at(1);
                    const results_list_value = list.at(2);

                    if (inputs_list_value.type != .list or results_list_value.type != .list) {
                        return error.InvalidType;
                    }

                    const inputs_list = inputs_list_value.data.list;
                    const results_list = results_list_value.data.list;

                    // Skip "inputs" and "results" keywords and build types
                    const num_inputs = if (inputs_list.len() > 0) inputs_list.len() - 1 else 0;
                    const num_results = if (results_list.len() > 0) results_list.len() - 1 else 0;

                    var input_types = try self.allocator.alloc(mlir.MlirType, num_inputs);
                    defer self.allocator.free(input_types);

                    for (0..num_inputs) |i| {
                        const input_value = inputs_list.at(i + 1);
                        input_types[i] = try self.buildTypeFromValue(input_value);
                    }

                    var result_types = try self.allocator.alloc(mlir.MlirType, num_results);
                    defer self.allocator.free(result_types);

                    for (0..num_results) |i| {
                        const result_value = results_list.at(i + 1);
                        result_types[i] = try self.buildTypeFromValue(result_value);
                    }

                    // Create function type directly
                    const mlir_type = mlir.c.mlirFunctionTypeGet(
                        self.ctx.ctx,
                        @intCast(num_inputs),
                        input_types.ptr,
                        @intCast(num_results),
                        result_types.ptr,
                    );

                    return mlir.c.mlirTypeAttrGet(mlir_type);
                }
            }
            break :blk value;
        } else value;

        const mlir_type = try self.buildTypeFromValue(type_value);

        // Wrap it in a TypeAttr
        return mlir.c.mlirTypeAttrGet(mlir_type);
    }

    /// Build a symbol attribute - either FlatSymbolRefAttr or StringAttr depending on attribute name
    pub fn buildSymbolAttribute(self: *Builder, attr_expr: *const parser.AttrExpr, attr_name: []const u8) BuildError!mlir.MlirAttribute {
        const value = attr_expr.value;

        // The value should be a symbol (e.g., @add)
        if (value.type != .symbol) return error.InvalidAttribute;

        // The symbol atom includes the '@', so strip it
        const symbol_str = value.data.atom;
        if (symbol_str.len == 0 or symbol_str[0] != '@') return error.InvalidAttribute;

        const symbol_name = symbol_str[1..]; // Skip the '@'

        // sym_name needs StringAttr, sym needs FlatSymbolRefAttr
        if (std.mem.eql(u8, attr_name, "sym_name")) {
            // Create StringAttr
            const name_ref = mlir.c.mlirStringRefCreate(symbol_name.ptr, symbol_name.len);
            return mlir.c.mlirStringAttrGet(self.ctx.ctx, name_ref);
        } else {
            // Create FlatSymbolRefAttr
            const symbol_ref = mlir.c.mlirStringRefCreate(symbol_name.ptr, symbol_name.len);
            return mlir.c.mlirFlatSymbolRefAttrGet(self.ctx.ctx, symbol_ref);
        }
    }

    /// Build an attribute value
    fn buildAttributeValue(self: *Builder, attr_expr: *const parser.AttrExpr) BuildError!mlir.MlirAttribute {
        const value = attr_expr.value;

        // Handle typed literals: (: value type)
        if (value.type == .has_type) {
            const val = value.data.has_type.value;
            const type_val = value.data.has_type.type_expr;

            // The type should be a type or function_type wrapper
            if (type_val.type != .type and type_val.type != .function_type) {
                return error.InvalidAttribute;
            }

            // Build the type from the type Value
            const mlir_type = try self.buildTypeFromValue(type_val);

            // Extract the integer value
            if (val.type != .number) {
                return error.InvalidAttribute;
            }

            const int_val = std.fmt.parseInt(i64, val.data.atom, 10) catch return error.InvalidAttribute;

            // Create a typed integer attribute
            return mlir.c.mlirIntegerAttrGet(mlir_type, int_val);
        }

        // Serialize the attribute expression to MLIR syntax
        const attr_str = try self.serializeValueToMLIR(value);
        defer self.allocator.free(attr_str);

        // Use MLIR's built-in attribute parser
        return mlir.Attribute.parse(self.ctx, attr_str) catch error.InvalidAttribute;
    }

    /// Build a region
    fn buildRegion(self: *Builder, region: *const parser.Region) BuildError!mlir.MlirRegion {
        const mlir_region = mlir.Region.create();

        for (region.blocks) |*block| {
            const mlir_block = try self.buildBlock(block);
            mlir.Region.appendBlock(mlir_region, mlir_block);
        }

        return mlir_region;
    }

    /// Build a block
    fn buildBlock(self: *Builder, block: *const parser.Block) BuildError!mlir.MlirBlock {
        // Parse block arguments
        var arg_types = std.ArrayList(mlir.MlirType){};
        defer arg_types.deinit(self.allocator);

        var arg_locs = std.ArrayList(mlir.MlirLocation){};
        defer arg_locs.deinit(self.allocator);

        for (block.arguments) |*arg| {
            const ty = try self.buildType(&arg.type);
            try arg_types.append(self.allocator, ty);
            try arg_locs.append(self.allocator, self.location);
        }

        // Create the block
        const mlir_block = mlir.Block.create(arg_types.items, arg_locs.items);

        // Register block arguments in value map
        for (block.arguments, 0..) |*arg, i| {
            const value = mlir.Block.getArgument(mlir_block, i);
            try self.value_map.put(arg.value_id, value);
        }

        // Build operations in the block
        for (block.operations) |*operation| {
            const mlir_op = try self.buildOperation(operation);
            mlir.Block.appendOperation(mlir_block, mlir_op);
        }

        return mlir_block;
    }
};
