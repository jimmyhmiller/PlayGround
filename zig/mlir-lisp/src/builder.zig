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
} || std.mem.Allocator.Error;

/// Builder context for constructing MLIR IR
pub const Builder = struct {
    allocator: std.mem.Allocator,
    ctx: *mlir.Context,
    location: mlir.MlirLocation,

    /// Maps SSA value IDs (%c0, %sum, etc.) to MLIR values
    value_map: std.StringHashMap(mlir.MlirValue),

    pub fn init(allocator: std.mem.Allocator, ctx: *mlir.Context) Builder {
        return Builder{
            .allocator = allocator,
            .ctx = ctx,
            .location = mlir.Location.unknown(ctx),
            .value_map = std.StringHashMap(mlir.MlirValue).init(allocator),
        };
    }

    pub fn deinit(self: *Builder) void {
        self.value_map.deinit();
    }

    /// Build a complete MLIR module from our parsed AST
    pub fn buildModule(self: *Builder, parsed_module: *parser.MlirModule) BuildError!mlir.Module {
        std.debug.print("DEBUG buildModule: building {} top-level operations\n", .{parsed_module.operations.len});
        for (parsed_module.operations, 0..) |*op, i| {
            std.debug.print("DEBUG buildModule:   op[{}] name={s}, ptr={*}\n", .{i, op.name, op});
        }

        var mod = try mlir.Module.create(self.location);

        // Get the module's body block
        const body = mod.getBody();

        // Build each top-level operation
        for (parsed_module.operations, 0..) |*operation, idx| {
            std.debug.print("DEBUG buildModule: about to build top-level op[{}]\n", .{idx});
            const mlir_op = try self.buildOperation(operation);
            mlir.Block.appendOperation(body, mlir_op);
        }

        return mod;
    }

    /// Build a single operation
    fn buildOperation(self: *Builder, operation: *const parser.Operation) BuildError!mlir.MlirOperation {
        std.debug.print("DEBUG buildOperation: name={s}, {} attributes\n", .{operation.name, operation.attributes.len});
        for (operation.attributes, 0..) |attr, i| {
            std.debug.print("DEBUG buildOperation:   attr[{}].key=\"{s}\" (len={})\n", .{i, attr.key, attr.key.len});
        }
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

        std.debug.print("DEBUG buildOperation: iterating over {} regions\n", .{operation.regions.len});
        for (operation.regions, 0..) |*region, region_idx| {
            std.debug.print("DEBUG buildOperation: building region[{}], blocks={}\n", .{region_idx, region.blocks.len});
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
            .type_expr => {
                try writer.writeAll("!");
                try self.serializeValueToMLIRImpl(value.data.type_expr, buffer);
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

        // Type is wrapped with type_expr marker
        if (value.type != .type_expr) return error.InvalidType;

        // Build the type from the value recursively
        return self.buildTypeFromValue(value.data.type_expr);
    }

    /// Build an MLIR type from a reader Value (recursive)
    /// Handles both simple types (!i32) and function types (!function (inputs ...) (results ...))
    pub fn buildTypeFromValue(self: *Builder, value: *const reader.Value) BuildError!mlir.MlirType {
        switch (value.type) {
            .identifier => {
                // Simple type like "i32", "f64", etc.
                // The identifier is the type name
                const type_name = value.data.atom;
                return mlir.Type.parse(self.ctx, type_name) catch error.InvalidType;
            },
            .list => {
                // This is a composite type like (!function (inputs ...) (results ...))
                // We need to recursively handle it based on the first element
                const list = value.data.list;
                if (list.len() == 0) return error.InvalidType;

                // First element tells us what kind of type this is
                const first = list.at(0);

                // For function types: (!function (inputs ...) (results ...))
                if (first.type == .type_expr) {
                    const type_name = first.data.type_expr;
                    if (type_name.type == .identifier and std.mem.eql(u8, type_name.data.atom, "function")) {
                        // This is a function type
                        if (list.len() < 3) return error.InvalidType;

                        // Second element should be (inputs ...)
                        const inputs_expr = list.at(1);
                        if (inputs_expr.type != .list) return error.InvalidType;
                        const inputs_list = inputs_expr.data.list;
                        if (inputs_list.len() < 1) return error.InvalidType;

                        const inputs_kw = inputs_list.at(0);
                        if (inputs_kw.type != .identifier or !std.mem.eql(u8, inputs_kw.data.atom, "inputs")) {
                            return error.InvalidType;
                        }

                        // Third element should be (results ...)
                        const results_expr = list.at(2);
                        if (results_expr.type != .list) return error.InvalidType;
                        const results_list = results_expr.data.list;
                        if (results_list.len() < 1) return error.InvalidType;

                        const results_kw = results_list.at(0);
                        if (results_kw.type != .identifier or !std.mem.eql(u8, results_kw.data.atom, "results")) {
                            return error.InvalidType;
                        }

                        // Build input types recursively
                        const num_inputs = inputs_list.len() - 1; // Skip the "inputs" keyword
                        var input_types = try self.allocator.alloc(mlir.MlirType, num_inputs);
                        defer self.allocator.free(input_types);

                        for (0..num_inputs) |i| {
                            const input_value = inputs_list.at(i + 1);
                            if (input_value.type != .type_expr) return error.InvalidType;
                            // Recursively build each input type
                            input_types[i] = try self.buildTypeFromValue(input_value.data.type_expr);
                        }

                        // Build result types recursively
                        const num_results = results_list.len() - 1; // Skip the "results" keyword
                        var result_types = try self.allocator.alloc(mlir.MlirType, num_results);
                        defer self.allocator.free(result_types);

                        for (0..num_results) |i| {
                            const result_value = results_list.at(i + 1);
                            if (result_value.type != .type_expr) return error.InvalidType;
                            // Recursively build each result type
                            result_types[i] = try self.buildTypeFromValue(result_value.data.type_expr);
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
                }

                return error.InvalidType;
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
            std.debug.print("DEBUG: Building symbol attribute for key: {s}\n", .{attr.key});
            const attr_value = try self.buildSymbolAttribute(&attr.value, attr.key);
            return mlir.namedAttribute(self.ctx, attr.key, attr_value);
        }

        std.debug.print("DEBUG: Building generic attribute for key: {s}\n", .{attr.key});
        const attr_value = self.buildAttributeValue(&attr.value) catch |err| {
            std.debug.print("ERROR: buildAttributeValue failed for key=\"{s}\", error={}\n", .{attr.key, err});
            return err;
        };
        return mlir.namedAttribute(self.ctx, attr.key, attr_value);
    }

    /// Build a TypeAttr from an attribute expression containing a type
    pub fn buildTypeAttribute(self: *Builder, attr_expr: *const parser.AttrExpr) BuildError!mlir.MlirAttribute {
        const value = attr_expr.value;

        // The attribute value is already unwrapped from the !type marker
        // So we can build directly from the value (which could be identifier or list for function type)
        const mlir_type = try self.buildTypeFromValue(value);

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

            // The type should be a type_expr wrapper - unwrap it
            if (type_val.type != .type_expr) {
                return error.InvalidAttribute;
            }

            // Build the type from the wrapped type expression
            const mlir_type = try self.buildTypeFromValue(type_val.data.type_expr);

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

        std.debug.print("DEBUG buildRegion: iterating over {} blocks\n", .{region.blocks.len});
        for (region.blocks, 0..) |*block, block_idx| {
            std.debug.print("DEBUG buildRegion: building block[{}], operations={}\n", .{block_idx, block.operations.len});
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
        std.debug.print("DEBUG: Building {} operations in block\n", .{block.operations.len});
        for (block.operations, 0..) |*operation, idx| {
            std.debug.print("DEBUG: About to build operation[{}] at address {*}, name.ptr={*}, name.len={}\n", .{idx, operation, operation.name.ptr, operation.name.len});
            const mlir_op = try self.buildOperation(operation);
            mlir.Block.appendOperation(mlir_block, mlir_op);
        }

        return mlir_block;
    }
};
