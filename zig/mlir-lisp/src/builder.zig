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
    UnknownBlock,
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

    /// Maps block IDs (^bb1, ^bb2, etc.) to MLIR blocks (scoped to current region)
    block_map: std.StringHashMap(mlir.MlirBlock),

    /// Maps type alias names (!my_vec, etc.) to their type definitions
    type_alias_map: std.StringHashMap([]const u8),

    /// Maps attribute alias names (#alias_scope, etc.) to their attribute definitions
    attribute_alias_map: std.StringHashMap([]const u8),

    pub fn init(allocator: std.mem.Allocator, ctx: *mlir.Context) Builder {
        return Builder{
            .allocator = allocator,
            .ctx = ctx,
            .location = mlir.Location.unknown(ctx),
            .value_map = std.StringHashMap(mlir.MlirValue).init(allocator),
            .block_map = std.StringHashMap(mlir.MlirBlock).init(allocator),
            .type_alias_map = std.StringHashMap([]const u8).init(allocator),
            .attribute_alias_map = std.StringHashMap([]const u8).init(allocator),
        };
    }

    pub fn deinit(self: *Builder) void {
        self.value_map.deinit();
        self.block_map.deinit();
        self.type_alias_map.deinit();
        self.attribute_alias_map.deinit();
    }

    /// Build a complete MLIR module from our parsed AST
    pub fn buildModule(self: *Builder, parsed_module: *parser.MlirModule) BuildError!mlir.Module {
        // Register all type aliases and attribute aliases first
        try self.registerTypeAliases(parsed_module.type_aliases);
        try self.registerAttributeAliases(parsed_module.attribute_aliases);

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

    /// Build a filtered MLIR module from our parsed AST
    /// Only includes operations where predicate(operation.name) returns true
    pub fn buildModuleFiltered(
        self: *Builder,
        parsed_module: *parser.MlirModule,
        predicate: *const fn ([]const u8) bool,
    ) BuildError!mlir.Module {
        // Register all type aliases and attribute aliases first
        try self.registerTypeAliases(parsed_module.type_aliases);
        try self.registerAttributeAliases(parsed_module.attribute_aliases);

        // Create a new module
        var mod = try mlir.Module.create(self.location);

        // Get the module's body block
        const body = mod.getBody();

        // Build only operations that match the predicate
        for (parsed_module.operations) |*operation| {
            if (predicate(operation.name)) {
                const mlir_op = try self.buildOperation(operation);
                mlir.Block.appendOperation(body, mlir_op);
            }
        }

        return mod;
    }

    /// Build a single operation into a module
    /// Useful for building metadata modules that are single operations
    pub fn buildSingleOperation(
        self: *Builder,
        operation: *const parser.Operation,
    ) BuildError!mlir.Module {
        const mlir_op = try self.buildOperation(operation);
        return try mlir.Module.fromOperation(mlir_op);
    }

    /// Build a module from a list of operations (not full parsed module)
    /// Used for building application code after filtering out metadata
    pub fn buildFromOperations(
        self: *Builder,
        operations: []const *const parser.Operation,
    ) BuildError!mlir.Module {
        // Special case: if we have exactly one operation and it's a builtin.module,
        // use that directly instead of creating a wrapper module
        if (operations.len == 1) {
            const operation = operations[0];
            if (std.mem.eql(u8, operation.name, "builtin.module")) {
                const mlir_op = try self.buildOperation(operation);
                return try mlir.Module.fromOperation(mlir_op);
            }
        }

        // Otherwise, create a new module and add all operations to it
        var mod = try mlir.Module.create(self.location);

        // Get the module's body block
        const body = mod.getBody();

        // Build each operation
        for (operations) |operation| {
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

    /// Register attribute aliases in the builder context
    fn registerAttributeAliases(self: *Builder, attribute_aliases: []const parser.AttributeAlias) BuildError!void {
        for (attribute_aliases) |alias| {
            try self.attribute_alias_map.put(alias.name, alias.definition);
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

        for (operation.successors) |successor| {
            const block = self.block_map.get(successor.block_id) orelse return error.UnknownBlock;
            try successors.append(self.allocator, block);
        }

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

        try self.serializeValueToMLIRImpl(value, &buffer, false);
        return buffer.toOwnedSlice(self.allocator);
    }

    /// Helper to recursively serialize a Value to MLIR syntax
    /// is_dict_key indicates if we're currently serializing a dictionary key (to handle keyword formatting)
    fn serializeValueToMLIRImpl(self: *Builder, value: *const reader.Value, buffer: *std.ArrayList(u8), is_dict_key: bool) BuildError!void {
        const writer = buffer.writer(self.allocator);

        switch (value.type) {
            .identifier => try writer.writeAll(value.data.atom),
            .value_id => try writer.writeAll(value.data.atom),
            .block_id => try writer.writeAll(value.data.atom),
            .symbol => try writer.writeAll(value.data.atom),
            .keyword => {
                // In dictionary keys, keywords don't have the ':' prefix
                if (is_dict_key) {
                    try writer.writeAll(value.keywordToName());
                } else {
                    try writer.print(":{s}", .{value.keywordToName()});
                }
            },
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
                    try self.serializeValueToMLIRImpl(inputs.at(i), buffer, false);
                }
                try writer.writeAll(") (results");
                const results = value.data.function_type.results;
                for (0..results.len()) |i| {
                    try writer.writeAll(" ");
                    try self.serializeValueToMLIRImpl(results.at(i), buffer, false);
                }
                try writer.writeAll("))");
            },
            .attr_expr => {
                try writer.writeAll("#");
                try self.serializeValueToMLIRImpl(value.data.attr_expr, buffer, false);
            },
            .has_type => {
                // Serialize as "value : type" for MLIR
                try self.serializeValueToMLIRImpl(value.data.has_type.value, buffer, is_dict_key);
                try writer.writeAll(" : ");
                try self.serializeValueToMLIRImpl(value.data.has_type.type_expr, buffer, false);
            },
            .list => {
                try writer.writeAll("(");
                const list = value.data.list;
                var i: usize = 0;
                while (i < list.len()) : (i += 1) {
                    if (i > 0) try writer.writeAll(" ");
                    try self.serializeValueToMLIRImpl(list.at(i), buffer, false);
                }
                try writer.writeAll(")");
            },
            .vector => {
                // MLIR arrays always use comma separators
                const vec = value.data.vector;

                try writer.writeAll("[");
                var i: usize = 0;
                while (i < vec.len()) : (i += 1) {
                    if (i > 0) try writer.writeAll(", ");
                    try self.serializeValueToMLIRImpl(vec.at(i), buffer, false);
                }
                try writer.writeAll("]");
            },
            .map => {
                // Maps are MLIR dictionary attributes: {key1 = value1, key2 = value2}
                try writer.writeAll("{");
                const map = value.data.map;
                var i: usize = 0;
                while (i + 1 < map.len()) : (i += 2) {
                    if (i > 0) try writer.writeAll(", ");

                    // Serialize key (with is_dict_key = true to strip ':' from keywords)
                    try self.serializeValueToMLIRImpl(map.at(i), buffer, true);

                    // Check if value is just 'true' - if so, it's a unit attribute (no value needed)
                    const val = map.at(i + 1);
                    if (val.type != .true_lit) {
                        try writer.writeAll(" = ");
                        try self.serializeValueToMLIRImpl(val, buffer, false);
                    }
                    // If it's true_lit, we just write the key name (unit attribute)
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
        // Special case: if results is a single empty list (), treat as void (0 results)
        const num_results = if (results.len() == 1 and results.at(0).type == .list and results.at(0).data.list.len() == 0)
            0
        else
            results.len();

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
            .list => {
                // Empty list () represents void/unit type (empty tuple)
                const list = value.data.list;
                if (list.len() == 0) {
                    // Create empty tuple type for void
                    return mlir.Type.parse(self.ctx, "()") catch error.InvalidType;
                }
                return error.InvalidType;
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

        // Special handling for unit attributes - when value is just 'true'
        // These are attributes like 'constant', 'dso_local', 'no_unwind' that are just flags
        // BUT NOT for 'value' attribute which is used for constants like llvm.mlir.constant with value=true
        if (attr.value.value.type == .true_lit and !std.mem.eql(u8, attr.key, "value")) {
            const attr_value = mlir.c.mlirUnitAttrGet(self.ctx.ctx);
            return mlir.namedAttribute(self.ctx, attr.key, attr_value);
        }

        const attr_value = try self.buildAttributeValue(&attr.value, attr.key);
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
                    // Two possible formats:
                    // 1. (!function !llvm.func<...>) - 2 elements, already in MLIR format
                    // 2. (!function (inputs ...) (results ...)) - 3+ elements, explicit format

                    if (list.len() == 2) {
                        // Format 1: (!function !llvm.func<...>)
                        // The second element is the actual type, just use it directly
                        break :blk list.at(1);
                    } else if (list.len() >= 3) {
                        // Format 2: (!function (inputs ...) (results ...))
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
                    } else {
                        return error.InvalidType;
                    }
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
    fn buildAttributeValue(self: *Builder, attr_expr: *const parser.AttrExpr, attr_key: []const u8) BuildError!mlir.MlirAttribute {
        const value = attr_expr.value;

        // Handle bare types - wrap them in TypeAttr for IRDL operations
        if (value.type == .type) {
            const mlir_type = try self.buildTypeFromValue(value);
            return mlir.c.mlirTypeAttrGet(mlir_type);
        }

        // Handle typed literals: (: value type)
        if (value.type == .has_type) {
            const val = value.data.has_type.value;
            const type_val = value.data.has_type.type_expr;

            // The type should be a type or function_type wrapper
            if (type_val.type != .type and type_val.type != .function_type) {
                std.debug.print("ERROR: Invalid attribute - expected type or function_type, got: {s}\n", .{@tagName(type_val.type)});
                std.debug.print("Attribute key: {s}\n", .{attr_key});
                return error.InvalidAttribute;
            }

            // Build the type from the type Value
            const mlir_type = try self.buildTypeFromValue(type_val);

            // Handle numeric values specially for typed attributes
            if (val.type == .number) {
                const num_str = val.data.atom;

                // Check if it's a float type (f16, f32, f64, bf16, etc.)
                const type_str = if (type_val.type == .type)
                    type_val.data.type
                else if (type_val.type == .identifier)
                    type_val.data.atom
                else
                    "";

                const is_float_type = std.mem.startsWith(u8, type_str, "f") or
                                     std.mem.eql(u8, type_str, "bf16");

                if (is_float_type) {
                    // Parse as float and create float attribute
                    const float_val = std.fmt.parseFloat(f64, num_str) catch |err| {
                        std.debug.print("ERROR: Invalid attribute - failed to parse float: {s}\n", .{num_str});
                        std.debug.print("Attribute key: {s}\n", .{attr_key});
                        std.debug.print("Parse error: {any}\n", .{err});
                        return error.InvalidAttribute;
                    };

                    // Create a typed float attribute
                    return mlir.c.mlirFloatAttrDoubleGet(self.ctx.ctx, mlir_type, float_val);
                } else {
                    // Parse as integer
                    const int_val = std.fmt.parseInt(i64, num_str, 10) catch |err| {
                        std.debug.print("ERROR: Invalid attribute - failed to parse integer: {s}\n", .{num_str});
                        std.debug.print("Attribute key: {s}\n", .{attr_key});
                        std.debug.print("Parse error: {any}\n", .{err});
                        return error.InvalidAttribute;
                    };

                    // Create a typed integer attribute
                    return mlir.c.mlirIntegerAttrGet(mlir_type, int_val);
                }
            }

            // For other typed values (like dense<...> : tensor<...>), serialize the entire
            // typed expression and parse it through MLIR's parser
            // Fall through to the general serialization path below
        }

        // Serialize the attribute expression to MLIR syntax
        const attr_str = try self.serializeValueToMLIR(value);
        defer self.allocator.free(attr_str);

        // Resolve attribute aliases in the string
        const resolved_attr_str = try self.resolveAttributeAliases(attr_str);
        defer self.allocator.free(resolved_attr_str);

        // Use MLIR's built-in attribute parser
        return mlir.Attribute.parse(self.ctx, resolved_attr_str) catch {
            std.debug.print("ERROR: Failed to parse attribute\n", .{});
            std.debug.print("Attribute key: {s}\n", .{attr_key});
            std.debug.print("Serialized value: {s}\n", .{attr_str});
            std.debug.print("Resolved value: {s}\n", .{resolved_attr_str});
            return error.InvalidAttribute;
        };
    }

    /// Resolve attribute alias references in a string
    /// Recursively expands #alias_name references with their definitions
    fn resolveAttributeAliases(self: *Builder, input: []const u8) BuildError![]const u8 {
        var result = std.ArrayList(u8){};
        errdefer result.deinit(self.allocator);

        var i: usize = 0;
        while (i < input.len) {
            // Look for '#' which starts an attribute alias reference
            if (input[i] == '#') {
                // Find the end of the identifier (skip the '#')
                var j = i + 1;
                while (j < input.len and (std.ascii.isAlphanumeric(input[j]) or input[j] == '_' or input[j] == '.')) : (j += 1) {}

                const alias_name_with_hash = input[i..j];
                const alias_name = input[i + 1 .. j]; // without the '#'

                // Try to resolve the alias (stored without #)
                if (self.attribute_alias_map.get(alias_name)) |definition| {
                    // Recursively resolve aliases in the definition
                    const resolved_def = try self.resolveAttributeAliases(definition);
                    defer self.allocator.free(resolved_def);
                    try result.appendSlice(self.allocator, resolved_def);
                    i = j;
                } else {
                    // Not an alias, keep the original '#alias_name'
                    try result.appendSlice(self.allocator, alias_name_with_hash);
                    i = j;
                }
            } else {
                try result.append(self.allocator, input[i]);
                i += 1;
            }
        }

        return result.toOwnedSlice(self.allocator);
    }

    /// Build a region
    fn buildRegion(self: *Builder, region: *const parser.Region) BuildError!mlir.MlirRegion {
        const mlir_region = mlir.Region.create();

        // Save the current block_map state to restore after building this region
        // (block labels are scoped to regions)
        const saved_block_map = self.block_map;
        self.block_map = std.StringHashMap(mlir.MlirBlock).init(self.allocator);
        defer {
            self.block_map.deinit();
            self.block_map = saved_block_map;
        }

        // Phase 1: Create all blocks and register their labels
        var mlir_blocks = try self.allocator.alloc(mlir.MlirBlock, region.blocks.len);
        defer self.allocator.free(mlir_blocks);

        for (region.blocks, 0..) |*block, i| {
            mlir_blocks[i] = try self.createBlockWithArguments(block);
            mlir.Region.appendBlock(mlir_region, mlir_blocks[i]);

            // Register block label if it has one
            if (block.label) |label| {
                try self.block_map.put(label, mlir_blocks[i]);
            }
        }

        // Phase 2: Build operations in each block (now successors can be resolved)
        for (region.blocks, 0..) |*block, i| {
            try self.buildBlockOperations(block, mlir_blocks[i]);
        }

        return mlir_region;
    }

    /// Create a block with its arguments (but without operations)
    fn createBlockWithArguments(self: *Builder, block: *const parser.Block) BuildError!mlir.MlirBlock {
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

        return mlir_block;
    }

    /// Build operations in a block (called after all blocks in region are created)
    fn buildBlockOperations(self: *Builder, block: *const parser.Block, mlir_block: mlir.MlirBlock) BuildError!void {
        for (block.operations) |*operation| {
            const mlir_op = try self.buildOperation(operation);
            mlir.Block.appendOperation(mlir_block, mlir_op);
        }
    }
};
