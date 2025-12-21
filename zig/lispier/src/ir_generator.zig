const std = @import("std");
const ast = @import("ast.zig");
const reader_types = @import("reader_types.zig");
const mlir_integration = @import("mlir_integration.zig");

/// C imports for MLIR - use the same import as mlir_integration to ensure type compatibility
const c = mlir_integration.c;

/// Helper to create null-terminated string for C API
fn allocPrintZ(allocator: std.mem.Allocator, comptime fmt: []const u8, args: anytype) ![:0]u8 {
    const str = try std.fmt.allocPrint(allocator, fmt, args);
    defer allocator.free(str);
    const result = try allocator.allocSentinel(u8, str.len, 0);
    @memcpy(result, str);
    return result;
}

pub const GeneratorError = error{
    UndefinedSymbol,
    TypeParseError,
    AttributeParseError,
    InvalidOperandType,
    RegionCreationFailed,
    BlockCreationFailed,
    OperationCreationFailed,
    UnsupportedNodeType,
    MissingResultType,
    MissingType,
    OutOfMemory,
};

/// Symbol table for tracking SSA values by name
pub const SymbolTable = struct {
    allocator: std.mem.Allocator,
    scopes: std.ArrayList(Scope),

    const Scope = struct {
        values: std.StringHashMap(c.MlirValue),
        allocator: std.mem.Allocator,

        fn init(allocator: std.mem.Allocator) Scope {
            return .{
                .values = std.StringHashMap(c.MlirValue).init(allocator),
                .allocator = allocator,
            };
        }

        fn deinit(self: *Scope) void {
            // Free duplicated keys
            var it = self.values.keyIterator();
            while (it.next()) |key| {
                self.allocator.free(key.*);
            }
            self.values.deinit();
        }
    };

    pub fn init(allocator: std.mem.Allocator) !SymbolTable {
        var st = SymbolTable{
            .allocator = allocator,
            .scopes = std.ArrayList(Scope){},
        };
        // Start with global scope
        try st.scopes.append(allocator, Scope.init(allocator));
        return st;
    }

    pub fn deinit(self: *SymbolTable) void {
        for (self.scopes.items) |*scope| {
            scope.deinit();
        }
        self.scopes.deinit(self.allocator);
    }

    pub fn pushScope(self: *SymbolTable) !void {
        try self.scopes.append(self.allocator, Scope.init(self.allocator));
    }

    pub fn popScope(self: *SymbolTable) void {
        if (self.scopes.items.len > 1) {
            const last_idx = self.scopes.items.len - 1;
            var scope = self.scopes.items[last_idx];
            scope.deinit();
            self.scopes.items.len -= 1;
        }
    }

    pub fn define(self: *SymbolTable, name: []const u8, value: c.MlirValue) !void {
        const current = &self.scopes.items[self.scopes.items.len - 1];
        const owned_name = try self.allocator.dupe(u8, name);
        try current.values.put(owned_name, value);
    }

    pub fn lookup(self: *SymbolTable, name: []const u8) ?c.MlirValue {
        // Search from innermost to outermost scope
        var i = self.scopes.items.len;
        while (i > 0) {
            i -= 1;
            if (self.scopes.items[i].values.get(name)) |value| {
                return value;
            }
        }
        return null;
    }
};

/// IR Generator - converts AST nodes to MLIR IR
pub const IRGenerator = struct {
    allocator: std.mem.Allocator,
    ctx: c.MlirContext,
    module: c.MlirModule,
    symbol_table: SymbolTable,
    location: c.MlirLocation,

    pub fn init(allocator: std.mem.Allocator, mlir_ctx: c.MlirContext) !IRGenerator {
        const location = c.mlirLocationUnknownGet(mlir_ctx);

        return .{
            .allocator = allocator,
            .ctx = mlir_ctx,
            .module = c.MlirModule{ .ptr = null },
            .symbol_table = try SymbolTable.init(allocator),
            .location = location,
        };
    }

    pub fn deinit(self: *IRGenerator) void {
        self.symbol_table.deinit();
        if (self.module.ptr != null) {
            c.mlirModuleDestroy(self.module);
        }
    }

    /// Generate MLIR module from AST nodes
    pub fn generate(self: *IRGenerator, nodes: []*ast.Node) !c.MlirModule {
        // Create empty module
        self.module = c.mlirModuleCreateEmpty(self.location);

        // Get module body block
        const module_body = c.mlirModuleGetBody(self.module);

        for (nodes) |node| {
            _ = try self.generateNode(node, module_body);
        }

        return self.module;
    }

    /// Generate IR for a single node, returns the result value if any
    fn generateNode(self: *IRGenerator, node: *ast.Node, block: c.MlirBlock) GeneratorError!?c.MlirValue {
        return switch (node.node_type) {
            .Module => try self.generateModule(node.data.module, block),
            .Operation => try self.generateOperation(node.data.operation, block),
            .Region => error.UnsupportedNodeType, // Regions are generated as part of operations
            .Block => error.UnsupportedNodeType, // Blocks are generated as part of regions
            .Def => try self.generateDef(node.data.binding, block),
            .Let => try self.generateLet(node.data.let_expr, block),
            .TypeAnnotation => try self.generateTypeAnnotation(node.data.type_annotation.value, node.data.type_annotation.typ, block),
            .FunctionType => error.UnsupportedNodeType, // Only used in attributes
            .Literal => try self.generateLiteral(node.data.literal, block),
        };
    }

    /// Generate module contents
    fn generateModule(self: *IRGenerator, module: *ast.Module, block: c.MlirBlock) !?c.MlirValue {
        for (module.body.items) |child| {
            _ = try self.generateNode(child, block);
        }
        return null;
    }

    /// Generate an MLIR operation
    fn generateOperation(self: *IRGenerator, op: *ast.Operation, block: c.MlirBlock) !?c.MlirValue {
        // Build qualified operation name
        const qualified_name = try op.getQualifiedName(self.allocator);
        defer self.allocator.free(qualified_name);

        // Null-terminate for C API
        const name_z = try allocPrintZ(self.allocator, "{s}", .{qualified_name});
        defer self.allocator.free(name_z);

        const name_ref = c.MlirStringRef{
            .data = name_z.ptr,
            .length = name_z.len,
        };

        // Initialize operation state
        var state = c.mlirOperationStateGet(name_ref, self.location);

        // First pass: infer type from typed operands
        // For operations like arith.addi, all operands share the same type
        var inferred_type: ?c.MlirType = null;
        for (op.operands.items) |operand_node| {
            if (try self.getOperandType(operand_node)) |t| {
                inferred_type = t;
                break;
            }
        }

        // Generate operands with type inference
        var operand_values = std.ArrayList(c.MlirValue){};
        defer operand_values.deinit(self.allocator);

        for (op.operands.items) |operand_node| {
            const value = try self.resolveOperand(operand_node, block, inferred_type);
            try operand_values.append(self.allocator, value);
        }

        if (operand_values.items.len > 0) {
            c.mlirOperationStateAddOperands(&state, @intCast(operand_values.items.len), operand_values.items.ptr);
        }

        // Generate result types
        var result_mlir_types = std.ArrayList(c.MlirType){};
        defer result_mlir_types.deinit(self.allocator);

        for (op.result_types.items) |result_type| {
            const mlir_type = try self.parseType(result_type.name);
            try result_mlir_types.append(self.allocator, mlir_type);
        }

        if (result_mlir_types.items.len > 0) {
            c.mlirOperationStateAddResults(&state, @intCast(result_mlir_types.items.len), result_mlir_types.items.ptr);
        } else if (op.regions.items.len == 0) {
            // Only enable type inference for operations without regions
            // that aren't terminators. Operations with regions (like func.func)
            // and terminators (like func.return) don't support type inference.
            const skip_inference = std.mem.eql(u8, qualified_name, "func.return") or
                std.mem.eql(u8, qualified_name, "cf.br") or
                std.mem.eql(u8, qualified_name, "cf.cond_br") or
                std.mem.eql(u8, qualified_name, "scf.yield");

            if (!skip_inference) {
                c.mlirOperationStateEnableResultTypeInference(&state);
            }
        }

        // Generate attributes
        try self.addAttributes(&state, &op.attributes);

        // Generate regions
        for (op.regions.items) |region| {
            const mlir_region = try self.generateRegion(region);
            c.mlirOperationStateAddOwnedRegions(&state, 1, &mlir_region);
        }

        // Create the operation
        const mlir_op = c.mlirOperationCreate(&state);

        // Append to block
        c.mlirBlockAppendOwnedOperation(block, mlir_op);

        // Return first result (if any)
        const num_results = c.mlirOperationGetNumResults(mlir_op);
        if (num_results > 0) {
            return c.mlirOperationGetResult(mlir_op, 0);
        }
        return null;
    }

    /// Generate a region with its blocks
    fn generateRegion(self: *IRGenerator, region: *ast.Region) !c.MlirRegion {
        const mlir_region = c.mlirRegionCreate();

        for (region.blocks.items) |block| {
            const mlir_block = try self.generateBlock(block);
            c.mlirRegionAppendOwnedBlock(mlir_region, mlir_block);
        }

        return mlir_region;
    }

    /// Generate a block with its arguments and operations
    fn generateBlock(self: *IRGenerator, block: *ast.Block) !c.MlirBlock {
        // Push new scope for block arguments
        try self.symbol_table.pushScope();
        errdefer self.symbol_table.popScope();

        // Prepare block argument types and locations
        var arg_types = std.ArrayList(c.MlirType){};
        defer arg_types.deinit(self.allocator);

        var arg_locs = std.ArrayList(c.MlirLocation){};
        defer arg_locs.deinit(self.allocator);

        for (block.arguments.items) |arg| {
            if (arg.type) |t| {
                const mlir_type = try self.parseType(t.name);
                try arg_types.append(self.allocator, mlir_type);
                try arg_locs.append(self.allocator, self.location);
            } else {
                return error.MissingType;
            }
        }

        // Create block
        const mlir_block = c.mlirBlockCreate(
            @intCast(arg_types.items.len),
            if (arg_types.items.len > 0) arg_types.items.ptr else null,
            if (arg_locs.items.len > 0) arg_locs.items.ptr else null,
        );

        // Register block arguments in symbol table
        for (block.arguments.items, 0..) |arg, i| {
            const block_arg = c.mlirBlockGetArgument(mlir_block, @intCast(i));
            try self.symbol_table.define(arg.name, block_arg);
        }

        // Generate operations in block
        for (block.operations.items) |op_node| {
            _ = try self.generateNode(op_node, mlir_block);
        }

        self.symbol_table.popScope();
        return mlir_block;
    }

    /// Generate def (variable binding)
    fn generateDef(self: *IRGenerator, binding: *ast.Binding, block: c.MlirBlock) !?c.MlirValue {
        // Generate the value expression
        const value = try self.generateNode(binding.value, block);

        if (value) |v| {
            if (binding.names.items.len == 1) {
                // Single result
                try self.symbol_table.define(binding.names.items[0], v);
                return v;
            } else {
                // Multi-result destructuring
                // Get the defining operation
                const op = c.mlirOpResultGetOwner(v);
                const num_results = c.mlirOperationGetNumResults(op);

                if (num_results != binding.names.items.len) {
                    return error.InvalidOperandType;
                }

                for (binding.names.items, 0..) |name, i| {
                    const result = c.mlirOperationGetResult(op, @intCast(i));
                    try self.symbol_table.define(name, result);
                }
                return v;
            }
        }

        return null;
    }

    /// Generate let expression
    fn generateLet(self: *IRGenerator, let_expr: *ast.LetExpr, block: c.MlirBlock) !?c.MlirValue {
        // Push scope for let bindings
        try self.symbol_table.pushScope();
        defer self.symbol_table.popScope();

        // Generate bindings
        for (let_expr.bindings.items) |binding| {
            _ = try self.generateDef(binding, block);
        }

        // Generate body, return last value
        var last_value: ?c.MlirValue = null;
        for (let_expr.body.items) |body_node| {
            last_value = try self.generateNode(body_node, block);
        }

        return last_value;
    }

    /// Generate type annotation (typed literal -> constant)
    fn generateTypeAnnotation(self: *IRGenerator, value_node: *ast.Node, typ: *ast.Type, block: c.MlirBlock) !?c.MlirValue {
        // For literals with type annotations, generate appropriate constant ops
        if (value_node.node_type == .Literal) {
            const literal = value_node.data.literal;
            if (literal.type == .Number) {
                return try self.generateConstant(literal.data.number, typ.name, block);
            }
        }

        // Otherwise, just generate the value
        return try self.generateNode(value_node, block);
    }

    /// Generate a constant operation with a type name string
    fn generateConstant(self: *IRGenerator, value: f64, type_name: []const u8, block: c.MlirBlock) !c.MlirValue {
        const mlir_type = try self.parseType(type_name);
        return self.generateConstantWithType(value, mlir_type, block);
    }

    /// Generate a constant operation with an MlirType directly
    fn generateConstantWithType(self: *IRGenerator, value: f64, mlir_type: c.MlirType, block: c.MlirBlock) !c.MlirValue {
        // Create arith.constant operation
        const op_name = c.MlirStringRef{
            .data = "arith.constant",
            .length = 14,
        };
        var state = c.mlirOperationStateGet(op_name, self.location);

        // Add result type
        var result_type = mlir_type;
        c.mlirOperationStateAddResults(&state, 1, &result_type);

        // Create value attribute based on type
        const attr = self.createNumberAttributeWithType(value, mlir_type);
        const attr_name = c.MlirStringRef{
            .data = "value",
            .length = 5,
        };
        const name_id = c.mlirIdentifierGet(self.ctx, attr_name);
        const named_attr = c.MlirNamedAttribute{
            .name = name_id,
            .attribute = attr,
        };
        c.mlirOperationStateAddAttributes(&state, 1, &named_attr);

        const mlir_op = c.mlirOperationCreate(&state);
        c.mlirBlockAppendOwnedOperation(block, mlir_op);

        return c.mlirOperationGetResult(mlir_op, 0);
    }

    /// Generate literal (symbol reference)
    fn generateLiteral(self: *IRGenerator, value: *reader_types.Value, block: c.MlirBlock) !?c.MlirValue {
        _ = block;
        return switch (value.type) {
            .Symbol => {
                // Symbol reference - lookup in symbol table
                const sym = value.data.symbol;
                return self.symbol_table.lookup(sym.name) orelse error.UndefinedSymbol;
            },
            .List => {
                // Lists that made it to IR generation are compiler directives like require-dialect
                // They don't produce any IR, so just return null
                return null;
            },
            else => error.UnsupportedNodeType,
        };
    }

    /// Get the type of an operand without generating code
    /// Used for type inference
    fn getOperandType(self: *IRGenerator, node: *ast.Node) !?c.MlirType {
        return switch (node.node_type) {
            .Literal => {
                const literal = node.data.literal;
                if (literal.type == .Symbol) {
                    // Look up the symbol and get its type
                    if (self.symbol_table.lookup(literal.data.symbol.name)) |value| {
                        return c.mlirValueGetType(value);
                    }
                }
                // Bare number literal - no type info yet
                return null;
            },
            .TypeAnnotation => {
                const ta = node.data.type_annotation;
                return try self.parseType(ta.typ.name);
            },
            .Operation => {
                // Can't easily get type without generating - skip for now
                return null;
            },
            else => null,
        };
    }

    /// Resolve an operand node to an MLIR value
    /// inferred_type is used for bare number literals when no explicit type is given
    fn resolveOperand(self: *IRGenerator, node: *ast.Node, block: c.MlirBlock, inferred_type: ?c.MlirType) (GeneratorError || error{OutOfMemory})!c.MlirValue {
        return switch (node.node_type) {
            .Literal => {
                const literal = node.data.literal;
                if (literal.type == .Symbol) {
                    return self.symbol_table.lookup(literal.data.symbol.name) orelse error.UndefinedSymbol;
                }
                if (literal.type == .Number) {
                    // Bare number - use inferred type or default to i64
                    if (inferred_type) |typ| {
                        return self.generateConstantWithType(literal.data.number, typ, block);
                    } else {
                        // Default to i64 for untyped integers
                        return self.generateConstant(literal.data.number, "i64", block);
                    }
                }
                return error.InvalidOperandType;
            },
            .TypeAnnotation => {
                const ta = node.data.type_annotation;
                // Generate typed constant
                if (ta.value.node_type == .Literal and ta.value.data.literal.type == .Number) {
                    return self.generateConstant(ta.value.data.literal.data.number, ta.typ.name, block);
                }
                return self.resolveOperand(ta.value, block, inferred_type);
            },
            .Operation => {
                // Generate the operation and return its result
                const result = self.generateOperation(node.data.operation, block) catch |err| return err;
                return result orelse error.InvalidOperandType;
            },
            else => error.InvalidOperandType,
        };
    }

    /// Parse a type string into MlirType
    fn parseType(self: *IRGenerator, type_name: []const u8) !c.MlirType {
        // Handle common built-in types
        if (std.mem.eql(u8, type_name, "i1")) return c.mlirIntegerTypeGet(self.ctx, 1);
        if (std.mem.eql(u8, type_name, "i8")) return c.mlirIntegerTypeGet(self.ctx, 8);
        if (std.mem.eql(u8, type_name, "i16")) return c.mlirIntegerTypeGet(self.ctx, 16);
        if (std.mem.eql(u8, type_name, "i32")) return c.mlirIntegerTypeGet(self.ctx, 32);
        if (std.mem.eql(u8, type_name, "i64")) return c.mlirIntegerTypeGet(self.ctx, 64);
        if (std.mem.eql(u8, type_name, "f16")) return c.mlirF16TypeGet(self.ctx);
        if (std.mem.eql(u8, type_name, "f32")) return c.mlirF32TypeGet(self.ctx);
        if (std.mem.eql(u8, type_name, "f64")) return c.mlirF64TypeGet(self.ctx);
        if (std.mem.eql(u8, type_name, "index")) return c.mlirIndexTypeGet(self.ctx);

        // For complex types (memref, tensor, etc.), use parser
        const type_z = try allocPrintZ(self.allocator, "{s}", .{type_name});
        defer self.allocator.free(type_z);

        const type_ref = c.MlirStringRef{
            .data = type_z.ptr,
            .length = type_z.len,
        };
        const parsed_type = c.mlirTypeParseGet(self.ctx, type_ref);

        if (c.mlirTypeIsNull(parsed_type)) {
            return error.TypeParseError;
        }

        return parsed_type;
    }

    /// Create a number attribute for the given value and type name
    fn createNumberAttribute(self: *IRGenerator, value: f64, type_name: []const u8) !c.MlirAttribute {
        const mlir_type = try self.parseType(type_name);
        return self.createNumberAttributeWithType(value, mlir_type);
    }

    /// Create a number attribute for the given value and MlirType directly
    fn createNumberAttributeWithType(self: *IRGenerator, value: f64, mlir_type: c.MlirType) c.MlirAttribute {
        // Check if it's an integer type
        if (c.mlirTypeIsAInteger(mlir_type)) {
            return c.mlirIntegerAttrGet(mlir_type, @intFromFloat(value));
        }

        // Check if it's a float type
        if (c.mlirTypeIsAFloat(mlir_type)) {
            return c.mlirFloatAttrDoubleGet(self.ctx, mlir_type, value);
        }

        // Index type - also use integer attribute
        if (c.mlirTypeIsAIndex(mlir_type)) {
            return c.mlirIntegerAttrGet(mlir_type, @intFromFloat(value));
        }

        // Default to integer attribute for unknown types
        return c.mlirIntegerAttrGet(mlir_type, @intFromFloat(value));
    }

    /// Add attributes from AST to operation state
    fn addAttributes(self: *IRGenerator, state: *c.MlirOperationState, attributes: *std.StringHashMap(ast.AttributeValue)) !void {
        var named_attrs = std.ArrayList(c.MlirNamedAttribute){};
        defer named_attrs.deinit(self.allocator);

        var it = attributes.iterator();
        while (it.next()) |entry| {
            const name_z = try allocPrintZ(self.allocator, "{s}", .{entry.key_ptr.*});
            defer self.allocator.free(name_z);

            const name_ref = c.MlirStringRef{
                .data = name_z.ptr,
                .length = name_z.len,
            };
            const name_id = c.mlirIdentifierGet(self.ctx, name_ref);

            const attr = try self.convertAttributeValue(entry.value_ptr.*);

            try named_attrs.append(self.allocator, c.MlirNamedAttribute{
                .name = name_id,
                .attribute = attr,
            });
        }

        if (named_attrs.items.len > 0) {
            c.mlirOperationStateAddAttributes(state, @intCast(named_attrs.items.len), named_attrs.items.ptr);
        }
    }

    /// Convert AST AttributeValue to MlirAttribute
    fn convertAttributeValue(self: *IRGenerator, value: ast.AttributeValue) !c.MlirAttribute {
        return switch (value) {
            .string => |s| blk: {
                const str_z = try allocPrintZ(self.allocator, "{s}", .{s});
                defer self.allocator.free(str_z);
                const str_ref = c.MlirStringRef{
                    .data = str_z.ptr,
                    .length = str_z.len,
                };
                break :blk c.mlirStringAttrGet(self.ctx, str_ref);
            },
            .number => |n| blk: {
                // Default to f64 for numbers without explicit type
                const f64_type = c.mlirF64TypeGet(self.ctx);
                break :blk c.mlirFloatAttrDoubleGet(self.ctx, f64_type, n);
            },
            .boolean => |b| c.mlirBoolAttrGet(self.ctx, if (b) 1 else 0),
            .type => |t| c.mlirTypeAttrGet(try self.parseType(t.name)),
            .function_type => |ft| try self.convertFunctionType(ft),
            .array => |arr| blk: {
                var attrs = std.ArrayList(c.MlirAttribute){};
                defer attrs.deinit(self.allocator);

                for (arr.items) |item| {
                    try attrs.append(self.allocator, try self.convertAttributeValue(item));
                }

                break :blk c.mlirArrayAttrGet(self.ctx, @intCast(attrs.items.len), if (attrs.items.len > 0) attrs.items.ptr else null);
            },
            .typed_number => |tn| blk: {
                // Create a typed number attribute
                break :blk try self.createNumberAttribute(tn.value, tn.typ.name);
            },
        };
    }

    /// Convert function type to type attribute
    fn convertFunctionType(self: *IRGenerator, ft: *ast.FunctionType) !c.MlirAttribute {
        var input_types = std.ArrayList(c.MlirType){};
        defer input_types.deinit(self.allocator);

        var result_types = std.ArrayList(c.MlirType){};
        defer result_types.deinit(self.allocator);

        for (ft.arg_types.items) |t| {
            try input_types.append(self.allocator, try self.parseType(t.name));
        }

        for (ft.return_types.items) |t| {
            try result_types.append(self.allocator, try self.parseType(t.name));
        }

        const func_type = c.mlirFunctionTypeGet(
            self.ctx,
            @intCast(input_types.items.len),
            if (input_types.items.len > 0) input_types.items.ptr else null,
            @intCast(result_types.items.len),
            if (result_types.items.len > 0) result_types.items.ptr else null,
        );

        return c.mlirTypeAttrGet(func_type);
    }

    /// Print the generated module to stderr for debugging
    pub fn printModule(self: *IRGenerator) void {
        if (self.module.ptr == null) return;

        const callback = struct {
            fn cb(str: c.MlirStringRef, user_data: ?*anyopaque) callconv(.c) void {
                _ = user_data;
                if (str.data != null and str.length > 0) {
                    std.debug.print("{s}", .{str.data[0..str.length]});
                }
            }
        }.cb;

        const module_op = c.mlirModuleGetOperation(self.module);
        c.mlirOperationPrint(module_op, callback, null);
        std.debug.print("\n", .{});
    }

    /// Verify the generated module against MLIR's verifier
    /// Returns true if the module is valid, false otherwise
    pub fn verify(self: *IRGenerator) bool {
        if (self.module.ptr == null) return false;

        const module_op = c.mlirModuleGetOperation(self.module);
        return c.mlirOperationVerify(module_op);
    }

    /// Print the module to a string (for testing)
    pub fn printModuleToString(self: *IRGenerator, allocator: std.mem.Allocator) ![]u8 {
        if (self.module.ptr == null) return allocator.dupe(u8, "");

        var result = std.ArrayList(u8){};
        errdefer result.deinit(allocator);

        const Context = struct {
            list: *std.ArrayList(u8),
            alloc: std.mem.Allocator,
        };

        var ctx = Context{
            .list = &result,
            .alloc = allocator,
        };

        const callback = struct {
            fn cb(str: c.MlirStringRef, user_data: ?*anyopaque) callconv(.c) void {
                const context: *Context = @ptrCast(@alignCast(user_data.?));
                if (str.data != null and str.length > 0) {
                    context.list.appendSlice(context.alloc, str.data[0..str.length]) catch {};
                }
            }
        }.cb;

        const module_op = c.mlirModuleGetOperation(self.module);
        c.mlirOperationPrint(module_op, callback, &ctx);

        return result.toOwnedSlice(allocator);
    }
};

// Tests
test "symbol table basic operations" {
    const allocator = std.testing.allocator;

    var st = try SymbolTable.init(allocator);
    defer st.deinit();

    // Create a dummy value (we can't easily create a real MlirValue in tests)
    const dummy_value = c.MlirValue{ .ptr = @ptrFromInt(0x1234) };

    try st.define("x", dummy_value);
    try std.testing.expect(st.lookup("x") != null);
    try std.testing.expect(st.lookup("y") == null);
}

test "symbol table scoping" {
    const allocator = std.testing.allocator;

    var st = try SymbolTable.init(allocator);
    defer st.deinit();

    const v1 = c.MlirValue{ .ptr = @ptrFromInt(0x1111) };
    const v2 = c.MlirValue{ .ptr = @ptrFromInt(0x2222) };

    try st.define("x", v1);
    try st.pushScope();
    try st.define("x", v2); // Shadow in inner scope

    // Inner scope sees v2
    const inner_x = st.lookup("x");
    try std.testing.expect(inner_x != null);
    try std.testing.expect(inner_x.?.ptr == v2.ptr);

    st.popScope();

    // Outer scope sees v1
    const outer_x = st.lookup("x");
    try std.testing.expect(outer_x != null);
    try std.testing.expect(outer_x.?.ptr == v1.ptr);
}

test "symbol table nested scope lookup" {
    const allocator = std.testing.allocator;

    var st = try SymbolTable.init(allocator);
    defer st.deinit();

    const v1 = c.MlirValue{ .ptr = @ptrFromInt(0x1111) };

    try st.define("outer", v1);
    try st.pushScope();

    // Can still see outer variable from inner scope
    try std.testing.expect(st.lookup("outer") != null);

    st.popScope();
}
