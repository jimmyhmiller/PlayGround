const std = @import("std");
const ast = @import("ast.zig");

pub const ValidationError = error{
    UnknownDialect,
    OutOfMemory,
};

/// C imports for MLIR
const c = @cImport({
    @cInclude("mlir-c/IR.h");
    @cInclude("mlir-c/Support.h");
    @cInclude("mlir-c/Dialect/Arith.h");
    @cInclude("mlir-c/Dialect/Func.h");
    @cInclude("mlir-c/Dialect/ControlFlow.h");
    @cInclude("mlir-c/Dialect/SCF.h");
    @cInclude("mlir-c/Dialect/MemRef.h");
    @cInclude("mlir-c/Dialect/Vector.h");
    @cInclude("mlir-c/Dialect/LLVM.h");
    @cInclude("mlir-introspection.h");
});

/// Dialect registry for validating operations
pub const DialectRegistry = struct {
    ctx: c.MlirContext,
    allocator: std.mem.Allocator,
    loaded_dialects: std.StringHashMap(void),

    pub fn init(allocator: std.mem.Allocator) !DialectRegistry {
        const ctx = c.mlirContextCreate();

        return .{
            .ctx = ctx,
            .allocator = allocator,
            .loaded_dialects = std.StringHashMap(void).init(allocator),
        };
    }

    pub fn deinit(self: *DialectRegistry) void {
        // Free all the keys (dialect names) we allocated
        var it = self.loaded_dialects.keyIterator();
        while (it.next()) |key| {
            self.allocator.free(key.*);
        }
        self.loaded_dialects.deinit();
        c.mlirContextDestroy(self.ctx);
    }

    /// Load a dialect by name
    pub fn loadDialect(self: *DialectRegistry, name: []const u8) !void {
        if (self.loaded_dialects.contains(name)) {
            return;
        }

        // Register the dialect handle based on name
        var found = false;

        if (std.mem.eql(u8, name, "arith")) {
            const handle = c.mlirGetDialectHandle__arith__();
            c.mlirDialectHandleRegisterDialect(handle, self.ctx);
            _ = c.mlirDialectHandleLoadDialect(handle, self.ctx);
            found = true;
        } else if (std.mem.eql(u8, name, "func")) {
            const handle = c.mlirGetDialectHandle__func__();
            c.mlirDialectHandleRegisterDialect(handle, self.ctx);
            _ = c.mlirDialectHandleLoadDialect(handle, self.ctx);
            found = true;
        } else if (std.mem.eql(u8, name, "cf")) {
            const handle = c.mlirGetDialectHandle__cf__();
            c.mlirDialectHandleRegisterDialect(handle, self.ctx);
            _ = c.mlirDialectHandleLoadDialect(handle, self.ctx);
            found = true;
        } else if (std.mem.eql(u8, name, "scf")) {
            const handle = c.mlirGetDialectHandle__scf__();
            c.mlirDialectHandleRegisterDialect(handle, self.ctx);
            _ = c.mlirDialectHandleLoadDialect(handle, self.ctx);
            found = true;
        } else if (std.mem.eql(u8, name, "memref")) {
            const handle = c.mlirGetDialectHandle__memref__();
            c.mlirDialectHandleRegisterDialect(handle, self.ctx);
            _ = c.mlirDialectHandleLoadDialect(handle, self.ctx);
            found = true;
        } else if (std.mem.eql(u8, name, "vector")) {
            const handle = c.mlirGetDialectHandle__vector__();
            c.mlirDialectHandleRegisterDialect(handle, self.ctx);
            _ = c.mlirDialectHandleLoadDialect(handle, self.ctx);
            found = true;
        } else if (std.mem.eql(u8, name, "llvm")) {
            const handle = c.mlirGetDialectHandle__llvm__();
            c.mlirDialectHandleRegisterDialect(handle, self.ctx);
            _ = c.mlirDialectHandleLoadDialect(handle, self.ctx);
            found = true;
        }

        if (!found) {
            return error.UnknownDialect;
        }

        // Mark as loaded
        const owned_name = try self.allocator.dupe(u8, name);
        try self.loaded_dialects.put(owned_name, {});
    }

    /// Check if an operation exists in a dialect
    pub fn validateOperation(self: *DialectRegistry, namespace: []const u8, op_name: []const u8) !bool {
        try self.loadDialect(namespace);

        // Create qualified operation name
        const qualified_name = try std.fmt.allocPrint(self.allocator, "{s}.{s}\x00", .{ namespace, op_name });
        defer self.allocator.free(qualified_name);

        const namespace_cstr = try std.fmt.allocPrint(self.allocator, "{s}\x00", .{namespace});
        defer self.allocator.free(namespace_cstr);

        const op_name_ref = c.mlirStringRefCreateFromCString(qualified_name.ptr);
        const ns_ref = c.mlirStringRefCreateFromCString(namespace_cstr.ptr);

        return c.mlirOperationBelongsToDialect(self.ctx, op_name_ref, ns_ref);
    }

    /// Enumerate all operations in a dialect
    pub fn enumerateOperations(self: *DialectRegistry, namespace: []const u8) !std.ArrayList([]const u8) {
        try self.loadDialect(namespace);

        var operations = std.ArrayList([]const u8){};
        errdefer {
            for (operations.items) |op| {
                self.allocator.free(op);
            }
            operations.deinit(self.allocator);
        }

        const namespace_cstr = try std.fmt.allocPrint(self.allocator, "{s}\x00", .{namespace});
        defer self.allocator.free(namespace_cstr);

        const ns_ref = c.mlirStringRefCreateFromCString(namespace_cstr.ptr);

        const Context = struct {
            ops: *std.ArrayList([]const u8),
            allocator: std.mem.Allocator,
        };

        var ctx = Context{
            .ops = &operations,
            .allocator = self.allocator,
        };

        const callback = struct {
            fn cb(op_name: c.MlirStringRef, user_data: ?*anyopaque) callconv(.c) bool {
                const context: *Context = @ptrCast(@alignCast(user_data.?));
                const name_slice = op_name.data[0..op_name.length];
                const owned_name = context.allocator.dupe(u8, name_slice) catch return false;
                context.ops.append(context.allocator, owned_name) catch return false;
                return true;
            }
        }.cb;

        _ = c.mlirEnumerateDialectOperations(self.ctx, ns_ref, callback, &ctx);

        return operations;
    }

    /// List all loaded dialects
    pub fn listLoadedDialects(self: *DialectRegistry) !std.ArrayList([]const u8) {
        var dialects = std.ArrayList([]const u8){};
        errdefer {
            for (dialects.items) |d| {
                self.allocator.free(d);
            }
            dialects.deinit(self.allocator);
        }

        var it = self.loaded_dialects.keyIterator();
        while (it.next()) |key| {
            const owned = try self.allocator.dupe(u8, key.*);
            try dialects.append(self.allocator, owned);
        }

        return dialects;
    }
};

/// Validator for AST nodes
pub const ASTValidator = struct {
    registry: *DialectRegistry,
    errors: std.ArrayList(ErrorInfo),
    allocator: std.mem.Allocator,

    pub const ErrorInfo = struct {
        message: []const u8,
        node_type: ast.NodeType,
    };

    pub fn init(allocator: std.mem.Allocator, registry: *DialectRegistry) ASTValidator {
        return .{
            .registry = registry,
            .errors = std.ArrayList(ErrorInfo){},
            .allocator = allocator,
        };
    }

    pub fn deinit(self: *ASTValidator) void {
        for (self.errors.items) |err| {
            self.allocator.free(err.message);
        }
        self.errors.deinit(self.allocator);
    }

    /// Validate a node and its children
    pub fn validate(self: *ASTValidator, node: *ast.Node) ValidationError!bool {
        switch (node.node_type) {
            .Module => try self.validateModule(node.data.module),
            .Operation => try self.validateOperation(node.data.operation),
            .Region => try self.validateRegion(node.data.region),
            .Block => try self.validateBlock(node.data.block),
            .Def => try self.validateBinding(node.data.binding),
            .Let => try self.validateLet(node.data.let_expr),
            .TypeAnnotation => {
                _ = try self.validate(node.data.type_annotation.value);
            },
            else => {},
        }

        return self.errors.items.len == 0;
    }

    fn validateModule(self: *ASTValidator, module: *ast.Module) !void {
        for (module.body.items) |child| {
            _ = try self.validate(child);
        }
    }

    fn validateOperation(self: *ASTValidator, op: *ast.Operation) !void {
        // Check if operation exists in its dialect
        if (op.namespace) |ns| {
            const valid = self.registry.validateOperation(ns, op.name) catch |err| {
                if (err == error.UnknownDialect) {
                    const msg = try std.fmt.allocPrint(
                        self.allocator,
                        "Unknown dialect: {s}",
                        .{ns},
                    );
                    try self.errors.append(self.allocator, .{
                        .message = msg,
                        .node_type = .Operation,
                    });
                    return;
                }
                return err;
            };

            if (!valid) {
                const msg = try std.fmt.allocPrint(
                    self.allocator,
                    "Unknown operation: {s}.{s}",
                    .{ ns, op.name },
                );
                try self.errors.append(self.allocator, .{
                    .message = msg,
                    .node_type = .Operation,
                });
            }
        } else {
            const msg = try std.fmt.allocPrint(
                self.allocator,
                "Operation missing namespace: {s}",
                .{op.name},
            );
            try self.errors.append(self.allocator, .{
                .message = msg,
                .node_type = .Operation,
            });
        }

        // Validate operands
        for (op.operands.items) |operand| {
            _ = try self.validate(operand);
        }

        // Validate regions
        for (op.regions.items) |region| {
            try self.validateRegion(region);
        }
    }

    fn validateRegion(self: *ASTValidator, region: *ast.Region) !void {
        for (region.blocks.items) |block| {
            try self.validateBlock(block);
        }
    }

    fn validateBlock(self: *ASTValidator, block: *ast.Block) !void {
        for (block.operations.items) |op| {
            _ = try self.validate(op);
        }
    }

    fn validateBinding(self: *ASTValidator, binding: *ast.Binding) !void {
        _ = try self.validate(binding.value);
    }

    fn validateLet(self: *ASTValidator, let_expr: *ast.LetExpr) !void {
        for (let_expr.bindings.items) |binding| {
            _ = try self.validateBinding(binding);
        }
        for (let_expr.body.items) |node| {
            _ = try self.validate(node);
        }
    }

    pub fn getErrors(self: *ASTValidator) []const ErrorInfo {
        return self.errors.items;
    }
};

test "dialect registry load arith" {
    const allocator = std.testing.allocator;

    var registry = try DialectRegistry.init(allocator);
    defer registry.deinit();

    try registry.loadDialect("arith");

    var dialects = try registry.listLoadedDialects();
    defer {
        for (dialects.items) |d| {
            allocator.free(d);
        }
        dialects.deinit(allocator);
    }

    try std.testing.expect(dialects.items.len > 0);
}

test "dialect registry validate operation" {
    const allocator = std.testing.allocator;

    var registry = try DialectRegistry.init(allocator);
    defer registry.deinit();

    const valid = try registry.validateOperation("arith", "addi");
    try std.testing.expect(valid);

    // Test that invalid operations are rejected
    const invalid = try registry.validateOperation("arith", "nonexistent");
    try std.testing.expect(!invalid);
}

test "dialect registry enumerate operations" {
    const allocator = std.testing.allocator;

    var registry = try DialectRegistry.init(allocator);
    defer registry.deinit();

    var ops = try registry.enumerateOperations("arith");
    defer {
        for (ops.items) |op| {
            allocator.free(op);
        }
        ops.deinit(allocator);
    }

    // Should have operations
    try std.testing.expect(ops.items.len > 0);

    // Check that addi is in the list
    var found_addi = false;
    for (ops.items) |op| {
        if (std.mem.indexOf(u8, op, "addi")) |_| {
            found_addi = true;
            break;
        }
    }
    try std.testing.expect(found_addi);
}
