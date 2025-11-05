/// Dialect Registry
/// Tracks IRDL dialect definitions and transform operations discovered during compilation
const std = @import("std");
const mlir = @import("mlir/c.zig");

pub const DialectRegistry = struct {
    allocator: std.mem.Allocator,
    /// IRDL dialect operations found (e.g., irdl.dialect)
    irdl_ops: std.ArrayList(mlir.MlirOperation),
    /// Transform operations found (e.g., transform.with_pdl_patterns, transform.sequence)
    transform_ops: std.ArrayList(mlir.MlirOperation),
    /// Set of dialect names that have been loaded
    loaded_dialects: std.StringHashMap(void),

    pub fn init(allocator: std.mem.Allocator) DialectRegistry {
        return DialectRegistry{
            .allocator = allocator,
            .irdl_ops = std.ArrayList(mlir.MlirOperation){},
            .transform_ops = std.ArrayList(mlir.MlirOperation){},
            .loaded_dialects = std.StringHashMap(void).init(allocator),
        };
    }

    pub fn deinit(self: *DialectRegistry) void {
        self.irdl_ops.deinit(self.allocator);
        self.transform_ops.deinit(self.allocator);

        var iter = self.loaded_dialects.keyIterator();
        while (iter.next()) |key| {
            self.allocator.free(key.*);
        }
        self.loaded_dialects.deinit();
    }

    /// Scan a module for IRDL and transform operations
    pub fn scanModule(self: *DialectRegistry, module: *mlir.Module) !void {
        // Find all IRDL dialect definitions
        const irdl_ops = try module.collectOperationsByPrefix(self.allocator, "irdl.");
        defer self.allocator.free(irdl_ops);

        for (irdl_ops) |op| {
            try self.irdl_ops.append(self.allocator, op);
        }

        // Find all transform operations
        const transform_ops = try module.collectOperationsByPrefix(self.allocator, "transform.");
        defer self.allocator.free(transform_ops);

        for (transform_ops) |op| {
            try self.transform_ops.append(self.allocator, op);
        }
    }

    /// Check if any IRDL dialects were found
    pub fn hasIRDLDialects(self: *DialectRegistry) bool {
        return self.irdl_ops.items.len > 0;
    }

    /// Check if any transform operations were found
    pub fn hasTransforms(self: *DialectRegistry) bool {
        return self.transform_ops.items.len > 0;
    }

    /// Mark a dialect as loaded
    pub fn markDialectLoaded(self: *DialectRegistry, name: []const u8) !void {
        const owned_name = try self.allocator.dupe(u8, name);
        try self.loaded_dialects.put(owned_name, {});
    }

    /// Check if a dialect has been loaded
    pub fn isDialectLoaded(self: *DialectRegistry, name: []const u8) bool {
        return self.loaded_dialects.contains(name);
    }

    /// Get all IRDL operations
    pub fn getIRDLOperations(self: *DialectRegistry) []const mlir.MlirOperation {
        return self.irdl_ops.items;
    }

    /// Get all transform operations
    pub fn getTransformOperations(self: *DialectRegistry) []const mlir.MlirOperation {
        return self.transform_ops.items;
    }

    /// Clear all registered operations (useful between compilation units)
    pub fn clear(self: *DialectRegistry) void {
        self.irdl_ops.clearRetainingCapacity();
        self.transform_ops.clearRetainingCapacity();
    }
};

test "DialectRegistry basic operations" {
    const allocator = std.testing.allocator;

    var registry = DialectRegistry.init(allocator);
    defer registry.deinit();

    try std.testing.expect(!registry.hasIRDLDialects());
    try std.testing.expect(!registry.hasTransforms());

    try registry.markDialectLoaded("mlsp");
    try std.testing.expect(registry.isDialectLoaded("mlsp"));
    try std.testing.expect(!registry.isDialectLoaded("other"));
}
