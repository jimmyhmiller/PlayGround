/// Dialect Registry
/// Tracks IRDL dialect definitions and transform operations discovered during compilation
///
/// Architecture:
/// - Scans MLIR modules for irdl.* operations (dialect definitions)
/// - Scans for transform.* operations (rewrite patterns and lowerings)
/// - Supports multiple transform types: named_sequence, with_pdl_patterns, sequence
/// - Validates that transforms are properly structured for application
const std = @import("std");
const mlir = @import("mlir/c.zig");

/// Types of transform operations we support
pub const TransformType = enum {
    named_sequence,      // transform.named_sequence (MLIR standard)
    with_pdl_patterns,   // transform.with_pdl_patterns (PDL-based transforms)
    sequence,            // transform.sequence (transform sequences)

    /// Classify a transform operation by its name
    pub fn fromOpName(op_name: []const u8) ?TransformType {
        if (std.mem.eql(u8, op_name, "transform.named_sequence")) return .named_sequence;
        if (std.mem.eql(u8, op_name, "transform.with_pdl_patterns")) return .with_pdl_patterns;
        if (std.mem.eql(u8, op_name, "transform.sequence")) return .sequence;
        return null;
    }

    /// Check if this is a valid root transform (can be applied at top-level)
    pub fn isRootTransform(self: TransformType) bool {
        return switch (self) {
            .named_sequence, .with_pdl_patterns => true,
            .sequence => false, // Usually nested inside other transforms
        };
    }
};

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

        // Find all transform.* operations
        const all_transform_ops = try module.collectOperationsByPrefix(self.allocator, "transform.");
        defer self.allocator.free(all_transform_ops);

        // Filter to only supported transform types
        for (all_transform_ops) |op| {
            const op_name = mlir.Operation.getName(op);

            // Check if this is a supported transform type
            if (TransformType.fromOpName(op_name)) |_| {
                try self.transform_ops.append(self.allocator, op);
            }
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

    /// Get the type of a transform operation
    pub fn getTransformType(op: mlir.MlirOperation) ?TransformType {
        const op_name = mlir.Operation.getName(op);
        return TransformType.fromOpName(op_name);
    }

    /// Check if a transform operation is a valid root transform
    pub fn isRootTransform(op: mlir.MlirOperation) bool {
        if (getTransformType(op)) |transform_type| {
            return transform_type.isRootTransform();
        }
        return false;
    }

    /// Get all root transforms (suitable for top-level application)
    pub fn getRootTransforms(self: *DialectRegistry, allocator: std.mem.Allocator) ![]mlir.MlirOperation {
        var roots = std.ArrayList(mlir.MlirOperation){};
        errdefer roots.deinit(allocator);

        for (self.transform_ops.items) |op| {
            if (isRootTransform(op)) {
                try roots.append(allocator, op);
            }
        }

        return try roots.toOwnedSlice(allocator);
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
