const std = @import("std");
const parser = @import("parser.zig");
const Operation = parser.Operation;

/// Type of metadata module
pub const MetadataType = enum {
    transform,
    irdl,
    unknown,
};

/// Detect the type of metadata module by scanning its operations
pub fn detectMetadataType(operation: *const Operation) MetadataType {
    // Only builtin.module can be metadata
    if (!std.mem.eql(u8, operation.name, "builtin.module")) {
        return .unknown;
    }

    // Scan regions for operations that indicate type
    return scanRegionsForType(operation);
}

fn scanRegionsForType(operation: *const Operation) MetadataType {
    var has_irdl = false;
    var has_transform = false;

    for (operation.regions) |region| {
        for (region.blocks) |block| {
            for (block.operations) |op| {
                // Check operation name prefixes
                if (std.mem.startsWith(u8, op.name, "irdl.")) {
                    has_irdl = true;
                } else if (std.mem.startsWith(u8, op.name, "transform.")) {
                    has_transform = true;
                }

                // Recursively check nested operations
                const nested_type = scanRegionsForType(&op);
                switch (nested_type) {
                    .irdl => has_irdl = true,
                    .transform => has_transform = true,
                    .unknown => {},
                }
            }
        }
    }

    // Prioritize IRDL over transform if both found
    if (has_irdl) {
        return .irdl;
    } else if (has_transform) {
        return .transform;
    } else {
        return .unknown;
    }
}

/// Check if an operation is a metadata module (has {:metadata unit} attribute)
pub fn isMetadataModule(operation: *const Operation) bool {
    // Only builtin.module can be metadata
    if (!std.mem.eql(u8, operation.name, "builtin.module")) {
        return false;
    }

    // Check for {:metadata unit} attribute
    for (operation.attributes) |attr| {
        if (std.mem.eql(u8, attr.key, "metadata")) {
            return true;
        }
    }

    return false;
}

/// Check if an operation is an application module (not metadata)
pub fn isApplicationModule(operation: *const Operation) bool {
    return !isMetadataModule(operation);
}
