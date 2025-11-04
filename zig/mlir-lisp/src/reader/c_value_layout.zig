const std = @import("std");
const reader = @import("../reader.zig");
const vector = @import("../collections/vector.zig");

/// Flat C-compatible layout for Value types
/// All fields are always present; interpretation depends on type_tag
///
/// Memory layout: 56 bytes total
/// - type_tag: 1 byte (ValueType enum)
/// - padding: 7 bytes (alignment)
/// - data_ptr: 8 bytes (universal pointer field)
/// - data_len: 8 bytes (universal length field)
/// - data_capacity: 8 bytes (for collections)
/// - data_elem_size: 8 bytes (for collections)
/// - extra_ptr1: 8 bytes (for complex types)
/// - extra_ptr2: 8 bytes (for complex types)
pub const CValueLayout = extern struct {
    /// Type tag indicating which Value variant this represents
    type_tag: u8,

    /// Padding for 8-byte alignment
    _padding: [7]u8,

    /// Universal pointer field
    /// - For atoms (identifier, number, string, etc.): points to string data
    /// - For collections (list, vector, map): points to element array
    /// - Null for types that don't use it
    data_ptr: ?[*]u8,

    /// Universal length field
    /// - For atoms: string length in bytes
    /// - For collections: number of elements
    /// - Zero for types that don't use it
    data_len: usize,

    /// Capacity field (collections only)
    /// - For collections: allocated capacity
    /// - Zero for non-collection types
    data_capacity: usize,

    /// Element size field (collections only)
    /// - For collections: size of each element in bytes
    /// - Zero for non-collection types
    data_elem_size: usize,

    /// First extra pointer
    /// - For attr_expr: pointer to wrapped CValueLayout
    /// - For has_type: pointer to value CValueLayout
    /// - For function_type: pointer to inputs CVectorLayout
    /// - Null otherwise
    extra_ptr1: ?[*]u8,

    /// Second extra pointer
    /// - For has_type: pointer to type_expr CValueLayout
    /// - For function_type: pointer to results CVectorLayout
    /// - Null otherwise
    extra_ptr2: ?[*]u8,

    /// Create an empty layout for a given type
    pub fn empty(value_type: reader.ValueType) CValueLayout {
        return .{
            .type_tag = @intFromEnum(value_type),
            ._padding = [_]u8{0} ** 7,
            .data_ptr = null,
            .data_len = 0,
            .data_capacity = 0,
            .data_elem_size = 0,
            .extra_ptr1 = null,
            .extra_ptr2 = null,
        };
    }

    /// Check if this layout represents an atom type
    pub fn isAtom(self: CValueLayout) bool {
        const vtype: reader.ValueType = @enumFromInt(self.type_tag);
        return switch (vtype) {
            .identifier, .number, .string, .value_id, .block_id,
            .symbol, .keyword, .true_lit, .false_lit, .type => true,
            else => false,
        };
    }

    /// Check if this layout represents a collection type
    pub fn isCollection(self: CValueLayout) bool {
        const vtype: reader.ValueType = @enumFromInt(self.type_tag);
        return switch (vtype) {
            .list, .vector, .map => true,
            else => false,
        };
    }
};

// Compile-time assertions to verify layout
comptime {
    // Verify total size is 56 bytes
    if (@sizeOf(CValueLayout) != 56) {
        @compileError("CValueLayout must be exactly 56 bytes");
    }

    // Verify alignment is 8 bytes
    if (@alignOf(CValueLayout) != 8) {
        @compileError("CValueLayout must be 8-byte aligned");
    }

    // Verify field offsets
    if (@offsetOf(CValueLayout, "type_tag") != 0) {
        @compileError("type_tag must be at offset 0");
    }
    if (@offsetOf(CValueLayout, "data_ptr") != 8) {
        @compileError("data_ptr must be at offset 8");
    }
    if (@offsetOf(CValueLayout, "data_len") != 16) {
        @compileError("data_len must be at offset 16");
    }
    if (@offsetOf(CValueLayout, "data_capacity") != 24) {
        @compileError("data_capacity must be at offset 24");
    }
    if (@offsetOf(CValueLayout, "data_elem_size") != 32) {
        @compileError("data_elem_size must be at offset 32");
    }
    if (@offsetOf(CValueLayout, "extra_ptr1") != 40) {
        @compileError("extra_ptr1 must be at offset 40");
    }
    if (@offsetOf(CValueLayout, "extra_ptr2") != 48) {
        @compileError("extra_ptr2 must be at offset 48");
    }
}

/// Convert a Value to CValueLayout
/// Note: Creates a VIEW of the data, not a copy (except for complex types)
/// The returned layout is only valid as long as the original value exists
pub fn valueToCLayout(_: std.mem.Allocator, value: *const reader.Value) !CValueLayout {
    var layout = CValueLayout.empty(value.type);

    switch (value.type) {
        // Atom types - just copy the string slice
        .identifier, .number, .string, .value_id, .block_id,
        .symbol, .keyword, .type => {
            const atom = value.data.atom;
            layout.data_ptr = @constCast(atom.ptr);
            layout.data_len = atom.len;
        },

        // Boolean literals - no data needed, tag is sufficient
        .true_lit, .false_lit => {
            // No data fields needed
        },

        // Collection types - convert vector to flat representation
        .list, .vector, .map => {
            const vec = switch (value.type) {
                .list => value.data.list,
                .vector => value.data.vector,
                .map => value.data.map,
                else => unreachable,
            };

            if (vec.buf) |buf| {
                layout.data_ptr = @ptrCast(@constCast(buf.ptr));
                layout.data_len = buf.len;
                layout.data_capacity = buf.len; // PersistentVector doesn't track capacity separately
                layout.data_elem_size = @sizeOf(*reader.Value);
            }
        },

        // Function type - store inputs and results as pointers
        .function_type => {
            // Allocate and store inputs vector layout
            const inputs_buf = value.data.function_type.inputs.buf;
            if (inputs_buf) |buf| {
                layout.extra_ptr1 = @ptrCast(@constCast(buf.ptr));
            }

            // Allocate and store results vector layout
            const results_buf = value.data.function_type.results.buf;
            if (results_buf) |buf| {
                layout.extra_ptr2 = @ptrCast(@constCast(buf.ptr));
            }

            // Store counts in data_len and data_capacity
            layout.data_len = if (inputs_buf) |buf| buf.len else 0;
            layout.data_capacity = if (results_buf) |buf| buf.len else 0;
            layout.data_elem_size = @sizeOf(*reader.Value);
        },

        // Attribute expression - store wrapped value pointer
        .attr_expr => {
            layout.extra_ptr1 = @ptrCast(value.data.attr_expr);
        },

        // Typed literal - store value and type pointers
        .has_type => {
            layout.extra_ptr1 = @ptrCast(value.data.has_type.value);
            layout.extra_ptr2 = @ptrCast(value.data.has_type.type_expr);
        },
    }

    return layout;
}

/// Convert CValueLayout back to Value
/// This creates a NEW value that may own copies of the data
pub fn cLayoutToValue(allocator: std.mem.Allocator, layout: CValueLayout) !*reader.Value {
    const value_ptr = try allocator.create(reader.Value);
    errdefer allocator.destroy(value_ptr);

    const value_type: reader.ValueType = @enumFromInt(layout.type_tag);
    value_ptr.type = value_type;

    switch (value_type) {
        // Atom types - copy the string
        .identifier, .number, .string, .value_id, .block_id,
        .symbol, .keyword, .type => {
            if (layout.data_ptr) |ptr| {
                const slice = ptr[0..layout.data_len];
                value_ptr.data = .{ .atom = slice };
            } else {
                value_ptr.data = .{ .atom = "" };
            }
        },

        // Boolean literals
        .true_lit, .false_lit => {
            value_ptr.data = .{ .atom = "" }; // Booleans don't need data
        },

        // Collection types
        .list, .vector, .map => {
            var vec = vector.PersistentVector(*reader.Value).init(allocator, null);

            if (layout.data_ptr) |ptr| {
                // Convert raw pointer back to slice
                const elem_ptr: [*]*reader.Value = @ptrCast(@alignCast(ptr));
                const buf = try allocator.alloc(*reader.Value, layout.data_len);
                @memcpy(buf, elem_ptr[0..layout.data_len]);
                vec.buf = buf;
            }

            value_ptr.data = switch (value_type) {
                .list => .{ .list = vec },
                .vector => .{ .vector = vec },
                .map => .{ .map = vec },
                else => unreachable,
            };
        },

        // Function type
        .function_type => {
            var inputs = vector.PersistentVector(*reader.Value).init(allocator, null);
            var results = vector.PersistentVector(*reader.Value).init(allocator, null);

            if (layout.extra_ptr1) |ptr| {
                const elem_ptr: [*]*reader.Value = @ptrCast(@alignCast(ptr));
                const buf = try allocator.alloc(*reader.Value, layout.data_len);
                @memcpy(buf, elem_ptr[0..layout.data_len]);
                inputs.buf = buf;
            }

            if (layout.extra_ptr2) |ptr| {
                const elem_ptr: [*]*reader.Value = @ptrCast(@alignCast(ptr));
                const buf = try allocator.alloc(*reader.Value, layout.data_capacity);
                @memcpy(buf, elem_ptr[0..layout.data_capacity]);
                results.buf = buf;
            }

            value_ptr.data = .{ .function_type = .{
                .inputs = inputs,
                .results = results,
            }};
        },

        // Attribute expression
        .attr_expr => {
            if (layout.extra_ptr1) |ptr| {
                const wrapped: *reader.Value = @ptrCast(@alignCast(ptr));
                value_ptr.data = .{ .attr_expr = wrapped };
            } else {
                return error.InvalidLayout;
            }
        },

        // Typed literal
        .has_type => {
            if (layout.extra_ptr1 == null or layout.extra_ptr2 == null) {
                return error.InvalidLayout;
            }

            const val: *reader.Value = @ptrCast(@alignCast(layout.extra_ptr1.?));
            const type_expr: *reader.Value = @ptrCast(@alignCast(layout.extra_ptr2.?));

            value_ptr.data = .{ .has_type = .{
                .value = val,
                .type_expr = type_expr,
            }};
        },
    }

    return value_ptr;
}

// ============================================================================
// Tests
// ============================================================================

test "CValueLayout - size and alignment" {
    try std.testing.expectEqual(@as(usize, 56), @sizeOf(CValueLayout));
    try std.testing.expectEqual(@as(usize, 8), @alignOf(CValueLayout));
}

test "CValueLayout - field offsets" {
    try std.testing.expectEqual(@as(usize, 0), @offsetOf(CValueLayout, "type_tag"));
    try std.testing.expectEqual(@as(usize, 8), @offsetOf(CValueLayout, "data_ptr"));
    try std.testing.expectEqual(@as(usize, 16), @offsetOf(CValueLayout, "data_len"));
    try std.testing.expectEqual(@as(usize, 24), @offsetOf(CValueLayout, "data_capacity"));
    try std.testing.expectEqual(@as(usize, 32), @offsetOf(CValueLayout, "data_elem_size"));
    try std.testing.expectEqual(@as(usize, 40), @offsetOf(CValueLayout, "extra_ptr1"));
    try std.testing.expectEqual(@as(usize, 48), @offsetOf(CValueLayout, "extra_ptr2"));
}

test "CValueLayout - empty layout" {
    const layout = CValueLayout.empty(.identifier);
    try std.testing.expectEqual(@as(u8, @intFromEnum(reader.ValueType.identifier)), layout.type_tag);
    try std.testing.expectEqual(@as(?[*]u8, null), layout.data_ptr);
    try std.testing.expectEqual(@as(usize, 0), layout.data_len);
}

test "valueToCLayout - identifier" {
    const allocator = std.testing.allocator;

    var value = reader.Value{
        .type = .identifier,
        .data = .{ .atom = "test_name" },
    };

    const layout = try valueToCLayout(allocator, &value);

    try std.testing.expectEqual(@as(u8, @intFromEnum(reader.ValueType.identifier)), layout.type_tag);
    try std.testing.expect(layout.data_ptr != null);
    try std.testing.expectEqual(@as(usize, 9), layout.data_len);
    try std.testing.expect(layout.isAtom());
    try std.testing.expect(!layout.isCollection());
}

test "valueToCLayout - boolean" {
    const allocator = std.testing.allocator;

    var value = reader.Value{
        .type = .true_lit,
        .data = .{ .atom = "" },
    };

    const layout = try valueToCLayout(allocator, &value);

    try std.testing.expectEqual(@as(u8, @intFromEnum(reader.ValueType.true_lit)), layout.type_tag);
}

test "valueToCLayout - list" {
    const allocator = std.testing.allocator;

    // Create a simple list with one element
    const elem = try allocator.create(reader.Value);
    defer allocator.destroy(elem);
    elem.* = reader.Value{
        .type = .number,
        .data = .{ .atom = "42" },
    };

    var vec = vector.PersistentVector(*reader.Value).init(allocator, null);
    defer vec.deinit();

    var vec2 = try vec.push(elem);
    defer vec2.deinit();

    var value = reader.Value{
        .type = .list,
        .data = .{ .list = vec2 },
    };

    const layout = try valueToCLayout(allocator, &value);

    try std.testing.expectEqual(@as(u8, @intFromEnum(reader.ValueType.list)), layout.type_tag);
    try std.testing.expect(layout.data_ptr != null);
    try std.testing.expectEqual(@as(usize, 1), layout.data_len);
    try std.testing.expectEqual(@as(usize, @sizeOf(*reader.Value)), layout.data_elem_size);
    try std.testing.expect(!layout.isAtom());
    try std.testing.expect(layout.isCollection());
}
