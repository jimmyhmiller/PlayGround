const std = @import("std");
const vector = @import("vector.zig");
const map = @import("map.zig");

/// C-compatible layout for PersistentVector
/// This struct has a stable ABI and can be directly accessed from MLIR code
/// Memory layout: [data_ptr, len, capacity, elem_size]
pub const CVectorLayout = extern struct {
    /// Pointer to the element array (null if empty)
    data: ?[*]u8,
    /// Number of elements in the vector
    len: usize,
    /// Allocated capacity (in elements)
    capacity: usize,
    /// Size of each element in bytes
    elem_size: usize,

    /// Create an empty vector layout
    pub fn empty(comptime T: type) CVectorLayout {
        return .{
            .data = null,
            .len = 0,
            .capacity = 0,
            .elem_size = @sizeOf(T),
        };
    }

    /// Get element at index (unsafe - no bounds checking)
    pub fn getUnchecked(self: CVectorLayout, comptime T: type, index: usize) T {
        const ptr: [*]T = @ptrCast(@alignCast(self.data.?));
        return ptr[index];
    }

    /// Set element at index (unsafe - no bounds checking, breaks immutability)
    pub fn setUnchecked(self: *CVectorLayout, comptime T: type, index: usize, value: T) void {
        const ptr: [*]T = @ptrCast(@alignCast(self.data.?));
        ptr[index] = value;
    }
};

/// C-compatible layout for PersistentMap (backed by vector of entries)
/// Memory layout: [entries_data_ptr, len, capacity, entry_size]
pub const CMapLayout = extern struct {
    /// Pointer to the entries array (null if empty)
    entries: ?[*]u8,
    /// Number of key-value pairs
    len: usize,
    /// Allocated capacity (in entries)
    capacity: usize,
    /// Size of each entry in bytes
    entry_size: usize,

    /// Create an empty map layout
    pub fn empty(comptime K: type, comptime V: type) CMapLayout {
        const Entry = struct { key: K, value: V };
        return .{
            .entries = null,
            .len = 0,
            .capacity = 0,
            .entry_size = @sizeOf(Entry),
        };
    }
};

/// Convert PersistentVector to C-compatible layout
/// Note: This creates a VIEW of the vector's data, not a copy
/// The returned layout is only valid as long as the original vector exists
pub fn vectorToCLayout(comptime T: type, vec: vector.PersistentVector(T)) CVectorLayout {
    if (vec.buf) |buf| {
        return .{
            .data = @ptrCast(@constCast(buf.ptr)),
            .len = buf.len,
            .capacity = buf.len, // PersistentVector doesn't track capacity separately
            .elem_size = @sizeOf(T),
        };
    } else {
        return CVectorLayout.empty(T);
    }
}

/// Convert PersistentVector to a HEAP-ALLOCATED C layout
/// The caller owns the returned pointer and must free it using destroyCVectorLayout
pub fn vectorToCLayoutAlloc(comptime T: type, allocator: std.mem.Allocator, vec: vector.PersistentVector(T)) !*CVectorLayout {
    const layout = try allocator.create(CVectorLayout);
    layout.* = vectorToCLayout(T, vec);
    return layout;
}

/// Free a heap-allocated CVectorLayout (does NOT free the data itself)
pub fn destroyCVectorLayout(allocator: std.mem.Allocator, layout: *CVectorLayout) void {
    allocator.destroy(layout);
}

/// Convert C layout back to PersistentVector
/// Note: This creates a NEW vector that OWNS a COPY of the data
pub fn cLayoutToVector(comptime T: type, allocator: std.mem.Allocator, layout: CVectorLayout) !vector.PersistentVector(T) {
    if (layout.data == null or layout.len == 0) {
        return vector.PersistentVector(T).init(allocator, null);
    }

    // Allocate new buffer and copy data
    const buf = try allocator.alloc(T, layout.len);
    const src_ptr: [*]const T = @ptrCast(@alignCast(layout.data.?));
    @memcpy(buf, src_ptr[0..layout.len]);

    return vector.PersistentVector(T).init(allocator, buf);
}

/// Convert PersistentMap to C-compatible layout
pub fn mapToCLayout(comptime K: type, comptime V: type, m: map.PersistentMap(K, V)) CMapLayout {
    const Entry = struct { key: K, value: V };
    const vec_slice = m.vec.slice();

    if (vec_slice.len > 0) {
        return .{
            .entries = @ptrCast(@constCast(vec_slice.ptr)),
            .len = vec_slice.len,
            .capacity = vec_slice.len,
            .entry_size = @sizeOf(Entry),
        };
    } else {
        return CMapLayout.empty(K, V);
    }
}

// ============================================================================
// Tests
// ============================================================================

test "CVectorLayout - empty vector" {
    const empty = CVectorLayout.empty(i32);
    try std.testing.expect(empty.data == null);
    try std.testing.expectEqual(@as(usize, 0), empty.len);
    try std.testing.expectEqual(@as(usize, 4), empty.elem_size);
}

test "vectorToCLayout - view of PersistentVector" {
    const allocator = std.testing.allocator;

    var vec = vector.PersistentVector(i32).init(allocator, null);
    defer vec.deinit();

    var vec2 = try vec.push(42);
    defer vec2.deinit();

    var vec3 = try vec2.push(100);
    defer vec3.deinit();

    const layout = vectorToCLayout(i32, vec3);

    try std.testing.expectEqual(@as(usize, 2), layout.len);
    try std.testing.expectEqual(@as(usize, 4), layout.elem_size);
    try std.testing.expect(layout.data != null);

    // Access elements via unsafe getUnchecked
    try std.testing.expectEqual(@as(i32, 42), layout.getUnchecked(i32, 0));
    try std.testing.expectEqual(@as(i32, 100), layout.getUnchecked(i32, 1));
}

test "cLayoutToVector - creates owned copy" {
    const allocator = std.testing.allocator;

    var vec = vector.PersistentVector(i32).init(allocator, null);
    defer vec.deinit();

    var vec2 = try vec.push(10);
    defer vec2.deinit();

    var vec3 = try vec2.push(20);
    defer vec3.deinit();

    const layout = vectorToCLayout(i32, vec3);

    // Convert back to vector (creates copy)
    var new_vec = try cLayoutToVector(i32, allocator, layout);
    defer new_vec.deinit();

    // Verify the copy
    try std.testing.expectEqual(@as(usize, 2), new_vec.len());
    try std.testing.expectEqual(@as(i32, 10), new_vec.at(0));
    try std.testing.expectEqual(@as(i32, 20), new_vec.at(1));

    // Original vec3 is still intact
    try std.testing.expectEqual(@as(usize, 2), vec3.len());
}

test "vectorToCLayoutAlloc - heap allocation" {
    const allocator = std.testing.allocator;

    var vec = vector.PersistentVector(u8).init(allocator, null);
    defer vec.deinit();

    var vec2 = try vec.push(255);
    defer vec2.deinit();

    // Allocate on heap
    const layout_ptr = try vectorToCLayoutAlloc(u8, allocator, vec2);
    defer destroyCVectorLayout(allocator, layout_ptr);

    try std.testing.expectEqual(@as(usize, 1), layout_ptr.len);
    try std.testing.expectEqual(@as(u8, 255), layout_ptr.getUnchecked(u8, 0));
}

test "mapToCLayout" {
    const allocator = std.testing.allocator;

    var m = map.PersistentMap([]const u8, i32).init(allocator);
    defer m.deinit();

    var m2 = try m.set("foo", 42);
    defer m2.deinit();

    var m3 = try m2.set("bar", 100);
    defer m3.deinit();

    const layout = mapToCLayout([]const u8, i32, m3);

    try std.testing.expectEqual(@as(usize, 2), layout.len);
    try std.testing.expect(layout.entries != null);
}
