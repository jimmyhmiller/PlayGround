const std = @import("std");
const vector = @import("vector.zig");
const map = @import("map.zig");
const reader = @import("../reader.zig");

// Opaque types for C API
pub const CValue = opaque {};
pub const CVectorValue = opaque {};
pub const CMapStrValue = opaque {};

// Helper to convert opaque allocator pointer to std.mem.Allocator
fn getAllocator(allocator: ?*anyopaque) std.mem.Allocator {
    const alloc_ptr: *std.mem.Allocator = @ptrCast(@alignCast(allocator));
    return alloc_ptr.*;
}

// ============================================================================
// Allocator Management
// ============================================================================

// Create a C allocator (malloc/free based) for use with the API
export fn allocator_create_c() ?*anyopaque {
    const alloc_ptr = std.heap.c_allocator.create(std.mem.Allocator) catch return null;
    alloc_ptr.* = std.heap.c_allocator;
    return @ptrCast(alloc_ptr);
}

// Destroy an allocator created with allocator_create_c
export fn allocator_destroy(allocator: ?*anyopaque) void {
    if (allocator == null) return;
    const alloc_ptr: *std.mem.Allocator = @ptrCast(@alignCast(allocator));
    const alloc = alloc_ptr.*;
    alloc.destroy(alloc_ptr);
}

// ============================================================================
// Vector API (Value pointers) - Primary collection type for reader
// ============================================================================

export fn vector_value_create(allocator: ?*anyopaque) ?*CVectorValue {
    if (allocator == null) return null;
    const alloc = getAllocator(allocator);
    const vec_ptr = alloc.create(vector.PersistentVector(*reader.Value)) catch return null;
    vec_ptr.* = vector.PersistentVector(*reader.Value).init(alloc, null);
    return @ptrCast(vec_ptr);
}

export fn vector_value_destroy(allocator: ?*anyopaque, vec: ?*CVectorValue) void {
    if (vec == null or allocator == null) return;
    const alloc = getAllocator(allocator);
    const vec_ptr: *vector.PersistentVector(*reader.Value) = @ptrCast(@alignCast(vec));
    vec_ptr.deinit();
    alloc.destroy(vec_ptr);
}

export fn vector_value_push(allocator: ?*anyopaque, vec: ?*CVectorValue, value: ?*CValue) ?*CVectorValue {
    if (vec == null or value == null or allocator == null) return null;
    const alloc = getAllocator(allocator);
    var vec_ptr: *vector.PersistentVector(*reader.Value) = @ptrCast(@alignCast(vec));
    const val_ptr: *reader.Value = @ptrCast(@alignCast(value));
    const new_vec = vec_ptr.push(val_ptr) catch return null;
    const new_vec_ptr = alloc.create(vector.PersistentVector(*reader.Value)) catch return null;
    new_vec_ptr.* = new_vec;
    return @ptrCast(new_vec_ptr);
}

export fn vector_value_pop(allocator: ?*anyopaque, vec: ?*CVectorValue) ?*CVectorValue {
    if (vec == null or allocator == null) return null;
    const alloc = getAllocator(allocator);
    var vec_ptr: *vector.PersistentVector(*reader.Value) = @ptrCast(@alignCast(vec));
    const new_vec = vec_ptr.pop() catch return null;
    const new_vec_ptr = alloc.create(vector.PersistentVector(*reader.Value)) catch return null;
    new_vec_ptr.* = new_vec;
    return @ptrCast(new_vec_ptr);
}

export fn vector_value_at(vec: ?*const CVectorValue, index: usize) ?*CValue {
    if (vec == null) return null;
    const vec_ptr: *const vector.PersistentVector(*reader.Value) = @ptrCast(@alignCast(vec));
    const val = vec_ptr.at(index);
    return @ptrCast(val);
}

export fn vector_value_len(vec: ?*const CVectorValue) usize {
    if (vec == null) return 0;
    const vec_ptr: *const vector.PersistentVector(*reader.Value) = @ptrCast(@alignCast(vec));
    return vec_ptr.len();
}

export fn vector_value_is_empty(vec: ?*const CVectorValue) bool {
    if (vec == null) return true;
    const vec_ptr: *const vector.PersistentVector(*reader.Value) = @ptrCast(@alignCast(vec));
    return vec_ptr.isEmpty();
}

// ============================================================================
// Map API (string -> Value) - For manipulating map data structures
// ============================================================================

export fn map_str_value_create(allocator: ?*anyopaque) ?*CMapStrValue {
    if (allocator == null) return null;
    const alloc = getAllocator(allocator);
    const map_ptr = alloc.create(map.PersistentMap([]const u8, *reader.Value)) catch return null;
    map_ptr.* = map.PersistentMap([]const u8, *reader.Value).init(alloc);
    return @ptrCast(map_ptr);
}

export fn map_str_value_destroy(allocator: ?*anyopaque, m: ?*CMapStrValue) void {
    if (m == null or allocator == null) return;
    const alloc = getAllocator(allocator);
    const map_ptr: *map.PersistentMap([]const u8, *reader.Value) = @ptrCast(@alignCast(m));
    map_ptr.deinit();
    alloc.destroy(map_ptr);
}

export fn map_str_value_get(m: ?*const CMapStrValue, key: [*:0]const u8, found: ?*bool) ?*CValue {
    if (m == null) {
        if (found != null) found.?.* = false;
        return null;
    }
    const map_ptr: *const map.PersistentMap([]const u8, *reader.Value) = @ptrCast(@alignCast(m));
    const key_slice = std.mem.span(key);
    if (map_ptr.get(key_slice)) |value| {
        if (found != null) found.?.* = true;
        return @ptrCast(value);
    }
    if (found != null) found.?.* = false;
    return null;
}

export fn map_str_value_set(allocator: ?*anyopaque, m: ?*CMapStrValue, key: [*:0]const u8, value: ?*CValue) ?*CMapStrValue {
    if (m == null or value == null or allocator == null) return null;
    const alloc = getAllocator(allocator);
    var map_ptr: *map.PersistentMap([]const u8, *reader.Value) = @ptrCast(@alignCast(m));
    const key_slice = std.mem.span(key);
    const val_ptr: *reader.Value = @ptrCast(@alignCast(value));
    const new_map = map_ptr.set(key_slice, val_ptr) catch return null;
    const new_map_ptr = alloc.create(map.PersistentMap([]const u8, *reader.Value)) catch return null;
    new_map_ptr.* = new_map;
    return @ptrCast(new_map_ptr);
}

// ============================================================================
// Value API - For creating and manipulating Value structures
// ============================================================================

// Value type enum for C
pub const CValueType = enum(c_int) {
    identifier = 0,
    number = 1,
    string = 2,
    value_id = 3,
    block_id = 4,
    symbol = 5,
    keyword = 6,
    true_lit = 7,
    false_lit = 8,
    list = 9,
    vector = 10,
    map = 11,
    type = 12,
    function_type = 13,
    attr_expr = 14,
    has_type = 15,
};

export fn value_get_type(val: ?*const CValue) CValueType {
    if (val == null) return .identifier;
    const val_ptr: *const reader.Value = @ptrCast(@alignCast(val));
    return @enumFromInt(@intFromEnum(val_ptr.type));
}

export fn value_get_atom(allocator: ?*anyopaque, val: ?*const CValue) ?[*:0]const u8 {
    if (val == null or allocator == null) return null;
    const val_ptr: *const reader.Value = @ptrCast(@alignCast(val));

    // Only atom types have atom data
    switch (val_ptr.type) {
        .identifier, .number, .string, .value_id, .block_id, .symbol, .keyword, .true_lit, .false_lit => {
            const atom = val_ptr.data.atom;
            // Note: This assumes the atom string is null-terminated or we need to make a copy
            // For safety, we should probably copy to a null-terminated buffer
            const alloc = getAllocator(allocator);
            const null_term = alloc.allocSentinel(u8, atom.len, 0) catch return null;
            @memcpy(null_term, atom);
            return null_term.ptr;
        },
        else => return null,
    }
}

export fn value_get_list(allocator: ?*anyopaque, val: ?*const CValue) ?*CVectorValue {
    if (val == null or allocator == null) return null;
    const val_ptr: *const reader.Value = @ptrCast(@alignCast(val));
    if (val_ptr.type != .list) return null;

    const alloc = getAllocator(allocator);
    const vec_ptr = alloc.create(vector.PersistentVector(*reader.Value)) catch return null;
    vec_ptr.* = val_ptr.data.list;
    return @ptrCast(vec_ptr);
}

export fn value_get_vector(allocator: ?*anyopaque, val: ?*const CValue) ?*CVectorValue {
    if (val == null or allocator == null) return null;
    const val_ptr: *const reader.Value = @ptrCast(@alignCast(val));
    if (val_ptr.type != .vector) return null;

    const alloc = getAllocator(allocator);
    const vec_ptr = alloc.create(vector.PersistentVector(*reader.Value)) catch return null;
    vec_ptr.* = val_ptr.data.vector;
    return @ptrCast(vec_ptr);
}

export fn value_get_map(allocator: ?*anyopaque, val: ?*const CValue) ?*CVectorValue {
    if (val == null or allocator == null) return null;
    const val_ptr: *const reader.Value = @ptrCast(@alignCast(val));
    if (val_ptr.type != .map) return null;

    const alloc = getAllocator(allocator);
    const vec_ptr = alloc.create(vector.PersistentVector(*reader.Value)) catch return null;
    vec_ptr.* = val_ptr.data.map;
    return @ptrCast(vec_ptr);
}

export fn value_get_type_string(val: ?*const CValue) [*:0]const u8 {
    if (val == null) return "";
    const val_ptr: *const reader.Value = @ptrCast(@alignCast(val));
    if (val_ptr.type != .type) return "";
    // type is a string, return it as a C string
    return val_ptr.data.type.ptr;
}

export fn value_get_attr_expr(val: ?*const CValue) ?*CValue {
    if (val == null) return null;
    const val_ptr: *const reader.Value = @ptrCast(@alignCast(val));
    if (val_ptr.type != .attr_expr) return null;
    return @ptrCast(val_ptr.data.attr_expr);
}

export fn value_get_has_type_value(val: ?*const CValue) ?*CValue {
    if (val == null) return null;
    const val_ptr: *const reader.Value = @ptrCast(@alignCast(val));
    if (val_ptr.type != .has_type) return null;
    return @ptrCast(val_ptr.data.has_type.value);
}

export fn value_get_has_type_type_expr(val: ?*const CValue) ?*CValue {
    if (val == null) return null;
    const val_ptr: *const reader.Value = @ptrCast(@alignCast(val));
    if (val_ptr.type != .has_type) return null;
    return @ptrCast(val_ptr.data.has_type.type_expr);
}

// Create atom values
export fn value_create_identifier(allocator: ?*anyopaque, atom: [*:0]const u8) ?*CValue {
    if (allocator == null) return null;
    const alloc = getAllocator(allocator);
    const val = alloc.create(reader.Value) catch return null;
    const atom_slice = std.mem.span(atom);
    val.* = reader.Value{
        .type = .identifier,
        .data = .{ .atom = atom_slice },
    };
    return @ptrCast(val);
}

export fn value_create_number(allocator: ?*anyopaque, atom: [*:0]const u8) ?*CValue {
    if (allocator == null) return null;
    const alloc = getAllocator(allocator);
    const val = alloc.create(reader.Value) catch return null;
    const atom_slice = std.mem.span(atom);
    val.* = reader.Value{
        .type = .number,
        .data = .{ .atom = atom_slice },
    };
    return @ptrCast(val);
}

export fn value_create_string(allocator: ?*anyopaque, atom: [*:0]const u8) ?*CValue {
    if (allocator == null) return null;
    const alloc = getAllocator(allocator);
    const val = alloc.create(reader.Value) catch return null;
    const atom_slice = std.mem.span(atom);
    val.* = reader.Value{
        .type = .string,
        .data = .{ .atom = atom_slice },
    };
    return @ptrCast(val);
}

export fn value_create_value_id(allocator: ?*anyopaque, atom: [*:0]const u8) ?*CValue {
    if (allocator == null) return null;
    const alloc = getAllocator(allocator);
    const val = alloc.create(reader.Value) catch return null;
    const atom_slice = std.mem.span(atom);
    val.* = reader.Value{
        .type = .value_id,
        .data = .{ .atom = atom_slice },
    };
    return @ptrCast(val);
}

export fn value_create_block_id(allocator: ?*anyopaque, atom: [*:0]const u8) ?*CValue {
    if (allocator == null) return null;
    const alloc = getAllocator(allocator);
    const val = alloc.create(reader.Value) catch return null;
    const atom_slice = std.mem.span(atom);
    val.* = reader.Value{
        .type = .block_id,
        .data = .{ .atom = atom_slice },
    };
    return @ptrCast(val);
}

export fn value_create_symbol(allocator: ?*anyopaque, atom: [*:0]const u8) ?*CValue {
    if (allocator == null) return null;
    const alloc = getAllocator(allocator);
    const val = alloc.create(reader.Value) catch return null;
    const atom_slice = std.mem.span(atom);
    val.* = reader.Value{
        .type = .symbol,
        .data = .{ .atom = atom_slice },
    };
    return @ptrCast(val);
}

export fn value_create_keyword(allocator: ?*anyopaque, atom: [*:0]const u8) ?*CValue {
    if (allocator == null) return null;
    const alloc = getAllocator(allocator);
    const val = alloc.create(reader.Value) catch return null;
    const atom_slice = std.mem.span(atom);
    val.* = reader.Value{
        .type = .keyword,
        .data = .{ .atom = atom_slice },
    };
    return @ptrCast(val);
}

export fn value_create_true(allocator: ?*anyopaque) ?*CValue {
    if (allocator == null) return null;
    const alloc = getAllocator(allocator);
    const val = alloc.create(reader.Value) catch return null;
    val.* = reader.Value{
        .type = .true_lit,
        .data = .{ .atom = "true" },
    };
    return @ptrCast(val);
}

export fn value_create_false(allocator: ?*anyopaque) ?*CValue {
    if (allocator == null) return null;
    const alloc = getAllocator(allocator);
    const val = alloc.create(reader.Value) catch return null;
    val.* = reader.Value{
        .type = .false_lit,
        .data = .{ .atom = "false" },
    };
    return @ptrCast(val);
}

// Create collection values
export fn value_create_list(allocator: ?*anyopaque, vec: ?*CVectorValue) ?*CValue {
    if (vec == null or allocator == null) return null;
    const alloc = getAllocator(allocator);
    const vec_ptr: *vector.PersistentVector(*reader.Value) = @ptrCast(@alignCast(vec));
    const val = alloc.create(reader.Value) catch return null;
    val.* = reader.Value{
        .type = .list,
        .data = .{ .list = vec_ptr.* },
    };
    return @ptrCast(val);
}

export fn value_create_vector(allocator: ?*anyopaque, vec: ?*CVectorValue) ?*CValue {
    if (vec == null or allocator == null) return null;
    const alloc = getAllocator(allocator);
    const vec_ptr: *vector.PersistentVector(*reader.Value) = @ptrCast(@alignCast(vec));
    const val = alloc.create(reader.Value) catch return null;
    val.* = reader.Value{
        .type = .vector,
        .data = .{ .vector = vec_ptr.* },
    };
    return @ptrCast(val);
}

export fn value_create_map(allocator: ?*anyopaque, vec: ?*CVectorValue) ?*CValue {
    if (vec == null or allocator == null) return null;
    const alloc = getAllocator(allocator);
    const vec_ptr: *vector.PersistentVector(*reader.Value) = @ptrCast(@alignCast(vec));
    const val = alloc.create(reader.Value) catch return null;
    val.* = reader.Value{
        .type = .map,
        .data = .{ .map = vec_ptr.* },
    };
    return @ptrCast(val);
}

// Create special expression values
export fn value_create_type(allocator: ?*anyopaque, type_str: [*:0]const u8) ?*CValue {
    if (type_str == null or allocator == null) return null;
    const alloc = getAllocator(allocator);
    // type takes a string directly (like "i32", "!llvm.ptr")
    const str_slice = std.mem.span(type_str);
    const val = alloc.create(reader.Value) catch return null;
    val.* = reader.Value{
        .type = .type,
        .data = .{ .type = str_slice },
    };
    return @ptrCast(val);
}

export fn value_create_attr_expr(allocator: ?*anyopaque, inner: ?*CValue) ?*CValue {
    if (inner == null or allocator == null) return null;
    const alloc = getAllocator(allocator);
    const inner_ptr: *reader.Value = @ptrCast(@alignCast(inner));
    const val = alloc.create(reader.Value) catch return null;
    val.* = reader.Value{
        .type = .attr_expr,
        .data = .{ .attr_expr = inner_ptr },
    };
    return @ptrCast(val);
}

export fn value_create_has_type(allocator: ?*anyopaque, value: ?*CValue, type_expr: ?*CValue) ?*CValue {
    if (value == null or type_expr == null or allocator == null) return null;
    const alloc = getAllocator(allocator);
    const value_ptr: *reader.Value = @ptrCast(@alignCast(value));
    const type_ptr: *reader.Value = @ptrCast(@alignCast(type_expr));
    const val = alloc.create(reader.Value) catch return null;
    val.* = reader.Value{
        .type = .has_type,
        .data = .{ .has_type = .{ .value = value_ptr, .type_expr = type_ptr } },
    };
    return @ptrCast(val);
}

export fn value_destroy(allocator: ?*anyopaque, val: ?*CValue) void {
    if (val == null or allocator == null) return;
    const alloc = getAllocator(allocator);
    var val_ptr: *reader.Value = @ptrCast(@alignCast(val));
    val_ptr.deinit(alloc);
    alloc.destroy(val_ptr);
}

// Free null-terminated strings returned by value_get_atom
export fn value_free_atom(allocator: ?*anyopaque, atom: ?[*:0]const u8) void {
    if (atom == null or allocator == null) return;
    const alloc = getAllocator(allocator);
    const len = std.mem.len(atom.?);
    alloc.free(atom.?[0..len :0]);
}
