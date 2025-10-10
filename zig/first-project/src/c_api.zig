const std = @import("std");
const value = @import("value.zig");
const reader = @import("reader.zig");

pub const Value = value.Value;
pub const Reader = reader.Reader;
const PersistentLinkedList = @import("collections/linked_list.zig").PersistentLinkedList;
const PersistentVector = @import("collections/vector.zig").PersistentVector;
const PersistentMap = @import("collections/map.zig").PersistentMap;

// Global allocator for C API (thread-local)
threadlocal var gpa: std.heap.GeneralPurposeAllocator(.{}) = undefined;
threadlocal var allocator_initialized: bool = false;

fn ensureAllocator() std.mem.Allocator {
    if (!allocator_initialized) {
        gpa = std.heap.GeneralPurposeAllocator(.{}){};
        allocator_initialized = true;
    }
    return gpa.allocator();
}

// ============================================================================
// Reader API
// ============================================================================

pub export fn reader_create() ?*Reader {
    const alloc = ensureAllocator();
    // Store the allocator in a static location so the pointer is stable
    const static_alloc = struct {
        var a: std.mem.Allocator = undefined;
        var initialized: bool = false;
    };
    if (!static_alloc.initialized) {
        static_alloc.a = alloc;
        static_alloc.initialized = true;
    }
    var mutable_alloc = static_alloc.a;
    const r = mutable_alloc.create(Reader) catch return null;
    r.* = Reader.init(&static_alloc.a);
    return r;
}

pub export fn reader_destroy(r: ?*Reader) void {
    if (r) |reader_ptr| {
        var alloc = ensureAllocator();
        alloc.destroy(reader_ptr);
    }
}

pub export fn reader_read_string(r: ?*Reader, source: [*:0]const u8) ?*Value {
    const reader_ptr = r orelse return null;
    const src = std.mem.span(source);
    return reader_ptr.readString(src) catch null;
}

// ============================================================================
// Value Type Checking API
// ============================================================================

pub export fn value_is_symbol(v: ?*const Value) bool {
    return if (v) |val| val.isSymbol() else false;
}

pub export fn value_is_keyword(v: ?*const Value) bool {
    return if (v) |val| val.isKeyword() else false;
}

pub export fn value_is_string(v: ?*const Value) bool {
    return if (v) |val| val.isString() else false;
}

pub export fn value_is_int(v: ?*const Value) bool {
    return if (v) |val| val.isInt() else false;
}

pub export fn value_is_float(v: ?*const Value) bool {
    return if (v) |val| val.isFloat() else false;
}

pub export fn value_is_list(v: ?*const Value) bool {
    return if (v) |val| val.isList() else false;
}

pub export fn value_is_vector(v: ?*const Value) bool {
    return if (v) |val| val.isVector() else false;
}

pub export fn value_is_map(v: ?*const Value) bool {
    return if (v) |val| val.isMap() else false;
}

pub export fn value_is_nil(v: ?*const Value) bool {
    return if (v) |val| val.isNil() else false;
}

// ============================================================================
// Value Accessors API
// ============================================================================

pub export fn value_get_symbol(v: ?*const Value) ?[*:0]const u8 {
    if (v) |val| {
        if (val.isSymbol()) {
            // Note: assuming symbols are null-terminated or we need to create a null-terminated copy
            return @ptrCast(val.symbol.ptr);
        }
    }
    return null;
}

pub export fn value_get_keyword(v: ?*const Value) ?[*:0]const u8 {
    if (v) |val| {
        if (val.isKeyword()) {
            return @ptrCast(val.keyword.ptr);
        }
    }
    return null;
}

pub export fn value_get_string(v: ?*const Value) ?[*:0]const u8 {
    if (v) |val| {
        if (val.isString()) {
            return @ptrCast(val.string.ptr);
        }
    }
    return null;
}

pub export fn value_get_int(v: ?*const Value) i64 {
    if (v) |val| {
        if (val.isInt()) {
            return val.int;
        }
    }
    return 0;
}

pub export fn value_get_float(v: ?*const Value) f64 {
    if (v) |val| {
        if (val.isFloat()) {
            return val.float;
        }
    }
    return 0.0;
}

// ============================================================================
// List API
// ============================================================================

pub export fn list_is_empty(v: ?*const Value) bool {
    if (v) |val| {
        if (val.isList()) {
            return val.list.isEmpty();
        }
    }
    return true;
}

pub export fn list_length(v: ?*const Value) usize {
    if (v) |val| {
        if (val.isList()) {
            return val.list.len();
        }
    }
    return 0;
}

pub export fn list_first(v: ?*const Value) ?*Value {
    if (v) |val| {
        if (val.isList()) {
            if (!val.list.isEmpty()) {
                return val.list.value;
            }
        }
    }
    return null;
}

pub export fn list_rest(v: ?*const Value) ?*Value {
    if (v) |val| {
        if (val.isList()) {
            if (!val.list.isEmpty()) {
                var alloc = ensureAllocator();
                const rest_val = alloc.create(Value) catch return null;
                // Cast away const - the list is persistent/immutable anyway
                rest_val.* = Value{ .list = @constCast(val.list.next.?) };
                return rest_val;
            }
        }
    }
    return null;
}

// Get nth element of list (0-indexed)
pub export fn list_nth(v: ?*const Value, n: usize) ?*Value {
    if (v) |val| {
        if (val.isList()) {
            var current: ?*const PersistentLinkedList(*Value) = val.list;
            var i: usize = 0;
            while (current != null and !current.?.isEmpty()) {
                if (i == n) {
                    return current.?.value;
                }
                current = current.?.next;
                i += 1;
            }
        }
    }
    return null;
}

// Create empty list
pub export fn list_empty() ?*Value {
    const alloc = ensureAllocator();
    return value.createList(alloc) catch null;
}

// Cons a value onto a list
pub export fn list_cons(head: ?*Value, tail: ?*const Value) ?*Value {
    const head_val = head orelse return null;
    const tail_val = tail orelse return null;

    if (!tail_val.isList()) return null;

    const alloc = ensureAllocator();
    const new_list = PersistentLinkedList(*Value).cons(alloc, head_val, tail_val.list) catch return null;
    const new_val = alloc.create(Value) catch return null;
    new_val.* = Value{ .list = new_list };
    return new_val;
}

// ============================================================================
// Vector API
// ============================================================================

pub export fn vector_length(v: ?*const Value) usize {
    if (v) |val| {
        if (val.isVector()) {
            return val.vector.len();
        }
    }
    return 0;
}

pub export fn vector_is_empty(v: ?*const Value) bool {
    if (v) |val| {
        if (val.isVector()) {
            return val.vector.isEmpty();
        }
    }
    return true;
}

pub export fn vector_nth(v: ?*const Value, n: usize) ?*Value {
    if (v) |val| {
        if (val.isVector()) {
            if (n < val.vector.len()) {
                return val.vector.at(n);
            }
        }
    }
    return null;
}

// Create empty vector
pub export fn vector_empty() ?*Value {
    const alloc = ensureAllocator();
    return value.createVector(alloc) catch null;
}

// Push a value onto a vector (returns new vector)
pub export fn vector_push(vec: ?*const Value, val: ?*Value) ?*Value {
    const vec_val = vec orelse return null;
    const push_val = val orelse return null;

    if (!vec_val.isVector()) return null;

    const alloc = ensureAllocator();
    var mutable_vec = vec_val.vector;
    const new_vec = mutable_vec.push(push_val) catch return null;
    const new_val = alloc.create(Value) catch return null;
    new_val.* = Value{ .vector = new_vec };
    return new_val;
}

// Pop a value from a vector (returns new vector)
pub export fn vector_pop(vec: ?*const Value) ?*Value {
    const vec_val = vec orelse return null;

    if (!vec_val.isVector()) return null;

    const alloc = ensureAllocator();
    var mutable_vec = vec_val.vector;
    const new_vec = mutable_vec.pop() catch return null;
    const new_val = alloc.create(Value) catch return null;
    new_val.* = Value{ .vector = new_vec };
    return new_val;
}

// ============================================================================
// Map API
// ============================================================================

pub export fn map_get(m: ?*const Value, key: ?*Value) ?*Value {
    const map_val = m orelse return null;
    const key_val = key orelse return null;

    if (!map_val.isMap()) return null;

    return map_val.map.get(key_val);
}

pub export fn map_set(m: ?*const Value, key: ?*Value, val: ?*Value) ?*Value {
    const map_val = m orelse return null;
    const key_val = key orelse return null;
    const set_val = val orelse return null;

    if (!map_val.isMap()) return null;

    const alloc = ensureAllocator();
    var mutable_map = map_val.map;
    const new_map = mutable_map.set(key_val, set_val) catch return null;
    const new_val = alloc.create(Value) catch return null;
    new_val.* = Value{ .map = new_map };
    return new_val;
}

// Create empty map
pub export fn map_empty() ?*Value {
    const alloc = ensureAllocator();
    return value.createMap(alloc) catch null;
}

// Get number of entries in map
pub export fn map_count(m: ?*const Value) usize {
    if (m) |map_val| {
        if (map_val.isMap()) {
            return map_val.map.vec.len();
        }
    }
    return 0;
}

// Iterator support for maps
pub const MapIterator = struct {
    map: PersistentMap(*Value, *Value),
    index: usize,
};

pub export fn map_iterator_create(m: ?*const Value) ?*MapIterator {
    const map_val = m orelse return null;
    if (!map_val.isMap()) return null;

    const alloc = ensureAllocator();
    const iter = alloc.create(MapIterator) catch return null;
    iter.* = MapIterator{
        .map = map_val.map,
        .index = 0,
    };
    return iter;
}

pub export fn map_iterator_next(iter: ?*MapIterator, out_key: *?*Value, out_value: *?*Value) bool {
    const it = iter orelse return false;

    const slice = it.map.vec.slice();
    if (it.index >= slice.len) {
        return false;
    }

    const entry = slice[it.index];
    out_key.* = entry.key;
    out_value.* = entry.value;
    it.index += 1;
    return true;
}

pub export fn map_iterator_destroy(iter: ?*MapIterator) void {
    if (iter) |it| {
        var alloc = ensureAllocator();
        alloc.destroy(it);
    }
}

// ============================================================================
// Value Creation API
// ============================================================================

pub export fn value_create_int(i: i64) ?*Value {
    const alloc = ensureAllocator();
    return value.createInt(alloc, i) catch null;
}

pub export fn value_create_float(f: f64) ?*Value {
    const alloc = ensureAllocator();
    return value.createFloat(alloc, f) catch null;
}

pub export fn value_create_string(s: [*:0]const u8) ?*Value {
    const alloc = ensureAllocator();
    const str = std.mem.span(s);
    return value.createString(alloc, str) catch null;
}

pub export fn value_create_symbol(s: [*:0]const u8) ?*Value {
    const alloc = ensureAllocator();
    const str = std.mem.span(s);
    return value.createSymbol(alloc, str) catch null;
}

pub export fn value_create_keyword(s: [*:0]const u8) ?*Value {
    const alloc = ensureAllocator();
    const str = std.mem.span(s);
    return value.createKeyword(alloc, str) catch null;
}

pub export fn value_create_nil() ?*Value {
    const alloc = ensureAllocator();
    return value.createNil(alloc) catch null;
}

// ============================================================================
// Utility API
// ============================================================================

pub export fn value_to_string(v: ?*const Value, buf: [*]u8, buf_size: usize) i32 {
    const val = v orelse return -1;

    var buffer: [2048]u8 = undefined;
    var stream = std.io.fixedBufferStream(&buffer);
    val.format("", .{}, stream.writer()) catch return -1;

    const len = @min(stream.pos, buf_size - 1);
    @memcpy(buf[0..len], buffer[0..len]);
    buf[len] = 0; // null terminate
    return @intCast(len);
}
