const std = @import("std");
const vector = @import("vector.zig");

pub fn PersistentMap(comptime K: type, comptime V: type) type {
    return PersistentMapWithEq(K, V, null);
}

pub fn PersistentMapWithEq(comptime K: type, comptime V: type, comptime eqFn: ?fn (K, K) bool) type {
    return struct {
        const Entry = struct {
            key: K,
            value: V,
        };

        const Vector = vector.PersistentVector(Entry);

        alloc: std.mem.Allocator,
        vec: Vector,

        pub fn init(alloc: std.mem.Allocator) @This() {
            return .{
                .alloc = alloc,
                .vec = Vector.init(alloc, null),
            };
        }

        pub fn deinit(self: *@This()) void {
            self.vec.deinit();
        }

        pub fn get(self: @This(), key: K) ?V {
            const slice = self.vec.slice();
            for (slice) |entry| {
                if (keysEqual(entry.key, key)) {
                    return entry.value;
                }
            }
            return null;
        }

        fn keysEqual(a: K, b: K) bool {
            if (eqFn) |eq| {
                return eq(a, b);
            }

            // Comptime type checking for automatic equality selection
            const info = @typeInfo(K);
            return switch (info) {
                .pointer => |ptr| blk: {
                    // String slices
                    if (ptr.size == .slice) {
                        if (ptr.child == u8) {
                            break :blk std.mem.eql(u8, a, b);
                        }
                        // Could add other slice types here
                        // e.g., if (ptr.child == u16) break :blk std.mem.eql(u16, a, b);
                    }
                    // C strings
                    if (ptr.size == .many and ptr.sentinel != null and ptr.child == u8) {
                        break :blk std.mem.eql(u8, std.mem.span(a), std.mem.span(b));
                    }
                    // Single pointers - check if pointed-to type has eql method
                    if (ptr.size == .one) {
                        if (@hasDecl(ptr.child, "eql")) {
                            break :blk a.eql(b);
                        }
                        break :blk a == b;
                    }
                    break :blk a == b;
                },
                .array => |arr| blk: {
                    // Fixed-size arrays
                    if (arr.child == u8) {
                        break :blk std.mem.eql(u8, &a, &b);
                    }
                    break :blk std.mem.eql(arr.child, &a, &b);
                },
                .@"struct" => blk: {
                    // For structs, we could check if they have an `eql` method
                    if (@hasDecl(K, "eql")) {
                        break :blk a.eql(b);
                    }
                    // Otherwise use default equality
                    break :blk a == b;
                },
                .@"enum", .int, .float, .bool => a == b,
                .optional => blk: {
                    if (a == null and b == null) break :blk true;
                    if (a == null or b == null) break :blk false;
                    // For now, just use == for non-null optionals
                    break :blk a.? == b.?;
                },
                else => a == b,
            };
        }

        pub fn set(self: *@This(), key: K, value: V) !@This() {
            const slice = self.vec.slice();

            // Check if key exists and create new vector with updated value
            for (slice, 0..) |entry, i| {
                if (keysEqual(entry.key, key)) {
                    // Key exists, create new vector with updated value
                    const new_buf = try self.alloc.alloc(Entry, slice.len);
                    @memcpy(new_buf, slice);
                    new_buf[i] = .{ .key = key, .value = value };
                    return .{
                        .alloc = self.alloc,
                        .vec = Vector.init(self.alloc, new_buf),
                    };
                }
            }

            // Key doesn't exist, append new entry
            const new_vec = try self.vec.push(.{ .key = key, .value = value });
            return .{
                .alloc = self.alloc,
                .vec = new_vec,
            };
        }

        pub const Iterator = struct {
            vec_iter: Vector.Iterator,

            pub fn next(self: *Iterator) ?struct { key: K, value: V } {
                if (self.vec_iter.next()) |entry| {
                    return .{ .key = entry.key, .value = entry.value };
                }
                return null;
            }
        };

        pub fn iterator(self: @This()) Iterator {
            return Iterator{ .vec_iter = self.vec.iterator() };
        }
    };
}

test "map test" {
    const gpa = std.testing.allocator;
    var m = PersistentMap([]const u8, u8).init(gpa);
    defer m.deinit();
    try std.testing.expect(m.get("a") == null);
    var m2 = try m.set("a", 1);
    defer m2.deinit();
    try std.testing.expect(m.get("a") == null);
    try std.testing.expect(m2.get("a") == 1);
    var m3 = try m2.set("b", 2);
    defer m3.deinit();
    try std.testing.expect(m2.get("b") == null);
    try std.testing.expect(m3.get("a") == 1);
    try std.testing.expect(m3.get("b") == 2);
}

test "map with custom equality" {
    const allocator = std.testing.allocator;

    // Custom equality function for case-insensitive string comparison
    const caseInsensitiveEq = struct {
        fn eq(a: []const u8, b: []const u8) bool {
            if (a.len != b.len) return false;
            for (a, b) |char_a, char_b| {
                if (std.ascii.toLower(char_a) != std.ascii.toLower(char_b)) {
                    return false;
                }
            }
            return true;
        }
    }.eq;

    var m = PersistentMapWithEq([]const u8, u8, caseInsensitiveEq).init(allocator);
    defer m.deinit();

    var m2 = try m.set("Hello", 1);
    defer m2.deinit();

    // These should find the same value due to case-insensitive comparison
    try std.testing.expect(m2.get("hello").? == 1);
    try std.testing.expect(m2.get("HELLO").? == 1);
    try std.testing.expect(m2.get("HeLLo").? == 1);
}

test "map with automatic type-based equality" {
    const allocator = std.testing.allocator;

    // Test with strings - automatic string equality
    {
        var m = PersistentMap([]const u8, i32).init(allocator);
        defer m.deinit();

        var m2 = try m.set("hello", 100);
        defer m2.deinit();
        var m3 = try m2.set("world", 200);
        defer m3.deinit();
        var m4 = try m3.set("hello", 150); // Updates existing key
        defer m4.deinit();

        // String comparison works automatically
        try std.testing.expect(m4.get("hello").? == 150);
        try std.testing.expect(m4.get("world").? == 200);
        try std.testing.expect(m4.get("missing") == null);

        // Works even with different string literal instances
        const another_hello = "hello";
        try std.testing.expect(m4.get(another_hello).? == 150);
    }

    // Test with arrays (automatic array equality)
    {
        var m = PersistentMap([3]u8, u32).init(allocator);
        defer m.deinit();

        const key1 = [3]u8{ 'a', 'b', 'c' };
        const key2 = [3]u8{ 'a', 'b', 'c' };

        var m2 = try m.set(key1, 42);
        defer m2.deinit();

        // Should find the value with equivalent array
        try std.testing.expect(m2.get(key2).? == 42);
    }

    // Test with structs that have eql method
    {
        const Point = struct {
            x: i32,
            y: i32,

            pub fn eql(self: @This(), other: @This()) bool {
                return self.x == other.x and self.y == other.y;
            }
        };

        var m = PersistentMap(Point, []const u8).init(allocator);
        defer m.deinit();

        var m2 = try m.set(.{ .x = 10, .y = 20 }, "point1");
        defer m2.deinit();

        try std.testing.expect(std.mem.eql(u8, m2.get(.{ .x = 10, .y = 20 }).?, "point1"));
    }

    // Test with optionals
    {
        var m = PersistentMap(?u32, []const u8).init(allocator);
        defer m.deinit();

        var m2 = try m.set(null, "nil");
        defer m2.deinit();
        var m3 = try m2.set(42, "forty-two");
        defer m3.deinit();

        try std.testing.expect(std.mem.eql(u8, m3.get(null).?, "nil"));
        try std.testing.expect(std.mem.eql(u8, m3.get(42).?, "forty-two"));
    }
}
