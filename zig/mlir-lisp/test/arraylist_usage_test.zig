const std = @import("std");
const testing = std.testing;

test "ArrayList correct usage in Zig 0.15.1" {
    const allocator = testing.allocator;

    // CORRECT: Initialize ArrayList with struct literal {}
    // NO .init() method exists in Zig 0.15.1
    // Pass allocator to each method call instead
    var list = std.ArrayList(i32){};
    defer list.deinit(allocator);

    // Append items - pass allocator to each call
    try list.append(allocator, 1);
    try list.append(allocator, 2);
    try list.append(allocator, 3);

    // Access items
    try testing.expectEqual(@as(usize, 3), list.items.len);
    try testing.expectEqual(@as(i32, 1), list.items[0]);
    try testing.expectEqual(@as(i32, 2), list.items[1]);
    try testing.expectEqual(@as(i32, 3), list.items[2]);
}

test "ArrayList with owned slice" {
    const allocator = testing.allocator;

    var list = std.ArrayList(u8){};
    // Note: Don't defer deinit when converting to owned slice

    try list.appendSlice(allocator, "Hello");
    try list.append(allocator, ' ');
    try list.appendSlice(allocator, "World");

    const owned = try list.toOwnedSlice(allocator);
    defer allocator.free(owned);

    try testing.expectEqualStrings("Hello World", owned);
}

test "ArrayList with custom type" {
    const allocator = testing.allocator;

    const Point = struct {
        x: f32,
        y: f32,
    };

    var points = std.ArrayList(Point){};
    defer points.deinit(allocator);

    try points.append(allocator, .{ .x = 1.0, .y = 2.0 });
    try points.append(allocator, .{ .x = 3.0, .y = 4.0 });

    try testing.expectEqual(@as(usize, 2), points.items.len);
    try testing.expectEqual(@as(f32, 1.0), points.items[0].x);
    try testing.expectEqual(@as(f32, 4.0), points.items[1].y);
}

test "ArrayList helper function pattern" {
    const allocator = testing.allocator;

    const Helper = struct {
        fn createList() std.ArrayList(i32) {
            // Just return empty struct literal
            return std.ArrayList(i32){};
        }
    };

    var list = Helper.createList();
    defer list.deinit(allocator);

    try list.append(allocator, 42);
    try testing.expectEqual(@as(i32, 42), list.items[0]);
}

test "ArrayList ensureTotalCapacity and resize" {
    const allocator = testing.allocator;

    var list = std.ArrayList(i32){};
    defer list.deinit(allocator);

    // Pre-allocate capacity - pass allocator
    try list.ensureTotalCapacity(allocator, 10);
    try testing.expect(list.capacity >= 10);

    // Resize to specific length - pass allocator
    try list.resize(allocator, 5);
    try testing.expectEqual(@as(usize, 5), list.items.len);

    // Set values
    for (list.items, 0..) |*item, i| {
        item.* = @intCast(i * 10);
    }

    try testing.expectEqual(@as(i32, 0), list.items[0]);
    try testing.expectEqual(@as(i32, 40), list.items[4]);
}

test "ArrayList clearAndFree vs clearRetainingCapacity" {
    const allocator = testing.allocator;

    var list = std.ArrayList(i32){};
    defer list.deinit(allocator);

    // Add some items
    try list.append(allocator, 1);
    try list.append(allocator, 2);
    try list.append(allocator, 3);

    // Clear but keep capacity
    list.clearRetainingCapacity();
    try testing.expectEqual(@as(usize, 0), list.items.len);
    try testing.expect(list.capacity > 0);

    // Can reuse without reallocation
    try list.append(allocator, 10);
    try testing.expectEqual(@as(i32, 10), list.items[0]);

    // Clear and free memory
    list.clearAndFree(allocator);
    try testing.expectEqual(@as(usize, 0), list.items.len);
    try testing.expectEqual(@as(usize, 0), list.capacity);
}
