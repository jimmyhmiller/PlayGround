const std = @import("std");

pub fn PersistentLinkedList(comptime T: type) type {
    return struct {
        const Self = @This();

        tag: enum { empty, cons },
        value: ?T,
        next: ?*const Self,
        allocator: std.mem.Allocator,

        pub fn empty(allocator: std.mem.Allocator) !*Self {
            const node = try allocator.create(Self);
            node.* = Self{
                .tag = .empty,
                .value = null,
                .next = null,
                .allocator = allocator,
            };
            return node;
        }

        pub fn cons(allocator: std.mem.Allocator, value: T, next: *const Self) !*Self {
            const node = try allocator.create(Self);
            node.* = Self{
                .tag = .cons,
                .value = value,
                .next = next,
                .allocator = allocator,
            };
            return node;
        }

        pub fn deinit(self: *const Self) void {
            self.allocator.destroy(self);
        }

        pub fn isEmpty(self: *const Self) bool {
            return self.tag == .empty;
        }

        pub fn len(self: *const Self) usize {
            if (self.tag == .empty) {
                return 0;
            }
            return 1 + self.next.?.len();
        }

        pub fn push(self: *const Self, allocator: std.mem.Allocator, value: T) !*Self {
            return try Self.cons(allocator, value, self);
        }

        pub fn pop(self: *const Self) *const Self {
            if (self.tag == .empty) {
                return self;
            }
            return self.next.?;
        }

        pub const Iterator = struct {
            current: ?*const Self,

            pub fn next(self: *Iterator) ?T {
                if (self.current) |node| {
                    if (node.tag == .empty) {
                        return null;
                    }
                    const value = node.value.?;
                    self.current = node.next;
                    return value;
                }
                return null;
            }

            pub fn peek(self: *const Iterator) ?T {
                if (self.current) |node| {
                    if (node.tag == .empty) {
                        return null;
                    }
                    return node.value;
                }
                return null;
            }
        };

        pub fn iterator(self: *const Self) Iterator {
            return Iterator{ .current = self };
        }
    };
}

test "linked list test" {
    const allocator = std.testing.allocator;

    // Our list does not mutate itself
    const list = try PersistentLinkedList(u8).empty(allocator);
    defer list.deinit();
    const new_list = try list.push(allocator, 1);
    defer new_list.deinit();
    const popped = new_list.pop();
    // popped is just a reference to list, so no deinit needed
    try std.testing.expect(popped.isEmpty());
    try std.testing.expect(list.isEmpty());
    try std.testing.expect(list.len() == 0);
    try std.testing.expect(new_list.len() == 1);
    try std.testing.expect(new_list.value.? == 1);
    try std.testing.expect(new_list.next.?.isEmpty());
}
