const std = @import("std");

pub fn PersistentVector(comptime T: type) type {
    return struct {
        alloc: std.mem.Allocator,
        buf: ?[]T = null,

        pub fn init(alloc: std.mem.Allocator, buf: ?[]T) PersistentVector(T) {
            return PersistentVector(T){
                .alloc = alloc,
                .buf = buf,
            };
        }

        pub fn deinit(self: *PersistentVector(T)) void {
            if (self.buf != null) {
                self.alloc.free(self.buf.?);
                self.buf = null;
            }
        }

        pub fn at(self: PersistentVector(T), index: usize) T {
            return self.buf.?[index];
        }

        pub fn slice(self: PersistentVector(T)) []T {
            if (self.buf == null) {
                return &[_]T{}; // return empty slice
            }
            return self.buf.?[0..self.buf.?.len];
        }

        pub fn len(self: PersistentVector(T)) usize {
            if (self.buf == null) {
                return 0;
            }
            return self.buf.?.len;
        }

        pub fn isEmpty(self: PersistentVector(T)) bool {
            if (self.buf == null) {
                return true;
            }
            return self.buf.?.len == 0;
        }

        pub fn push(self: *PersistentVector(T), value: T) !PersistentVector(T) {
            if (self.buf == null) {
                const new_buf = try self.alloc.alloc(T, 1);
                new_buf[0] = value;
                return PersistentVector(T).init(self.alloc, new_buf);
            }
            const old_len = self.buf.?.len;
            const new_buf = try self.alloc.alloc(T, old_len + 1);
            @memcpy(new_buf[0..old_len], self.buf.?);
            new_buf[old_len] = value;
            return PersistentVector(T).init(self.alloc, new_buf);
        }

        pub fn pop(self: *PersistentVector(T)) !PersistentVector(T) {
            if (self.buf == null or self.buf.?.len == 0) {
                return PersistentVector(T).init(self.alloc, null);
            }
            const old_len = self.buf.?.len;
            if (old_len == 1) {
                return PersistentVector(T).init(self.alloc, null);
            }
            const new_buf = try self.alloc.alloc(T, old_len - 1);
            @memcpy(new_buf, self.buf.?[0 .. old_len - 1]);
            return PersistentVector(T).init(self.alloc, new_buf);
        }

        pub const Iterator = struct {
            buf: ?[]T,
            index: usize,

            pub fn next(self: *Iterator) ?T {
                if (self.buf == null or self.index >= self.buf.?.len) {
                    return null;
                }
                const value = self.buf.?[self.index];
                self.index += 1;
                return value;
            }
        };

        pub fn iterator(self: PersistentVector(T)) Iterator {
            return Iterator{ .buf = self.buf, .index = 0 };
        }
    };
}

test "vector test" {
    const gpa = std.testing.allocator;
    var vec = PersistentVector(u8).init(gpa, null);
    defer vec.deinit();
    var new_vector = try vec.push(1);
    defer new_vector.deinit();
    var empty = try new_vector.pop();
    defer empty.deinit();
    try std.testing.expect(empty.isEmpty());

    try std.testing.expect(vec.isEmpty());
    try std.testing.expect(vec.len() == 0);
    try std.testing.expect(new_vector.len() == 1);
    try std.testing.expect(new_vector.at(0) == 1);
}
