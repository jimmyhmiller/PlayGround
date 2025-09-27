const std = @import("std");
const vector = @import("collections/vector.zig");
const linked_list = @import("collections/linked_list.zig");
const collections_map = @import("collections/map.zig");

const PersistentVector = vector.PersistentVector;
const PersistentLinkedList = linked_list.PersistentLinkedList;
const PersistentMap = collections_map.PersistentMap;

pub const NamespaceDecl = struct {
    name: []const u8,
};

pub const Value = union(enum) {
    symbol: []const u8,
    keyword: []const u8, // keywords start with `:` but we store without the `:`
    string: []const u8,
    int: i64,
    float: f64,
    list: *PersistentLinkedList(*Value),
    vector: PersistentVector(*Value),
    map: PersistentMap(*Value, *Value),
    namespace: NamespaceDecl,
    nil,

    pub fn format(self: Value, comptime fmt: []const u8, options: std.fmt.FormatOptions, writer: anytype) !void {
        _ = fmt;
        _ = options;

        switch (self) {
            .symbol => |s| try writer.print("{s}", .{s}),
            .keyword => |k| try writer.print(":{s}", .{k}),
            .string => |s| try writer.print("\"{s}\"", .{s}),
            .int => |i| try writer.print("{d}", .{i}),
            .float => |f| try writer.print("{d:.2}", .{f}),
            .list => |l| {
                try writer.print("(", .{});
                // For now, just print a simple representation
                var current: ?*const @TypeOf(l.*) = l;
                var first = true;
                while (current != null and !current.?.isEmpty()) {
                    if (!first) try writer.print(" ", .{});
                    try current.?.value.?.format("", .{}, writer);
                    current = current.?.next;
                    first = false;
                }
                try writer.print(")", .{});
            },
            .vector => |v| {
                try writer.print("[", .{});
                const slice = v.slice();
                for (slice, 0..) |val, i| {
                    if (i > 0) try writer.print(" ", .{});
                    try val.format("", .{}, writer);
                }
                try writer.print("]", .{});
            },
            .map => |m| {
                try writer.print("{{", .{});
                // For now, just indicate it's a map - proper iteration would need more work
                const slice = m.vec.slice();
                for (slice, 0..) |entry, i| {
                    if (i > 0) try writer.print(" ", .{});
                    try entry.key.format("", .{}, writer);
                    try writer.print(" ", .{});
                    try entry.value.format("", .{}, writer);
                }
                try writer.print("}}", .{});
            },
            .namespace => |ns| {
                try writer.print("(ns {s})", .{ns.name});
            },
            .nil => try writer.print("nil", .{}),
        }
    }

    pub fn eql(self: *const Value, other: *const Value) bool {
        const self_tag = std.meta.activeTag(self.*);
        const other_tag = std.meta.activeTag(other.*);

        if (self_tag != other_tag) return false;

        return switch (self.*) {
            .symbol => |s| std.mem.eql(u8, s, other.symbol),
            .keyword => |k| std.mem.eql(u8, k, other.keyword),
            .string => |s| std.mem.eql(u8, s, other.string),
            .int => |i| i == other.int,
            .float => |f| f == other.float,
            .list => |l| l == other.list, // pointer comparison for now
            .vector => |v| blk: {
                if (v.buf == null and other.vector.buf == null) break :blk true;
                if (v.buf == null or other.vector.buf == null) break :blk false;
                break :blk v.buf.?.ptr == other.vector.buf.?.ptr;
            },
            .map => |m| m.vec.buf.?.ptr == other.map.vec.buf.?.ptr, // pointer comparison for now
            .namespace => |ns| std.mem.eql(u8, ns.name, other.namespace.name),
            .nil => true,
        };
    }

    pub fn isSymbol(self: *const Value) bool {
        return std.meta.activeTag(self.*) == .symbol;
    }

    pub fn isKeyword(self: *const Value) bool {
        return std.meta.activeTag(self.*) == .keyword;
    }

    pub fn isString(self: *const Value) bool {
        return std.meta.activeTag(self.*) == .string;
    }

    pub fn isInt(self: *const Value) bool {
        return std.meta.activeTag(self.*) == .int;
    }

    pub fn isFloat(self: *const Value) bool {
        return std.meta.activeTag(self.*) == .float;
    }

    pub fn isList(self: *const Value) bool {
        return std.meta.activeTag(self.*) == .list;
    }

    pub fn isVector(self: *const Value) bool {
        return std.meta.activeTag(self.*) == .vector;
    }

    pub fn isMap(self: *const Value) bool {
        return std.meta.activeTag(self.*) == .map;
    }

    pub fn isNamespace(self: *const Value) bool {
        return std.meta.activeTag(self.*) == .namespace;
    }

    pub fn isNil(self: *const Value) bool {
        return std.meta.activeTag(self.*) == .nil;
    }
};

// Helper functions to create values
pub fn createSymbol(allocator: std.mem.Allocator, name: []const u8) !*Value {
    const val = try allocator.create(Value);
    // Copy the string to arena memory
    const owned_name = try allocator.dupe(u8, name);
    val.* = Value{ .symbol = owned_name };
    return val;
}

pub fn createKeyword(allocator: std.mem.Allocator, name: []const u8) !*Value {
    const val = try allocator.create(Value);
    // Copy the string to arena memory (without the ':')
    const owned_name = try allocator.dupe(u8, name);
    val.* = Value{ .keyword = owned_name };
    return val;
}

pub fn createString(allocator: std.mem.Allocator, str: []const u8) !*Value {
    const val = try allocator.create(Value);
    // Copy the string to arena memory
    const owned_str = try allocator.dupe(u8, str);
    val.* = Value{ .string = owned_str };
    return val;
}

pub fn createInt(allocator: std.mem.Allocator, i: i64) !*Value {
    const val = try allocator.create(Value);
    val.* = Value{ .int = i };
    return val;
}

pub fn createFloat(allocator: std.mem.Allocator, f: f64) !*Value {
    const val = try allocator.create(Value);
    val.* = Value{ .float = f };
    return val;
}

pub fn createNil(allocator: std.mem.Allocator) !*Value {
    const val = try allocator.create(Value);
    val.* = Value{ .nil = {} };
    return val;
}

pub fn createList(allocator: std.mem.Allocator) !*Value {
    const val = try allocator.create(Value);
    const list = try PersistentLinkedList(*Value).empty(allocator);
    val.* = Value{ .list = list };
    return val;
}

pub fn createVector(allocator: std.mem.Allocator) !*Value {
    const val = try allocator.create(Value);
    val.* = Value{ .vector = PersistentVector(*Value).init(allocator, null) };
    return val;
}

pub fn createMap(allocator: std.mem.Allocator) !*Value {
    const val = try allocator.create(Value);
    val.* = Value{ .map = PersistentMap(*Value, *Value).init(allocator) };
    return val;
}

pub fn createNamespace(allocator: std.mem.Allocator, name: []const u8) !*Value {
    const val = try allocator.create(Value);
    const owned_name = try allocator.dupe(u8, name);
    val.* = Value{ .namespace = NamespaceDecl{ .name = owned_name } };
    return val;
}

test "value creation and formatting" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const allocator = arena.allocator();

    // Test basic values
    const sym = try createSymbol(allocator, "hello");
    const kw = try createKeyword(allocator, "world");
    const str = try createString(allocator, "test");
    const int_val = try createInt(allocator, 42);
    const float_val = try createFloat(allocator, 3.14);
    const nil_val = try createNil(allocator);
    const ns_val = try createNamespace(allocator, "my.namespace");

    // Test type checks
    try std.testing.expect(sym.isSymbol());
    try std.testing.expect(kw.isKeyword());
    try std.testing.expect(str.isString());
    try std.testing.expect(int_val.isInt());
    try std.testing.expect(float_val.isFloat());
    try std.testing.expect(nil_val.isNil());
    try std.testing.expect(ns_val.isNamespace());

    // Test values
    try std.testing.expect(std.mem.eql(u8, sym.symbol, "hello"));
    try std.testing.expect(std.mem.eql(u8, kw.keyword, "world"));
    try std.testing.expect(std.mem.eql(u8, str.string, "test"));
    try std.testing.expect(int_val.int == 42);
    try std.testing.expect(float_val.float == 3.14);
    try std.testing.expect(std.mem.eql(u8, ns_val.namespace.name, "my.namespace"));

    // Test formatting (basic smoke test)
    var buf: [256]u8 = undefined;
    var stream = std.io.fixedBufferStream(&buf);
    try sym.format("", .{}, stream.writer());
    try std.testing.expect(std.mem.startsWith(u8, buf[0..stream.pos], "hello"));
}
