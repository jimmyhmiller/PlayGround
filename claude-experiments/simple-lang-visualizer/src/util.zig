const std = @import("std");

/// Helper to allocate a null-terminated formatted string (replacement for removed allocPrintZ in Zig 0.15)
pub fn allocPrintZ(allocator: std.mem.Allocator, comptime fmt: []const u8, args: anytype) ![:0]u8 {
    const text = try std.fmt.allocPrint(allocator, fmt, args);
    defer allocator.free(text);
    return try allocator.dupeZ(u8, text);
}
