const std = @import("std");
const mlir = @import("src/root.zig");

pub fn main() !void {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();

    const source = "%0 = \"test.op\"() : () -> !llvm<ptr<i32>>";
    std.debug.print("Original: {s}\n", .{source});

    var module1 = try mlir.parse(allocator, source);
    defer module1.deinit();

    const printed1 = try mlir.print(allocator, module1);
    defer allocator.free(printed1);
    std.debug.print("First print: {s}\n", .{printed1});
}
