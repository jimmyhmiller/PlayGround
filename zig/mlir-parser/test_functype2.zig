const std = @import("std");
const mlir = @import("src/root.zig");

pub fn main() !void {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();

    // Corrected syntax with parentheses around the nested function type
    const source = "%0 = \"test.op\"() : () -> ((i32, i32, f32) -> i64)";
    std.debug.print("Original: {s}\n", .{source});

    var module1 = try mlir.parse(allocator, source);
    defer module1.deinit();

    const printed1 = try mlir.print(allocator, module1);
    defer allocator.free(printed1);
    std.debug.print("First print: {s}\n", .{printed1});
    
    var module2 = try mlir.parse(allocator, printed1);
    defer module2.deinit();
    
    const printed2 = try mlir.print(allocator, module2);
    defer allocator.free(printed2);
    std.debug.print("Second print: {s}\n", .{printed2});
}
