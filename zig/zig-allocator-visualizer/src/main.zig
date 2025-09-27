const std = @import("std");
const zig_allocator_visualizer = @import("zig_allocator_visualizer");

pub fn main() !void {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();

    var tracking = zig_allocator_visualizer.TrackingAllocator.init(gpa.allocator());
    defer tracking.deinit();

    const allocator = tracking.allocator();

    std.debug.print("=== Zig Allocator Visualizer Demo ===\n", .{});

    // Simulate some allocations
    const mem1 = try allocator.alloc(u8, 100);
    defer allocator.free(mem1);

    const mem2 = try allocator.alloc(u32, 50);
    defer allocator.free(mem2);

    const mem3 = try allocator.alloc(u64, 25);
    defer allocator.free(mem3);

    // Print statistics
    std.debug.print("Allocations made: {d}\n", .{tracking.getAllocationCount()});
    std.debug.print("Total bytes allocated: {d}\n", .{tracking.getTotalBytesAllocated()});
    std.debug.print("Current bytes in use: {d}\n", .{tracking.getCurrentBytesAllocated()});

    // Generate SVG visualization
    try tracking.writeSvg("memory_demo.svg");
    std.debug.print("Memory visualization saved to memory_demo.svg\n", .{});

    try zig_allocator_visualizer.bufferedPrint();
}

test "simple test" {
    const gpa = std.testing.allocator;
    var list: std.ArrayList(i32) = .empty;
    defer list.deinit(gpa); // Try commenting this out and see if zig detects the memory leak!
    try list.append(gpa, 42);
    try std.testing.expectEqual(@as(i32, 42), list.pop());
}

test "fuzz example" {
    const Context = struct {
        fn testOne(context: @This(), input: []const u8) anyerror!void {
            _ = context;
            // Try passing `--fuzz` to `zig build test` and see if it manages to fail this test case!
            try std.testing.expect(!std.mem.eql(u8, "canyoufindme", input));
        }
    };
    try std.testing.fuzz(Context{}, Context.testOne, .{});
}
