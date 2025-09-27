const std = @import("std");
const zig_allocator_visualizer = @import("zig_allocator_visualizer");

pub fn main() !void {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();

    var tracking = zig_allocator_visualizer.TrackingAllocator.init(gpa.allocator());
    defer tracking.deinit();

    const allocator = tracking.allocator();

    std.debug.print("=== Zig Allocator Visualizer SDL Demo ===\n", .{});
    std.debug.print("Starting SDL visualization...\n", .{});

    // Start SDL visualization
    var sdl_renderer = try zig_allocator_visualizer.SdlRenderer.init(gpa.allocator(), &tracking);
    defer sdl_renderer.deinit();

    // Start visualization in background thread
    const viz_thread = try std.Thread.spawn(.{}, runVisualization, .{&sdl_renderer});

    // Give a moment for SDL to initialize
    std.Thread.sleep(100_000_000); // 100ms

    std.debug.print("Performing allocations...\n", .{});

    // Simulate some allocations over time
    var allocations: std.ArrayList([]u8) = .{};
    defer allocations.deinit(allocator);

    // Small allocations
    for (0..10) |i| {
        const size = 16 + i * 8;
        const mem = try allocator.alloc(u8, size);
        try allocations.append(allocator,mem);
        std.debug.print("Allocated {d} bytes\n", .{size});
        std.Thread.sleep(200_000_000); // 200ms
    }

    // Medium allocations
    for (0..5) |i| {
        const size = 256 + i * 128;
        const mem = try allocator.alloc(u8, size);
        try allocations.append(allocator,mem);
        std.debug.print("Allocated {d} bytes\n", .{size});
        std.Thread.sleep(300_000_000); // 300ms
    }

    // Large allocation
    const large = try allocator.alloc(u8, 4096);
    try allocations.append(allocator, large);
    std.debug.print("Allocated 4096 bytes\n", .{});
    std.Thread.sleep(500_000_000); // 500ms

    // Free some allocations to show fragmentation
    std.debug.print("Freeing some allocations...\n", .{});
    for (0..5) |i| {
        allocator.free(allocations.items[i]);
        std.debug.print("Freed allocation {d}\n", .{i});
        std.Thread.sleep(200_000_000); // 200ms
    }

    // More allocations after freeing (to show fragmentation)
    std.debug.print("Creating more allocations...\n", .{});
    for (0..5) |_| {
        const mem = try allocator.alloc(u8, 64);
        try allocations.append(allocator,mem);
        std.debug.print("Allocated 64 bytes\n", .{});
        std.Thread.sleep(150_000_000); // 150ms
    }

    std.debug.print("\nVisualization running. Close the SDL window to exit.\n", .{});
    std.debug.print("Current stats:\n", .{});
    std.debug.print("  Allocations: {d}\n", .{tracking.getAllocationCount()});
    std.debug.print("  Total allocated: {d} bytes\n", .{tracking.getTotalBytesAllocated()});
    std.debug.print("  Current usage: {d} bytes\n", .{tracking.getCurrentBytesAllocated()});

    // Wait for SDL window to be closed
    viz_thread.join();

    // Clean up remaining allocations
    for (allocations.items[5..]) |mem| {
        allocator.free(mem);
    }

    // Generate final SVG report
    try tracking.writeSvg("final_memory_map.svg");
    std.debug.print("\nFinal memory visualization saved to final_memory_map.svg\n", .{});
}

fn runVisualization(renderer: *zig_allocator_visualizer.SdlRenderer) !void {
    try renderer.startVisualization();
}