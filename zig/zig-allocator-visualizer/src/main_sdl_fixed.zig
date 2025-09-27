const std = @import("std");
const zig_allocator_visualizer = @import("zig_allocator_visualizer");

pub fn main() !void {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();

    var tracking = zig_allocator_visualizer.TrackingAllocator.init(gpa.allocator());
    defer tracking.deinit();

    const allocator = tracking.allocator();

    std.debug.print("=== Zig Allocator Visualizer SDL Demo ===\n", .{});
    std.debug.print("Starting SDL visualization on main thread (macOS compatible)...\n", .{});

    // Start SDL visualization
    var sdl_renderer = try zig_allocator_visualizer.SdlRenderer.init(gpa.allocator(), &tracking);
    defer sdl_renderer.deinit();

    // Start allocation simulation in background thread
    const alloc_thread = try std.Thread.spawn(.{}, simulateAllocations, .{ allocator, &tracking });

    // Run SDL visualization on main thread (required on macOS)
    try sdl_renderer.startVisualization();

    // Wait for allocation thread to complete
    alloc_thread.join();

    // Generate final SVG report
    try tracking.writeSvg("final_memory_map_sdl.svg");
    std.debug.print("\nFinal memory visualization saved to final_memory_map_sdl.svg\n", .{});
}

fn simulateAllocations(allocator: std.mem.Allocator, tracking: *zig_allocator_visualizer.TrackingAllocator) !void {
    // Give SDL time to initialize
    std.Thread.sleep(500_000_000); // 500ms

    std.debug.print("Starting allocation simulation in background...\n", .{});

    var allocations: std.ArrayList([]u8) = .{};
    defer allocations.deinit(allocator);

    // Small allocations
    for (0..10) |i| {
        const size = 16 + i * 8;
        const mem = try allocator.alloc(u8, size);
        try allocations.append(allocator, mem);
        std.debug.print("Allocated {d} bytes\n", .{size});
        std.Thread.sleep(300_000_000); // 300ms
    }

    // Medium allocations
    for (0..5) |i| {
        const size = 256 + i * 128;
        const mem = try allocator.alloc(u8, size);
        try allocations.append(allocator, mem);
        std.debug.print("Allocated {d} bytes\n", .{size});
        std.Thread.sleep(400_000_000); // 400ms
    }

    // Large allocation
    const large = try allocator.alloc(u8, 4096);
    try allocations.append(allocator, large);
    std.debug.print("Allocated 4096 bytes\n", .{});
    std.Thread.sleep(600_000_000); // 600ms

    // Free some allocations to show fragmentation
    std.debug.print("Freeing some allocations to show fragmentation...\n", .{});
    for (0..5) |i| {
        allocator.free(allocations.items[i]);
        std.debug.print("Freed allocation {d}\n", .{i});
        std.Thread.sleep(250_000_000); // 250ms
    }

    // More allocations after freeing (to show fragmentation)
    std.debug.print("Creating more allocations to show fragmentation...\n", .{});
    for (0..8) |_| {
        const mem = try allocator.alloc(u8, 64);
        try allocations.append(allocator, mem);
        std.debug.print("Allocated 64 bytes\n", .{});
        std.Thread.sleep(200_000_000); // 200ms
    }

    std.debug.print("\nCurrent stats:\n", .{});
    std.debug.print("  Allocations: {d}\n", .{tracking.getAllocationCount()});
    std.debug.print("  Total allocated: {d} bytes\n", .{tracking.getTotalBytesAllocated()});
    std.debug.print("  Current usage: {d} bytes\n", .{tracking.getCurrentBytesAllocated()});

    std.debug.print("\nAllocation simulation complete.\n", .{});
    std.debug.print("You can see the live visualization in the SDL window.\n", .{});
    std.debug.print("Close the window when you're done watching the visualization.\n", .{});

    // Keep allocations alive for a while longer for viewing
    std.Thread.sleep(10_000_000_000); // 10 seconds

    // Clean up remaining allocations
    for (allocations.items[5..]) |mem| {
        allocator.free(mem);
    }
}