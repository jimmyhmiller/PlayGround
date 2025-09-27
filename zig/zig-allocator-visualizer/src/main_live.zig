const std = @import("std");
const c = @cImport({
    @cInclude("SDL2/SDL.h");
});

const zig_allocator_visualizer = @import("root.zig");

const builtin = @import("builtin");
const is_macos = builtin.os.tag == .macos;

// Shared state between threads
const SharedState = struct {
    tracking: *zig_allocator_visualizer.TrackingAllocator,
    program_finished: std.atomic.Value(bool),
    should_quit: std.atomic.Value(bool),

    fn init(tracking: *zig_allocator_visualizer.TrackingAllocator) SharedState {
        return .{
            .tracking = tracking,
            .program_finished = std.atomic.Value(bool).init(false),
            .should_quit = std.atomic.Value(bool).init(false),
        };
    }
};

// This creates a live monitoring version that hooks into a real program
pub fn main() !void {
    // Use page allocator as base
    const base_allocator = std.heap.page_allocator;

    // Create tracking allocator that will monitor all allocations
    var tracking = zig_allocator_visualizer.TrackingAllocator.init(base_allocator);
    defer tracking.deinit();

    // Create our monitored allocator
    const monitored_allocator = tracking.allocator();

    std.debug.print("=== Live Zig Allocator Visualizer ===\n", .{});
    std.debug.print("Monitoring real program allocations...\n", .{});

    // Create shared state
    var shared_state = SharedState.init(&tracking);

    if (is_macos) {
        // On macOS, run SDL on main thread and program on worker thread
        var program_thread = try std.Thread.spawn(.{}, programThread, .{ monitored_allocator, &shared_state });
        defer program_thread.join();

        // Run visualization on main thread
        try visualizationThreadMain(&shared_state);
    } else {
        // On other platforms, run program on main thread and SDL on worker thread
        var viz_thread = try std.Thread.spawn(.{}, visualizationThreadMain, .{&shared_state});
        defer viz_thread.join();

        // Run the actual program that we want to monitor
        try runRealProgram(monitored_allocator);
        shared_state.program_finished.store(true, .release);

        std.debug.print("Program finished. Visualization will continue showing final state.\n", .{});

        // Keep visualization running
        std.Thread.sleep(std.time.ns_per_s * 5); // Show final state for 5 seconds
        shared_state.should_quit.store(true, .release);
    }
}

fn programThread(allocator: std.mem.Allocator, shared_state: *SharedState) !void {
    try runRealProgram(allocator);
    shared_state.program_finished.store(true, .release);
    std.debug.print("Program finished. Visualization will continue showing final state.\n", .{});
}

fn visualizationThreadMain(shared_state: *SharedState) !void {
    // Initialize SDL
    if (c.SDL_Init(c.SDL_INIT_VIDEO) != 0) {
        std.debug.print("SDL init failed: {s}\n", .{c.SDL_GetError()});
        return;
    }
    defer c.SDL_Quit();

    const window = c.SDL_CreateWindow(
        "Live Zig Allocator Visualizer",
        c.SDL_WINDOWPOS_CENTERED,
        c.SDL_WINDOWPOS_CENTERED,
        1200,
        800,
        c.SDL_WINDOW_SHOWN | c.SDL_WINDOW_RESIZABLE,
    ) orelse {
        std.debug.print("Failed to create window: {s}\n", .{c.SDL_GetError()});
        return;
    };
    defer c.SDL_DestroyWindow(window);

    const renderer = c.SDL_CreateRenderer(
        window,
        -1,
        c.SDL_RENDERER_ACCELERATED | c.SDL_RENDERER_PRESENTVSYNC,
    ) orelse {
        std.debug.print("Failed to create renderer: {s}\n", .{c.SDL_GetError()});
        return;
    };
    defer c.SDL_DestroyRenderer(renderer);

    var quit = false;
    var event: c.SDL_Event = undefined;
    var sort_by_address = false;

    while (!quit and !shared_state.should_quit.load(.acquire)) {
        // Handle events
        while (c.SDL_PollEvent(&event) != 0) {
            switch (event.type) {
                c.SDL_QUIT => {
                    quit = true;
                    shared_state.should_quit.store(true, .release);
                },
                c.SDL_KEYDOWN => {
                    switch (event.key.keysym.sym) {
                        c.SDLK_SPACE => sort_by_address = !sort_by_address,
                        c.SDLK_ESCAPE => {
                            quit = true;
                            shared_state.should_quit.store(true, .release);
                        },
                        else => {},
                    }
                },
                else => {},
            }
        }

        // Render the current state
        try renderLiveVisualization(renderer, window, shared_state.tracking, sort_by_address);

        // Limit to ~60 FPS
        std.Thread.sleep(16_666_667); // ~16ms

        // If program finished and we're on macOS, wait a bit then quit
        if (is_macos and shared_state.program_finished.load(.acquire)) {
            std.Thread.sleep(std.time.ns_per_s * 3); // Show final state for 3 seconds
            quit = true;
        }
    }
}

fn renderLiveVisualization(renderer: *c.SDL_Renderer, window: *c.SDL_Window, tracking: *zig_allocator_visualizer.TrackingAllocator, sort_by_address: bool) !void {
    // Get window size
    var window_width: i32 = undefined;
    var window_height: i32 = undefined;
    c.SDL_GetWindowSize(window, &window_width, &window_height);

    // Clear screen
    _ = c.SDL_SetRenderDrawColor(renderer, 20, 20, 20, 255);
    _ = c.SDL_RenderClear(renderer);

    // Draw title
    // (In a real implementation, you'd use SDL_ttf for text)

    const tracker = tracking.getTracker();

    // Draw stats bars
    renderStats(renderer, tracker, window_width);

    // Draw memory visualization
    try renderMemoryVisualization(renderer, tracker, window_width, window_height, sort_by_address);

    // Draw instructions
    renderInstructions(renderer, window_width, window_height, sort_by_address);

    c.SDL_RenderPresent(renderer);
}

fn renderStats(renderer: *c.SDL_Renderer, tracker: *zig_allocator_visualizer.AllocationTracker, window_width: i32) void {
    const stats_y = 20;
    const bar_height = 15;
    const bar_width = window_width - 40;

    const current_usage = tracker.total_bytes_allocated - tracker.total_bytes_freed;
    const total_allocated = tracker.total_bytes_allocated;

    // Memory usage bar
    if (total_allocated > 0) {
        const usage_percent = @as(f32, @floatFromInt(current_usage)) / @as(f32, @floatFromInt(total_allocated));
        const filled_width = @as(i32, @intFromFloat(@as(f32, @floatFromInt(bar_width)) * usage_percent));

        // Background
        _ = c.SDL_SetRenderDrawColor(renderer, 50, 50, 50, 255);
        const bg_rect = c.SDL_Rect{ .x = 20, .y = stats_y, .w = bar_width, .h = bar_height };
        _ = c.SDL_RenderFillRect(renderer, &bg_rect);

        // Filled portion
        _ = c.SDL_SetRenderDrawColor(renderer, 100, 200, 100, 255);
        const fill_rect = c.SDL_Rect{ .x = 20, .y = stats_y, .w = filled_width, .h = bar_height };
        _ = c.SDL_RenderFillRect(renderer, &fill_rect);

        // Border
        _ = c.SDL_SetRenderDrawColor(renderer, 200, 200, 200, 255);
        _ = c.SDL_RenderDrawRect(renderer, &bg_rect);
    }

    // Allocation count bar
    const count_y = stats_y + 25;
    const count_width = @min(tracker.allocation_count * 5, @as(usize, @intCast(bar_width)));
    _ = c.SDL_SetRenderDrawColor(renderer, 100, 150, 255, 255);
    const count_rect = c.SDL_Rect{
        .x = 20,
        .y = count_y,
        .w = @intCast(count_width),
        .h = bar_height,
    };
    _ = c.SDL_RenderFillRect(renderer, &count_rect);
}

fn renderMemoryVisualization(renderer: *c.SDL_Renderer, tracker: *zig_allocator_visualizer.AllocationTracker, window_width: i32, window_height: i32, sort_by_address: bool) !void {
    const map_x = 20;
    const map_y = 80;
    const map_width = window_width - 40;
    const map_height = window_height - 120;

    // Draw border
    _ = c.SDL_SetRenderDrawColor(renderer, 100, 100, 100, 255);
    const border_rect = c.SDL_Rect{ .x = map_x, .y = map_y, .w = map_width, .h = map_height };
    _ = c.SDL_RenderDrawRect(renderer, &border_rect);

    if (tracker.memory_ordered.items.len == 0) return;

    // Sort allocations
    const allocator = std.heap.page_allocator;
    var sorted_allocs: std.ArrayList(zig_allocator_visualizer.AllocationInfo) = .{};
    defer sorted_allocs.deinit(allocator);

    for (tracker.memory_ordered.items) |info| {
        try sorted_allocs.append(allocator, info);
    }

    if (sort_by_address) {
        std.sort.insertion(zig_allocator_visualizer.AllocationInfo, sorted_allocs.items, {}, compareByAddress);
    } else {
        std.sort.insertion(zig_allocator_visualizer.AllocationInfo, sorted_allocs.items, {}, compareByTime);
    }

    // Render as grid
    const item_width = 8;
    const item_height = 20;
    const spacing = 2;
    const items_per_row = @as(usize, @intCast(@divTrunc(map_width - 10, item_width + spacing)));

    for (sorted_allocs.items, 0..) |info, i| {
        const row = @divTrunc(i, items_per_row);
        const col = i % items_per_row;

        const x = map_x + 5 + @as(i32, @intCast(col)) * (item_width + spacing);
        const y = map_y + 5 + @as(i32, @intCast(row)) * (item_height + spacing);

        if (y + item_height > map_y + map_height - 5) break; // Don't draw outside bounds

        // Color based on size
        const color = getColorForSize(info.size);
        _ = c.SDL_SetRenderDrawColor(renderer, color.r, color.g, color.b, 200);

        const rect = c.SDL_Rect{ .x = x, .y = y, .w = item_width, .h = item_height };
        _ = c.SDL_RenderFillRect(renderer, &rect);

        // Border for visibility
        _ = c.SDL_SetRenderDrawColor(renderer, 255, 255, 255, 100);
        _ = c.SDL_RenderDrawRect(renderer, &rect);
    }
}

fn renderInstructions(renderer: *c.SDL_Renderer, window_width: i32, window_height: i32, sort_by_address: bool) void {
    _ = renderer;
    _ = window_width;
    _ = window_height;
    _ = sort_by_address;
    // In a real implementation, you'd render text here showing:
    // "SPACE: Toggle sort (Time/Address)"
    // "ESC: Quit"
    // Current sort mode
}

fn compareByAddress(context: void, a: zig_allocator_visualizer.AllocationInfo, b: zig_allocator_visualizer.AllocationInfo) bool {
    _ = context;
    return @intFromPtr(a.ptr) < @intFromPtr(b.ptr);
}

fn compareByTime(context: void, a: zig_allocator_visualizer.AllocationInfo, b: zig_allocator_visualizer.AllocationInfo) bool {
    _ = context;
    return a.timestamp < b.timestamp;
}

const Color = struct { r: u8, g: u8, b: u8 };

fn getColorForSize(size: usize) Color {
    if (size < 64) {
        return .{ .r = 100, .g = 255, .b = 100 }; // Green for tiny
    } else if (size < 1024) {
        return .{ .r = 100, .g = 200, .b = 255 }; // Blue for small
    } else if (size < 16384) {
        return .{ .r = 255, .g = 200, .b = 100 }; // Orange for medium
    } else {
        return .{ .r = 255, .g = 100, .b = 100 }; // Red for large
    }
}

// This is the real program we want to monitor
fn runRealProgram(allocator: std.mem.Allocator) !void {
    std.debug.print("Starting real program with monitored allocations...\n", .{});

    // Simulate a real program doing various work

    // 1. Parse some JSON-like data
    std.debug.print("Phase 1: Parsing data structures...\n", .{});
    var data_structures: std.ArrayList([]u8) = .{};
    defer {
        for (data_structures.items) |item| {
            allocator.free(item);
        }
        data_structures.deinit(allocator);
    }

    for (0..50) |i| {
        const size = 32 + (i * 17) % 200; // Varying sizes
        const item = try allocator.alloc(u8, size);
        // Simulate using the memory
        for (item, 0..) |*byte, j| {
            byte.* = @intCast(j % 256);
        }
        try data_structures.append(allocator, item);

        if (i % 10 == 0) {
            std.Thread.sleep(100_000_000); // 100ms pause to see allocations happen
        }
    }

    std.Thread.sleep(500_000_000); // 500ms pause

    // 2. Build some structured data (simulating hash map behavior)
    std.debug.print("Phase 2: Building structured data...\n", .{});
    var structured_data: [30][]u8 = undefined;
    var structured_count: usize = 0;

    for (0..30) |i| {
        const value_size = 64 + (i * 23) % 500;
        const value = try allocator.alloc(u8, value_size);

        // Fill with data
        for (value, 0..) |*byte, j| {
            byte.* = @intCast((i + j) % 256);
        }

        structured_data[i] = value;
        structured_count += 1;

        if (i % 5 == 0) {
            std.Thread.sleep(150_000_000); // 150ms pause
        }
    }

    // Clean up structured data at the end
    defer {
        for (0..structured_count) |i| {
            allocator.free(structured_data[i]);
        }
    }

    std.Thread.sleep(500_000_000); // 500ms pause

    // 3. Allocate some large buffers
    std.debug.print("Phase 3: Large buffer allocations...\n", .{});
    var large_buffers: std.ArrayList([]u8) = .{};
    defer {
        for (large_buffers.items) |buffer| {
            allocator.free(buffer);
        }
        large_buffers.deinit(allocator);
    }

    for (0..5) |i| {
        const size = 4096 + i * 2048; // 4KB, 6KB, 8KB, 10KB, 12KB
        const buffer = try allocator.alloc(u8, size);

        // Simulate processing
        for (buffer, 0..) |*byte, j| {
            byte.* = @intCast((i * 100 + j) % 256);
        }

        try large_buffers.append(allocator, buffer);
        std.Thread.sleep(300_000_000); // 300ms pause between large allocations
    }

    std.Thread.sleep(1000_000_000); // 1 second pause

    // 4. Free some data structures to show fragmentation
    std.debug.print("Phase 4: Creating fragmentation...\n", .{});
    // Free every other item from data_structures
    var free_index: usize = data_structures.items.len;
    while (free_index > 0) {
        free_index -= 1;
        if (free_index % 2 == 1) {
            const item = data_structures.orderedRemove(free_index);
            allocator.free(item);
            std.Thread.sleep(50_000_000); // 50ms between frees
        }
    }

    std.Thread.sleep(1000_000_000); // 1 second pause

    // 5. Allocate more to show how memory gets reused
    std.debug.print("Phase 5: Memory reuse...\n", .{});
    for (0..20) |reuse_index| {
        const size = 100 + (reuse_index * 13) % 150;
        const item = try allocator.alloc(u8, size);
        try data_structures.append(allocator, item);
        std.Thread.sleep(100_000_000); // 100ms pause
    }

    std.debug.print("Real program finished. Final state will be displayed.\n", .{});
    std.Thread.sleep(2000_000_000); // 2 second final pause
}