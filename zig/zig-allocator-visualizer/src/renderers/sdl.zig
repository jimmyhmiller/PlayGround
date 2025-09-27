const std = @import("std");
const c = @cImport({
    @cInclude("SDL2/SDL.h");
});

const TrackingAllocator = @import("../tracking_allocator.zig").TrackingAllocator;
const AllocationInfo = @import("../allocation_tracker.zig").AllocationInfo;
const AllocationTracker = @import("../allocation_tracker.zig").AllocationTracker;
const Visualization = @import("../visualization.zig");

const builtin = @import("builtin");
const is_macos = builtin.os.tag == .macos;

pub const SdlRenderer = struct {
    allocator: std.mem.Allocator,
    window: ?*c.SDL_Window,
    renderer: ?*c.SDL_Renderer,
    tracking_allocator: *TrackingAllocator,
    render_thread: ?std.Thread,
    should_quit: std.atomic.Value(bool),
    event_queue: EventQueue,
    mutex: std.Thread.Mutex,
    width: i32,
    height: i32,

    const Self = @This();

    const RenderEvent = union(enum) {
        AllocationAdded: AllocationInfo,
        AllocationFreed: usize,
        Refresh,
        Quit,
    };

    const EventQueue = struct {
        events: [1024]RenderEvent,
        head: std.atomic.Value(usize),
        tail: std.atomic.Value(usize),

        fn init() EventQueue {
            return .{
                .events = undefined,
                .head = std.atomic.Value(usize).init(0),
                .tail = std.atomic.Value(usize).init(0),
            };
        }

        fn push(self: *EventQueue, event: RenderEvent) bool {
            const current_tail = self.tail.load(.acquire);
            const next_tail = (current_tail + 1) % self.events.len;

            if (next_tail == self.head.load(.acquire)) {
                return false; // Queue is full
            }

            self.events[current_tail] = event;
            self.tail.store(next_tail, .release);
            return true;
        }

        fn pop(self: *EventQueue) ?RenderEvent {
            const current_head = self.head.load(.acquire);
            if (current_head == self.tail.load(.acquire)) {
                return null; // Queue is empty
            }

            const event = self.events[current_head];
            const next_head = (current_head + 1) % self.events.len;
            self.head.store(next_head, .release);
            return event;
        }
    };

    pub fn init(allocator: std.mem.Allocator, tracking_allocator: *TrackingAllocator) !Self {
        if (c.SDL_Init(c.SDL_INIT_VIDEO) != 0) {
            std.debug.print("SDL init failed: {s}\n", .{c.SDL_GetError()});
            return error.SdlInitFailed;
        }

        return .{
            .allocator = allocator,
            .window = null,
            .renderer = null,
            .tracking_allocator = tracking_allocator,
            .render_thread = null,
            .should_quit = std.atomic.Value(bool).init(false),
            .event_queue = EventQueue.init(),
            .mutex = .{},
            .width = 1280,
            .height = 720,
        };
    }

    pub fn deinit(self: *Self) void {
        self.should_quit.store(true, .release);

        if (self.render_thread) |thread| {
            thread.join();
        }

        if (self.renderer) |renderer| {
            c.SDL_DestroyRenderer(renderer);
        }

        if (self.window) |window| {
            c.SDL_DestroyWindow(window);
        }

        c.SDL_Quit();
    }

    pub fn startVisualization(self: *Self) !void {
        if (is_macos) {
            // On macOS, SDL must be initialized on the main thread
            try self.initSdlResources();
            // Start worker thread for monitoring allocations
            self.render_thread = try std.Thread.spawn(.{}, workerThreadMain, .{self});
            // Run event loop on main thread
            try self.mainThreadEventLoop();
        } else {
            // On other platforms, can run everything in separate thread
            self.render_thread = try std.Thread.spawn(.{}, renderThreadMain, .{self});
        }
    }

    fn initSdlResources(self: *Self) !void {
        self.window = c.SDL_CreateWindow(
            "Zig Allocator Visualizer",
            c.SDL_WINDOWPOS_CENTERED,
            c.SDL_WINDOWPOS_CENTERED,
            self.width,
            self.height,
            c.SDL_WINDOW_SHOWN | c.SDL_WINDOW_RESIZABLE,
        ) orelse {
            std.debug.print("Failed to create window: {s}\n", .{c.SDL_GetError()});
            return error.WindowCreationFailed;
        };

        self.renderer = c.SDL_CreateRenderer(
            self.window,
            -1,
            c.SDL_RENDERER_ACCELERATED | c.SDL_RENDERER_PRESENTVSYNC,
        ) orelse {
            std.debug.print("Failed to create renderer: {s}\n", .{c.SDL_GetError()});
            return error.RendererCreationFailed;
        };
    }

    fn mainThreadEventLoop(self: *Self) !void {
        var event: c.SDL_Event = undefined;
        var last_render_time = std.time.milliTimestamp();
        const render_interval: i64 = 16; // ~60 FPS

        while (!self.should_quit.load(.acquire)) {
            // Process SDL events
            while (c.SDL_PollEvent(&event) != 0) {
                if (event.type == c.SDL_QUIT) {
                    self.should_quit.store(true, .release);
                    break;
                }
            }

            // Process render events from worker thread
            while (self.event_queue.pop()) |render_event| {
                switch (render_event) {
                    .Refresh => {},
                    .Quit => self.should_quit.store(true, .release),
                    else => {},
                }
            }

            // Render at target FPS
            const current_time = std.time.milliTimestamp();
            if (current_time - last_render_time >= render_interval) {
                try self.render();
                last_render_time = current_time;
            }

            std.Thread.sleep(1_000_000); // 1ms
        }
    }

    fn workerThreadMain(self: *Self) !void {
        var last_check = std.time.milliTimestamp();
        const check_interval: i64 = 100; // Check every 100ms

        while (!self.should_quit.load(.acquire)) {
            const current_time = std.time.milliTimestamp();
            if (current_time - last_check >= check_interval) {
                _ = self.event_queue.push(.Refresh);
                last_check = current_time;
            }

            std.Thread.sleep(10_000_000); // 10ms
        }
    }

    fn renderThreadMain(self: *Self) !void {
        try self.initSdlResources();

        var event: c.SDL_Event = undefined;
        var last_render_time = std.time.milliTimestamp();
        const render_interval: i64 = 16; // ~60 FPS

        while (!self.should_quit.load(.acquire)) {
            while (c.SDL_PollEvent(&event) != 0) {
                if (event.type == c.SDL_QUIT) {
                    self.should_quit.store(true, .release);
                    break;
                }
            }

            const current_time = std.time.milliTimestamp();
            if (current_time - last_render_time >= render_interval) {
                try self.render();
                last_render_time = current_time;
            }

            std.Thread.sleep(1_000_000); // 1ms
        }
    }

    fn render(self: *Self) !void {
        const renderer = self.renderer orelse return;

        // Clear screen
        _ = c.SDL_SetRenderDrawColor(renderer, 20, 20, 20, 255);
        _ = c.SDL_RenderClear(renderer);

        // Get current allocation state
        const tracker = self.tracking_allocator.getTracker();

        // Draw memory map
        try self.drawMemoryMap(tracker);

        // Draw stats
        try self.drawStats(tracker);

        // Present
        c.SDL_RenderPresent(renderer);
    }

    fn drawMemoryMap(self: *Self, tracker: *AllocationTracker) !void {
        const renderer = self.renderer orelse return;

        const map_x = 20;
        const map_y = 100;
        const map_width = self.width - 40;
        const map_height = self.height - 200;

        // Draw border
        _ = c.SDL_SetRenderDrawColor(renderer, 100, 100, 100, 255);
        const border_rect = c.SDL_Rect{
            .x = map_x,
            .y = map_y,
            .w = map_width,
            .h = map_height,
        };
        _ = c.SDL_RenderDrawRect(renderer, &border_rect);

        // Create memory view
        const mem_view = Visualization.MemoryMapView.init(
            tracker,
            @intCast(map_width),
            @intCast(map_height),
        );

        // Draw allocations
        self.mutex.lock();
        defer self.mutex.unlock();

        var color_index: u8 = 0;
        for (tracker.memory_ordered.items) |info| {

            // Calculate position
            const x = map_x + mem_view.mapAddressToX(@intFromPtr(info.ptr));
            const width = @max(2, mem_view.mapSizeToWidth(info.size));

            // Set color based on allocation size
            const color = getColorForSize(info.size);
            _ = c.SDL_SetRenderDrawColor(renderer, color.r, color.g, color.b, 200);

            // Draw rectangle
            const rect = c.SDL_Rect{
                .x = @intCast(x),
                .y = map_y + 10,
                .w = @intCast(width),
                .h = 20,
            };
            _ = c.SDL_RenderFillRect(renderer, &rect);

            color_index +%= 1;
        }
    }

    fn drawStats(self: *Self, tracker: *AllocationTracker) !void {
        // In a real implementation, we'd use SDL_ttf for text rendering
        // For now, we'll just draw some basic rectangles to represent stats

        const renderer = self.renderer orelse return;

        const stats_y = 20;
        const bar_height = 20;

        // Draw memory usage bar
        const current_usage = tracker.total_bytes_allocated - tracker.total_bytes_freed;
        const max_usage = tracker.total_bytes_allocated;

        if (max_usage > 0) {
            const usage_percent = @as(f32, @floatFromInt(current_usage)) / @as(f32, @floatFromInt(max_usage));
            const bar_width = @as(i32, @intFromFloat(@as(f32, @floatFromInt(self.width - 40)) * usage_percent));

            _ = c.SDL_SetRenderDrawColor(renderer, 50, 150, 250, 255);
            const usage_rect = c.SDL_Rect{
                .x = 20,
                .y = stats_y,
                .w = bar_width,
                .h = bar_height,
            };
            _ = c.SDL_RenderFillRect(renderer, &usage_rect);
        }

        // Draw allocation count indicator
        const count_width = @min(tracker.allocation_count * 2, @as(usize, @intCast(self.width - 40)));
        _ = c.SDL_SetRenderDrawColor(renderer, 250, 150, 50, 255);
        const count_rect = c.SDL_Rect{
            .x = 20,
            .y = stats_y + 30,
            .w = @intCast(count_width),
            .h = 10,
        };
        _ = c.SDL_RenderFillRect(renderer, &count_rect);
    }

    fn getColorForSize(size: usize) struct { r: u8, g: u8, b: u8 } {
        if (size < 64) {
            return .{ .r = 100, .g = 200, .b = 100 }; // Green for small
        } else if (size < 1024) {
            return .{ .r = 200, .g = 200, .b = 100 }; // Yellow for medium
        } else if (size < 65536) {
            return .{ .r = 200, .g = 100, .b = 100 }; // Orange for large
        } else {
            return .{ .r = 200, .g = 50, .b = 50 }; // Red for huge
        }
    }
};