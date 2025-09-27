const std = @import("std");
const zig_allocator_visualizer = @import("zig_allocator_visualizer");
const c = @cImport({
    @cInclude("SDL2/SDL.h");
});

const Color = struct { r: u8, g: u8, b: u8 };

const Button = struct {
    x: i32,
    y: i32,
    width: i32,
    height: i32,
    text: []const u8,
    action: ButtonAction,
    color: Color = .{ .r = 100, .g = 100, .b = 200 },
    hover_color: Color = .{ .r = 120, .g = 120, .b = 220 },
    is_hovered: bool = false,

    fn isClicked(self: Button, mouse_x: i32, mouse_y: i32) bool {
        return mouse_x >= self.x and mouse_x < (self.x +| self.width) and
               mouse_y >= self.y and mouse_y < (self.y +| self.height);
    }

    fn render(self: Button, renderer: *c.SDL_Renderer) void {
        const color = if (self.is_hovered) self.hover_color else self.color;
        _ = c.SDL_SetRenderDrawColor(renderer, color.r, color.g, color.b, 255);

        const rect = c.SDL_Rect{
            .x = self.x,
            .y = self.y,
            .w = self.width,
            .h = self.height,
        };
        _ = c.SDL_RenderFillRect(renderer, &rect);

        // Draw border
        _ = c.SDL_SetRenderDrawColor(renderer, 255, 255, 255, 255);
        _ = c.SDL_RenderDrawRect(renderer, &rect);
    }
};

const ButtonAction = enum {
    AllocSmall,
    AllocMedium,
    AllocLarge,
    AllocHuge,
    FreeOldest,
    FreeNewest,
    FreeRandom,
    FragmentMemory,
    ClearAll,
    ResetArena,
    ToggleSortMode,
};

const SortMode = enum {
    ByTime,
    ByMemoryAddress,
};

const InteractiveDemo = struct {
    allocator: std.mem.Allocator,
    tracking: *zig_allocator_visualizer.TrackingAllocator,
    tracked_allocator: std.mem.Allocator,
    arena_allocator: ?*std.heap.ArenaAllocator,
    renderer: ?*c.SDL_Renderer,
    allocations: std.ArrayList([]u8),
    buttons: [11]Button,
    rng: std.Random.DefaultPrng,
    sort_mode: SortMode,

    const Self = @This();

    fn init(allocator: std.mem.Allocator, tracking: *zig_allocator_visualizer.TrackingAllocator, arena_allocator: ?*std.heap.ArenaAllocator) !Self {
        const tracked_allocator = tracking.allocator();

        const buttons = [_]Button{
            .{ .x = 20, .y = 50, .width = 100, .height = 30, .text = "Small (64B)", .action = .AllocSmall, .color = .{ .r = 100, .g = 200, .b = 100 } },
            .{ .x = 130, .y = 50, .width = 100, .height = 30, .text = "Medium (1KB)", .action = .AllocMedium, .color = .{ .r = 200, .g = 200, .b = 100 } },
            .{ .x = 240, .y = 50, .width = 100, .height = 30, .text = "Large (16KB)", .action = .AllocLarge, .color = .{ .r = 200, .g = 150, .b = 100 } },
            .{ .x = 350, .y = 50, .width = 100, .height = 30, .text = "Huge (1MB)", .action = .AllocHuge, .color = .{ .r = 200, .g = 100, .b = 100 } },

            .{ .x = 20, .y = 90, .width = 100, .height = 30, .text = "Free Oldest", .action = .FreeOldest, .color = .{ .r = 150, .g = 100, .b = 150 } },
            .{ .x = 130, .y = 90, .width = 100, .height = 30, .text = "Free Newest", .action = .FreeNewest, .color = .{ .r = 150, .g = 100, .b = 150 } },
            .{ .x = 240, .y = 90, .width = 100, .height = 30, .text = "Free Random", .action = .FreeRandom, .color = .{ .r = 150, .g = 100, .b = 150 } },
            .{ .x = 350, .y = 90, .width = 100, .height = 30, .text = "Fragment", .action = .FragmentMemory, .color = .{ .r = 180, .g = 120, .b = 80 } },

            .{ .x = 20, .y = 130, .width = 100, .height = 30, .text = "Clear All", .action = .ClearAll, .color = .{ .r = 200, .g = 50, .b = 50 } },
            .{ .x = 130, .y = 130, .width = 100, .height = 30, .text = "Reset Arena", .action = .ResetArena, .color = .{ .r = 200, .g = 100, .b = 50 } },
            .{ .x = 250, .y = 130, .width = 120, .height = 30, .text = "Sort: Time", .action = .ToggleSortMode, .color = .{ .r = 100, .g = 150, .b = 200 } },
        };

        return Self{
            .allocator = allocator,
            .tracking = tracking,
            .tracked_allocator = tracked_allocator,
            .arena_allocator = arena_allocator,
            .renderer = null,
            .allocations = .{},
            .buttons = buttons,
            .rng = std.Random.DefaultPrng.init(@intCast(std.time.milliTimestamp())),
            .sort_mode = .ByTime,
        };
    }

    fn deinit(self: *Self) void {
        // Free all remaining allocations
        for (self.allocations.items) |allocation| {
            self.tracked_allocator.free(allocation);
        }
        self.allocations.deinit(self.allocator);
    }

    fn handleButtonClick(self: *Self, action: ButtonAction) !void {
        const random = self.rng.random();

        switch (action) {
            .AllocSmall => {
                const size = 32 + random.intRangeLessThan(usize, 0, 64);
                const mem = try self.tracked_allocator.alloc(u8, size);
                try self.allocations.append(self.allocator, mem);
                std.debug.print("Allocated {d} bytes (small) at address 0x{x}\n", .{size, @intFromPtr(mem.ptr)});
            },
            .AllocMedium => {
                const size = 512 + random.intRangeLessThan(usize, 0, 1024);
                const mem = try self.tracked_allocator.alloc(u8, size);
                try self.allocations.append(self.allocator, mem);
                std.debug.print("Allocated {d} bytes (medium) at address 0x{x}\n", .{size, @intFromPtr(mem.ptr)});
            },
            .AllocLarge => {
                const size = 8192 + random.intRangeLessThan(usize, 0, 16384);
                const mem = try self.tracked_allocator.alloc(u8, size);
                try self.allocations.append(self.allocator, mem);
                std.debug.print("Allocated {d} bytes (large) at address 0x{x}\n", .{size, @intFromPtr(mem.ptr)});
            },
            .AllocHuge => {
                const size = 512 * 1024 + random.intRangeLessThan(usize, 0, 512 * 1024);
                const mem = try self.tracked_allocator.alloc(u8, size);
                try self.allocations.append(self.allocator, mem);
                std.debug.print("Allocated {d} bytes (huge) at address 0x{x}\n", .{size, @intFromPtr(mem.ptr)});
            },
            .FreeOldest => {
                if (self.allocations.items.len > 0) {
                    const mem = self.allocations.orderedRemove(0);
                    std.debug.print("Freed oldest allocation at address 0x{x} ({d} bytes)\n", .{@intFromPtr(mem.ptr), mem.len});
                    self.tracked_allocator.free(mem);
                }
            },
            .FreeNewest => {
                if (self.allocations.items.len > 0) {
                    const mem = self.allocations.pop() orelse return;
                    std.debug.print("Freed newest allocation at address 0x{x} ({d} bytes)\n", .{@intFromPtr(mem.ptr), mem.len});
                    self.tracked_allocator.free(mem);
                }
            },
            .FreeRandom => {
                if (self.allocations.items.len > 0) {
                    const index = random.intRangeLessThan(usize, 0, self.allocations.items.len);
                    const mem = self.allocations.orderedRemove(index);
                    std.debug.print("Freed random allocation at address 0x{x} ({d} bytes)\n", .{@intFromPtr(mem.ptr), mem.len});
                    self.tracked_allocator.free(mem);
                }
            },
            .FragmentMemory => {
                // Allocate several blocks, then free every other one
                std.debug.print("Creating fragmentation pattern...\n", .{});
                var temp_allocs: [10][]u8 = undefined;
                for (0..10) |i| {
                    const size = 256 + i * 32;
                    temp_allocs[i] = try self.tracked_allocator.alloc(u8, size);
                    try self.allocations.append(self.allocator, temp_allocs[i]);
                    std.debug.print("Fragment alloc [{d}]: {d} bytes at address 0x{x}\n", .{i, size, @intFromPtr(temp_allocs[i].ptr)});
                }
                // Free every other allocation
                var i: usize = 9;
                while (i > 0) {
                    if (i % 2 == 1) {
                        // Find and remove this allocation
                        for (self.allocations.items, 0..) |alloc, idx| {
                            if (alloc.ptr == temp_allocs[i].ptr) {
                                const removed = self.allocations.orderedRemove(idx);
                                std.debug.print("Fragment free [{d}]: address 0x{x} ({d} bytes)\n", .{i, @intFromPtr(removed.ptr), removed.len});
                                self.tracked_allocator.free(removed);
                                break;
                            }
                        }
                    }
                    i -= 1;
                }
                std.debug.print("Fragmentation pattern created\n", .{});
            },
            .ClearAll => {
                for (self.allocations.items) |allocation| {
                    self.tracked_allocator.free(allocation);
                }
                self.allocations.clearRetainingCapacity();
                std.debug.print("Cleared all allocations\n", .{});
            },
            .ResetArena => {
                if (self.arena_allocator) |arena| {
                    // Clear our tracking list since all allocations will be invalid
                    self.allocations.clearRetainingCapacity();

                    // Reset the arena - this frees all allocations at once
                    _ = arena.reset(.retain_capacity);
                    std.debug.print("Reset arena allocator - all allocations freed in bulk\n", .{});
                } else {
                    std.debug.print("No arena allocator available\n", .{});
                }
            },
            .ToggleSortMode => {
                self.sort_mode = switch (self.sort_mode) {
                    .ByTime => .ByMemoryAddress,
                    .ByMemoryAddress => .ByTime,
                };
                // Update button text
                for (&self.buttons) |*button| {
                    if (button.action == .ToggleSortMode) {
                        button.text = switch (self.sort_mode) {
                            .ByTime => "Sort: Time",
                            .ByMemoryAddress => "Sort: Memory",
                        };
                        break;
                    }
                }
                std.debug.print("Sort mode: {}\n", .{self.sort_mode});
            },
        }
    }

    fn updateButtons(self: *Self, mouse_x: i32, mouse_y: i32) void {
        for (&self.buttons) |*button| {
            button.is_hovered = button.isClicked(mouse_x, mouse_y);
        }
    }

    fn render(self: *Self, renderer: *c.SDL_Renderer, window: *c.SDL_Window) !void {
        // Clear screen
        _ = c.SDL_SetRenderDrawColor(renderer, 20, 20, 20, 255);
        _ = c.SDL_RenderClear(renderer);

        // Draw title
        // Note: In a real implementation, we'd use SDL_ttf for text rendering
        // For now, we'll just draw the buttons and stats

        // Draw buttons
        for (self.buttons) |button| {
            button.render(renderer);
        }

        // Draw stats (as colored bars)
        self.renderStats(renderer, window);

        // Draw legend
        self.renderLegend(renderer, window);

        // Draw memory map
        try self.renderMemoryMap(renderer, window);

        // Present
        c.SDL_RenderPresent(renderer);
    }

    fn renderStats(self: *Self, renderer: *c.SDL_Renderer, window: *c.SDL_Window) void {
        // Get current window size and scale stats bar width
        var window_width: i32 = undefined;
        var window_height: i32 = undefined;
        c.SDL_GetWindowSize(window, &window_width, &window_height);

        const stats_y = 170;
        const bar_height = 20;
        const bar_width = @min(400, window_width - 300); // Max 400px, but scale down if window is small

        // Memory usage bar
        const current_usage = self.tracking.getCurrentBytesAllocated();
        const total_allocated = self.tracking.getTotalBytesAllocated();

        if (total_allocated > 0) {
            const usage_percent = @as(f32, @floatFromInt(current_usage)) / @as(f32, @floatFromInt(total_allocated));
            const filled_width = @as(i32, @intFromFloat(@as(f32, @floatFromInt(bar_width)) * usage_percent));

            // Background
            _ = c.SDL_SetRenderDrawColor(renderer, 50, 50, 50, 255);
            const bg_rect = c.SDL_Rect{ .x = 20, .y = stats_y, .w = bar_width, .h = bar_height };
            _ = c.SDL_RenderFillRect(renderer, &bg_rect);

            // Filled portion
            _ = c.SDL_SetRenderDrawColor(renderer, 50, 150, 250, 255);
            const fill_rect = c.SDL_Rect{ .x = 20, .y = stats_y, .w = filled_width, .h = bar_height };
            _ = c.SDL_RenderFillRect(renderer, &fill_rect);

            // Border
            _ = c.SDL_SetRenderDrawColor(renderer, 200, 200, 200, 255);
            _ = c.SDL_RenderDrawRect(renderer, &bg_rect);
        }

        // Allocation count indicator
        const count_width = @min(self.tracking.getAllocationCount() * 3, @as(usize, @intCast(bar_width)));
        _ = c.SDL_SetRenderDrawColor(renderer, 250, 150, 50, 255);
        const count_rect = c.SDL_Rect{
            .x = 20,
            .y = stats_y + 30,
            .w = @intCast(count_width),
            .h = 10,
        };
        _ = c.SDL_RenderFillRect(renderer, &count_rect);
    }

    fn renderLegend(self: *Self, renderer: *c.SDL_Renderer, window: *c.SDL_Window) void {
        // Get current window size and position legend on the right
        var window_width: i32 = undefined;
        var window_height: i32 = undefined;
        c.SDL_GetWindowSize(window, &window_width, &window_height);

        const legend_x = window_width - 270; // 270px from right edge
        const legend_y = 170;
        const box_size = 15;
        _ = 20; // spacing not used

        // Draw legend title area
        const legend_height: i32 = if (self.sort_mode == .ByMemoryAddress) 50 else 25;
        _ = c.SDL_SetRenderDrawColor(renderer, 40, 40, 40, 255);
        const legend_bg = c.SDL_Rect{
            .x = legend_x - 5,
            .y = legend_y - 5,
            .w = 250,
            .h = legend_height,
        };
        _ = c.SDL_RenderFillRect(renderer, &legend_bg);

        // Legend items: Size categories
        const categories = [_]SizeCategory{ .tiny, .small, .medium, .large };
        _ = [_][]const u8{ "Tiny(<128B)", "Small(2KB)", "Med(64KB)", "Large(>64KB)" }; // labels not used in visual legend

        for (categories, 0..) |category, i| {
            const x = legend_x + @as(i32, @intCast(i)) * 60;
            const color = getColorForSizeCategory(category);

            // Draw color box
            _ = c.SDL_SetRenderDrawColor(renderer, color.r, color.g, color.b, 255);
            const box = c.SDL_Rect{
                .x = x,
                .y = legend_y,
                .w = box_size,
                .h = box_size,
            };
            _ = c.SDL_RenderFillRect(renderer, &box);

            // Draw border
            _ = c.SDL_SetRenderDrawColor(renderer, 255, 255, 255, 255);
            _ = c.SDL_RenderDrawRect(renderer, &box);
        }

        // Add gap legend when in memory address mode
        if (self.sort_mode == .ByMemoryAddress) {
            // Draw gap example
            _ = c.SDL_SetRenderDrawColor(renderer, 60, 60, 60, 255);
            const gap_box = c.SDL_Rect{
                .x = legend_x,
                .y = legend_y + 25,
                .w = box_size,
                .h = box_size,
            };
            _ = c.SDL_RenderFillRect(renderer, &gap_box);

            // Add stripes to gap example
            _ = c.SDL_SetRenderDrawColor(renderer, 80, 80, 80, 255);
            var stripe_x: i32 = legend_x;
            while (stripe_x < legend_x + box_size) {
                _ = c.SDL_RenderDrawLine(renderer, stripe_x, legend_y + 25, stripe_x, legend_y + 25 + box_size);
                stripe_x += 3;
            }

            // Draw border
            _ = c.SDL_SetRenderDrawColor(renderer, 120, 120, 120, 255);
            _ = c.SDL_RenderDrawRect(renderer, &gap_box);
        }

    }

    fn renderMemoryMap(self: *Self, renderer: *c.SDL_Renderer, window: *c.SDL_Window) !void {
        // Get current window size
        var window_width: i32 = undefined;
        var window_height: i32 = undefined;
        c.SDL_GetWindowSize(window, &window_width, &window_height);

        // Calculate dynamic map dimensions - use most of the window
        const map_x = 20;
        const map_y = 220;
        const legend_width = 280; // Reserve space for legend
        const map_width = window_width - map_x - legend_width - 20; // 20px right margin
        const map_height = window_height - map_y - 40; // 40px bottom margin

        // Draw border
        _ = c.SDL_SetRenderDrawColor(renderer, 100, 100, 100, 255);
        const border_rect = c.SDL_Rect{
            .x = map_x,
            .y = map_y,
            .w = map_width,
            .h = map_height,
        };
        _ = c.SDL_RenderDrawRect(renderer, &border_rect);

        // Get tracker for memory map
        const tracker = self.tracking.getTracker();

        // Grid-based visualization: allocation order with size-based width
        const grid_start_x = map_x + 5;
        const grid_start_y = map_y + 5;
        const row_height = 25;
        const spacing = 2;
        const available_width = map_width - 10;

        var current_x: i32 = grid_start_x;
        var current_y: i32 = grid_start_y;
        var allocation_index: usize = 0;

        // Create a copy of allocations and sort based on current mode
        var sorted_allocs: std.ArrayList(zig_allocator_visualizer.AllocationInfo) = .{};
        defer sorted_allocs.deinit(self.allocator);

        // Copy all allocations
        for (tracker.memory_ordered.items) |info| {
            sorted_allocs.append(self.allocator, info) catch continue;
        }

        // Sort based on current mode
        switch (self.sort_mode) {
            .ByTime => {
                std.sort.insertion(zig_allocator_visualizer.AllocationInfo, sorted_allocs.items, {}, compareByTimestamp);
            },
            .ByMemoryAddress => {
                std.sort.insertion(zig_allocator_visualizer.AllocationInfo, sorted_allocs.items, {}, compareByMemoryAddress);
            },
        }

        // Different rendering based on sort mode
        switch (self.sort_mode) {
            .ByTime => {
                // Time-based: show as grid with size-based widths
                for (sorted_allocs.items) |info| {
                    const size_category = getSizeCategory(info.size);
                    const width = getWidthForSizeCategory(size_category);

                    // Check if we need to wrap to next row
                    if (current_x + width > grid_start_x + available_width) {
                        current_x = grid_start_x;
                        current_y += row_height + spacing;
                        if (current_y + row_height > map_y + map_height - 5) {
                            break;
                        }
                    }

                    // Get color for size category
                    const color = getColorForSizeCategory(size_category);
                    _ = c.SDL_SetRenderDrawColor(renderer, color.r, color.g, color.b, 220);

                    // Make large allocations taller for better visibility
                    const height: i32 = switch (size_category) {
                        .large => row_height + 10, // Make large allocations taller
                        else => row_height,
                    };

                    const rect = c.SDL_Rect{
                        .x = current_x,
                        .y = current_y,
                        .w = width,
                        .h = height,
                    };
                    _ = c.SDL_RenderFillRect(renderer, &rect);

                    // Draw border for visibility
                    _ = c.SDL_SetRenderDrawColor(renderer, 255, 255, 255, 180);
                    _ = c.SDL_RenderDrawRect(renderer, &rect);

                    current_x += width + spacing;
                    allocation_index += 1;
                }
            },
            .ByMemoryAddress => {
                // Memory address mode: show proportional to actual memory layout with gaps
                try self.renderMemoryAddressLayout(renderer, sorted_allocs.items, map_x, map_y, map_width, map_height);
            },
        }
    }

    const SizeCategory = enum {
        tiny,    // < 128 bytes
        small,   // 128 - 2KB
        medium,  // 2KB - 64KB
        large,   // > 64KB
    };

    fn getSizeCategory(size: usize) SizeCategory {
        if (size < 128) {
            return .tiny;
        } else if (size < 2048) {
            return .small;
        } else if (size < 65536) {
            return .medium;
        } else {
            return .large;
        }
    }

    fn getWidthForSizeCategory(category: SizeCategory) i32 {
        return switch (category) {
            .tiny => 12,
            .small => 35,
            .medium => 80,
            .large => 150, // Make large allocations much more prominent
        };
    }

    fn getColorForSizeCategory(category: SizeCategory) Color {
        return switch (category) {
            .tiny => .{ .r = 100, .g = 255, .b = 100 },   // Bright green for tiny
            .small => .{ .r = 255, .g = 255, .b = 100 },  // Yellow for small
            .medium => .{ .r = 255, .g = 150, .b = 100 }, // Orange for medium
            .large => .{ .r = 255, .g = 100, .b = 100 },  // Red for large
        };
    }

    fn compareByTimestamp(context: void, a: zig_allocator_visualizer.AllocationInfo, b: zig_allocator_visualizer.AllocationInfo) bool {
        _ = context;
        return a.timestamp < b.timestamp;
    }

    fn compareByMemoryAddress(context: void, a: zig_allocator_visualizer.AllocationInfo, b: zig_allocator_visualizer.AllocationInfo) bool {
        _ = context;
        return @intFromPtr(a.ptr) < @intFromPtr(b.ptr);
    }

    fn renderMemoryAddressLayout(self: *Self, renderer: *c.SDL_Renderer, allocations: []zig_allocator_visualizer.AllocationInfo, map_x: i32, map_y: i32, map_width: i32, map_height: i32) !void {
        _ = self;
        if (allocations.len == 0) return;

        // Fixed address space visualization
        // Each row represents a fixed range of memory addresses
        const bytes_per_row: usize = 16384; // 16KB per row
        const row_height: i32 = 20;
        const draw_x = map_x + 10;
        const draw_y = map_y + 10;
        const draw_width = map_width - 20;
        const draw_height = map_height - 20;

        // Find the address range
        var min_addr: usize = std.math.maxInt(usize);
        var max_addr: usize = 0;
        for (allocations) |info| {
            const addr = @intFromPtr(info.ptr);
            const end_addr = addr + info.size;
            if (addr < min_addr) min_addr = addr;
            if (end_addr > max_addr) max_addr = end_addr;
        }

        // Align to row boundaries
        const start_addr = (min_addr / bytes_per_row) * bytes_per_row;
        const end_addr = ((max_addr + bytes_per_row - 1) / bytes_per_row) * bytes_per_row;

        // Calculate how many rows we need
        const total_rows = (end_addr - start_addr) / bytes_per_row;
        const max_visible_rows = @as(usize, @intCast(@divTrunc(draw_height, row_height + 2)));
        const rows_to_show = @min(total_rows, max_visible_rows);

        // Pixels per byte for this scale
        const pixels_per_byte = @as(f32, @floatFromInt(draw_width)) / @as(f32, @floatFromInt(bytes_per_row));

        // Draw background
        _ = c.SDL_SetRenderDrawColor(renderer, 20, 20, 20, 255);
        const bg_rect = c.SDL_Rect{
            .x = draw_x,
            .y = draw_y,
            .w = draw_width,
            .h = @intCast(rows_to_show * @as(usize, @intCast(row_height + 2))),
        };
        _ = c.SDL_RenderFillRect(renderer, &bg_rect);

        // Draw row backgrounds and grid lines
        for (0..rows_to_show) |row| {
            const y = draw_y + @as(i32, @intCast(row)) * (row_height + 2);

            // Row separator
            _ = c.SDL_SetRenderDrawColor(renderer, 40, 40, 40, 255);
            _ = c.SDL_RenderDrawLine(renderer, draw_x, y, draw_x + draw_width, y);

            // Draw KB markers
            _ = c.SDL_SetRenderDrawColor(renderer, 35, 35, 35, 255);
            const kb_pixels = @as(i32, @intFromFloat(1024.0 * pixels_per_byte));
            var marker_x = draw_x;
            while (marker_x < draw_x + draw_width) {
                _ = c.SDL_RenderDrawLine(renderer, marker_x, y, marker_x, y + row_height);
                marker_x += kb_pixels;
            }
        }

        // Draw allocations at their fixed positions
        for (allocations) |info| {
            const addr = @intFromPtr(info.ptr);
            const size = info.size;

            // Skip if outside visible range
            if (addr < start_addr or addr >= start_addr + rows_to_show * bytes_per_row) continue;

            // Calculate position
            const relative_addr = addr - start_addr;
            const row = relative_addr / bytes_per_row;
            const byte_in_row = relative_addr % bytes_per_row;

            _ = draw_y + @as(i32, @intCast(row)) * (row_height + 2) + 1; // y not used in loop
            const x = draw_x + @as(i32, @intFromFloat(@as(f32, @floatFromInt(byte_in_row)) * pixels_per_byte));
            _ = @max(2, @as(i32, @intFromFloat(@as(f32, @floatFromInt(size)) * pixels_per_byte))); // width not used in loop

            // Get color based on size category
            const size_category = getSizeCategory(info.size);
            const color = getColorForSizeCategory(size_category);

            // Handle allocations that span multiple rows
            var remaining_size = size;
            var current_row = row;
            var current_x = x;

            while (remaining_size > 0 and current_row < rows_to_show) {
                const bytes_in_current_row = if (current_row == row)
                    @min(remaining_size, bytes_per_row - byte_in_row)
                else
                    @min(remaining_size, bytes_per_row);

                const segment_width = @max(2, @as(i32, @intFromFloat(@as(f32, @floatFromInt(bytes_in_current_row)) * pixels_per_byte)));
                const segment_y = draw_y + @as(i32, @intCast(current_row)) * (row_height + 2) + 1;

                // Draw allocation segment
                _ = c.SDL_SetRenderDrawColor(renderer, color.r, color.g, color.b, 220);
                const alloc_rect = c.SDL_Rect{
                    .x = current_x,
                    .y = segment_y,
                    .w = segment_width,
                    .h = row_height - 2,
                };
                _ = c.SDL_RenderFillRect(renderer, &alloc_rect);

                // Draw border
                _ = c.SDL_SetRenderDrawColor(renderer, 255, 255, 255, 180);
                _ = c.SDL_RenderDrawRect(renderer, &alloc_rect);

                remaining_size -= bytes_in_current_row;
                current_row += 1;
                current_x = draw_x; // Next row starts at the beginning
            }
        }

        // Highlight gaps between consecutive allocations
        for (0..allocations.len - 1) |i| {
            const current = allocations[i];
            const next = allocations[i + 1];
            const current_end = @intFromPtr(current.ptr) + current.size;
            const next_start = @intFromPtr(next.ptr);

            if (next_start > current_end and next_start - current_end > 256) { // Only show gaps > 256 bytes
                const gap_addr = current_end;
                const gap_size = next_start - current_end;

                // Skip if outside visible range
                if (gap_addr < start_addr or gap_addr >= start_addr + rows_to_show * bytes_per_row) continue;

                const relative_gap_addr = gap_addr - start_addr;
                const gap_row = relative_gap_addr / bytes_per_row;
                const gap_byte_in_row = relative_gap_addr % bytes_per_row;

                const gap_y = draw_y + @as(i32, @intCast(gap_row)) * (row_height + 2) + 1;
                const gap_x = draw_x + @as(i32, @intFromFloat(@as(f32, @floatFromInt(gap_byte_in_row)) * pixels_per_byte));
                const gap_width = @max(2, @as(i32, @intFromFloat(@as(f32, @floatFromInt(@min(gap_size, bytes_per_row - gap_byte_in_row))) * pixels_per_byte)));

                // Draw gap indicator with stripes
                _ = c.SDL_SetRenderDrawColor(renderer, 100, 50, 50, 100);
                var stripe_x: i32 = gap_x;
                while (stripe_x < gap_x + gap_width) {
                    _ = c.SDL_RenderDrawLine(renderer, stripe_x, gap_y, stripe_x, gap_y + row_height - 2);
                    stripe_x += 3;
                }
            }
        }

        // Draw address labels
        _ = c.SDL_SetRenderDrawColor(renderer, 150, 150, 150, 255);
        for (0..@min(rows_to_show, 5)) |row| {
            const row_addr = start_addr + row * bytes_per_row;
            // Simple visual indicator - draw address hash as a pattern
            const addr_hash = @as(u8, @truncate(row_addr >> 12));
            const y = draw_y + @as(i32, @intCast(row)) * (row_height + 2) + row_height / 2;
            _ = c.SDL_RenderDrawPoint(renderer, draw_x - 5, y);
            if (addr_hash & 1 != 0) _ = c.SDL_RenderDrawPoint(renderer, draw_x - 4, y);
            if (addr_hash & 2 != 0) _ = c.SDL_RenderDrawPoint(renderer, draw_x - 3, y);
        }
    }

    fn getColorForSize(size: usize) Color {
        if (size < 128) {
            return .{ .r = 100, .g = 200, .b = 100 }; // Green for small
        } else if (size < 2048) {
            return .{ .r = 200, .g = 200, .b = 100 }; // Yellow for medium
        } else if (size < 65536) {
            return .{ .r = 200, .g = 150, .b = 100 }; // Orange for large
        } else {
            return .{ .r = 200, .g = 50, .b = 50 }; // Red for huge
        }
    }
};

pub fn main() !void {
    // Use page allocator as the base for the arena allocator
    const base_allocator = std.heap.page_allocator;

    // Use an arena allocator for simpler allocation patterns
    var arena_allocator = std.heap.ArenaAllocator.init(base_allocator);
    defer arena_allocator.deinit();

    var tracking = zig_allocator_visualizer.TrackingAllocator.init(arena_allocator.allocator());
    defer tracking.deinit();

    std.debug.print("=== Interactive Zig Allocator Visualizer ===\n", .{});
    std.debug.print("Click buttons to allocate/free memory and watch the visualization!\n", .{});
    std.debug.print("Using Arena Allocator - allocations will be freed in bulk\n", .{});

    var demo = try InteractiveDemo.init(base_allocator, &tracking, &arena_allocator);
    defer demo.deinit();

    // Initialize SDL
    if (c.SDL_Init(c.SDL_INIT_VIDEO) != 0) {
        std.debug.print("SDL init failed: {s}\n", .{c.SDL_GetError()});
        return;
    }
    defer c.SDL_Quit();

    const window = c.SDL_CreateWindow(
        "Interactive Zig Allocator Visualizer",
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

    demo.renderer = renderer;

    // Main event loop
    var quit = false;
    var event: c.SDL_Event = undefined;
    var mouse_x: i32 = 0;
    var mouse_y: i32 = 0;

    while (!quit) {
        // Handle events
        while (c.SDL_PollEvent(&event) != 0) {
            switch (event.type) {
                c.SDL_QUIT => quit = true,
                c.SDL_WINDOWEVENT => {
                    if (event.window.event == c.SDL_WINDOWEVENT_RESIZED) {
                        std.debug.print("Window resized to {}x{}\n", .{ event.window.data1, event.window.data2 });
                    }
                },
                c.SDL_MOUSEBUTTONDOWN => {
                    if (event.button.button == c.SDL_BUTTON_LEFT) {
                        mouse_x = event.button.x;
                        mouse_y = event.button.y;

                        // Check button clicks
                        for (demo.buttons) |button| {
                            if (button.isClicked(mouse_x, mouse_y)) {
                                demo.handleButtonClick(button.action) catch |err| {
                                    std.debug.print("Error handling button click: {}\n", .{err});
                                };
                                break;
                            }
                        }
                    }
                },
                c.SDL_MOUSEMOTION => {
                    mouse_x = event.motion.x;
                    mouse_y = event.motion.y;
                },
                else => {},
            }
        }

        // Update button hover states
        demo.updateButtons(mouse_x, mouse_y);

        // Render
        demo.render(renderer, window) catch |err| {
            std.debug.print("Error rendering: {}\n", .{err});
        };

        // Small delay to limit CPU usage
        std.Thread.sleep(16_666_667); // ~60 FPS
    }

    // Final stats
    std.debug.print("\nFinal Stats:\n", .{});
    std.debug.print("  Total allocations made: {d}\n", .{tracking.getAllocationCount()});
    std.debug.print("  Total bytes allocated: {d}\n", .{tracking.getTotalBytesAllocated()});
    std.debug.print("  Total bytes freed: {d}\n", .{tracking.getTotalBytesFreed()});
    std.debug.print("  Final usage: {d} bytes\n", .{tracking.getCurrentBytesAllocated()});

    // Generate final SVG
    try tracking.writeSvg("interactive_final.svg");
    std.debug.print("Final visualization saved to interactive_final.svg\n", .{});
}