const std = @import("std");
const AllocationTracker = @import("../allocation_tracker.zig").AllocationTracker;
const AllocationInfo = @import("../allocation_tracker.zig").AllocationInfo;
const Visualization = @import("../visualization.zig");

pub const SvgRenderer = struct {
    allocator: std.mem.Allocator,
    buffer: std.ArrayList(u8),
    width: u32,
    height: u32,
    color_map: std.StringHashMap(Color),
    next_color_index: usize,

    const Self = @This();

    const Color = struct {
        r: u8,
        g: u8,
        b: u8,

        fn toHex(self: Color) [7]u8 {
            var buf: [7]u8 = undefined;
            _ = std.fmt.bufPrint(&buf, "#{x:0>2}{x:0>2}{x:0>2}", .{ self.r, self.g, self.b }) catch unreachable;
            return buf;
        }
    };

    const PALETTE = [_]Color{
        .{ .r = 0x3B, .g = 0x82, .b = 0xF6 }, // Blue
        .{ .r = 0x10, .g = 0xB9, .b = 0x81 }, // Green
        .{ .r = 0xF5, .g = 0x9E, .b = 0x0B }, // Orange
        .{ .r = 0xEF, .g = 0x44, .b = 0x44 }, // Red
        .{ .r = 0x84, .g = 0x56, .b = 0xF3 }, // Purple
        .{ .r = 0x06, .g = 0xB6, .b = 0xD4 }, // Cyan
        .{ .r = 0xF4, .g = 0x72, .b = 0xB6 }, // Pink
        .{ .r = 0x65, .g = 0xA3, .b = 0x0D }, // Lime
    };

    pub fn init(allocator: std.mem.Allocator) Self {
        return .{
            .allocator = allocator,
            .buffer = .{},
            .width = 1200,
            .height = 800,
            .color_map = std.StringHashMap(Color).init(allocator),
            .next_color_index = 0,
        };
    }

    pub fn deinit(self: *Self) void {
        self.buffer.deinit(self.allocator);
        self.color_map.deinit();
    }

    pub fn render(self: *Self, tracker: *AllocationTracker) !void {
        try self.beginSvg();
        try self.renderBackground();
        try self.renderTitle();
        try self.renderStats(tracker);
        try self.renderMemoryMap(tracker);
        try self.renderLegend();
        try self.endSvg();
    }

    fn beginSvg(self: *Self) !void {
        try self.buffer.writer(self.allocator).print(
            \\<?xml version="1.0" encoding="UTF-8"?>
            \\<svg width="{d}" height="{d}" xmlns="http://www.w3.org/2000/svg">
            \\
        , .{ self.width, self.height });
    }

    fn endSvg(self: *Self) !void {
        try self.buffer.appendSlice(self.allocator, "</svg>\n");
    }

    fn renderBackground(self: *Self) !void {
        try self.buffer.writer(self.allocator).print(
            \\  <rect x="0" y="0" width="{d}" height="{d}" fill="#1a1a1a"/>
            \\
        , .{ self.width, self.height });
    }

    fn renderTitle(self: *Self) !void {
        try self.buffer.appendSlice(self.allocator,
            \\  <text x="20" y="30" font-family="monospace" font-size="20" fill="#ffffff">
            \\    Memory Allocation Visualization
            \\  </text>
            \\
        );
    }

    fn renderStats(self: *Self, tracker: *AllocationTracker) !void {
        const total_allocated = tracker.total_bytes_allocated;
        const total_freed = tracker.total_bytes_freed;
        const current_usage = total_allocated - total_freed;
        const fragmentation = tracker.getFragmentation();

        try self.buffer.writer(self.allocator).print(
            \\  <g transform="translate(20, 60)">
            \\    <text font-family="monospace" font-size="14" fill="#888888">
            \\      <tspan x="0" dy="0">Total Allocated: {d} bytes</tspan>
            \\      <tspan x="0" dy="20">Total Freed: {d} bytes</tspan>
            \\      <tspan x="0" dy="20">Current Usage: {d} bytes</tspan>
            \\      <tspan x="0" dy="20">Fragmentation: {d:.2}%</tspan>
            \\      <tspan x="0" dy="20">Active Allocations: {d}</tspan>
            \\    </text>
            \\  </g>
            \\
        , .{
            total_allocated,
            total_freed,
            current_usage,
            fragmentation * 100,
            tracker.allocation_count,
        });
    }

    fn renderMemoryMap(self: *Self, tracker: *AllocationTracker) !void {
        const map_y_start = 200;
        const map_height = 400;
        const map_x_start = 20;
        const map_width = self.width - 40;

        // Draw border
        try self.buffer.writer(self.allocator).print(
            \\  <rect x="{d}" y="{d}" width="{d}" height="{d}" fill="none" stroke="#444444" stroke-width="2"/>
            \\
        , .{ map_x_start, map_y_start, map_width, map_height });

        // Create memory map view
        const mem_view = Visualization.MemoryMapView.init(tracker, map_width, map_height);

        // Render allocations
        const row_height = 20;
        var current_y: u32 = 0;
        var current_row_end: usize = 0;

        for (tracker.memory_ordered.items) |info| {

            const color = try self.getColorForType(info.type_name);
            const addr = @intFromPtr(info.ptr);

            // Check if we need a new row
            if (addr >= current_row_end) {
                current_y += row_height + 2;
                if (current_y >= map_height) break;
            }

            const x = map_x_start + mem_view.mapAddressToX(addr);
            const width = @max(2, mem_view.mapSizeToWidth(info.size));
            const y = map_y_start + current_y;

            current_row_end = addr + info.size;

            try self.buffer.writer(self.allocator).print(
                \\  <rect x="{d}" y="{d}" width="{d}" height="{d}" fill="{s}" opacity="0.8">
                \\    <title>Address: 0x{x}, Size: {d} bytes, Type: {s}</title>
                \\  </rect>
                \\
            , .{
                x,
                y,
                width,
                row_height,
                &color.toHex(),
                addr,
                info.size,
                info.type_name,
            });
        }
    }

    fn renderLegend(self: *Self) !void {
        const legend_x = self.width - 300;
        const legend_y = 200;

        try self.buffer.writer(self.allocator).print(
            \\  <g transform="translate({d}, {d})">
            \\    <text x="0" y="0" font-family="monospace" font-size="16" fill="#ffffff">Type Legend</text>
            \\
        , .{ legend_x, legend_y });

        var iter = self.color_map.iterator();
        var y_offset: u32 = 25;

        while (iter.next()) |entry| {
            const type_name = entry.key_ptr.*;
            const color = entry.value_ptr.*;

            try self.buffer.writer(self.allocator).print(
                \\    <rect x="0" y="{d}" width="15" height="15" fill="{s}"/>
                \\    <text x="20" y="{d}" font-family="monospace" font-size="12" fill="#cccccc">{s}</text>
                \\
            , .{
                y_offset,
                &color.toHex(),
                y_offset + 11,
                type_name,
            });

            y_offset += 20;
        }

        try self.buffer.appendSlice(self.allocator,"  </g>\n");
    }

    fn getColorForType(self: *Self, type_name: []const u8) !Color {
        const result = try self.color_map.getOrPut(type_name);
        if (!result.found_existing) {
            result.value_ptr.* = PALETTE[self.next_color_index % PALETTE.len];
            self.next_color_index += 1;
        }
        return result.value_ptr.*;
    }

    pub fn writeToFile(self: *Self, path: []const u8) !void {
        const file = try std.fs.cwd().createFile(path, .{});
        defer file.close();
        try file.writeAll(self.buffer.items);
    }

    pub fn getContent(self: *Self) []const u8 {
        return self.buffer.items;
    }
};