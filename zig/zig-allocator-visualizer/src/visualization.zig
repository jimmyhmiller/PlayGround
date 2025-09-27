const std = @import("std");
const AllocationInfo = @import("allocation_tracker.zig").AllocationInfo;
const AllocationTracker = @import("allocation_tracker.zig").AllocationTracker;

pub const IterationType = enum {
    MemoryOrder,
    ByType,
    ByTime,
    BySize,
};

pub fn Iterator(comptime iter_type: IterationType) type {
    return struct {
        tracker: *AllocationTracker,
        index: usize,
        items: []AllocationInfo,
        allocator: std.mem.Allocator,

        const Self = @This();

        pub fn init(tracker: *AllocationTracker) Self {
            var self = Self{
                .tracker = tracker,
                .index = 0,
                .items = undefined,
                .allocator = tracker.allocator,
            };

            switch (iter_type) {
                .MemoryOrder => {
                    self.items = tracker.memory_ordered.items;
                },
                .ByType => {
                    self.items = self.gatherByType() catch &.{};
                },
                .ByTime => {
                    self.items = self.sortByTime() catch &.{};
                },
                .BySize => {
                    self.items = self.sortBySize() catch &.{};
                },
            }

            return self;
        }

        pub fn next(self: *Self) ?AllocationInfo {
            while (self.index < self.items.len) {
                const item = self.items[self.index];
                self.index += 1;
                return item;
            }
            return null;
        }

        pub fn reset(self: *Self) void {
            self.index = 0;
        }

        pub fn count(self: *Self) usize {
            return self.items.len;
        }

        fn gatherByType(self: *Self) ![]AllocationInfo {
            var result: std.ArrayList(AllocationInfo) = .{};

            var type_iter = self.tracker.type_index.iterator();
            while (type_iter.next()) |entry| {
                for (entry.value_ptr.items) |info| {
                    try result.append(self.allocator, info);
                }
            }

            return result.toOwnedSlice(self.allocator);
        }

        fn sortByTime(self: *Self) ![]AllocationInfo {
            var result: std.ArrayList(AllocationInfo) = .{};

            for (self.tracker.memory_ordered.items) |info| {
                try result.append(self.allocator, info);
            }

            std.sort.pdq(AllocationInfo, result.items, {}, compareByTime);
            return result.toOwnedSlice(self.allocator);
        }

        fn sortBySize(self: *Self) ![]AllocationInfo {
            var result: std.ArrayList(AllocationInfo) = .{};

            for (self.tracker.memory_ordered.items) |info| {
                try result.append(self.allocator, info);
            }

            std.sort.pdq(AllocationInfo, result.items, {}, compareBySize);
            return result.toOwnedSlice(self.allocator);
        }

        fn compareByTime(context: void, a: AllocationInfo, b: AllocationInfo) bool {
            _ = context;
            return a.timestamp < b.timestamp;
        }

        fn compareBySize(context: void, a: AllocationInfo, b: AllocationInfo) bool {
            _ = context;
            return a.size > b.size;
        }
    };
}

pub const Visualizer = struct {
    pub const VTable = struct {
        renderAllocation: *const fn (self: *anyopaque, info: AllocationInfo) anyerror!void,
        beginRender: *const fn (self: *anyopaque) anyerror!void,
        endRender: *const fn (self: *anyopaque) anyerror!void,
    };

    ptr: *anyopaque,
    vtable: *const VTable,

    pub fn renderAllocation(self: Visualizer, info: AllocationInfo) !void {
        return self.vtable.renderAllocation(self.ptr, info);
    }

    pub fn beginRender(self: Visualizer) !void {
        return self.vtable.beginRender(self.ptr);
    }

    pub fn endRender(self: Visualizer) !void {
        return self.vtable.endRender(self.ptr);
    }
};

pub const MemoryMapView = struct {
    min_address: usize,
    max_address: usize,
    width: u32,
    height: u32,
    scale: f32,

    pub fn init(tracker: *AllocationTracker, width: u32, height: u32) MemoryMapView {
        var min_addr: usize = std.math.maxInt(usize);
        var max_addr: usize = 0;

        for (tracker.memory_ordered.items) |info| {

            const addr = @intFromPtr(info.ptr);
            const end_addr = @intFromPtr(info.getEndPtr());

            if (addr < min_addr) min_addr = addr;
            if (end_addr > max_addr) max_addr = end_addr;
        }

        const address_range = if (max_addr > min_addr) max_addr - min_addr else 1;
        const scale = @as(f32, @floatFromInt(width)) / @as(f32, @floatFromInt(address_range));

        return .{
            .min_address = min_addr,
            .max_address = max_addr,
            .width = width,
            .height = height,
            .scale = scale,
        };
    }

    pub fn mapAddressToX(self: MemoryMapView, addr: usize) u32 {
        const offset = addr - self.min_address;
        return @intFromFloat(@as(f32, @floatFromInt(offset)) * self.scale);
    }

    pub fn mapSizeToWidth(self: MemoryMapView, size: usize) u32 {
        return @max(1, @as(u32, @intFromFloat(@as(f32, @floatFromInt(size)) * self.scale)));
    }
};