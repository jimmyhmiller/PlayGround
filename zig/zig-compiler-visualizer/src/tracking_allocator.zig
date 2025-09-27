const std = @import("std");
const AllocationTracker = @import("allocation_tracker.zig").AllocationTracker;
const Visualization = @import("visualization.zig");
const SvgRenderer = @import("renderers/svg.zig").SvgRenderer;

pub const TrackingAllocator = struct {
    backing_allocator: std.mem.Allocator,
    tracker: AllocationTracker,
    mutex: std.Thread.Mutex,
    vtable: std.mem.Allocator.VTable,

    const Self = @This();

    pub fn init(backing_allocator: std.mem.Allocator) Self {
        const self = Self{
            .backing_allocator = backing_allocator,
            .tracker = AllocationTracker.init(backing_allocator),
            .mutex = .{},
            .vtable = .{
                .alloc = alloc,
                .resize = resize,
                .free = free,
                .remap = std.mem.Allocator.noRemap,
            },
        };
        return self;
    }

    pub fn deinit(self: *Self) void {
        self.mutex.lock();
        defer self.mutex.unlock();
        self.tracker.deinit();
    }

    pub fn allocator(self: *Self) std.mem.Allocator {
        return .{
            .ptr = self,
            .vtable = &self.vtable,
        };
    }

    fn alloc(ctx: *anyopaque, len: usize, ptr_align: std.mem.Alignment, ret_addr: usize) ?[*]u8 {
        const self: *Self = @ptrCast(@alignCast(ctx));

        self.mutex.lock();
        defer self.mutex.unlock();

        const result = self.backing_allocator.rawAlloc(len, ptr_align, ret_addr);
        if (result) |ptr| {
            self.tracker.trackAllocation(ptr, len, @intFromEnum(ptr_align), ret_addr) catch {
                // If we can't track, still return the allocation
                // but log the error in debug builds
                if (std.debug.runtime_safety) {
                    std.debug.print("Failed to track allocation at {*}\n", .{ptr});
                }
            };
        }
        return result;
    }

    fn resize(ctx: *anyopaque, buf: []u8, buf_align: std.mem.Alignment, new_len: usize, ret_addr: usize) bool {
        const self: *Self = @ptrCast(@alignCast(ctx));

        self.mutex.lock();
        defer self.mutex.unlock();

        const result = self.backing_allocator.rawResize(buf, buf_align, new_len, ret_addr);
        if (result) {
            self.tracker.updateAllocation(buf.ptr, buf.len, new_len) catch {
                if (std.debug.runtime_safety) {
                    std.debug.print("Failed to update allocation tracking for resize at {*}\n", .{buf.ptr});
                }
            };
        }
        return result;
    }

    fn free(ctx: *anyopaque, buf: []u8, buf_align: std.mem.Alignment, ret_addr: usize) void {
        const self: *Self = @ptrCast(@alignCast(ctx));

        self.mutex.lock();
        defer self.mutex.unlock();

        self.tracker.trackDeallocation(buf.ptr, buf.len) catch {
            if (std.debug.runtime_safety) {
                std.debug.print("Failed to track deallocation at {*}\n", .{buf.ptr});
            }
        };

        self.backing_allocator.rawFree(buf, buf_align, ret_addr);
    }

    pub fn getTracker(self: *Self) *AllocationTracker {
        return &self.tracker;
    }

    pub fn writeSvg(self: *Self, path: []const u8) !void {
        self.mutex.lock();
        defer self.mutex.unlock();

        var renderer = SvgRenderer.init(self.backing_allocator);
        defer renderer.deinit();

        try renderer.render(&self.tracker);
        try renderer.writeToFile(path);
    }

    pub fn iterateAllocations(self: *Self, comptime IterType: Visualization.IterationType) Visualization.Iterator(IterType) {
        return self.tracker.iterate(IterType);
    }

    pub fn getAllocationCount(self: *Self) usize {
        self.mutex.lock();
        defer self.mutex.unlock();
        return self.tracker.allocation_count;
    }

    pub fn getTotalBytesAllocated(self: *Self) usize {
        self.mutex.lock();
        defer self.mutex.unlock();
        return self.tracker.total_bytes_allocated;
    }

    pub fn getTotalBytesFreed(self: *Self) usize {
        self.mutex.lock();
        defer self.mutex.unlock();
        return self.tracker.total_bytes_freed;
    }

    pub fn getCurrentBytesAllocated(self: *Self) usize {
        self.mutex.lock();
        defer self.mutex.unlock();
        return self.tracker.total_bytes_allocated - self.tracker.total_bytes_freed;
    }
};