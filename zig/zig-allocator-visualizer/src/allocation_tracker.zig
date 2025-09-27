const std = @import("std");

pub const AllocationInfo = struct {
    ptr: [*]u8,
    size: usize,
    alignment: u8,
    timestamp: i64,
    thread_id: std.Thread.Id,
    type_name: []const u8,
    stack_trace: ?[]usize,

    pub fn getEndPtr(self: AllocationInfo) [*]u8 {
        return self.ptr + self.size;
    }

    pub fn overlaps(self: AllocationInfo, other: AllocationInfo) bool {
        const self_start = @intFromPtr(self.ptr);
        const self_end = @intFromPtr(self.getEndPtr());
        const other_start = @intFromPtr(other.ptr);
        const other_end = @intFromPtr(other.getEndPtr());

        return !(self_end <= other_start or other_end <= self_start);
    }
};

pub const AllocationTracker = struct {
    allocator: std.mem.Allocator,
    allocations: std.hash_map.AutoHashMap(*anyopaque, AllocationInfo),
    memory_ordered: std.ArrayList(AllocationInfo),
    type_index: std.StringHashMap(std.ArrayList(AllocationInfo)),
    allocation_count: usize,
    total_bytes_allocated: usize,
    total_bytes_freed: usize,
    enable_stack_traces: bool,

    const Self = @This();

    pub fn init(allocator: std.mem.Allocator) Self {
        return .{
            .allocator = allocator,
            .allocations = std.hash_map.AutoHashMap(*anyopaque, AllocationInfo).init(allocator),
            .memory_ordered = .{},
            .type_index = std.StringHashMap(std.ArrayList(AllocationInfo)).init(allocator),
            .allocation_count = 0,
            .total_bytes_allocated = 0,
            .total_bytes_freed = 0,
            .enable_stack_traces = false,
        };
    }

    pub fn deinit(self: *Self) void {
        self.allocations.deinit();
        self.memory_ordered.deinit(self.allocator);

        var type_iter = self.type_index.iterator();
        while (type_iter.next()) |entry| {
            entry.value_ptr.deinit(self.allocator);
        }
        self.type_index.deinit();
    }

    pub fn trackAllocation(
        self: *Self,
        ptr: [*]u8,
        size: usize,
        alignment: u8,
        ret_addr: usize,
    ) !void {
        _ = ret_addr;

        const info = AllocationInfo{
            .ptr = ptr,
            .size = size,
            .alignment = alignment,
            .timestamp = std.time.milliTimestamp(),
            .thread_id = std.Thread.getCurrentId(),
            .type_name = "unknown",
            .stack_trace = null,
        };

        try self.allocations.put(@ptrCast(ptr), info);
        try self.insertOrdered(info, self.allocator);

        self.allocation_count += 1;
        self.total_bytes_allocated += size;
    }

    pub fn trackAllocationWithType(
        self: *Self,
        ptr: [*]u8,
        size: usize,
        alignment: u8,
        comptime T: type,
    ) !void {
        const info = AllocationInfo{
            .ptr = ptr,
            .size = size,
            .alignment = alignment,
            .timestamp = std.time.milliTimestamp(),
            .thread_id = std.Thread.getCurrentId(),
            .type_name = @typeName(T),
            .stack_trace = null,
        };

        try self.allocations.put(@ptrCast(ptr), info);
        try self.insertOrdered(info, self.allocator);
        try self.indexByType(info);

        self.allocation_count += 1;
        self.total_bytes_allocated += size;
    }

    pub fn updateAllocation(
        self: *Self,
        ptr: [*]u8,
        old_size: usize,
        new_size: usize,
    ) !void {
        if (self.allocations.getPtr(@ptrCast(ptr))) |info| {
            self.total_bytes_allocated += new_size;
            self.total_bytes_allocated -= old_size;
            info.size = new_size;

            // Update in ordered list
            for (self.memory_ordered.items) |*item| {
                if (item.ptr == ptr) {
                    item.size = new_size;
                    break;
                }
            }
        }
    }

    pub fn trackDeallocation(self: *Self, ptr: [*]u8, size: usize) !void {
        if (self.allocations.get(@ptrCast(ptr))) |info| {
            self.total_bytes_freed += size;

            // Remove from allocations map
            _ = self.allocations.remove(@ptrCast(ptr));

            // Remove from memory_ordered list
            var i: usize = 0;
            while (i < self.memory_ordered.items.len) {
                if (self.memory_ordered.items[i].ptr == ptr) {
                    _ = self.memory_ordered.orderedRemove(i);
                    break;
                }
                i += 1;
            }

            // Remove from type index if it exists
            if (self.type_index.getPtr(info.type_name)) |type_list| {
                var j: usize = 0;
                while (j < type_list.items.len) {
                    if (type_list.items[j].ptr == ptr) {
                        _ = type_list.orderedRemove(j);
                        break;
                    }
                    j += 1;
                }
            }
        }
    }

    fn insertOrdered(self: *Self, info: AllocationInfo, allocator: std.mem.Allocator) !void {
        const ptr_addr = @intFromPtr(info.ptr);

        // Binary search for insertion point
        var left: usize = 0;
        var right = self.memory_ordered.items.len;

        while (left < right) {
            const mid = left + (right - left) / 2;
            const mid_addr = @intFromPtr(self.memory_ordered.items[mid].ptr);

            if (mid_addr < ptr_addr) {
                left = mid + 1;
            } else {
                right = mid;
            }
        }

        try self.memory_ordered.insert(allocator, left, info);
    }

    fn indexByType(self: *Self, info: AllocationInfo) !void {
        const result = try self.type_index.getOrPut(info.type_name);
        if (!result.found_existing) {
            result.value_ptr.* = .{};
        }
        try result.value_ptr.append(self.allocator, info);
    }

    pub fn getAllocationAt(self: *Self, ptr: *anyopaque) ?AllocationInfo {
        return self.allocations.get(ptr);
    }

    pub fn getAllocationsInRange(self: *Self, start: usize, end: usize) []const AllocationInfo {
        var count: usize = 0;
        for (self.memory_ordered.items) |info| {
            const addr = @intFromPtr(info.ptr);
            if (addr >= start and addr < end) {
                count += 1;
            }
        }
        return self.memory_ordered.items[0..count];
    }

    pub fn getFragmentation(self: *Self) f64 {
        if (self.memory_ordered.items.len < 2) return 0.0;

        var total_gaps: usize = 0;
        var gap_count: usize = 0;

        for (0..self.memory_ordered.items.len - 1) |i| {
            const current = self.memory_ordered.items[i];
            const next = self.memory_ordered.items[i + 1];

            const current_end = @intFromPtr(current.getEndPtr());
            const next_start = @intFromPtr(next.ptr);

            if (next_start > current_end) {
                total_gaps += next_start - current_end;
                gap_count += 1;
            }
        }

        const total_allocated = self.total_bytes_allocated - self.total_bytes_freed;
        if (total_allocated == 0) return 0.0;

        return @as(f64, @floatFromInt(total_gaps)) / @as(f64, @floatFromInt(total_allocated));
    }

    pub fn iterate(self: *Self, comptime iter_type: @import("visualization.zig").IterationType) @import("visualization.zig").Iterator(iter_type) {
        return @import("visualization.zig").Iterator(iter_type).init(self);
    }
};