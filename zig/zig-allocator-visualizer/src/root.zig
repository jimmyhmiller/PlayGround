const std = @import("std");

pub const TrackingAllocator = @import("tracking_allocator.zig").TrackingAllocator;
pub const AllocationTracker = @import("allocation_tracker.zig").AllocationTracker;
pub const AllocationInfo = @import("allocation_tracker.zig").AllocationInfo;
pub const Visualization = @import("visualization.zig");
pub const SvgRenderer = @import("renderers/svg.zig").SvgRenderer;
pub const SdlRenderer = @import("renderers/sdl.zig").SdlRenderer;

pub fn bufferedPrint() !void {
    std.debug.print("Zig Allocator Visualizer loaded!\n", .{});
}