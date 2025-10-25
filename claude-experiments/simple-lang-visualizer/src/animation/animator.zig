const std = @import("std");
const easing = @import("easing.zig");

/// Type of value being animated
pub const AnimatedValue = union(enum) {
    float: *f32,
    vec2: *Vec2,
    color: *Color,

    pub const Vec2 = struct {
        x: f32,
        y: f32,
    };

    pub const Color = struct {
        r: u8,
        g: u8,
        b: u8,
        a: u8,
    };
};

/// Single animation tween
pub const Animation = struct {
    target: AnimatedValue,
    start_value: union(enum) {
        float: f32,
        vec2: AnimatedValue.Vec2,
        color: AnimatedValue.Color,
    },
    end_value: union(enum) {
        float: f32,
        vec2: AnimatedValue.Vec2,
        color: AnimatedValue.Color,
    },
    duration: f32,
    elapsed: f32,
    easing_fn: easing.EasingFunction,
    finished: bool,

    /// Update the animation
    pub fn update(self: *Animation, delta_time: f32) void {
        if (self.finished) return;

        self.elapsed += delta_time;
        const t = @min(self.elapsed / self.duration, 1.0);
        const eased = self.easing_fn(t);

        switch (self.target) {
            .float => |ptr| {
                const start = self.start_value.float;
                const end = self.end_value.float;
                ptr.* = start + (end - start) * eased;
            },
            .vec2 => |ptr| {
                const start = self.start_value.vec2;
                const end = self.end_value.vec2;
                ptr.x = start.x + (end.x - start.x) * eased;
                ptr.y = start.y + (end.y - start.y) * eased;
            },
            .color => |ptr| {
                const start = self.start_value.color;
                const end = self.end_value.color;
                ptr.r = @intFromFloat(@as(f32, @floatFromInt(start.r)) + (@as(f32, @floatFromInt(end.r)) - @as(f32, @floatFromInt(start.r))) * eased);
                ptr.g = @intFromFloat(@as(f32, @floatFromInt(start.g)) + (@as(f32, @floatFromInt(end.g)) - @as(f32, @floatFromInt(start.g))) * eased);
                ptr.b = @intFromFloat(@as(f32, @floatFromInt(start.b)) + (@as(f32, @floatFromInt(end.b)) - @as(f32, @floatFromInt(start.b))) * eased);
                ptr.a = @intFromFloat(@as(f32, @floatFromInt(start.a)) + (@as(f32, @floatFromInt(end.a)) - @as(f32, @floatFromInt(start.a))) * eased);
            },
        }

        if (t >= 1.0) {
            self.finished = true;
        }
    }
};

/// Animation manager
pub const Animator = struct {
    animations: std.ArrayList(Animation),
    allocator: std.mem.Allocator,

    pub fn init(allocator: std.mem.Allocator) Animator {
        return .{
            .animations = std.ArrayList(Animation){},
            .allocator = allocator,
        };
    }

    pub fn deinit(self: *Animator) void {
        self.animations.deinit(self.allocator);
    }

    /// Add a new float animation
    pub fn animateFloat(
        self: *Animator,
        target: *f32,
        end_value: f32,
        duration: f32,
        easing_fn: easing.EasingFunction,
    ) !void {
        const anim = Animation{
            .target = .{ .float = target },
            .start_value = .{ .float = target.* },
            .end_value = .{ .float = end_value },
            .duration = duration,
            .elapsed = 0.0,
            .easing_fn = easing_fn,
            .finished = false,
        };
        try self.animations.append(self.allocator, anim);
    }

    /// Add a new Vec2 animation
    pub fn animateVec2(
        self: *Animator,
        target: *AnimatedValue.Vec2,
        end_value: AnimatedValue.Vec2,
        duration: f32,
        easing_fn: easing.EasingFunction,
    ) !void {
        const anim = Animation{
            .target = .{ .vec2 = target },
            .start_value = .{ .vec2 = target.* },
            .end_value = .{ .vec2 = end_value },
            .duration = duration,
            .elapsed = 0.0,
            .easing_fn = easing_fn,
            .finished = false,
        };
        try self.animations.append(self.allocator, anim);
    }

    /// Add a new Color animation
    pub fn animateColor(
        self: *Animator,
        target: *AnimatedValue.Color,
        end_value: AnimatedValue.Color,
        duration: f32,
        easing_fn: easing.EasingFunction,
    ) !void {
        const anim = Animation{
            .target = .{ .color = target },
            .start_value = .{ .color = target.* },
            .end_value = .{ .color = end_value },
            .duration = duration,
            .elapsed = 0.0,
            .easing_fn = easing_fn,
            .finished = false,
        };
        try self.animations.append(self.allocator, anim);
    }

    /// Update all animations
    pub fn update(self: *Animator, delta_time: f32) void {
        for (self.animations.items) |*anim| {
            anim.update(delta_time);
        }

        // Remove finished animations
        var i: usize = 0;
        while (i < self.animations.items.len) {
            if (self.animations.items[i].finished) {
                _ = self.animations.swapRemove(i);
            } else {
                i += 1;
            }
        }
    }

    /// Check if any animations are running
    pub fn hasActiveAnimations(self: *const Animator) bool {
        return self.animations.items.len > 0;
    }

    /// Clear all animations
    pub fn clear(self: *Animator) void {
        self.animations.clearRetainingCapacity();
    }
};
