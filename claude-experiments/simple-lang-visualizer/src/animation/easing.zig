const std = @import("std");

/// Easing functions for smooth animations
/// All functions take a normalized time t (0.0 to 1.0) and return a normalized value

/// Linear interpolation - constant speed
pub fn linear(t: f32) f32 {
    return t;
}

/// Ease in (slow start)
pub fn easeIn(t: f32) f32 {
    return t * t;
}

/// Ease out (slow end)
pub fn easeOut(t: f32) f32 {
    return t * (2.0 - t);
}

/// Ease in-out (slow start and end)
pub fn easeInOut(t: f32) f32 {
    if (t < 0.5) {
        return 2.0 * t * t;
    } else {
        const t2 = t - 1.0;
        return 1.0 - 2.0 * t2 * t2;
    }
}

/// Cubic ease in
pub fn cubicIn(t: f32) f32 {
    return t * t * t;
}

/// Cubic ease out
pub fn cubicOut(t: f32) f32 {
    const t2 = t - 1.0;
    return 1.0 + t2 * t2 * t2;
}

/// Cubic ease in-out
pub fn cubicInOut(t: f32) f32 {
    if (t < 0.5) {
        return 4.0 * t * t * t;
    } else {
        const t2 = t - 1.0;
        return 1.0 + 4.0 * t2 * t2 * t2;
    }
}

/// Elastic ease out (bounce effect)
pub fn elasticOut(t: f32) f32 {
    if (t == 0.0 or t == 1.0) return t;
    const p: f32 = 0.3;
    const s: f32 = p / 4.0;
    return @exp(-10.0 * t) * @sin((t - s) * (2.0 * std.math.pi) / p) + 1.0;
}

/// Bounce ease out
pub fn bounceOut(t: f32) f32 {
    if (t < 1.0 / 2.75) {
        return 7.5625 * t * t;
    } else if (t < 2.0 / 2.75) {
        const t2 = t - 1.5 / 2.75;
        return 7.5625 * t2 * t2 + 0.75;
    } else if (t < 2.5 / 2.75) {
        const t2 = t - 2.25 / 2.75;
        return 7.5625 * t2 * t2 + 0.9375;
    } else {
        const t2 = t - 2.625 / 2.75;
        return 7.5625 * t2 * t2 + 0.984375;
    }
}

/// Back ease out (overshoots then settles)
pub fn backOut(t: f32) f32 {
    const c1: f32 = 1.70158;
    const c3 = c1 + 1.0;
    const t2 = t - 1.0;
    return 1.0 + c3 * t2 * t2 * t2 + c1 * t2 * t2;
}

/// EasingFunction type for function pointers
pub const EasingFunction = *const fn (f32) f32;
