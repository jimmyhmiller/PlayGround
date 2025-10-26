const std = @import("std");
const rl = @import("raylib");
const ts = @import("../types/type_system.zig");

/// Convert our Color to raylib Color
pub fn toRaylibColor(color: ts.Color) rl.Color {
    return rl.Color{
        .r = color.r,
        .g = color.g,
        .b = color.b,
        .a = color.a,
    };
}

/// Convert our Vec2 to raylib Vector2
pub fn toRaylibVec2(vec: ts.Vec2) rl.Vector2 {
    return rl.Vector2{
        .x = vec.x,
        .y = vec.y,
    };
}

/// Draw a rounded rectangle with outline
pub fn drawRoundedBox(pos: ts.Vec2, width: f32, height: f32, color: ts.Color, outline_color: ts.Color, roundness: f32) void {
    const rect = rl.Rectangle{
        .x = pos.x,
        .y = pos.y,
        .width = width,
        .height = height,
    };

    rl.drawRectangleRounded(rect, roundness, 10, toRaylibColor(color));
    rl.drawRectangleRoundedLines(rect, roundness, 10, toRaylibColor(outline_color));
}

/// Draw a rounded rectangle with double border (for enums/sum types)
pub fn drawDoubleBorderedBox(pos: ts.Vec2, width: f32, height: f32, color: ts.Color, outline_color: ts.Color, roundness: f32) void {
    const rect = rl.Rectangle{
        .x = pos.x,
        .y = pos.y,
        .width = width,
        .height = height,
    };

    // Draw filled box
    rl.drawRectangleRounded(rect, roundness, 10, toRaylibColor(color));

    // Draw outer border
    rl.drawRectangleRoundedLines(rect, roundness, 10, toRaylibColor(outline_color));

    // Draw inner border (2px inset)
    const inner_rect = rl.Rectangle{
        .x = pos.x + 3,
        .y = pos.y + 3,
        .width = width - 6,
        .height = height - 6,
    };
    const inner_outline = outline_color.withAlpha(@as(u8, @intFromFloat(@as(f32, @floatFromInt(outline_color.a)) * 0.5)));
    rl.drawRectangleRoundedLines(inner_rect, roundness, 10, toRaylibColor(inner_outline));
}

/// Draw text centered in a box with custom font
pub fn drawCenteredText(text: [:0]const u8, pos: ts.Vec2, width: f32, height: f32, font_size: f32, color: ts.Color, font: rl.Font) void {
    const text_vec = rl.measureTextEx(font, text, font_size, 0.5);
    const text_x = pos.x + (width - text_vec.x) / 2.0;
    const text_y = pos.y + (height - font_size) / 2.0;

    rl.drawTextEx(font, text, rl.Vector2{ .x = text_x, .y = text_y }, font_size, 0.5, toRaylibColor(color));
}

/// Draw text at position with custom font
pub fn drawText(text: [:0]const u8, x: f32, y: f32, font_size: f32, color: ts.Color, font: rl.Font) void {
    rl.drawTextEx(font, text, rl.Vector2{ .x = x, .y = y }, font_size, 0.5, toRaylibColor(color));
}

/// Draw a connecting line with bezier curve
pub fn drawConnection(start: ts.Vec2, end: ts.Vec2, color: ts.Color, thickness: f32) void {
    // Draw bezier curve
    rl.drawLineBezier(
        toRaylibVec2(start),
        toRaylibVec2(end),
        thickness,
        toRaylibColor(color),
    );

    // Draw arrowhead at end
    const arrow_size = 8.0;
    const angle = std.math.atan2(end.y - start.y, end.x - start.x);
    const arrow_angle = 0.4; // radians

    const p1 = ts.Vec2.init(
        end.x - arrow_size * @cos(angle - arrow_angle),
        end.y - arrow_size * @sin(angle - arrow_angle),
    );
    const p2 = ts.Vec2.init(
        end.x - arrow_size * @cos(angle + arrow_angle),
        end.y - arrow_size * @sin(angle + arrow_angle),
    );

    rl.drawTriangle(
        toRaylibVec2(end),
        toRaylibVec2(p1),
        toRaylibVec2(p2),
        toRaylibColor(color),
    );
}

/// Draw a circle (for tuple elements, connection points)
pub fn drawCircle(center: ts.Vec2, radius: f32, color: ts.Color, outline_color: ts.Color) void {
    rl.drawCircleV(toRaylibVec2(center), radius, toRaylibColor(color));
    rl.drawCircleLinesV(toRaylibVec2(center), radius, toRaylibColor(outline_color));
}

/// Draw a tree branch (for enum variants)
pub fn drawBranch(start: ts.Vec2, end: ts.Vec2, color: ts.Color, thickness: f32) void {
    // Draw a tapered line (trunk to branch)
    const steps = 20;
    var i: i32 = 0;
    while (i < steps) : (i += 1) {
        const t = @as(f32, @floatFromInt(i)) / @as(f32, @floatFromInt(steps));
        const next_t = @as(f32, @floatFromInt(i + 1)) / @as(f32, @floatFromInt(steps));

        const x1 = start.x + (end.x - start.x) * t;
        const y1 = start.y + (end.y - start.y) * t;
        const x2 = start.x + (end.x - start.x) * next_t;
        const y2 = start.y + (end.y - start.y) * next_t;

        const current_thickness = thickness * (1.0 - t * 0.5);

        rl.drawLineEx(
            rl.Vector2{ .x = x1, .y = y1 },
            rl.Vector2{ .x = x2, .y = y2 },
            current_thickness,
            toRaylibColor(color),
        );
    }
}

/// Draw a ghost/translucent box for optional types
pub fn drawGhostBox(pos: ts.Vec2, width: f32, height: f32, color: ts.Color) void {
    const ghost_color = ts.Color{
        .r = color.r,
        .g = color.g,
        .b = color.b,
        .a = 80, // Very translucent
    };

    const rect = rl.Rectangle{
        .x = pos.x,
        .y = pos.y,
        .width = width,
        .height = height,
    };

    // Draw dashed outline
    const dash_length: f32 = 10;
    const gap_length: f32 = 5;

    // Top
    var x = pos.x;
    while (x < pos.x + width) : (x += dash_length + gap_length) {
        const end_x = @min(x + dash_length, pos.x + width);
        rl.drawLineEx(
            rl.Vector2{ .x = x, .y = pos.y },
            rl.Vector2{ .x = end_x, .y = pos.y },
            2,
            toRaylibColor(color),
        );
    }

    // Bottom
    x = pos.x;
    while (x < pos.x + width) : (x += dash_length + gap_length) {
        const end_x = @min(x + dash_length, pos.x + width);
        rl.drawLineEx(
            rl.Vector2{ .x = x, .y = pos.y + height },
            rl.Vector2{ .x = end_x, .y = pos.y + height },
            2,
            toRaylibColor(color),
        );
    }

    // Left
    var y = pos.y;
    while (y < pos.y + height) : (y += dash_length + gap_length) {
        const end_y = @min(y + dash_length, pos.y + height);
        rl.drawLineEx(
            rl.Vector2{ .x = pos.x, .y = y },
            rl.Vector2{ .x = pos.x, .y = end_y },
            2,
            toRaylibColor(color),
        );
    }

    // Right
    y = pos.y;
    while (y < pos.y + height) : (y += dash_length + gap_length) {
        const end_y = @min(y + dash_length, pos.y + height);
        rl.drawLineEx(
            rl.Vector2{ .x = pos.x + width, .y = y },
            rl.Vector2{ .x = pos.x + width, .y = end_y },
            2,
            toRaylibColor(color),
        );
    }

    rl.drawRectangleRec(rect, toRaylibColor(ghost_color));
}

/// Draw arrow for function type transformations
pub fn drawArrow(start: ts.Vec2, end: ts.Vec2, color: ts.Color, thickness: f32) void {
    // Draw line
    rl.drawLineEx(toRaylibVec2(start), toRaylibVec2(end), thickness, toRaylibColor(color));

    // Draw arrowhead
    const dx = end.x - start.x;
    const dy = end.y - start.y;
    const length = @sqrt(dx * dx + dy * dy);

    if (length > 0) {
        const norm_x = dx / length;
        const norm_y = dy / length;

        const arrow_size: f32 = 15;
        const arrow_angle: f32 = 0.5;

        // Calculate arrowhead points
        const p1_x = end.x - arrow_size * (norm_x * @cos(arrow_angle) - norm_y * @sin(arrow_angle));
        const p1_y = end.y - arrow_size * (norm_y * @cos(arrow_angle) + norm_x * @sin(arrow_angle));

        const p2_x = end.x - arrow_size * (norm_x * @cos(-arrow_angle) - norm_y * @sin(-arrow_angle));
        const p2_y = end.y - arrow_size * (norm_y * @cos(-arrow_angle) + norm_x * @sin(-arrow_angle));

        rl.drawTriangle(
            toRaylibVec2(end),
            rl.Vector2{ .x = p1_x, .y = p1_y },
            rl.Vector2{ .x = p2_x, .y = p2_y },
            toRaylibColor(color),
        );
    }
}

/// Draw a particle (for animations)
pub fn drawParticle(pos: ts.Vec2, size: f32, color: ts.Color, alpha: u8) void {
    const particle_color = ts.Color{
        .r = color.r,
        .g = color.g,
        .b = color.b,
        .a = alpha,
    };
    rl.drawCircleV(toRaylibVec2(pos), size, toRaylibColor(particle_color));
}

/// Draw grid lines for layout debugging
pub fn drawGrid(spacing: f32, color: ts.Color) void {
    const screen_width = rl.getScreenWidth();
    const screen_height = rl.getScreenHeight();

    var x: f32 = 0;
    while (x < @as(f32, @floatFromInt(screen_width))) : (x += spacing) {
        rl.drawLine(
            @intFromFloat(x),
            0,
            @intFromFloat(x),
            screen_height,
            toRaylibColor(color),
        );
    }

    var y: f32 = 0;
    while (y < @as(f32, @floatFromInt(screen_height))) : (y += spacing) {
        rl.drawLine(
            0,
            @intFromFloat(y),
            screen_width,
            @intFromFloat(y),
            toRaylibColor(color),
        );
    }
}

/// Draw a rounded shadow beneath a box (for 3D depth effect)
pub fn drawShadow(pos: ts.Vec2, width: f32, height: f32, offset: f32, blur: u8) void {
    const shadow_color = ts.Color.init(0, 0, 0, blur);
    const shadow_rect = rl.Rectangle{
        .x = pos.x + offset,
        .y = pos.y + offset,
        .width = width,
        .height = height,
    };
    rl.drawRectangleRounded(shadow_rect, 0.2, 10, toRaylibColor(shadow_color));
}

/// Draw a solid rounded box (no gradient - looks cleaner)
pub fn drawGradientBox(pos: ts.Vec2, width: f32, height: f32, top_color: ts.Color, bottom_color: ts.Color, roundness: f32) void {
    _ = bottom_color; // Not using gradient for now - solid looks better
    const rect = rl.Rectangle{
        .x = pos.x,
        .y = pos.y,
        .width = width,
        .height = height,
    };

    // Just draw a solid color with the top color
    rl.drawRectangleRounded(rect, roundness, 10, toRaylibColor(top_color));

    // Draw very subtle border with reduced opacity
    const border_color = ts.Color.init(
        if (top_color.r >= 20) top_color.r - 20 else 0,
        if (top_color.g >= 20) top_color.g - 20 else 0,
        if (top_color.b >= 20) top_color.b - 20 else 0,
        @min(60, top_color.a / 2),  // Much more subtle - low opacity
    );
    rl.drawRectangleRoundedLines(rect, roundness, 10, toRaylibColor(border_color));
}

/// Draw a badge/pill shape with text
pub fn drawBadge(pos: ts.Vec2, text: [:0]const u8, font_size: f32, bg_color: ts.Color, text_color: ts.Color, font: rl.Font) void {
    const text_vec = rl.measureTextEx(font, text, font_size, 0.5);
    const padding: f32 = 8.0;
    const width = text_vec.x + padding * 2.0;
    const height = font_size + padding;

    const rect = rl.Rectangle{
        .x = pos.x,
        .y = pos.y,
        .width = width,
        .height = height,
    };

    rl.drawRectangleRounded(rect, 0.5, 10, toRaylibColor(bg_color));

    const text_x = pos.x + padding;
    const text_y = pos.y + padding / 2.0;
    rl.drawTextEx(font, text, rl.Vector2{ .x = text_x, .y = text_y }, font_size, 0.5, toRaylibColor(text_color));
}

/// Draw a simple icon glyph (using Unicode characters)
pub fn drawIconGlyph(pos: ts.Vec2, glyph: [:0]const u8, size: f32, color: ts.Color, font: rl.Font) void {
    rl.drawTextEx(font, glyph, rl.Vector2{ .x = pos.x, .y = pos.y }, size, 0.5, toRaylibColor(color));
}

/// Draw a glow effect (multiple circles with decreasing opacity)
pub fn drawGlow(center: ts.Vec2, radius: f32, color: ts.Color) void {
    const layers = 5;
    var i: i32 = layers;
    while (i > 0) : (i -= 1) {
        const layer_radius = radius * (@as(f32, @floatFromInt(i)) / @as(f32, @floatFromInt(layers)));
        const layer_alpha = color.a / @as(u8, @intCast(i + 1));
        const layer_color = ts.Color.init(color.r, color.g, color.b, layer_alpha);
        rl.drawCircleV(toRaylibVec2(center), layer_radius, toRaylibColor(layer_color));
    }
}

/// Draw a bezier curve with multiple control points
pub fn drawSmoothCurve(start: ts.Vec2, end: ts.Vec2, color: ts.Color, thickness: f32) void {
    // Create control points for a smooth S-curve
    const mid_x = (start.x + end.x) / 2.0;
    const control1 = rl.Vector2{ .x = mid_x, .y = start.y };
    const control2 = rl.Vector2{ .x = mid_x, .y = end.y };

    // Draw the curve using line segments
    const segments = 30;
    var i: i32 = 0;
    while (i < segments) : (i += 1) {
        const t1 = @as(f32, @floatFromInt(i)) / @as(f32, @floatFromInt(segments));
        const t2 = @as(f32, @floatFromInt(i + 1)) / @as(f32, @floatFromInt(segments));

        const p1 = cubicBezier(toRaylibVec2(start), control1, control2, toRaylibVec2(end), t1);
        const p2 = cubicBezier(toRaylibVec2(start), control1, control2, toRaylibVec2(end), t2);

        rl.drawLineEx(p1, p2, thickness, toRaylibColor(color));
    }
}

/// Cubic bezier calculation
fn cubicBezier(p0: rl.Vector2, p1: rl.Vector2, p2: rl.Vector2, p3: rl.Vector2, t: f32) rl.Vector2 {
    const u = 1.0 - t;
    const tt = t * t;
    const uu = u * u;
    const uuu = uu * u;
    const ttt = tt * t;

    return rl.Vector2{
        .x = uuu * p0.x + 3.0 * uu * t * p1.x + 3.0 * u * tt * p2.x + ttt * p3.x,
        .y = uuu * p0.y + 3.0 * uu * t * p1.y + 3.0 * u * tt * p2.y + ttt * p3.y,
    };
}

/// Draw an animated dashed line (rotating dash pattern)
pub fn drawDashedLineAnimated(start: ts.Vec2, end: ts.Vec2, color: ts.Color, thickness: f32, dash_length: f32, gap_length: f32, offset: f32) void {
    const dx = end.x - start.x;
    const dy = end.y - start.y;
    const total_length = @sqrt(dx * dx + dy * dy);
    const pattern_length = dash_length + gap_length;

    var current_length: f32 = -offset;
    while (current_length < total_length) : (current_length += pattern_length) {
        const dash_start = @max(0.0, current_length);
        const dash_end = @min(total_length, current_length + dash_length);

        if (dash_start < dash_end) {
            const t_start = dash_start / total_length;
            const t_end = dash_end / total_length;

            const x1 = start.x + dx * t_start;
            const y1 = start.y + dy * t_start;
            const x2 = start.x + dx * t_end;
            const y2 = start.y + dy * t_end;

            rl.drawLineEx(
                rl.Vector2{ .x = x1, .y = y1 },
                rl.Vector2{ .x = x2, .y = y2 },
                thickness,
                toRaylibColor(color),
            );
        }
    }
}
