const std = @import("std");
const rl = @import("raylib");
const ts = @import("../types/type_system.zig");
const prim = @import("primitives.zig");
const util = @import("../util.zig");
const theme = @import("theme.zig");

/// Constants for rendering
pub const RenderConstants = struct {
    pub const box_width: f32 = 200;
    pub const box_height: f32 = 60;
    pub const field_height: f32 = 30;
    pub const padding: f32 = 10;
    pub const roundness: f32 = 0.2;
    pub const branch_length: f32 = 100;
    pub const branch_angle: f32 = 0.5; // radians
};

/// Render a primitive type
fn renderPrimitive(typ: *const ts.Type, pos: ts.Vec2, font: rl.Font, current_theme: *const theme.Theme) void {
    const anim = typ.metadata.animation;
    const width = RenderConstants.box_width * anim.scale;
    const height = RenderConstants.box_height * anim.scale;

    // Center the scaled box
    const scaled_pos = ts.Vec2.init(
        pos.x + (RenderConstants.box_width - width) / 2.0,
        pos.y + (RenderConstants.box_height - height) / 2.0,
    );

    const colors = current_theme.colorsForType(.primitive);
    const alpha = @as(u8, @intFromFloat(255.0 * anim.opacity));

    // Draw subtle shadow for depth
    if (anim.opacity > 0.5) {
        prim.drawShadow(scaled_pos, width, height, 2.0, @intFromFloat(15.0 * anim.opacity));
    }

    // Draw gradient box
    const top_color = colors.primary.withAlpha(@as(u8, @intFromFloat(@as(f32, @floatFromInt(colors.primary.a)) * anim.opacity)));
    const bottom_color = colors.secondary.withAlpha(@as(u8, @intFromFloat(@as(f32, @floatFromInt(colors.secondary.a)) * anim.opacity)));
    prim.drawGradientBox(scaled_pos, width, height, top_color, bottom_color, RenderConstants.roundness);

    // Draw type name
    const name_z = std.heap.page_allocator.dupeZ(u8, typ.name) catch return;
    defer std.heap.page_allocator.free(name_z);

    const text_color = colors.text_header.withAlpha(alpha);
    prim.drawCenteredText(
        name_z,
        scaled_pos,
        width,
        height,
        20 * anim.scale,
        text_color,
        font,
    );
}

/// Render a struct type
fn renderStruct(typ: *const ts.Type, pos: ts.Vec2, font: rl.Font, current_theme: *const theme.Theme) void {
    const struct_def = typ.definition.@"struct";
    const anim = typ.metadata.animation;
    const width = RenderConstants.box_width * anim.scale;
    const header_height = RenderConstants.box_height * anim.scale;
    const field_height = RenderConstants.field_height * anim.scale;
    const total_height = header_height + @as(f32, @floatFromInt(struct_def.fields.len)) * field_height;

    // Center the scaled box
    const base_width = RenderConstants.box_width;
    const base_height = RenderConstants.box_height + @as(f32, @floatFromInt(struct_def.fields.len)) * RenderConstants.field_height;
    const scaled_pos = ts.Vec2.init(
        pos.x + (base_width - width) / 2.0,
        pos.y + (base_height - total_height) / 2.0,
    );

    const colors = current_theme.colorsForType(.@"struct");
    const alpha = @as(u8, @intFromFloat(255.0 * anim.opacity));
    const bg_color = colors.secondary.withAlpha(@min(alpha, colors.secondary.a));
    const outline_color = colors.primary.withAlpha(alpha);

    // Draw subtle shadow
    if (anim.opacity > 0.5) {
        prim.drawShadow(scaled_pos, width, total_height, 2.0, @intFromFloat(15.0 * anim.opacity));
    }

    // Draw main container
    prim.drawRoundedBox(scaled_pos, width, total_height, bg_color, outline_color, RenderConstants.roundness);

    // Draw header with gradient
    const header_top = colors.primary.withAlpha(alpha);
    const header_bottom = colors.accent.withAlpha(alpha);
    prim.drawGradientBox(scaled_pos, width, header_height, header_top, header_bottom, RenderConstants.roundness);

    // Draw struct name in header
    const name_z = std.heap.page_allocator.dupeZ(u8, typ.name) catch return;
    defer std.heap.page_allocator.free(name_z);

    const text_color = colors.text_header.withAlpha(alpha);
    prim.drawCenteredText(
        name_z,
        scaled_pos,
        width,
        header_height,
        20 * anim.scale,
        text_color,
        font,
    );

    // Draw fields
    var i: usize = 0;
    while (i < struct_def.fields.len) : (i += 1) {
        const field = @constCast(struct_def.fields.at(i));
        const field_anim = field.animation;
        const field_y = scaled_pos.y + header_height + @as(f32, @floatFromInt(i)) * field_height;
        const field_pos = ts.Vec2.init(scaled_pos.x + RenderConstants.padding * anim.scale, field_y + RenderConstants.padding * anim.scale);

        // Only draw if field is visible (has started appearing)
        if (field_anim.opacity > 0.0) {
            const field_alpha = @as(u8, @intFromFloat(255.0 * field_anim.opacity * anim.opacity));

            // Draw field name and type
            const field_text = util.allocPrintZ(
                std.heap.page_allocator,
                "{s}: {s}",
                .{ field.name, field.type_ref },
            ) catch return;
            defer std.heap.page_allocator.free(field_text);

            const field_text_color = colors.text.withAlpha(field_alpha);
            prim.drawText(
                field_text,
                field_pos.x,
                field_pos.y,
                16 * anim.scale * field_anim.scale,
                field_text_color,
                font,
            );
        }

        // Draw separator line
        if (i < struct_def.fields.len - 1) {
            const sep_alpha = @as(u8, @intFromFloat(@as(f32, @floatFromInt(outline_color.a)) * 0.4 * anim.opacity));
            rl.drawLineEx(
                rl.Vector2{ .x = scaled_pos.x + 10, .y = field_y + field_height },
                rl.Vector2{ .x = scaled_pos.x + width - 10, .y = field_y + field_height },
                1,
                prim.toRaylibColor(outline_color.withAlpha(sep_alpha)),
            );
        }
    }
}

/// Render an enum type - like a struct with variant rows
fn renderEnum(typ: *const ts.Type, pos: ts.Vec2, font: rl.Font, current_theme: *const theme.Theme) void {
    const enum_def = typ.definition.@"enum";
    const anim = typ.metadata.animation;

    const width = RenderConstants.box_width * 1.5; // Wider for variants
    const header_height = RenderConstants.box_height * anim.scale;
    const variant_height = RenderConstants.field_height * anim.scale;
    const total_height = header_height + @as(f32, @floatFromInt(enum_def.variants.len)) * variant_height;

    // Center the scaled box
    const base_width = RenderConstants.box_width * 1.5;
    const base_height = RenderConstants.box_height + @as(f32, @floatFromInt(enum_def.variants.len)) * RenderConstants.field_height;
    const scaled_pos = ts.Vec2.init(
        pos.x + (base_width - width) / 2.0,
        pos.y + (base_height - total_height) / 2.0,
    );

    const colors = current_theme.colorsForType(.@"enum");
    const alpha = @as(u8, @intFromFloat(255.0 * anim.opacity));

    // Draw subtle shadow
    if (anim.opacity > 0.5) {
        prim.drawShadow(scaled_pos, width, total_height, 2.0, @intFromFloat(15.0 * anim.opacity));
    }

    // Draw main container
    const bg_color = colors.secondary.withAlpha(@min(alpha, colors.secondary.a));
    const outline_color = colors.primary.withAlpha(alpha);
    prim.drawRoundedBox(scaled_pos, width, total_height, bg_color, outline_color, RenderConstants.roundness);

    // Draw header
    const header_top = colors.primary.withAlpha(alpha);
    const header_bottom = colors.accent.withAlpha(alpha);
    prim.drawGradientBox(scaled_pos, width, header_height, header_top, header_bottom, RenderConstants.roundness);

    // Draw enum name in header
    const name_z = std.heap.page_allocator.dupeZ(u8, typ.name) catch return;
    defer std.heap.page_allocator.free(name_z);

    const text_color = colors.text_header.withAlpha(alpha);
    prim.drawCenteredText(
        name_z,
        scaled_pos,
        width,
        header_height,
        20 * anim.scale,
        text_color,
        font,
    );

    // Draw variants as rows (like struct fields)
    var i: usize = 0;
    while (i < enum_def.variants.len) : (i += 1) {
        const variant = @constCast(enum_def.variants.at(i));
        const variant_anim = variant.animation;
        const variant_y = scaled_pos.y + header_height + @as(f32, @floatFromInt(i)) * variant_height;
        const variant_pos = ts.Vec2.init(scaled_pos.x + RenderConstants.padding * anim.scale, variant_y + RenderConstants.padding * anim.scale);

        // Only draw if variant is visible
        if (variant_anim.opacity > 0.0) {
            const variant_alpha = @as(u8, @intFromFloat(255.0 * variant_anim.opacity * anim.opacity));

            // Draw variant name with payload
            const variant_text = if (variant.payload) |payload|
                util.allocPrintZ(
                    std.heap.page_allocator,
                    "| {s}({s})",
                    .{ variant.name, payload },
                ) catch return
            else
                util.allocPrintZ(
                    std.heap.page_allocator,
                    "| {s}",
                    .{variant.name},
                ) catch return;
            defer std.heap.page_allocator.free(variant_text);

            const variant_text_color = colors.text.withAlpha(variant_alpha);
            prim.drawText(
                variant_text,
                variant_pos.x,
                variant_pos.y,
                16 * anim.scale * variant_anim.scale,
                variant_text_color,
                font,
            );
        }

        // Draw separator line
        if (i < enum_def.variants.len - 1) {
            const sep_alpha = @as(u8, @intFromFloat(@as(f32, @floatFromInt(outline_color.a)) * 0.4 * anim.opacity));
            rl.drawLineEx(
                rl.Vector2{ .x = scaled_pos.x + 10, .y = variant_y + variant_height },
                rl.Vector2{ .x = scaled_pos.x + width - 10, .y = variant_y + variant_height },
                1,
                prim.toRaylibColor(outline_color.withAlpha(sep_alpha)),
            );
        }
    }
}

/// Render a function type
fn renderFunction(typ: *const ts.Type, pos: ts.Vec2, font: rl.Font, current_theme: *const theme.Theme) void {
    _ = current_theme; // TODO: Use theme colors
    const func_def = typ.definition.function;

    const param_width: f32 = 100;
    const param_height: f32 = 40;
    const arrow_length: f32 = 80;
    const return_width: f32 = 100;
    const return_height: f32 = 40;

    var current_y = pos.y;

    // Draw parameters
    var i: usize = 0;
    while (i < func_def.parameters.len) : (i += 1) {
        const param = @constCast(func_def.parameters.at(i));
        const param_pos = ts.Vec2.init(pos.x, current_y);
        const param_color = typ.metadata.color.withAlpha(120);

        prim.drawRoundedBox(
            param_pos,
            param_width,
            param_height,
            param_color,
            typ.metadata.color,
            RenderConstants.roundness,
        );

        const param_text = util.allocPrintZ(
            std.heap.page_allocator,
            "{s}",
            .{param.type_ref},
        ) catch return;
        defer std.heap.page_allocator.free(param_text);

        prim.drawCenteredText(
            param_text,
            param_pos,
            param_width,
            param_height,
            14,
            ts.Color.init(50, 50, 50, 255),
            font,
        );

        current_y += param_height + 10;
    }

    // Calculate midpoint for arrow
    const params_mid_y = pos.y + (@as(f32, @floatFromInt(func_def.parameters.len)) * (param_height + 10)) / 2;

    // Draw arrow (transformation)
    const arrow_start = ts.Vec2.init(pos.x + param_width + 10, params_mid_y);
    const arrow_end = ts.Vec2.init(arrow_start.x + arrow_length, params_mid_y);

    prim.drawArrow(arrow_start, arrow_end, typ.metadata.color, 3);

    // Draw function name on arrow
    const name_z = std.heap.page_allocator.dupeZ(u8, typ.name) catch return;
    defer std.heap.page_allocator.free(name_z);

    prim.drawText(
        name_z,
        arrow_start.x + arrow_length / 2 - 30,
        params_mid_y - 20,
        14,
        typ.metadata.color,
        font,
    );

    // Draw return type
    const return_pos = ts.Vec2.init(
        arrow_end.x + 10,
        params_mid_y - return_height / 2,
    );

    const return_color = typ.metadata.color.withAlpha(150);
    prim.drawRoundedBox(
        return_pos,
        return_width,
        return_height,
        return_color,
        typ.metadata.color,
        RenderConstants.roundness,
    );

    const return_text = util.allocPrintZ(
        std.heap.page_allocator,
        "{s}",
        .{func_def.return_type},
    ) catch return;
    defer std.heap.page_allocator.free(return_text);

    prim.drawCenteredText(
        return_text,
        return_pos,
        return_width,
        return_height,
        14,
        ts.Color.init(255, 255, 255, 255),
        font,
    );
}

/// Render a tuple type
fn renderTuple(typ: *const ts.Type, pos: ts.Vec2, font: rl.Font, current_theme: *const theme.Theme) void {
    _ = current_theme; // TODO: Use theme colors
    const tuple_def = typ.definition.tuple;
    const element_size: f32 = 60;
    const spacing: f32 = 20;

    // Draw connecting line
    const line_y = pos.y + element_size / 2;
    const line_start_x = pos.x;
    const line_end_x = pos.x + @as(f32, @floatFromInt(tuple_def.elements.len)) * (element_size + spacing);

    rl.drawLineEx(
        rl.Vector2{ .x = line_start_x, .y = line_y },
        rl.Vector2{ .x = line_end_x, .y = line_y },
        2,
        prim.toRaylibColor(typ.metadata.color),
    );

    // Draw elements
    var i: usize = 0;
    while (i < tuple_def.elements.len) : (i += 1) {
        const element = @constCast(tuple_def.elements.at(i));
        const elem_x = pos.x + @as(f32, @floatFromInt(i)) * (element_size + spacing);
        const elem_pos = ts.Vec2.init(elem_x, pos.y);

        prim.drawCircle(
            ts.Vec2.init(elem_x + element_size / 2, pos.y + element_size / 2),
            element_size / 2,
            typ.metadata.color.withAlpha(120),
            typ.metadata.color,
        );

        // Draw element type
        const elem_text = util.allocPrintZ(
            std.heap.page_allocator,
            "{s}",
            .{element.type_ref},
        ) catch return;
        defer std.heap.page_allocator.free(elem_text);

        prim.drawCenteredText(
            elem_text,
            elem_pos,
            element_size,
            element_size,
            12,
            ts.Color.init(255, 255, 255, 255),
            font,
        );
    }

    // Draw tuple name above
    const name_z = std.heap.page_allocator.dupeZ(u8, typ.name) catch return;
    defer std.heap.page_allocator.free(name_z);

    prim.drawText(
        name_z,
        pos.x,
        pos.y - 25,
        18,
        typ.metadata.color,
        font,
    );
}

/// Render an optional type
fn renderOptional(typ: *const ts.Type, pos: ts.Vec2, font: rl.Font, current_theme: *const theme.Theme) void {
    _ = current_theme;
    const width = RenderConstants.box_width;
    const height = RenderConstants.box_height;

    // Draw ghost box
    prim.drawGhostBox(pos, width, height, typ.metadata.color);

    // Draw inner type reference
    const optional_def = typ.definition.optional;
    const text = util.allocPrintZ(
        std.heap.page_allocator,
        "?{s}",
        .{optional_def.inner_type},
    ) catch return;
    defer std.heap.page_allocator.free(text);

    prim.drawCenteredText(
        text,
        pos,
        width,
        height,
        20,
        typ.metadata.color,
        font,
    );
}

/// Bounds of a type visualization
pub const Bounds = struct {
    x: f32,
    y: f32,
    width: f32,
    height: f32,
};

/// Get the bounding box of a type for hit detection
pub fn getTypeBounds(typ: *const ts.Type) Bounds {
    const pos = typ.metadata.position;

    return switch (typ.definition) {
        .primitive => Bounds{
            .x = pos.x,
            .y = pos.y,
            .width = RenderConstants.box_width,
            .height = RenderConstants.box_height,
        },
        .@"struct" => |s| blk: {
            const header_height = RenderConstants.box_height;
            const field_height = RenderConstants.field_height;
            const total_height = header_height + @as(f32, @floatFromInt(s.fields.len)) * field_height;
            break :blk Bounds{
                .x = pos.x,
                .y = pos.y,
                .width = RenderConstants.box_width,
                .height = total_height,
            };
        },
        .@"enum" => |e| blk: {
            const trunk_height: f32 = 60;
            const branch_length = RenderConstants.branch_length;
            const variant_height: f32 = 40;
            const total_height = trunk_height + branch_length + variant_height;
            const variant_count = @as(f32, @floatFromInt(e.variants.len));
            const spread = branch_length * 2; // Approximate horizontal spread
            _ = variant_count;
            break :blk Bounds{
                .x = pos.x - spread / 2,
                .y = pos.y,
                .width = 80 + spread,
                .height = total_height,
            };
        },
        .function => |f| blk: {
            const param_height: f32 = 40;
            const total_height = @as(f32, @floatFromInt(f.parameters.len)) * (param_height + 10);
            break :blk Bounds{
                .x = pos.x,
                .y = pos.y,
                .width = 300,
                .height = @max(total_height, 60),
            };
        },
        .tuple => |t| blk: {
            const element_size: f32 = 60;
            const spacing: f32 = 20;
            const width = @as(f32, @floatFromInt(t.elements.len)) * (element_size + spacing);
            break :blk Bounds{
                .x = pos.x,
                .y = pos.y - 25,
                .width = width,
                .height = element_size + 25,
            };
        },
        .optional => Bounds{
            .x = pos.x,
            .y = pos.y,
            .width = RenderConstants.box_width,
            .height = RenderConstants.box_height,
        },
        .recursive => Bounds{
            .x = pos.x,
            .y = pos.y,
            .width = RenderConstants.box_width,
            .height = RenderConstants.box_height * 3, // Approximate height for recursive types
        },
    };
}

/// Main render function that dispatches to specific type renderers
pub fn renderType(typ: *const ts.Type, font: rl.Font) void {
    const pos = typ.metadata.position;
    const current_theme = theme.getCurrentTheme() catch return;

    switch (typ.definition) {
        .primitive => renderPrimitive(typ, pos, font, current_theme),
        .@"struct" => renderStruct(typ, pos, font, current_theme),
        .@"enum" => renderEnum(typ, pos, font, current_theme),
        .function => renderFunction(typ, pos, font, current_theme),
        .tuple => renderTuple(typ, pos, font, current_theme),
        .optional => renderOptional(typ, pos, font, current_theme),
        .recursive => renderStruct(typ, pos, font, current_theme), // Render like struct for now
    }
}
