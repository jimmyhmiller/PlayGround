const std = @import("std");
const rl = @import("raylib");
const ts = @import("types/type_system.zig");
const type_builder = @import("types/type_builder.zig");
const type_renderer = @import("render/type_renderer.zig");
const prim = @import("render/primitives.zig");
const theme = @import("render/theme.zig");
const demo_controller = @import("demo/demo_controller.zig");
const animator = @import("animation/animator.zig");
const easing = @import("animation/easing.zig");
const util = @import("util.zig");

pub fn main() !void {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();

    // Initialize empty type system
    var type_system = ts.TypeSystem.init(allocator);
    defer type_system.deinit();

    var builder = type_builder.TypeBuilder.init(&type_system, allocator);

    // Initialize demo controller
    var demo_ctrl = demo_controller.DemoController.init(allocator);
    defer demo_ctrl.deinit();

    // Initialize animator
    var anim = animator.Animator.init(allocator);
    defer anim.deinit();

    // Load the Task Management System demo
    std.debug.print("Loading Task Management System demo...\n", .{});
    const task_demo = try demo_controller.createTaskManagementDemo(allocator);
    demo_ctrl.loadDemo(task_demo);

    // Initialization
    const screenWidth = 1400;
    const screenHeight = 900;

    rl.initWindow(screenWidth, screenHeight, "Data Type Visualizer - Demo Mode");
    defer rl.closeWindow();

    rl.setTargetFPS(60);

    // Load SF Mono font
    const font = rl.loadFontEx("/System/Library/Fonts/SFNSMono.ttf", 96, null) catch blk: {
        std.debug.print("Failed to load SF Mono, trying Courier New...\n", .{});
        break :blk rl.loadFontEx("/System/Library/Fonts/Supplemental/Courier New.ttf", 96, null) catch blk2: {
            std.debug.print("Failed to load Courier New, using default font\n", .{});
            break :blk2 try rl.getFontDefault();
        };
    };
    defer {
        if (font.texture.id != 1) {
            rl.unloadFont(font);
        }
    }

    rl.setTextureFilter(font.texture, rl.TextureFilter.bilinear);

    var last_frame_time = rl.getTime();

    // Dragging state
    var dragging_type: ?[]const u8 = null;
    var drag_offset = ts.Vec2.init(0, 0);

    // Main game loop
    while (!rl.windowShouldClose()) {
        const current_time = rl.getTime();
        const delta_time = @as(f32, @floatCast(current_time - last_frame_time));
        last_frame_time = current_time;

        // Handle mouse dragging
        if (rl.isMouseButtonPressed(rl.MouseButton.left)) {
            const mouse_pos = rl.getMousePosition();
            // Check if we clicked on any type
            var type_it = type_system.types.iterator();
            while (type_it.next()) |entry| {
                const typ = entry.value_ptr.*;
                const bounds = type_renderer.getTypeBounds(typ);
                if (mouse_pos.x >= bounds.x and mouse_pos.x <= bounds.x + bounds.width and
                    mouse_pos.y >= bounds.y and mouse_pos.y <= bounds.y + bounds.height) {
                    dragging_type = entry.key_ptr.*;
                    drag_offset = ts.Vec2.init(mouse_pos.x - typ.metadata.position.x, mouse_pos.y - typ.metadata.position.y);
                    break;
                }
            }
        }

        if (rl.isMouseButtonReleased(rl.MouseButton.left)) {
            dragging_type = null;
        }

        // Update dragged type position
        if (dragging_type) |type_id| {
            if (type_system.getType(type_id)) |typ| {
                const mouse_pos = rl.getMousePosition();
                typ.metadata.position = ts.Vec2.init(mouse_pos.x - drag_offset.x, mouse_pos.y - drag_offset.y);
            }
        }

        // Handle input
        if (rl.isKeyPressed(rl.KeyboardKey.space) or rl.isKeyPressed(rl.KeyboardKey.right)) {
            if (demo_ctrl.nextStage()) {
                if (demo_ctrl.getCurrentStage()) |stage| {
                    try executeStageAction(&builder, &type_system, &anim, stage);
                }
            }
        }

        if (rl.isKeyPressed(rl.KeyboardKey.left)) {
            _ = demo_ctrl.previousStage();
            // TODO: Implement reverse actions
        }

        if (rl.isKeyPressed(rl.KeyboardKey.r)) {
            demo_ctrl.reset();
            // Clear type system
            var type_it = type_system.types.iterator();
            while (type_it.next()) |entry| {
                entry.value_ptr.*.deinit(allocator);
                allocator.destroy(entry.value_ptr.*);
            }
            type_system.types.clearRetainingCapacity();
            anim.clear();
        }

        if (rl.isKeyPressed(rl.KeyboardKey.a)) {
            demo_ctrl.toggleAutoPlay();
        }

        if (rl.isKeyPressed(rl.KeyboardKey.t)) {
            try theme.toggleTheme();
        }

        // Update demo controller
        demo_ctrl.update(delta_time);

        // Update animations
        anim.update(delta_time);

        // Update type animations
        var type_it = type_system.types.iterator();
        while (type_it.next()) |entry| {
            const typ = entry.value_ptr.*;
            updateTypeAnimation(&typ.metadata.animation, delta_time);

            // Update field/variant animations
            switch (typ.definition) {
                .@"struct" => |*s| {
                    var i: usize = 0;
                    while (i < s.fields.len) : (i += 1) {
                        const field = @constCast(s.fields.at(i));
                        updateTypeAnimation(&field.animation, delta_time);
                    }
                },
                .@"enum" => |*e| {
                    var i: usize = 0;
                    while (i < e.variants.len) : (i += 1) {
                        const variant = @constCast(e.variants.at(i));
                        updateTypeAnimation(&variant.animation, delta_time);
                    }
                },
                .function => |*f| {
                    var i: usize = 0;
                    while (i < f.parameters.len) : (i += 1) {
                        const param = @constCast(f.parameters.at(i));
                        updateTypeAnimation(&param.animation, delta_time);
                    }
                },
                .tuple => |*t| {
                    var i: usize = 0;
                    while (i < t.elements.len) : (i += 1) {
                        const elem = @constCast(t.elements.at(i));
                        updateTypeAnimation(&elem.animation, delta_time);
                    }
                },
                else => {},
            }
        }

        // Draw
        const current_theme = try theme.getCurrentTheme();
        rl.beginDrawing();
        defer rl.endDrawing();

        rl.clearBackground(prim.toRaylibColor(current_theme.background));

        // Draw grid
        prim.drawGrid(50, current_theme.grid);

        // Render all types
        var render_it = type_system.types.iterator();
        while (render_it.next()) |entry| {
            const typ = entry.value_ptr.*;
            type_renderer.renderType(typ, font, &type_system);
        }

        // Draw UI overlay
        const ui_y_start: f32 = 20;
        rl.drawTextEx(font, "Live Type Visualizer Demo", rl.Vector2{ .x = 20, .y = ui_y_start }, 30, 0.5, prim.toRaylibColor(current_theme.ui_text));

        // Demo info
        if (demo_ctrl.current_demo) |demo| {
            const demo_name = try util.allocPrintZ(allocator, "Demo: {s}", .{demo.name});
            defer allocator.free(demo_name);
            rl.drawTextEx(font, demo_name, rl.Vector2{ .x = 20, .y = ui_y_start + 40 }, 18, 0.5, prim.toRaylibColor(current_theme.ui_text));

            if (demo_ctrl.getCurrentStage()) |stage| {
                const stage_text = try util.allocPrintZ(allocator, "Stage {}/{}: {s}", .{ stage.id + 1, demo.stages.len, stage.description });
                defer allocator.free(stage_text);
                rl.drawTextEx(font, stage_text, rl.Vector2{ .x = 20, .y = ui_y_start + 70 }, 16, 0.5, prim.toRaylibColor(current_theme.ui_text_secondary));
            }
        }

        // Controls
        const controls_y: f32 = @as(f32, @floatFromInt(screenHeight)) - 120;
        rl.drawTextEx(font, "Controls:", rl.Vector2{ .x = 20, .y = controls_y }, 18, 0.5, prim.toRaylibColor(current_theme.ui_text));
        rl.drawTextEx(font, "SPACE/RIGHT - Next Stage", rl.Vector2{ .x = 20, .y = controls_y + 25 }, 14, 0.5, prim.toRaylibColor(current_theme.ui_text_secondary));
        rl.drawTextEx(font, "R - Reset Demo", rl.Vector2{ .x = 20, .y = controls_y + 45 }, 14, 0.5, prim.toRaylibColor(current_theme.ui_text_secondary));
        rl.drawTextEx(font, "A - Toggle Auto-play", rl.Vector2{ .x = 20, .y = controls_y + 65 }, 14, 0.5, prim.toRaylibColor(current_theme.ui_text_secondary));
        rl.drawTextEx(font, "T - Toggle Theme", rl.Vector2{ .x = 20, .y = controls_y + 85 }, 14, 0.5, prim.toRaylibColor(current_theme.ui_text_secondary));

        // Auto-play indicator
        if (demo_ctrl.auto_play) {
            const auto_text = "AUTO-PLAY ON";
            rl.drawTextEx(font, auto_text, rl.Vector2{ .x = @as(f32, @floatFromInt(screenWidth)) - 200, .y = ui_y_start }, 18, 0.5, prim.toRaylibColor(current_theme.valid_reference));
        }

        // Progress bar
        const progress = demo_ctrl.getProgress();
        const bar_width: f32 = 300;
        const bar_height: f32 = 10;
        const bar_x: f32 = @as(f32, @floatFromInt(screenWidth)) - bar_width - 20;
        const bar_y: f32 = ui_y_start + 40;

        rl.drawRectangle(@intFromFloat(bar_x), @intFromFloat(bar_y), @intFromFloat(bar_width), @intFromFloat(bar_height), prim.toRaylibColor(current_theme.ui_button));
        rl.drawRectangle(@intFromFloat(bar_x), @intFromFloat(bar_y), @intFromFloat(bar_width * progress), @intFromFloat(bar_height), prim.toRaylibColor(current_theme.valid_reference));

        // FPS counter
        rl.drawFPS(screenWidth - 100, screenHeight - 30);
    }
}

/// Execute a demo stage action
fn executeStageAction(
    builder: *type_builder.TypeBuilder,
    type_system: *ts.TypeSystem,
    anim: *animator.Animator,
    stage: *const demo_controller.DemoStage,
) !void {
    switch (stage.action) {
        .add_type => |action| {
            const color = try ts.Color.fromHex(action.color);
            try builder.createType(action.type_id, action.type_name, action.type_kind, action.position, color);

            // Animate the type appearing
            if (type_system.getType(action.type_id)) |typ| {
                try anim.animateFloat(&typ.metadata.animation.opacity, 1.0, 0.5, easing.easeOut);
                try anim.animateFloat(&typ.metadata.animation.scale, 1.0, 0.5, easing.backOut);
            }
        },
        .add_struct_field => |action| {
            try builder.addStructField(action.type_id, action.field_name, action.field_type, action.description);

            // Animate the field appearing
            if (type_system.getType(action.type_id)) |typ| {
                if (typ.definition == .@"struct") {
                    if (typ.definition.@"struct".fields.len > 0) {
                        const last_field = @constCast(typ.definition.@"struct".fields.at(typ.definition.@"struct".fields.len - 1));
                        try anim.animateFloat(&last_field.animation.opacity, 1.0, 0.4, easing.easeOut);
                        try anim.animateFloat(&last_field.animation.scale, 1.0, 0.4, easing.backOut);
                    }
                }
            }
        },
        .add_enum_variant => |action| {
            try builder.addEnumVariant(action.type_id, action.variant_name, action.payload, action.description);

            // Animate the variant appearing
            if (type_system.getType(action.type_id)) |typ| {
                if (typ.definition == .@"enum") {
                    if (typ.definition.@"enum".variants.len > 0) {
                        const last_variant = @constCast(typ.definition.@"enum".variants.at(typ.definition.@"enum".variants.len - 1));
                        try anim.animateFloat(&last_variant.animation.opacity, 1.0, 0.4, easing.easeOut);
                        try anim.animateFloat(&last_variant.animation.scale, 1.0, 0.4, easing.elasticOut);
                    }
                }
            }
        },
        .add_function_param => |action| {
            try builder.addFunctionParameter(action.type_id, action.param_name, action.param_type);

            // Animate the parameter appearing
            if (type_system.getType(action.type_id)) |typ| {
                if (typ.definition == .function) {
                    if (typ.definition.function.parameters.len > 0) {
                        const last_param = @constCast(typ.definition.function.parameters.at(typ.definition.function.parameters.len - 1));
                        try anim.animateFloat(&last_param.animation.opacity, 1.0, 0.4, easing.easeOut);
                        try anim.animateFloat(&last_param.animation.scale, 1.0, 0.4, easing.backOut);
                    }
                }
            }
        },
        .add_tuple_element => |action| {
            try builder.addTupleElement(action.type_id, action.element_type, action.description);

            // Animate the element appearing
            if (type_system.getType(action.type_id)) |typ| {
                if (typ.definition == .tuple) {
                    if (typ.definition.tuple.elements.len > 0) {
                        const last_elem = @constCast(typ.definition.tuple.elements.at(typ.definition.tuple.elements.len - 1));
                        try anim.animateFloat(&last_elem.animation.opacity, 1.0, 0.4, easing.easeOut);
                        try anim.animateFloat(&last_elem.animation.scale, 1.0, 0.4, easing.bounceOut);
                    }
                }
            }
        },
        .complete => {},
    }
}

/// Update animation state manually (for appearing animations)
fn updateTypeAnimation(anim_state: *ts.AnimationState, delta_time: f32) void {
    if (anim_state.appearing) {
        anim_state.animation_time += delta_time;
        // Animations are handled by the Animator, but we track time here
    }
}
