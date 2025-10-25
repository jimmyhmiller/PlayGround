const std = @import("std");
const ts = @import("type_system.zig");

pub const ParseError = error{
    InvalidJson,
    MissingField,
    InvalidTypeKind,
    InvalidColor,
} || std.mem.Allocator.Error || std.json.ParseError(std.json.Scanner);

/// Parse a Vec2 from JSON
fn parseVec2(obj: std.json.Value) !ts.Vec2 {
    const x = obj.object.get("x") orelse return error.MissingField;
    const y = obj.object.get("y") orelse return error.MissingField;

    const x_val: f32 = switch (x) {
        .integer => |i| @floatFromInt(i),
        .float => |f| @floatCast(f),
        else => return error.InvalidJson,
    };

    const y_val: f32 = switch (y) {
        .integer => |i| @floatFromInt(i),
        .float => |f| @floatCast(f),
        else => return error.InvalidJson,
    };

    return ts.Vec2.init(x_val, y_val);
}

/// Parse a Color from JSON hex string
fn parseColor(hex: []const u8) !ts.Color {
    return ts.Color.fromHex(hex) catch error.InvalidColor;
}

/// Parse metadata from JSON
fn parseMetadata(allocator: std.mem.Allocator, obj: std.json.Value) !ts.Metadata {
    _ = allocator;
    const color_str = obj.object.get("color") orelse return error.MissingField;
    const position = obj.object.get("position") orelse return error.MissingField;
    const collapsed = obj.object.get("collapsed") orelse return error.MissingField;

    const color = try parseColor(color_str.string);
    const pos = try parseVec2(position);

    return .{
        .color = color,
        .position = pos,
        .collapsed = collapsed.bool,
        .animation = ts.AnimationState.init(),
    };
}

/// Parse struct fields from JSON
fn parseStructFields(allocator: std.mem.Allocator, fields_array: std.json.Value) !std.ArrayList(ts.StructField) {
    var fields = std.ArrayList(ts.StructField){};
    errdefer {
        for (fields.items) |field| {
            field.deinit(allocator);
        }
        fields.deinit(allocator);
    }

    for (fields_array.array.items) |field_obj| {
        const name = field_obj.object.get("name") orelse return error.MissingField;
        const type_ref = field_obj.object.get("type_ref") orelse return error.MissingField;
        const description = field_obj.object.get("description") orelse return error.MissingField;

        const field = try ts.StructField.init(
            allocator,
            name.string,
            type_ref.string,
            description.string,
        );
        try fields.append(allocator, field);
    }

    return fields;
}

/// Parse enum variants from JSON
fn parseEnumVariants(allocator: std.mem.Allocator, variants_array: std.json.Value) !std.ArrayList(ts.EnumVariant) {
    var variants = std.ArrayList(ts.EnumVariant){};
    errdefer {
        for (variants.items) |variant| {
            variant.deinit(allocator);
        }
        variants.deinit(allocator);
    }

    for (variants_array.array.items) |variant_obj| {
        const name = variant_obj.object.get("name") orelse return error.MissingField;
        const description = variant_obj.object.get("description") orelse return error.MissingField;
        const payload = variant_obj.object.get("payload");

        const variant = try ts.EnumVariant.init(
            allocator,
            name.string,
            if (payload) |p| p.string else null,
            description.string,
        );
        try variants.append(allocator, variant);
    }

    return variants;
}

/// Parse function parameters from JSON
fn parseFunctionParameters(allocator: std.mem.Allocator, params_array: std.json.Value) !std.ArrayList(ts.FunctionParameter) {
    var params = std.ArrayList(ts.FunctionParameter){};
    errdefer {
        for (params.items) |param| {
            param.deinit(allocator);
        }
        params.deinit(allocator);
    }

    for (params_array.array.items) |param_obj| {
        const name = param_obj.object.get("name") orelse return error.MissingField;
        const type_ref = param_obj.object.get("type_ref") orelse return error.MissingField;

        const param = try ts.FunctionParameter.init(
            allocator,
            name.string,
            type_ref.string,
        );
        try params.append(allocator, param);
    }

    return params;
}

/// Parse tuple elements from JSON
fn parseTupleElements(allocator: std.mem.Allocator, elements_array: std.json.Value) !std.ArrayList(ts.TupleElement) {
    var elements = std.ArrayList(ts.TupleElement){};
    errdefer {
        for (elements.items) |elem| {
            elem.deinit(allocator);
        }
        elements.deinit(allocator);
    }

    for (elements_array.array.items) |elem_obj| {
        const index = elem_obj.object.get("index") orelse return error.MissingField;
        const type_ref = elem_obj.object.get("type_ref") orelse return error.MissingField;
        const description = elem_obj.object.get("description") orelse return error.MissingField;

        const elem = try ts.TupleElement.init(
            allocator,
            @intCast(index.integer),
            type_ref.string,
            description.string,
        );
        try elements.append(allocator, elem);
    }

    return elements;
}

/// Parse type definition from JSON
fn parseTypeDefinition(allocator: std.mem.Allocator, kind: ts.TypeKind, def_obj: std.json.Value) !ts.TypeDefinition {
    return switch (kind) {
        .primitive => .{ .primitive = {} },
        .@"struct" => blk: {
            const struct_obj = def_obj.object.get("struct") orelse return error.MissingField;
            const fields_array = struct_obj.object.get("fields") orelse return error.MissingField;
            const fields = try parseStructFields(allocator, fields_array);
            break :blk .{ .@"struct" = .{ .fields = fields } };
        },
        .@"enum" => blk: {
            const enum_obj = def_obj.object.get("enum") orelse return error.MissingField;
            const variants_array = enum_obj.object.get("variants") orelse return error.MissingField;
            const variants = try parseEnumVariants(allocator, variants_array);
            break :blk .{ .@"enum" = .{ .variants = variants } };
        },
        .function => blk: {
            const func_obj = def_obj.object.get("function") orelse return error.MissingField;
            const params_array = func_obj.object.get("parameters") orelse return error.MissingField;
            const return_type = func_obj.object.get("return_type") orelse return error.MissingField;

            const params = try parseFunctionParameters(allocator, params_array);
            const ret_type = try allocator.dupe(u8, return_type.string);

            break :blk .{ .function = .{
                .parameters = params,
                .return_type = ret_type,
            } };
        },
        .tuple => blk: {
            const tuple_obj = def_obj.object.get("tuple") orelse return error.MissingField;
            const elements_array = tuple_obj.object.get("elements") orelse return error.MissingField;
            const elements = try parseTupleElements(allocator, elements_array);
            break :blk .{ .tuple = .{ .elements = elements } };
        },
        .optional => blk: {
            const optional_obj = def_obj.object.get("optional") orelse return error.MissingField;
            const inner_type = optional_obj.object.get("inner_type") orelse return error.MissingField;
            const inner = try allocator.dupe(u8, inner_type.string);
            break :blk .{ .optional = .{ .inner_type = inner } };
        },
        .recursive => blk: {
            const recursive_obj = def_obj.object.get("recursive") orelse return error.MissingField;
            const base_type = recursive_obj.object.get("base_type") orelse return error.MissingField;
            const self_ref = recursive_obj.object.get("self_reference") orelse return error.MissingField;

            const base = try allocator.dupe(u8, base_type.string);
            const self_reference = try allocator.dupe(u8, self_ref.string);

            break :blk .{ .recursive = .{
                .base_type = base,
                .self_reference = self_reference,
            } };
        },
    };
}

/// Parse a single type from JSON
fn parseType(allocator: std.mem.Allocator, type_obj: std.json.Value) !ts.Type {
    const id = type_obj.object.get("id") orelse return error.MissingField;
    const name = type_obj.object.get("name") orelse return error.MissingField;
    const kind_str = type_obj.object.get("kind") orelse return error.MissingField;
    const description = type_obj.object.get("description") orelse return error.MissingField;
    const metadata_obj = type_obj.object.get("metadata") orelse return error.MissingField;

    // Parse kind
    const kind: ts.TypeKind = std.meta.stringToEnum(ts.TypeKind, kind_str.string) orelse
        return error.InvalidTypeKind;

    const metadata = try parseMetadata(allocator, metadata_obj);

    // Parse definition if not primitive
    const definition = if (kind == .primitive)
        ts.TypeDefinition{ .primitive = {} }
    else blk: {
        const def_obj = type_obj.object.get("definition") orelse return error.MissingField;
        break :blk try parseTypeDefinition(allocator, kind, def_obj);
    };

    return ts.Type{
        .id = try allocator.dupe(u8, id.string),
        .name = try allocator.dupe(u8, name.string),
        .description = try allocator.dupe(u8, description.string),
        .metadata = metadata,
        .definition = definition,
    };
}

/// Parse instance value recursively
fn parseInstanceValue(allocator: std.mem.Allocator, value: std.json.Value) std.mem.Allocator.Error!ts.InstanceValue {
    return switch (value) {
        .null => .null_value,
        .bool => |b| .{ .bool_value = b },
        .integer => |i| .{ .int_value = i },
        .float => |f| .{ .float_value = f },
        .number_string => |s| .{ .float_value = std.fmt.parseFloat(f64, s) catch 0.0 },
        .string => |s| .{ .string_value = try allocator.dupe(u8, s) },
        .array => |arr| blk: {
            var array_values = std.ArrayList(ts.InstanceValue){};
            errdefer {
                for (array_values.items) |*item| {
                    item.deinit(allocator);
                }
                array_values.deinit(allocator);
            }
            for (arr.items) |item| {
                try array_values.append(allocator, try parseInstanceValue(allocator, item));
            }
            break :blk .{ .array_value = array_values };
        },
        .object => |obj| blk: {
            var object_values = std.StringHashMap(ts.InstanceValue).init(allocator);
            errdefer {
                var it = object_values.iterator();
                while (it.next()) |entry| {
                    allocator.free(entry.key_ptr.*);
                    entry.value_ptr.deinit(allocator);
                }
                object_values.deinit();
            }
            var it = obj.iterator();
            while (it.next()) |entry| {
                const key = try allocator.dupe(u8, entry.key_ptr.*);
                const val = try parseInstanceValue(allocator, entry.value_ptr.*);
                try object_values.put(key, val);
            }
            break :blk .{ .object_value = object_values };
        },
    };
}

/// Parse an instance from JSON
fn parseInstance(allocator: std.mem.Allocator, instance_obj: std.json.Value) !ts.Instance {
    const type_ref = instance_obj.object.get("type_ref") orelse return error.MissingField;
    const name = instance_obj.object.get("name") orelse return error.MissingField;
    const description = instance_obj.object.get("description") orelse return error.MissingField;
    const value = instance_obj.object.get("value") orelse return error.MissingField;

    return ts.Instance{
        .type_ref = try allocator.dupe(u8, type_ref.string),
        .name = try allocator.dupe(u8, name.string),
        .description = try allocator.dupe(u8, description.string),
        .value = try parseInstanceValue(allocator, value),
    };
}

/// Parse visualization from JSON
fn parseVisualization(allocator: std.mem.Allocator, viz_obj: std.json.Value) !ts.Visualization {
    const name = viz_obj.object.get("name") orelse return error.MissingField;
    const layout_str = viz_obj.object.get("layout") orelse return error.MissingField;
    const root_types_array = viz_obj.object.get("root_types") orelse return error.MissingField;
    const show_instances = viz_obj.object.get("show_instances") orelse return error.MissingField;
    const animation_speed = viz_obj.object.get("animation_speed") orelse return error.MissingField;
    const camera_obj = viz_obj.object.get("camera") orelse return error.MissingField;

    const layout = std.meta.stringToEnum(ts.LayoutType, layout_str.string) orelse return error.InvalidJson;

    var root_types = std.ArrayList([]const u8){};
    errdefer {
        for (root_types.items) |type_id| {
            allocator.free(type_id);
        }
        root_types.deinit(allocator);
    }
    for (root_types_array.array.items) |type_id| {
        try root_types.append(allocator, try allocator.dupe(u8, type_id.string));
    }

    const zoom = camera_obj.object.get("zoom") orelse return error.MissingField;
    const center = camera_obj.object.get("center") orelse return error.MissingField;

    const zoom_val: f32 = switch (zoom) {
        .integer => |i| @floatFromInt(i),
        .float => |f| @floatCast(f),
        else => return error.InvalidJson,
    };

    const animation_speed_val: f32 = switch (animation_speed) {
        .integer => |i| @floatFromInt(i),
        .float => |f| @floatCast(f),
        else => return error.InvalidJson,
    };

    return ts.Visualization{
        .name = try allocator.dupe(u8, name.string),
        .layout = layout,
        .root_types = root_types,
        .show_instances = show_instances.bool,
        .animation_speed = animation_speed_val,
        .camera = .{
            .zoom = zoom_val,
            .center = try parseVec2(center),
        },
    };
}

/// Parse a complete type system from JSON file
pub fn parseFile(allocator: std.mem.Allocator, file_path: []const u8) !ts.TypeSystem {
    // Read file
    const file = try std.fs.cwd().openFile(file_path, .{});
    defer file.close();

    const file_size = try file.getEndPos();
    const buffer = try allocator.alloc(u8, file_size);
    defer allocator.free(buffer);

    _ = try file.readAll(buffer);

    // Parse JSON
    const parsed = try std.json.parseFromSlice(std.json.Value, allocator, buffer, .{});
    defer parsed.deinit();

    const root = parsed.value;
    const types_obj = root.object.get("types") orelse return error.MissingField;
    const type_defs = types_obj.object.get("type_definitions") orelse return error.MissingField;
    const instances_array = types_obj.object.get("instances") orelse return error.MissingField;
    const viz_array = types_obj.object.get("visualizations") orelse return error.MissingField;

    var type_system = ts.TypeSystem.init(allocator);
    errdefer type_system.deinit();

    // Parse types
    for (type_defs.array.items) |type_obj| {
        const typ = try parseType(allocator, type_obj);
        try type_system.addType(typ);
    }

    // Parse instances
    for (instances_array.array.items) |instance_obj| {
        const instance = try parseInstance(allocator, instance_obj);
        try type_system.instances.append(allocator, instance);
    }

    // Parse visualizations
    for (viz_array.array.items) |viz_obj| {
        const viz = try parseVisualization(allocator, viz_obj);
        try type_system.visualizations.append(allocator, viz);
    }

    return type_system;
}
