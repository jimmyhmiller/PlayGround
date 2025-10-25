const std = @import("std");

/// Position in 2D space
pub const Vec2 = struct {
    x: f32,
    y: f32,

    pub fn init(x: f32, y: f32) Vec2 {
        return .{ .x = x, .y = y };
    }

    pub fn add(self: Vec2, other: Vec2) Vec2 {
        return .{ .x = self.x + other.x, .y = self.y + other.y };
    }

    pub fn sub(self: Vec2, other: Vec2) Vec2 {
        return .{ .x = self.x - other.x, .y = self.y - other.y };
    }

    pub fn scale(self: Vec2, scalar: f32) Vec2 {
        return .{ .x = self.x * scalar, .y = self.y * scalar };
    }

    pub fn distance(self: Vec2, other: Vec2) f32 {
        const dx = other.x - self.x;
        const dy = other.y - self.y;
        return @sqrt(dx * dx + dy * dy);
    }
};

/// Color representation
pub const Color = struct {
    r: u8,
    g: u8,
    b: u8,
    a: u8,

    pub fn fromHex(hex: []const u8) !Color {
        if (hex.len != 7 or hex[0] != '#') {
            return error.InvalidHexColor;
        }
        const r = try std.fmt.parseInt(u8, hex[1..3], 16);
        const g = try std.fmt.parseInt(u8, hex[3..5], 16);
        const b = try std.fmt.parseInt(u8, hex[5..7], 16);
        return Color{ .r = r, .g = g, .b = b, .a = 255 };
    }

    pub fn init(r: u8, g: u8, b: u8, a: u8) Color {
        return .{ .r = r, .g = g, .b = b, .a = a };
    }

    pub fn withAlpha(self: Color, alpha: u8) Color {
        return .{ .r = self.r, .g = self.g, .b = self.b, .a = alpha };
    }
};

/// Animation state for smooth transitions
pub const AnimationState = struct {
    appearing: bool = false,
    opacity: f32 = 1.0,
    scale: f32 = 1.0,
    animation_time: f32 = 0.0,

    pub fn init() AnimationState {
        return .{};
    }

    pub fn initAppearing() AnimationState {
        return .{
            .appearing = true,
            .opacity = 0.0,
            .scale = 0.0,
            .animation_time = 0.0,
        };
    }
};

/// Metadata for visualization
pub const Metadata = struct {
    color: Color,
    position: Vec2,
    collapsed: bool,
    animation: AnimationState,

    pub fn init(color: Color, position: Vec2) Metadata {
        return .{
            .color = color,
            .position = position,
            .collapsed = false,
            .animation = AnimationState.init(),
        };
    }

    pub fn initAppearing(color: Color, position: Vec2) Metadata {
        return .{
            .color = color,
            .position = position,
            .collapsed = false,
            .animation = AnimationState.initAppearing(),
        };
    }
};

/// Type kind enumeration
pub const TypeKind = enum {
    primitive,
    @"struct",
    @"enum",
    function,
    tuple,
    optional,
    recursive,
};

/// Field in a struct
pub const StructField = struct {
    name: []const u8,
    type_ref: []const u8,
    description: []const u8,
    animation: AnimationState = AnimationState.init(),

    pub fn init(allocator: std.mem.Allocator, name: []const u8, type_ref: []const u8, description: []const u8) !StructField {
        return .{
            .name = try allocator.dupe(u8, name),
            .type_ref = try allocator.dupe(u8, type_ref),
            .description = try allocator.dupe(u8, description),
            .animation = AnimationState.init(),
        };
    }

    pub fn initAppearing(allocator: std.mem.Allocator, name: []const u8, type_ref: []const u8, description: []const u8) !StructField {
        return .{
            .name = try allocator.dupe(u8, name),
            .type_ref = try allocator.dupe(u8, type_ref),
            .description = try allocator.dupe(u8, description),
            .animation = AnimationState.initAppearing(),
        };
    }

    pub fn deinit(self: StructField, allocator: std.mem.Allocator) void {
        allocator.free(self.name);
        allocator.free(self.type_ref);
        allocator.free(self.description);
    }
};

/// Variant in an enum
pub const EnumVariant = struct {
    name: []const u8,
    payload: ?[]const u8,
    description: []const u8,
    animation: AnimationState = AnimationState.init(),

    pub fn init(allocator: std.mem.Allocator, name: []const u8, payload: ?[]const u8, description: []const u8) !EnumVariant {
        return .{
            .name = try allocator.dupe(u8, name),
            .payload = if (payload) |p| try allocator.dupe(u8, p) else null,
            .description = try allocator.dupe(u8, description),
            .animation = AnimationState.init(),
        };
    }

    pub fn initAppearing(allocator: std.mem.Allocator, name: []const u8, payload: ?[]const u8, description: []const u8) !EnumVariant {
        return .{
            .name = try allocator.dupe(u8, name),
            .payload = if (payload) |p| try allocator.dupe(u8, p) else null,
            .description = try allocator.dupe(u8, description),
            .animation = AnimationState.initAppearing(),
        };
    }

    pub fn deinit(self: EnumVariant, allocator: std.mem.Allocator) void {
        allocator.free(self.name);
        if (self.payload) |p| allocator.free(p);
        allocator.free(self.description);
    }
};

/// Function parameter
pub const FunctionParameter = struct {
    name: []const u8,
    type_ref: []const u8,
    animation: AnimationState = AnimationState.init(),

    pub fn init(allocator: std.mem.Allocator, name: []const u8, type_ref: []const u8) !FunctionParameter {
        return .{
            .name = try allocator.dupe(u8, name),
            .type_ref = try allocator.dupe(u8, type_ref),
            .animation = AnimationState.init(),
        };
    }

    pub fn initAppearing(allocator: std.mem.Allocator, name: []const u8, type_ref: []const u8) !FunctionParameter {
        return .{
            .name = try allocator.dupe(u8, name),
            .type_ref = try allocator.dupe(u8, type_ref),
            .animation = AnimationState.initAppearing(),
        };
    }

    pub fn deinit(self: FunctionParameter, allocator: std.mem.Allocator) void {
        allocator.free(self.name);
        allocator.free(self.type_ref);
    }
};

/// Tuple element
pub const TupleElement = struct {
    index: usize,
    type_ref: []const u8,
    description: []const u8,
    animation: AnimationState = AnimationState.init(),

    pub fn init(allocator: std.mem.Allocator, index: usize, type_ref: []const u8, description: []const u8) !TupleElement {
        return .{
            .index = index,
            .type_ref = try allocator.dupe(u8, type_ref),
            .description = try allocator.dupe(u8, description),
            .animation = AnimationState.init(),
        };
    }

    pub fn initAppearing(allocator: std.mem.Allocator, index: usize, type_ref: []const u8, description: []const u8) !TupleElement {
        return .{
            .index = index,
            .type_ref = try allocator.dupe(u8, type_ref),
            .description = try allocator.dupe(u8, description),
            .animation = AnimationState.initAppearing(),
        };
    }

    pub fn deinit(self: TupleElement, allocator: std.mem.Allocator) void {
        allocator.free(self.type_ref);
        allocator.free(self.description);
    }
};

/// Type definition details
pub const TypeDefinition = union(TypeKind) {
    primitive: void,
    @"struct": struct {
        fields: std.SegmentedList(StructField, 8),
    },
    @"enum": struct {
        variants: std.SegmentedList(EnumVariant, 8),
    },
    function: struct {
        parameters: std.SegmentedList(FunctionParameter, 8),
        return_type: []const u8,
    },
    tuple: struct {
        elements: std.SegmentedList(TupleElement, 8),
    },
    optional: struct {
        inner_type: []const u8,
    },
    recursive: struct {
        base_type: []const u8,
        self_reference: []const u8,
    },

    pub fn deinit(self: *TypeDefinition, allocator: std.mem.Allocator) void {
        switch (self.*) {
            .primitive => {},
            .@"struct" => |*s| {
                var i: usize = 0;
                while (i < s.fields.len) : (i += 1) {
                    const field = s.fields.at(i);
                    field.deinit(allocator);
                }
                s.fields.deinit(allocator);
            },
            .@"enum" => |*e| {
                var i: usize = 0;
                while (i < e.variants.len) : (i += 1) {
                    const variant = e.variants.at(i);
                    variant.deinit(allocator);
                }
                e.variants.deinit(allocator);
            },
            .function => |*f| {
                var i: usize = 0;
                while (i < f.parameters.len) : (i += 1) {
                    const param = f.parameters.at(i);
                    param.deinit(allocator);
                }
                f.parameters.deinit(allocator);
                allocator.free(f.return_type);
            },
            .tuple => |*t| {
                var i: usize = 0;
                while (i < t.elements.len) : (i += 1) {
                    const elem = t.elements.at(i);
                    elem.deinit(allocator);
                }
                t.elements.deinit(allocator);
            },
            .optional => |*o| {
                allocator.free(o.inner_type);
            },
            .recursive => |*r| {
                allocator.free(r.base_type);
                allocator.free(r.self_reference);
            },
        }
    }
};

/// A complete type with metadata
pub const Type = struct {
    id: []const u8,
    name: []const u8,
    description: []const u8,
    metadata: Metadata,
    definition: TypeDefinition,

    pub fn deinit(self: *Type, allocator: std.mem.Allocator) void {
        allocator.free(self.id);
        allocator.free(self.name);
        allocator.free(self.description);
        self.definition.deinit(allocator);
    }
};

/// Instance value (simplified JSON representation)
pub const InstanceValue = union(enum) {
    null_value,
    bool_value: bool,
    int_value: i64,
    float_value: f64,
    string_value: []const u8,
    array_value: std.ArrayList(InstanceValue),
    object_value: std.StringHashMap(InstanceValue),

    pub fn deinit(self: *InstanceValue, allocator: std.mem.Allocator) void {
        switch (self.*) {
            .string_value => |s| allocator.free(s),
            .array_value => |*arr| {
                for (arr.items) |*item| {
                    item.deinit(allocator);
                }
                arr.deinit(allocator);
            },
            .object_value => |*obj| {
                var it = obj.iterator();
                while (it.next()) |entry| {
                    allocator.free(entry.key_ptr.*);
                    entry.value_ptr.deinit(allocator);
                }
                obj.deinit();
            },
            else => {},
        }
    }
};

/// An instance of a type with concrete values
pub const Instance = struct {
    type_ref: []const u8,
    name: []const u8,
    description: []const u8,
    value: InstanceValue,

    pub fn deinit(self: *Instance, allocator: std.mem.Allocator) void {
        allocator.free(self.type_ref);
        allocator.free(self.name);
        allocator.free(self.description);
        self.value.deinit(allocator);
    }
};

/// Layout types for visualization
pub const LayoutType = enum {
    tree,
    grid,
    flow,
    circular,
};

/// Visualization configuration
pub const Visualization = struct {
    name: []const u8,
    layout: LayoutType,
    root_types: std.ArrayList([]const u8),
    show_instances: bool,
    animation_speed: f32,
    camera: struct {
        zoom: f32,
        center: Vec2,
    },

    pub fn deinit(self: *Visualization, allocator: std.mem.Allocator) void {
        allocator.free(self.name);
        for (self.root_types.items) |type_id| {
            allocator.free(type_id);
        }
        self.root_types.deinit(allocator);
    }
};

/// Complete type system
pub const TypeSystem = struct {
    allocator: std.mem.Allocator,
    types: std.StringHashMap(*Type),
    instances: std.ArrayList(Instance),
    visualizations: std.ArrayList(Visualization),

    pub fn init(allocator: std.mem.Allocator) TypeSystem {
        return .{
            .allocator = allocator,
            .types = std.StringHashMap(*Type).init(allocator),
            .instances = std.ArrayList(Instance){},
            .visualizations = std.ArrayList(Visualization){},
        };
    }

    pub fn deinit(self: *TypeSystem) void {
        var type_it = self.types.iterator();
        while (type_it.next()) |entry| {
            self.allocator.free(entry.key_ptr.*);
            entry.value_ptr.*.deinit(self.allocator);
            self.allocator.destroy(entry.value_ptr.*);
        }
        self.types.deinit();

        for (self.instances.items) |*instance| {
            instance.deinit(self.allocator);
        }
        self.instances.deinit(self.allocator);

        for (self.visualizations.items) |*viz| {
            viz.deinit(self.allocator);
        }
        self.visualizations.deinit(self.allocator);
    }

    pub fn getType(self: *TypeSystem, id: []const u8) ?*Type {
        return self.types.get(id);
    }

    pub fn addType(self: *TypeSystem, typ: Type) !void {
        const id_copy = try self.allocator.dupe(u8, typ.id);
        const type_ptr = try self.allocator.create(Type);
        type_ptr.* = typ;
        try self.types.put(id_copy, type_ptr);
    }
};
