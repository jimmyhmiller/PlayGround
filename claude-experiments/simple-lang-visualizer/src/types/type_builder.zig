const std = @import("std");
const ts = @import("type_system.zig");

/// Type builder for incremental construction
pub const TypeBuilder = struct {
    type_system: *ts.TypeSystem,
    allocator: std.mem.Allocator,

    pub fn init(type_system: *ts.TypeSystem, allocator: std.mem.Allocator) TypeBuilder {
        return .{
            .type_system = type_system,
            .allocator = allocator,
        };
    }

    /// Create a new type
    pub fn createType(
        self: *TypeBuilder,
        type_id: []const u8,
        name: []const u8,
        kind: ts.TypeKind,
        position: ts.Vec2,
        color: ts.Color,
    ) !void {
        const description = try std.fmt.allocPrint(
            self.allocator,
            "{s} type",
            .{name},
        );
        errdefer self.allocator.free(description);

        const definition: ts.TypeDefinition = switch (kind) {
            .primitive => .{ .primitive = {} },
            .@"struct" => .{ .@"struct" = .{ .fields = std.SegmentedList(ts.StructField, 8){} } },
            .@"enum" => .{ .@"enum" = .{ .variants = std.SegmentedList(ts.EnumVariant, 8){} } },
            .function => .{
                .function = .{
                    .parameters = std.SegmentedList(ts.FunctionParameter, 8){},
                    .return_type = try self.allocator.dupe(u8, "void"),
                },
            },
            .tuple => .{ .tuple = .{ .elements = std.SegmentedList(ts.TupleElement, 8){} } },
            .optional => .{
                .optional = .{
                    .inner_type = try self.allocator.dupe(u8, "T"),
                },
            },
            .recursive => .{
                .recursive = .{
                    .base_type = try self.allocator.dupe(u8, "T"),
                    .self_reference = try self.allocator.dupe(u8, type_id),
                },
            },
        };

        const typ = ts.Type{
            .id = try self.allocator.dupe(u8, type_id),
            .name = try self.allocator.dupe(u8, name),
            .description = description,
            .metadata = ts.Metadata.initAppearing(color, position),
            .definition = definition,
        };

        try self.type_system.addType(typ);
    }

    /// Add a field to a struct
    pub fn addStructField(
        self: *TypeBuilder,
        type_id: []const u8,
        field_name: []const u8,
        field_type: []const u8,
        description: []const u8,
    ) !void {
        const typ = self.type_system.getType(type_id) orelse return error.TypeNotFound;

        if (typ.definition != .@"struct") return error.NotAStruct;

        const field = try ts.StructField.initAppearing(
            self.allocator,
            field_name,
            field_type,
            description,
        );

        try typ.definition.@"struct".fields.append(self.allocator, field);
    }

    /// Add a variant to an enum
    pub fn addEnumVariant(
        self: *TypeBuilder,
        type_id: []const u8,
        variant_name: []const u8,
        payload: ?[]const u8,
        description: []const u8,
    ) !void {
        const typ = self.type_system.getType(type_id) orelse return error.TypeNotFound;

        if (typ.definition != .@"enum") return error.NotAnEnum;

        const variant = try ts.EnumVariant.initAppearing(
            self.allocator,
            variant_name,
            payload,
            description,
        );

        try typ.definition.@"enum".variants.append(self.allocator, variant);
    }

    /// Add a parameter to a function
    pub fn addFunctionParameter(
        self: *TypeBuilder,
        type_id: []const u8,
        param_name: []const u8,
        param_type: []const u8,
    ) !void {
        const typ = self.type_system.getType(type_id) orelse return error.TypeNotFound;

        if (typ.definition != .function) return error.NotAFunction;

        const param = try ts.FunctionParameter.initAppearing(
            self.allocator,
            param_name,
            param_type,
        );

        try typ.definition.function.parameters.append(self.allocator, param);
    }

    /// Set function return type
    pub fn setFunctionReturnType(
        self: *TypeBuilder,
        type_id: []const u8,
        return_type: []const u8,
    ) !void {
        const typ = self.type_system.getType(type_id) orelse return error.TypeNotFound;

        if (typ.definition != .function) return error.NotAFunction;

        // Free old return type
        self.allocator.free(typ.definition.function.return_type);

        // Set new return type
        typ.definition.function.return_type = try self.allocator.dupe(u8, return_type);
    }

    /// Add an element to a tuple
    pub fn addTupleElement(
        self: *TypeBuilder,
        type_id: []const u8,
        element_type: []const u8,
        description: []const u8,
    ) !void {
        const typ = self.type_system.getType(type_id) orelse return error.TypeNotFound;

        if (typ.definition != .tuple) return error.NotATuple;

        const index = typ.definition.tuple.elements.len;
        const element = try ts.TupleElement.initAppearing(
            self.allocator,
            index,
            element_type,
            description,
        );

        try typ.definition.tuple.elements.append(self.allocator, element);
    }

    /// Update type position (for auto-layout)
    pub fn updatePosition(
        self: *TypeBuilder,
        type_id: []const u8,
        position: ts.Vec2,
    ) !void {
        const typ = self.type_system.getType(type_id) orelse return error.TypeNotFound;
        typ.metadata.position = position;
    }

    /// Update type color
    pub fn updateColor(
        self: *TypeBuilder,
        type_id: []const u8,
        color: ts.Color,
    ) !void {
        const typ = self.type_system.getType(type_id) orelse return error.TypeNotFound;
        typ.metadata.color = color;
    }

    /// Delete a type
    pub fn deleteType(
        self: *TypeBuilder,
        type_id: []const u8,
    ) !void {
        const typ = self.type_system.getType(type_id) orelse return error.TypeNotFound;
        typ.deinit(self.allocator);
        self.allocator.destroy(typ);
        _ = self.type_system.types.remove(type_id);
    }
};
