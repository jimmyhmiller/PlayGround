const std = @import("std");
const Value = @import("../value.zig").Value;
const ArrayList = std.ArrayList;
const HashMap = std.HashMap;

// Enhanced type system for bidirectional checking
pub const Type = union(enum) {
    // Primitive types
    int,    // Keep for compatibility
    float,  // Keep for compatibility
    string,
    bool,
    nil,

    // Specific integer types
    u8,
    u16,
    u32,
    u64,
    usize,
    i8,
    i16,
    i32,
    i64,
    isize,

    // Specific float types
    f32,
    f64,

    // Composite types
    function: *FunctionType,
    vector: *Type,
    map: *MapType,

    // Type variables for inference
    type_var: u32,

    pub fn format(self: Type, comptime fmt: []const u8, options: std.fmt.FormatOptions, writer: anytype) !void {
        _ = fmt;
        _ = options;
        switch (self) {
            .int => try writer.print("Int"),
            .float => try writer.print("Float"),
            .string => try writer.print("String"),
            .bool => try writer.print("Bool"),
            .nil => try writer.print("Nil"),
            .u8 => try writer.print("U8"),
            .u16 => try writer.print("U16"),
            .u32 => try writer.print("U32"),
            .u64 => try writer.print("U64"),
            .usize => try writer.print("Usize"),
            .i8 => try writer.print("I8"),
            .i16 => try writer.print("I16"),
            .i32 => try writer.print("I32"),
            .i64 => try writer.print("I64"),
            .isize => try writer.print("Isize"),
            .f32 => try writer.print("F32"),
            .f64 => try writer.print("F64"),
            .function => |f| {
                try writer.print("(-> [");
                for (f.param_types, 0..) |param, i| {
                    if (i > 0) try writer.print(" ");
                    try param.format("", .{}, writer);
                }
                try writer.print("] ");
                try f.return_type.format("", .{}, writer);
                try writer.print(")");
            },
            .vector => |elem_type| {
                try writer.print("[");
                try elem_type.format("", .{}, writer);
                try writer.print("]");
            },
            .map => |m| {
                try writer.print("{");
                try m.key_type.format("", .{}, writer);
                try writer.print(" ");
                try m.value_type.format("", .{}, writer);
                try writer.print("}");
            },
            .type_var => |id| try writer.print("?{d}", .{id}),
        }
    }
};

pub const FunctionType = struct {
    param_types: []const Type,
    return_type: Type,
};

pub const MapType = struct {
    key_type: Type,
    value_type: Type,
};

// Typed expression - decorates AST nodes with their inferred types
pub const TypedExpression = struct {
    value: *Value,
    type: Type,

    pub fn init(allocator: *std.mem.Allocator, value: *Value, type_info: Type) !*TypedExpression {
        const typed_expr = try allocator.create(TypedExpression);
        typed_expr.* = TypedExpression{
            .value = value,
            .type = type_info,
        };
        return typed_expr;
    }
};

// Typed value - a fully annotated AST where every node has a type
pub const TypedValue = union(enum) {
    int: struct { value: i64, type: Type },
    float: struct { value: f64, type: Type },
    string: struct { value: []const u8, type: Type },
    nil: struct { type: Type },
    symbol: struct { name: []const u8, type: Type },
    keyword: struct { name: []const u8, type: Type },
    list: struct { elements: []*TypedValue, type: Type },
    vector: struct { elements: []*TypedValue, type: Type },
    map: struct { entries: []*MapEntry, type: Type },

    pub const MapEntry = struct {
        key: *TypedValue,
        value: *TypedValue,
    };

    pub fn getType(self: *const TypedValue) Type {
        return switch (self.*) {
            .int => |v| v.type,
            .float => |v| v.type,
            .string => |v| v.type,
            .nil => |v| v.type,
            .symbol => |v| v.type,
            .keyword => |v| v.type,
            .list => |v| v.type,
            .vector => |v| v.type,
            .map => |v| v.type,
        };
    }
};

// Type environment for variable bindings
pub const TypeEnv = HashMap([]const u8, Type, StringContext, std.hash_map.default_max_load_percentage);

const StringContext = struct {
    pub fn hash(self: @This(), s: []const u8) u64 {
        _ = self;
        return std.hash_map.hashString(s);
    }

    pub fn eql(self: @This(), a: []const u8, b: []const u8) bool {
        _ = self;
        return std.mem.eql(u8, a, b);
    }
};

// Type checking errors
pub const TypeCheckError = error{
    UnboundVariable,
    TypeMismatch,
    CannotSynthesize,
    CannotApplyNonFunction,
    ArgumentCountMismatch,
    InvalidTypeAnnotation,
    OutOfMemory,
};

// Bidirectional type checker
pub const BidirectionalTypeChecker = struct {
    allocator: *std.mem.Allocator,
    env: TypeEnv,
    next_var_id: u32,

    pub fn init(allocator: *std.mem.Allocator) BidirectionalTypeChecker {
        return BidirectionalTypeChecker{
            .allocator = allocator,
            .env = TypeEnv.init(allocator.*),
            .next_var_id = 0,
        };
    }

    pub fn deinit(self: *BidirectionalTypeChecker) void {
        self.env.deinit();
    }

    // Fresh type variable generation
    fn freshVar(self: *BidirectionalTypeChecker) Type {
        const id = self.next_var_id;
        self.next_var_id += 1;
        return Type{ .type_var = id };
    }

    // Synthesis mode: expr ⇒ type
    pub fn synthesize(self: *BidirectionalTypeChecker, expr: *Value) TypeCheckError!*TypedExpression {
        switch (expr.*) {
            .int => |i| {
                _ = i;
                return try TypedExpression.init(self.allocator, expr, Type.int);
            },

            .float => |f| {
                _ = f;
                return try TypedExpression.init(self.allocator, expr, Type.float);
            },

            .string => |s| {
                _ = s;
                return try TypedExpression.init(self.allocator, expr, Type.string);
            },

            .nil => {
                return try TypedExpression.init(self.allocator, expr, Type.nil);
            },

            .symbol => |name| {
                if (self.env.get(name)) |var_type| {
                    return try TypedExpression.init(self.allocator, expr, var_type);
                } else {
                    return TypeCheckError.UnboundVariable;
                }
            },

            .list => |list| {
                if (list.isEmpty()) {
                    // Empty list - could be any list type
                    const elem_type = self.freshVar();
                    const list_type = try self.allocator.create(Type);
                    list_type.* = elem_type;
                    return try TypedExpression.init(self.allocator, expr, Type{ .vector = list_type });
                }

                // Check if it's a special form
                if (list.value) |first| {
                    if (first.isSymbol()) {
                        if (std.mem.eql(u8, first.symbol, "def")) {
                            return try self.synthesizeDef(expr, list);
                        } else if (std.mem.eql(u8, first.symbol, "fn")) {
                            return TypeCheckError.CannotSynthesize; // Functions need type annotations
                        } else if (std.mem.eql(u8, first.symbol, "+") or
                                   std.mem.eql(u8, first.symbol, "-") or
                                   std.mem.eql(u8, first.symbol, "*") or
                                   std.mem.eql(u8, first.symbol, "/") or
                                   std.mem.eql(u8, first.symbol, "%")) {
                            return try self.synthesizeArithmetic(expr, list);
                        }
                    }
                }

                // Function application
                return try self.synthesizeApplication(expr, list);
            },

            .vector => |vec| {
                if (vec.len() == 0) {
                    const elem_type = self.freshVar();
                    const vector_type = try self.allocator.create(Type);
                    vector_type.* = elem_type;
                    return try TypedExpression.init(self.allocator, expr, Type{ .vector = vector_type });
                }

                // Synthesize type of first element and check others against it
                const first_elem = vec.at(0);
                const first_typed = try self.synthesize(first_elem);

                var i: usize = 1;
                while (i < vec.len()) {
                    const elem = vec.at(i);
                    _ = try self.check(elem, first_typed.type);
                    i += 1;
                }

                const elem_type = try self.allocator.create(Type);
                elem_type.* = first_typed.type;
                return try TypedExpression.init(self.allocator, expr, Type{ .vector = elem_type });
            },

            .map => {
                // For now, return a generic map type
                const key_type = try self.allocator.create(Type);
                const value_type = try self.allocator.create(Type);
                key_type.* = self.freshVar();
                value_type.* = self.freshVar();

                const map_type = try self.allocator.create(MapType);
                map_type.* = MapType{
                    .key_type = key_type.*,
                    .value_type = value_type.*,
                };

                return try TypedExpression.init(self.allocator, expr, Type{ .map = map_type });
            },

            .keyword => {
                return try TypedExpression.init(self.allocator, expr, Type.string);
            },
        }
    }

    // Checking mode: expr ⇐ type
    pub fn check(self: *BidirectionalTypeChecker, expr: *Value, expected: Type) TypeCheckError!*TypedExpression {
        switch (expr.*) {
            .list => |list| {
                if (list.value) |first| {
                    if (first.isSymbol() and std.mem.eql(u8, first.symbol, "fn")) {
                        return try self.checkFunction(expr, list, expected);
                    }
                }

                // Fall back to synthesis and subtype check
                const synthesized = try self.synthesize(expr);
                if (try self.isSubtype(synthesized.type, expected)) {
                    return try TypedExpression.init(self.allocator, expr, expected);
                } else {
                    return TypeCheckError.TypeMismatch;
                }
            },

            else => {
                // Fall back to synthesis and subtype check
                const synthesized = try self.synthesize(expr);
                if (try self.isSubtype(synthesized.type, expected)) {
                    return try TypedExpression.init(self.allocator, expr, expected);
                } else {
                    return TypeCheckError.TypeMismatch;
                }
            }
        }
    }

    // Type checking for def forms: (def name (: type) body)
    fn synthesizeDef(self: *BidirectionalTypeChecker, expr: *Value, list: anytype) TypeCheckError!*TypedExpression {
        var current: ?*const @TypeOf(list.*) = list.next;

        // Get variable name
        const name_node = current orelse return TypeCheckError.InvalidTypeAnnotation;
        if (!name_node.value.?.isSymbol()) return TypeCheckError.InvalidTypeAnnotation;
        const var_name = name_node.value.?.symbol;
        current = name_node.next;

        // Get type annotation (: type)
        const type_annotation_node = current orelse return TypeCheckError.InvalidTypeAnnotation;
        const annotated_type = try self.parseTypeAnnotation(type_annotation_node.value.?);
        current = type_annotation_node.next;

        // Get body
        const body_node = current orelse return TypeCheckError.InvalidTypeAnnotation;
        const body = body_node.value.?;

        // Check body against annotated type
        _ = try self.check(body, annotated_type);

        // Add binding to environment
        try self.env.put(var_name, annotated_type);

        return try TypedExpression.init(self.allocator, expr, annotated_type);
    }

    // Type checking for function expressions: (fn [params] body)
    fn checkFunction(self: *BidirectionalTypeChecker, expr: *Value, list: anytype, expected: Type) TypeCheckError!*TypedExpression {
        if (expected != .function) return TypeCheckError.TypeMismatch;

        var current = list.next; // Skip 'fn'

        // Get parameter list
        const param_list_node = current orelse return TypeCheckError.InvalidTypeAnnotation;
        if (!param_list_node.value.?.isVector()) return TypeCheckError.InvalidTypeAnnotation;
        const param_vector = param_list_node.value.?.vector;
        current = param_list_node.next;

        // Get body
        const body_node = current orelse return TypeCheckError.InvalidTypeAnnotation;
        const body = body_node.value.?;

        // Check parameter count matches
        if (param_vector.len() != expected.function.param_types.len) {
            return TypeCheckError.ArgumentCountMismatch;
        }

        // Create new environment with parameter bindings
        var old_env = self.env;
        self.env = TypeEnv.init(self.allocator.*);

        // Copy old bindings
        var iter = old_env.iterator();
        while (iter.next()) |entry| {
            try self.env.put(entry.key_ptr.*, entry.value_ptr.*);
        }

        // Add parameter bindings
        var i: usize = 0;
        while (i < param_vector.len()) {
            const param = param_vector.at(i);
            if (!param.isSymbol()) return TypeCheckError.InvalidTypeAnnotation;
            try self.env.put(param.symbol, expected.function.param_types[i]);
            i += 1;
        }

        // Check body against return type
        const typed_body = try self.check(body, expected.function.return_type);
        _ = typed_body;

        // Restore environment
        self.env.deinit();
        self.env = old_env;

        return try TypedExpression.init(self.allocator, expr, expected);
    }

    // Typed function checking for checkTyped method
    fn checkFunctionTyped(self: *BidirectionalTypeChecker, expr: *Value, list: anytype, expected: Type) TypeCheckError!*TypedValue {
        if (expected != .function) return TypeCheckError.TypeMismatch;

        var current = list.next; // Skip 'fn'

        // Get parameter list
        const param_list_node = current orelse return TypeCheckError.InvalidTypeAnnotation;
        if (!param_list_node.value.?.isVector()) return TypeCheckError.InvalidTypeAnnotation;
        const param_vector = param_list_node.value.?.vector;
        current = param_list_node.next;

        // Get body
        const body_node = current orelse return TypeCheckError.InvalidTypeAnnotation;
        const body = body_node.value.?;

        // Check parameter count matches
        if (param_vector.len() != expected.function.param_types.len) {
            return TypeCheckError.ArgumentCountMismatch;
        }

        // Create new environment with parameter bindings
        var old_env = self.env;
        self.env = TypeEnv.init(self.allocator.*);

        // Copy old bindings
        var iter = old_env.iterator();
        while (iter.next()) |entry| {
            try self.env.put(entry.key_ptr.*, entry.value_ptr.*);
        }

        // Add parameter bindings
        var i: usize = 0;
        while (i < param_vector.len()) {
            const param = param_vector.at(i);
            if (!param.isSymbol()) return TypeCheckError.InvalidTypeAnnotation;
            try self.env.put(param.symbol, expected.function.param_types[i]);
            i += 1;
        }

        // Check body against return type
        _ = try self.checkTyped(body, expected.function.return_type);

        // Restore environment
        self.env.deinit();
        self.env = old_env;

        // Create the typed function value
        const result = try self.allocator.create(TypedValue);
        result.* = TypedValue{ .list = .{
            .elements = &[0]*TypedValue{}, // TODO: Add proper function representation
            .type = expected
        } };
        _ = expr; // Acknowledge unused parameter
        return result;
    }

    // Function application type checking
    fn synthesizeApplication(self: *BidirectionalTypeChecker, expr: *Value, list: anytype) TypeCheckError!*TypedExpression {
        var current: ?*const @TypeOf(list.*) = list;

        // Get function
        const func_node = (current orelse return TypeCheckError.CannotSynthesize).value orelse return TypeCheckError.CannotSynthesize;
        const func_typed = try self.synthesize(func_node);

        if (func_typed.type != .function) {
            return TypeCheckError.CannotApplyNonFunction;
        }

        if (current) |curr_node| {
            if (curr_node.next) |next_node| {
                current = next_node;
            } else {
                return TypeCheckError.CannotApplyNonFunction;
            }
        } else {
            return TypeCheckError.CannotApplyNonFunction;
        }

        // Check arguments against parameter types
        var arg_index: usize = 0;
        while (current != null and arg_index < func_typed.type.function.param_types.len) {
            const arg = (current orelse break).value orelse break;
            const param_type = func_typed.type.function.param_types[arg_index];
            _ = try self.check(arg, param_type);

            if (current) |curr_node| {
                current = curr_node.next;
            }
            arg_index += 1;
        }

        if (arg_index != func_typed.type.function.param_types.len) {
            return TypeCheckError.ArgumentCountMismatch;
        }

        return try TypedExpression.init(self.allocator, expr, func_typed.type.function.return_type);
    }

    // Arithmetic operations type checking
    fn synthesizeArithmetic(self: *BidirectionalTypeChecker, expr: *Value, list: anytype) TypeCheckError!*TypedExpression {
        var current: ?*const @TypeOf(list.*) = list.next; // Skip operator

        // Get first operand to determine type
        const first_arg = (current orelse return TypeCheckError.InvalidTypeAnnotation).value orelse return TypeCheckError.InvalidTypeAnnotation;
        const first_typed = try self.synthesize(first_arg);

        if (!isNumericType(first_typed.type)) {
            return TypeCheckError.TypeMismatch;
        }

        var result_type = first_typed.type;
        current = (current orelse return TypeCheckError.InvalidTypeAnnotation).next;

        // Get the operator
        const op_node = list.value orelse return TypeCheckError.InvalidTypeAnnotation;
        if (!op_node.isSymbol()) return TypeCheckError.InvalidTypeAnnotation;
        const op = op_node.symbol;

        // For division of integers, result should be float
        if (std.mem.eql(u8, op, "/") and isIntegerType(result_type)) {
            result_type = if (std.meta.activeTag(result_type) == .int) Type.float else Type.f64;
        }

        // For modulo, ensure integer type
        if (std.mem.eql(u8, op, "%") and !isIntegerType(first_typed.type)) {
            return TypeCheckError.TypeMismatch;
        }

        // Check remaining operands
        while (current != null) {
            const arg = (current orelse break).value orelse break;
            _ = try self.check(arg, first_typed.type);
            if (current) |curr_node| {
                current = curr_node.next;
            }
        }

        return try TypedExpression.init(self.allocator, expr, result_type);
    }

    // Parse type annotations: (: (-> [Int] Int))
    pub fn parseTypeAnnotation(self: *BidirectionalTypeChecker, annotation: *Value) TypeCheckError!Type {
        if (!annotation.isList()) return TypeCheckError.InvalidTypeAnnotation;

        var current: ?*const @TypeOf(annotation.list.*) = annotation.list;

        // Expect ':' (which is parsed as a keyword, not a symbol)
        const colon = (current orelse return TypeCheckError.InvalidTypeAnnotation).value orelse return TypeCheckError.InvalidTypeAnnotation;
        if (!colon.isKeyword() or !std.mem.eql(u8, colon.keyword, "")) {
            return TypeCheckError.InvalidTypeAnnotation;
        }

        if (current) |curr_node| {
            if (curr_node.next) |next_node| {
                current = next_node;
            } else {
                return TypeCheckError.InvalidTypeAnnotation;
            }
        } else {
            return TypeCheckError.InvalidTypeAnnotation;
        }
        const type_expr = (current orelse return TypeCheckError.InvalidTypeAnnotation).value orelse return TypeCheckError.InvalidTypeAnnotation;

        return try self.parseType(type_expr);
    }

    // Parse type expressions
    fn parseType(self: *BidirectionalTypeChecker, type_expr: *Value) TypeCheckError!Type {
        if (type_expr.isSymbol()) {
            const type_name = type_expr.symbol;
            // Legacy types
            if (std.mem.eql(u8, type_name, "Int")) return Type.int;
            if (std.mem.eql(u8, type_name, "Float")) return Type.float;
            if (std.mem.eql(u8, type_name, "String")) return Type.string;
            if (std.mem.eql(u8, type_name, "Bool")) return Type.bool;
            if (std.mem.eql(u8, type_name, "Nil")) return Type.nil;

            // Specific integer types
            if (std.mem.eql(u8, type_name, "U8")) return Type.u8;
            if (std.mem.eql(u8, type_name, "U16")) return Type.u16;
            if (std.mem.eql(u8, type_name, "U32")) return Type.u32;
            if (std.mem.eql(u8, type_name, "U64")) return Type.u64;
            if (std.mem.eql(u8, type_name, "Usize")) return Type.usize;
            if (std.mem.eql(u8, type_name, "I8")) return Type.i8;
            if (std.mem.eql(u8, type_name, "I16")) return Type.i16;
            if (std.mem.eql(u8, type_name, "I32")) return Type.i32;
            if (std.mem.eql(u8, type_name, "I64")) return Type.i64;
            if (std.mem.eql(u8, type_name, "Isize")) return Type.isize;

            // Specific float types
            if (std.mem.eql(u8, type_name, "F32")) return Type.f32;
            if (std.mem.eql(u8, type_name, "F64")) return Type.f64;

            return TypeCheckError.InvalidTypeAnnotation;
        }

        if (type_expr.isList()) {
            var current: ?*const @TypeOf(type_expr.list.*) = type_expr.list;
            const head = (current orelse return TypeCheckError.InvalidTypeAnnotation).value orelse return TypeCheckError.InvalidTypeAnnotation;

            if (head.isSymbol()) {
                if (std.mem.eql(u8, head.symbol, "->")) {
                    // Function type: (-> [param_types] return_type)
                    if (current) |curr_node| {
            if (curr_node.next) |next_node| {
                current = next_node;
            } else {
                return TypeCheckError.InvalidTypeAnnotation;
            }
        } else {
            return TypeCheckError.InvalidTypeAnnotation;
        }
                    const param_list = (current orelse return TypeCheckError.InvalidTypeAnnotation).value orelse return TypeCheckError.InvalidTypeAnnotation;

                    if (!param_list.isVector()) return TypeCheckError.InvalidTypeAnnotation;

                    if (current) |curr_node| {
            if (curr_node.next) |next_node| {
                current = next_node;
            } else {
                return TypeCheckError.InvalidTypeAnnotation;
            }
        } else {
            return TypeCheckError.InvalidTypeAnnotation;
        }
                    const return_type_expr = (current orelse return TypeCheckError.InvalidTypeAnnotation).value orelse return TypeCheckError.InvalidTypeAnnotation;

                    // Parse parameter types
                    const param_count = param_list.vector.len();
                    const param_types = try self.allocator.alloc(Type, param_count);

                    var i: usize = 0;
                    while (i < param_count) {
                        const param_type_expr = param_list.vector.at(i);
                        param_types[i] = try self.parseType(param_type_expr);
                        i += 1;
                    }

                    // Parse return type
                    const return_type = try self.parseType(return_type_expr);

                    const func_type = try self.allocator.create(FunctionType);
                    func_type.* = FunctionType{
                        .param_types = param_types,
                        .return_type = return_type,
                    };

                    return Type{ .function = func_type };
                }
            }
        }

        if (type_expr.isVector()) {
            // Vector type: [element_type]
            if (type_expr.vector.len() != 1) return TypeCheckError.InvalidTypeAnnotation;
            const elem_type_expr = type_expr.vector.at(0);
            const elem_type = try self.parseType(elem_type_expr);

            const vector_elem_type = try self.allocator.create(Type);
            vector_elem_type.* = elem_type;

            return Type{ .vector = vector_elem_type };
        }

        return TypeCheckError.InvalidTypeAnnotation;
    }

    // Subtyping check
    fn isSubtype(self: *BidirectionalTypeChecker, sub: Type, super: Type) TypeCheckError!bool {
        _ = self;

        // Simple structural equality for now
        return typesEqual(sub, super);
    }

    // Check if a type is numeric
    fn isNumericType(t: Type) bool {
        return switch (t) {
            .int, .float,
            .u8, .u16, .u32, .u64, .usize,
            .i8, .i16, .i32, .i64, .isize,
            .f32, .f64 => true,
            else => false,
        };
    }

    // Check if a type is integer
    fn isIntegerType(t: Type) bool {
        return switch (t) {
            .int,
            .u8, .u16, .u32, .u64, .usize,
            .i8, .i16, .i32, .i64, .isize => true,
            else => false,
        };
    }

    // Check if a type is floating point
    fn isFloatType(t: Type) bool {
        return switch (t) {
            .float, .f32, .f64 => true,
            else => false,
        };
    }

    // Type equality
    pub fn typesEqual(a: Type, b: Type) bool {
        if (std.meta.activeTag(a) != std.meta.activeTag(b)) return false;

        switch (a) {
            .int, .float, .string, .bool, .nil,
            .u8, .u16, .u32, .u64, .usize,
            .i8, .i16, .i32, .i64, .isize,
            .f32, .f64 => return true,
            .function => |a_func| {
                const b_func = b.function;
                if (a_func.param_types.len != b_func.param_types.len) return false;

                for (a_func.param_types, b_func.param_types) |a_param, b_param| {
                    if (!typesEqual(a_param, b_param)) return false;
                }

                return typesEqual(a_func.return_type, b_func.return_type);
            },
            .vector => |a_elem| return typesEqual(a_elem.*, b.vector.*),
            .map => |a_map| {
                const b_map = b.map;
                return typesEqual(a_map.key_type, b_map.key_type) and
                       typesEqual(a_map.value_type, b_map.value_type);
            },
            .type_var => |a_id| return a_id == b.type_var,
        }
    }

    // Synthesis mode that produces a fully typed AST
    pub fn synthesizeTyped(self: *BidirectionalTypeChecker, expr: *Value) TypeCheckError!*TypedValue {
        const result = try self.allocator.create(TypedValue);

        switch (expr.*) {
            .int => |i| {
                result.* = TypedValue{ .int = .{ .value = i, .type = Type.int } };
                return result;
            },

            .float => |f| {
                result.* = TypedValue{ .float = .{ .value = f, .type = Type.float } };
                return result;
            },

            .string => |s| {
                result.* = TypedValue{ .string = .{ .value = s, .type = Type.string } };
                return result;
            },

            .nil => {
                result.* = TypedValue{ .nil = .{ .type = Type.nil } };
                return result;
            },

            .symbol => |name| {
                if (self.env.get(name)) |var_type| {
                    result.* = TypedValue{ .symbol = .{ .name = name, .type = var_type } };
                    return result;
                } else {
                    return TypeCheckError.UnboundVariable;
                }
            },

            .keyword => |name| {
                result.* = TypedValue{ .keyword = .{ .name = name, .type = Type.string } };
                return result;
            },

            .vector => |vec| {
                const elem_count = vec.len();
                const typed_elements = try self.allocator.alloc(*TypedValue, elem_count);

                if (elem_count == 0) {
                    const elem_type = self.freshVar();
                    const vector_type_ptr = try self.allocator.create(Type);
                    vector_type_ptr.* = elem_type;
                    result.* = TypedValue{ .vector = .{
                        .elements = typed_elements,
                        .type = Type{ .vector = vector_type_ptr }
                    } };
                    return result;
                }

                // Type check first element
                typed_elements[0] = try self.synthesizeTyped(vec.at(0));
                const elem_type = typed_elements[0].getType();

                // Check remaining elements against first element's type
                var i: usize = 1;
                while (i < elem_count) : (i += 1) {
                    typed_elements[i] = try self.checkTyped(vec.at(i), elem_type);
                }

                const vector_type_ptr = try self.allocator.create(Type);
                vector_type_ptr.* = elem_type;
                result.* = TypedValue{ .vector = .{
                    .elements = typed_elements,
                    .type = Type{ .vector = vector_type_ptr }
                } };
                return result;
            },

            .list => |list| {
                // Count elements first
                var count: usize = 0;
                var curr: ?*const @TypeOf(list.*) = list;
                while (curr != null) {
                    const node = curr.?;
                    if (node.value != null) count += 1;
                    curr = node.next;
                }

                const typed_elements = try self.allocator.alloc(*TypedValue, count);

                // Handle special forms
                if (list.value) |first| {
                    if (first.isSymbol()) {
                        if (std.mem.eql(u8, first.symbol, "+") or
                           std.mem.eql(u8, first.symbol, "-") or
                           std.mem.eql(u8, first.symbol, "*") or
                           std.mem.eql(u8, first.symbol, "/") or
                           std.mem.eql(u8, first.symbol, "%")) {
                            // Arithmetic operation - determine result type from first operand
                            var current: ?*const @TypeOf(list.*) = list.next; // Skip operator
                            if (current == null or current.?.value == null) {
                                return TypeCheckError.InvalidTypeAnnotation;
                            }

                            // Get type from first operand
                            const first_operand = current.?.value.?;
                            const first_typed = try self.synthesizeTyped(first_operand);
                            const result_type = first_typed.getType();

                            if (!isNumericType(result_type)) {
                                return TypeCheckError.TypeMismatch;
                            }

                            // For division, check special cases
                            if (std.mem.eql(u8, first.symbol, "/")) {
                                // Integer division should produce float result
                                if (isIntegerType(result_type)) {
                                    // Convert to appropriate float type
                                    _ = if (std.meta.activeTag(result_type) == .int) Type.float else Type.f64; // TODO: Use this for proper type conversion

                                    var idx: usize = 0;
                                    typed_elements[idx] = first_typed;
                                    idx += 1;
                                    current = current.?.next;

                                    while (current != null) {
                                        const node = current.?;
                                        if (node.value) |val| {
                                            typed_elements[idx] = try self.checkTyped(val, result_type);
                                            idx += 1;
                                        }
                                        current = node.next;
                                    }

                                    result.* = TypedValue{ .list = .{
                                        .elements = typed_elements[0..idx],
                                        .type = Type.float // float_result_type  // TODO: Fix this properly
                                    } };
                                    return result;
                                }
                            }

                            // For modulo, only allow integer types
                            if (std.mem.eql(u8, first.symbol, "%")) {
                                if (!isIntegerType(result_type)) {
                                    return TypeCheckError.TypeMismatch;
                                }
                            }

                            // Type check remaining operands against the first operand's type
                            var idx: usize = 0;
                            typed_elements[idx] = first_typed;
                            idx += 1;
                            current = current.?.next;

                            while (current != null) {
                                const node = current.?;
                                if (node.value) |val| {
                                    typed_elements[idx] = try self.checkTyped(val, result_type);
                                    idx += 1;
                                }
                                current = node.next;
                            }

                            result.* = TypedValue{ .list = .{
                                .elements = typed_elements[0..idx],
                                .type = result_type
                            } };
                            return result;
                        } else if (std.mem.eql(u8, first.symbol, "def")) {
                            // Handle def form: (def name (: type) body)
                            var current: ?*const @TypeOf(list.*) = list.next;

                            // Get variable name
                            const name_node = current orelse return TypeCheckError.InvalidTypeAnnotation;
                            if (!name_node.value.?.isSymbol()) return TypeCheckError.InvalidTypeAnnotation;
                            const var_name = name_node.value.?.symbol;
                            current = name_node.next;

                            // Get type annotation
                            const type_annotation_node = current orelse return TypeCheckError.InvalidTypeAnnotation;
                            const annotated_type = try self.parseTypeAnnotation(type_annotation_node.value.?);
                            current = type_annotation_node.next;

                            // Type check body
                            const body_node = current orelse return TypeCheckError.InvalidTypeAnnotation;
                            const body = body_node.value.?;
                            const typed_body = try self.checkTyped(body, annotated_type);

                            // Add binding to environment
                            try self.env.put(var_name, annotated_type);

                            // Return the typed body as the result
                            return typed_body;
                        }
                        // Add other special forms as needed
                    }
                }

                // Default: check if it's a function application
                if (list.value) |first_val| {
                    const first_typed = try self.synthesizeTyped(first_val);
                    if (first_typed.getType() == .function) {
                        // This is a function application - synthesize the application
                        const func_type = first_typed.getType().function;

                        // Count and check arguments
                        var arg_count: usize = 0;
                        var current: ?*const @TypeOf(list.*) = list.next;
                        while (current != null) {
                            if (current.?.value != null) {
                                arg_count += 1;
                            }
                            current = current.?.next;
                        }

                        if (arg_count != func_type.param_types.len) {
                            return TypeCheckError.ArgumentCountMismatch;
                        }

                        // Type check arguments
                        current = list.next;
                        var arg_idx: usize = 0;
                        while (current != null and arg_idx < func_type.param_types.len) {
                            const arg_node = current.?;
                            if (arg_node.value) |arg_val| {
                                _ = try self.checkTyped(arg_val, func_type.param_types[arg_idx]);
                            }
                            current = arg_node.next;
                            arg_idx += 1;
                        }

                        // Return the function's return type
                        result.* = TypedValue{ .list = .{
                            .elements = typed_elements[0..1], // Just store the function for now
                            .type = func_type.return_type
                        } };
                        return result;
                    }
                }

                // Default: type check all elements
                var idx: usize = 0;
                var current: ?*const @TypeOf(list.*) = list;
                while (current != null) {
                    const node = current.?;
                    if (node.value) |val| {
                        typed_elements[idx] = try self.synthesizeTyped(val);
                        idx += 1;
                    }
                    current = node.next;
                }

                // For now, lists have a generic type
                result.* = TypedValue{ .list = .{
                    .elements = typed_elements,
                    .type = self.freshVar()
                } };
                return result;
            },

            .map => {
                // Maps not fully implemented yet
                result.* = TypedValue{ .map = .{
                    .entries = try self.allocator.alloc(*TypedValue.MapEntry, 0),
                    .type = Type{ .map = try self.allocator.create(MapType) }
                } };
                return result;
            },
        }
    }

    // Checking mode that produces a fully typed AST
    pub fn checkTyped(self: *BidirectionalTypeChecker, expr: *Value, expected: Type) TypeCheckError!*TypedValue {
        // Handle special forms that need expected type
        switch (expr.*) {
            .list => |list| {
                if (list.value) |first| {
                    if (first.isSymbol() and std.mem.eql(u8, first.symbol, "fn")) {
                        return try self.checkFunctionTyped(expr, list, expected);
                    }
                }
                // Fall through to synthesis + subtype check
            },
            else => {
                // Fall through to synthesis + subtype check
            },
        }

        // First synthesize, then verify it matches expected type
        const typed = try self.synthesizeTyped(expr);
        const actual = typed.getType();

        if (!try self.isSubtype(actual, expected)) {
            return TypeCheckError.TypeMismatch;
        }

        // Update the type to the expected type (for subsumption)
        switch (typed.*) {
            .int => |*v| v.type = expected,
            .float => |*v| v.type = expected,
            .string => |*v| v.type = expected,
            .nil => |*v| v.type = expected,
            .symbol => |*v| v.type = expected,
            .keyword => |*v| v.type = expected,
            .list => |*v| v.type = expected,
            .vector => |*v| v.type = expected,
            .map => |*v| v.type = expected,
        }

        return typed;
    }

    // Public interface for type checking a complete expression (fully typed)
    pub fn typeCheck(self: *BidirectionalTypeChecker, expr: *Value) TypeCheckError!*TypedValue {
        return try self.synthesizeTyped(expr);
    }

    // Type check multiple top-level expressions and return fully typed ASTs
    pub fn typeCheckAll(self: *BidirectionalTypeChecker, expressions: []const *Value) !ArrayList(*TypedValue) {
        var results = ArrayList(*TypedValue){};

        for (expressions) |expr| {
            const typed = try self.synthesizeTyped(expr);
            try results.append(self.allocator.*, typed);
        }

        return results;
    }

    // Two-pass type checking for forward references (fully typed)
    pub fn typeCheckAllTwoPass(self: *BidirectionalTypeChecker, expressions: []const *Value) !ArrayList(*TypedValue) {
        // Pass 1: Collect all top-level definitions and their type signatures
        for (expressions) |expr| {
            if (expr.isList()) {
                var current: ?*const @TypeOf(expr.list.*) = expr.list;
                if (current) |node| {
                    if (node.value) |first| {
                        if (first.isSymbol() and std.mem.eql(u8, first.symbol, "def")) {
                            // Extract name and type annotation without checking body
                            current = node.next;

                            // Get variable name
                            const name_node = current orelse continue;
                            if (!name_node.value.?.isSymbol()) continue;
                            const var_name = name_node.value.?.symbol;
                            current = name_node.next;

                            // Get type annotation
                            const type_annotation_node = current orelse continue;
                            const annotated_type = self.parseTypeAnnotation(type_annotation_node.value.?) catch continue;

                            // Add to environment for forward references
                            try self.env.put(var_name, annotated_type);
                        }
                    }
                }
            }
        }

        // Pass 2: Type check all expressions with forward references available
        var results = ArrayList(*TypedValue){};
        for (expressions) |expr| {
            if (expr.isList()) {
                var current: ?*const @TypeOf(expr.list.*) = expr.list;
                if (current) |node| {
                    if (node.value) |first| {
                        if (first.isSymbol() and std.mem.eql(u8, first.symbol, "def")) {
                            // For def expressions, just check the body against the already-established type
                            current = node.next;
                            // Get variable name
                            const name_node = current orelse continue;
                            if (!name_node.value.?.isSymbol()) continue;
                            _ = name_node.value.?.symbol;
                            current = name_node.next;

                            // Get type annotation
                            const type_annotation_node = current orelse continue;
                            const annotated_type = self.parseTypeAnnotation(type_annotation_node.value.?) catch continue;
                            current = type_annotation_node.next;

                            // Get body and type check it
                            const body_node = current orelse continue;
                            const body = body_node.value.?;
                            const typed_body = try self.checkTyped(body, annotated_type);

                            try results.append(self.allocator.*, typed_body);
                            continue;
                        }
                    }
                }
            }
            // For non-def expressions, use normal synthesis
            const typed = try self.synthesizeTyped(expr);
            try results.append(self.allocator.*, typed);
        }

        return results;
    }
};

// Utility functions for creating types
pub fn createIntType(allocator: *std.mem.Allocator) !*Type {
    const type_ptr = try allocator.create(Type);
    type_ptr.* = Type.int;
    return type_ptr;
}

pub fn createFunctionType(allocator: *std.mem.Allocator, param_types: []const Type, return_type: Type) !Type {
    const func_type = try allocator.create(FunctionType);
    func_type.* = FunctionType{
        .param_types = param_types,
        .return_type = return_type,
    };
    return Type{ .function = func_type };
}