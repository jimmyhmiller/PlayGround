const std = @import("std");
const Value = @import("value.zig").Value;
const ArrayList = std.ArrayList;
const HashMap = std.HashMap;

// Enhanced type system for bidirectional checking
pub const Type = union(enum) {
    // Primitive types
    int, // Keep for compatibility
    float, // Keep for compatibility
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
    struct_type: *StructType,
    enum_type: *EnumType,
    pointer: *Type,
    array: *ArrayType,

    // Meta-type
    type_type, // The type of types

    // Type variables for inference
    type_var: u32,

    // FFI types
    extern_function: *ExternFunctionType,
    extern_type: *ExternType,
    c_string, // Pointer to null-terminated C string
    void, // Void type for void pointers and returns

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
            .struct_type => |s| {
                try writer.print("(Struct");
                for (s.fields) |field| {
                    try writer.print(" [");
                    try writer.print("{s}", .{field.name});
                    try writer.print(" ");
                    try field.field_type.format("", .{}, writer);
                    try writer.print("]");
                }
                try writer.print(")");
            },
            .enum_type => |e| {
                try writer.print("(Enum");
                for (e.variants) |variant| {
                    try writer.print(" {s}", .{variant.name});
                }
                try writer.print(")");
            },
            .pointer => |pointee| {
                try writer.print("(Pointer ");
                try pointee.format("", .{}, writer);
                try writer.print(")");
            },
            .array => |a| {
                try writer.print("(Array ");
                try a.element_type.format("", .{}, writer);
                try writer.print(" {d})", .{a.size});
            },
            .type_type => try writer.print("Type"),
            .type_var => |id| try writer.print("?{d}", .{id}),
            .extern_function => |f| {
                try writer.print("(extern-fn {s} [", .{f.name});
                for (f.param_types, 0..) |param, i| {
                    if (i > 0) try writer.print(" ");
                    try param.format("", .{}, writer);
                }
                if (f.variadic) try writer.print(" ...");
                try writer.print("] -> ");
                try f.return_type.format("", .{}, writer);
                try writer.print(")");
            },
            .extern_type => |t| try writer.print("(extern-type {s})", .{t.name}),
            .c_string => try writer.print("CString"),
            .void => try writer.print("Void"),
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

pub const StructField = struct {
    name: []const u8,
    field_type: Type,
};

pub const StructType = struct {
    name: []const u8,
    fields: []const StructField,
};

pub const EnumVariant = struct {
    name: []const u8,
    qualified_name: ?[]const u8 = null,
};

pub const EnumType = struct {
    name: []const u8,
    variants: []EnumVariant,
};

pub const ExternFunctionType = struct {
    name: []const u8,
    param_types: []const Type,
    return_type: Type,
    variadic: bool,
};

pub const ExternType = struct {
    name: []const u8,
    is_opaque: bool,
    is_union: bool = false,
    fields: ?[]const StructField = null, // For extern-struct with known fields
};

pub const ArrayType = struct {
    element_type: Type,
    size: usize,
};

// Typed expression - decorates AST nodes with their inferred types
pub const TypedExpression = struct {
    value: *Value,
    type: Type,

    pub fn init(allocator: std.mem.Allocator, value: *Value, type_info: Type) !*TypedExpression {
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
    type_value: struct { value_type: Type, type: Type }, // A type as a value
    namespace: struct { name: []const u8, type: Type },
    struct_instance: struct { field_values: []*TypedValue, type: Type },

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
            .type_value => |v| v.type,
            .namespace => |v| v.type,
            .struct_instance => |v| v.type,
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

pub const ErrorInfo = union(enum) {
    unbound: struct { name: []const u8 },
};

pub const TypeCheckErrorDetail = struct { index: usize, expr: *Value, err: TypeCheckError, info: ?ErrorInfo };

pub const TypeCheckReport = struct {
    typed: ArrayList(*TypedValue),
    errors: ArrayList(TypeCheckErrorDetail),
};

// Bidirectional type checker
pub const BidirectionalTypeChecker = struct {
    allocator: std.mem.Allocator,
    env: TypeEnv,
    type_defs: TypeEnv, // Separate map for type definitions (struct/enum types)
    builtins: std.StringHashMap(void), // Set of builtin special forms
    next_var_id: u32,
    errors: ArrayList(TypeCheckErrorDetail),
    index: usize,

    const BindingSnapshot = struct {
        name: []const u8,
        had_previous: bool,
        previous: Type,
        entry_ptr: *Type,
    };

    pub fn init(allocator: std.mem.Allocator) BidirectionalTypeChecker {
        var checker = BidirectionalTypeChecker{
            .allocator = allocator,
            .env = TypeEnv.init(allocator),
            .type_defs = TypeEnv.init(allocator),
            .builtins = std.StringHashMap(void).init(allocator),
            .next_var_id = 0,
            .errors = ArrayList(TypeCheckErrorDetail){},
            .index = 0,
        };
        checker.initBuiltins() catch |err| {
            std.debug.print("FATAL: Failed to initialize builtins: {}\n", .{err});
            @panic("Could not initialize type checker builtins");
        };
        return checker;
    }

    fn initBuiltins(self: *BidirectionalTypeChecker) !void {
        // Special forms that don't need type checking like regular variables
        try self.builtins.put("def", {});
        try self.builtins.put("extern-fn", {});
        try self.builtins.put("extern-type", {});
        try self.builtins.put("extern-union", {});
        try self.builtins.put("extern-struct", {});
        try self.builtins.put("extern-var", {});
        try self.builtins.put("include-header", {});
        try self.builtins.put("link-library", {});
        try self.builtins.put("compiler-flag", {});
        try self.builtins.put("let", {});
        try self.builtins.put("fn", {});
        try self.builtins.put("if", {});
        try self.builtins.put("while", {});
        try self.builtins.put("c-for", {});
        try self.builtins.put("set!", {});
        try self.builtins.put("and", {});
        try self.builtins.put("or", {});
        try self.builtins.put("not", {});
        // Operators
        try self.builtins.put("+", {});
        try self.builtins.put("-", {});
        try self.builtins.put("*", {});
        try self.builtins.put("/", {});
        try self.builtins.put("%", {});
        try self.builtins.put("<", {});
        try self.builtins.put(">", {});
        try self.builtins.put("<=", {});
        try self.builtins.put(">=", {});
        try self.builtins.put("=", {});
        try self.builtins.put("!=", {});
        try self.builtins.put("&", {});
        try self.builtins.put("|", {});
        try self.builtins.put("^", {});
        try self.builtins.put("<<", {});
        try self.builtins.put(">>", {});
        // Array operations
        try self.builtins.put("array", {});
        try self.builtins.put("array-ref", {});
        try self.builtins.put("array-set!", {});
        try self.builtins.put("array-length", {});
        try self.builtins.put("array-ptr", {});
        try self.builtins.put("allocate-array", {});
        try self.builtins.put("deallocate-array", {});
        try self.builtins.put("pointer-index-read", {});
        try self.builtins.put("pointer-index-write!", {});
        // C emission primitives
        try self.builtins.put("c-binary-op", {});
        try self.builtins.put("c-unary-op", {});
        try self.builtins.put("c-fold-binary-op", {});
    }

    pub fn deinit(self: *BidirectionalTypeChecker) void {
        self.env.deinit();
        self.type_defs.deinit();
        self.builtins.deinit();
    }

    // Fresh type variable generation
    fn freshVar(self: *BidirectionalTypeChecker) Type {
        const id = self.next_var_id;
        self.next_var_id += 1;
        return Type{ .type_var = id };
    }

    fn registerEnumVariants(self: *BidirectionalTypeChecker, type_name: []const u8, enum_type: *EnumType) !void {
        enum_type.name = type_name;
        for (enum_type.variants) |*variant| {
            if (variant.qualified_name == null) {
                variant.qualified_name = try std.fmt.allocPrint(self.allocator, "{s}/{s}", .{ type_name, variant.name });
            }
            try self.env.put(variant.qualified_name.?, Type{ .enum_type = enum_type });
        }
    }

    fn restoreBindingSnapshots(self: *BidirectionalTypeChecker, snapshots: []const BindingSnapshot) void {
        var i: usize = snapshots.len;
        while (i > 0) {
            i -= 1;
            const snapshot = snapshots[i];
            if (snapshot.had_previous) {
                snapshot.entry_ptr.* = snapshot.previous;
            } else {
                _ = self.env.remove(snapshot.name);
            }
        }
    }

    // Helper: Get next node value from linked list
    fn getNextValue(list_node: anytype) TypeCheckError!*Value {
        const node = list_node.next orelse return TypeCheckError.InvalidTypeAnnotation;
        return node.value orelse TypeCheckError.InvalidTypeAnnotation;
    }

    // Helper: Advance to next node in linked list
    fn advanceNode(current_ptr: anytype) TypeCheckError!void {
        if (current_ptr.*) |curr_node| {
            if (curr_node.next) |next_node| {
                current_ptr.* = next_node;
            } else {
                return TypeCheckError.InvalidTypeAnnotation;
            }
        } else {
            return TypeCheckError.InvalidTypeAnnotation;
        }
    }

    // Helper: Get operator symbol from list head
    fn getOperator(list: anytype) TypeCheckError![]const u8 {
        const op_node = list.value orelse return TypeCheckError.InvalidTypeAnnotation;
        if (!op_node.isSymbol()) return TypeCheckError.InvalidTypeAnnotation;
        return op_node.symbol;
    }

    // Helper: Extract binary operands (left and right)
    const BinaryOperands = struct {
        left: *Value,
        right: *Value,
    };

    fn getBinaryOperands(list: anytype) TypeCheckError!BinaryOperands {
        const left_node = list.next orelse return TypeCheckError.InvalidTypeAnnotation;
        const left = left_node.value orelse return TypeCheckError.InvalidTypeAnnotation;

        const right_node = left_node.next orelse return TypeCheckError.InvalidTypeAnnotation;
        const right = right_node.value orelse return TypeCheckError.InvalidTypeAnnotation;

        // Ensure no extra arguments
        if (right_node.next) |tail| {
            if (tail.value != null) {
                return TypeCheckError.InvalidTypeAnnotation;
            }
        }

        return BinaryOperands{ .left = left, .right = right };
    }

    // Helper: Extract unary operand
    fn getUnaryOperand(list: anytype) TypeCheckError!*Value {
        const operand_node = list.next orelse return TypeCheckError.InvalidTypeAnnotation;
        const operand = operand_node.value orelse return TypeCheckError.InvalidTypeAnnotation;

        // Ensure no extra arguments
        if (operand_node.next) |tail| {
            if (tail.value != null) {
                return TypeCheckError.InvalidTypeAnnotation;
            }
        }

        return operand;
    }

    // Helper: Check if two types are compatible (either direction)
    fn areTypesCompatible(self: *BidirectionalTypeChecker, type1: Type, type2: Type) TypeCheckError!bool {
        var compatible = try self.isSubtype(type1, type2);
        if (!compatible) {
            compatible = try self.isSubtype(type2, type1);
        }
        return compatible;
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

            .namespace => |ns| {
                _ = ns;
                return try TypedExpression.init(self.allocator, expr, Type.nil);
            },

            .macro_def => |m| {
                // Macros should have been expanded already, this is an error
                _ = m;
                return TypeCheckError.InvalidTypeAnnotation;
            },

            .symbol => |name| {
                // Handle boolean literals
                if (std.mem.eql(u8, name, "true") or std.mem.eql(u8, name, "false")) {
                    return try TypedExpression.init(self.allocator, expr, Type.bool);
                }

                // Check builtins first (special forms and operators)
                if (self.builtins.contains(name)) {
                    // Builtins are handled as special forms in list context
                    // In isolation, they don't have a meaningful type
                    return TypeCheckError.CannotSynthesize;
                }

                if (self.env.get(name)) |var_type| {
                    return try TypedExpression.init(self.allocator, expr, var_type);
                } else {
                    const err = TypeCheckError.UnboundVariable;
                    try self.errors.append(self.allocator, .{
                        .index = self.index,
                        .expr = expr,
                        .err = err,
                        .info = ErrorInfo{ .unbound = .{ .name = name } },
                    });
                    return err;
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
                        } else if (std.mem.eql(u8, first.symbol, "extern-fn")) {
                            return try self.synthesizeExternFn(expr, list);
                        } else if (std.mem.eql(u8, first.symbol, "extern-type")) {
                            return try self.synthesizeExternType(expr, list, false, false);
                        } else if (std.mem.eql(u8, first.symbol, "extern-union")) {
                            return try self.synthesizeExternType(expr, list, true, false);
                        } else if (std.mem.eql(u8, first.symbol, "extern-struct")) {
                            return try self.synthesizeExternType(expr, list, false, true);
                        } else if (std.mem.eql(u8, first.symbol, "extern-var")) {
                            return try self.synthesizeExternVar(expr, list);
                        } else if (std.mem.eql(u8, first.symbol, "include-header") or
                            std.mem.eql(u8, first.symbol, "link-library") or
                            std.mem.eql(u8, first.symbol, "compiler-flag"))
                        {
                            // These are compiler directives, not type-checked values
                            return try TypedExpression.init(self.allocator, expr, Type.nil);
                        } else if (std.mem.eql(u8, first.symbol, "let")) {
                            return try self.synthesizeLet(expr, list);
                        } else if (std.mem.eql(u8, first.symbol, "fn")) {
                            return TypeCheckError.CannotSynthesize; // Functions need type annotations
                        } else if (std.mem.eql(u8, first.symbol, "if")) {
                            return try self.synthesizeIf(expr, list);
                        } else if (std.mem.eql(u8, first.symbol, "while")) {
                            return try self.synthesizeWhile(expr, list);
                        } else if (std.mem.eql(u8, first.symbol, "c-for")) {
                            return try self.synthesizeCFor(expr, list);
                        } else if (std.mem.eql(u8, first.symbol, "set!")) {
                            return try self.synthesizeSet(expr, list);
                        } else if (self.isComparisonOperator(first.symbol)) {
                            return try self.synthesizeComparison(expr, list);
                        } else if (std.mem.eql(u8, first.symbol, "and") or
                            std.mem.eql(u8, first.symbol, "or") or
                            std.mem.eql(u8, first.symbol, "not"))
                        {
                            return try self.synthesizeLogical(expr, list);
                        } else if (self.isBitwiseOperator(first.symbol)) {
                            return try self.synthesizeBitwise(expr, list);
                        } else if (std.mem.eql(u8, first.symbol, "+") or
                            std.mem.eql(u8, first.symbol, "-") or
                            std.mem.eql(u8, first.symbol, "*") or
                            std.mem.eql(u8, first.symbol, "/") or
                            std.mem.eql(u8, first.symbol, "%"))
                        {
                            return try self.synthesizeArithmetic(expr, list);
                        } else if (std.mem.eql(u8, first.symbol, "c-binary-op")) {
                            return try self.synthesizeCBinaryOp(expr, list);
                        } else if (std.mem.eql(u8, first.symbol, "c-unary-op")) {
                            return try self.synthesizeCUnaryOp(expr, list);
                        } else if (std.mem.eql(u8, first.symbol, "c-fold-binary-op")) {
                            return try self.synthesizeCFoldBinaryOp(expr, list);
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
            },
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

    // Type checking for extern-fn: (extern-fn SDL_Init [flags U32] -> I32)
    fn synthesizeExternFn(self: *BidirectionalTypeChecker, expr: *Value, list: anytype) TypeCheckError!*TypedExpression {
        var current: ?*const @TypeOf(list.*) = list.next;

        // Get function name
        const name_node = current orelse return TypeCheckError.InvalidTypeAnnotation;
        if (!name_node.value.?.isSymbol()) return TypeCheckError.InvalidTypeAnnotation;
        const fn_name = name_node.value.?.symbol;
        current = name_node.next;

        // Get parameter types vector
        const params_node = current orelse return TypeCheckError.InvalidTypeAnnotation;
        if (!params_node.value.?.isVector()) return TypeCheckError.InvalidTypeAnnotation;
        const params_vec = params_node.value.?.vector;
        current = params_node.next;

        // Check for arrow ->
        const arrow_node = current orelse return TypeCheckError.InvalidTypeAnnotation;
        if (!arrow_node.value.?.isSymbol() or !std.mem.eql(u8, arrow_node.value.?.symbol, "->"))
            return TypeCheckError.InvalidTypeAnnotation;
        current = arrow_node.next;

        // Get return type
        const return_type_node = current orelse return TypeCheckError.InvalidTypeAnnotation;
        const return_type = try self.parseType(return_type_node.value.?);

        // Parse parameter types - params_vec contains [name1 type1 name2 type2 ...]
        // So we need to parse every other element starting at index 1
        if (params_vec.len() % 2 != 0) return TypeCheckError.InvalidTypeAnnotation;
        const param_count = params_vec.len() / 2;
        var param_types = try self.allocator.alloc(Type, param_count);
        var i: usize = 0;
        while (i < param_count) : (i += 1) {
            // Parse type at index (i * 2) + 1
            param_types[i] = try self.parseType(params_vec.at((i * 2) + 1));
        }

        // Create extern function type
        const extern_fn_type = try self.allocator.create(ExternFunctionType);
        extern_fn_type.* = ExternFunctionType{
            .name = fn_name,
            .param_types = param_types,
            .return_type = return_type,
            .variadic = false,
        };

        const fn_type = Type{ .extern_function = extern_fn_type };

        // Add to environment as a regular function type for compatibility
        const regular_fn_type = try self.allocator.create(FunctionType);
        regular_fn_type.* = FunctionType{
            .param_types = param_types,
            .return_type = return_type,
        };
        try self.env.put(fn_name, Type{ .function = regular_fn_type });

        return try TypedExpression.init(self.allocator, expr, fn_type);
    }

    // Type checking for extern-type: (extern-type SDL_Window) or (extern-union SDL_Event) or (extern-struct SDL_Event [type U32])
    fn synthesizeExternType(self: *BidirectionalTypeChecker, expr: *Value, list: anytype, is_union: bool, has_fields: bool) TypeCheckError!*TypedExpression {
        var current: ?*const @TypeOf(list.*) = list.next;

        // Get type name
        const name_node = current orelse return TypeCheckError.InvalidTypeAnnotation;
        if (!name_node.value.?.isSymbol()) return TypeCheckError.InvalidTypeAnnotation;
        const type_name = name_node.value.?.symbol;

        var fields: ?[]const StructField = null;

        if (has_fields) {
            // Parse fields similar to regular struct parsing
            current = name_node.next;

            // Count fields
            var field_count: usize = 0;
            var count_current = current;
            while (count_current != null) {
                if (count_current.?.value != null) field_count += 1;
                count_current = count_current.?.next;
            }

            if (field_count > 0) {
                const fields_array = try self.allocator.alloc(StructField, field_count);
                var field_idx: usize = 0;

                while (current != null and field_idx < field_count) {
                    const field_node = current.?.value orelse return TypeCheckError.InvalidTypeAnnotation;
                    if (!field_node.isVector() or field_node.vector.len() != 2) {
                        return TypeCheckError.InvalidTypeAnnotation;
                    }

                    const field_name_val = field_node.vector.at(0);
                    const field_type_val = field_node.vector.at(1);

                    if (!field_name_val.isSymbol()) return TypeCheckError.InvalidTypeAnnotation;

                    const field_name = field_name_val.symbol;
                    const field_type = try self.parseType(field_type_val);

                    fields_array[field_idx] = StructField{
                        .name = field_name,
                        .field_type = field_type,
                    };

                    field_idx += 1;
                    current = current.?.next;
                }

                fields = fields_array;
            }
        }

        // Create extern type
        const extern_type = try self.allocator.create(ExternType);
        extern_type.* = ExternType{
            .name = type_name,
            .is_opaque = fields == null, // Only opaque if no fields defined
            .is_union = is_union,
            .fields = fields,
        };

        const type_val = Type{ .extern_type = extern_type };

        // Add to environment - extern types are typically used as pointer types
        try self.env.put(type_name, type_val);

        return try TypedExpression.init(self.allocator, expr, Type.type_type);
    }

    // Type checking for extern-var: (extern-var SDL_INIT_VIDEO U32)
    fn synthesizeExternVar(self: *BidirectionalTypeChecker, expr: *Value, list: anytype) TypeCheckError!*TypedExpression {
        var current: ?*const @TypeOf(list.*) = list.next;

        // Get variable name
        const name_node = current orelse return TypeCheckError.InvalidTypeAnnotation;
        if (!name_node.value.?.isSymbol()) return TypeCheckError.InvalidTypeAnnotation;
        const var_name = name_node.value.?.symbol;
        current = name_node.next;

        // Get type
        const type_node = current orelse return TypeCheckError.InvalidTypeAnnotation;
        const var_type = try self.parseType(type_node.value.?);

        // Add to environment
        try self.env.put(var_name, var_type);

        return try TypedExpression.init(self.allocator, expr, var_type);
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

        // Check parameter count matches
        if (param_vector.len() != expected.function.param_types.len) {
            return TypeCheckError.ArgumentCountMismatch;
        }

        // Create new environment with parameter bindings
        var old_env = self.env;
        self.env = TypeEnv.init(self.allocator);

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

        // Check all body expressions, last one must match return type
        var last_typed: ?*TypedExpression = null;
        while (current != null) {
            const node = current.?;
            if (node.value) |body_expr| {
                last_typed = try self.synthesize(body_expr);
            }
            current = node.next;
        }

        // Check that last expression matches return type
        if (last_typed) |typed| {
            if (!(try self.isSubtype(typed.type, expected.function.return_type))) {
                self.env.deinit();
                self.env = old_env;
                return TypeCheckError.TypeMismatch;
            }
        } else {
            self.env.deinit();
            self.env = old_env;
            return TypeCheckError.InvalidTypeAnnotation;
        }

        // Restore environment
        self.env.deinit();
        self.env = old_env;

        return try TypedExpression.init(self.allocator, expr, expected);
    }

    // Typed function checking for checkTyped method
    fn checkFunctionTyped(self: *BidirectionalTypeChecker, _: *Value, list: anytype, expected: Type) TypeCheckError!*TypedValue {
        if (expected != .function) return TypeCheckError.TypeMismatch;

        var current = list.next; // Skip 'fn'

        // Get parameter list
        const param_list_node = current orelse return TypeCheckError.InvalidTypeAnnotation;
        if (!param_list_node.value.?.isVector()) return TypeCheckError.InvalidTypeAnnotation;
        const param_vector = param_list_node.value.?.vector;
        current = param_list_node.next;

        // Check parameter count matches
        if (param_vector.len() != expected.function.param_types.len) {
            return TypeCheckError.ArgumentCountMismatch;
        }

        // Create new environment with parameter bindings
        var old_env = self.env;
        self.env = TypeEnv.init(self.allocator);

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

        // Check all body expressions and collect them, last one must match return type
        var body_exprs = ArrayList(*TypedValue){};
        defer body_exprs.deinit(self.allocator);

        var last_typed: ?*TypedValue = null;
        var body_count: usize = 0;
        while (current != null) {
            const node = current.?;
            if (node.value) |body_expr| {
                body_count += 1;
                const typed_body = self.synthesizeTyped(body_expr) catch |err| {
                    std.debug.print("ERROR: Body expr #{} failed with error: {}\n", .{ body_count, err });
                    return err;
                };
                try body_exprs.append(self.allocator, typed_body);
                last_typed = typed_body;
            }
            current = node.next;
        }

        // Check that last expression matches return type
        if (last_typed) |typed| {
            if (!(try self.isSubtype(typed.getType(), expected.function.return_type))) {
                std.debug.print("ERROR: Function body TypeMismatch - expected return: {any}, actual: {any}\n", .{ expected.function.return_type, typed.getType() });
                self.env.deinit();
                self.env = old_env;
                return TypeCheckError.TypeMismatch;
            }
        } else {
            self.env.deinit();
            self.env = old_env;
            return TypeCheckError.InvalidTypeAnnotation;
        }

        // Restore environment
        self.env.deinit();
        self.env = old_env;

        // Create the typed function value with proper representation
        // Format: (fn [params...] body...)
        const element_count = 2 + body_exprs.items.len; // 'fn' symbol + params vector + body expressions
        const typed_elements = try self.allocator.alloc(*TypedValue, element_count);

        // Element 0: 'fn' symbol
        const fn_symbol = try self.allocator.create(TypedValue);
        fn_symbol.* = TypedValue{ .symbol = .{ .name = "fn", .type = expected } };
        typed_elements[0] = fn_symbol;

        // Element 1: params vector (typed)
        const params_typed = try self.allocator.create(TypedValue);
        const param_typed_elements = try self.allocator.alloc(*TypedValue, param_vector.len());
        i = 0;
        while (i < param_vector.len()) {
            const param = param_vector.at(i);
            const param_typed_val = try self.allocator.create(TypedValue);
            param_typed_val.* = TypedValue{ .symbol = .{ .name = param.symbol, .type = expected.function.param_types[i] } };
            param_typed_elements[i] = param_typed_val;
            i += 1;
        }
        params_typed.* = TypedValue{ .vector = .{ .elements = param_typed_elements, .type = self.freshVar() } };
        typed_elements[1] = params_typed;

        // Elements 2+: body expressions
        for (body_exprs.items, 0..) |body_typed, idx| {
            typed_elements[2 + idx] = body_typed;
        }

        const result = try self.allocator.create(TypedValue);
        result.* = TypedValue{
            .list = .{
                .elements = typed_elements,
                .type = expected,
            },
        };
        return result;
    }

    // Function application type checking
    fn synthesizeApplication(self: *BidirectionalTypeChecker, expr: *Value, list: anytype) TypeCheckError!*TypedExpression {
        var current: ?*const @TypeOf(list.*) = list;

        // Get function/constructor
        const func_node = (current orelse return TypeCheckError.CannotSynthesize).value orelse return TypeCheckError.CannotSynthesize;
        const func_typed = try self.synthesize(func_node);

        // Check if this is a type name that's actually a struct constructor
        var struct_type_opt: ?Type = null;
        if (func_typed.type == .type_type and func_node.isSymbol()) {
            // Look up the actual type in type_defs (for regular structs)
            if (self.type_defs.get(func_node.symbol)) |actual_type| {
                struct_type_opt = actual_type;
            }
        } else if ((func_typed.type == .extern_type or func_typed.type == .struct_type)) {
            // Extern types and struct types are stored directly in env
            struct_type_opt = func_typed.type;
        }

        // Check if this is a struct constructor (struct_type or extern_type with fields)
        const is_struct_constructor = if (struct_type_opt) |st| switch (st) {
            .struct_type => true,
            .extern_type => |et| et.fields != null,
            else => false,
        } else false;

        if (is_struct_constructor) {
            // Struct construction
            const the_struct_type = struct_type_opt orelse func_typed.type;
            const fields = switch (the_struct_type) {
                .struct_type => |st| st.fields,
                .extern_type => |et| et.fields.?,
                else => unreachable,
            };

            if (current) |curr_node| {
                if (curr_node.next) |next_node| {
                    current = next_node;
                } else {
                    return TypeCheckError.ArgumentCountMismatch;
                }
            } else {
                return TypeCheckError.ArgumentCountMismatch;
            }

            // Check arguments against field types
            var arg_index: usize = 0;
            while (current != null and arg_index < fields.len) {
                const arg = (current orelse break).value orelse break;
                const field_type = fields[arg_index].field_type;
                _ = try self.check(arg, field_type);

                if (current) |curr_node| {
                    current = curr_node.next;
                }
                arg_index += 1;
            }

            if (arg_index != fields.len) {
                return TypeCheckError.ArgumentCountMismatch;
            }

            return try TypedExpression.init(self.allocator, expr, the_struct_type);
        }

        // Regular function application or function pointer call
        // Support both .function and .pointer(.function)
        const fn_type = if (func_typed.type == .function)
            func_typed.type.function
        else if (func_typed.type == .pointer and func_typed.type.pointer.* == .function)
            func_typed.type.pointer.function
        else
            return TypeCheckError.CannotApplyNonFunction;

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
        while (current != null and arg_index < fn_type.param_types.len) {
            const arg = (current orelse break).value orelse break;
            const param_type = fn_type.param_types[arg_index];
            _ = try self.check(arg, param_type);

            if (current) |curr_node| {
                current = curr_node.next;
            }
            arg_index += 1;
        }

        if (arg_index != fn_type.param_types.len) {
            return TypeCheckError.ArgumentCountMismatch;
        }

        return try TypedExpression.init(self.allocator, expr, fn_type.return_type);
    }

    // Arithmetic operations type checking
    fn synthesizeArithmetic(self: *BidirectionalTypeChecker, expr: *Value, list: anytype) TypeCheckError!*TypedExpression {
        var operands: [64]*Value = undefined;
        var operand_count: usize = 0;

        var merged_type_opt: ?Type = null;

        var current: ?*const @TypeOf(list.*) = list.next;
        while (current != null) {
            const node = current.?;
            if (node.value) |operand| {
                if (operand_count >= operands.len) {
                    return TypeCheckError.InvalidTypeAnnotation;
                }
                operands[operand_count] = operand;

                const operand_typed = try self.synthesize(operand);
                if (!isNumericType(operand_typed.type)) {
                    return TypeCheckError.TypeMismatch;
                }

                merged_type_opt = if (merged_type_opt) |prev|
                    try self.mergeNumericTypes(prev, operand_typed.type)
                else
                    operand_typed.type;

                operand_count += 1;
            } else {
                break;
            }
            current = node.next;
        }

        if (operand_count == 0) {
            return TypeCheckError.InvalidTypeAnnotation;
        }

        // Get the operator
        const op_node = list.value orelse return TypeCheckError.InvalidTypeAnnotation;
        if (!op_node.isSymbol()) return TypeCheckError.InvalidTypeAnnotation;
        const op = op_node.symbol;

        var result_type = merged_type_opt.?;

        // For division of integers, result should be float
        if (std.mem.eql(u8, op, "/") and isIntegerType(result_type)) {
            result_type = if (std.meta.activeTag(result_type) == .int) Type.float else Type.f64;
        }

        // For modulo, ensure integer type
        if (std.mem.eql(u8, op, "%") and !isIntegerType(result_type)) {
            return TypeCheckError.TypeMismatch;
        }

        if (operand_count == 1 and !std.mem.eql(u8, op, "-")) {
            return TypeCheckError.InvalidTypeAnnotation;
        }

        var idx: usize = 0;
        while (idx < operand_count) : (idx += 1) {
            _ = try self.check(operands[idx], result_type);
        }

        return try TypedExpression.init(self.allocator, expr, result_type);
    }

    fn synthesizeComparison(self: *BidirectionalTypeChecker, expr: *Value, list: anytype) TypeCheckError!*TypedExpression {
        const op = try getOperator(list);
        const operands = try getBinaryOperands(list);

        const left_typed = try self.synthesize(operands.left);
        const right_typed = try self.synthesize(operands.right);

        if (isRelationalOperator(op)) {
            if (!isNumericType(left_typed.type) or !isNumericType(right_typed.type)) {
                return TypeCheckError.TypeMismatch;
            }
        }

        if (!try self.areTypesCompatible(left_typed.type, right_typed.type)) {
            return TypeCheckError.TypeMismatch;
        }

        return try TypedExpression.init(self.allocator, expr, Type.bool);
    }

    fn synthesizeLogical(self: *BidirectionalTypeChecker, expr: *Value, list: anytype) TypeCheckError!*TypedExpression {
        const op = try getOperator(list);

        if (std.mem.eql(u8, op, "not")) {
            // Unary operator
            const operand = try getUnaryOperand(list);
            const operand_typed = try self.synthesize(operand);

            if (!isTruthyType(operand_typed.type)) {
                return TypeCheckError.TypeMismatch;
            }

            return try TypedExpression.init(self.allocator, expr, Type.bool);
        }

        // Binary operators: and, or
        const operands = try getBinaryOperands(list);
        const left_typed = try self.synthesize(operands.left);
        const right_typed = try self.synthesize(operands.right);

        if (!isTruthyType(left_typed.type) or !isTruthyType(right_typed.type)) {
            return TypeCheckError.TypeMismatch;
        }

        return try TypedExpression.init(self.allocator, expr, Type.bool);
    }

    fn synthesizeBitwise(self: *BidirectionalTypeChecker, expr: *Value, list: anytype) TypeCheckError!*TypedExpression {
        const op = try getOperator(list);

        if (std.mem.eql(u8, op, "bit-not")) {
            // Unary operator
            const operand = try getUnaryOperand(list);
            const operand_typed = try self.synthesize(operand);

            if (!isIntegerType(operand_typed.type)) {
                return TypeCheckError.TypeMismatch;
            }

            return try TypedExpression.init(self.allocator, expr, operand_typed.type);
        }

        // Binary operators: bit-and, bit-or, bit-xor, bit-shl, bit-shr
        const operands = try getBinaryOperands(list);
        const left_typed = try self.synthesize(operands.left);
        const right_typed = try self.synthesize(operands.right);

        if (!isIntegerType(left_typed.type) or !isIntegerType(right_typed.type)) {
            return TypeCheckError.TypeMismatch;
        }

        // For shifts, result type is left operand type
        if (std.mem.eql(u8, op, "bit-shl") or std.mem.eql(u8, op, "bit-shr")) {
            return try TypedExpression.init(self.allocator, expr, left_typed.type);
        }

        // For bit-and, bit-or, bit-xor, merge types like arithmetic
        const merged_type = try self.mergeNumericTypes(left_typed.type, right_typed.type);
        return try TypedExpression.init(self.allocator, expr, merged_type);
    }

    // C emission primitives - minimal type checking, trust macro expansion
    fn synthesizeCBinaryOp(self: *BidirectionalTypeChecker, expr: *Value, list: anytype) TypeCheckError!*TypedExpression {
        // (c-binary-op "op-string" left right)
        var current = list.next orelse return TypeCheckError.InvalidTypeAnnotation;

        // Get operator string (must be a string literal)
        const op_node = current.value orelse return TypeCheckError.InvalidTypeAnnotation;
        if (!op_node.isString()) return TypeCheckError.InvalidTypeAnnotation;
        const op = op_node.string;

        current = current.next orelse return TypeCheckError.InvalidTypeAnnotation;

        // Get left operand
        const left = current.value orelse return TypeCheckError.InvalidTypeAnnotation;
        const left_typed = try self.synthesize(left);

        current = current.next orelse return TypeCheckError.InvalidTypeAnnotation;

        // Get right operand
        const right = current.value orelse return TypeCheckError.InvalidTypeAnnotation;
        const right_typed = try self.synthesize(right);

        // Ensure no extra arguments
        if (current.next) |tail| {
            if (tail.value != null) {
                return TypeCheckError.InvalidTypeAnnotation;
            }
        }

        // Determine result type based on operator
        const result_type: Type = if (isComparisonOp(op) or isLogicalOp(op))
            Type.bool
        else if (isArithmeticOp(op)) blk: {
            // For arithmetic, merge types
            const merged = try self.mergeNumericTypes(left_typed.type, right_typed.type);
            // Division of integers produces float
            if (std.mem.eql(u8, op, "/") and isIntegerType(merged)) {
                break :blk if (std.meta.activeTag(merged) == .int) Type.float else Type.f64;
            }
            break :blk merged;
        } else if (isBitwiseOp(op))
            try self.mergeNumericTypes(left_typed.type, right_typed.type)
        else
            // Unknown operator, use left type
            left_typed.type;

        return try TypedExpression.init(self.allocator, expr, result_type);
    }

    fn synthesizeCUnaryOp(self: *BidirectionalTypeChecker, expr: *Value, list: anytype) TypeCheckError!*TypedExpression {
        // (c-unary-op "op-string" operand)
        var current = list.next orelse return TypeCheckError.InvalidTypeAnnotation;

        // Get operator string
        const op_node = current.value orelse return TypeCheckError.InvalidTypeAnnotation;
        if (!op_node.isString()) return TypeCheckError.InvalidTypeAnnotation;
        const op = op_node.string;

        current = current.next orelse return TypeCheckError.InvalidTypeAnnotation;

        // Get operand
        const operand = current.value orelse return TypeCheckError.InvalidTypeAnnotation;
        const operand_typed = try self.synthesize(operand);

        // Ensure no extra arguments
        if (current.next) |tail| {
            if (tail.value != null) {
                return TypeCheckError.InvalidTypeAnnotation;
            }
        }

        // Result type is same as operand for unary ops (-, ~)
        // For !, result is bool
        const result_type: Type = if (std.mem.eql(u8, op, "!"))
            Type.bool
        else
            operand_typed.type;

        return try TypedExpression.init(self.allocator, expr, result_type);
    }

    fn synthesizeCFoldBinaryOp(self: *BidirectionalTypeChecker, expr: *Value, list: anytype) TypeCheckError!*TypedExpression {
        // (c-fold-binary-op "op-string" arg1 arg2 arg3 ...)
        var current_node = list.next orelse return TypeCheckError.InvalidTypeAnnotation;

        // Get operator string
        const op_node = current_node.value orelse return TypeCheckError.InvalidTypeAnnotation;
        if (!op_node.isString()) return TypeCheckError.InvalidTypeAnnotation;
        const op = op_node.string;

        current_node = current_node.next orelse return TypeCheckError.InvalidTypeAnnotation;

        // Type check all operands and merge types
        var merged_type_opt: ?Type = null;
        var operand_count: usize = 0;

        var current_opt: ?*const @TypeOf(current_node.*) = current_node;
        while (current_opt) |node| {
            if (node.value) |operand| {
                const operand_typed = try self.synthesize(operand);

                merged_type_opt = if (merged_type_opt) |prev|
                    try self.mergeNumericTypes(prev, operand_typed.type)
                else
                    operand_typed.type;

                operand_count += 1;
            } else {
                break;
            }
            current_opt = node.next;
        }

        if (operand_count == 0) {
            return TypeCheckError.InvalidTypeAnnotation;
        }

        var result_type = merged_type_opt.?;

        // For division of integers, result should be float
        if (std.mem.eql(u8, op, "/") and isIntegerType(result_type)) {
            result_type = if (std.meta.activeTag(result_type) == .int) Type.float else Type.f64;
        }

        return try TypedExpression.init(self.allocator, expr, result_type);
    }

    fn synthesizeIf(self: *BidirectionalTypeChecker, expr: *Value, list: anytype) TypeCheckError!*TypedExpression {
        const cond_node = list.next orelse return TypeCheckError.InvalidTypeAnnotation;
        const cond_expr = cond_node.value orelse return TypeCheckError.InvalidTypeAnnotation;
        const cond_typed = try self.synthesize(cond_expr);

        if (!isTruthyType(cond_typed.type)) {
            return TypeCheckError.TypeMismatch;
        }

        const then_node = cond_node.next orelse return TypeCheckError.InvalidTypeAnnotation;
        const then_expr = then_node.value orelse return TypeCheckError.InvalidTypeAnnotation;
        const then_typed = try self.synthesize(then_expr);

        const else_node = then_node.next orelse return TypeCheckError.InvalidTypeAnnotation;
        const else_expr = else_node.value orelse return TypeCheckError.InvalidTypeAnnotation;
        const else_typed = try self.synthesize(else_expr);

        if (else_node.next) |tail| {
            if (tail.value != null) {
                return TypeCheckError.InvalidTypeAnnotation;
            }
        }

        const result_type = try self.mergeBranchTypes(then_typed.type, else_typed.type);

        return try TypedExpression.init(self.allocator, expr, result_type);
    }

    fn synthesizeWhile(self: *BidirectionalTypeChecker, expr: *Value, list: anytype) TypeCheckError!*TypedExpression {
        // (while cond body*)
        const cond_node = list.next orelse return TypeCheckError.InvalidTypeAnnotation;
        const cond_expr = cond_node.value orelse return TypeCheckError.InvalidTypeAnnotation;
        const cond_typed = try self.synthesize(cond_expr);

        if (!isTruthyType(cond_typed.type)) {
            return TypeCheckError.TypeMismatch;
        }

        // Type check all body expressions
        var current = cond_node.next;
        while (current) |node| {
            if (node.value) |body_expr| {
                _ = try self.synthesize(body_expr);
            }
            current = node.next;
        }

        // While loops always return nil
        return try TypedExpression.init(self.allocator, expr, Type.void);
    }

    fn synthesizeCFor(self: *BidirectionalTypeChecker, expr: *Value, list: anytype) TypeCheckError!*TypedExpression {
        // (c-for [var (: Type) init] condition step body*)
        const init_binding_node = list.next orelse return TypeCheckError.InvalidTypeAnnotation;
        const init_binding_value = init_binding_node.value orelse return TypeCheckError.InvalidTypeAnnotation;

        if (!init_binding_value.isVector() or init_binding_value.vector.len() != 3) {
            return TypeCheckError.InvalidTypeAnnotation;
        }

        const var_name_val = init_binding_value.vector.at(0);
        const annotation_val = init_binding_value.vector.at(1);
        const init_val = init_binding_value.vector.at(2);

        if (!var_name_val.isSymbol()) return TypeCheckError.InvalidTypeAnnotation;
        const var_name = var_name_val.symbol;

        // Parse type annotation
        const var_type = try self.parseTypeAnnotation(annotation_val);

        // Check init value against the declared type
        _ = try self.check(init_val, var_type);

        // Save current binding (if any) to restore later
        const entry = try self.env.getOrPut(var_name);
        const had_previous = entry.found_existing;
        const previous = if (had_previous) entry.value_ptr.* else Type.nil;
        entry.value_ptr.* = var_type;

        const snapshot = BindingSnapshot{
            .name = var_name,
            .had_previous = had_previous,
            .previous = previous,
            .entry_ptr = entry.value_ptr,
        };

        errdefer self.restoreBindingSnapshots(&[_]BindingSnapshot{snapshot});

        // Get condition
        const cond_node = init_binding_node.next orelse return TypeCheckError.InvalidTypeAnnotation;
        const cond_expr = cond_node.value orelse return TypeCheckError.InvalidTypeAnnotation;
        const cond_typed = try self.synthesize(cond_expr);

        if (!isTruthyType(cond_typed.type)) {
            self.restoreBindingSnapshots(&[_]BindingSnapshot{snapshot});
            return TypeCheckError.TypeMismatch;
        }

        // Get step expression
        const step_node = cond_node.next orelse return TypeCheckError.InvalidTypeAnnotation;
        const step_expr = step_node.value orelse return TypeCheckError.InvalidTypeAnnotation;

        // Type check step expression (can be nil for empty step)
        if (!step_expr.isNil()) {
            _ = try self.synthesize(step_expr);
        }

        // Type check all body expressions
        var current = step_node.next;
        while (current) |node| {
            if (node.value) |body_expr| {
                _ = try self.synthesize(body_expr);
            }
            current = node.next;
        }

        // Restore binding
        self.restoreBindingSnapshots(&[_]BindingSnapshot{snapshot});

        // c-for loops always return void
        return try TypedExpression.init(self.allocator, expr, Type.void);
    }

    fn synthesizeSet(self: *BidirectionalTypeChecker, expr: *Value, list: anytype) TypeCheckError!*TypedExpression {
        // (set! var-name value)
        const operands = try getBinaryOperands(list);

        if (!operands.left.isSymbol()) return TypeCheckError.InvalidTypeAnnotation;
        const var_name = operands.left.symbol;

        // Check that the variable exists in the environment
        const var_type = self.env.get(var_name) orelse return TypeCheckError.UnboundVariable;

        // Type check the value and ensure it matches the variable's type
        const value_typed = try self.check(operands.right, var_type);
        _ = value_typed;

        // set! returns void/nil
        return try TypedExpression.init(self.allocator, expr, Type.void);
    }

    fn synthesizeLet(self: *BidirectionalTypeChecker, expr: *Value, list: anytype) TypeCheckError!*TypedExpression {
        var current: ?*const @TypeOf(list.*) = list.next;

        const bindings_node = current orelse return TypeCheckError.InvalidTypeAnnotation;
        const bindings_value = bindings_node.value orelse return TypeCheckError.InvalidTypeAnnotation;
        if (!bindings_value.isVector()) return TypeCheckError.InvalidTypeAnnotation;
        const bindings_vec = bindings_value.vector;

        if (bindings_vec.len() % 3 != 0) return TypeCheckError.InvalidTypeAnnotation;
        const binding_count = bindings_vec.len() / 3;

        const snapshots_storage = try self.allocator.alloc(BindingSnapshot, binding_count);
        var inserted: usize = 0;
        const snapshot_slice = snapshots_storage[0..binding_count];
        errdefer self.restoreBindingSnapshots(snapshot_slice[0..inserted]);

        var idx: usize = 0;
        while (idx < binding_count) : (idx += 1) {
            const name_val = bindings_vec.at(idx * 3);
            const annotation_val = bindings_vec.at(idx * 3 + 1);
            const value_val = bindings_vec.at(idx * 3 + 2);

            if (!name_val.isSymbol()) return TypeCheckError.InvalidTypeAnnotation;

            const annotated_type = try self.parseTypeAnnotation(annotation_val);

            _ = try self.check(value_val, annotated_type);

            const entry = try self.env.getOrPut(name_val.symbol);
            const had_previous = entry.found_existing;
            const previous = if (had_previous) entry.value_ptr.* else Type.nil;
            entry.value_ptr.* = annotated_type;

            snapshot_slice[inserted] = BindingSnapshot{
                .name = name_val.symbol,
                .had_previous = had_previous,
                .previous = previous,
                .entry_ptr = entry.value_ptr,
            };
            inserted += 1;
        }

        current = bindings_node.next;
        if (current == null) return TypeCheckError.InvalidTypeAnnotation;

        var last_typed: ?*TypedExpression = null;
        while (current != null) {
            const node = current.?;
            if (node.value) |body_expr| {
                last_typed = try self.synthesize(body_expr);
            }
            current = node.next;
        }

        const body_typed = last_typed orelse return TypeCheckError.InvalidTypeAnnotation;
        const result = try TypedExpression.init(self.allocator, expr, body_typed.type);

        self.restoreBindingSnapshots(snapshot_slice[0..inserted]);

        return result;
    }

    fn synthesizeTypedLet(self: *BidirectionalTypeChecker, expr: *Value, list: anytype) TypeCheckError!*TypedValue {
        var current: ?*const @TypeOf(list.*) = list.next;

        const bindings_node = current orelse return TypeCheckError.InvalidTypeAnnotation;
        const bindings_value = bindings_node.value orelse return TypeCheckError.InvalidTypeAnnotation;
        if (!bindings_value.isVector()) return TypeCheckError.InvalidTypeAnnotation;
        const bindings_vec = bindings_value.vector;

        if (bindings_vec.len() % 3 != 0) return TypeCheckError.InvalidTypeAnnotation;
        const binding_count = bindings_vec.len() / 3;

        const snapshots_storage = try self.allocator.alloc(BindingSnapshot, binding_count);
        var inserted: usize = 0;
        const snapshot_slice = snapshots_storage[0..binding_count];
        errdefer self.restoreBindingSnapshots(snapshot_slice[0..inserted]);

        var idx: usize = 0;
        while (idx < binding_count) : (idx += 1) {
            const name_val = bindings_vec.at(idx * 3);
            const annotation_val = bindings_vec.at(idx * 3 + 1);
            const value_val = bindings_vec.at(idx * 3 + 2);

            if (!name_val.isSymbol()) return TypeCheckError.InvalidTypeAnnotation;

            const annotated_type = try self.parseTypeAnnotation(annotation_val);

            _ = try self.checkTyped(value_val, annotated_type);

            const entry = try self.env.getOrPut(name_val.symbol);
            const had_previous = entry.found_existing;
            const previous = if (had_previous) entry.value_ptr.* else Type.nil;
            entry.value_ptr.* = annotated_type;

            snapshot_slice[inserted] = BindingSnapshot{
                .name = name_val.symbol,
                .had_previous = had_previous,
                .previous = previous,
                .entry_ptr = entry.value_ptr,
            };
            inserted += 1;
        }

        current = bindings_node.next;
        if (current == null) return TypeCheckError.InvalidTypeAnnotation;

        var last_typed: ?*TypedValue = null;
        var let_body_count: usize = 0;
        while (current != null) {
            const node = current.?;
            if (node.value) |body_expr| {
                let_body_count += 1;
                last_typed = self.synthesizeTyped(body_expr) catch |err| {
                    std.debug.print("ERROR: Let body expr #{} failed: {}\n", .{ let_body_count, err });
                    return err;
                };
            }
            current = node.next;
        }

        const body_typed = last_typed orelse return TypeCheckError.InvalidTypeAnnotation;

        self.restoreBindingSnapshots(snapshot_slice[0..inserted]);

        _ = expr;
        return body_typed;
    }

    fn synthesizeTypedComparison(self: *BidirectionalTypeChecker, expr: *Value, list: anytype, storage: []*TypedValue) TypeCheckError!*TypedValue {
        if (storage.len < 3) return TypeCheckError.InvalidTypeAnnotation;
        _ = expr;

        const op = try getOperator(list);
        const operands = try getBinaryOperands(list);

        const left_typed = try self.synthesizeTyped(operands.left);
        const left_type = left_typed.getType();

        var right_typed: *TypedValue = undefined;

        if (isRelationalOperator(op)) {
            if (!isNumericType(left_type)) {
                return TypeCheckError.TypeMismatch;
            }
            right_typed = try self.checkTyped(operands.right, left_type);
        } else {
            right_typed = try self.synthesizeTyped(operands.right);
            if (!try self.areTypesCompatible(left_type, right_typed.getType())) {
                return TypeCheckError.TypeMismatch;
            }
        }

        // Include operator symbol as first element
        const op_typed = try self.allocator.create(TypedValue);
        op_typed.* = TypedValue{ .symbol = .{ .name = op, .type = Type.bool } };
        storage[0] = op_typed;
        storage[1] = left_typed;
        storage[2] = right_typed;

        const result = try self.allocator.create(TypedValue);
        result.* = TypedValue{ .list = .{
            .elements = storage[0..3],
            .type = Type.bool,
        } };
        return result;
    }

    fn synthesizeTypedLogical(self: *BidirectionalTypeChecker, expr: *Value, list: anytype, storage: []*TypedValue) TypeCheckError!*TypedValue {
        _ = expr;
        const op = try getOperator(list);

        if (std.mem.eql(u8, op, "not")) {
            // Unary operator
            if (storage.len < 2) return TypeCheckError.InvalidTypeAnnotation;

            const operand = try getUnaryOperand(list);
            const operand_typed = try self.synthesizeTyped(operand);

            if (!isTruthyType(operand_typed.getType())) {
                return TypeCheckError.TypeMismatch;
            }

            // Include operator symbol as first element
            const op_typed = try self.allocator.create(TypedValue);
            op_typed.* = TypedValue{ .symbol = .{ .name = op, .type = Type.bool } };
            storage[0] = op_typed;
            storage[1] = operand_typed;

            const result = try self.allocator.create(TypedValue);
            result.* = TypedValue{ .list = .{
                .elements = storage[0..2],
                .type = Type.bool,
            } };
            return result;
        }

        // Binary operators: and, or
        if (storage.len < 3) return TypeCheckError.InvalidTypeAnnotation;

        const operands = try getBinaryOperands(list);
        const left_typed = try self.synthesizeTyped(operands.left);
        const right_typed = try self.synthesizeTyped(operands.right);

        if (!isTruthyType(left_typed.getType()) or !isTruthyType(right_typed.getType())) {
            return TypeCheckError.TypeMismatch;
        }

        // Include operator symbol as first element
        const op_typed = try self.allocator.create(TypedValue);
        op_typed.* = TypedValue{ .symbol = .{ .name = op, .type = Type.bool } };
        storage[0] = op_typed;
        storage[1] = left_typed;
        storage[2] = right_typed;

        const result = try self.allocator.create(TypedValue);
        result.* = TypedValue{ .list = .{
            .elements = storage[0..3],
            .type = Type.bool,
        } };
        return result;
    }

    fn synthesizeTypedCStr(self: *BidirectionalTypeChecker, expr: *Value, list: anytype, storage: []*TypedValue) TypeCheckError!*TypedValue {
        _ = expr;

        // c-str takes one argument: a string literal
        const arg_node = list.next orelse return TypeCheckError.InvalidTypeAnnotation;
        const arg_expr = arg_node.value orelse return TypeCheckError.InvalidTypeAnnotation;

        // Type check the argument - should be a string
        const arg_typed = try self.synthesizeTyped(arg_expr);
        if (arg_typed.getType() != .string) {
            return TypeCheckError.TypeMismatch;
        }

        storage[0] = arg_typed;

        // Return type is CString
        const result = try self.allocator.create(TypedValue);
        result.* = TypedValue{ .list = .{
            .elements = storage[0..1],
            .type = Type.c_string,
        } };
        return result;
    }

    fn synthesizeTypedBitwise(self: *BidirectionalTypeChecker, expr: *Value, list: anytype, storage: []*TypedValue) TypeCheckError!*TypedValue {
        _ = expr;
        const op = try getOperator(list);

        if (std.mem.eql(u8, op, "bit-not")) {
            // Unary operator
            if (storage.len < 1) return TypeCheckError.InvalidTypeAnnotation;

            const operand = try getUnaryOperand(list);
            const operand_typed = try self.synthesizeTyped(operand);
            const operand_type = operand_typed.getType();

            if (!isIntegerType(operand_type)) {
                return TypeCheckError.TypeMismatch;
            }

            storage[0] = operand_typed;

            const result = try self.allocator.create(TypedValue);
            result.* = TypedValue{ .list = .{
                .elements = storage[0..1],
                .type = operand_type,
            } };
            return result;
        }

        // Binary operators
        if (storage.len < 2) return TypeCheckError.InvalidTypeAnnotation;

        const operands = try getBinaryOperands(list);
        const left_typed = try self.synthesizeTyped(operands.left);
        const left_type = left_typed.getType();
        const right_typed = try self.synthesizeTyped(operands.right);
        const right_type = right_typed.getType();

        if (!isIntegerType(left_type) or !isIntegerType(right_type)) {
            return TypeCheckError.TypeMismatch;
        }

        storage[0] = left_typed;
        storage[1] = right_typed;

        // For shifts, result type is left operand type
        const result_type = if (std.mem.eql(u8, op, "bit-shl") or std.mem.eql(u8, op, "bit-shr"))
            left_type
        else
            try self.mergeNumericTypes(left_type, right_type);

        const result = try self.allocator.create(TypedValue);
        result.* = TypedValue{ .list = .{
            .elements = storage[0..2],
            .type = result_type,
        } };
        return result;
    }

    fn synthesizeTypedIf(self: *BidirectionalTypeChecker, expr: *Value, list: anytype, storage: []*TypedValue) TypeCheckError!*TypedValue {
        if (storage.len < 3) return TypeCheckError.InvalidTypeAnnotation;

        _ = expr;
        const cond_node = list.next orelse return TypeCheckError.InvalidTypeAnnotation;
        const cond_expr = cond_node.value orelse return TypeCheckError.InvalidTypeAnnotation;
        const cond_typed = try self.synthesizeTyped(cond_expr);
        if (!isTruthyType(cond_typed.getType())) {
            return TypeCheckError.TypeMismatch;
        }

        const then_node = cond_node.next orelse return TypeCheckError.InvalidTypeAnnotation;
        const then_expr = then_node.value orelse return TypeCheckError.InvalidTypeAnnotation;
        const then_typed = try self.synthesizeTyped(then_expr);

        const else_node = then_node.next orelse return TypeCheckError.InvalidTypeAnnotation;
        const else_expr = else_node.value orelse return TypeCheckError.InvalidTypeAnnotation;
        const else_typed = try self.synthesizeTyped(else_expr);

        if (else_node.next) |tail| {
            if (tail.value != null) {
                return TypeCheckError.InvalidTypeAnnotation;
            }
        }

        const result_type = self.mergeBranchTypes(then_typed.getType(), else_typed.getType()) catch |err| {
            std.debug.print("ERROR: mergeBranchTypes failed: {}\n", .{err});
            return err;
        };

        storage[0] = cond_typed;
        storage[1] = then_typed;
        storage[2] = else_typed;

        const result = try self.allocator.create(TypedValue);
        result.* = TypedValue{ .list = .{
            .elements = storage[0..3],
            .type = result_type,
        } };
        return result;
    }

    fn synthesizeTypedWhile(self: *BidirectionalTypeChecker, expr: *Value, list: anytype, storage: []*TypedValue) TypeCheckError!*TypedValue {
        _ = expr;

        // (while cond body*)
        const cond_node = list.next orelse return TypeCheckError.InvalidTypeAnnotation;
        const cond_expr = cond_node.value orelse return TypeCheckError.InvalidTypeAnnotation;
        const cond_typed = try self.synthesizeTyped(cond_expr);

        if (!isTruthyType(cond_typed.getType())) {
            return TypeCheckError.TypeMismatch;
        }

        // Type check all body expressions
        var current = cond_node.next;
        var elem_count: usize = 1; // Start at 1 for condition
        while (current) |node| {
            if (node.value) |body_expr| {
                if (elem_count >= storage.len) return TypeCheckError.InvalidTypeAnnotation;
                storage[elem_count] = try self.synthesizeTyped(body_expr);
                elem_count += 1;
            }
            current = node.next;
        }

        storage[0] = cond_typed;

        const result = try self.allocator.create(TypedValue);
        result.* = TypedValue{ .list = .{
            .elements = storage[0..elem_count],
            .type = Type.void,
        } };
        return result;
    }

    fn synthesizeTypedCFor(self: *BidirectionalTypeChecker, expr: *Value, list: anytype, storage: []*TypedValue) TypeCheckError!*TypedValue {
        _ = expr;

        // (c-for [var (: Type) init] condition step body*)
        const init_binding_node = list.next orelse return TypeCheckError.InvalidTypeAnnotation;
        const init_binding_value = init_binding_node.value orelse return TypeCheckError.InvalidTypeAnnotation;

        if (!init_binding_value.isVector() or init_binding_value.vector.len() != 3) {
            return TypeCheckError.InvalidTypeAnnotation;
        }

        const var_name_val = init_binding_value.vector.at(0);
        const annotation_val = init_binding_value.vector.at(1);
        const init_val = init_binding_value.vector.at(2);

        if (!var_name_val.isSymbol()) return TypeCheckError.InvalidTypeAnnotation;
        const var_name = var_name_val.symbol;

        // Parse type annotation
        const var_type = try self.parseTypeAnnotation(annotation_val);

        // Check init value against the declared type
        const init_typed = try self.checkTyped(init_val, var_type);

        // Save current binding (if any) to restore later
        const entry = try self.env.getOrPut(var_name);
        const had_previous = entry.found_existing;
        const previous = if (had_previous) entry.value_ptr.* else Type.nil;
        entry.value_ptr.* = var_type;

        const snapshot = BindingSnapshot{
            .name = var_name,
            .had_previous = had_previous,
            .previous = previous,
            .entry_ptr = entry.value_ptr,
        };

        errdefer self.restoreBindingSnapshots(&[_]BindingSnapshot{snapshot});

        // Get condition
        const cond_node = init_binding_node.next orelse return TypeCheckError.InvalidTypeAnnotation;
        const cond_expr = cond_node.value orelse return TypeCheckError.InvalidTypeAnnotation;
        const cond_typed = try self.synthesizeTyped(cond_expr);

        if (!isTruthyType(cond_typed.getType())) {
            self.restoreBindingSnapshots(&[_]BindingSnapshot{snapshot});
            return TypeCheckError.TypeMismatch;
        }

        // Get step expression
        const step_node = cond_node.next orelse return TypeCheckError.InvalidTypeAnnotation;
        const step_expr = step_node.value orelse return TypeCheckError.InvalidTypeAnnotation;

        // Type check step expression (can be nil for empty step)
        const step_typed = if (!step_expr.isNil())
            try self.synthesizeTyped(step_expr)
        else
            blk: {
                const nil_typed = try self.allocator.create(TypedValue);
                nil_typed.* = TypedValue{ .nil = .{ .type = Type.nil } };
                break :blk nil_typed;
            };

        // Type check all body expressions
        var current = step_node.next;
        var elem_count: usize = 3; // Start at 3 for init, cond, step
        while (current) |node| {
            if (node.value) |body_expr| {
                if (elem_count >= storage.len) return TypeCheckError.InvalidTypeAnnotation;
                storage[elem_count] = try self.synthesizeTyped(body_expr);
                elem_count += 1;
            }
            current = node.next;
        }

        // Restore binding
        self.restoreBindingSnapshots(&[_]BindingSnapshot{snapshot});

        storage[0] = init_typed;
        storage[1] = cond_typed;
        storage[2] = step_typed;

        const result = try self.allocator.create(TypedValue);
        result.* = TypedValue{ .list = .{
            .elements = storage[0..elem_count],
            .type = Type.void,
        } };
        return result;
    }

    fn synthesizeTypedSet(self: *BidirectionalTypeChecker, expr: *Value, list: anytype, storage: []*TypedValue) TypeCheckError!*TypedValue {
        _ = expr;

        // (set! var-name value)
        const operands = try getBinaryOperands(list);

        if (!operands.left.isSymbol()) return TypeCheckError.InvalidTypeAnnotation;
        const var_name = operands.left.symbol;

        // Check that the variable exists in the environment
        const var_type = self.env.get(var_name) orelse {
            std.debug.print("ERROR: set! - Unbound variable: '{s}'\n", .{var_name});
            return TypeCheckError.UnboundVariable;
        };

        // Type check the value and ensure it matches the variable's type
        const value_typed = try self.checkTyped(operands.right, var_type);
        storage[0] = value_typed;

        const result = try self.allocator.create(TypedValue);
        result.* = TypedValue{ .list = .{
            .elements = storage[0..1],
            .type = Type.void,
        } };
        return result;
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

        try advanceNode(&current);
        const type_expr = (current orelse return TypeCheckError.InvalidTypeAnnotation).value orelse return TypeCheckError.InvalidTypeAnnotation;

        return try self.parseType(type_expr);
    }

    // Helper: Parse primitive type name
    fn parsePrimitiveType(type_name: []const u8) ?Type {
        // Use a static map for cleaner code
        const TypeMapping = struct { name: []const u8, type_value: Type };
        const mappings = [_]TypeMapping{
            // Legacy types
            .{ .name = "Int", .type_value = Type.int },
            .{ .name = "Float", .type_value = Type.float },
            .{ .name = "String", .type_value = Type.string },
            .{ .name = "Bool", .type_value = Type.bool },
            .{ .name = "Nil", .type_value = Type.nil },
            // Unsigned integers
            .{ .name = "U8", .type_value = Type.u8 },
            .{ .name = "U16", .type_value = Type.u16 },
            .{ .name = "U32", .type_value = Type.u32 },
            .{ .name = "U64", .type_value = Type.u64 },
            .{ .name = "Usize", .type_value = Type.usize },
            // Signed integers
            .{ .name = "I8", .type_value = Type.i8 },
            .{ .name = "I16", .type_value = Type.i16 },
            .{ .name = "I32", .type_value = Type.i32 },
            .{ .name = "I64", .type_value = Type.i64 },
            .{ .name = "Isize", .type_value = Type.isize },
            // Floats
            .{ .name = "F32", .type_value = Type.f32 },
            .{ .name = "F64", .type_value = Type.f64 },
            // FFI types
            .{ .name = "CString", .type_value = Type.c_string },
            .{ .name = "Void", .type_value = Type.void },
            // Meta-type
            .{ .name = "Type", .type_value = Type.type_type },
        };

        for (mappings) |mapping| {
            if (std.mem.eql(u8, type_name, mapping.name)) {
                return mapping.type_value;
            }
        }
        return null;
    }

    // Parse type expressions
    fn parseType(self: *BidirectionalTypeChecker, type_expr: *Value) TypeCheckError!Type {
        if (type_expr.isSymbol()) {
            const type_name = type_expr.symbol;

            // Try primitive types first
            if (parsePrimitiveType(type_name)) |primitive_type| {
                return primitive_type;
            }

            // Check if it's a user-defined type in type_defs first
            if (self.type_defs.get(type_name)) |type_def| {
                return type_def;
            }

            // Fall back to env for backward compatibility
            if (self.env.get(type_name)) |env_type| {
                return env_type;
            }

            return TypeCheckError.InvalidTypeAnnotation;
        }

        if (type_expr.isList()) {
            var current: ?*const @TypeOf(type_expr.list.*) = type_expr.list;
            const head = (current orelse return TypeCheckError.InvalidTypeAnnotation).value orelse return TypeCheckError.InvalidTypeAnnotation;

            if (head.isSymbol()) {
                if (std.mem.eql(u8, head.symbol, "->")) {
                    // Function type: (-> [param_types] return_type)
                    try advanceNode(&current);
                    const param_list = (current orelse return TypeCheckError.InvalidTypeAnnotation).value orelse return TypeCheckError.InvalidTypeAnnotation;

                    if (!param_list.isVector()) return TypeCheckError.InvalidTypeAnnotation;

                    try advanceNode(&current);
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

        if (type_expr.isList()) {
            var current: ?*const @TypeOf(type_expr.list.*) = type_expr.list;

            // Check if it's a Struct type: (Struct [field1 Type1] [field2 Type2] ...)
            if (current) |node| {
                if (node.value) |first| {
                    if (first.isSymbol()) {
                        if (std.mem.eql(u8, first.symbol, "Struct")) {
                            current = node.next;

                            // Count fields
                            var field_count: usize = 0;
                            var count_current = current;
                            while (count_current != null) {
                                if (count_current.?.value != null) field_count += 1;
                                count_current = count_current.?.next;
                            }

                            // Parse fields
                            const fields = try self.allocator.alloc(StructField, field_count);
                            var field_idx: usize = 0;

                            while (current != null and field_idx < field_count) {
                                const field_node = current.?.value orelse return TypeCheckError.InvalidTypeAnnotation;
                                if (!field_node.isVector() or field_node.vector.len() != 2) {
                                    return TypeCheckError.InvalidTypeAnnotation;
                                }

                                const field_name_val = field_node.vector.at(0);
                                const field_type_val = field_node.vector.at(1);

                                if (!field_name_val.isSymbol()) return TypeCheckError.InvalidTypeAnnotation;

                                const field_name = field_name_val.symbol;
                                const field_type = try self.parseType(field_type_val);

                                fields[field_idx] = StructField{
                                    .name = field_name,
                                    .field_type = field_type,
                                };

                                field_idx += 1;
                                current = current.?.next;
                            }

                            // Create struct type - for now use empty name, but this should come from def
                            const struct_type = try self.allocator.create(StructType);
                            struct_type.* = StructType{
                                .name = "",
                                .fields = fields,
                            };

                            return Type{ .struct_type = struct_type };
                        } else if (std.mem.eql(u8, first.symbol, "Enum")) {
                            current = node.next;

                            // Count variants
                            var variant_count: usize = 0;
                            var count_current = current;
                            while (count_current != null) {
                                if (count_current.?.value != null) variant_count += 1;
                                count_current = count_current.?.next;
                            }

                            if (variant_count == 0) return TypeCheckError.InvalidTypeAnnotation;

                            // Parse variants
                            const variants = try self.allocator.alloc(EnumVariant, variant_count);
                            var variant_idx: usize = 0;

                            while (current != null and variant_idx < variant_count) {
                                const variant_node = current.?.value orelse return TypeCheckError.InvalidTypeAnnotation;
                                if (!variant_node.isSymbol()) return TypeCheckError.InvalidTypeAnnotation;

                                variants[variant_idx] = EnumVariant{
                                    .name = variant_node.symbol,
                                    .qualified_name = null,
                                };

                                variant_idx += 1;
                                current = current.?.next;
                            }

                            const enum_type = try self.allocator.create(EnumType);
                            enum_type.* = EnumType{
                                .name = "",
                                .variants = variants,
                            };

                            return Type{ .enum_type = enum_type };
                        } else if (std.mem.eql(u8, first.symbol, "Pointer")) {
                            // (Pointer T) - pointer type constructor
                            current = node.next;

                            if (current == null) {
                                return TypeCheckError.InvalidTypeAnnotation; // Missing pointee type
                            }

                            const pointee_node = current.?;
                            if (pointee_node.value == null) {
                                return TypeCheckError.InvalidTypeAnnotation;
                            }

                            const pointee_type = try self.parseType(pointee_node.value.?);

                            // Create pointer type
                            const ptr_type = try self.allocator.create(Type);
                            ptr_type.* = pointee_type;

                            return Type{ .pointer = ptr_type };
                        } else if (std.mem.eql(u8, first.symbol, "Array")) {
                            // (Array ElementType Size) - array type constructor
                            current = node.next;

                            if (current == null) {
                                return TypeCheckError.InvalidTypeAnnotation; // Missing element type
                            }

                            const elem_type_node = current.?;
                            if (elem_type_node.value == null) {
                                return TypeCheckError.InvalidTypeAnnotation;
                            }

                            const element_type = try self.parseType(elem_type_node.value.?);

                            // Get size
                            current = elem_type_node.next;
                            if (current == null) {
                                return TypeCheckError.InvalidTypeAnnotation; // Missing array size
                            }

                            const size_node = current.?;
                            if (size_node.value == null) {
                                return TypeCheckError.InvalidTypeAnnotation;
                            }

                            // Size must be an integer literal
                            if (!size_node.value.?.isInt()) {
                                return TypeCheckError.InvalidTypeAnnotation;
                            }

                            const size: usize = @intCast(size_node.value.?.int);

                            // Create array type
                            const array_type = try self.allocator.create(ArrayType);
                            array_type.* = ArrayType{
                                .element_type = element_type,
                                .size = size,
                            };

                            return Type{ .array = array_type };
                        }
                    }
                }
            }
        }

        return TypeCheckError.InvalidTypeAnnotation;
    }

    // Subtyping check
    fn isSubtype(self: *BidirectionalTypeChecker, sub: Type, super: Type) TypeCheckError!bool {
        _ = self;

        if (typesEqual(sub, super)) {
            return true;
        }

        // void and nil are compatible
        if ((sub == .void and super == .nil) or (sub == .nil and super == .void)) {
            return true;
        }

        // General integers can flow into any specific integer type
        if (sub == .int and isIntegerType(super)) {
            return true;
        }

        // Specific integer types are subtypes of the generic Int
        if (super == .int and isIntegerType(sub)) {
            return true;
        }

        // General floats can flow into any specific float type
        if (sub == .float and isFloatType(super)) {
            return true;
        }

        // Specific float types are subtypes of the generic Float
        if (super == .float and isFloatType(sub)) {
            return true;
        }

        // Integers can flow into floating point types
        if (isNumericType(sub) and !isFloatType(sub) and isFloatType(super)) {
            return true;
        }

        // c_string (const char*) is a subtype of (Pointer U8)
        if (sub == .c_string and super == .pointer) {
            if (super.pointer.* == .u8) {
                return true;
            }
        }

        return false;
    }

    fn isComparisonOperator(_: *BidirectionalTypeChecker, symbol: []const u8) bool {
        return isRelationalOperator(symbol) or isEqualityOperator(symbol);
    }

    fn isRelationalOperator(symbol: []const u8) bool {
        return std.mem.eql(u8, symbol, "<") or
            std.mem.eql(u8, symbol, ">") or
            std.mem.eql(u8, symbol, "<=") or
            std.mem.eql(u8, symbol, ">=");
    }

    fn isEqualityOperator(symbol: []const u8) bool {
        return std.mem.eql(u8, symbol, "=") or std.mem.eql(u8, symbol, "==") or std.mem.eql(u8, symbol, "!=");
    }

    fn isLogicalOperator(_: *BidirectionalTypeChecker, symbol: []const u8) bool {
        return std.mem.eql(u8, symbol, "and") or
            std.mem.eql(u8, symbol, "or") or
            std.mem.eql(u8, symbol, "not");
    }

    fn isBitwiseOperator(_: *BidirectionalTypeChecker, symbol: []const u8) bool {
        return std.mem.eql(u8, symbol, "bit-and") or
            std.mem.eql(u8, symbol, "bit-or") or
            std.mem.eql(u8, symbol, "bit-xor") or
            std.mem.eql(u8, symbol, "bit-not") or
            std.mem.eql(u8, symbol, "bit-shl") or
            std.mem.eql(u8, symbol, "bit-shr");
    }

    fn isTruthyType(t: Type) bool {
        return t == .bool or isNumericType(t);
    }

    fn mergeBranchTypes(self: *BidirectionalTypeChecker, then_type: Type, else_type: Type) TypeCheckError!Type {
        if (try self.isSubtype(then_type, else_type)) {
            return else_type;
        }

        if (try self.isSubtype(else_type, then_type)) {
            return then_type;
        }

        return TypeCheckError.TypeMismatch;
    }

    fn mergeNumericTypes(self: *BidirectionalTypeChecker, a: Type, b: Type) TypeCheckError!Type {
        if (!isNumericType(a) or !isNumericType(b)) {
            return TypeCheckError.TypeMismatch;
        }

        if (try self.isSubtype(a, b)) {
            return b;
        }

        if (try self.isSubtype(b, a)) {
            return a;
        }

        if (isFloatType(a) or isFloatType(b)) {
            return selectFloatType(a, b);
        }

        return TypeCheckError.TypeMismatch;
    }

    // Check if a type is numeric
    fn isNumericType(t: Type) bool {
        return switch (t) {
            .int, .float, .u8, .u16, .u32, .u64, .usize, .i8, .i16, .i32, .i64, .isize, .f32, .f64 => true,
            else => false,
        };
    }

    // Check if a type is integer
    fn isIntegerType(t: Type) bool {
        return switch (t) {
            .int, .u8, .u16, .u32, .u64, .usize, .i8, .i16, .i32, .i64, .isize => true,
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

    // Helper functions for C emission primitives
    fn isComparisonOp(op: []const u8) bool {
        return std.mem.eql(u8, op, "<") or
            std.mem.eql(u8, op, ">") or
            std.mem.eql(u8, op, "<=") or
            std.mem.eql(u8, op, ">=") or
            std.mem.eql(u8, op, "==") or
            std.mem.eql(u8, op, "!=");
    }

    fn isLogicalOp(op: []const u8) bool {
        return std.mem.eql(u8, op, "&&") or
            std.mem.eql(u8, op, "||");
    }

    fn isArithmeticOp(op: []const u8) bool {
        return std.mem.eql(u8, op, "+") or
            std.mem.eql(u8, op, "-") or
            std.mem.eql(u8, op, "*") or
            std.mem.eql(u8, op, "/") or
            std.mem.eql(u8, op, "%");
    }

    fn isBitwiseOp(op: []const u8) bool {
        return std.mem.eql(u8, op, "&") or
            std.mem.eql(u8, op, "|") or
            std.mem.eql(u8, op, "^") or
            std.mem.eql(u8, op, "<<") or
            std.mem.eql(u8, op, ">>");
    }

    // Type equality
    pub fn typesEqual(a: Type, b: Type) bool {
        if (std.meta.activeTag(a) != std.meta.activeTag(b)) return false;

        switch (a) {
            .int, .float, .string, .bool, .nil, .u8, .u16, .u32, .u64, .usize, .i8, .i16, .i32, .i64, .isize, .f32, .f64, .type_type, .c_string, .void => return true,
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
            .struct_type => |a_struct| {
                const b_struct = b.struct_type;
                if (!std.mem.eql(u8, a_struct.name, b_struct.name)) return false;
                if (a_struct.fields.len != b_struct.fields.len) return false;

                for (a_struct.fields, b_struct.fields) |a_field, b_field| {
                    if (!std.mem.eql(u8, a_field.name, b_field.name)) return false;
                    if (!typesEqual(a_field.field_type, b_field.field_type)) return false;
                }
                return true;
            },
            .enum_type => |a_enum| {
                const b_enum = b.enum_type;
                if (!std.mem.eql(u8, a_enum.name, b_enum.name)) return false;
                if (a_enum.variants.len != b_enum.variants.len) return false;

                for (a_enum.variants, b_enum.variants) |a_variant, b_variant| {
                    if (!std.mem.eql(u8, a_variant.name, b_variant.name)) return false;
                }
                return true;
            },
            .pointer => |a_pointee| {
                if (b != .pointer) return false;
                return typesEqual(a_pointee.*, b.pointer.*);
            },
            .array => |a_array| {
                if (b != .array) return false;
                const b_array = b.array;
                if (a_array.size != b_array.size) return false;
                return typesEqual(a_array.element_type, b_array.element_type);
            },
            .extern_function => |a_extern| {
                if (b != .extern_function) return false;
                const b_extern = b.extern_function;
                if (!std.mem.eql(u8, a_extern.name, b_extern.name)) return false;
                if (a_extern.param_types.len != b_extern.param_types.len) return false;
                for (a_extern.param_types, b_extern.param_types) |a_param, b_param| {
                    if (!typesEqual(a_param, b_param)) return false;
                }
                return typesEqual(a_extern.return_type, b_extern.return_type);
            },
            .extern_type => |a_extern| {
                if (b != .extern_type) return false;
                const b_extern = b.extern_type;
                return std.mem.eql(u8, a_extern.name, b_extern.name);
            },
            .type_var => |a_id| return a_id == b.type_var,
        }
    }

    fn selectFloatType(a: Type, b: Type) Type {
        if (a == .float or b == .float) return Type.float;
        const rank_a = floatRank(a);
        const rank_b = floatRank(b);
        if (rank_a >= rank_b) {
            return if (rank_a == 64) Type.f64 else Type.f32;
        } else {
            return if (rank_b == 64) Type.f64 else Type.f32;
        }
    }

    fn floatRank(t: Type) u8 {
        return switch (t) {
            .float => 64,
            .f64 => 64,
            .f32 => 32,
            else => 0,
        };
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

            .macro_def => |m| {
                // Macros should have been expanded already
                _ = m;
                return TypeCheckError.InvalidTypeAnnotation;
            },

            .namespace => |ns| {
                result.* = TypedValue{ .namespace = .{ .name = ns.name, .type = Type.nil } };
                return result;
            },

            .symbol => |name| {
                // Handle boolean literals
                if (std.mem.eql(u8, name, "true") or std.mem.eql(u8, name, "false")) {
                    result.* = TypedValue{ .symbol = .{ .name = name, .type = Type.bool } };
                    return result;
                }

                // Check if this is a type definition first
                if (self.type_defs.get(name)) |type_def| {
                    result.* = TypedValue{ .type_value = .{
                        .value_type = type_def,
                        .type = Type.type_type,
                    } };
                    return result;
                }

                // Otherwise it's a regular variable
                if (self.env.get(name)) |var_type| {
                    result.* = TypedValue{ .symbol = .{ .name = name, .type = var_type } };
                    return result;
                } else {
                    std.debug.print("ERROR: Unbound variable: '{s}'\n", .{name});
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
                    result.* = TypedValue{ .vector = .{ .elements = typed_elements, .type = Type{ .vector = vector_type_ptr } } };
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
                result.* = TypedValue{ .vector = .{ .elements = typed_elements, .type = Type{ .vector = vector_type_ptr } } };
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
                            std.mem.eql(u8, first.symbol, "%"))
                        {
                            var operands: [64]*Value = undefined;
                            var operand_count: usize = 0;
                            var merged_type_opt: ?Type = null;

                            var node_iter: ?*const @TypeOf(list.*) = list.next;
                            while (node_iter != null) {
                                const node = node_iter.?;
                                if (node.value) |operand| {
                                    if (operand_count >= operands.len) {
                                        return TypeCheckError.InvalidTypeAnnotation;
                                    }
                                    operands[operand_count] = operand;

                                    const operand_typed = try self.synthesizeTyped(operand);
                                    const operand_type = operand_typed.getType();
                                    if (!isNumericType(operand_type)) {
                                        return TypeCheckError.TypeMismatch;
                                    }

                                    merged_type_opt = if (merged_type_opt) |prev|
                                        try self.mergeNumericTypes(prev, operand_type)
                                    else
                                        operand_type;

                                    // destroy temporary typed value to avoid leaks
                                    self.allocator.destroy(operand_typed);

                                    operand_count += 1;
                                } else {
                                    break;
                                }
                                node_iter = node.next;
                            }

                            if (operand_count == 0) {
                                return TypeCheckError.InvalidTypeAnnotation;
                            }

                            var result_type = merged_type_opt.?;

                            if (std.mem.eql(u8, first.symbol, "/") and isIntegerType(result_type)) {
                                result_type = if (std.meta.activeTag(result_type) == .int) Type.float else Type.f64;
                            }

                            if (std.mem.eql(u8, first.symbol, "%") and !isIntegerType(result_type)) {
                                return TypeCheckError.TypeMismatch;
                            }

                            if (operand_count == 1 and !std.mem.eql(u8, first.symbol, "-")) {
                                return TypeCheckError.InvalidTypeAnnotation;
                            }

                            // Include operator symbol as first element
                            const op_typed = try self.allocator.create(TypedValue);
                            op_typed.* = TypedValue{ .symbol = .{ .name = first.symbol, .type = result_type } };
                            typed_elements[0] = op_typed;

                            var idx: usize = 0;
                            while (idx < operand_count) : (idx += 1) {
                                typed_elements[idx + 1] = try self.checkTyped(operands[idx], result_type);
                            }

                            result.* = TypedValue{ .list = .{
                                .elements = typed_elements[0..(operand_count + 1)],
                                .type = result_type,
                            } };
                            return result;
                        } else if (std.mem.eql(u8, first.symbol, "c-binary-op")) {
                            // (c-binary-op "op" left right)
                            var node_iter: ?*const @TypeOf(list.*) = list.next;

                            // Get operator string
                            const op_node = node_iter orelse return TypeCheckError.InvalidTypeAnnotation;
                            const op_val = op_node.value orelse return TypeCheckError.InvalidTypeAnnotation;
                            if (!op_val.isString()) return TypeCheckError.InvalidTypeAnnotation;
                            const op = op_val.string;
                            node_iter = op_node.next;

                            // Get left operand
                            const left_node = node_iter orelse return TypeCheckError.InvalidTypeAnnotation;
                            const left = left_node.value orelse return TypeCheckError.InvalidTypeAnnotation;
                            const left_typed = try self.synthesizeTyped(left);
                            const left_type = left_typed.getType();
                            node_iter = left_node.next;

                            // Get right operand
                            const right_node = node_iter orelse return TypeCheckError.InvalidTypeAnnotation;
                            const right = right_node.value orelse return TypeCheckError.InvalidTypeAnnotation;
                            const right_typed = try self.synthesizeTyped(right);
                            const right_type = right_typed.getType();

                            // Determine result type and operand check type
                            const result_type: Type = if (isComparisonOp(op))
                                Type.bool
                            else if (isLogicalOp(op))
                                Type.bool
                            else if (isArithmeticOp(op)) blk: {
                                const merged = try self.mergeNumericTypes(left_type, right_type);
                                if (std.mem.eql(u8, op, "/") and isIntegerType(merged)) {
                                    break :blk if (std.meta.activeTag(merged) == .int) Type.float else Type.f64;
                                }
                                break :blk merged;
                            } else if (isBitwiseOp(op))
                                try self.mergeNumericTypes(left_type, right_type)
                            else
                                left_type;

                            // For comparison ops, check operands against their merged type, not bool
                            const operand_check_type: Type = if (isComparisonOp(op))
                                try self.mergeNumericTypes(left_type, right_type)
                            else
                                result_type;

                            typed_elements[0] = try self.checkTyped(left, operand_check_type);
                            typed_elements[1] = try self.checkTyped(right, operand_check_type);

                            result.* = TypedValue{ .list = .{
                                .elements = typed_elements[0..2],
                                .type = result_type,
                            } };
                            return result;
                        } else if (std.mem.eql(u8, first.symbol, "c-unary-op")) {
                            // (c-unary-op "op" operand)
                            var node_iter: ?*const @TypeOf(list.*) = list.next;

                            // Get operator string
                            const op_node = node_iter orelse return TypeCheckError.InvalidTypeAnnotation;
                            const op_val = op_node.value orelse return TypeCheckError.InvalidTypeAnnotation;
                            if (!op_val.isString()) return TypeCheckError.InvalidTypeAnnotation;
                            const op = op_val.string;
                            node_iter = op_node.next;

                            // Get operand
                            const operand_node = node_iter orelse return TypeCheckError.InvalidTypeAnnotation;
                            const operand = operand_node.value orelse return TypeCheckError.InvalidTypeAnnotation;
                            const operand_typed = try self.synthesizeTyped(operand);
                            const operand_type = operand_typed.getType();

                            // Determine result type
                            const result_type: Type = if (std.mem.eql(u8, op, "!"))
                                Type.bool
                            else
                                operand_type;

                            typed_elements[0] = try self.checkTyped(operand, result_type);

                            result.* = TypedValue{ .list = .{
                                .elements = typed_elements[0..1],
                                .type = result_type,
                            } };
                            return result;
                        } else if (std.mem.eql(u8, first.symbol, "c-fold-binary-op")) {
                            // (c-fold-binary-op "op" arg1 arg2 arg3 ...)
                            var node_iter: ?*const @TypeOf(list.*) = list.next;

                            // Get operator string
                            const op_node = node_iter orelse return TypeCheckError.InvalidTypeAnnotation;
                            const op_val = op_node.value orelse return TypeCheckError.InvalidTypeAnnotation;
                            if (!op_val.isString()) return TypeCheckError.InvalidTypeAnnotation;
                            const op = op_val.string;
                            node_iter = op_node.next;

                            // Type check all operands and merge types
                            var operands: [64]*Value = undefined;
                            var operand_count: usize = 0;
                            var merged_type_opt: ?Type = null;

                            while (node_iter) |node| {
                                if (node.value) |operand| {
                                    if (operand_count >= operands.len) return TypeCheckError.InvalidTypeAnnotation;
                                    operands[operand_count] = operand;

                                    const operand_typed = try self.synthesizeTyped(operand);
                                    const operand_type = operand_typed.getType();

                                    merged_type_opt = if (merged_type_opt) |prev|
                                        try self.mergeNumericTypes(prev, operand_type)
                                    else
                                        operand_type;

                                    self.allocator.destroy(operand_typed);
                                    operand_count += 1;
                                } else {
                                    break;
                                }
                                node_iter = node.next;
                            }

                            if (operand_count == 0) return TypeCheckError.InvalidTypeAnnotation;

                            var result_type = merged_type_opt.?;
                            if (std.mem.eql(u8, op, "/") and isIntegerType(result_type)) {
                                result_type = if (std.meta.activeTag(result_type) == .int) Type.float else Type.f64;
                            }

                            var idx: usize = 0;
                            while (idx < operand_count) : (idx += 1) {
                                typed_elements[idx] = try self.checkTyped(operands[idx], result_type);
                            }

                            result.* = TypedValue{ .list = .{
                                .elements = typed_elements[0..operand_count],
                                .type = result_type,
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

                            // Get the second argument (must be type annotation)
                            const second_node = current orelse return TypeCheckError.InvalidTypeAnnotation;
                            const second_val = second_node.value.?;

                            // Check if it's a type annotation: (def name (: type) body)
                            if (second_val.isList()) {
                                const anno_current: ?*const @TypeOf(second_val.list.*) = second_val.list;
                                if (anno_current) |node| {
                                    if (node.value) |anno_first| {
                                        if (anno_first.isKeyword() and std.mem.eql(u8, anno_first.keyword, "")) {
                                            // This is a type annotation form
                                            const annotated_type = try self.parseTypeAnnotation(second_val);
                                            current = second_node.next;

                                            // Type check body
                                            const body_node = current orelse return TypeCheckError.InvalidTypeAnnotation;
                                            const body = body_node.value.?;

                                            // Check body against annotated type
                                            const typed_body = try self.checkTyped(body, annotated_type);

                                            // If this is a type definition (annotated_type is type_type),
                                            // extract the actual type from typed_body and register it
                                            if (annotated_type == .type_type) {
                                                if (typed_body.* == .type_value) {
                                                    const actual_type = typed_body.type_value.value_type;
                                                    try self.env.put(var_name, annotated_type); // Point has type Type
                                                    try self.type_defs.put(var_name, actual_type); // Point -> struct_type{Point}

                                                    // Set the name if it's a struct or enum
                                                    if (actual_type == .struct_type) {
                                                        actual_type.struct_type.name = var_name;
                                                    } else if (actual_type == .enum_type) {
                                                        try self.registerEnumVariants(var_name, actual_type.enum_type);
                                                    }
                                                }
                                            } else {
                                                // Add binding to environment for non-type definitions
                                                try self.env.put(var_name, annotated_type);
                                            }

                                            // Return the typed body as the result
                                            return typed_body;
                                        }
                                    }
                                }
                            }

                            // Fallback: regular value definition (def name value)
                            const typed_value = try self.synthesizeTyped(second_val);
                            try self.env.put(var_name, typed_value.getType());
                            return typed_value;
                        } else if (std.mem.eql(u8, first.symbol, "extern-fn") or
                            std.mem.eql(u8, first.symbol, "extern-type") or
                            std.mem.eql(u8, first.symbol, "extern-union") or
                            std.mem.eql(u8, first.symbol, "extern-struct") or
                            std.mem.eql(u8, first.symbol, "extern-var") or
                            std.mem.eql(u8, first.symbol, "include-header") or
                            std.mem.eql(u8, first.symbol, "link-library") or
                            std.mem.eql(u8, first.symbol, "compiler-flag"))
                        {
                            // Extern forms: call synthesize to get the TypedExpression, then wrap it
                            const typed_expr = try self.synthesize(expr);
                            result.* = TypedValue{
                                .list = .{
                                    .elements = typed_elements[0..0], // Empty for now
                                    .type = typed_expr.type,
                                },
                            };
                            return result;
                        } else if (std.mem.eql(u8, first.symbol, "fn")) {
                            // Call synthesize to get the function type, then construct typed list
                            const fn_typed_expr = try self.synthesize(expr);
                            // For now, call synthesize on the whole thing to get the type, but return untyped structure
                            // This is a workaround - proper solution would fully type-check function body
                            // Store the 'fn' symbol
                            typed_elements[0] = try self.allocator.create(TypedValue);
                            typed_elements[0].* = TypedValue{ .symbol = .{ .name = "fn", .type = fn_typed_expr.type } };

                            // Type check remaining elements (params vector and body)
                            var idx: usize = 1;
                            var current: ?*const @TypeOf(list.*) = list.next;
                            while (current != null and idx < count) {
                                const node = current.?;
                                if (node.value) |val| {
                                    typed_elements[idx] = try self.synthesizeTyped(val);
                                    idx += 1;
                                }
                                current = node.next;
                            }

                            result.* = TypedValue{ .list = .{ .elements = typed_elements[0..count], .type = fn_typed_expr.type } };
                            return result;
                        } else if (std.mem.eql(u8, first.symbol, "let")) {
                            return try self.synthesizeTypedLet(expr, list);
                        } else if (std.mem.eql(u8, first.symbol, "if")) {
                            return try self.synthesizeTypedIf(expr, list, typed_elements);
                        } else if (std.mem.eql(u8, first.symbol, "while")) {
                            return try self.synthesizeTypedWhile(expr, list, typed_elements);
                        } else if (std.mem.eql(u8, first.symbol, "c-for")) {
                            return try self.synthesizeTypedCFor(expr, list, typed_elements);
                        } else if (std.mem.eql(u8, first.symbol, "set!")) {
                            return try self.synthesizeTypedSet(expr, list, typed_elements);
                        } else if (self.isComparisonOperator(first.symbol)) {
                            return try self.synthesizeTypedComparison(expr, list, typed_elements);
                        } else if (std.mem.eql(u8, first.symbol, "and") or
                            std.mem.eql(u8, first.symbol, "or") or
                            std.mem.eql(u8, first.symbol, "not"))
                        {
                            return try self.synthesizeTypedLogical(expr, list, typed_elements);
                        } else if (self.isBitwiseOperator(first.symbol)) {
                            return try self.synthesizeTypedBitwise(expr, list, typed_elements);
                        } else if (std.mem.eql(u8, first.symbol, "c-str")) {
                            // c-str takes a string literal and returns CString (const char*)
                            return try self.synthesizeTypedCStr(expr, list, typed_elements);
                        } else if (std.mem.eql(u8, first.symbol, "allocate")) {
                            // allocate takes a type and a value, returns (Pointer T)
                            const args_init = list.next; // Skip 'allocate'

                            if (args_init == null) return TypeCheckError.InvalidTypeAnnotation;
                            var args = args_init;
                            const type_arg_node = args.?;
                            if (type_arg_node.value == null) return TypeCheckError.InvalidTypeAnnotation;

                            // Parse the type argument
                            const pointee_type = try self.parseType(type_arg_node.value.?);

                            // Get the value argument (optional for extern types)
                            args = type_arg_node.next;

                            // Return type is (Pointer T)
                            const ptr_type = try self.allocator.create(Type);
                            ptr_type.* = pointee_type;

                            // For extern types, allow allocation without initialization
                            if (pointee_type == .extern_type) {
                                // No initialization value needed for extern types
                                result.* = TypedValue{ .list = .{
                                    .elements = typed_elements[0..0],
                                    .type = Type{ .pointer = ptr_type },
                                } };
                                return result;
                            }

                            if (args == null) return TypeCheckError.InvalidTypeAnnotation;
                            const value_node = args.?;
                            if (value_node.value == null) return TypeCheckError.InvalidTypeAnnotation;

                            // Type check the value against the pointee type
                            const typed_value = try self.checkTyped(value_node.value.?, pointee_type);

                            typed_elements[0] = typed_value;
                            result.* = TypedValue{ .list = .{
                                .elements = typed_elements[0..1],
                                .type = Type{ .pointer = ptr_type },
                            } };
                            return result;
                        } else if (std.mem.eql(u8, first.symbol, "uninitialized")) {
                            // uninitialized takes a type, returns a value of that type (uninitialized)
                            const args_init = list.next; // Skip 'uninitialized'

                            if (args_init == null) return TypeCheckError.InvalidTypeAnnotation;
                            const type_arg_node = args_init.?;
                            if (type_arg_node.value == null) return TypeCheckError.InvalidTypeAnnotation;

                            // Parse the type argument
                            const value_type = try self.parseType(type_arg_node.value.?);

                            // Return a value of the given type (uninitialized)
                            result.* = TypedValue{ .list = .{
                                .elements = typed_elements[0..0],
                                .type = value_type,
                            } };
                            return result;
                        } else if (std.mem.eql(u8, first.symbol, "cast")) {
                            // cast takes a type and a value, returns the value as that type
                            const args_init = list.next; // Skip 'cast'

                            if (args_init == null) return TypeCheckError.InvalidTypeAnnotation;
                            var args = args_init;
                            const type_arg_node = args.?;
                            if (type_arg_node.value == null) return TypeCheckError.InvalidTypeAnnotation;

                            // Parse the target type
                            const target_type = try self.parseType(type_arg_node.value.?);

                            // Get the value argument
                            args = type_arg_node.next;
                            if (args == null) return TypeCheckError.InvalidTypeAnnotation;
                            const value_node = args.?;
                            if (value_node.value == null) return TypeCheckError.InvalidTypeAnnotation;

                            // Type check the value (synthesize its type)
                            const typed_value = try self.synthesizeTyped(value_node.value.?);

                            // Create a marker symbol to identify this as a cast operation
                            const cast_marker = try self.allocator.create(TypedValue);
                            cast_marker.* = TypedValue{ .symbol = .{ .name = "cast", .type = Type.type_type } };

                            typed_elements[0] = cast_marker;
                            typed_elements[1] = typed_value;
                            result.* = TypedValue{ .list = .{
                                .elements = typed_elements[0..2],
                                .type = target_type,
                            } };
                            return result;
                        } else if (std.mem.eql(u8, first.symbol, "dereference")) {
                            const args = list.next;

                            if (args == null) return TypeCheckError.InvalidTypeAnnotation;
                            const ptr_node = args.?;
                            if (ptr_node.value == null) return TypeCheckError.InvalidTypeAnnotation;

                            // Infer the pointer type
                            const typed_ptr = try self.synthesizeTyped(ptr_node.value.?);

                            // Check that it's actually a pointer
                            if (typed_ptr.getType() != .pointer) {
                                return TypeCheckError.TypeMismatch;
                            }

                            const pointee_type = typed_ptr.getType().pointer.*;

                            typed_elements[0] = typed_ptr;
                            result.* = TypedValue{ .list = .{
                                .elements = typed_elements[0..1],
                                .type = pointee_type,
                            } };
                            return result;
                        } else if (std.mem.eql(u8, first.symbol, "pointer-write!")) {
                            var args = list.next;

                            // Get pointer argument
                            if (args == null) return TypeCheckError.InvalidTypeAnnotation;
                            const ptr_node = args.?;
                            if (ptr_node.value == null) return TypeCheckError.InvalidTypeAnnotation;

                            const typed_ptr = try self.synthesizeTyped(ptr_node.value.?);
                            if (typed_ptr.getType() != .pointer) {
                                return TypeCheckError.TypeMismatch;
                            }

                            const pointee_type = typed_ptr.getType().pointer.*;

                            // Get value argument
                            args = ptr_node.next;
                            if (args == null) return TypeCheckError.InvalidTypeAnnotation;
                            const value_node = args.?;
                            if (value_node.value == null) return TypeCheckError.InvalidTypeAnnotation;

                            // Check value against pointee type
                            const typed_value = try self.checkTyped(value_node.value.?, pointee_type);

                            typed_elements[0] = typed_ptr;
                            typed_elements[1] = typed_value;
                            result.* = TypedValue{ .list = .{
                                .elements = typed_elements[0..2],
                                .type = Type.nil,
                            } };
                            return result;
                        } else if (std.mem.eql(u8, first.symbol, "address-of")) {
                            const args = list.next;

                            if (args == null) return TypeCheckError.InvalidTypeAnnotation;
                            const var_node = args.?;
                            if (var_node.value == null) return TypeCheckError.InvalidTypeAnnotation;

                            // The argument must be a symbol (variable name)
                            if (!var_node.value.?.isSymbol()) {
                                return TypeCheckError.InvalidTypeAnnotation;
                            }

                            const var_name = var_node.value.?.symbol;
                            const var_type = self.env.get(var_name) orelse return TypeCheckError.UnboundVariable;

                            // Create pointer type to this variable's type
                            const ptr_type = try self.allocator.create(Type);
                            ptr_type.* = var_type;

                            typed_elements[0] = try self.allocator.create(TypedValue);
                            typed_elements[0].* = TypedValue{ .symbol = .{
                                .name = var_name,
                                .type = Type{ .pointer = ptr_type },
                            } };
                            result.* = TypedValue{ .list = .{
                                .elements = typed_elements[0..1],
                                .type = Type{ .pointer = ptr_type },
                            } };
                            return result;
                        } else if (std.mem.eql(u8, first.symbol, "pointer-field-read")) {
                            var args = list.next;

                            // Get pointer
                            if (args == null) return TypeCheckError.InvalidTypeAnnotation;
                            const ptr_node = args.?;
                            if (ptr_node.value == null) return TypeCheckError.InvalidTypeAnnotation;

                            const typed_ptr = try self.synthesizeTyped(ptr_node.value.?);
                            if (typed_ptr.getType() != .pointer) {
                                return TypeCheckError.TypeMismatch;
                            }

                            const pointee_type = typed_ptr.getType().pointer.*;
                            if (pointee_type != .struct_type) {
                                return TypeCheckError.TypeMismatch;
                            }

                            // Get field name
                            args = ptr_node.next;
                            if (args == null) return TypeCheckError.InvalidTypeAnnotation;
                            const field_node = args.?;
                            if (field_node.value == null or !field_node.value.?.isSymbol()) {
                                return TypeCheckError.InvalidTypeAnnotation;
                            }

                            const field_name = field_node.value.?.symbol;
                            const struct_type = pointee_type.struct_type;

                            // Find the field
                            var field_type: ?Type = null;
                            for (struct_type.fields) |field| {
                                if (std.mem.eql(u8, field.name, field_name)) {
                                    field_type = field.field_type;
                                    break;
                                }
                            }

                            if (field_type == null) {
                                return TypeCheckError.InvalidTypeAnnotation; // Field not found
                            }

                            typed_elements[0] = typed_ptr;
                            typed_elements[1] = try self.allocator.create(TypedValue);
                            typed_elements[1].* = TypedValue{ .symbol = .{ .name = field_name, .type = Type.nil } };
                            result.* = TypedValue{ .list = .{
                                .elements = typed_elements[0..2],
                                .type = field_type.?,
                            } };
                            return result;
                        } else if (std.mem.eql(u8, first.symbol, "pointer-field-write!")) {
                            var args = list.next;

                            // Get pointer
                            if (args == null) return TypeCheckError.InvalidTypeAnnotation;
                            const ptr_node = args.?;
                            if (ptr_node.value == null) return TypeCheckError.InvalidTypeAnnotation;

                            const typed_ptr = try self.synthesizeTyped(ptr_node.value.?);
                            if (typed_ptr.getType() != .pointer) {
                                return TypeCheckError.TypeMismatch;
                            }

                            const pointee_type = typed_ptr.getType().pointer.*;
                            if (pointee_type != .struct_type) {
                                return TypeCheckError.TypeMismatch;
                            }

                            // Get field name
                            args = ptr_node.next;
                            if (args == null) return TypeCheckError.InvalidTypeAnnotation;
                            const field_node = args.?;
                            if (field_node.value == null or !field_node.value.?.isSymbol()) {
                                return TypeCheckError.InvalidTypeAnnotation;
                            }

                            const field_name = field_node.value.?.symbol;
                            const struct_type = pointee_type.struct_type;

                            // Find the field
                            var field_type: ?Type = null;
                            for (struct_type.fields) |field| {
                                if (std.mem.eql(u8, field.name, field_name)) {
                                    field_type = field.field_type;
                                    break;
                                }
                            }

                            if (field_type == null) {
                                return TypeCheckError.InvalidTypeAnnotation; // Field not found
                            }

                            // Get value argument
                            args = field_node.next;
                            if (args == null) return TypeCheckError.InvalidTypeAnnotation;
                            const value_node = args.?;
                            if (value_node.value == null) return TypeCheckError.InvalidTypeAnnotation;

                            // Check value against field type
                            const typed_value = try self.checkTyped(value_node.value.?, field_type.?);

                            typed_elements[0] = typed_ptr;
                            typed_elements[1] = try self.allocator.create(TypedValue);
                            typed_elements[1].* = TypedValue{ .symbol = .{ .name = field_name, .type = Type.nil } };
                            typed_elements[2] = typed_value;
                            result.* = TypedValue{ .list = .{
                                .elements = typed_elements[0..3],
                                .type = Type.nil,
                            } };
                            return result;
                        } else if (std.mem.eql(u8, first.symbol, "pointer-equal?")) {
                            var args = list.next;

                            // Get first pointer
                            if (args == null) return TypeCheckError.InvalidTypeAnnotation;
                            const ptr1_node = args.?;
                            if (ptr1_node.value == null) return TypeCheckError.InvalidTypeAnnotation;

                            const typed_ptr1 = try self.synthesizeTyped(ptr1_node.value.?);
                            if (typed_ptr1.getType() != .pointer) {
                                return TypeCheckError.TypeMismatch;
                            }

                            // Get second pointer
                            args = ptr1_node.next;
                            if (args == null) return TypeCheckError.InvalidTypeAnnotation;
                            const ptr2_node = args.?;
                            if (ptr2_node.value == null) return TypeCheckError.InvalidTypeAnnotation;

                            const typed_ptr2 = try self.checkTyped(ptr2_node.value.?, typed_ptr1.getType());

                            typed_elements[0] = typed_ptr1;
                            typed_elements[1] = typed_ptr2;
                            result.* = TypedValue{ .list = .{
                                .elements = typed_elements[0..2],
                                .type = Type.bool,
                            } };
                            return result;
                        } else if (std.mem.eql(u8, first.symbol, "deallocate")) {
                            const args = list.next;

                            if (args == null) return TypeCheckError.InvalidTypeAnnotation;
                            const ptr_node = args.?;
                            if (ptr_node.value == null) return TypeCheckError.InvalidTypeAnnotation;

                            const typed_ptr = try self.synthesizeTyped(ptr_node.value.?);
                            if (typed_ptr.getType() != .pointer) {
                                return TypeCheckError.TypeMismatch;
                            }

                            typed_elements[0] = typed_ptr;
                            result.* = TypedValue{ .list = .{
                                .elements = typed_elements[0..1],
                                .type = Type.nil,
                            } };
                            return result;
                        } else if (std.mem.eql(u8, first.symbol, "array")) {
                            // (array Type Size [InitValue])
                            var args = list.next;
                            if (args == null) return TypeCheckError.InvalidTypeAnnotation;

                            const type_node = args.?;
                            if (type_node.value == null) return TypeCheckError.InvalidTypeAnnotation;

                            // Parse element type from type expression
                            const elem_type = try self.parseType(type_node.value.?);

                            args = type_node.next;
                            if (args == null) return TypeCheckError.InvalidTypeAnnotation;

                            const size_node = args.?;
                            if (size_node.value == null or !size_node.value.?.isInt()) {
                                return TypeCheckError.InvalidTypeAnnotation;
                            }
                            const size: usize = @intCast(size_node.value.?.int);

                            // Check for optional init value
                            var element_count: usize = 2;
                            args = size_node.next;
                            if (args != null and args.?.value != null) {
                                // Type check init value
                                const init_typed = try self.checkTyped(args.?.value.?, elem_type);
                                typed_elements[2] = init_typed;
                                element_count = 3;
                            }

                            // Create array type
                            const array_type = try self.allocator.create(ArrayType);
                            array_type.* = ArrayType{
                                .element_type = elem_type,
                                .size = size,
                            };

                            typed_elements[0] = try self.allocator.create(TypedValue);
                            typed_elements[0].* = TypedValue{ .symbol = .{ .name = "array", .type = Type.nil } };
                            typed_elements[1] = try self.allocator.create(TypedValue);
                            typed_elements[1].* = TypedValue{ .int = .{ .value = @intCast(size), .type = Type.int } };

                            result.* = TypedValue{ .list = .{
                                .elements = typed_elements[0..element_count],
                                .type = Type{ .array = array_type },
                            } };
                            return result;
                        } else if (std.mem.eql(u8, first.symbol, "array-ref")) {
                            // (array-ref array index)
                            var args = list.next;
                            if (args == null) return TypeCheckError.InvalidTypeAnnotation;

                            const array_node = args.?;
                            if (array_node.value == null) return TypeCheckError.InvalidTypeAnnotation;

                            const array_typed = try self.synthesizeTyped(array_node.value.?);
                            const array_type = array_typed.getType();

                            if (array_type != .array) {
                                return TypeCheckError.TypeMismatch;
                            }

                            args = array_node.next;
                            if (args == null) return TypeCheckError.InvalidTypeAnnotation;

                            const index_node = args.?;
                            if (index_node.value == null) return TypeCheckError.InvalidTypeAnnotation;

                            const index_typed = try self.synthesizeTyped(index_node.value.?);
                            const index_type = index_typed.getType();

                            // Index must be an integer type
                            if (!isIntegerType(index_type) and index_type != .int) {
                                return TypeCheckError.TypeMismatch;
                            }

                            typed_elements[0] = try self.allocator.create(TypedValue);
                            typed_elements[0].* = TypedValue{ .symbol = .{ .name = "array-ref", .type = Type.nil } };
                            typed_elements[1] = array_typed;
                            typed_elements[2] = index_typed;

                            result.* = TypedValue{ .list = .{
                                .elements = typed_elements[0..3],
                                .type = array_type.array.element_type,
                            } };
                            return result;
                        } else if (std.mem.eql(u8, first.symbol, "array-set!")) {
                            // (array-set! array index value)
                            var args = list.next;
                            if (args == null) return TypeCheckError.InvalidTypeAnnotation;

                            const array_node = args.?;
                            if (array_node.value == null) return TypeCheckError.InvalidTypeAnnotation;

                            const array_typed = try self.synthesizeTyped(array_node.value.?);
                            const array_type = array_typed.getType();

                            if (array_type != .array) {
                                return TypeCheckError.TypeMismatch;
                            }

                            args = array_node.next;
                            if (args == null) return TypeCheckError.InvalidTypeAnnotation;

                            const index_node = args.?;
                            if (index_node.value == null) return TypeCheckError.InvalidTypeAnnotation;

                            const index_typed = try self.synthesizeTyped(index_node.value.?);
                            const index_type = index_typed.getType();

                            // Index must be an integer type
                            if (!isIntegerType(index_type) and index_type != .int) {
                                return TypeCheckError.TypeMismatch;
                            }

                            args = index_node.next;
                            if (args == null) return TypeCheckError.InvalidTypeAnnotation;

                            const value_node = args.?;
                            if (value_node.value == null) return TypeCheckError.InvalidTypeAnnotation;

                            // Check that value has the right type for the array
                            const value_typed = try self.checkTyped(value_node.value.?, array_type.array.element_type);

                            typed_elements[0] = try self.allocator.create(TypedValue);
                            typed_elements[0].* = TypedValue{ .symbol = .{ .name = "array-set!", .type = Type.nil } };
                            typed_elements[1] = array_typed;
                            typed_elements[2] = index_typed;
                            typed_elements[3] = value_typed;

                            result.* = TypedValue{ .list = .{
                                .elements = typed_elements[0..4],
                                .type = Type.nil,
                            } };
                            return result;
                        } else if (std.mem.eql(u8, first.symbol, "array-length")) {
                            // (array-length array)
                            const args = list.next;
                            if (args == null) return TypeCheckError.InvalidTypeAnnotation;

                            const array_node = args.?;
                            if (array_node.value == null) return TypeCheckError.InvalidTypeAnnotation;

                            const array_typed = try self.synthesizeTyped(array_node.value.?);
                            const array_type = array_typed.getType();

                            if (array_type != .array) {
                                return TypeCheckError.TypeMismatch;
                            }

                            typed_elements[0] = try self.allocator.create(TypedValue);
                            typed_elements[0].* = TypedValue{ .symbol = .{ .name = "array-length", .type = Type.nil } };
                            typed_elements[1] = array_typed;

                            result.* = TypedValue{ .list = .{
                                .elements = typed_elements[0..2],
                                .type = Type.int,
                            } };
                            return result;
                        } else if (std.mem.eql(u8, first.symbol, "array-ptr")) {
                            // (array-ptr array index) -> (Pointer ElementType)
                            const args = list.next;
                            if (args == null) return TypeCheckError.InvalidTypeAnnotation;

                            const array_node = args.?;
                            if (array_node.value == null) return TypeCheckError.InvalidTypeAnnotation;

                            const index_node = array_node.next;
                            if (index_node == null or index_node.?.value == null) return TypeCheckError.InvalidTypeAnnotation;

                            const array_typed = try self.synthesizeTyped(array_node.value.?);
                            const index_typed = try self.synthesizeTyped(index_node.?.value.?);

                            const array_type = array_typed.getType();
                            const index_type = index_typed.getType();

                            if (array_type != .array) {
                                return TypeCheckError.TypeMismatch;
                            }

                            if (!isIntegerType(index_type) and index_type != .int) {
                                return TypeCheckError.TypeMismatch;
                            }

                            // Create pointer to element type
                            const element_type_ptr = try self.allocator.create(Type);
                            element_type_ptr.* = array_type.array.element_type;

                            typed_elements[0] = try self.allocator.create(TypedValue);
                            typed_elements[0].* = TypedValue{ .symbol = .{ .name = "array-ptr", .type = Type.nil } };
                            typed_elements[1] = array_typed;
                            typed_elements[2] = index_typed;

                            result.* = TypedValue{ .list = .{
                                .elements = typed_elements[0..3],
                                .type = Type{ .pointer = element_type_ptr },
                            } };
                            return result;
                        } else if (std.mem.eql(u8, first.symbol, "allocate-array")) {
                            // (allocate-array Type size) or (allocate-array Type size init-value)
                            // Returns (Pointer Type)
                            const args = list.next;
                            if (args == null) return TypeCheckError.InvalidTypeAnnotation;

                            const type_node = args.?;
                            if (type_node.value == null) return TypeCheckError.InvalidTypeAnnotation;

                            const size_node = type_node.next;
                            if (size_node == null or size_node.?.value == null) return TypeCheckError.InvalidTypeAnnotation;

                            // Optional init value
                            const init_node = size_node.?.next;

                            const element_type = try self.parseType(type_node.value.?);
                            const size_typed = try self.synthesizeTyped(size_node.?.value.?);

                            const size_type = size_typed.getType();
                            if (!isIntegerType(size_type) and size_type != .int) {
                                return TypeCheckError.TypeMismatch;
                            }

                            var element_count: usize = 3;
                            if (init_node != null and init_node.?.value != null) {
                                const init_typed = try self.checkTyped(init_node.?.value.?, element_type);
                                typed_elements[3] = init_typed;
                                element_count = 4;
                            }

                            typed_elements[0] = try self.allocator.create(TypedValue);
                            typed_elements[0].* = TypedValue{ .symbol = .{ .name = "allocate-array", .type = Type.nil } };
                            typed_elements[1] = try self.allocator.create(TypedValue);
                            typed_elements[1].* = TypedValue{ .type_value = .{ .value_type = element_type, .type = Type.type_type } };
                            typed_elements[2] = size_typed;

                            const element_type_ptr = try self.allocator.create(Type);
                            element_type_ptr.* = element_type;

                            result.* = TypedValue{ .list = .{
                                .elements = typed_elements[0..element_count],
                                .type = Type{ .pointer = element_type_ptr },
                            } };
                            return result;
                        } else if (std.mem.eql(u8, first.symbol, "deallocate-array")) {
                            // (deallocate-array pointer) -> Nil
                            const args = list.next;
                            if (args == null) return TypeCheckError.InvalidTypeAnnotation;

                            const ptr_node = args.?;
                            if (ptr_node.value == null) return TypeCheckError.InvalidTypeAnnotation;

                            const ptr_typed = try self.synthesizeTyped(ptr_node.value.?);
                            const ptr_type = ptr_typed.getType();

                            if (ptr_type != .pointer) {
                                return TypeCheckError.TypeMismatch;
                            }

                            typed_elements[0] = try self.allocator.create(TypedValue);
                            typed_elements[0].* = TypedValue{ .symbol = .{ .name = "deallocate-array", .type = Type.nil } };
                            typed_elements[1] = ptr_typed;

                            result.* = TypedValue{ .list = .{
                                .elements = typed_elements[0..2],
                                .type = Type.nil,
                            } };
                            return result;
                        } else if (std.mem.eql(u8, first.symbol, "pointer-index-read")) {
                            // (pointer-index-read pointer index) -> ElementType
                            const args = list.next;
                            if (args == null) return TypeCheckError.InvalidTypeAnnotation;

                            const ptr_node = args.?;
                            if (ptr_node.value == null) return TypeCheckError.InvalidTypeAnnotation;

                            const index_node = ptr_node.next;
                            if (index_node == null or index_node.?.value == null) return TypeCheckError.InvalidTypeAnnotation;

                            const ptr_typed = try self.synthesizeTyped(ptr_node.value.?);
                            const index_typed = try self.synthesizeTyped(index_node.?.value.?);

                            const ptr_type = ptr_typed.getType();
                            const index_type = index_typed.getType();

                            if (ptr_type != .pointer) {
                                return TypeCheckError.TypeMismatch;
                            }

                            if (!isIntegerType(index_type) and index_type != .int) {
                                return TypeCheckError.TypeMismatch;
                            }

                            typed_elements[0] = try self.allocator.create(TypedValue);
                            typed_elements[0].* = TypedValue{ .symbol = .{ .name = "pointer-index-read", .type = Type.nil } };
                            typed_elements[1] = ptr_typed;
                            typed_elements[2] = index_typed;

                            result.* = TypedValue{ .list = .{
                                .elements = typed_elements[0..3],
                                .type = ptr_type.pointer.*,
                            } };
                            return result;
                        } else if (std.mem.eql(u8, first.symbol, "pointer-index-write!")) {
                            // (pointer-index-write! pointer index value) -> Nil
                            const args = list.next;
                            if (args == null) return TypeCheckError.InvalidTypeAnnotation;

                            const ptr_node = args.?;
                            if (ptr_node.value == null) return TypeCheckError.InvalidTypeAnnotation;

                            const index_node = ptr_node.next;
                            if (index_node == null or index_node.?.value == null) return TypeCheckError.InvalidTypeAnnotation;

                            const value_node = index_node.?.next;
                            if (value_node == null or value_node.?.value == null) return TypeCheckError.InvalidTypeAnnotation;

                            const ptr_typed = try self.synthesizeTyped(ptr_node.value.?);
                            const index_typed = try self.synthesizeTyped(index_node.?.value.?);

                            const ptr_type = ptr_typed.getType();
                            const index_type = index_typed.getType();

                            if (ptr_type != .pointer) {
                                return TypeCheckError.TypeMismatch;
                            }

                            if (!isIntegerType(index_type) and index_type != .int) {
                                return TypeCheckError.TypeMismatch;
                            }

                            const element_type = ptr_type.pointer.*;
                            const value_typed = try self.checkTyped(value_node.?.value.?, element_type);

                            typed_elements[0] = try self.allocator.create(TypedValue);
                            typed_elements[0].* = TypedValue{ .symbol = .{ .name = "pointer-index-write!", .type = Type.nil } };
                            typed_elements[1] = ptr_typed;
                            typed_elements[2] = index_typed;
                            typed_elements[3] = value_typed;

                            result.* = TypedValue{ .list = .{
                                .elements = typed_elements[0..4],
                                .type = Type.nil,
                            } };
                            return result;
                        }
                        // Add other special forms as needed
                    }
                }

                // Check for field access: (. struct-expr field-name)
                if (list.value) |first_val| {
                    if (first_val.isSymbol() and std.mem.eql(u8, first_val.symbol, ".")) {
                        // Field access syntax: (. struct-expr field-name)
                        const struct_node = list.next orelse return TypeCheckError.InvalidTypeAnnotation;
                        const field_node = struct_node.next orelse return TypeCheckError.InvalidTypeAnnotation;

                        const struct_expr = struct_node.value orelse return TypeCheckError.InvalidTypeAnnotation;
                        const field_name_val = field_node.value orelse return TypeCheckError.InvalidTypeAnnotation;

                        if (!field_name_val.isSymbol()) return TypeCheckError.InvalidTypeAnnotation;
                        const field_name = field_name_val.symbol;

                        // Type check the struct expression
                        const struct_typed = try self.synthesizeTyped(struct_expr);
                        const struct_type_result = struct_typed.getType();

                        // Find the field - works for both struct_type and extern_type with fields
                        var field_type: ?Type = null;
                        var fields: ?[]const StructField = null;

                        if (struct_type_result == .struct_type) {
                            fields = struct_type_result.struct_type.fields;
                        } else if (struct_type_result == .extern_type) {
                            fields = struct_type_result.extern_type.fields;
                        } else if (struct_type_result == .pointer) {
                            // Handle pointer to struct/extern type - dereference first
                            const pointee = struct_type_result.pointer.*;
                            if (pointee == .struct_type) {
                                fields = pointee.struct_type.fields;
                            } else if (pointee == .extern_type) {
                                fields = pointee.extern_type.fields;
                            }
                        }

                        if (fields == null) {
                            return TypeCheckError.TypeMismatch; // Not a struct or extern type with fields
                        }

                        for (fields.?) |field| {
                            if (std.mem.eql(u8, field.name, field_name)) {
                                field_type = field.field_type;
                                break;
                            }
                        }

                        if (field_type == null) {
                            return TypeCheckError.TypeMismatch; // Field not found
                        }

                        // Return a list representing the field access with the field's type
                        typed_elements[0] = try self.allocator.create(TypedValue);
                        typed_elements[0].* = TypedValue{ .symbol = .{ .name = ".", .type = Type.nil } };
                        typed_elements[1] = struct_typed;
                        typed_elements[2] = try self.allocator.create(TypedValue);
                        typed_elements[2].* = TypedValue{ .symbol = .{ .name = field_name, .type = Type.nil } };

                        result.* = TypedValue{ .list = .{
                            .elements = typed_elements[0..3],
                            .type = field_type.?,
                        } };
                        return result;
                    }
                }

                // Default: check if it's a function application or struct construction
                if (list.value) |first_val| {
                    const first_typed = try self.synthesizeTyped(first_val);

                    // Check for struct construction: (Point 1 2)
                    // Handle both regular structs (type_type) and extern structs (extern_type)
                    const first_type = first_typed.getType();
                    var the_type_opt: ?Type = null;

                    if (first_type == .type_type) {
                        the_type_opt = first_typed.type_value.value_type;
                    } else if (first_type == .extern_type) {
                        the_type_opt = first_type;
                    }

                    if (the_type_opt) |the_type| {
                        if (the_type == .struct_type) {
                            const struct_type = the_type.struct_type;

                            // Count arguments
                            var arg_count: usize = 0;
                            var current: ?*const @TypeOf(list.*) = list.next;
                            while (current != null) {
                                if (current.?.value != null) {
                                    arg_count += 1;
                                }
                                current = current.?.next;
                            }

                            if (arg_count != struct_type.fields.len) {
                                return TypeCheckError.ArgumentCountMismatch;
                            }

                            // Type check field values
                            const field_values = try self.allocator.alloc(*TypedValue, arg_count);
                            current = list.next;
                            var field_idx: usize = 0;
                            while (current != null and field_idx < struct_type.fields.len) {
                                const arg_node = current.?;
                                if (arg_node.value) |arg_val| {
                                    field_values[field_idx] = try self.checkTyped(arg_val, struct_type.fields[field_idx].field_type);
                                }
                                current = arg_node.next;
                                field_idx += 1;
                            }

                            result.* = TypedValue{ .struct_instance = .{
                                .field_values = field_values,
                                .type = the_type,
                            } };
                            return result;
                        } else if (the_type == .extern_type) {
                            const extern_type = the_type.extern_type;

                            // Only allow construction if it has fields
                            if (extern_type.fields == null) {
                                return TypeCheckError.TypeMismatch;
                            }

                            const extern_fields = extern_type.fields.?;

                            // Count arguments
                            var arg_count: usize = 0;
                            var current: ?*const @TypeOf(list.*) = list.next;
                            while (current != null) {
                                if (current.?.value != null) {
                                    arg_count += 1;
                                }
                                current = current.?.next;
                            }

                            if (arg_count != extern_fields.len) {
                                return TypeCheckError.ArgumentCountMismatch;
                            }

                            // Type check field values
                            const field_values = try self.allocator.alloc(*TypedValue, arg_count);
                            current = list.next;
                            var field_idx: usize = 0;
                            while (current != null and field_idx < extern_fields.len) {
                                const arg_node = current.?;
                                if (arg_node.value) |arg_val| {
                                    field_values[field_idx] = try self.checkTyped(arg_val, extern_fields[field_idx].field_type);
                                }
                                current = arg_node.next;
                                field_idx += 1;
                            }

                            result.* = TypedValue{ .struct_instance = .{
                                .field_values = field_values,
                                .type = the_type,
                            } };
                            return result;
                        }
                    }

                    // Check if this is a function or function pointer application
                    const is_function_call = first_type == .function or
                        (first_type == .pointer and first_type.pointer.* == .function);

                    if (is_function_call) {
                        // This is a function application - synthesize the application
                        const func_type = if (first_type == .function)
                            first_type.function
                        else
                            first_type.pointer.function;

                        // Store the typed function in elements array
                        typed_elements[0] = first_typed;

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

                        // Type check and store arguments
                        current = list.next;
                        var arg_idx: usize = 0;
                        while (current != null and arg_idx < func_type.param_types.len) {
                            const arg_node = current.?;
                            if (arg_node.value) |arg_val| {
                                typed_elements[arg_idx + 1] = try self.checkTyped(arg_val, func_type.param_types[arg_idx]);
                            }
                            current = arg_node.next;
                            arg_idx += 1;
                        }

                        // Return the function's return type
                        result.* = TypedValue{
                            .list = .{
                                .elements = typed_elements[0..(arg_count + 1)], // Store function + all arguments
                                .type = func_type.return_type,
                            },
                        };
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
                result.* = TypedValue{ .list = .{ .elements = typed_elements, .type = self.freshVar() } };
                return result;
            },

            .map => {
                // Maps not fully implemented yet
                result.* = TypedValue{ .map = .{ .entries = try self.allocator.alloc(*TypedValue.MapEntry, 0), .type = Type{ .map = try self.allocator.create(MapType) } } };
                return result;
            },
        }
    }

    // Checking mode that produces a fully typed AST
    pub fn checkTyped(self: *BidirectionalTypeChecker, expr: *Value, expected: Type) TypeCheckError!*TypedValue {
        // Handle special forms that need expected type
        switch (expr.*) {
            .macro_def => |m| {
                // Macros should have been expanded already
                _ = m;
                return TypeCheckError.InvalidTypeAnnotation;
            },
            .symbol => |name| {
                // Check for pointer-null constant
                if (std.mem.eql(u8, name, "pointer-null")) {
                    // pointer-null is polymorphic, its type depends on context
                    if (expected == .pointer) {
                        const result = try self.allocator.create(TypedValue);
                        result.* = TypedValue{ .nil = .{ .type = expected } };
                        return result;
                    } else {
                        return TypeCheckError.TypeMismatch;
                    }
                }
            },
            .list => |list| {
                if (list.value) |first| {
                    if (first.isSymbol()) {
                        if (std.mem.eql(u8, first.symbol, "fn")) {
                            return try self.checkFunctionTyped(expr, list, expected);
                        }
                        // Handle (Struct ...) and (Enum ...) when expecting Type
                        if (expected == .type_type) {
                            if (std.mem.eql(u8, first.symbol, "Struct") or std.mem.eql(u8, first.symbol, "Enum")) {
                                const the_type = try self.parseType(expr);
                                const result = try self.allocator.create(TypedValue);
                                result.* = TypedValue{ .type_value = .{
                                    .value_type = the_type,
                                    .type = Type.type_type,
                                } };
                                return result;
                            }
                        }
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
            std.debug.print("ERROR: TypeMismatch - expected: {any}, actual: {any}\n", .{ expected, actual });
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
            .type_value => |*v| v.type = expected,
            .namespace => |*v| v.type = expected,
            .struct_instance => |*v| v.type = expected,
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
            try results.append(self.allocator, typed);
        }

        return results;
    }

    // Two-pass type checking for forward references (fully typed)
    pub fn typeCheckAllTwoPass(self: *BidirectionalTypeChecker, expressions: []const *Value) !TypeCheckReport {
        // Pass 1: Collect all top-level definitions and their type signatures
        for (expressions) |expr| {
            if (expr.isList()) {
                var current: ?*const @TypeOf(expr.list.*) = expr.list;
                if (current) |node| {
                    if (node.value) |first| {
                        // Handle extern declarations in pass 1
                        if (first.isSymbol() and (std.mem.eql(u8, first.symbol, "extern-fn") or
                            std.mem.eql(u8, first.symbol, "extern-type") or
                            std.mem.eql(u8, first.symbol, "extern-union") or
                            std.mem.eql(u8, first.symbol, "extern-struct") or
                            std.mem.eql(u8, first.symbol, "extern-var") or
                            std.mem.eql(u8, first.symbol, "include-header") or
                            std.mem.eql(u8, first.symbol, "link-library") or
                            std.mem.eql(u8, first.symbol, "compiler-flag")))
                        {
                            _ = self.synthesize(expr) catch |err| {
                                std.debug.print("Pass 1: Failed to synthesize {s}: {}\n", .{ first.symbol, err });
                                continue;
                            };
                            continue;
                        }

                        if (first.isSymbol() and std.mem.eql(u8, first.symbol, "def")) {
                            // Extract name and type annotation without checking body
                            current = node.next;

                            // Get variable name
                            const name_node = current orelse continue;
                            if (!name_node.value.?.isSymbol()) continue;
                            const var_name = name_node.value.?.symbol;
                            current = name_node.next;

                            // Get second argument (must be type annotation)
                            const second_node = current orelse continue;
                            const second_val = second_node.value.?;

                            // Parse type annotation: (def Point (: Type) ...)
                            const annotated_type = self.parseTypeAnnotation(second_val) catch continue;

                            // If this is a type definition (: Type), peek at the body to get the actual type
                            var actual_type = annotated_type;
                            if (annotated_type == .type_type) {
                                // Get the body (the third argument)
                                const body_node = second_node.next;
                                if (body_node) |bn| {
                                    if (bn.value) |body_val| {
                                        // Try to parse the body as a type (Struct or Enum)
                                        actual_type = self.parseType(body_val) catch annotated_type;
                                    }
                                }
                            }

                            const is_struct_type_def = actual_type == .struct_type and actual_type.struct_type.name.len == 0;
                            const is_enum_type_def = actual_type == .enum_type and actual_type.enum_type.name.len == 0;

                            if (is_struct_type_def) {
                                actual_type.struct_type.name = var_name;
                            }
                            if (is_enum_type_def) {
                                self.registerEnumVariants(var_name, actual_type.enum_type) catch continue;
                            }

                            // Add to environment for forward references
                            // For type definitions, store Type in env, actual type in type_defs
                            try self.env.put(var_name, annotated_type);
                            // Add to type_defs if this is a struct/enum type definition
                            if (is_struct_type_def or is_enum_type_def) {
                                try self.type_defs.put(var_name, actual_type);
                            }
                        }
                    }
                }
            }
        }

        // Pass 2: Type check all expressions with forward references available
        var results = ArrayList(*TypedValue){};

        for (expressions, 0..) |expr, index| {
            self.index = index;
            if (expr.isList()) {
                var current: ?*const @TypeOf(expr.list.*) = expr.list;
                if (current) |node| {
                    if (node.value) |first| {
                        // Skip extern declarations in pass 2 - they were handled in pass 1
                        if (first.isSymbol() and (std.mem.eql(u8, first.symbol, "extern-fn") or
                            std.mem.eql(u8, first.symbol, "extern-type") or
                            std.mem.eql(u8, first.symbol, "extern-union") or
                            std.mem.eql(u8, first.symbol, "extern-struct") or
                            std.mem.eql(u8, first.symbol, "extern-var") or
                            std.mem.eql(u8, first.symbol, "include-header") or
                            std.mem.eql(u8, first.symbol, "link-library") or
                            std.mem.eql(u8, first.symbol, "compiler-flag")))
                        {
                            // Synthesize again to get the typed value for the results
                            const typed = self.synthesizeTyped(expr) catch |err| {
                                if (err == TypeCheckError.OutOfMemory) return err;
                                try self.errors.append(self.allocator, .{
                                    .index = index,
                                    .expr = expr,
                                    .err = err,
                                    .info = null,
                                });
                                continue;
                            };
                            try results.append(self.allocator, typed);
                            continue;
                        }

                        if (first.isSymbol() and std.mem.eql(u8, first.symbol, "def")) {
                            // For def expressions, just check the body against the already-established type
                            current = node.next;
                            // Get variable name
                            const name_node = current orelse continue;
                            if (!name_node.value.?.isSymbol()) continue;
                            const var_name = name_node.value.?.symbol;
                            current = name_node.next;

                            // Get second argument (must be type annotation)
                            const second_node = current orelse continue;
                            const second_val = second_node.value.?;

                            // Parse type annotation: (def name (: Type) value)
                            const annotated_type = self.parseTypeAnnotation(second_val) catch |err| {
                                if (err == TypeCheckError.OutOfMemory) return err;
                                try self.errors.append(self.allocator, .{
                                    .index = index,
                                    .expr = second_val,
                                    .err = err,
                                    .info = null,
                                });
                                continue;
                            };
                            current = second_node.next;

                            const is_struct_type_def = annotated_type == .struct_type and annotated_type.struct_type.name.len == 0;
                            const is_enum_type_def = annotated_type == .enum_type and annotated_type.enum_type.name.len == 0;

                            if (is_struct_type_def) {
                                annotated_type.struct_type.name = var_name;
                            }
                            if (is_enum_type_def) {
                                self.registerEnumVariants(var_name, annotated_type.enum_type) catch continue;
                            }

                            // Get body and type check it
                            const body_node = current orelse continue;
                            const body = body_node.value.?;
                            const typed_body = self.checkTyped(body, annotated_type) catch |err| {
                                if (err == TypeCheckError.OutOfMemory) return err;
                                try self.errors.append(self.allocator, .{
                                    .index = index,
                                    .expr = body,
                                    .err = err,
                                    .info = null,
                                });
                                continue;
                            };

                            // If this is a type definition (annotated_type is type_type),
                            // extract the actual type from typed_body and register it
                            if (annotated_type == .type_type) {
                                if (typed_body.* == .type_value) {
                                    const actual_type = typed_body.type_value.value_type;
                                    try self.env.put(var_name, annotated_type); // Point has type Type
                                    try self.type_defs.put(var_name, actual_type); // Point -> struct_type{Point}

                                    // Set the name if it's a struct or enum
                                    if (actual_type == .struct_type) {
                                        actual_type.struct_type.name = var_name;
                                    } else if (actual_type == .enum_type) {
                                        self.registerEnumVariants(var_name, actual_type.enum_type) catch continue;
                                    }
                                }
                            } else if (is_struct_type_def or is_enum_type_def) {
                                try self.env.put(var_name, annotated_type);
                            }

                            try results.append(self.allocator, typed_body);
                            continue;
                        }
                    }
                }
            }
            // For non-def expressions, use normal synthesis
            const typed = self.synthesizeTyped(expr) catch |err| {
                if (err == TypeCheckError.OutOfMemory) return err;
                try self.errors.append(self.allocator, .{
                    .index = index,
                    .expr = expr,
                    .err = err,
                    .info = null,
                });
                continue;
            };
            try results.append(self.allocator, typed);
        }

        return TypeCheckReport{
            .typed = results,
            .errors = self.errors,
        };
    }
};

// Utility functions for creating types
pub fn createIntType(allocator: std.mem.Allocator) !*Type {
    const type_ptr = try allocator.create(Type);
    type_ptr.* = Type.int;
    return type_ptr;
}

pub fn createFunctionType(allocator: std.mem.Allocator, param_types: []const Type, return_type: Type) !Type {
    const func_type = try allocator.create(FunctionType);
    func_type.* = FunctionType{
        .param_types = param_types,
        .return_type = return_type,
    };
    return Type{ .function = func_type };
}
