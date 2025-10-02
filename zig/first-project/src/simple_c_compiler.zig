const std = @import("std");
const builtin = @import("builtin");
const reader = @import("reader.zig");
const value = @import("value.zig");
const type_checker = @import("type_checker.zig");

const Reader = reader.Reader;
const Value = value.Value;
const TypeChecker = type_checker.BidirectionalTypeChecker;
const Type = type_checker.Type;
const TypedValue = type_checker.TypedValue;

const IncludeFlags = struct {
    need_stdint: bool = false,
    need_stdbool: bool = false,
    need_stddef: bool = false,
};

pub const SimpleCCompiler = struct {
    allocator: *std.mem.Allocator,

    pub const TargetKind = enum {
        executable,
        bundle,
    };

    const NamespaceDef = struct {
        name: []const u8,
        expr: *Value,
        typed: *TypedValue,
        var_type: Type,
    };

    const NamespaceContext = struct {
        name: ?[]const u8,
        def_names: *std.StringHashMap(void),
        in_init_function: bool = false,
    };

    pub const Error = error{
        UnsupportedExpression,
        MissingOperand,
        InvalidDefinition,
        InvalidFunction,
        InvalidIfForm,
        UnsupportedType,
        TypeCheckFailed,
        UnboundVariable,
        TypeMismatch,
        CannotSynthesize,
        CannotApplyNonFunction,
        ArgumentCountMismatch,
        InvalidTypeAnnotation,
        UnexpectedToken,
        UnterminatedString,
        UnterminatedList,
        UnterminatedVector,
        UnterminatedMap,
        InvalidNumber,
        OutOfMemory,
    };

    const int_type_name = "long long";
    const int_printf_format = "%lld";

    pub fn init(allocator: *std.mem.Allocator) SimpleCCompiler {
        return SimpleCCompiler{ .allocator = allocator };
    }

    pub fn compileString(self: *SimpleCCompiler, source: []const u8, target: TargetKind) Error![]u8 {
        var reader_instance = Reader.init(self.allocator);
        var expressions = reader_instance.readAllString(source) catch |err| {
            return switch (err) {
                error.UnexpectedToken => Error.UnexpectedToken,
                error.UnterminatedString => Error.UnterminatedString,
                error.UnterminatedList => Error.UnterminatedList,
                error.UnterminatedVector => Error.UnterminatedVector,
                error.UnterminatedMap => Error.UnterminatedMap,
                error.InvalidNumber => Error.InvalidNumber,
                error.OutOfMemory => Error.OutOfMemory,
            };
        };
        defer expressions.deinit(self.allocator.*);

        var checker = TypeChecker.init(self.allocator.*);
        defer checker.deinit();

        var report = try checker.typeCheckAllTwoPass(expressions.items);
        defer report.typed.deinit(self.allocator.*);
        defer report.errors.deinit(self.allocator.*);

        if (report.errors.items.len != 0 or report.typed.items.len != expressions.items.len) {
            if (report.errors.items.len == 0) {
                std.debug.print("Type check failed: expected {d} typed expressions, got {d}\n",
                    .{ expressions.items.len, report.typed.items.len });
            }
            for (report.errors.items) |detail| {
                const maybe_str = self.formatValue(detail.expr) catch null;
                if (maybe_str) |expr_str| {
                    defer self.allocator.*.free(expr_str);
                    std.debug.print("Type error (expr {d}): {s} -> {s}\n",
                        .{ detail.index, @errorName(detail.err), expr_str });
                } else {
                    std.debug.print("Type error (expr {d}): {s}\n",
                        .{ detail.index, @errorName(detail.err) });
                }
            }
            return Error.TypeCheckFailed;
        }

        // Collect namespace info and definitions
        var namespace_name: ?[]const u8 = null;
        var namespace_defs = std.ArrayList(NamespaceDef){};
        defer namespace_defs.deinit(self.allocator.*);
        var non_def_exprs = std.ArrayList(struct { expr: *Value, typed: *TypedValue, idx: usize }){};
        defer non_def_exprs.deinit(self.allocator.*);

        for (expressions.items, 0..) |expr, idx| {
            const typed_val = report.typed.items[idx];

            // Check for namespace declaration
            if (expr.* == .namespace) {
                namespace_name = expr.namespace.name;
                continue;
            }

            // Check if this is a def
            if (expr.* == .list) {
                var iter = expr.list.iterator();
                if (iter.next()) |head_val| {
                    if (head_val.isSymbol() and std.mem.eql(u8, head_val.symbol, "def")) {
                        if (iter.next()) |name_val| {
                            if (name_val.isSymbol()) {
                                const def_name = name_val.symbol;
                                // Skip type definitions (struct/enum) - they're handled differently
                                if (checker.type_defs.get(def_name) == null) {
                                    if (checker.env.get(def_name)) |var_type| {
                                        // Include both regular vars and functions
                                        try namespace_defs.append(self.allocator.*, .{
                                            .name = def_name,
                                            .expr = expr,
                                            .typed = typed_val,
                                            .var_type = var_type,
                                        });
                                        continue;
                                    }
                                }
                            }
                        }
                    }
                }
            }

            // Not a namespace or def - it's a regular expression
            try non_def_exprs.append(self.allocator.*, .{ .expr = expr, .typed = typed_val, .idx = idx });
        }

        var forward_decls = std.ArrayList(u8){};
        defer forward_decls.deinit(self.allocator.*);
        var prelude = std.ArrayList(u8){};
        defer prelude.deinit(self.allocator.*);
        var body = std.ArrayList(u8){};
        defer body.deinit(self.allocator.*);

        const forward_writer = forward_decls.writer(self.allocator.*);
        const prelude_writer = prelude.writer(self.allocator.*);
        const body_writer = body.writer(self.allocator.*);

        var includes = IncludeFlags{};

        // First pass: emit forward declarations for functions and structs
        for (expressions.items, 0..) |expr, idx| {
            try self.emitForwardDecl(forward_writer, expr, report.typed.items[idx], &checker, &includes);
        }

        // Create empty context for non-namespace code (used in emitDefinition/emitFunctionDefinition)
        var empty_def_names = std.StringHashMap(void).init(self.allocator.*);
        defer empty_def_names.deinit();
        _ = &empty_def_names; // Mark as used to suppress warning
        var empty_ctx = NamespaceContext{
            .name = null,
            .def_names = &empty_def_names,
        };
        _ = &empty_ctx; // Mark as used (referenced in nested functions)

        // If we have namespace defs, generate the namespace struct
        var namespace_struct = std.ArrayList(u8){};
        defer namespace_struct.deinit(self.allocator.*);
        var namespace_init = std.ArrayList(u8){};
        defer namespace_init.deinit(self.allocator.*);

        if (namespace_defs.items.len > 0) {
            const ns_writer = namespace_struct.writer(self.allocator.*);
            const init_writer = namespace_init.writer(self.allocator.*);

            const ns_name = namespace_name orelse "user";
            const sanitized_ns = try self.sanitizeIdentifier(ns_name);
            defer self.allocator.*.free(sanitized_ns);

            // Generate namespace struct
            try ns_writer.print("typedef struct {{\n", .{});
            for (namespace_defs.items) |def| {
                const sanitized_field = try self.sanitizeIdentifier(def.name);
                defer self.allocator.*.free(sanitized_field);

                if (def.var_type == .function) {
                    // Function pointer: return_type (*name)(param_types...)
                    const fn_type = def.var_type.function;
                    const return_type_str = self.cTypeFor(fn_type.return_type, &includes) catch |err| {
                        if (err == Error.UnsupportedType) {
                            try ns_writer.print("    // unsupported function type for: {s}\n", .{def.name});
                            continue;
                        }
                        return err;
                    };
                    try ns_writer.print("    {s} (*{s})(", .{ return_type_str, sanitized_field });
                    for (fn_type.param_types, 0..) |param_type, i| {
                        if (i > 0) try ns_writer.print(", ", .{});
                        const param_type_str = self.cTypeFor(param_type, &includes) catch |err| {
                            if (err == Error.UnsupportedType) {
                                try ns_writer.print("/* unsupported */", .{});
                                continue;
                            }
                            return err;
                        };
                        try ns_writer.print("{s}", .{param_type_str});
                    }
                    try ns_writer.print(");\n", .{});
                } else {
                    // Regular variable
                    const c_type = self.cTypeFor(def.var_type, &includes) catch |err| {
                        if (err == Error.UnsupportedType) {
                            try ns_writer.print("    // unsupported type for: {s}\n", .{def.name});
                            continue;
                        }
                        return err;
                    };
                    try ns_writer.print("    {s} {s};\n", .{ c_type, sanitized_field });
                }
            }
            try ns_writer.print("}} Namespace_{s};\n\n", .{sanitized_ns});

            // Generate global namespace instance
            try ns_writer.print("Namespace_{s} g_{s};\n\n", .{ sanitized_ns, sanitized_ns });

            // Generate forward declarations for static functions
            for (namespace_defs.items) |def| {
                if (def.var_type == .function) {
                    const sanitized_field = try self.sanitizeIdentifier(def.name);
                    defer self.allocator.*.free(sanitized_field);

                    const fn_type = def.var_type.function;
                    const return_type_str = self.cTypeFor(fn_type.return_type, &includes) catch continue;

                    try ns_writer.print("static {s} {s}(", .{ return_type_str, sanitized_field });
                    for (fn_type.param_types, 0..) |param_type, i| {
                        if (i > 0) try ns_writer.print(", ", .{});
                        const param_type_str = self.cTypeFor(param_type, &includes) catch continue;
                        try ns_writer.print("{s}", .{param_type_str});
                    }
                    try ns_writer.print(");\n", .{});
                }
            }
            try ns_writer.print("\n", .{});

            // Build def names set for init function context
            var init_def_names = std.StringHashMap(void).init(self.allocator.*);
            defer init_def_names.deinit();
            for (namespace_defs.items) |def| {
                try init_def_names.put(def.name, {});
            }

            // Create init function context (can reference namespace vars as ns->field)
            var init_ctx = NamespaceContext{
                .name = ns_name,
                .def_names = &init_def_names,
                .in_init_function = true,
            };

            // Generate init function signature
            try init_writer.print("void init_namespace_{s}(Namespace_{s}* ns) {{\n", .{ sanitized_ns, sanitized_ns });

            // Emit initialization for each def
            for (namespace_defs.items) |def| {
                const sanitized_field = try self.sanitizeIdentifier(def.name);
                defer self.allocator.*.free(sanitized_field);

                if (def.var_type == .function) {
                    // For functions, just assign the address of the static function
                    try init_writer.print("    ns->{s} = &{s};\n", .{ sanitized_field, sanitized_field });
                } else {
                    // For regular variables, evaluate the expression
                    try init_writer.print("    ns->{s} = ", .{sanitized_field});

                    // Get the value expression from the def
                    var iter = def.expr.list.iterator();
                    _ = iter.next(); // skip 'def'
                    _ = iter.next(); // skip name

                    // Skip type annotation if present
                    var maybe_value = iter.next();
                    if (maybe_value) |val| {
                        if (val.isList()) {
                            var val_iter = val.list.iterator();
                            if (val_iter.next()) |first| {
                                // Check if this is a type annotation: (: ...)
                                // The keyword : is stored with empty name
                                if (first.isKeyword() and first.keyword.len == 0) {
                                    maybe_value = iter.next();
                                }
                            }
                        }
                    }

                    if (maybe_value) |value_expr| {
                        // Try writeExpressionTyped first (handles structs), fall back to writeExpression
                        self.writeExpressionTyped(init_writer, def.typed, &init_ctx) catch |err_typed| {
                            if (err_typed == Error.UnsupportedExpression) {
                                self.writeExpression(init_writer, value_expr, &init_ctx) catch |err| {
                                    switch (err) {
                                        Error.UnsupportedExpression, Error.MissingOperand, Error.InvalidIfForm => {
                                            try init_writer.print("0; // unsupported expression\n", .{});
                                            continue;
                                        },
                                        else => return err,
                                    }
                                };
                            } else {
                                return err_typed;
                            }
                        };
                    }
                    try init_writer.print(";\n", .{});
                }
            }

            try init_writer.print("}}\n\n", .{});

            // Emit static function definitions
            for (namespace_defs.items) |def| {
                if (def.var_type == .function) {
                    const sanitized_name = try self.sanitizeIdentifier(def.name);
                    defer self.allocator.*.free(sanitized_name);

                    // Get the function value expression
                    var def_iter = def.expr.list.iterator();
                    _ = def_iter.next(); // skip 'def'
                    _ = def_iter.next(); // skip name

                    // Skip type annotation if present
                    var maybe_fn_expr = def_iter.next();
                    if (maybe_fn_expr) |val| {
                        if (val.isList()) {
                            var val_iter = val.list.iterator();
                            if (val_iter.next()) |first| {
                                if (first.isKeyword() and first.keyword.len == 0) {
                                    maybe_fn_expr = def_iter.next();
                                }
                            }
                        }
                    }

                    if (maybe_fn_expr) |fn_expr| {
                        if (fn_expr.isList()) {
                            var fn_iter = fn_expr.list.iterator();
                            const maybe_fn = fn_iter.next();
                            if (maybe_fn) |fn_sym| {
                                if (fn_sym.isSymbol() and std.mem.eql(u8, fn_sym.symbol, "fn")) {
                                    // Emit the function
                                    const fn_type = def.var_type.function;
                                    const return_type_str = self.cTypeFor(fn_type.return_type, &includes) catch continue;

                                    try init_writer.print("static {s} {s}(", .{ return_type_str, sanitized_name });

                                    const params_val = fn_iter.next() orelse continue;
                                    if (!params_val.isVector()) continue;
                                    const params_vec = params_val.vector;

                                    for (fn_type.param_types, 0..) |param_type, i| {
                                        if (i > 0) try init_writer.print(", ", .{});
                                        const param_type_str = self.cTypeFor(param_type, &includes) catch continue;
                                        const param_val = params_vec.at(i);
                                        const sanitized_param = try self.sanitizeIdentifier(param_val.symbol);
                                        defer self.allocator.*.free(sanitized_param);
                                        try init_writer.print("{s} {s}", .{ param_type_str, sanitized_param });
                                    }

                                    try init_writer.print(") {{\n    return ", .{});

                                    // Get the body expression
                                    const fn_body = fn_iter.next() orelse continue;

                                    // Create empty context for function bodies
                                    // Functions call each other directly, not through namespace
                                    var fn_def_names = std.StringHashMap(void).init(self.allocator.*);
                                    defer fn_def_names.deinit();
                                    var fn_ctx = NamespaceContext{
                                        .name = null,
                                        .def_names = &fn_def_names,
                                    };

                                    // Write function body
                                    self.writeExpression(init_writer, fn_body, &fn_ctx) catch {
                                        try init_writer.print("0", .{});
                                    };

                                    try init_writer.print(";\n}}\n", .{});
                                }
                            }
                        }
                    }
                }
            }
        }

        // Build a set of namespace def names for quick lookup
        var namespace_def_names = std.StringHashMap(void).init(self.allocator.*);
        defer namespace_def_names.deinit();
        for (namespace_defs.items) |def| {
            try namespace_def_names.put(def.name, {});
        }

        // Create namespace context (use "user" as default if no namespace declared)
        const ns_name = namespace_name orelse "user";
        var ns_ctx = NamespaceContext{
            .name = ns_name,
            .def_names = &namespace_def_names,
        };

        // Second pass: emit full definitions and expressions (skip namespace defs)
        for (expressions.items, 0..) |expr, idx| {
            const typed_val = report.typed.items[idx];
            try self.emitTopLevel(prelude_writer, body_writer, expr, typed_val, &checker, &includes, &ns_ctx);
        }

        var output = std.ArrayList(u8){};
        defer output.deinit(self.allocator.*);

        try output.appendSlice(self.allocator.*, "#include <stdio.h>\n");
        if (includes.need_stdbool) {
            try output.appendSlice(self.allocator.*, "#include <stdbool.h>\n");
        }
        if (includes.need_stdint) {
            try output.appendSlice(self.allocator.*, "#include <stdint.h>\n");
        }
        if (includes.need_stddef) {
            try output.appendSlice(self.allocator.*, "#include <stddef.h>\n");
        }
        try output.appendSlice(self.allocator.*, "\n");

        if (forward_decls.items.len > 0) {
            try output.appendSlice(self.allocator.*, forward_decls.items);
            try output.appendSlice(self.allocator.*, "\n");
        }

        // Add namespace struct and global instance
        if (namespace_struct.items.len > 0) {
            try output.appendSlice(self.allocator.*, namespace_struct.items);
        }

        if (prelude.items.len > 0) {
            try output.appendSlice(self.allocator.*, prelude.items);
            try output.appendSlice(self.allocator.*, "\n");
        }

        // Add namespace init function
        if (namespace_init.items.len > 0) {
            try output.appendSlice(self.allocator.*, namespace_init.items);
        }

        switch (target) {
            .executable => {
                try output.appendSlice(self.allocator.*, "int main() {\n");
                // Initialize namespace if present
                if (namespace_defs.items.len > 0) {
                    const sanitized_ns = try self.sanitizeIdentifier(ns_name);
                    defer self.allocator.*.free(sanitized_ns);
                    try output.print(self.allocator.*, "    init_namespace_{s}(&g_{s});\n", .{ sanitized_ns, sanitized_ns });
                }
                if (body.items.len > 0) {
                    try output.appendSlice(self.allocator.*, body.items);
                }
                try output.appendSlice(self.allocator.*, "    return 0;\n}\n");
            },
            .bundle => {
                try output.appendSlice(self.allocator.*, "void lisp_main(void) {\n");
                // Initialize namespace if present
                if (namespace_defs.items.len > 0) {
                    const sanitized_ns = try self.sanitizeIdentifier(ns_name);
                    defer self.allocator.*.free(sanitized_ns);
                    try output.print(self.allocator.*, "    init_namespace_{s}(&g_{s});\n", .{ sanitized_ns, sanitized_ns });
                }
                if (body.items.len > 0) {
                    try output.appendSlice(self.allocator.*, body.items);
                }
                try output.appendSlice(self.allocator.*, "}\n");
            },
        }

        return output.toOwnedSlice(self.allocator.*);
    }

    fn emitForwardDecl(self: *SimpleCCompiler, forward_writer: anytype, expr: *Value, typed: *TypedValue, checker: *TypeChecker, includes: *IncludeFlags) Error!void {
        _ = typed;
        switch (expr.*) {
            .list => |list| {
                var iter = list.iterator();
                const head_val = iter.next() orelse return;
                if (!head_val.isSymbol()) return;

                const head = head_val.symbol;
                if (std.mem.eql(u8, head, "def")) {
                    const name_val = iter.next() orelse return;
                    if (!name_val.isSymbol()) return;
                    const name = name_val.symbol;

                    // Check if this is a struct definition by looking it up in type_defs
                    if (checker.type_defs.get(name)) |type_def| {
                        if (type_def == .struct_type) {
                            // This is a struct definition - emit struct declaration
                            const sanitized_name = try self.sanitizeIdentifier(name);
                            defer self.allocator.*.free(sanitized_name);

                            try forward_writer.print("typedef struct {{\n", .{});
                            for (type_def.struct_type.fields) |field| {
                                const field_type_str = self.cTypeFor(field.field_type, includes) catch {
                                    try forward_writer.print("    // unsupported field type: {s}\n", .{field.name});
                                    continue;
                                };
                                const sanitized_field = try self.sanitizeIdentifier(field.name);
                                defer self.allocator.*.free(sanitized_field);
                                try forward_writer.print("    {s} {s};\n", .{ field_type_str, sanitized_field });
                            }
                            try forward_writer.print("}} {s};\n\n", .{sanitized_name});
                            return;
                        }
                    }

                    // Check if this is a function definition

                    // Skip the type annotation if present
                    var maybe_value = iter.next();
                    if (maybe_value) |val| {
                        if (val.isList()) {
                            var val_iter = val.list.iterator();
                            if (val_iter.next()) |first| {
                                if (first.isSymbol() and std.mem.eql(u8, first.symbol, ":")) {
                                    maybe_value = iter.next();
                                }
                            }
                        }
                    }

                    if (maybe_value) |value_expr| {
                        if (value_expr.isList()) {
                            var fn_iter = value_expr.list.iterator();
                            const maybe_fn = fn_iter.next() orelse return;
                            if (maybe_fn.isSymbol() and std.mem.eql(u8, maybe_fn.symbol, "fn")) {
                                // This is a function definition - emit forward declaration
                                const var_type = checker.env.get(name) orelse return;
                                if (var_type != .function) return;

                                const return_type_str = self.cTypeFor(var_type.function.return_type, includes) catch return;
                                const sanitized_name = try self.sanitizeIdentifier(name);
                                defer self.allocator.*.free(sanitized_name);

                                try forward_writer.print("static {s} {s}(", .{ return_type_str, sanitized_name });

                                const params_val = fn_iter.next() orelse return;
                                if (!params_val.isVector()) return;
                                const params_vec = params_val.vector;

                                for (var_type.function.param_types, 0..) |param_type, i| {
                                    if (i > 0) try forward_writer.print(", ", .{});
                                    const param_type_str = self.cTypeFor(param_type, includes) catch return;
                                    const param_val = params_vec.at(i);
                                    const sanitized_param = try self.sanitizeIdentifier(param_val.symbol);
                                    defer self.allocator.*.free(sanitized_param);
                                    try forward_writer.print("{s} {s}", .{ param_type_str, sanitized_param });
                                }

                                try forward_writer.print(");\n", .{});
                            }
                        }
                    }
                }
            },
            else => {},
        }
    }

    fn emitTopLevel(self: *SimpleCCompiler, def_writer: anytype, body_writer: anytype, expr: *Value, typed: *TypedValue, checker: *TypeChecker, includes: *IncludeFlags, ns_ctx: *NamespaceContext) Error!void {
        switch (expr.*) {
            .namespace => |ns| {
                try body_writer.print("    // namespace {s}\n", .{ns.name});
            },
            .list => |list| {
                var iter = list.iterator();
                const head_val = iter.next() orelse {
                    try self.emitPrintStatement(body_writer, expr, typed, includes, ns_ctx);
                    return;
                };

                if (!head_val.isSymbol()) {
                    try self.emitPrintStatement(body_writer, expr, typed, includes, ns_ctx);
                    return;
                }

                const head = head_val.symbol;
                if (std.mem.eql(u8, head, "def")) {
                    // Check if this def is in the namespace (skip if it is)
                    if (iter.next()) |name_val| {
                        if (name_val.isSymbol()) {
                            if (ns_ctx.def_names.contains(name_val.symbol)) {
                                // Skip - it's handled in namespace init
                                return;
                            }
                        }
                    }
                    // Reset iterator and emit as normal
                    iter = list.iterator();
                    _ = iter.next(); // skip 'def' again
                    try self.emitDefinition(def_writer, expr, typed, checker, includes);
                    return;
                }

                try self.emitPrintStatement(body_writer, expr, typed, includes, ns_ctx);
            },
            else => {
                try self.emitPrintStatement(body_writer, expr, typed, includes, ns_ctx);
            },
        }
    }

    fn emitDefinition(self: *SimpleCCompiler, def_writer: anytype, list_expr: *Value, typed: *TypedValue, checker: *TypeChecker, includes: *IncludeFlags) Error!void {
        // Create empty context (definitions shouldn't reference namespace vars from the def line itself)
        var empty_def_names = std.StringHashMap(void).init(self.allocator.*);
        defer empty_def_names.deinit();
        var empty_ctx = NamespaceContext{
            .name = null,
            .def_names = &empty_def_names,
        };

        var iter = list_expr.list.iterator();
        _ = iter.next(); // Skip 'def'

        const name_val = iter.next() orelse return Error.InvalidDefinition;
        if (!name_val.isSymbol()) return Error.InvalidDefinition;
        const name = name_val.symbol;

        // Skip type definitions (struct/enum) - they're already declared in forward_decls
        if (checker.type_defs.get(name)) |_| {
            return;
        }

        var values_buf: [8]*Value = undefined;
        var value_count: usize = 0;
        while (iter.next()) |val| {
            if (value_count >= values_buf.len) {
                const repr = try self.formatValue(list_expr);
                defer self.allocator.*.free(repr);
                try def_writer.print("// unsupported definition: {s}\n", .{repr});
                return;
            }
            values_buf[value_count] = val;
            value_count += 1;
        }

        const value_expr = switch (value_count) {
            1 => values_buf[0],
            2 => values_buf[1],
            else => {
                const repr = try self.formatValue(list_expr);
                defer self.allocator.*.free(repr);
                try def_writer.print("// unsupported definition: {s}\n", .{repr});
                return;
            },
        };
        const var_type = checker.env.get(name) orelse {
            const repr = try self.formatValue(list_expr);
            defer self.allocator.*.free(repr);
            try def_writer.print("// unknown type for definition: {s}\n", .{repr});
            return;
        };

        if (value_expr.isList()) {
            var fn_iter = value_expr.list.iterator();
            const maybe_fn = fn_iter.next() orelse return Error.InvalidFunction;
            if (maybe_fn.isSymbol() and std.mem.eql(u8, maybe_fn.symbol, "fn")) {
                if (var_type != .function) {
                    return Error.UnsupportedType;
                }
                try self.emitFunctionDefinition(def_writer, name, value_expr, var_type, includes);
                return;
            }
        }

        const c_type = self.cTypeFor(var_type, includes) catch |err| {
            if (err == Error.UnsupportedType) {
                const repr = try self.formatValue(list_expr);
                defer self.allocator.*.free(repr);
                try def_writer.print("// unsupported definition: {s}\n", .{repr});
                return;
            }
            return err;
        };

        const sanitized_var_name = try self.sanitizeIdentifier(name);
        defer self.allocator.*.free(sanitized_var_name);
        try def_writer.print("{s} {s} = ", .{ c_type, sanitized_var_name });

        // Use writeExpressionTyped if we have the typed value, otherwise fall back to writeExpression
        self.writeExpressionTyped(def_writer, typed, &empty_ctx) catch |err| {
            switch (err) {
                Error.UnsupportedExpression, Error.MissingOperand, Error.InvalidIfForm => {
                    // Fall back to untyped version
                    self.writeExpression(def_writer, value_expr, &empty_ctx) catch |err2| {
                        switch (err2) {
                            Error.UnsupportedExpression, Error.MissingOperand, Error.InvalidIfForm => {
                                const repr = try self.formatValue(list_expr);
                                defer self.allocator.*.free(repr);
                                try def_writer.print("0; // unsupported definition: {s}\n", .{repr});
                                return;
                            },
                            else => return err2,
                        }
                    };
                    try def_writer.print(";\n", .{});
                    return;
                },
                else => return err,
            }
        };
        try def_writer.print(";\n", .{});
    }

    fn emitFunctionDefinition(self: *SimpleCCompiler, def_writer: anytype, name: []const u8, fn_expr: *Value, fn_type: Type, includes: *IncludeFlags) Error!void {
        if (fn_type != .function) return Error.UnsupportedType;

        // Create empty context (function bodies shouldn't reference namespace vars for now - that needs more work)
        var empty_def_names = std.StringHashMap(void).init(self.allocator.*);
        defer empty_def_names.deinit();
        var empty_ctx = NamespaceContext{
            .name = null,
            .def_names = &empty_def_names,
        };

        const param_types = fn_type.function.param_types;
        var fn_iter = fn_expr.list.iterator();
        _ = fn_iter.next(); // Skip 'fn'
        const params_val = fn_iter.next() orelse return Error.InvalidFunction;
        if (!params_val.isVector()) return Error.InvalidFunction;
        const params_vec = params_val.vector;

        if (param_types.len != params_vec.len()) {
            return Error.InvalidFunction;
        }

        const body_expr = fn_iter.next() orelse return Error.InvalidFunction;
        if (fn_iter.next() != null) return Error.InvalidFunction;

        const return_type_str = self.cTypeFor(fn_type.function.return_type, includes) catch |err| {
            if (err == Error.UnsupportedType) {
                const repr = try self.formatValue(fn_expr);
                defer self.allocator.*.free(repr);
                try def_writer.print("// unsupported function: {s}\n", .{repr});
                return;
            }
            return err;
        };

        if (params_vec.len() > 32) {
            const repr = try self.formatValue(fn_expr);
            defer self.allocator.*.free(repr);
            try def_writer.print("// unsupported function (too many parameters): {s}\n", .{repr});
            return;
        }

        var param_type_buf: [32][]const u8 = undefined;
        var index: usize = 0;
        while (index < params_vec.len()) : (index += 1) {
            const param_val = params_vec.at(index);
            if (!param_val.isSymbol()) return Error.InvalidFunction;
            const param_type_str = self.cTypeFor(param_types[index], includes) catch |err| {
                if (err == Error.UnsupportedType) {
                    const repr = try self.formatValue(fn_expr);
                    defer self.allocator.*.free(repr);
                    try def_writer.print("// unsupported function: {s}\n", .{repr});
                    return;
                }
                return err;
            };
            param_type_buf[index] = param_type_str;
        }

        const sanitized_name = try self.sanitizeIdentifier(name);
        defer self.allocator.*.free(sanitized_name);
        try def_writer.print("static {s} {s}(", .{ return_type_str, sanitized_name });
        index = 0;
        while (index < params_vec.len()) : (index += 1) {
            const param_val = params_vec.at(index);
            if (index > 0) {
                try def_writer.print(", ", .{});
            }
            const sanitized_param = try self.sanitizeIdentifier(param_val.symbol);
            defer self.allocator.*.free(sanitized_param);
            try def_writer.print("{s} {s}", .{ param_type_buf[index], sanitized_param });
        }
        try def_writer.writeAll(") {\n");
        try def_writer.print("    return ", .{});
        self.writeExpression(def_writer, body_expr, &empty_ctx) catch |err| {
            switch (err) {
                Error.UnsupportedExpression, Error.MissingOperand, Error.InvalidIfForm => {
                    const repr = try self.formatValue(fn_expr);
                    defer self.allocator.*.free(repr);
                    try def_writer.writeAll("0;\n}\n");
                    try def_writer.print("// unsupported function body: {s}\n", .{repr});
                    return;
                },
                else => return err,
            }
        };
        try def_writer.writeAll(";\n}\n");
    }

    fn sanitizeIdentifier(self: *SimpleCCompiler, name: []const u8) ![]u8 {
        // Check if it's a C keyword
        const c_keywords = [_][]const u8{
            "auto",     "break",    "case",     "char",     "const",    "continue",
            "default",  "do",       "double",   "else",     "enum",     "extern",
            "float",    "for",      "goto",     "if",       "inline",   "int",
            "long",     "register", "restrict", "return",   "short",    "signed",
            "sizeof",   "static",   "struct",   "switch",   "typedef",  "union",
            "unsigned", "void",     "volatile", "while",    "_Bool",    "_Complex",
            "_Imaginary",
        };

        var is_keyword = false;
        for (c_keywords) |kw| {
            if (std.mem.eql(u8, name, kw)) {
                is_keyword = true;
                break;
            }
        }

        if (is_keyword) {
            // Prefix with _ to avoid keyword collision
            var result = try self.allocator.*.alloc(u8, name.len + 1);
            result[0] = '_';
            for (name, 0..) |c, i| {
                result[i + 1] = if (c == '-' or c == '.') '_' else c;
            }
            return result;
        } else {
            var result = try self.allocator.*.alloc(u8, name.len);
            for (name, 0..) |c, i| {
                result[i] = if (c == '-' or c == '.') '_' else c;
            }
            return result;
        }
    }

    fn emitPrintStatement(self: *SimpleCCompiler, body_writer: anytype, expr: *Value, typed: *TypedValue, includes: *IncludeFlags, ns_ctx: *NamespaceContext) Error!void {
        const expr_type = typed.getType();
        const format = self.printfFormatFor(expr_type) catch |err| {
            if (err == Error.UnsupportedType) {
                const repr = try self.formatValue(expr);
                defer self.allocator.*.free(repr);
                try body_writer.print("    // unsupported expression: {s}\n", .{repr});
                return;
            }
            return err;
        };

        try body_writer.print("    printf(\"{s}\\n\", ", .{format});
        const wrap_bool = expr_type == .bool;
        if (wrap_bool) {
            includes.need_stdbool = true;
            try body_writer.writeAll("((");
        }

        // Try writeExpressionTyped first, fall back to writeExpression
        self.writeExpressionTyped(body_writer, typed, ns_ctx) catch |err_typed| {
            if (err_typed == Error.UnsupportedExpression) {
                self.writeExpression(body_writer, expr, ns_ctx) catch |err| {
                    switch (err) {
                        Error.UnsupportedExpression, Error.MissingOperand, Error.InvalidIfForm => {
                            const repr = try self.formatValue(expr);
                            defer self.allocator.*.free(repr);
                            try body_writer.print("0); // unsupported expression\n", .{});
                            try body_writer.print("    // {s}\n", .{repr});
                            return;
                        },
                        else => return err,
                    }
                };
            } else {
                return err_typed;
            }
        };
        if (wrap_bool) {
            try body_writer.writeAll(") ? 1 : 0)");
        }
        try body_writer.print(");\n", .{});
    }

    fn writeExpressionTyped(self: *SimpleCCompiler, writer: anytype, typed: *TypedValue, ns_ctx: *NamespaceContext) Error!void {
        switch (typed.*) {
            .struct_instance => |si| {
                // Emit C99 compound literal: (TypeName){field1, field2, ...}
                const struct_type = si.type.struct_type;
                const sanitized_name = try self.sanitizeIdentifier(struct_type.name);
                defer self.allocator.*.free(sanitized_name);

                try writer.print("({s}){{", .{sanitized_name});
                for (si.field_values, 0..) |field_val, i| {
                    if (i > 0) try writer.print(", ", .{});
                    try self.writeExpressionTyped(writer, field_val, ns_ctx);
                }
                try writer.print("}}", .{});
            },
            .list => |l| {
                if (l.elements.len > 0) {
                    const first = l.elements[0];
                    // Check for field access
                    if (first.* == .symbol and std.mem.eql(u8, first.symbol.name, ".")) {
                        if (l.elements.len == 3) {
                            try self.writeExpressionTyped(writer, l.elements[1], ns_ctx);
                            if (l.elements[2].* == .symbol) {
                                try writer.print(".{s}", .{l.elements[2].symbol.name});
                                return;
                            }
                        }
                    }
                }
                return Error.UnsupportedExpression;
            },
            .int => |i| try writer.print("{d}", .{i.value}),
            .float => |f| try writer.print("{d}", .{f.value}),
            .string => |s| try writer.print("\"{s}\"", .{s.value}),
            .symbol => |sym| {
                // Check if this symbol is in the namespace
                if (ns_ctx.def_names.contains(sym.name)) {
                    if (ns_ctx.name) |_| {
                        const sanitized_field = try self.sanitizeIdentifier(sym.name);
                        defer self.allocator.*.free(sanitized_field);

                        if (ns_ctx.in_init_function) {
                            // In init function, use ns->field
                            try writer.print("ns->{s}", .{sanitized_field});
                        } else {
                            // In regular code, use g_namespace.field
                            const sanitized_ns = try self.sanitizeIdentifier(ns_ctx.name.?);
                            defer self.allocator.*.free(sanitized_ns);
                            try writer.print("g_{s}.{s}", .{ sanitized_ns, sanitized_field });
                        }
                        return;
                    }
                }
                const sanitized = try self.sanitizeIdentifier(sym.name);
                defer self.allocator.*.free(sanitized);
                try writer.print("{s}", .{sanitized});
            },
            .nil => try writer.print("0", .{}),
            else => return Error.UnsupportedExpression,
        }
    }

    fn writeExpression(self: *SimpleCCompiler, writer: anytype, expr: *Value, ns_ctx: *NamespaceContext) Error!void {
        switch (expr.*) {
            .int => |i| try writer.print("{d}", .{i}),
            .float => |f| try writer.print("{d}", .{f}),
            .string => |s| try writer.print("\"{s}\"", .{s}),
            .symbol => |sym| {
                // Check if this symbol is in the namespace
                if (ns_ctx.def_names.contains(sym)) {
                    if (ns_ctx.name) |_| {
                        const sanitized_field = try self.sanitizeIdentifier(sym);
                        defer self.allocator.*.free(sanitized_field);

                        if (ns_ctx.in_init_function) {
                            // In init function, use ns->field
                            try writer.print("ns->{s}", .{sanitized_field});
                        } else {
                            // In regular code, use g_namespace.field
                            const sanitized_ns = try self.sanitizeIdentifier(ns_ctx.name.?);
                            defer self.allocator.*.free(sanitized_ns);
                            try writer.print("g_{s}.{s}", .{ sanitized_ns, sanitized_field });
                        }
                        return;
                    }
                }
                const sanitized = try self.sanitizeIdentifier(sym);
                defer self.allocator.*.free(sanitized);
                try writer.print("{s}", .{sanitized});
            },
            .keyword => |kw| try writer.print(":{s}", .{kw}),
            .nil => try writer.print("0", .{}),
            .namespace => |ns| try writer.print("/* namespace {s} */", .{ns.name}),
            .vector => {
                return Error.UnsupportedExpression;
            },
            .map => {
                return Error.UnsupportedExpression;
            },
            .list => |list| {
                var iter = list.iterator();
                const head_val = iter.next() orelse return Error.UnsupportedExpression;
                if (!head_val.isSymbol()) return Error.UnsupportedExpression;
                const op = head_val.symbol;

                if (std.mem.eql(u8, op, "if")) {
                    const condition = iter.next() orelse return Error.InvalidIfForm;
                    const then_expr = iter.next() orelse return Error.InvalidIfForm;
                    const else_expr = iter.next() orelse return Error.InvalidIfForm;
                    if (iter.next() != null) return Error.InvalidIfForm;

                    try writer.print("((", .{});
                    try self.writeExpression(writer, condition, ns_ctx);
                    try writer.print(") ? (", .{});
                    try self.writeExpression(writer, then_expr, ns_ctx);
                    try writer.print(") : (", .{});
                    try self.writeExpression(writer, else_expr, ns_ctx);
                    try writer.print("))", .{});
                    return;
                }

                if (self.isComparisonOperator(op)) {
                    const left = iter.next() orelse return Error.MissingOperand;
                    const right = iter.next() orelse return Error.MissingOperand;
                    if (iter.next() != null) return Error.UnsupportedExpression;

                    try writer.print("(", .{});
                    try self.writeExpression(writer, left, ns_ctx);
                    try writer.print(" {s} ", .{op});
                    try self.writeExpression(writer, right, ns_ctx);
                    try writer.print(")", .{});
                    return;
                }

                if (self.isArithmeticOperator(op)) {
                    var operands: [64]*Value = undefined;
                    var count: usize = 0;
                    while (iter.next()) |operand| {
                        if (count >= operands.len) return Error.UnsupportedExpression;
                        operands[count] = operand;
                        count += 1;
                    }

                    if (count == 0) return Error.MissingOperand;

                    if (std.mem.eql(u8, op, "-") and count == 1) {
                        try writer.writeAll("(-(");
                        try self.writeExpression(writer, operands[0], ns_ctx);
                        try writer.writeAll("))");
                        return;
                    }

                    if (count == 1) return Error.MissingOperand;

                    try writer.writeAll("(");
                    try self.writeExpression(writer, operands[0], ns_ctx);

                    var idx: usize = 1;
                    while (idx < count) : (idx += 1) {
                        try writer.print(" {s} ", .{op});
                        try self.writeExpression(writer, operands[idx], ns_ctx);
                    }

                    try writer.writeAll(")");
                    return;
                }

                // Field access: (. struct-expr field-name)
                if (std.mem.eql(u8, op, ".")) {
                    const struct_arg = iter.next() orelse return Error.MissingOperand;
                    const field_arg = iter.next() orelse return Error.MissingOperand;
                    if (iter.next() != null) return Error.UnsupportedExpression;

                    if (!field_arg.isSymbol()) return Error.UnsupportedExpression;

                    try self.writeExpression(writer, struct_arg, ns_ctx);
                    try writer.print(".{s}", .{field_arg.symbol});
                    return;
                }

                // Check if this is struct construction by looking up the op in type_defs
                // We need the checker for this, but writeExpression doesn't have it
                // For now, just emit as function call and we'll fix this separately

                const sanitized_op = try self.sanitizeIdentifier(op);
                defer self.allocator.*.free(sanitized_op);

                // Check if this is a namespace function
                if (ns_ctx.def_names.contains(op) and ns_ctx.name != null) {
                    const sanitized_ns = try self.sanitizeIdentifier(ns_ctx.name.?);
                    defer self.allocator.*.free(sanitized_ns);
                    try writer.print("g_{s}.{s}(", .{ sanitized_ns, sanitized_op });
                } else {
                    try writer.print("{s}(", .{sanitized_op});
                }
                var is_first = true;
                while (iter.next()) |arg| {
                    if (!is_first) {
                        try writer.print(", ", .{});
                    }
                    try self.writeExpression(writer, arg, ns_ctx);
                    is_first = false;
                }
                try writer.writeAll(")");
            },
        }
    }

    fn isArithmeticOperator(_: *SimpleCCompiler, symbol: []const u8) bool {
        return std.mem.eql(u8, symbol, "+") or
            std.mem.eql(u8, symbol, "-") or
            std.mem.eql(u8, symbol, "*") or
            std.mem.eql(u8, symbol, "/") or
            std.mem.eql(u8, symbol, "%");
    }

    fn isComparisonOperator(_: *SimpleCCompiler, symbol: []const u8) bool {
        return std.mem.eql(u8, symbol, "<") or
            std.mem.eql(u8, symbol, ">") or
            std.mem.eql(u8, symbol, "<=") or
            std.mem.eql(u8, symbol, ">=") or
            std.mem.eql(u8, symbol, "==") or
            std.mem.eql(u8, symbol, "!=");
    }

    fn cTypeFor(self: *SimpleCCompiler, type_info: Type, includes: *IncludeFlags) Error![]const u8 {
        _ = self;
        return switch (type_info) {
            .int => int_type_name,
            .float => "double",
            .string => "const char *",
            .bool => blk: {
                includes.need_stdbool = true;
                break :blk "bool";
            },
            .u8 => blk: {
                includes.need_stdint = true;
                break :blk "uint8_t";
            },
            .u16 => blk: {
                includes.need_stdint = true;
                break :blk "uint16_t";
            },
            .u32 => blk: {
                includes.need_stdint = true;
                break :blk "uint32_t";
            },
            .u64 => blk: {
                includes.need_stdint = true;
                break :blk "uint64_t";
            },
            .usize => blk: {
                includes.need_stddef = true;
                break :blk "size_t";
            },
            .i8 => blk: {
                includes.need_stdint = true;
                break :blk "int8_t";
            },
            .i16 => blk: {
                includes.need_stdint = true;
                break :blk "int16_t";
            },
            .i32 => blk: {
                includes.need_stdint = true;
                break :blk "int32_t";
            },
            .i64 => blk: {
                includes.need_stdint = true;
                break :blk "int64_t";
            },
            .isize => int_type_name,
            .f32 => "float",
            .f64 => "double",
            .struct_type => |st| st.name,
            else => Error.UnsupportedType,
        };
    }

    fn printfFormatFor(self: *SimpleCCompiler, type_info: Type) Error![]const u8 {
        _ = self;
        return switch (type_info) {
            .int, .i64 => int_printf_format,
            .i32, .i16, .i8 => "%d",
            .float, .f64, .f32 => "%f",
            .u8, .u16, .u32 => "%u",
            .u64 => "%llu",
            .usize => "%zu",
            .bool => "%d",
            .string => "%s",
            else => Error.UnsupportedType,
        };
    }

    fn formatValue(self: *SimpleCCompiler, expr: *Value) ![]u8 {
        var buf = std.ArrayList(u8){};
        defer buf.deinit(self.allocator.*);
        try expr.format("", .{}, buf.writer(self.allocator.*));
        return buf.toOwnedSlice(self.allocator.*);
    }
};

test "simple c compiler basic program" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();

    var allocator = arena.allocator();
    var compiler = SimpleCCompiler.init(&allocator);

    const source =
        "(ns my.app)\n" ++
        "(def answer (: Int) 41)\n" ++
        "(+ answer 1)";

    const output = try compiler.compileString(source, .executable);

    const expected =
        "#include <stdio.h>\n\n"
        ++ "long long answer = 41;\n\n"
        ++ "int main() {\n"
        ++ "    // namespace my.app\n"
        ++ "    printf(\"%lld\\n\", (answer + 1));\n"
        ++ "    return 0;\n"
        ++ "}\n";

    try std.testing.expectEqualStrings(expected, output);
}

test "simple c compiler fibonacci program with zig cc" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();

    var allocator = arena.allocator();
    var compiler = SimpleCCompiler.init(&allocator);

    const source =
        "(ns demo.core)\n"
        ++ "(def f0 (: Int) 0)\n"
        ++ "(def f1 (: Int) 1)\n"
        ++ "(def fib (: (-> [Int] Int)) (fn [n] (if (< n 2) n (+ (fib (- n 1)) (fib (- n 2))))))\n"
        ++ "(fib 10)";

    const c_source = try compiler.compileString(source, .executable);

    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();

    try tmp.dir.writeFile(.{ .sub_path = "main.c", .data = c_source });

    const exe_name = if (builtin.os.tag == .windows) "program.exe" else "program";

    var cc_child = std.process.Child.init(&.{ "zig", "cc", "main.c", "-o", exe_name }, std.testing.allocator);
    cc_child.cwd_dir = tmp.dir;
    try cc_child.spawn();
    const cc_term = try cc_child.wait();
    switch (cc_term) {
        .Exited => |code| try std.testing.expect(code == 0),
        else => return error.TestUnexpectedResult,
    }

    const exe_path = try tmp.dir.realpathAlloc(std.testing.allocator, exe_name);
    defer std.testing.allocator.free(exe_path);

    var run_child = std.process.Child.init(&.{ exe_path }, std.testing.allocator);
    run_child.stdout_behavior = .Pipe;
    try run_child.spawn();

    var stdout_file = run_child.stdout orelse return error.TestUnexpectedResult;
    const output = try stdout_file.readToEndAlloc(std.testing.allocator, 1024);
    defer std.testing.allocator.free(output);
    stdout_file.close();
    run_child.stdout = null;

    const run_term = try run_child.wait();
    switch (run_term) {
        .Exited => |code| try std.testing.expect(code == 0),
        else => return error.TestUnexpectedResult,
    }

    try std.testing.expectEqualStrings("55\n", output);
}

pub fn main() !void {
    var arena = std.heap.ArenaAllocator.init(std.heap.page_allocator);
    defer arena.deinit();
    var allocator = arena.allocator();

    var args = std.process.args();
    defer args.deinit();

    _ = args.next() orelse return;
    const source_path = args.next() orelse {
        std.debug.print("Usage: simple_c_compiler <source-file> [--run]\n", .{});
        return;
    };

    var run_flag = false;
    var bundle_flag = false;
    while (args.next()) |arg| {
        if (std.mem.eql(u8, arg, "--run")) {
            run_flag = true;
        } else if (std.mem.eql(u8, arg, "--bundle")) {
            bundle_flag = true;
        } else {
            std.debug.print("Unknown argument: {s}\n", .{arg});
            return;
        }
    }

    const target_kind: SimpleCCompiler.TargetKind = if (bundle_flag) .bundle else .executable;

    const source = try std.fs.cwd().readFileAlloc(allocator, source_path, std.math.maxInt(usize));
    defer allocator.free(source);

    var compiler = SimpleCCompiler.init(&allocator);
    const c_source = try compiler.compileString(source, target_kind);
    defer allocator.free(c_source);

    const basename = std.fs.path.basename(source_path);
    const stem = blk: {
        if (std.mem.lastIndexOfScalar(u8, basename, '.')) |idx| {
            break :blk basename[0..idx];
        } else {
            break :blk basename;
        }
    };

    const c_filename = try std.fmt.allocPrint(allocator, "{s}.c", .{stem});
    defer allocator.free(c_filename);

    const dir_opt = std.fs.path.dirname(source_path);
    if (dir_opt) |dir_path| {
        var dir = try std.fs.cwd().openDir(dir_path, .{});
        defer dir.close();
        try dir.writeFile(.{ .sub_path = c_filename, .data = c_source });
    } else {
        try std.fs.cwd().writeFile(.{ .sub_path = c_filename, .data = c_source });
    }

    const c_path = if (dir_opt) |dir_path|
        try std.fs.path.join(allocator, &.{ dir_path, c_filename })
    else
        try allocator.dupe(u8, c_filename);
    defer allocator.free(c_path);

    const c_real_path = try std.fs.cwd().realpathAlloc(allocator, c_path);
    defer allocator.free(c_real_path);

    std.debug.print("Generated C file: {s}\n", .{c_real_path});

    if (!run_flag) return;

    if (bundle_flag) {
        const bundle_name = try std.fmt.allocPrint(allocator, "{s}.bundle", .{stem});
        defer allocator.free(bundle_name);

        const bundle_path = if (dir_opt) |dir_path|
            try std.fs.path.join(allocator, &.{ dir_path, bundle_name })
        else
            try allocator.dupe(u8, bundle_name);
        defer allocator.free(bundle_path);

        var cc_child = std.process.Child.init(&.{ "zig", "cc", "-dynamiclib", c_path, "-o", bundle_path }, allocator);
        cc_child.stdin_behavior = .Inherit;
        cc_child.stdout_behavior = .Inherit;
        cc_child.stderr_behavior = .Inherit;
        try cc_child.spawn();
        const cc_term = try cc_child.wait();
        switch (cc_term) {
            .Exited => |code| {
                if (code != 0) {
                    std.debug.print("zig cc exited with code {d}\n", .{code});
                    return;
                }
            },
            else => {
                std.debug.print("zig cc failed to build the bundle\n", .{});
                return;
            },
        }

        const bundle_real_path = try std.fs.cwd().realpathAlloc(allocator, bundle_path);
        defer allocator.free(bundle_real_path);

        std.debug.print("Built bundle: {s}\n", .{bundle_real_path});

        if (!run_flag) return;

        var lib = try std.DynLib.open(bundle_real_path);
        defer lib.close();

        const entry_fn = lib.lookup(*const fn () callconv(.c) void, "lisp_main") orelse {
            std.debug.print("Bundle missing lisp_main entry\n", .{});
            return;
        };

        @call(.auto, entry_fn, .{});
        return;
    }

    const exe_name = if (builtin.os.tag == .windows)
        try std.fmt.allocPrint(allocator, "{s}.exe", .{stem})
    else
        try allocator.dupe(u8, stem);
    defer allocator.free(exe_name);

    const exe_path = if (dir_opt) |dir_path|
        try std.fs.path.join(allocator, &.{ dir_path, exe_name })
    else
        try allocator.dupe(u8, exe_name);
    defer allocator.free(exe_path);

    var cc_child = std.process.Child.init(&.{ "zig", "cc", c_path, "-o", exe_path }, allocator);
    cc_child.stdin_behavior = .Inherit;
    cc_child.stdout_behavior = .Inherit;
    cc_child.stderr_behavior = .Inherit;
    try cc_child.spawn();
    const cc_term = try cc_child.wait();
    switch (cc_term) {
        .Exited => |code| {
            if (code != 0) {
                std.debug.print("zig cc exited with code {d}\n", .{code});
                return;
            }
        },
        else => {
            std.debug.print("zig cc failed to build the program\n", .{});
            return;
        },
    }

    const exe_real_path = try std.fs.cwd().realpathAlloc(allocator, exe_path);
    defer allocator.free(exe_real_path);

    std.debug.print("Built executable: {s}\n", .{exe_real_path});

    if (!run_flag) return;

    var run_child = std.process.Child.init(&.{ exe_real_path }, allocator);
    run_child.stdin_behavior = .Inherit;
    run_child.stdout_behavior = .Inherit;
    run_child.stderr_behavior = .Inherit;
    try run_child.spawn();
    const run_term = try run_child.wait();
    switch (run_term) {
        .Exited => |code| {
            if (code != 0) {
                std.debug.print("Program exited with code {d}\n", .{code});
            }
        },
        else => std.debug.print("Program terminated abnormally\n", .{}),
    }
}
