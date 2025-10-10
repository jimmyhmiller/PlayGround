const std = @import("std");
const builtin = @import("builtin");
const reader = @import("reader.zig");
const value = @import("value.zig");
const type_checker = @import("type_checker.zig");
const macro_expander = @import("macro_expander.zig");

const Reader = reader.Reader;
const Value = value.Value;
const TypeChecker = type_checker.BidirectionalTypeChecker;
const Type = type_checker.Type;
const TypedValue = type_checker.TypedValue;
const MacroExpander = macro_expander.MacroExpander;

const IncludeFlags = struct {
    need_stdint: bool = false,
    need_stdbool: bool = false,
    need_stddef: bool = false,
    need_stdlib: bool = false,
    need_stdio: bool = false,
};

pub const SimpleCCompiler = struct {
    allocator: *std.mem.Allocator,
    linked_libraries: std.ArrayList([]const u8),
    include_paths: std.ArrayList([]const u8),
    compiler_flags: std.ArrayList([]const u8),

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

    fn parseSimpleType(self: *SimpleCCompiler, type_name: []const u8) Error!Type {
        _ = self;
        if (std.mem.eql(u8, type_name, "U8")) return Type.u8;
        if (std.mem.eql(u8, type_name, "U16")) return Type.u16;
        if (std.mem.eql(u8, type_name, "U32")) return Type.u32;
        if (std.mem.eql(u8, type_name, "U64")) return Type.u64;
        if (std.mem.eql(u8, type_name, "I8")) return Type.i8;
        if (std.mem.eql(u8, type_name, "I16")) return Type.i16;
        if (std.mem.eql(u8, type_name, "I32")) return Type.i32;
        if (std.mem.eql(u8, type_name, "I64")) return Type.i64;
        if (std.mem.eql(u8, type_name, "CString")) return Type.c_string;
        if (std.mem.eql(u8, type_name, "Void")) return Type.void;
        if (std.mem.eql(u8, type_name, "Int")) return Type.int;
        if (std.mem.eql(u8, type_name, "Float")) return Type.float;
        if (std.mem.eql(u8, type_name, "String")) return Type.string;
        if (std.mem.eql(u8, type_name, "Bool")) return Type.bool;
        return Error.UnsupportedType;
    }

    pub fn init(allocator: *std.mem.Allocator) SimpleCCompiler {
        return SimpleCCompiler{
            .allocator = allocator,
            .linked_libraries = std.ArrayList([]const u8){},
            .include_paths = std.ArrayList([]const u8){},
            .compiler_flags = std.ArrayList([]const u8){},
        };
    }

    pub fn deinit(self: *SimpleCCompiler) void {
        self.linked_libraries.deinit(self.allocator.*);
        self.include_paths.deinit(self.allocator.*);
        self.compiler_flags.deinit(self.allocator.*);
    }

    pub fn compileString(self: *SimpleCCompiler, source: []const u8, target: TargetKind) Error![]u8 {
        var reader_instance = Reader.init(self.allocator);
        const read_result = reader_instance.readAllString(source) catch |err| {
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
        var expressions = read_result.values;
        var line_numbers = read_result.line_numbers;
        defer expressions.deinit(self.allocator.*);
        defer line_numbers.deinit(self.allocator.*);

        // MACRO EXPANSION PHASE: Expand all macros before type checking
        var expander = MacroExpander.init(self.allocator.*);
        defer expander.deinit();

        var expanded_expressions = std.ArrayList(*Value){};
        defer expanded_expressions.deinit(self.allocator.*);

        for (expressions.items) |expr| {
            const expanded = expander.expand(expr) catch |err| {
                std.debug.print("Macro expansion error: {s}\n", .{@errorName(err)});
                return Error.UnsupportedExpression;
            };
            // Skip macro definitions - they shouldn't be type-checked or code-generated
            // They're only needed during the expansion phase
            if (!expanded.isMacro()) {
                try expanded_expressions.append(self.allocator.*, expanded);
            }
        }

        var checker = TypeChecker.init(self.allocator.*);
        defer checker.deinit();

        var report = try checker.typeCheckAllTwoPass(expanded_expressions.items);
        defer report.typed.deinit(self.allocator.*);
        defer report.errors.deinit(self.allocator.*);

        if (report.errors.items.len != 0 or report.typed.items.len != expanded_expressions.items.len) {
            if (report.errors.items.len == 0) {
                std.debug.print("Type check failed: expected {d} typed expressions, got {d}\n", .{ expanded_expressions.items.len, report.typed.items.len });

                // Show all expressions with their status
                std.debug.print("\nExpression details:\n", .{});
                for (expanded_expressions.items, 0..) |expr, idx| {
                    const line = if (idx < line_numbers.items.len) line_numbers.items[idx] else 0;
                    const maybe_str = self.formatValue(expr) catch null;
                    if (maybe_str) |expr_str| {
                        defer self.allocator.*.free(expr_str);
                        const status = if (idx < report.typed.items.len) "✓" else "✗";
                        std.debug.print("  {s} Line {d} (expr #{d}): {s}\n", .{ status, line, idx, expr_str });
                    }
                }
                std.debug.print("\nNote: {d} expressions type-checked successfully, {d} failed\n", .{ report.typed.items.len, expanded_expressions.items.len - report.typed.items.len });
            }
            for (report.errors.items) |detail| {
                const line = if (detail.index < line_numbers.items.len) line_numbers.items[detail.index] else 0;
                const maybe_str = self.formatValue(detail.expr) catch null;
                if (detail.info) |info| {
                    switch (info) {
                        .unbound => |unbound| {
                            std.debug.print("unbound variable {s}", .{unbound.name});
                        },
                    }
                }
                if (maybe_str) |expr_str| {
                    defer self.allocator.*.free(expr_str);
                    std.debug.print("Type error at line {d} (expr #{d}): {s} -> {s}\n", .{ line, detail.index, @errorName(detail.err), expr_str });
                } else {
                    std.debug.print("Type error at line {d} (expr #{d}): {s}\n", .{ line, detail.index, @errorName(detail.err) });
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

        // Note: expanded_expressions and report.typed have the same length
        // because we filtered out macros before type checking
        for (expanded_expressions.items, report.typed.items) |expr, typed_val| {

            // Check for namespace declaration
            if (expr.* == .namespace) {
                namespace_name = expr.namespace.name;
                continue;
            }

            // Check if this is a def or extern declaration
            if (expr.* == .list) {
                var iter = expr.list.iterator();
                if (iter.next()) |head_val| {
                    if (head_val.isSymbol()) {
                        // Skip extern declarations and directives - they're handled in forward decl pass
                        if (std.mem.eql(u8, head_val.symbol, "extern-fn") or
                            std.mem.eql(u8, head_val.symbol, "extern-type") or
                            std.mem.eql(u8, head_val.symbol, "extern-union") or
                            std.mem.eql(u8, head_val.symbol, "extern-struct") or
                            std.mem.eql(u8, head_val.symbol, "extern-var") or
                            std.mem.eql(u8, head_val.symbol, "include-header") or
                            std.mem.eql(u8, head_val.symbol, "link-library") or
                            std.mem.eql(u8, head_val.symbol, "compiler-flag"))
                        {
                            continue;
                        }
                    }
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
            try non_def_exprs.append(self.allocator.*, .{ .expr = expr, .typed = typed_val, .idx = 0 }); // idx not needed for non-defs
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

        // Check if any expression uses 'let' - if so, include stdint headers
        for (expanded_expressions.items) |expr| {
            if (self.hasLet(expr)) {
                includes.need_stdint = true;
                includes.need_stdbool = true;
                includes.need_stddef = true;
                break;
            }
        }

        // First pass: emit forward declarations for functions and structs
        for (expanded_expressions.items, report.typed.items) |expr, typed_val| {
            try self.emitForwardDecl(forward_writer, expr, typed_val, &checker, &includes);
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
                    try ns_writer.print("    ", .{});
                    self.emitArrayDecl(ns_writer, sanitized_field, def.var_type, &includes) catch |err| {
                        if (err == Error.UnsupportedType) {
                            try ns_writer.print("// unsupported type for: {s}\n", .{def.name});
                            continue;
                        }
                        return err;
                    };
                    try ns_writer.print(";\n", .{});
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
                } else if (def.var_type == .array) {
                    // Arrays can't be assigned in C, so we need to copy element by element
                    // For now, generate a loop to initialize the array
                    const array_type = def.var_type.array;

                    // Check if there's an init value
                    const typed_list = def.typed.list;
                    const has_init = typed_list.elements.len == 3; // array op has 3 elements when initialized

                    if (has_init) {
                        try init_writer.print("    for (size_t __i_{s} = 0; __i_{s} < {d}; __i_{s}++) {{\n", .{ sanitized_field, sanitized_field, array_type.size, sanitized_field });
                        try init_writer.print("        ns->{s}[__i_{s}] = ", .{ sanitized_field, sanitized_field });
                        self.writeExpressionTyped(init_writer, typed_list.elements[2], &init_ctx, &includes) catch |err| {
                            switch (err) {
                                Error.UnsupportedExpression => {
                                    try init_writer.print("0;\n    }}\n", .{});
                                    continue;
                                },
                                else => return err,
                            }
                        };
                        try init_writer.print(";\n    }}\n", .{});
                    }
                    // If no init value, array is left uninitialized (or zero-initialized by C)
                } else {
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
                        // Check if this is a let expression - handle it specially using compound statement
                        var is_let = false;
                        if (value_expr.isList()) {
                            var check_iter = value_expr.list.iterator();
                            if (check_iter.next()) |first| {
                                if (first.isSymbol() and std.mem.eql(u8, first.symbol, "let")) {
                                    is_let = true;
                                }
                            }
                        }

                        if (is_let) {
                            // Emit let as compound statement expression: ns->field = ({ bindings; body; });
                            try init_writer.print("    ns->{s} = ", .{sanitized_field});
                            self.emitLetAsCompoundStatement(init_writer, value_expr, def.typed, &init_ctx, &includes) catch |err| {
                                switch (err) {
                                    Error.UnsupportedExpression, Error.MissingOperand, Error.InvalidIfForm, Error.InvalidDefinition => {
                                        try init_writer.print("0; // unsupported let expression\n", .{});
                                        continue;
                                    },
                                    else => return err,
                                }
                            };
                            try init_writer.print(";\n", .{});
                        } else {
                            // For regular variables, evaluate the expression
                            try init_writer.print("    ns->{s} = ", .{sanitized_field});

                            // Use typed AST for code generation
                            self.writeExpressionTyped(init_writer, def.typed, &init_ctx, &includes) catch |err| {
                                switch (err) {
                                    Error.UnsupportedExpression, Error.MissingOperand, Error.InvalidIfForm => {
                                        try init_writer.print("0; // unsupported expression\n", .{});
                                        continue;
                                    },
                                    else => return err,
                                }
                            };
                            try init_writer.print(";\n", .{});
                        }
                    }
                }
            }

            try init_writer.print("}}\n\n", .{});

            // Emit static function definitions
            for (namespace_defs.items) |def| {
                if (def.var_type == .function) {
                    // Extract the function expression from the def (untyped AST)
                    var def_iter = def.expr.list.iterator();
                    _ = def_iter.next(); // skip 'def'
                    _ = def_iter.next(); // skip name
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

                    // Extract the typed function expression from def.typed
                    // def.typed is the typed body from the type checker (not the full def form)
                    // It should be a list: (fn [params...] body...)
                    const fn_typed = def.typed;

                    if (maybe_fn_expr) |fn_expr| {
                        // Use emitFunctionDefinition which handles both untyped and typed AST
                        self.emitFunctionDefinition(init_writer, def.name, fn_expr, fn_typed, def.var_type, &includes) catch |err| {
                            std.debug.print("Failed to emit namespace function {s}: {}\n", .{ def.name, err });
                            continue;
                        };
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
        for (expanded_expressions.items, report.typed.items) |expr, typed_val| {
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
        if (includes.need_stdlib) {
            try output.appendSlice(self.allocator.*, "#include <stdlib.h>\n");
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
        switch (expr.*) {
            .list => |list| {
                var iter = list.iterator();
                const head_val = iter.next() orelse return;
                if (!head_val.isSymbol()) return;

                const head = head_val.symbol;

                // Handle extern declarations
                if (std.mem.eql(u8, head, "extern-fn")) {
                    // extern-fn: function is declared in C header, we just acknowledge it exists
                    // Don't generate any extern declaration - the C header already has it
                    // This avoids conflicts when headers define functions with different signatures
                    _ = typed; // Function type info is only needed for type checking
                    return;
                } else if (std.mem.eql(u8, head, "extern-type")) {
                    // extern-type just creates a type alias or forward declaration
                    const name_val = iter.next() orelse return;
                    if (!name_val.isSymbol()) return;
                    const type_name = name_val.symbol;

                    // For SDL, most types are opaque structs
                    try forward_writer.print("typedef struct {s} {s};\n", .{ type_name, type_name });
                    return;
                } else if (std.mem.eql(u8, head, "extern-union")) {
                    // extern-union creates a union type forward declaration
                    const name_val = iter.next() orelse return;
                    if (!name_val.isSymbol()) return;
                    const type_name = name_val.symbol;

                    // Create union typedef
                    try forward_writer.print("typedef union {s} {s};\n", .{ type_name, type_name });
                    return;
                } else if (std.mem.eql(u8, head, "extern-struct")) {
                    // extern-struct: type is defined in C header, we just acknowledge it exists
                    // Don't generate any typedef - the C header already has it
                    return;
                } else if (std.mem.eql(u8, head, "extern-var")) {
                    // extern-var creates an extern variable declaration
                    const name_val = iter.next() orelse return;
                    if (!name_val.isSymbol()) return;
                    const var_name = name_val.symbol;

                    const type_val = iter.next() orelse return;
                    if (type_val.isSymbol()) {
                        // Parse the type - for now just handle simple types
                        const type_str = self.cTypeFor(try self.parseSimpleType(type_val.symbol), includes) catch return;
                        try forward_writer.print("extern {s} {s};\n", .{ type_str, var_name });
                    }
                    return;
                } else if (std.mem.eql(u8, head, "include-header")) {
                    // Handle include-header directive
                    const header_val = iter.next() orelse return;
                    if (!header_val.isString()) return;
                    const header_name = header_val.string;

                    // Emit the #include directive
                    try forward_writer.print("#include \"{s}\"\n", .{header_name});
                    return;
                } else if (std.mem.eql(u8, head, "link-library")) {
                    // Collect library for linking
                    const lib_val = iter.next() orelse return;
                    if (!lib_val.isString()) return;
                    const lib_name = lib_val.string;

                    // Store library name for later use during compilation
                    try self.linked_libraries.append(self.allocator.*, try self.allocator.*.dupe(u8, lib_name));
                    return;
                } else if (std.mem.eql(u8, head, "compiler-flag")) {
                    // Collect compiler flag
                    const flag_val = iter.next() orelse return;
                    if (!flag_val.isString()) return;
                    const flag = flag_val.string;

                    // Store compiler flag for later use during compilation
                    try self.compiler_flags.append(self.allocator.*, try self.allocator.*.dupe(u8, flag));
                    return;
                } else if (std.mem.eql(u8, head, "def")) {
                    const name_val = iter.next() orelse return;
                    if (!name_val.isSymbol()) return;
                    const name = name_val.symbol;

                    // Check if this is a struct or enum definition by looking it up in type_defs
                    if (checker.type_defs.get(name)) |type_def| {
                        if (type_def == .struct_type) {
                            // This is a struct definition - emit struct declaration
                            const sanitized_name = try self.sanitizeIdentifier(name);
                            defer self.allocator.*.free(sanitized_name);

                            try forward_writer.print("typedef struct {{\n", .{});
                            for (type_def.struct_type.fields) |field| {
                                const sanitized_field = try self.sanitizeIdentifier(field.name);
                                defer self.allocator.*.free(sanitized_field);
                                try forward_writer.print("    ", .{});
                                self.emitArrayDecl(forward_writer, sanitized_field, field.field_type, includes) catch {
                                    try forward_writer.print("// unsupported field type: {s}\n", .{field.name});
                                    continue;
                                };
                                try forward_writer.print(";\n", .{});
                            }
                            try forward_writer.print("}} {s};\n\n", .{sanitized_name});
                            return;
                        } else if (type_def == .enum_type) {
                            // This is an enum definition - emit enum declaration
                            const sanitized_name = try self.sanitizeIdentifier(name);
                            defer self.allocator.*.free(sanitized_name);

                            try forward_writer.print("typedef enum {{\n", .{});
                            for (type_def.enum_type.variants) |variant| {
                                const sanitized_variant = try self.sanitizeIdentifier(variant.qualified_name.?);
                                defer self.allocator.*.free(sanitized_variant);
                                try forward_writer.print("    {s},\n", .{sanitized_variant});
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
                                    const param_val = params_vec.at(i);
                                    const sanitized_param = try self.sanitizeIdentifier(param_val.symbol);
                                    defer self.allocator.*.free(sanitized_param);
                                    // Use emitArrayDecl which handles function pointers correctly
                                    self.emitArrayDecl(forward_writer, sanitized_param, param_type, includes) catch return;
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

                // Skip extern declarations - they're already emitted in forward declarations
                if (std.mem.eql(u8, head, "extern-fn") or
                    std.mem.eql(u8, head, "extern-type") or
                    std.mem.eql(u8, head, "extern-union") or
                    std.mem.eql(u8, head, "extern-struct") or
                    std.mem.eql(u8, head, "extern-var") or
                    std.mem.eql(u8, head, "include-header") or
                    std.mem.eql(u8, head, "link-library") or
                    std.mem.eql(u8, head, "compiler-flag"))
                {
                    return;
                }

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
                try self.emitFunctionDefinition(def_writer, name, value_expr, typed, var_type, includes);
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

        // Use typed AST for code generation
        self.writeExpressionTyped(def_writer, typed, &empty_ctx, includes) catch |err| {
            switch (err) {
                Error.UnsupportedExpression, Error.MissingOperand, Error.InvalidIfForm => {
                    const repr = try self.formatValue(list_expr);
                    defer self.allocator.*.free(repr);
                    try def_writer.print("0; // unsupported definition: {s}\n", .{repr});
                    return;
                },
                else => return err,
            }
        };
        try def_writer.print(";\n", .{});
    }

    fn emitFunctionDefinition(self: *SimpleCCompiler, def_writer: anytype, name: []const u8, fn_expr: *Value, fn_typed: *TypedValue, fn_type: Type, includes: *IncludeFlags) Error!void {
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

        // Extract typed body expressions from fn_typed
        // fn_typed is a list like (fn [params...] body...)
        if (fn_typed.* != .list) return Error.InvalidFunction;
        const typed_list = fn_typed.list;
        if (typed_list.elements.len < 2) return Error.InvalidFunction;

        const typed_body_count = typed_list.elements.len - 2; // Subtract 'fn' symbol and params vector
        if (typed_body_count == 0) return Error.InvalidFunction;

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

        const sanitized_name = try self.sanitizeIdentifier(name);
        defer self.allocator.*.free(sanitized_name);
        try def_writer.print("static {s} {s}(", .{ return_type_str, sanitized_name });
        var index: usize = 0;
        while (index < params_vec.len()) : (index += 1) {
            const param_val = params_vec.at(index);
            if (!param_val.isSymbol()) return Error.InvalidFunction;
            if (index > 0) {
                try def_writer.print(", ", .{});
            }
            const sanitized_param = try self.sanitizeIdentifier(param_val.symbol);
            defer self.allocator.*.free(sanitized_param);
            // Use emitArrayDecl which handles function pointers correctly
            self.emitArrayDecl(def_writer, sanitized_param, param_types[index], includes) catch |err| {
                if (err == Error.UnsupportedType) {
                    const repr = try self.formatValue(fn_expr);
                    defer self.allocator.*.free(repr);
                    try def_writer.print("/* unsupported */", .{});
                } else {
                    return err;
                }
            };
        }
        try def_writer.writeAll(") {\n");

        // Emit all body expressions except the last using typed AST
        for (typed_list.elements[2 .. typed_list.elements.len - 1]) |typed_stmt| {
            try def_writer.print("    ", .{});
            self.writeExpressionTyped(def_writer, typed_stmt, &empty_ctx, includes) catch |err| {
                switch (err) {
                    Error.UnsupportedExpression, Error.MissingOperand, Error.InvalidIfForm => {
                        try def_writer.print("/* unsupported statement */", .{});
                    },
                    else => return err,
                }
            };
            try def_writer.writeAll(";\n");
        }

        // Emit return statement with last expression using typed AST
        try def_writer.print("    return ", .{});
        const last_typed_expr = typed_list.elements[typed_list.elements.len - 1];
        self.writeExpressionTyped(def_writer, last_typed_expr, &empty_ctx, includes) catch |err| {
            switch (err) {
                Error.UnsupportedExpression, Error.MissingOperand, Error.InvalidIfForm => {
                    const repr = try self.formatValue(fn_expr);
                    defer self.allocator.*.free(repr);
                    try def_writer.writeAll("0;\n}\n");
                    try def_writer.print("// unsupported function body: {s}\n", .{repr});
                    std.debug.print("ERROR writing return expression: {s}\n", .{@errorName(err)});
                    return;
                },
                else => return err,
            }
        };
        try def_writer.writeAll(";\n}\n");
    }

    fn hasLet(self: *SimpleCCompiler, expr: *Value) bool {
        switch (expr.*) {
            .list => |list| {
                var iter = list.iterator();
                if (iter.next()) |first| {
                    if (first.isSymbol() and std.mem.eql(u8, first.symbol, "let")) {
                        return true;
                    }
                }
                // Check nested expressions
                iter = list.iterator();
                while (iter.next()) |child| {
                    if (self.hasLet(child)) return true;
                }
                return false;
            },
            .vector => |vec| {
                for (0..vec.len()) |i| {
                    if (self.hasLet(vec.at(i))) return true;
                }
                return false;
            },
            else => return false,
        }
    }

    // Helper to emit let expressions as C compound statement expressions: ({ stmts; result; })
    // This handles nested lets recursively by relying on writeExpressionTyped for the body
    fn emitLetAsCompoundStatement(self: *SimpleCCompiler, writer: anytype, value_expr: *Value, typed: *TypedValue, ns_ctx: *NamespaceContext, includes: *IncludeFlags) Error!void {
        _ = typed; // We'll use the untyped AST structure and recurse via writeExpressionTyped for typed code generation

        // Check if this is a let expression
        if (!value_expr.isList()) return Error.InvalidDefinition;

        var expr_iter = value_expr.list.iterator();
        const first = expr_iter.next() orelse return Error.InvalidDefinition;
        if (!first.isSymbol() or !std.mem.eql(u8, first.symbol, "let")) {
            return Error.InvalidDefinition;
        }

        // Get the bindings vector
        const bindings_val = expr_iter.next() orelse return Error.InvalidDefinition;
        if (!bindings_val.isVector()) return Error.InvalidDefinition;
        const bindings_vec = bindings_val.vector;

        if (bindings_vec.len() % 3 != 0) return Error.InvalidDefinition;
        const binding_count = bindings_vec.len() / 3;

        // Start compound statement expression
        try writer.print("({{ ", .{});

        // Emit each binding as a C variable declaration
        var idx: usize = 0;
        while (idx < binding_count) : (idx += 1) {
            const name_val = bindings_vec.at(idx * 3);
            const annotation_val = bindings_vec.at(idx * 3 + 1);
            const init_val = bindings_vec.at(idx * 3 + 2);

            if (!name_val.isSymbol()) return Error.InvalidDefinition;
            const var_name = name_val.symbol;

            // Parse type annotation to get C type
            const var_type = try self.parseTypeFromAnnotation(annotation_val);
            const sanitized = try self.sanitizeIdentifier(var_name);
            defer self.allocator.*.free(sanitized);

            // Special handling for function pointer types
            // Function pointer: int32_t (*varname)(int32_t, int32_t) = ...
            // vs regular type: int32_t varname = ...
            const is_fn_ptr = (var_type == .pointer and var_type.pointer.* == .function) or var_type == .function;

            if (is_fn_ptr) {
                // For function pointers, we need to inject the variable name into the type
                // Type format: "RetType (*)(Params)" -> "RetType (*varname)(Params)"
                const fn_type = if (var_type == .pointer) var_type.pointer.function else var_type.function;
                const ret_c_type = try self.cTypeFor(fn_type.return_type, includes);

                // Build parameter list
                var params_list = std.ArrayList(u8){};
                defer params_list.deinit(self.allocator.*);
                var params_writer = params_list.writer(self.allocator.*);

                for (fn_type.param_types, 0..) |param_type, i| {
                    if (i > 0) try params_writer.print(", ", .{});
                    const param_c_type = try self.cTypeFor(param_type, includes);
                    try params_writer.print("{s}", .{param_c_type});
                }

                // Emit: RetType (*varname)(Params) =
                try writer.print("{s} (*{s})({s}) = ", .{ ret_c_type, sanitized, params_list.items });
            } else {
                const c_type = try self.cTypeFor(var_type, includes);
                try writer.print("{s} {s} = ", .{ c_type, sanitized });
            }

            // Recursively emit the init value - it might be a nested let!
            // For now, emit it as untyped (this is a limitation - we'd need the type checker's typed values for each binding)
            try self.emitUntypedValueExpression(writer, init_val, ns_ctx, includes);
            try writer.print("; ", .{});
        }

        // Emit the body - collect all expressions first
        var body_exprs = std.ArrayList(*Value){};
        defer body_exprs.deinit(self.allocator.*);

        while (expr_iter.next()) |body_expr| {
            try body_exprs.append(self.allocator.*, body_expr);
        }

        if (body_exprs.items.len == 0) return Error.InvalidDefinition;

        // Emit all but the last expression as statements (with semicolons)
        for (body_exprs.items[0..body_exprs.items.len - 1]) |stmt_expr| {
            try self.emitUntypedValueExpression(writer, stmt_expr, ns_ctx, includes);
            try writer.print("; ", .{});
        }

        // Emit the final expression as the return value (no semicolon)
        const final_expr = body_exprs.items[body_exprs.items.len - 1];
        try self.emitUntypedValueExpression(writer, final_expr, ns_ctx, includes);

        // Close compound statement expression
        try writer.print("; }})", .{});
    }

    // Simplified helper to emit untyped value expressions
    fn emitUntypedValueExpression(self: *SimpleCCompiler, writer: anytype, expr: *Value, ns_ctx: *NamespaceContext, includes: *IncludeFlags) Error!void {
        if (expr.isSymbol()) {
            // Handle special symbols
            if (std.mem.eql(u8, expr.symbol, "pointer-null")) {
                try writer.print("NULL", .{});
                return;
            }
            if (std.mem.eql(u8, expr.symbol, "true")) {
                try writer.print("true", .{});
                return;
            }
            if (std.mem.eql(u8, expr.symbol, "false")) {
                try writer.print("false", .{});
                return;
            }

            // Check if this symbol is a namespace field
            if (ns_ctx.def_names.contains(expr.symbol)) {
                if (ns_ctx.name) |_| {
                    const sanitized_field = try self.sanitizeIdentifier(expr.symbol);
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

            const sanitized = try self.sanitizeIdentifier(expr.symbol);
            defer self.allocator.*.free(sanitized);
            try writer.print("{s}", .{sanitized});
        } else if (expr.isInt()) {
            try writer.print("{d}", .{expr.int});
        } else if (expr.isFloat()) {
            try writer.print("{d}", .{expr.float});
        } else if (expr.isString()) {
            try writer.print("\"{s}\"", .{expr.string});
        } else if (expr.isList()) {
            var list_iter = expr.list.iterator();
            const first = list_iter.next() orelse return Error.MissingOperand;

            if (first.isSymbol()) {
                const op = first.symbol;

                // Handle nested let recursively
                if (std.mem.eql(u8, op, "let")) {
                    // Recursively emit nested let as compound statement
                    try self.emitLetAsCompoundStatement(writer, expr, undefined, ns_ctx, includes);
                    return;
                }

                // Handle cast: (cast TargetType value)
                if (std.mem.eql(u8, op, "cast")) {
                    const type_expr = list_iter.next() orelse return Error.MissingOperand;
                    const value_expr = list_iter.next() orelse return Error.MissingOperand;

                    // Parse the target type
                    const target_type = try self.parseTypeFromAnnotation_Simple(type_expr);
                    const c_type = try self.cTypeFor(target_type, includes);

                    // Emit cast: (TargetType)(value)
                    try writer.print("(({s})(", .{c_type});
                    try self.emitUntypedValueExpression(writer, value_expr, ns_ctx, includes);
                    try writer.print("))", .{});
                    return;
                }

                // Handle c-str: (c-str string) - cast string literal to const char*
                if (std.mem.eql(u8, op, "c-str")) {
                    const str_expr = list_iter.next() orelse return Error.MissingOperand;
                    try writer.print("((const char*)", .{});
                    try self.emitUntypedValueExpression(writer, str_expr, ns_ctx, includes);
                    try writer.print(")", .{});
                    return;
                }

                // Handle arithmetic operators
                if (std.mem.eql(u8, op, "+") or std.mem.eql(u8, op, "-") or
                    std.mem.eql(u8, op, "*") or std.mem.eql(u8, op, "/") or
                    std.mem.eql(u8, op, "%")) {
                    try writer.print("(", .{});
                    var first_arg = true;
                    while (list_iter.next()) |arg| {
                        if (!first_arg) {
                            try writer.print(" {s} ", .{op});
                        }
                        try self.emitUntypedValueExpression(writer, arg, ns_ctx, includes);
                        first_arg = false;
                    }
                    try writer.print(")", .{});
                    return;
                }

                // Assume it's a function call
                const sanitized = try self.sanitizeIdentifier(op);
                defer self.allocator.*.free(sanitized);
                try writer.print("{s}(", .{sanitized});
                var first_arg = true;
                while (list_iter.next()) |arg| {
                    if (!first_arg) {
                        try writer.print(", ", .{});
                    }
                    try self.emitUntypedValueExpression(writer, arg, ns_ctx, includes);
                    first_arg = false;
                }
                try writer.print(")", .{});
                return;
            }
            std.debug.print("ERROR: Unsupported list form in let binding\n", .{});
            return Error.UnsupportedExpression;
        } else {
            std.debug.print("ERROR: Unsupported value type in let: {s}\n", .{@tagName(expr.*)});
            return Error.UnsupportedExpression;
        }
    }

    // Helper to parse type from type annotation expression
    fn parseTypeFromAnnotation(self: *SimpleCCompiler, annotation: *Value) Error!Type {
        // Type annotation is (: Type)
        if (!annotation.isList()) {
            std.debug.print("ERROR: Type annotation is not a list: {s}\n", .{@tagName(annotation.*)});
            return Error.InvalidTypeAnnotation;
        }
        var iter = annotation.list.iterator();
        const first = iter.next() orelse {
            std.debug.print("ERROR: Empty type annotation list\n", .{});
            return Error.InvalidTypeAnnotation;
        };
        if (!first.isKeyword() or first.keyword.len != 0) {
            std.debug.print("ERROR: Type annotation doesn't start with ':' keyword, got: {s}\n", .{@tagName(first.*)});
            return Error.InvalidTypeAnnotation;
        }

        const type_expr = iter.next() orelse {
            std.debug.print("ERROR: Type annotation missing type expression after ':'\n", .{});
            return Error.InvalidTypeAnnotation;
        };

        // Parse basic types
        if (type_expr.isSymbol()) {
            const type_name = type_expr.symbol;
            if (std.mem.eql(u8, type_name, "Int")) return Type.int;
            if (std.mem.eql(u8, type_name, "Float")) return Type.float;
            if (std.mem.eql(u8, type_name, "Bool")) return Type.bool;
            if (std.mem.eql(u8, type_name, "String")) return Type.string;
            if (std.mem.eql(u8, type_name, "I8")) return Type.i8;
            if (std.mem.eql(u8, type_name, "I16")) return Type.i16;
            if (std.mem.eql(u8, type_name, "I32")) return Type.i32;
            if (std.mem.eql(u8, type_name, "I64")) return Type.i64;
            if (std.mem.eql(u8, type_name, "U8")) return Type.u8;
            if (std.mem.eql(u8, type_name, "U16")) return Type.u16;
            if (std.mem.eql(u8, type_name, "U32")) return Type.u32;
            if (std.mem.eql(u8, type_name, "U64")) return Type.u64;
            if (std.mem.eql(u8, type_name, "F32")) return Type.f32;
            if (std.mem.eql(u8, type_name, "F64")) return Type.f64;

            // For unknown type names, treat them as extern types
            // This is a hack - we create an extern type with the given name
            const extern_type_ptr = try self.allocator.*.create(type_checker.ExternType);
            extern_type_ptr.* = .{
                .name = type_name,
                .is_opaque = true, // Assume opaque since we don't have field info
            };
            return Type{ .extern_type = extern_type_ptr };
        }

        // For complex types (like Pointer, Array, etc.), we'd need more parsing
        if (type_expr.isList()) {
            var type_iter = type_expr.list.iterator();
            if (type_iter.next()) |first_type| {
                if (first_type.isSymbol()) {
                    const type_constructor = first_type.symbol;

                    // Handle (Pointer T)
                    if (std.mem.eql(u8, type_constructor, "Pointer")) {
                        const pointee_expr = type_iter.next() orelse {
                            std.debug.print("ERROR: Pointer type missing pointee type\n", .{});
                            return Error.InvalidTypeAnnotation;
                        };
                        const pointee_type = try self.parseTypeFromAnnotation_Simple(pointee_expr);
                        const pointee_ptr = try self.allocator.*.create(Type);
                        pointee_ptr.* = pointee_type;
                        return Type{ .pointer = pointee_ptr };
                    }

                    std.debug.print("ERROR: Unsupported type constructor: {s}\n", .{type_constructor});
                    return Error.InvalidTypeAnnotation;
                }
            }
        }

        std.debug.print("ERROR: Complex type annotation not supported yet: {s}\n", .{@tagName(type_expr.*)});
        return Error.InvalidTypeAnnotation;
    }

    // Simplified type parser for pointee types (no type annotation wrapper)
    fn parseTypeFromAnnotation_Simple(self: *SimpleCCompiler, type_expr: *Value) Error!Type {
        if (type_expr.isSymbol()) {
            const type_name = type_expr.symbol;
            if (std.mem.eql(u8, type_name, "Int")) return Type.int;
            if (std.mem.eql(u8, type_name, "Float")) return Type.float;
            if (std.mem.eql(u8, type_name, "Bool")) return Type.bool;
            if (std.mem.eql(u8, type_name, "String")) return Type.string;
            if (std.mem.eql(u8, type_name, "Nil")) return Type.nil;
            if (std.mem.eql(u8, type_name, "I8")) return Type.i8;
            if (std.mem.eql(u8, type_name, "I16")) return Type.i16;
            if (std.mem.eql(u8, type_name, "I32")) return Type.i32;
            if (std.mem.eql(u8, type_name, "I64")) return Type.i64;
            if (std.mem.eql(u8, type_name, "U8")) return Type.u8;
            if (std.mem.eql(u8, type_name, "U16")) return Type.u16;
            if (std.mem.eql(u8, type_name, "U32")) return Type.u32;
            if (std.mem.eql(u8, type_name, "U64")) return Type.u64;
            if (std.mem.eql(u8, type_name, "F32")) return Type.f32;
            if (std.mem.eql(u8, type_name, "F64")) return Type.f64;

            // For unknown type names, treat them as extern types
            const extern_type_ptr = try self.allocator.*.create(type_checker.ExternType);
            extern_type_ptr.* = .{
                .name = type_name,
                .is_opaque = true,
            };
            return Type{ .extern_type = extern_type_ptr };
        }

        // Handle nested complex types like (Pointer (Pointer T))
        if (type_expr.isList()) {
            var type_iter = type_expr.list.iterator();
            if (type_iter.next()) |first_type| {
                if (first_type.isSymbol()) {
                    const type_constructor = first_type.symbol;

                    // Handle (Pointer T) recursively
                    if (std.mem.eql(u8, type_constructor, "Pointer")) {
                        const pointee_expr = type_iter.next() orelse {
                            std.debug.print("ERROR: Pointer type missing pointee type\n", .{});
                            return Error.InvalidTypeAnnotation;
                        };
                        const pointee_type = try self.parseTypeFromAnnotation_Simple(pointee_expr);
                        const pointee_ptr = try self.allocator.*.create(Type);
                        pointee_ptr.* = pointee_type;
                        return Type{ .pointer = pointee_ptr };
                    }

                    // Handle (-> [param_types...] return_type)
                    if (std.mem.eql(u8, type_constructor, "->")) {
                        const params_expr = type_iter.next() orelse {
                            std.debug.print("ERROR: Function type missing parameter list\n", .{});
                            return Error.InvalidTypeAnnotation;
                        };
                        const return_expr = type_iter.next() orelse {
                            std.debug.print("ERROR: Function type missing return type\n", .{});
                            return Error.InvalidTypeAnnotation;
                        };

                        // Parse parameter types
                        if (!params_expr.isVector()) {
                            std.debug.print("ERROR: Function parameter list must be a vector\n", .{});
                            return Error.InvalidTypeAnnotation;
                        }
                        const params_vec = params_expr.vector;
                        const param_types = try self.allocator.*.alloc(Type, params_vec.len());
                        for (0..params_vec.len()) |i| {
                            param_types[i] = try self.parseTypeFromAnnotation_Simple(params_vec.at(i));
                        }

                        // Parse return type
                        const return_type = try self.parseTypeFromAnnotation_Simple(return_expr);

                        // Create FunctionType on heap (Type.function expects *FunctionType)
                        const fn_type_ptr = try self.allocator.*.create(type_checker.FunctionType);
                        fn_type_ptr.* = type_checker.FunctionType{
                            .param_types = param_types,
                            .return_type = return_type,
                        };
                        return Type{ .function = fn_type_ptr };
                    }

                    std.debug.print("ERROR: Unsupported type constructor in simple parser: {s}\n", .{type_constructor});
                    return Error.InvalidTypeAnnotation;
                }
            }
        }

        std.debug.print("ERROR: Unsupported simple type: {s}\n", .{@tagName(type_expr.*)});
        return Error.InvalidTypeAnnotation;
    }

    fn sanitizeIdentifier(self: *SimpleCCompiler, name: []const u8) ![]u8 {
        // Check if it's a C keyword
        const c_keywords = [_][]const u8{
            "auto",       "break",    "case",     "char",   "const",   "continue",
            "default",    "do",       "double",   "else",   "enum",    "extern",
            "float",      "for",      "goto",     "if",     "inline",  "int",
            "long",       "register", "restrict", "return", "short",   "signed",
            "sizeof",     "static",   "struct",   "switch", "typedef", "union",
            "unsigned",   "void",     "volatile", "while",  "_Bool",   "_Complex",
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
                result[i + 1] = if (c == '-' or c == '.' or c == '/') '_' else c;
            }
            return result;
        } else {
            var result = try self.allocator.*.alloc(u8, name.len);
            for (name, 0..) |c, i| {
                result[i] = if (c == '-' or c == '.' or c == '/') '_' else c;
            }
            return result;
        }
    }

    fn emitPrintStatement(self: *SimpleCCompiler, body_writer: anytype, expr: *Value, typed: *TypedValue, includes: *IncludeFlags, ns_ctx: *NamespaceContext) Error!void {
        const expr_type = typed.getType();

        // Don't print Nil/void statements (e.g., pointer-write!, deallocate, while loops)
        if (expr_type == .nil or expr_type == .void) {
            try body_writer.print("    ", .{});
            // Use typed AST for code generation
            self.writeExpressionTyped(body_writer, typed, ns_ctx, includes) catch |err| {
                switch (err) {
                    Error.UnsupportedExpression, Error.MissingOperand, Error.InvalidIfForm => {
                        const repr = try self.formatValue(expr);
                        defer self.allocator.*.free(repr);
                        try body_writer.print("/* unsupported: {s} */", .{repr});
                    },
                    else => return err,
                }
            };
            try body_writer.print(";\n", .{});
            return;
        }

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

        // Use typed AST for code generation
        self.writeExpressionTyped(body_writer, typed, ns_ctx, includes) catch |err| {
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
        if (wrap_bool) {
            try body_writer.writeAll(") ? 1 : 0)");
        }
        try body_writer.print(");\n", .{});
    }

    fn writeExpressionTyped(self: *SimpleCCompiler, writer: anytype, typed: *TypedValue, ns_ctx: *NamespaceContext, includes: *IncludeFlags) Error!void {
        switch (typed.*) {
            .struct_instance => |si| {
                // Emit C99 compound literal: (TypeName){field1, field2, ...}
                // Handle both regular struct_type and extern_type
                const type_name = switch (si.type) {
                    .struct_type => |st| st.name,
                    .extern_type => |et| et.name,
                    else => return Error.TypeCheckFailed,
                };
                const sanitized_name = try self.sanitizeIdentifier(type_name);
                defer self.allocator.*.free(sanitized_name);

                try writer.print("({s}){{", .{sanitized_name});
                for (si.field_values, 0..) |field_val, i| {
                    if (i > 0) try writer.print(", ", .{});
                    try self.writeExpressionTyped(writer, field_val, ns_ctx, includes);
                }
                try writer.print("}}", .{});
            },
            .list => |l| {
                // Check for c-str operation first
                if (l.elements.len == 1 and l.type == .c_string) {
                    // c-str just passes through the string literal
                    try self.writeExpressionTyped(writer, l.elements[0], ns_ctx, includes);
                    return;
                }

                // Check for allocate operation (by type signature)
                // allocate creates a list with 0-1 elements and pointer type
                // BUT: distinguish from function calls that return pointers
                // Function calls have the function as the first element
                if ((l.elements.len == 0 or l.elements.len == 1) and l.type == .pointer) {
                    // If we have 1 element and it's a symbol with function/pointer type, it's a function call, not allocate
                    const is_function_call = if (l.elements.len == 1)
                        (l.elements[0].getType() == .function or
                         (l.elements[0].getType() == .pointer and l.elements[0].getType().pointer.* == .function))
                    else
                        false;

                    if (!is_function_call) {
                        const ptr_type = l.type;
                        const pointee = ptr_type.pointer.*;
                        includes.need_stdlib = true;
                        try writer.print("({{ ", .{});
                        const c_type = try self.cTypeFor(pointee, includes);
                        try writer.print("{s}* __tmp_ptr = malloc(sizeof({s})); ", .{ c_type, c_type });

                        // Initialize if value provided
                        if (l.elements.len == 1) {
                            try writer.print("*__tmp_ptr = ", .{});
                            try self.writeExpressionTyped(writer, l.elements[0], ns_ctx, includes);
                            try writer.print("; ", .{});
                        }

                        try writer.print("__tmp_ptr; }})", .{});
                        return;
                    }
                }

                // Check for deallocate first: 1 element (pointer), result type is nil
                if (l.elements.len == 1 and l.elements[0].getType() == .pointer and l.type == .nil) {
                    includes.need_stdlib = true;
                    try writer.print("free(", .{});
                    try self.writeExpressionTyped(writer, l.elements[0], ns_ctx, includes);
                    try writer.print(")", .{});
                    return;
                }

                // Check for cast: 2 elements (cast marker symbol, value)
                // cast generates: ((TargetType)value)
                if (l.elements.len == 2 and l.elements[0].* == .symbol and std.mem.eql(u8, l.elements[0].symbol.name, "cast")) {
                    const c_type = try self.cTypeFor(l.type, includes);
                    try writer.print("(({s})", .{c_type});
                    try self.writeExpressionTyped(writer, l.elements[1], ns_ctx, includes);
                    try writer.print(")", .{});
                    return;
                }

                // Check for dereference: 1 element (pointer), result type is NOT pointer and NOT nil
                if (l.elements.len == 1 and l.elements[0].getType() == .pointer and l.type != .pointer and l.type != .nil) {
                    try writer.print("(*", .{});
                    try self.writeExpressionTyped(writer, l.elements[0], ns_ctx, includes);
                    try writer.print(")", .{});
                    return;
                }

                if (l.elements.len > 0) {
                    const first = l.elements[0];
                    // Check for field access
                    if (first.* == .symbol and std.mem.eql(u8, first.symbol.name, ".")) {
                        if (l.elements.len == 3) {
                            try self.writeExpressionTyped(writer, l.elements[1], ns_ctx, includes);
                            if (l.elements[2].* == .symbol) {
                                try writer.print(".{s}", .{l.elements[2].symbol.name});
                                return;
                            }
                        }
                    }
                    // Check for pointer-write!: 2 elements (pointer, value), result type is nil
                    if (l.elements.len == 2 and l.elements[0].getType() == .pointer and l.type == .nil) {
                        try writer.print("(*", .{});
                        try self.writeExpressionTyped(writer, l.elements[0], ns_ctx, includes);
                        try writer.print(" = ", .{});
                        try self.writeExpressionTyped(writer, l.elements[1], ns_ctx, includes);
                        try writer.print(")", .{});
                        return;
                    }
                    // Check for pointer-equal?: 2 elements (both pointers), result type is bool
                    // This must come BEFORE pointer-field-read to avoid conflicts
                    if (l.elements.len == 2 and l.elements[0].getType() == .pointer and l.elements[1].getType() == .pointer and l.type == .bool) {
                        try writer.print("(", .{});
                        try self.writeExpressionTyped(writer, l.elements[0], ns_ctx, includes);
                        try writer.print(" == ", .{});
                        try self.writeExpressionTyped(writer, l.elements[1], ns_ctx, includes);
                        try writer.print(")", .{});
                        return;
                    }
                    // Check for pointer-field-read: 2 elements (pointer to struct, field symbol)
                    if (l.elements.len == 2 and l.elements[0].getType() == .pointer and l.elements[1].* == .symbol) {
                        const ptr_type = l.elements[0].getType().pointer.*;
                        if (ptr_type == .struct_type) {
                            try self.writeExpressionTyped(writer, l.elements[0], ns_ctx, includes);
                            try writer.print("->", .{});
                            const sanitized = try self.sanitizeIdentifier(l.elements[1].symbol.name);
                            defer self.allocator.*.free(sanitized);
                            try writer.print("{s}", .{sanitized});
                            return;
                        }
                    }
                    // Check for pointer-field-write!: 3 elements (pointer to struct, field symbol, value), result type is nil
                    if (l.elements.len == 3 and l.elements[0].getType() == .pointer and l.elements[1].* == .symbol and l.type == .nil) {
                        const ptr_type = l.elements[0].getType().pointer.*;
                        if (ptr_type == .struct_type) {
                            try self.writeExpressionTyped(writer, l.elements[0], ns_ctx, includes);
                            try writer.print("->", .{});
                            const sanitized = try self.sanitizeIdentifier(l.elements[1].symbol.name);
                            defer self.allocator.*.free(sanitized);
                            try writer.print("{s} = ", .{sanitized});
                            try self.writeExpressionTyped(writer, l.elements[2], ns_ctx, includes);
                            return;
                        }
                    }
                    // Check for address-of: 1 element (variable symbol), result type is pointer
                    // BUT: exclude function calls (symbol with function type)
                    if (l.elements.len == 1 and l.elements[0].* == .symbol and l.type == .pointer) {
                        const sym_type = l.elements[0].symbol.type;
                        const is_function_sym = sym_type == .function or
                            (sym_type == .pointer and sym_type.pointer.* == .function);

                        if (!is_function_sym) {
                            try writer.print("(&", .{});
                            try self.writeExpressionTyped(writer, l.elements[0], ns_ctx, includes);
                            try writer.print(")", .{});
                            return;
                        }
                    }
                    // Check for array operations by first element symbol
                    if (l.elements.len > 0 and l.elements[0].* == .symbol) {
                        const op = l.elements[0].symbol.name;

                        // array-ref: (array-ref array index) -> array[index]
                        if (std.mem.eql(u8, op, "array-ref") and l.elements.len == 3) {
                            try self.writeExpressionTyped(writer, l.elements[1], ns_ctx, includes);
                            try writer.print("[", .{});
                            try self.writeExpressionTyped(writer, l.elements[2], ns_ctx, includes);
                            try writer.print("]", .{});
                            return;
                        }

                        // array-set!: (array-set! array index value) -> array[index] = value
                        if (std.mem.eql(u8, op, "array-set!") and l.elements.len == 4 and l.type == .nil) {
                            try writer.print("(", .{});
                            try self.writeExpressionTyped(writer, l.elements[1], ns_ctx, includes);
                            try writer.print("[", .{});
                            try self.writeExpressionTyped(writer, l.elements[2], ns_ctx, includes);
                            try writer.print("] = ", .{});
                            try self.writeExpressionTyped(writer, l.elements[3], ns_ctx, includes);
                            try writer.print(")", .{});
                            return;
                        }

                        // array-length: (array-length array) -> compile-time size constant
                        if (std.mem.eql(u8, op, "array-length") and l.elements.len == 2) {
                            // The type checker already validated this is an array
                            // We return the compile-time constant from the array type
                            const array_type = l.elements[1].getType();
                            if (array_type == .array) {
                                try writer.print("{d}", .{array_type.array.size});
                                return;
                            }
                        }

                        // array: (array Type Size [InitValue])
                        // For typed arrays, we need to check if result type is array
                        if (std.mem.eql(u8, op, "array") and l.type == .array) {
                            const array_type = l.type.array;
                            const elem_c_type = try self.cTypeFor(array_type.element_type, includes);

                            // Check if there's an init value (element count is 3)
                            if (l.elements.len == 3) {
                                // Initialized array: {init_val, init_val, ...}
                                try writer.print("({{ {s} __tmp_arr[{d}]; for (size_t __i = 0; __i < {d}; __i++) __tmp_arr[__i] = ", .{ elem_c_type, array_type.size, array_type.size });
                                try self.writeExpressionTyped(writer, l.elements[2], ns_ctx, includes);
                                try writer.print("; __tmp_arr; }})", .{});
                            } else {
                                // Uninitialized array - just declare it
                                try writer.print("({{ {s} __tmp_arr[{d}]; __tmp_arr; }})", .{ elem_c_type, array_type.size });
                            }
                            return;
                        }

                        // array-ptr: (array-ptr array index) -> &array[index]
                        if (std.mem.eql(u8, op, "array-ptr") and l.elements.len == 3) {
                            try writer.print("(&", .{});
                            try self.writeExpressionTyped(writer, l.elements[1], ns_ctx, includes);
                            try writer.print("[", .{});
                            try self.writeExpressionTyped(writer, l.elements[2], ns_ctx, includes);
                            try writer.print("])", .{});
                            return;
                        }

                        // allocate-array: (allocate-array Type Size [InitValue])
                        if (std.mem.eql(u8, op, "allocate-array") and l.type == .pointer) {
                            const elem_type = l.type.pointer.*;
                            const elem_c_type = try self.cTypeFor(elem_type, includes);
                            includes.need_stdlib = true;

                            // Size is in l.elements[2] (after symbol and type)
                            const size_typed = l.elements[2];

                            if (l.elements.len == 4) {
                                // With initialization
                                try writer.print("({{ {s}* __arr = ({s}*)malloc(", .{ elem_c_type, elem_c_type });
                                try self.writeExpressionTyped(writer, size_typed, ns_ctx, includes);
                                try writer.print(" * sizeof({s})); for (size_t __i = 0; __i < ", .{elem_c_type});
                                try self.writeExpressionTyped(writer, size_typed, ns_ctx, includes);
                                try writer.print("; __i++) __arr[__i] = ", .{});
                                try self.writeExpressionTyped(writer, l.elements[3], ns_ctx, includes);
                                try writer.print("; __arr; }})", .{});
                            } else {
                                // Without initialization
                                try writer.print("({s}*)malloc(", .{elem_c_type});
                                try self.writeExpressionTyped(writer, size_typed, ns_ctx, includes);
                                try writer.print(" * sizeof({s}))", .{elem_c_type});
                            }
                            return;
                        }

                        // deallocate-array: free(ptr)
                        if (std.mem.eql(u8, op, "deallocate-array") and l.elements.len == 2) {
                            includes.need_stdlib = true;
                            try writer.print("free(", .{});
                            try self.writeExpressionTyped(writer, l.elements[1], ns_ctx, includes);
                            try writer.print(")", .{});
                            return;
                        }

                        // pointer-index-read: ptr[index]
                        if (std.mem.eql(u8, op, "pointer-index-read") and l.elements.len == 3) {
                            try self.writeExpressionTyped(writer, l.elements[1], ns_ctx, includes);
                            try writer.print("[", .{});
                            try self.writeExpressionTyped(writer, l.elements[2], ns_ctx, includes);
                            try writer.print("]", .{});
                            return;
                        }

                        // pointer-index-write!: ptr[index] = value
                        if (std.mem.eql(u8, op, "pointer-index-write!") and l.elements.len == 4 and l.type == .nil) {
                            try writer.print("(", .{});
                            try self.writeExpressionTyped(writer, l.elements[1], ns_ctx, includes);
                            try writer.print("[", .{});
                            try self.writeExpressionTyped(writer, l.elements[2], ns_ctx, includes);
                            try writer.print("] = ", .{});
                            try self.writeExpressionTyped(writer, l.elements[3], ns_ctx, includes);
                            try writer.print(")", .{});
                            return;
                        }

                        // printf: (printf format-string ...args) -> printf(format, args...)
                        if (std.mem.eql(u8, op, "printf") and l.type == .i32) {
                            try writer.print("printf(", .{});

                            // Emit all arguments (format string + variadic args)
                            for (l.elements[1..], 0..) |arg, i| {
                                if (i > 0) try writer.print(", ", .{});
                                try self.writeExpressionTyped(writer, arg, ns_ctx, includes);
                            }

                            try writer.print(")", .{});
                            return;
                        }
                    }
                }

                // Check for if expression (represented as 3-element list without 'if' symbol)
                // Pattern: [cond, then, else] where cond is typically a list (comparison/function call)
                if (l.elements.len == 3 and l.elements[0].* == .list) {
                    // This is likely an if expression (cond then else)
                    // Emit as C ternary: (cond ? then : else)
                    try writer.print("(", .{});
                    try self.writeExpressionTyped(writer, l.elements[0], ns_ctx, includes);
                    try writer.print(" ? ", .{});
                    try self.writeExpressionTyped(writer, l.elements[1], ns_ctx, includes);
                    try writer.print(" : ", .{});
                    try self.writeExpressionTyped(writer, l.elements[2], ns_ctx, includes);
                    try writer.print(")", .{});
                    return;
                }

                // Check for arithmetic/comparison/logical operators
                if (l.elements.len > 0 and l.elements[0].* == .symbol) {
                    const op = l.elements[0].symbol.name;

                    // Arithmetic operators: +, -, *, /, %
                    if (std.mem.eql(u8, op, "+") or std.mem.eql(u8, op, "-") or
                        std.mem.eql(u8, op, "*") or std.mem.eql(u8, op, "/") or std.mem.eql(u8, op, "%")) {
                        if (l.elements.len < 2) return Error.UnsupportedExpression;

                        // Handle unary minus
                        if (std.mem.eql(u8, op, "-") and l.elements.len == 2) {
                            try writer.print("(-(", .{});
                            try self.writeExpressionTyped(writer, l.elements[1], ns_ctx, includes);
                            try writer.print("))", .{});
                            return;
                        }

                        try writer.print("(", .{});
                        try self.writeExpressionTyped(writer, l.elements[1], ns_ctx, includes);
                        var i: usize = 2;
                        while (i < l.elements.len) : (i += 1) {
                            try writer.print(" {s} ", .{op});
                            try self.writeExpressionTyped(writer, l.elements[i], ns_ctx, includes);
                        }
                        try writer.print(")", .{});
                        return;
                    }

                    // Comparison operators
                    if (std.mem.eql(u8, op, "<") or std.mem.eql(u8, op, ">") or
                        std.mem.eql(u8, op, "<=") or std.mem.eql(u8, op, ">=") or
                        std.mem.eql(u8, op, "=") or std.mem.eql(u8, op, "!=")) {
                        if (l.elements.len != 3) return Error.UnsupportedExpression;
                        const c_op = if (std.mem.eql(u8, op, "=")) "==" else op;
                        try writer.print("(", .{});
                        try self.writeExpressionTyped(writer, l.elements[1], ns_ctx, includes);
                        try writer.print(" {s} ", .{c_op});
                        try self.writeExpressionTyped(writer, l.elements[2], ns_ctx, includes);
                        try writer.print(")", .{});
                        return;
                    }

                    // Logical operators
                    if (std.mem.eql(u8, op, "and") or std.mem.eql(u8, op, "or")) {
                        if (l.elements.len != 3) return Error.UnsupportedExpression;
                        const c_op = if (std.mem.eql(u8, op, "and")) "&&" else "||";
                        try writer.print("(", .{});
                        try self.writeExpressionTyped(writer, l.elements[1], ns_ctx, includes);
                        try writer.print(" {s} ", .{c_op});
                        try self.writeExpressionTyped(writer, l.elements[2], ns_ctx, includes);
                        try writer.print(")", .{});
                        return;
                    }

                    if (std.mem.eql(u8, op, "not")) {
                        if (l.elements.len != 2) return Error.UnsupportedExpression;
                        try writer.print("(!(", .{});
                        try self.writeExpressionTyped(writer, l.elements[1], ns_ctx, includes);
                        try writer.print("))", .{});
                        return;
                    }
                }

                // Check if this is a function or function pointer application
                // Function application has type .function or .pointer(.function) for the first element
                if (l.elements.len > 0) {
                    const first_elem = l.elements[0];
                    const first_type = first_elem.getType();
                    const is_function = first_type == .function or
                        (first_type == .pointer and first_type.pointer.* == .function);

                    if (is_function) {
                        // Emit function call
                        // For function pointers, we need: (*fn_ptr)(args...)
                        // For regular functions, we need: fn_name(args...)
                        const is_fn_ptr = first_type == .pointer;

                        if (is_fn_ptr) {
                            try writer.print("(*", .{});
                        }
                        try self.writeExpressionTyped(writer, first_elem, ns_ctx, includes);
                        if (is_fn_ptr) {
                            try writer.print(")", .{});
                        }

                        try writer.print("(", .{});
                        var i: usize = 1;
                        while (i < l.elements.len) : (i += 1) {
                            if (i > 1) try writer.print(", ", .{});
                            try self.writeExpressionTyped(writer, l.elements[i], ns_ctx, includes);
                        }
                        try writer.print(")", .{});
                        return;
                    }
                }

                return Error.UnsupportedExpression;
            },
            .int => |i| try writer.print("{d}", .{i.value}),
            .float => |f| try writer.print("{d}", .{f.value}),
            .string => |s| try writer.print("\"{s}\"", .{s.value}),
            .symbol => |sym| {
                // Check if this is pointer-null
                if (std.mem.eql(u8, sym.name, "pointer-null")) {
                    try writer.print("NULL", .{});
                    return;
                }

                // Check if this is an enum variant (has type enum_type)
                if (sym.type == .enum_type) {
                    // Enum variants like Color/Red become ENUM_VARIANT in C
                    const sanitized = try self.sanitizeIdentifier(sym.name);
                    defer self.allocator.*.free(sanitized);
                    try writer.print("{s}", .{sanitized});
                    return;
                }

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
            .nil => |n| {
                // Check if this is pointer-null (has pointer type)
                if (n.type == .pointer) {
                    try writer.print("NULL", .{});
                } else {
                    try writer.print("0", .{});
                }
            },
            else => return Error.UnsupportedExpression,
        }
    }

    // writeExpression has been removed - all code generation now uses writeExpressionTyped with the typed AST

    fn isComparisonOperator(_: *SimpleCCompiler, symbol: []const u8) bool {
        return std.mem.eql(u8, symbol, "<") or
            std.mem.eql(u8, symbol, ">") or
            std.mem.eql(u8, symbol, "<=") or
            std.mem.eql(u8, symbol, ">=") or
            std.mem.eql(u8, symbol, "=") or
            std.mem.eql(u8, symbol, "==") or
            std.mem.eql(u8, symbol, "!=");
    }

    fn isLogicalOperator(_: *SimpleCCompiler, symbol: []const u8) bool {
        return std.mem.eql(u8, symbol, "and") or
            std.mem.eql(u8, symbol, "or") or
            std.mem.eql(u8, symbol, "not");
    }

    fn isBitwiseOperator(_: *SimpleCCompiler, symbol: []const u8) bool {
        return std.mem.eql(u8, symbol, "bit-and") or
            std.mem.eql(u8, symbol, "bit-or") or
            std.mem.eql(u8, symbol, "bit-xor") or
            std.mem.eql(u8, symbol, "bit-not") or
            std.mem.eql(u8, symbol, "bit-shl") or
            std.mem.eql(u8, symbol, "bit-shr");
    }

    // Helper to emit array declarations which have special C syntax: type name[size]
    // For multi-dimensional arrays, we need to emit: type name[size1][size2]...
    fn emitArrayDecl(self: *SimpleCCompiler, writer: anytype, var_name: []const u8, var_type: Type, includes: *IncludeFlags) Error!void {
        if (var_type == .array) {
            // Get the base element type and collect all array dimensions
            var current_type = var_type;
            var dimensions: [16]usize = undefined; // Support up to 16 dimensions
            var dim_count: usize = 0;

            // Traverse nested array types to collect dimensions
            while (current_type == .array and dim_count < 16) {
                dimensions[dim_count] = current_type.array.size;
                dim_count += 1;
                current_type = current_type.array.element_type;
            }

            // Emit the base type
            const base_c_type = try self.cTypeFor(current_type, includes);
            try writer.print("{s} {s}", .{ base_c_type, var_name });

            // Emit all dimensions: [size1][size2]...
            var i: usize = 0;
            while (i < dim_count) : (i += 1) {
                try writer.print("[{d}]", .{dimensions[i]});
            }
        } else if (var_type == .pointer) {
            // Check if this is a pointer to a function
            const pointee = var_type.pointer.*;
            if (pointee == .function) {
                // Function pointer: return_type (*var_name)(params...)
                const fn_type = pointee.function;
                const return_type_str = try self.cTypeFor(fn_type.return_type, includes);
                try writer.print("{s} (*{s})(", .{ return_type_str, var_name });
                for (fn_type.param_types, 0..) |param_type, i| {
                    if (i > 0) try writer.print(", ", .{});
                    const param_type_str = try self.cTypeFor(param_type, includes);
                    try writer.print("{s}", .{param_type_str});
                }
                try writer.print(")", .{});
            } else {
                // Regular pointer
                const c_type = try self.cTypeFor(var_type, includes);
                try writer.print("{s} {s}", .{ c_type, var_name });
            }
        } else {
            const c_type = try self.cTypeFor(var_type, includes);
            try writer.print("{s} {s}", .{ c_type, var_name });
        }
    }

    // Helper function to write a type expression as C type syntax for casting
    // Handles: Int, U32, (Pointer T), (-> [Args...] ReturnType), etc.
    fn writeTypeAsCast(self: *SimpleCCompiler, writer: anytype, type_expr: *Value) Error!void {
        if (type_expr.isSymbol()) {
            const type_name = type_expr.symbol;
            const c_type = if (std.mem.eql(u8, type_name, "Int"))
                int_type_name
            else if (std.mem.eql(u8, type_name, "Float"))
                "double"
            else if (std.mem.eql(u8, type_name, "U8"))
                "uint8_t"
            else if (std.mem.eql(u8, type_name, "U16"))
                "uint16_t"
            else if (std.mem.eql(u8, type_name, "U32"))
                "uint32_t"
            else if (std.mem.eql(u8, type_name, "U64"))
                "uint64_t"
            else if (std.mem.eql(u8, type_name, "I8"))
                "int8_t"
            else if (std.mem.eql(u8, type_name, "I16"))
                "int16_t"
            else if (std.mem.eql(u8, type_name, "I32"))
                "int32_t"
            else if (std.mem.eql(u8, type_name, "I64"))
                "int64_t"
            else if (std.mem.eql(u8, type_name, "F32"))
                "float"
            else if (std.mem.eql(u8, type_name, "F64"))
                "double"
            else if (std.mem.eql(u8, type_name, "Bool"))
                "bool"
            else if (std.mem.eql(u8, type_name, "Nil"))
                "void"
            else
                type_name; // Custom type name
            try writer.print("{s}", .{c_type});
        } else if (type_expr.isList()) {
            var type_iter = type_expr.list.iterator();
            const first = type_iter.next() orelse return Error.UnsupportedType;
            if (!first.isSymbol()) return Error.UnsupportedType;

            if (std.mem.eql(u8, first.symbol, "Pointer")) {
                // (Pointer T) becomes T*
                const pointee = type_iter.next() orelse return Error.UnsupportedType;

                // Special case: (Pointer (-> ...)) needs parentheses: RetType (*)(Args...)
                if (pointee.isList()) {
                    var pointee_iter = pointee.list.iterator();
                    const pointee_first = pointee_iter.next() orelse return Error.UnsupportedType;
                    if (pointee_first.isSymbol() and std.mem.eql(u8, pointee_first.symbol, "->")) {
                        // Function pointer: (Pointer (-> [Args...] ReturnType))
                        // C syntax: ReturnType (*)(ArgType1, ArgType2, ...)

                        const args_vec = pointee_iter.next() orelse return Error.UnsupportedType;
                        const ret_type = pointee_iter.next() orelse return Error.UnsupportedType;

                        if (!args_vec.isVector()) return Error.UnsupportedType;

                        // Write return type
                        try self.writeTypeAsCast(writer, ret_type);
                        try writer.print(" (*)(", .{});

                        // Write parameter types
                        const args = args_vec.vector;
                        for (0..args.len()) |i| {
                            if (i > 0) try writer.print(", ", .{});
                            try self.writeTypeAsCast(writer, args.at(i));
                        }
                        try writer.print(")", .{});
                        return;
                    }
                }

                // Regular pointer: T*
                try self.writeTypeAsCast(writer, pointee);
                try writer.print("*", .{});
            } else if (std.mem.eql(u8, first.symbol, "->")) {
                // Bare function type (not pointer): (-> [Args...] ReturnType)
                // This is unusual in casts but handle it
                return Error.UnsupportedType;
            } else {
                return Error.UnsupportedType;
            }
        } else {
            return Error.UnsupportedType;
        }
    }

    fn cTypeFor(self: *SimpleCCompiler, type_info: Type, includes: *IncludeFlags) Error![]const u8 {
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
            .enum_type => |et| et.name,
            .function => |fn_type| {
                // Function pointer: ReturnType (*)(Param1Type, Param2Type, ...)
                const ret_c_type = try self.cTypeFor(fn_type.return_type, includes);

                // Build parameter list
                var params_list = std.ArrayList([]const u8){};
                defer params_list.deinit(self.allocator.*);

                for (fn_type.param_types) |param_type| {
                    const param_c_type = try self.cTypeFor(param_type, includes);
                    try params_list.append(self.allocator.*, param_c_type);
                }

                // Join with ", "
                const params = try std.mem.join(self.allocator.*, ", ", params_list.items);
                defer self.allocator.free(params);

                // Format final string
                return try std.fmt.allocPrint(self.allocator.*, "{s} (*)({s})", .{ ret_c_type, params });
            },
            .pointer => |pointee| {
                // Special handling for pointer to function type
                // In Lisp: (Pointer (-> [Args] Return)) represents a function pointer
                // In C: ReturnType (*)(Args) is a function pointer
                if (pointee.* == .function) {
                    const fn_type = pointee.function;
                    const ret_c_type = try self.cTypeFor(fn_type.return_type, includes);

                    // Build parameter list
                    var params_list = std.ArrayList([]const u8){};
                    defer params_list.deinit(self.allocator.*);

                    for (fn_type.param_types) |param_type| {
                        const param_c_type = try self.cTypeFor(param_type, includes);
                        try params_list.append(self.allocator.*, param_c_type);
                    }

                    const params = try std.mem.join(self.allocator.*, ", ", params_list.items);
                    defer self.allocator.free(params);

                    // Function pointer: ReturnType (*)(Params)
                    return try std.fmt.allocPrint(self.allocator.*, "{s} (*)({s})", .{ ret_c_type, params });
                } else {
                    const pointee_c_type = try self.cTypeFor(pointee.*, includes);
                    // Allocate string for "pointee_type*"
                    const ptr_type_str = try std.fmt.allocPrint(self.allocator.*, "{s}*", .{pointee_c_type});
                    return ptr_type_str;
                }
            },
            .c_string => "const char*",
            .void => "void",
            .nil => "void",
            .extern_type => |et| et.name,
            .extern_function => Error.UnsupportedType, // Can't have values of extern function type directly
            .array => Error.UnsupportedType, // Arrays need special declaration syntax, handled separately
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
            .enum_type => "%d", // Enums are represented as ints in C
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
        "#include <stdio.h>\n\n" ++ "typedef struct {\n" ++ "    long long answer;\n" ++ "} Namespace_my_app;\n\n" ++ "Namespace_my_app g_my_app;\n\n\n" ++ "void init_namespace_my_app(Namespace_my_app* ns) {\n" ++ "    ns->answer = 41;\n" ++ "}\n\n" ++ "int main() {\n" ++ "    init_namespace_my_app(&g_my_app);\n" ++ "    // namespace my.app\n" ++ "    printf(\"%lld\\n\", (g_my_app.answer + 1));\n" ++ "    return 0;\n" ++ "}\n";

    try std.testing.expectEqualStrings(expected, output);
}

test "simple c compiler fibonacci program with zig cc" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();

    var allocator = arena.allocator();
    var compiler = SimpleCCompiler.init(&allocator);

    const source =
        "(ns demo.core)\n" ++ "(def f0 (: Int) 0)\n" ++ "(def f1 (: Int) 1)\n" ++ "(def fib (: (-> [Int] Int)) (fn [n] (if (< n 2) n (+ (fib (- n 1)) (fib (- n 2))))))\n" ++ "(fib 10)";

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

    var run_child = std.process.Child.init(&.{exe_path}, std.testing.allocator);
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
    defer compiler.deinit();
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

        // Build command with linked libraries
        var cc_args = std.ArrayList([]const u8){};
        defer cc_args.deinit(allocator);
        try cc_args.appendSlice(allocator, &.{ "zig", "cc", "-dynamiclib", c_path, "-o", bundle_path });

        // Add compiler flags
        for (compiler.compiler_flags.items) |flag| {
            try cc_args.append(allocator, flag);
        }

        // Add linked libraries
        for (compiler.linked_libraries.items) |lib| {
            const lib_arg = try std.fmt.allocPrint(allocator, "-l{s}", .{lib});
            try cc_args.append(allocator, lib_arg);
        }

        var cc_child = std.process.Child.init(cc_args.items, allocator);
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

    // Build command with linked libraries
    var cc_args = std.ArrayList([]const u8){};
    defer cc_args.deinit(allocator);
    try cc_args.appendSlice(allocator, &.{ "zig", "cc", c_path, "-o", exe_path });

    // Add compiler flags
    for (compiler.compiler_flags.items) |flag| {
        try cc_args.append(allocator, flag);
    }

    // Add linked libraries
    for (compiler.linked_libraries.items) |lib| {
        const lib_arg = try std.fmt.allocPrint(allocator, "-l{s}", .{lib});
        try cc_args.append(allocator, lib_arg);
    }

    var cc_child = std.process.Child.init(cc_args.items, allocator);
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

    var run_child = std.process.Child.init(&.{exe_real_path}, allocator);
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
