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

        var prelude = std.ArrayList(u8){};
        defer prelude.deinit(self.allocator.*);
        var body = std.ArrayList(u8){};
        defer body.deinit(self.allocator.*);

        const prelude_writer = prelude.writer(self.allocator.*);
        const body_writer = body.writer(self.allocator.*);

        var includes = IncludeFlags{};

        for (expressions.items, 0..) |expr, idx| {
            try self.emitTopLevel(prelude_writer, body_writer, expr, report.typed.items[idx], &checker, &includes);
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

        if (prelude.items.len > 0) {
            try output.appendSlice(self.allocator.*, prelude.items);
            try output.appendSlice(self.allocator.*, "\n");
        }

        switch (target) {
            .executable => {
                try output.appendSlice(self.allocator.*, "int main() {\n");
                if (body.items.len > 0) {
                    try output.appendSlice(self.allocator.*, body.items);
                }
                try output.appendSlice(self.allocator.*, "    return 0;\n}\n");
            },
            .bundle => {
                try output.appendSlice(self.allocator.*, "void lisp_main(void) {\n");
                if (body.items.len > 0) {
                    try output.appendSlice(self.allocator.*, body.items);
                }
                try output.appendSlice(self.allocator.*, "}\n");
            },
        }

        return output.toOwnedSlice(self.allocator.*);
    }

    fn emitTopLevel(self: *SimpleCCompiler, def_writer: anytype, body_writer: anytype, expr: *Value, typed: *TypedValue, checker: *TypeChecker, includes: *IncludeFlags) Error!void {
        switch (expr.*) {
            .namespace => |ns| {
                try body_writer.print("    // namespace {s}\n", .{ns.name});
            },
            .list => |list| {
                var iter = list.iterator();
                const head_val = iter.next() orelse {
                    try self.emitPrintStatement(body_writer, expr, typed, includes);
                    return;
                };

                if (!head_val.isSymbol()) {
                    try self.emitPrintStatement(body_writer, expr, typed, includes);
                    return;
                }

                const head = head_val.symbol;
                if (std.mem.eql(u8, head, "def")) {
                    try self.emitDefinition(def_writer, expr, checker, includes);
                    return;
                }

                try self.emitPrintStatement(body_writer, expr, typed, includes);
            },
            else => {
                try self.emitPrintStatement(body_writer, expr, typed, includes);
            },
        }
    }

    fn emitDefinition(self: *SimpleCCompiler, def_writer: anytype, list_expr: *Value, checker: *TypeChecker, includes: *IncludeFlags) Error!void {
        var iter = list_expr.list.iterator();
        _ = iter.next(); // Skip 'def'

        const name_val = iter.next() orelse return Error.InvalidDefinition;
        if (!name_val.isSymbol()) return Error.InvalidDefinition;
        const name = name_val.symbol;

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

        try def_writer.print("{s} {s} = ", .{ c_type, name });
        self.writeExpression(def_writer, value_expr) catch |err| {
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

    fn emitFunctionDefinition(self: *SimpleCCompiler, def_writer: anytype, name: []const u8, fn_expr: *Value, fn_type: Type, includes: *IncludeFlags) Error!void {
        if (fn_type != .function) return Error.UnsupportedType;

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

        try def_writer.print("static {s} {s}(", .{ return_type_str, name });
        index = 0;
        while (index < params_vec.len()) : (index += 1) {
            const param_val = params_vec.at(index);
            if (index > 0) {
                try def_writer.print(", ", .{});
            }
            try def_writer.print("{s} {s}", .{ param_type_buf[index], param_val.symbol });
        }
        try def_writer.writeAll(") {\n");
        try def_writer.print("    return ", .{});
        self.writeExpression(def_writer, body_expr) catch |err| {
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

    fn emitPrintStatement(self: *SimpleCCompiler, body_writer: anytype, expr: *Value, typed: *TypedValue, includes: *IncludeFlags) Error!void {
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
        self.writeExpression(body_writer, expr) catch |err| {
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

    fn writeExpression(self: *SimpleCCompiler, writer: anytype, expr: *Value) Error!void {
        switch (expr.*) {
            .int => |i| try writer.print("{d}", .{i}),
            .float => |f| try writer.print("{d}", .{f}),
            .string => |s| try writer.print("\"{s}\"", .{s}),
            .symbol => |sym| try writer.print("{s}", .{sym}),
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
                    try self.writeExpression(writer, condition);
                    try writer.print(") ? (", .{});
                    try self.writeExpression(writer, then_expr);
                    try writer.print(") : (", .{});
                    try self.writeExpression(writer, else_expr);
                    try writer.print("))", .{});
                    return;
                }

                if (self.isComparisonOperator(op)) {
                    const left = iter.next() orelse return Error.MissingOperand;
                    const right = iter.next() orelse return Error.MissingOperand;
                    if (iter.next() != null) return Error.UnsupportedExpression;

                    try writer.print("(", .{});
                    try self.writeExpression(writer, left);
                    try writer.print(" {s} ", .{op});
                    try self.writeExpression(writer, right);
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
                        try self.writeExpression(writer, operands[0]);
                        try writer.writeAll("))");
                        return;
                    }

                    if (count == 1) return Error.MissingOperand;

                    try writer.writeAll("(");
                    try self.writeExpression(writer, operands[0]);

                    var idx: usize = 1;
                    while (idx < count) : (idx += 1) {
                        try writer.print(" {s} ", .{op});
                        try self.writeExpression(writer, operands[idx]);
                    }

                    try writer.writeAll(")");
                    return;
                }

                try writer.print("{s}(", .{op});
                var is_first = true;
                while (iter.next()) |arg| {
                    if (!is_first) {
                        try writer.print(", ", .{});
                    }
                    try self.writeExpression(writer, arg);
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
