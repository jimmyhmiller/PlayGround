const std = @import("std");
const mlir_lisp = @import("mlir_lisp");
const macro_expander = mlir_lisp.macro_expander;
const MacroExpander = macro_expander.MacroExpander;
const builtin_macros = mlir_lisp.builtin_macros;
const reader = mlir_lisp;
const Reader = mlir_lisp.Reader;
const Value = mlir_lisp.Value;
const ValueType = mlir_lisp.ValueType;
const tokenizer = mlir_lisp;
const Tokenizer = mlir_lisp.Tokenizer;
const vector = mlir_lisp;
const PersistentVector = mlir_lisp.PersistentVector;
const linked_list = mlir_lisp;
const PersistentLinkedList = mlir_lisp.PersistentLinkedList;

// ============================================================================
// Helper Functions
// ============================================================================

fn readString(allocator: std.mem.Allocator, source: []const u8) !*Value {
    var tok = Tokenizer.init(allocator, source);
    var r = try Reader.init(allocator, &tok);
    return try r.read();
}

fn valueToString(allocator: std.mem.Allocator, value: *Value) ![]const u8 {
    var buffer = std.ArrayList(u8){};
    defer buffer.deinit(allocator);
    try printValue(allocator, &buffer, value);
    return try buffer.toOwnedSlice(allocator);
}

fn printValue(allocator: std.mem.Allocator, buffer: *std.ArrayList(u8), value: *Value) !void {
    switch (value.type) {
        .identifier, .number, .string, .value_id, .block_id, .symbol, .keyword, .type => {
            try buffer.appendSlice(allocator, value.data.atom);
        },
        .true_lit => try buffer.appendSlice(allocator, "true"),
        .false_lit => try buffer.appendSlice(allocator, "false"),
        .list => {
            try buffer.append(allocator, '(');
            const list = value.data.list;
            for (list.slice(), 0..) |elem, i| {
                if (i > 0) try buffer.append(allocator, ' ');
                try printValue(allocator, buffer, elem);
            }
            try buffer.append(allocator, ')');
        },
        .vector => {
            try buffer.append(allocator, '[');
            const vec = value.data.vector;
            for (vec.slice(), 0..) |elem, i| {
                if (i > 0) try buffer.append(allocator, ' ');
                try printValue(allocator, buffer, elem);
            }
            try buffer.append(allocator, ']');
        },
        .map => {
            try buffer.append(allocator, '{');
            const map = value.data.map;
            const slice = map.slice();
            var i: usize = 0;
            while (i < slice.len) : (i += 2) {
                if (i > 0) try buffer.append(allocator, ' ');
                try printValue(allocator, buffer, slice[i]);
                try buffer.append(allocator, ' ');
                if (i + 1 < slice.len) {
                    try printValue(allocator, buffer, slice[i + 1]);
                }
            }
            try buffer.append(allocator, '}');
        },
        .attr_expr => {
            try buffer.append(allocator, '#');
            try printValue(allocator, buffer, value.data.attr_expr);
        },
        .has_type => {
            try buffer.appendSlice(allocator, "(: ");
            try printValue(allocator, buffer, value.data.has_type.value);
            try buffer.append(allocator, ' ');
            try printValue(allocator, buffer, value.data.has_type.type_expr);
            try buffer.append(allocator, ')');
        },
        .function_type => {
            try buffer.appendSlice(allocator, "(!function ");
            try buffer.append(allocator, '(');
            const inputs = value.data.function_type.inputs;
            for (inputs.slice(), 0..) |elem, i| {
                if (i > 0) try buffer.append(allocator, ' ');
                try printValue(allocator, buffer, elem);
            }
            try buffer.appendSlice(allocator, ") (");
            const results = value.data.function_type.results;
            for (results.slice(), 0..) |elem, i| {
                if (i > 0) try buffer.append(allocator, ' ');
                try printValue(allocator, buffer, elem);
            }
            try buffer.appendSlice(allocator, "))");
        },
    }
}

// ============================================================================
// Tests
// ============================================================================

test "macro expander - basic expansion" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const allocator = arena.allocator();

    var expander = MacroExpander.init(allocator);
    defer expander.deinit();

    // Register built-in macros
    try builtin_macros.registerBuiltinMacros(&expander);

    // Read input: (if true 1 0)
    const input = try readString(allocator, "(if true 1 0)");

    // Expand
    const expanded = try expander.expandAll(input);

    // Convert to string for comparison
    const result_str = try valueToString(allocator, expanded);

    // Should expand to something like (scf.if true (region [...]) (region [...]))
    try std.testing.expect(std.mem.indexOf(u8, result_str, "scf.if") != null);
    try std.testing.expect(std.mem.indexOf(u8, result_str, "region") != null);
}

test "macro expander - when macro" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const allocator = arena.allocator();

    var expander = MacroExpander.init(allocator);
    defer expander.deinit();

    try builtin_macros.registerBuiltinMacros(&expander);

    const input = try readString(allocator, "(when condition body)");

    const expanded = try expander.expandAll(input);

    const result_str = try valueToString(allocator, expanded);

    // Should expand to scf.if with empty else branch
    try std.testing.expect(std.mem.indexOf(u8, result_str, "scf.if") != null);
    try std.testing.expect(std.mem.indexOf(u8, result_str, "region") != null);
}

test "macro expander - unless macro" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const allocator = arena.allocator();

    var expander = MacroExpander.init(allocator);
    defer expander.deinit();

    try builtin_macros.registerBuiltinMacros(&expander);

    const input = try readString(allocator, "(unless condition body)");

    const expanded = try expander.expandAll(input);

    const result_str = try valueToString(allocator, expanded);

    // Should expand to scf.if with swapped branches
    try std.testing.expect(std.mem.indexOf(u8, result_str, "scf.if") != null);
}

test "macro expander - no expansion for non-macro" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const allocator = arena.allocator();

    var expander = MacroExpander.init(allocator);
    defer expander.deinit();

    const input = try readString(allocator, "(some-function arg1 arg2)");

    const expanded = try expander.expandAll(input);

    // Should be the same (pointer equality)
    try std.testing.expect(expanded == input);
}

test "macro expander - nested expansion" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const allocator = arena.allocator();

    var expander = MacroExpander.init(allocator);
    defer expander.deinit();

    try builtin_macros.registerBuiltinMacros(&expander);

    // Nested: (if cond1 (when cond2 body) else)
    const input = try readString(allocator, "(if cond1 (when cond2 body) else)");

    const expanded = try expander.expandAll(input);

    const result_str = try valueToString(allocator, expanded);

    // Both if and when should be expanded
    const scf_count = std.mem.count(u8, result_str, "scf.if");
    try std.testing.expect(scf_count >= 2); // At least 2 scf.if calls
}

test "macro expander - gensym uniqueness" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const allocator = arena.allocator();

    var expander = MacroExpander.init(allocator);
    defer expander.deinit();

    const sym1 = try expander.gensym("var");
    const sym2 = try expander.gensym("var");
    const sym3 = try expander.gensym("var");

    // All should be unique
    try std.testing.expect(!std.mem.eql(u8, sym1, sym2));
    try std.testing.expect(!std.mem.eql(u8, sym2, sym3));
    try std.testing.expect(!std.mem.eql(u8, sym1, sym3));

    // Should follow pattern
    try std.testing.expectEqualStrings("var_G0", sym1);
    try std.testing.expectEqualStrings("var_G1", sym2);
    try std.testing.expectEqualStrings("var_G2", sym3);
}

test "macro expander - iterative expansion" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const allocator = arena.allocator();

    var expander = MacroExpander.init(allocator);
    defer expander.deinit();

    // Create macro A that expands to macro B
    const MacroA = struct {
        fn expand(
            alloc: std.mem.Allocator,
            args: *const PersistentLinkedList(*Value),
        ) anyerror!*Value {
            _ = args;
            // Return (macro-b arg)
            var vec = PersistentVector(*Value).init(alloc, null);
            const ident = try alloc.create(Value);
            ident.* = Value{
                .type = .identifier,
                .data = .{ .atom = "macro-b" },
            };
            vec = try vec.push(ident);
            const arg = try alloc.create(Value);
            arg.* = Value{
                .type = .identifier,
                .data = .{ .atom = "arg" },
            };
            vec = try vec.push(arg);
            const list = try alloc.create(Value);
            list.* = Value{
                .type = .list,
                .data = .{ .list = vec },
            };
            return list;
        }
    };

    // Create macro B that expands to final form
    const MacroB = struct {
        fn expand(
            alloc: std.mem.Allocator,
            args: *const PersistentLinkedList(*Value),
        ) anyerror!*Value {
            _ = args;
            // Return (final result)
            var vec = PersistentVector(*Value).init(alloc, null);
            const ident = try alloc.create(Value);
            ident.* = Value{
                .type = .identifier,
                .data = .{ .atom = "final" },
            };
            vec = try vec.push(ident);
            const result = try alloc.create(Value);
            result.* = Value{
                .type = .identifier,
                .data = .{ .atom = "result" },
            };
            vec = try vec.push(result);
            const list = try alloc.create(Value);
            list.* = Value{
                .type = .list,
                .data = .{ .list = vec },
            };
            return list;
        }
    };

    try expander.registerMacro("macro-a", &MacroA.expand);
    try expander.registerMacro("macro-b", &MacroB.expand);

    const input = try readString(allocator, "(macro-a x)");

    const expanded = try expander.expandAll(input);

    const result_str = try valueToString(allocator, expanded);

    // Should expand through both macros to final form
    try std.testing.expect(std.mem.indexOf(u8, result_str, "final") != null);
    try std.testing.expect(std.mem.indexOf(u8, result_str, "result") != null);
    try std.testing.expect(std.mem.indexOf(u8, result_str, "macro-a") == null);
    try std.testing.expect(std.mem.indexOf(u8, result_str, "macro-b") == null);
}

test "macro expander - argument passing" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const allocator = arena.allocator();

    var expander = MacroExpander.init(allocator);
    defer expander.deinit();

    // Create a macro that echoes its arguments
    const EchoMacro = struct {
        fn expand(
            alloc: std.mem.Allocator,
            args: *const PersistentLinkedList(*Value),
        ) anyerror!*Value {
            // Return (echo-result arg1 arg2 ...)
            var vec = PersistentVector(*Value).init(alloc, null);
            const ident = try alloc.create(Value);
            ident.* = Value{
                .type = .identifier,
                .data = .{ .atom = "echo-result" },
            };
            vec = try vec.push(ident);

            // Add all arguments
            var iter = args.iterator();
            while (iter.next()) |arg| {
                vec = try vec.push(arg);
            }

            const list = try alloc.create(Value);
            list.* = Value{
                .type = .list,
                .data = .{ .list = vec },
            };
            return list;
        }
    };

    try expander.registerMacro("echo", &EchoMacro.expand);

    const input = try readString(allocator, "(echo a b c)");

    const expanded = try expander.expandAll(input);

    const result_str = try valueToString(allocator, expanded);

    // Should have all arguments
    try std.testing.expect(std.mem.indexOf(u8, result_str, "echo-result") != null);
    try std.testing.expect(std.mem.indexOf(u8, result_str, "a") != null);
    try std.testing.expect(std.mem.indexOf(u8, result_str, "b") != null);
    try std.testing.expect(std.mem.indexOf(u8, result_str, "c") != null);
}
