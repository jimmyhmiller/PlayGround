const std = @import("std");
const testing = std.testing;
const reader = @import("../src/reader.zig");
const Reader = reader.Reader;
const Value = reader.Value;
const ValueType = reader.ValueType;
const Tokenizer = @import("../src/tokenizer.zig").Tokenizer;
const macro_compressor = @import("../src/macro_compressor.zig");
const MacroExpander = @import("../src/macro_expander.zig").MacroExpander;
const builtin_macros = @import("../src/builtin_macros.zig");
const Printer = @import("../src/printer.zig").Printer;

/// Helper to parse a string into a Value
fn parseString(allocator: std.mem.Allocator, source: []const u8) !*Value {
    var tok = Tokenizer.init(allocator, source);
    var r = try Reader.init(allocator, &tok);
    return try r.read();
}

/// Helper to print a Value to string
fn valueToString(allocator: std.mem.Allocator, value: *Value) ![]const u8 {
    var printer = Printer.init(allocator);
    defer printer.deinit();
    try printer.print(value);
    return try printer.getString();
}

test "compress arith.addi to + macro" {
    const allocator = testing.allocator;

    const source =
        \\(operation
        \\  (name arith.addi)
        \\  (result-types i64)
        \\  (operands %a %b))
    ;

    const value = try parseString(allocator, source);
    defer {
        value.deinit(allocator);
        allocator.destroy(value);
    }

    const compressed = try macro_compressor.compressMacros(allocator, value);
    defer {
        compressed.deinit(allocator);
        allocator.destroy(compressed);
    }

    const result_str = try valueToString(allocator, compressed);
    defer allocator.free(result_str);

    // Expected: (+ (: i64) %a %b)
    try testing.expect(std.mem.indexOf(u8, result_str, "+") != null);
    try testing.expect(std.mem.indexOf(u8, result_str, "%a") != null);
    try testing.expect(std.mem.indexOf(u8, result_str, "%b") != null);
}

test "compress arith.muli to * macro" {
    const allocator = testing.allocator;

    const source =
        \\(operation
        \\  (name arith.muli)
        \\  (result-types i32)
        \\  (operands %x %y))
    ;

    const value = try parseString(allocator, source);
    defer {
        value.deinit(allocator);
        allocator.destroy(value);
    }

    const compressed = try macro_compressor.compressMacros(allocator, value);
    defer {
        compressed.deinit(allocator);
        allocator.destroy(compressed);
    }

    const result_str = try valueToString(allocator, compressed);
    defer allocator.free(result_str);

    // Expected: (* (: i32) %x %y)
    try testing.expect(std.mem.indexOf(u8, result_str, "*") != null);
    try testing.expect(std.mem.indexOf(u8, result_str, "%x") != null);
    try testing.expect(std.mem.indexOf(u8, result_str, "%y") != null);
}

test "compress arith.constant to constant macro" {
    const allocator = testing.allocator;

    const source =
        \\(operation
        \\  (name arith.constant)
        \\  (result-bindings [%c])
        \\  (result-types i64)
        \\  (attributes {:value (: 42 i64)}))
    ;

    const value = try parseString(allocator, source);
    defer {
        value.deinit(allocator);
        allocator.destroy(value);
    }

    const compressed = try macro_compressor.compressMacros(allocator, value);
    defer {
        compressed.deinit(allocator);
        allocator.destroy(compressed);
    }

    const result_str = try valueToString(allocator, compressed);
    defer allocator.free(result_str);

    // Expected: (constant %c (: 42 i64))
    try testing.expect(std.mem.indexOf(u8, result_str, "constant") != null);
    try testing.expect(std.mem.indexOf(u8, result_str, "%c") != null);
    try testing.expect(std.mem.indexOf(u8, result_str, "42") != null);
}

test "compress func.return to return macro" {
    const allocator = testing.allocator;

    const source =
        \\(operation
        \\  (name func.return)
        \\  (operands %result))
    ;

    const value = try parseString(allocator, source);
    defer {
        value.deinit(allocator);
        allocator.destroy(value);
    }

    const compressed = try macro_compressor.compressMacros(allocator, value);
    defer {
        compressed.deinit(allocator);
        allocator.destroy(compressed);
    }

    const result_str = try valueToString(allocator, compressed);
    defer allocator.free(result_str);

    // Expected: (return %result)
    try testing.expect(std.mem.indexOf(u8, result_str, "return") != null);
    try testing.expect(std.mem.indexOf(u8, result_str, "%result") != null);
}

test "compress generic operation to op macro" {
    const allocator = testing.allocator;

    const source =
        \\(operation
        \\  (name memref.load)
        \\  (result-bindings [%val])
        \\  (result-types f32)
        \\  (operands %memref %idx))
    ;

    const value = try parseString(allocator, source);
    defer {
        value.deinit(allocator);
        allocator.destroy(value);
    }

    const compressed = try macro_compressor.compressMacros(allocator, value);
    defer {
        compressed.deinit(allocator);
        allocator.destroy(compressed);
    }

    const result_str = try valueToString(allocator, compressed);
    defer allocator.free(result_str);

    // Expected: (op %val (: f32) (memref.load [%memref %idx]))
    try testing.expect(std.mem.indexOf(u8, result_str, "op") != null);
    try testing.expect(std.mem.indexOf(u8, result_str, "%val") != null);
    try testing.expect(std.mem.indexOf(u8, result_str, "memref.load") != null);
}

test "round-trip: expand then compress preserves semantics" {
    const allocator = testing.allocator;

    // Start with macro form
    const source = "(+ (: i64) %a %b)";

    const value = try parseString(allocator, source);
    defer {
        value.deinit(allocator);
        allocator.destroy(value);
    }

    // Expand macros
    var macro_expander = MacroExpander.init(allocator);
    defer macro_expander.deinit();
    try builtin_macros.registerBuiltinMacros(&macro_expander);

    const expanded = try macro_expander.expandAll(value);
    defer {
        expanded.deinit(allocator);
        allocator.destroy(expanded);
    }

    // Compress back to macros
    const compressed = try macro_compressor.compressMacros(allocator, expanded);
    defer {
        compressed.deinit(allocator);
        allocator.destroy(compressed);
    }

    const result_str = try valueToString(allocator, compressed);
    defer allocator.free(result_str);

    // Should have + macro back
    try testing.expect(std.mem.indexOf(u8, result_str, "+") != null);
    try testing.expect(std.mem.indexOf(u8, result_str, "%a") != null);
    try testing.expect(std.mem.indexOf(u8, result_str, "%b") != null);
}

test "compress nested operations" {
    const allocator = testing.allocator;

    const source =
        \\(operation
        \\  (name arith.addi)
        \\  (result-types i64)
        \\  (operands
        \\    (operation
        \\      (name arith.muli)
        \\      (result-types i64)
        \\      (operands %a %b))
        \\    %c))
    ;

    const value = try parseString(allocator, source);
    defer {
        value.deinit(allocator);
        allocator.destroy(value);
    }

    const compressed = try macro_compressor.compressMacros(allocator, value);
    defer {
        compressed.deinit(allocator);
        allocator.destroy(compressed);
    }

    const result_str = try valueToString(allocator, compressed);
    defer allocator.free(result_str);

    // Should have both + and * macros
    try testing.expect(std.mem.indexOf(u8, result_str, "+") != null);
    try testing.expect(std.mem.indexOf(u8, result_str, "*") != null);
}

test "compress operation without result bindings" {
    const allocator = testing.allocator;

    const source =
        \\(operation
        \\  (name memref.store)
        \\  (operands %val %memref %idx))
    ;

    const value = try parseString(allocator, source);
    defer {
        value.deinit(allocator);
        allocator.destroy(value);
    }

    const compressed = try macro_compressor.compressMacros(allocator, value);
    defer {
        compressed.deinit(allocator);
        allocator.destroy(compressed);
    }

    const result_str = try valueToString(allocator, compressed);
    defer allocator.free(result_str);

    // Expected: (op (memref.store [%val %memref %idx]))
    try testing.expect(std.mem.indexOf(u8, result_str, "op") != null);
    try testing.expect(std.mem.indexOf(u8, result_str, "memref.store") != null);
}
