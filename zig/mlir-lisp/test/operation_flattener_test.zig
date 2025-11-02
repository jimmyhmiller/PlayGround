const std = @import("std");
const testing = std.testing;
const mlir_lisp = @import("mlir_lisp");
const OperationFlattener = mlir_lisp.OperationFlattener;
const Reader = mlir_lisp.Reader;
const Tokenizer = mlir_lisp.Tokenizer;
const Value = mlir_lisp.Value;

/// Helper to parse source code into a Value
fn parseSource(allocator: std.mem.Allocator, source: []const u8) !*Value {
    var tok = Tokenizer.init(allocator, source);
    var r = try Reader.init(allocator, &tok);
    return try r.read();
}

/// Helper to print a Value to a string for comparison
fn valueToString(allocator: std.mem.Allocator, value: *const Value) ![]const u8 {
    var list = std.ArrayList(u8){};
    defer list.deinit(allocator);
    try value.print(list.writer(allocator));
    return try list.toOwnedSlice(allocator);
}

test "operation flattener - single nested operand" {
    // Use arena allocator to avoid double-free issues
    var arena = std.heap.ArenaAllocator.init(testing.allocator);
    defer arena.deinit();
    const allocator = arena.allocator();

    const source =
        \\(block
        \\  (arguments [])
        \\  (operation
        \\    (name func.return)
        \\    (operands (operation
        \\               (name arith.constant)
        \\               (result-types [i64])
        \\               (attributes { :value (: 99 i64) })))))
    ;

    const value = try parseSource(allocator, source);
    var flattener = OperationFlattener.init(allocator);
    const flattened = try flattener.flattenModule(value);

    const result_str = try valueToString(allocator, flattened);

    // Should have generated %result_G0 for the constant
    try testing.expect(std.mem.indexOf(u8, result_str, "%result_G0") != null);

    // The arith.constant should appear before func.return
    const const_pos = std.mem.indexOf(u8, result_str, "arith.constant");
    const return_pos = std.mem.indexOf(u8, result_str, "func.return");
    try testing.expect(const_pos != null);
    try testing.expect(return_pos != null);
    try testing.expect(const_pos.? < return_pos.?);
}

test "operation flattener - multiple nested operands" {
    var arena = std.heap.ArenaAllocator.init(testing.allocator);
    defer arena.deinit();
    const allocator = arena.allocator();

    const source =
        \\(block
        \\  (arguments [])
        \\  (operation
        \\    (name arith.addi)
        \\    (result-bindings [%sum])
        \\    (result-types [i64])
        \\    (operands
        \\      (operation
        \\        (name arith.constant)
        \\        (result-types [i64])
        \\        (attributes { :value (: 10 i64) }))
        \\      (operation
        \\        (name arith.constant)
        \\        (result-types [i64])
        \\        (attributes { :value (: 32 i64) })))))
    ;

    const value = try parseSource(allocator, source);
    var flattener = OperationFlattener.init(allocator);
    const flattened = try flattener.flattenModule(value);

    const result_str = try valueToString(allocator, flattened);

    // Should have generated %result_G0 and %result_G1
    try testing.expect(std.mem.indexOf(u8, result_str, "%result_G0") != null);
    try testing.expect(std.mem.indexOf(u8, result_str, "%result_G1") != null);

    // Both constants should appear before addi
    const const1_pos = std.mem.indexOf(u8, result_str, "(: 10 i64)");
    const const2_pos = std.mem.indexOf(u8, result_str, "(: 32 i64)");
    const addi_pos = std.mem.indexOf(u8, result_str, "arith.addi");
    try testing.expect(const1_pos != null);
    try testing.expect(const2_pos != null);
    try testing.expect(addi_pos != null);
    try testing.expect(const1_pos.? < addi_pos.?);
    try testing.expect(const2_pos.? < addi_pos.?);

    // First constant should come before second constant (left-to-right order)
    try testing.expect(const1_pos.? < const2_pos.?);
}

test "operation flattener - mixed nested and value IDs" {
    var arena = std.heap.ArenaAllocator.init(testing.allocator);
    defer arena.deinit();
    const allocator = arena.allocator();

    const source =
        \\(block
        \\  (arguments [%x : i64])
        \\  (operation
        \\    (name arith.addi)
        \\    (result-bindings [%result])
        \\    (result-types [i64])
        \\    (operands
        \\      (operation
        \\        (name arith.constant)
        \\        (result-types [i64])
        \\        (attributes { :value (: 42 i64) }))
        \\      %x)))
    ;

    const value = try parseSource(allocator, source);
    var flattener = OperationFlattener.init(allocator);
    const flattened = try flattener.flattenModule(value);

    const result_str = try valueToString(allocator, flattened);

    // Should have generated %result_G0 for constant
    try testing.expect(std.mem.indexOf(u8, result_str, "%result_G0") != null);

    // %x should still be present
    try testing.expect(std.mem.indexOf(u8, result_str, "%x") != null);

    // Constant should appear before addi
    const const_pos = std.mem.indexOf(u8, result_str, "(: 42 i64)");
    const addi_pos = std.mem.indexOf(u8, result_str, "arith.addi");
    try testing.expect(const_pos != null);
    try testing.expect(addi_pos != null);
    try testing.expect(const_pos.? < addi_pos.?);
}

test "operation flattener - deeply nested operations (3 levels)" {
    var arena = std.heap.ArenaAllocator.init(testing.allocator);
    defer arena.deinit();
    const allocator = arena.allocator();

    const source =
        \\(block
        \\  (arguments [])
        \\  (operation
        \\    (name arith.muli)
        \\    (result-bindings [%product])
        \\    (result-types [i64])
        \\    (operands
        \\      (operation
        \\        (name arith.addi)
        \\        (result-types [i64])
        \\        (operands
        \\          (operation
        \\            (name arith.constant)
        \\            (result-types [i64])
        \\            (attributes { :value (: 5 i64) }))
        \\          (operation
        \\            (name arith.constant)
        \\            (result-types [i64])
        \\            (attributes { :value (: 3 i64) }))))
        \\      (operation
        \\        (name arith.constant)
        \\        (result-types [i64])
        \\        (attributes { :value (: 2 i64) })))))
    ;

    const value = try parseSource(allocator, source);
    var flattener = OperationFlattener.init(allocator);
    const flattened = try flattener.flattenModule(value);

    const result_str = try valueToString(allocator, flattened);

    // Should have generated 4 bindings: G0, G1 for constants, G2 for addi, G3 for last constant
    try testing.expect(std.mem.indexOf(u8, result_str, "%result_G0") != null);
    try testing.expect(std.mem.indexOf(u8, result_str, "%result_G1") != null);
    try testing.expect(std.mem.indexOf(u8, result_str, "%result_G2") != null);
    try testing.expect(std.mem.indexOf(u8, result_str, "%result_G3") != null);

    // Check evaluation order: 5, 3, (5+3), 2, ((5+3)*2)
    const const5_pos = std.mem.indexOf(u8, result_str, "(: 5 i64)");
    const const3_pos = std.mem.indexOf(u8, result_str, "(: 3 i64)");
    const addi_pos = std.mem.indexOf(u8, result_str, "arith.addi");
    const const2_pos = std.mem.indexOf(u8, result_str, "(: 2 i64)");
    const muli_pos = std.mem.indexOf(u8, result_str, "arith.muli");

    try testing.expect(const5_pos != null);
    try testing.expect(const3_pos != null);
    try testing.expect(addi_pos != null);
    try testing.expect(const2_pos != null);
    try testing.expect(muli_pos != null);

    try testing.expect(const5_pos.? < const3_pos.?);
    try testing.expect(const3_pos.? < addi_pos.?);
    try testing.expect(addi_pos.? < const2_pos.?);
    try testing.expect(const2_pos.? < muli_pos.?);
}

test "operation flattener - user-provided bindings are preserved" {
    var arena = std.heap.ArenaAllocator.init(testing.allocator);
    defer arena.deinit();
    const allocator = arena.allocator();

    const source =
        \\(block
        \\  (arguments [])
        \\  (operation
        \\    (name func.return)
        \\    (operands
        \\      (operation
        \\        (name arith.constant)
        \\        (result-bindings [%my_constant])
        \\        (result-types [i64])
        \\        (attributes { :value (: 99 i64) })))))
    ;

    const value = try parseSource(allocator, source);
    var flattener = OperationFlattener.init(allocator);
    const flattened = try flattener.flattenModule(value);

    const result_str = try valueToString(allocator, flattened);

    // Should use %my_constant, not generate a new binding
    try testing.expect(std.mem.indexOf(u8, result_str, "%my_constant") != null);
    // Should NOT generate %result_G0
    try testing.expect(std.mem.indexOf(u8, result_str, "%result_G0") == null);
}

test "operation flattener - pass through non-nested operations" {
    var arena = std.heap.ArenaAllocator.init(testing.allocator);
    defer arena.deinit();
    const allocator = arena.allocator();

    const source =
        \\(block
        \\  (arguments [%x : i64])
        \\  (operation
        \\    (name func.return)
        \\    (operands %x)))
    ;

    const value = try parseSource(allocator, source);
    var flattener = OperationFlattener.init(allocator);
    const flattened = try flattener.flattenModule(value);

    const result_str = try valueToString(allocator, flattened);

    // Should still have %x
    try testing.expect(std.mem.indexOf(u8, result_str, "%x") != null);
    // Should not generate any bindings
    try testing.expect(std.mem.indexOf(u8, result_str, "%result_G") == null);
}

test "operation flattener - nested operations with regions" {
    var arena = std.heap.ArenaAllocator.init(testing.allocator);
    defer arena.deinit();
    const allocator = arena.allocator();

    const source =
        \\(block
        \\  (arguments [])
        \\  (operation
        \\    (name scf.if)
        \\    (operands
        \\      (operation
        \\        (name arith.constant)
        \\        (result-types [i1])
        \\        (attributes { :value (: 1 i1) })))
        \\    (regions
        \\      (region
        \\        (block
        \\          (arguments [])
        \\          (operation
        \\            (name scf.yield)))))))
    ;

    const value = try parseSource(allocator, source);
    var flattener = OperationFlattener.init(allocator);
    const flattened = try flattener.flattenModule(value);

    const result_str = try valueToString(allocator, flattened);

    // Should have generated %result_G0 for the constant
    try testing.expect(std.mem.indexOf(u8, result_str, "%result_G0") != null);

    // The constant should appear before scf.if
    const const_pos = std.mem.indexOf(u8, result_str, "arith.constant");
    const if_pos = std.mem.indexOf(u8, result_str, "scf.if");
    try testing.expect(const_pos != null);
    try testing.expect(if_pos != null);
    try testing.expect(const_pos.? < if_pos.?);

    // The region should still be present
    try testing.expect(std.mem.indexOf(u8, result_str, "regions") != null);
    try testing.expect(std.mem.indexOf(u8, result_str, "scf.yield") != null);
}
