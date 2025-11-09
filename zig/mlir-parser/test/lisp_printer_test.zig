//! Tests for MLIR to Lisp S-expression printer

const std = @import("std");
const mlir = @import("mlir_parser");

test "lisp printer - simple constant" {
    const source = "%0 = \"arith.constant\"() <{value = 42 : i32}> : () -> i32";

    var module = try mlir.parse(std.testing.allocator, source);
    defer module.deinit();

    const result = try mlir.printLisp(std.testing.allocator, module);
    defer std.testing.allocator.free(result);

    // Check structure
    try std.testing.expect(std.mem.indexOf(u8, result, "(mlir") != null);
    try std.testing.expect(std.mem.indexOf(u8, result, "(operation") != null);
    try std.testing.expect(std.mem.indexOf(u8, result, "(name arith.constant)") != null);
    try std.testing.expect(std.mem.indexOf(u8, result, "(result-bindings [%0])") != null);
    try std.testing.expect(std.mem.indexOf(u8, result, "(result-types i32)") != null);

    // Check attributes - should have typed literal
    try std.testing.expect(std.mem.indexOf(u8, result, ":value") != null);
    try std.testing.expect(std.mem.indexOf(u8, result, "(: 42 i32)") != null);

    std.debug.print("\n=== Simple Constant Output ===\n{s}\n", .{result});
}

test "lisp printer - arithmetic operation with operands" {
    const source =
        \\%0 = "arith.constant"() <{value = 42 : i32}> : () -> i32
        \\%1 = "arith.constant"() <{value = 13 : i32}> : () -> i32
        \\%2 = "arith.addi"(%0, %1) : (i32, i32) -> i32
    ;

    var module = try mlir.parse(std.testing.allocator, source);
    defer module.deinit();

    const result = try mlir.printLisp(std.testing.allocator, module);
    defer std.testing.allocator.free(result);

    // Check that we have three operations
    const op_count = std.mem.count(u8, result, "(operation");
    try std.testing.expectEqual(@as(usize, 3), op_count);

    // Check addi operation
    try std.testing.expect(std.mem.indexOf(u8, result, "(name arith.addi)") != null);
    try std.testing.expect(std.mem.indexOf(u8, result, "(operands %0 %1)") != null);
    try std.testing.expect(std.mem.indexOf(u8, result, "(result-bindings [%2])") != null);

    std.debug.print("\n=== Arithmetic Operation Output ===\n{s}\n", .{result});
}

test "lisp printer - module wrapper" {
    const source =
        \\"builtin.module"() ({
        \\  %0 = "arith.constant"() <{value = 42 : i32}> : () -> i32
        \\}) : () -> ()
    ;

    var module = try mlir.parse(std.testing.allocator, source);
    defer module.deinit();

    const result = try mlir.printLisp(std.testing.allocator, module);
    defer std.testing.allocator.free(result);

    // Check structure
    try std.testing.expect(std.mem.indexOf(u8, result, "(mlir") != null);
    try std.testing.expect(std.mem.indexOf(u8, result, "(name builtin.module)") != null);
    try std.testing.expect(std.mem.indexOf(u8, result, "(regions") != null);
    try std.testing.expect(std.mem.indexOf(u8, result, "(region") != null);

    std.debug.print("\n=== Module Wrapper Output ===\n{s}\n", .{result});
}

test "lisp printer - control flow with successors" {
    const source =
        \\"test.op"() ({
        \\^bb0:
        \\  %0 = "arith.constant"() <{value = 1 : i32}> : () -> i32
        \\  %1 = "arith.constant"() <{value = 0 : i32}> : () -> i32
        \\  %2 = "arith.cmpi"(%0, %1) <{predicate = 0 : i64}> : (i32, i32) -> i1
        \\  "test.cond_br"(%2)[^bb1, ^bb2] : (i1) -> ()
        \\^bb1:
        \\  "test.return"() : () -> ()
        \\^bb2:
        \\  "test.return"() : () -> ()
        \\}) : () -> ()
    ;

    var module = try mlir.parse(std.testing.allocator, source);
    defer module.deinit();

    const result = try mlir.printLisp(std.testing.allocator, module);
    defer std.testing.allocator.free(result);

    // Check for successors
    try std.testing.expect(std.mem.indexOf(u8, result, "(successors") != null);
    try std.testing.expect(std.mem.indexOf(u8, result, "(successor ^bb1)") != null);
    try std.testing.expect(std.mem.indexOf(u8, result, "(successor ^bb2)") != null);

    // Check for blocks
    try std.testing.expect(std.mem.indexOf(u8, result, "(block [^bb0]") != null);
    try std.testing.expect(std.mem.indexOf(u8, result, "(block [^bb1]") != null);
    try std.testing.expect(std.mem.indexOf(u8, result, "(block [^bb2]") != null);

    std.debug.print("\n=== Control Flow Output ===\n{s}\n", .{result});
}

test "lisp printer - nested regions (scf.if)" {
    const source =
        \\"builtin.module"() ({
        \\  %0 = "arith.constant"() <{value = true}> : () -> i1
        \\  %1 = "scf.if"(%0) ({
        \\    %3 = "arith.constant"() <{value = 42 : i32}> : () -> i32
        \\    "scf.yield"(%3) : (i32) -> ()
        \\  }, {
        \\    %2 = "arith.constant"() <{value = 0 : i32}> : () -> i32
        \\    "scf.yield"(%2) : (i32) -> ()
        \\  }) : (i1) -> i32
        \\}) : () -> ()
    ;

    var module = try mlir.parse(std.testing.allocator, source);
    defer module.deinit();

    const result = try mlir.printLisp(std.testing.allocator, module);
    defer std.testing.allocator.free(result);

    // Check for multiple regions
    const region_count = std.mem.count(u8, result, "(region");
    try std.testing.expect(region_count >= 3); // Module region + 2 scf.if regions

    // Check scf.if structure
    try std.testing.expect(std.mem.indexOf(u8, result, "(name scf.if)") != null);
    try std.testing.expect(std.mem.indexOf(u8, result, "(name scf.yield)") != null);

    std.debug.print("\n=== Nested Regions Output ===\n{s}\n", .{result});
}

test "lisp printer - types" {
    const source =
        \\%0 = "test.op"() : () -> i32
        \\%1 = "test.op"() : () -> f64
        \\%2 = "test.op"() : () -> index
    ;

    var module = try mlir.parse(std.testing.allocator, source);
    defer module.deinit();

    const result = try mlir.printLisp(std.testing.allocator, module);
    defer std.testing.allocator.free(result);

    // Check type outputs
    try std.testing.expect(std.mem.indexOf(u8, result, "(result-types i32)") != null);
    try std.testing.expect(std.mem.indexOf(u8, result, "(result-types f64)") != null);
    try std.testing.expect(std.mem.indexOf(u8, result, "(result-types index)") != null);

    std.debug.print("\n=== Types Output ===\n{s}\n", .{result});
}

test "lisp printer - multiple results" {
    const source = "%0, %1 = \"test.op\"() : () -> (i32, i32)";

    var module = try mlir.parse(std.testing.allocator, source);
    defer module.deinit();

    const result = try mlir.printLisp(std.testing.allocator, module);
    defer std.testing.allocator.free(result);

    // Check multiple result bindings
    try std.testing.expect(std.mem.indexOf(u8, result, "(result-bindings [%0 %1])") != null);
    try std.testing.expect(std.mem.indexOf(u8, result, "(result-types i32 i32)") != null);

    std.debug.print("\n=== Multiple Results Output ===\n{s}\n", .{result});
}
