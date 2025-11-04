const std = @import("std");
const mlir_lisp = @import("mlir_lisp");
const Tokenizer = mlir_lisp.Tokenizer;
const Reader = mlir_lisp.Reader;
const Parser = mlir_lisp.Parser;
const Printer = mlir_lisp.Printer;
const testing = std.testing;

test "printer - simple constant round-trip" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const allocator = arena.allocator();

    const input =
        \\(mlir
        \\  (operation
        \\    (name arith.constant)
        \\    (result-bindings [%c0])
        \\    (result-types i32)
        \\    (attributes { :value (#int 42) })
        \\    (location (#unknown))))
    ;

    // Parse
    var tok = Tokenizer.init(allocator, input);
    var r = try Reader.init(allocator, &tok);
    const value = try r.read();

    var p = Parser.init(allocator, "");
    var module = try p.parseModule(value);
    defer module.deinit();

    // Print
    var pr = Printer.init(allocator);
    defer pr.deinit();
    try pr.printModule(&module);

    const output = pr.getOutput();

    // Verify we have the expected structure
    try testing.expect(std.mem.indexOf(u8, output, "(operation") != null);
    try testing.expect(std.mem.indexOf(u8, output, "(name arith.constant)") != null);
    try testing.expect(std.mem.indexOf(u8, output, "(result-bindings [%c0])") != null);
    try testing.expect(std.mem.indexOf(u8, output, "(result-types i32)") != null);
    try testing.expect(std.mem.indexOf(u8, output, ":value") != null);
    try testing.expect(std.mem.indexOf(u8, output, "#int") != null);
}

test "printer - function with operations round-trip" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const allocator = arena.allocator();

    const input =
        \\(mlir
        \\  (operation
        \\    (name func.func)
        \\    (attributes {
        \\      :sym (#sym @add)
        \\      :type (!function (inputs i32 i32) (results i32))
        \\    })
        \\    (regions
        \\      (region
        \\        (block [^entry]
        \\          (arguments [ (: %x i32) (: %y i32) ])
        \\          (operation
        \\            (name arith.addi)
        \\            (result-bindings [%sum])
        \\            (operands %x %y)
        \\            (result-types i32))
        \\          (operation
        \\            (name func.return)
        \\            (operands %sum)))))))
    ;

    // Parse
    var tok = Tokenizer.init(allocator, input);
    var r = try Reader.init(allocator, &tok);
    const value = try r.read();

    var p = Parser.init(allocator, "");
    var module = try p.parseModule(value);
    defer module.deinit();

    // Print
    var pr = Printer.init(allocator);
    defer pr.deinit();
    try pr.printModule(&module);

    const output = pr.getOutput();

    // Verify structure
    try testing.expect(std.mem.indexOf(u8, output, "(name func.func)") != null);
    try testing.expect(std.mem.indexOf(u8, output, ":sym") != null);
    try testing.expect(std.mem.indexOf(u8, output, "#sym") != null);
    try testing.expect(std.mem.indexOf(u8, output, "@add") != null);
    try testing.expect(std.mem.indexOf(u8, output, "(regions") != null);
    try testing.expect(std.mem.indexOf(u8, output, "(region") != null);
    try testing.expect(std.mem.indexOf(u8, output, "(block") != null);
    try testing.expect(std.mem.indexOf(u8, output, "[^entry]") != null);
    try testing.expect(std.mem.indexOf(u8, output, "(arguments [(: %x i32) (: %y i32)])") != null);
    try testing.expect(std.mem.indexOf(u8, output, "(name arith.addi)") != null);
    try testing.expect(std.mem.indexOf(u8, output, "(result-bindings [%sum])") != null);
    try testing.expect(std.mem.indexOf(u8, output, "(operands %x %y)") != null);
}

test "printer - control flow with successors" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const allocator = arena.allocator();

    const input =
        \\(mlir
        \\  (operation
        \\    (name func.func)
        \\    (regions
        \\      (region
        \\        (block [^entry]
        \\          (arguments [ (: %cond i1) (: %x i32) (: %y i32) ])
        \\          (operation
        \\            (name cf.cond_br)
        \\            (operands %cond)
        \\            (successors
        \\              (successor ^then (%x))
        \\              (successor ^else (%y)))))
        \\        (block [^then]
        \\          (arguments [ (: %t i32) ])
        \\          (operation (name func.return) (operands %t)))
        \\        (block [^else]
        \\          (arguments [ (: %e i32) ])
        \\          (operation (name func.return) (operands %e)))))))
    ;

    // Parse
    var tok = Tokenizer.init(allocator, input);
    var r = try Reader.init(allocator, &tok);
    const value = try r.read();

    var p = Parser.init(allocator, "");
    var module = try p.parseModule(value);
    defer module.deinit();

    // Print
    var pr = Printer.init(allocator);
    defer pr.deinit();
    try pr.printModule(&module);

    const output = pr.getOutput();

    // Verify successors are preserved
    try testing.expect(std.mem.indexOf(u8, output, "(successors") != null);
    try testing.expect(std.mem.indexOf(u8, output, "(successor ^then (%x))") != null);
    try testing.expect(std.mem.indexOf(u8, output, "(successor ^else (%y))") != null);
    try testing.expect(std.mem.indexOf(u8, output, "[^then]") != null);
    try testing.expect(std.mem.indexOf(u8, output, "[^else]") != null);
}

test "printer - empty block arguments" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const allocator = arena.allocator();

    const input =
        \\(mlir
        \\  (operation
        \\    (name func.func)
        \\    (regions
        \\      (region
        \\        (block []
        \\          (arguments [])
        \\          (operation (name func.return)))))))
    ;

    // Parse
    var tok = Tokenizer.init(allocator, input);
    var r = try Reader.init(allocator, &tok);
    const value = try r.read();

    var p = Parser.init(allocator, "");
    var module = try p.parseModule(value);
    defer module.deinit();

    // Print
    var pr = Printer.init(allocator);
    defer pr.deinit();
    try pr.printModule(&module);

    const output = pr.getOutput();

    // Verify empty structures are preserved
    try testing.expect(std.mem.indexOf(u8, output, "[]") != null);
    try testing.expect(std.mem.indexOf(u8, output, "(arguments [])") != null);
}

test "printer - preserve semantic round-trip" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const allocator = arena.allocator();

    const input =
        \\(mlir
        \\  (operation
        \\    (name arith.constant)
        \\    (result-bindings [%c0])
        \\    (result-types i32)
        \\    (attributes { :value (#int 42) })))
    ;

    // Parse
    var tok = Tokenizer.init(allocator, input);
    var r = try Reader.init(allocator, &tok);
    const value = try r.read();

    var p = Parser.init(allocator, "");
    var module = try p.parseModule(value);
    defer module.deinit();

    // Print
    var pr = Printer.init(allocator);
    defer pr.deinit();
    try pr.printModule(&module);

    const output = pr.getOutput();

    // Parse the output again
    var tok2 = Tokenizer.init(allocator, output);
    var r2 = try Reader.init(allocator, &tok2);
    const value2 = try r2.read();

    var p2 = Parser.init(allocator, "");
    var module2 = try p2.parseModule(value2);
    defer module2.deinit();

    // Verify semantic equivalence
    try testing.expectEqual(module.operations.len, module2.operations.len);
    try testing.expectEqualStrings(module.operations[0].name, module2.operations[0].name);
    try testing.expectEqual(module.operations[0].result_bindings.len, module2.operations[0].result_bindings.len);
    try testing.expectEqualStrings(module.operations[0].result_bindings[0], module2.operations[0].result_bindings[0]);
}
