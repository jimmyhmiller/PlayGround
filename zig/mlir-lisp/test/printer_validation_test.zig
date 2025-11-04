const std = @import("std");
const mlir_lisp = @import("mlir_lisp");
const Tokenizer = mlir_lisp.Tokenizer;
const Reader = mlir_lisp.Reader;
const Parser = mlir_lisp.Parser;
const Printer = mlir_lisp.Printer;
const testing = std.testing;

// VALIDATION TESTS - Testing edge cases and scenarios not covered in original tests

test "printer validation - multiple operations in module" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const allocator = arena.allocator();

    const input =
        \\(mlir
        \\  (operation
        \\    (name arith.constant)
        \\    (result-bindings [%c1])
        \\    (result-types i32)
        \\    (attributes { :value (#int 10) }))
        \\  (operation
        \\    (name arith.constant)
        \\    (result-bindings [%c2])
        \\    (result-types i32)
        \\    (attributes { :value (#int 20) }))
        \\  (operation
        \\    (name arith.addi)
        \\    (result-bindings [%sum])
        \\    (operands %c1 %c2)
        \\    (result-types i32)))
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

    // Parse again for round-trip validation
    var tok2 = Tokenizer.init(allocator, output);
    var r2 = try Reader.init(allocator, &tok2);
    const value2 = try r2.read();

    var p2 = Parser.init(allocator, "");
    var module2 = try p2.parseModule(value2);
    defer module2.deinit();

    // Validate multiple operations preserved
    try testing.expectEqual(@as(usize, 3), module.operations.len);
    try testing.expectEqual(@as(usize, 3), module2.operations.len);
    try testing.expectEqualStrings("arith.constant", module2.operations[0].name);
    try testing.expectEqualStrings("arith.constant", module2.operations[1].name);
    try testing.expectEqualStrings("arith.addi", module2.operations[2].name);
}

test "printer validation - nested regions and blocks" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const allocator = arena.allocator();

    const input =
        \\(mlir
        \\  (operation
        \\    (name scf.if)
        \\    (operands %cond)
        \\    (regions
        \\      (region
        \\        (block []
        \\          (arguments [])
        \\          (operation
        \\            (name scf.if)
        \\            (operands %inner_cond)
        \\            (regions
        \\              (region
        \\                (block []
        \\                  (arguments [])
        \\                  (operation (name arith.constant)))))))))))
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

    // Parse again
    var tok2 = Tokenizer.init(allocator, output);
    var r2 = try Reader.init(allocator, &tok2);
    const value2 = try r2.read();

    var p2 = Parser.init(allocator, "");
    var module2 = try p2.parseModule(value2);
    defer module2.deinit();

    // Validate nested structure preserved
    try testing.expectEqual(@as(usize, 1), module2.operations[0].regions.len);
    try testing.expectEqual(@as(usize, 1), module2.operations[0].regions[0].blocks[0].operations.len);
    try testing.expectEqual(@as(usize, 1), module2.operations[0].regions[0].blocks[0].operations[0].regions.len);
}

test "printer validation - multiple result bindings and types" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const allocator = arena.allocator();

    const input =
        \\(mlir
        \\  (operation
        \\    (name custom.multi_result)
        \\    (result-bindings [%r1 %r2 %r3])
        \\    (result-types i32 i64 f32)
        \\    (operands %x %y %z)))
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

    // Parse again
    var tok2 = Tokenizer.init(allocator, output);
    var r2 = try Reader.init(allocator, &tok2);
    const value2 = try r2.read();

    var p2 = Parser.init(allocator, "");
    var module2 = try p2.parseModule(value2);
    defer module2.deinit();

    // Validate multiple results preserved
    try testing.expectEqual(@as(usize, 3), module2.operations[0].result_bindings.len);
    try testing.expectEqual(@as(usize, 3), module2.operations[0].result_types.len);
    try testing.expectEqualStrings("%r1", module2.operations[0].result_bindings[0]);
    try testing.expectEqualStrings("%r2", module2.operations[0].result_bindings[1]);
    try testing.expectEqualStrings("%r3", module2.operations[0].result_bindings[2]);
}

test "printer validation - complex attribute types" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const allocator = arena.allocator();

    const input =
        \\(mlir
        \\  (operation
        \\    (name test.op)
        \\    (attributes {
        \\      :int_attr (#int 42)
        \\      :float_attr (#float 3.14)
        \\      :string_attr (#str "hello")
        \\      :bool_attr (#bool true)
        \\      :sym_attr (#sym @symbol)
        \\      :array_attr (#array [1 2 3])
        \\      :nested_attr (#dict { :x 1 :y 2 })
        \\    })))
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

    // Verify complex attributes present
    try testing.expect(std.mem.indexOf(u8, output, ":int_attr") != null);
    try testing.expect(std.mem.indexOf(u8, output, ":float_attr") != null);
    try testing.expect(std.mem.indexOf(u8, output, ":string_attr") != null);
    try testing.expect(std.mem.indexOf(u8, output, ":bool_attr") != null);
    try testing.expect(std.mem.indexOf(u8, output, ":sym_attr") != null);
    try testing.expect(std.mem.indexOf(u8, output, ":array_attr") != null);
    try testing.expect(std.mem.indexOf(u8, output, ":nested_attr") != null);

    // Parse again for full validation
    var tok2 = Tokenizer.init(allocator, output);
    var r2 = try Reader.init(allocator, &tok2);
    const value2 = try r2.read();

    var p2 = Parser.init(allocator, "");
    var module2 = try p2.parseModule(value2);
    defer module2.deinit();

    try testing.expectEqual(@as(usize, 7), module2.operations[0].attributes.len);
}

test "printer validation - multiple blocks in region" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const allocator = arena.allocator();

    const input =
        \\(mlir
        \\  (operation
        \\    (name func.func)
        \\    (regions
        \\      (region
        \\        (block [^bb0]
        \\          (arguments [(: %arg0 i32)])
        \\          (operation
        \\            (name cf.br)
        \\            (successors (successor ^bb1 (%arg0)))))
        \\        (block [^bb1]
        \\          (arguments [(: %arg1 i32)])
        \\          (operation
        \\            (name cf.br)
        \\            (successors (successor ^bb2 (%arg1)))))
        \\        (block [^bb2]
        \\          (arguments [(: %arg2 i32)])
        \\          (operation
        \\            (name func.return)
        \\            (operands %arg2)))))))
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

    // Parse again
    var tok2 = Tokenizer.init(allocator, output);
    var r2 = try Reader.init(allocator, &tok2);
    const value2 = try r2.read();

    var p2 = Parser.init(allocator, "");
    var module2 = try p2.parseModule(value2);
    defer module2.deinit();

    // Validate multiple blocks preserved
    try testing.expectEqual(@as(usize, 3), module2.operations[0].regions[0].blocks.len);
    try testing.expectEqualStrings("^bb0", module2.operations[0].regions[0].blocks[0].label.?);
    try testing.expectEqualStrings("^bb1", module2.operations[0].regions[0].blocks[1].label.?);
    try testing.expectEqualStrings("^bb2", module2.operations[0].regions[0].blocks[2].label.?);
}

test "printer validation - complex type expressions" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const allocator = arena.allocator();

    const input =
        \\(mlir
        \\  (operation
        \\    (name test.op)
        \\    (result-types
        \\      !tensor
        \\      !llvm.ptr
        \\      i32)))
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

    // Verify type expressions preserved - just check basics
    try testing.expect(std.mem.indexOf(u8, output, "!tensor") != null);
    try testing.expect(std.mem.indexOf(u8, output, "!llvm.ptr") != null);
    try testing.expect(std.mem.indexOf(u8, output, "(result-types") != null);

    // Parse again
    var tok2 = Tokenizer.init(allocator, output);
    var r2 = try Reader.init(allocator, &tok2);
    const value2 = try r2.read();

    var p2 = Parser.init(allocator, "");
    var module2 = try p2.parseModule(value2);
    defer module2.deinit();

    try testing.expectEqual(@as(usize, 3), module2.operations[0].result_types.len);
}

test "printer validation - successor with multiple operands" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const allocator = arena.allocator();

    const input =
        \\(mlir
        \\  (operation
        \\    (name cf.br)
        \\    (successors
        \\      (successor ^bb1 (%x %y %z %w)))))
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

    // Parse again
    var tok2 = Tokenizer.init(allocator, output);
    var r2 = try Reader.init(allocator, &tok2);
    const value2 = try r2.read();

    var p2 = Parser.init(allocator, "");
    var module2 = try p2.parseModule(value2);
    defer module2.deinit();

    // Validate multiple successor operands preserved
    try testing.expectEqual(@as(usize, 1), module2.operations[0].successors.len);
    try testing.expectEqual(@as(usize, 4), module2.operations[0].successors[0].operands.len);
    try testing.expectEqualStrings("%x", module2.operations[0].successors[0].operands[0]);
    try testing.expectEqualStrings("%y", module2.operations[0].successors[0].operands[1]);
    try testing.expectEqualStrings("%z", module2.operations[0].successors[0].operands[2]);
    try testing.expectEqualStrings("%w", module2.operations[0].successors[0].operands[3]);
}

test "printer validation - string literals" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const allocator = arena.allocator();

    const input =
        \\(mlir
        \\  (operation
        \\    (name test.op)
        \\    (attributes {
        \\      :name (#str "hello")
        \\      :path (#str "file")
        \\    })))
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

    // Verify strings preserved
    try testing.expect(std.mem.indexOf(u8, output, "\"hello\"") != null);
    try testing.expect(std.mem.indexOf(u8, output, "\"file\"") != null);

    // Parse again
    var tok2 = Tokenizer.init(allocator, output);
    var r2 = try Reader.init(allocator, &tok2);
    const value2 = try r2.read();

    var p2 = Parser.init(allocator, "");
    var module2 = try p2.parseModule(value2);
    defer module2.deinit();

    try testing.expectEqual(@as(usize, 2), module2.operations[0].attributes.len);
}

test "printer validation - operation with location metadata" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const allocator = arena.allocator();

    const input =
        \\(mlir
        \\  (operation
        \\    (name arith.constant)
        \\    (result-bindings [%c0])
        \\    (result-types i32)
        \\    (location (#fused (#file "test.mlir" 10 5) (#callsite "main")))))
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

    // Verify location preserved
    try testing.expect(std.mem.indexOf(u8, output, "(location") != null);
    try testing.expect(std.mem.indexOf(u8, output, "#fused") != null);

    // Parse again
    var tok2 = Tokenizer.init(allocator, output);
    var r2 = try Reader.init(allocator, &tok2);
    const value2 = try r2.read();

    var p2 = Parser.init(allocator, "");
    var module2 = try p2.parseModule(value2);
    defer module2.deinit();

    try testing.expect(module2.operations[0].location != null);
}

test "printer validation - full round-trip with complex nested structure" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const allocator = arena.allocator();

    const input =
        \\(mlir
        \\  (operation
        \\    (name builtin.module)
        \\    (regions
        \\      (region
        \\        (block []
        \\          (arguments [])
        \\          (operation
        \\            (name func.func)
        \\            (attributes {
        \\              :sym (#sym @main)
        \\              :type (!function (inputs) (results i32))
        \\            })
        \\            (regions
        \\              (region
        \\                (block [^entry]
        \\                  (arguments [])
        \\                  (operation
        \\                    (name arith.constant)
        \\                    (result-bindings [%c0])
        \\                    (result-types i32)
        \\                    (attributes { :value (#int 0) }))
        \\                  (operation
        \\                    (name scf.while)
        \\                    (operands %c0)
        \\                    (result-types i32)
        \\                    (regions
        \\                      (region
        \\                        (block []
        \\                          (arguments [(: %iter i32)])
        \\                          (operation
        \\                            (name arith.constant)
        \\                            (result-bindings [%limit])
        \\                            (result-types i32)
        \\                            (attributes { :value (#int 10) }))
        \\                          (operation
        \\                            (name arith.cmpi)
        \\                            (result-bindings [%cond])
        \\                            (operands %iter %limit)
        \\                            (result-types i1)
        \\                            (attributes { :predicate (#str "slt") }))
        \\                          (operation
        \\                            (name scf.condition)
        \\                            (operands %cond %iter))))
        \\                      (region
        \\                        (block []
        \\                          (arguments [(: %arg i32)])
        \\                          (operation
        \\                            (name arith.constant)
        \\                            (result-bindings [%c1])
        \\                            (result-types i32)
        \\                            (attributes { :value (#int 1) }))
        \\                          (operation
        \\                            (name arith.addi)
        \\                            (result-bindings [%next])
        \\                            (operands %arg %c1)
        \\                            (result-types i32))
        \\                          (operation
        \\                            (name scf.yield)
        \\                            (operands %next))))))
        \\                  (operation
        \\                    (name func.return)
        \\                    (operands %c0)))))))))))
    ;

    // Parse original
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

    // Parse printed output
    var tok2 = Tokenizer.init(allocator, output);
    var r2 = try Reader.init(allocator, &tok2);
    const value2 = try r2.read();

    var p2 = Parser.init(allocator, "");
    var module2 = try p2.parseModule(value2);
    defer module2.deinit();

    // Print again for third iteration
    var pr3 = Printer.init(allocator);
    defer pr3.deinit();
    try pr3.printModule(&module2);

    const output3 = pr3.getOutput();

    // Parse third time
    var tok3 = Tokenizer.init(allocator, output3);
    var r3 = try Reader.init(allocator, &tok3);
    const value3 = try r3.read();

    var p3 = Parser.init(allocator, "");
    var module3 = try p3.parseModule(value3);
    defer module3.deinit();

    // Validate structure is preserved through multiple round-trips
    try testing.expectEqual(module.operations.len, module2.operations.len);
    try testing.expectEqual(module.operations.len, module3.operations.len);

    // Verify deep nesting preserved
    const outer_region = module3.operations[0].regions[0];
    const func_op = outer_region.blocks[0].operations[0];
    const entry_block = func_op.regions[0].blocks[0];
    const while_op = entry_block.operations[1];

    try testing.expectEqualStrings("scf.while", while_op.name);
    try testing.expectEqual(@as(usize, 2), while_op.regions.len);
}
