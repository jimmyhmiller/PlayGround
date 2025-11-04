const std = @import("std");
const mlir_lisp = @import("mlir_lisp");
const Tokenizer = mlir_lisp.Tokenizer;
const Reader = mlir_lisp.Reader;
const Parser = mlir_lisp.Parser;

// Example 1 — simple constant
const example1 =
    \\(mlir
    \\  (operation
    \\    (name arith.constant)
    \\    (result-bindings [%c0])
    \\    (result-types i32)
    \\    (attributes { :value (#int 42) })
    \\    (location (#unknown))))
;

test "parser - example 1 simple constant" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const allocator = arena.allocator();

    var tok = Tokenizer.init(allocator, example1);
    var reader = try Reader.init(allocator, &tok);
    const value = try reader.read();

    var parser = Parser.init(allocator, "");
    var module = try parser.parseModule(value);
    defer module.deinit();

    // Verify module structure
    try std.testing.expect(module.operations.len == 1);

    const op = module.operations[0];
    try std.testing.expectEqualStrings("arith.constant", op.name);
    try std.testing.expect(op.result_bindings.len == 1);
    try std.testing.expectEqualStrings("%c0", op.result_bindings[0]);
    try std.testing.expect(op.result_types.len == 1);
    try std.testing.expect(op.attributes.len == 1);
    try std.testing.expectEqualStrings("value", op.attributes[0].key);
    try std.testing.expect(op.location != null);
}

// Example 2 — add inside a function
const example2 =
    \\(mlir
    \\  (operation
    \\    (name func.func)
    \\    (attributes {
    \\      :sym  (#sym @add)
    \\      :type (!function (inputs i32 i32) (results i32))
    \\      :visibility :public
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

test "parser - example 2 function with regions" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const allocator = arena.allocator();

    var tok = Tokenizer.init(allocator, example2);
    var reader = try Reader.init(allocator, &tok);
    const value = try reader.read();

    var parser = Parser.init(allocator, "");
    var module = try parser.parseModule(value);
    defer module.deinit();

    // Verify module structure
    try std.testing.expect(module.operations.len == 1);

    const func_op = module.operations[0];
    try std.testing.expectEqualStrings("func.func", func_op.name);
    try std.testing.expect(func_op.attributes.len == 3);
    try std.testing.expect(func_op.regions.len == 1);

    const region = func_op.regions[0];
    try std.testing.expect(region.blocks.len == 1);

    const block = region.blocks[0];
    try std.testing.expect(block.label != null);
    try std.testing.expectEqualStrings("^entry", block.label.?);
    try std.testing.expect(block.arguments.len == 2);
    try std.testing.expectEqualStrings("%x", block.arguments[0].value_id);
    try std.testing.expectEqualStrings("%y", block.arguments[1].value_id);
    try std.testing.expect(block.operations.len == 2);

    // Check nested operations
    const addi_op = block.operations[0];
    try std.testing.expectEqualStrings("arith.addi", addi_op.name);
    try std.testing.expect(addi_op.result_bindings.len == 1);
    try std.testing.expectEqualStrings("%sum", addi_op.result_bindings[0]);
    try std.testing.expect(addi_op.operands.len == 2);
    try std.testing.expectEqualStrings("%x", addi_op.operands[0]);
    try std.testing.expectEqualStrings("%y", addi_op.operands[1]);

    const return_op = block.operations[1];
    try std.testing.expectEqualStrings("func.return", return_op.name);
    try std.testing.expect(return_op.operands.len == 1);
    try std.testing.expectEqualStrings("%sum", return_op.operands[0]);
}

// Example 3 — function call with two constants
const example3 =
    \\(mlir
    \\  (operation
    \\    (name func.func)
    \\    (attributes {
    \\      :sym (#sym @main)
    \\      :type (!function (inputs) (results i32))
    \\    })
    \\    (regions
    \\      (region
    \\        (block [^entry]
    \\          (arguments [])
    \\          (operation
    \\            (name arith.constant)
    \\            (result-bindings [%a])
    \\            (result-types i32)
    \\            (attributes { :value (#int 1) }))
    \\          (operation
    \\            (name arith.constant)
    \\            (result-bindings [%b])
    \\            (result-types i32)
    \\            (attributes { :value (#int 2) }))
    \\          (operation
    \\            (name func.call)
    \\            (result-bindings [%r])
    \\            (result-types i32)
    \\            (operands %a %b)
    \\            (attributes { :callee (#flat-symbol @add) }))
    \\          (operation
    \\            (name func.return)
    \\            (operands %r)))))))
;

test "parser - example 3 multiple operations" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const allocator = arena.allocator();

    var tok = Tokenizer.init(allocator, example3);
    var reader = try Reader.init(allocator, &tok);
    const value = try reader.read();

    var parser = Parser.init(allocator, "");
    var module = try parser.parseModule(value);
    defer module.deinit();

    // Verify module structure
    try std.testing.expect(module.operations.len == 1);

    const func_op = module.operations[0];
    try std.testing.expectEqualStrings("func.func", func_op.name);

    const region = func_op.regions[0];
    const block = region.blocks[0];
    try std.testing.expectEqualStrings("^entry", block.label.?);
    try std.testing.expect(block.arguments.len == 0);
    try std.testing.expect(block.operations.len == 4);

    // Verify operation names
    try std.testing.expectEqualStrings("arith.constant", block.operations[0].name);
    try std.testing.expectEqualStrings("arith.constant", block.operations[1].name);
    try std.testing.expectEqualStrings("func.call", block.operations[2].name);
    try std.testing.expectEqualStrings("func.return", block.operations[3].name);

    // Verify func.call has operands
    const call_op = block.operations[2];
    try std.testing.expect(call_op.operands.len == 2);
    try std.testing.expectEqualStrings("%a", call_op.operands[0]);
    try std.testing.expectEqualStrings("%b", call_op.operands[1]);
}

// Example 4 — control flow with multiple blocks
const example4 =
    \\(mlir
    \\  (operation
    \\    (name func.func)
    \\    (attributes {
    \\      :sym (#sym @branchy)
    \\      :type (!function (inputs i1 i32 i32) (results i32))
    \\    })
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

test "parser - example 4 control flow" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const allocator = arena.allocator();

    var tok = Tokenizer.init(allocator, example4);
    var reader = try Reader.init(allocator, &tok);
    const value = try reader.read();

    var parser = Parser.init(allocator, "");
    var module = try parser.parseModule(value);
    defer module.deinit();

    // Verify module structure
    try std.testing.expect(module.operations.len == 1);

    const func_op = module.operations[0];
    try std.testing.expectEqualStrings("func.func", func_op.name);

    const region = func_op.regions[0];
    try std.testing.expect(region.blocks.len == 3);

    // Entry block
    const entry = region.blocks[0];
    try std.testing.expectEqualStrings("^entry", entry.label.?);
    try std.testing.expect(entry.arguments.len == 3);
    try std.testing.expect(entry.operations.len == 1);

    const br_op = entry.operations[0];
    try std.testing.expectEqualStrings("cf.cond_br", br_op.name);
    try std.testing.expect(br_op.successors.len == 2);

    // Check successors
    try std.testing.expectEqualStrings("^then", br_op.successors[0].block_id);
    try std.testing.expect(br_op.successors[0].operands.len == 1);
    try std.testing.expectEqualStrings("%x", br_op.successors[0].operands[0]);

    try std.testing.expectEqualStrings("^else", br_op.successors[1].block_id);
    try std.testing.expect(br_op.successors[1].operands.len == 1);
    try std.testing.expectEqualStrings("%y", br_op.successors[1].operands[0]);

    // Then block
    const then_block = region.blocks[1];
    try std.testing.expectEqualStrings("^then", then_block.label.?);
    try std.testing.expect(then_block.arguments.len == 1);
    try std.testing.expectEqualStrings("%t", then_block.arguments[0].value_id);

    // Else block
    const else_block = region.blocks[2];
    try std.testing.expectEqualStrings("^else", else_block.label.?);
    try std.testing.expect(else_block.arguments.len == 1);
    try std.testing.expectEqualStrings("%e", else_block.arguments[0].value_id);
}

// Example 5 — recursive fibonacci with scf.if
const example5 =
    \\(mlir
    \\  (operation
    \\    (name func.func)
    \\    (attributes {
    \\      :sym (#sym @fibonacci)
    \\      :type (!function (inputs i32) (results i32))
    \\      :visibility :public
    \\    })
    \\    (regions
    \\      (region
    \\        (block [^entry]
    \\          (arguments [ (: %n i32) ])
    \\          (operation
    \\            (name arith.constant)
    \\            (result-bindings [%c1])
    \\            (result-types i32)
    \\            (attributes { :value (#int 1) }))
    \\          (operation
    \\            (name arith.cmpi)
    \\            (result-bindings [%cond])
    \\            (result-types i1)
    \\            (operands %n %c1)
    \\            (attributes { :predicate (#string "sle") }))
    \\          (operation
    \\            (name scf.if)
    \\            (result-bindings [%result])
    \\            (result-types i32)
    \\            (operands %cond)
    \\            (regions
    \\              (region
    \\                (block
    \\                  (arguments [])
    \\                  (operation
    \\                    (name scf.yield)
    \\                    (operands %n))))
    \\              (region
    \\                (block
    \\                  (arguments [])
    \\                  (operation
    \\                    (name arith.constant)
    \\                    (result-bindings [%c1_rec])
    \\                    (result-types i32)
    \\                    (attributes { :value (#int 1) }))
    \\                  (operation
    \\                    (name arith.subi)
    \\                    (result-bindings [%n_minus_1])
    \\                    (result-types i32)
    \\                    (operands %n %c1_rec))
    \\                  (operation
    \\                    (name func.call)
    \\                    (result-bindings [%fib_n_minus_1])
    \\                    (result-types i32)
    \\                    (operands %n_minus_1)
    \\                    (attributes { :callee (#flat-symbol @fibonacci) }))
    \\                  (operation
    \\                    (name arith.constant)
    \\                    (result-bindings [%c2])
    \\                    (result-types i32)
    \\                    (attributes { :value (#int 2) }))
    \\                  (operation
    \\                    (name arith.subi)
    \\                    (result-bindings [%n_minus_2])
    \\                    (result-types i32)
    \\                    (operands %n %c2))
    \\                  (operation
    \\                    (name func.call)
    \\                    (result-bindings [%fib_n_minus_2])
    \\                    (result-types i32)
    \\                    (operands %n_minus_2)
    \\                    (attributes { :callee (#flat-symbol @fibonacci) }))
    \\                  (operation
    \\                    (name arith.addi)
    \\                    (result-bindings [%sum])
    \\                    (result-types i32)
    \\                    (operands %fib_n_minus_1 %fib_n_minus_2))
    \\                  (operation
    \\                    (name scf.yield)
    \\                    (operands %sum))))))
    \\          (operation
    \\            (name func.return)
    \\            (operands %result)))))))
;

test "parser - example 5 recursive fibonacci" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const allocator = arena.allocator();

    var tok = Tokenizer.init(allocator, example5);
    var reader = try Reader.init(allocator, &tok);
    const value = try reader.read();

    var parser = Parser.init(allocator, "");
    var module = try parser.parseModule(value);
    defer module.deinit();

    // Verify module structure
    try std.testing.expect(module.operations.len == 1);

    const func_op = module.operations[0];
    try std.testing.expectEqualStrings("func.func", func_op.name);
    try std.testing.expect(func_op.attributes.len == 3);
    try std.testing.expect(func_op.regions.len == 1);

    const region = func_op.regions[0];
    try std.testing.expect(region.blocks.len == 1);

    const entry_block = region.blocks[0];
    try std.testing.expectEqualStrings("^entry", entry_block.label.?);
    try std.testing.expect(entry_block.arguments.len == 1);
    try std.testing.expectEqualStrings("%n", entry_block.arguments[0].value_id);

    // Check operations in entry block
    try std.testing.expect(entry_block.operations.len == 4);
    try std.testing.expectEqualStrings("arith.constant", entry_block.operations[0].name);
    try std.testing.expectEqualStrings("arith.cmpi", entry_block.operations[1].name);
    try std.testing.expectEqualStrings("scf.if", entry_block.operations[2].name);
    try std.testing.expectEqualStrings("func.return", entry_block.operations[3].name);

    // Verify scf.if has nested regions
    const scf_if_op = entry_block.operations[2];
    try std.testing.expect(scf_if_op.regions.len == 2);
    try std.testing.expect(scf_if_op.result_bindings.len == 1);
    try std.testing.expectEqualStrings("%result", scf_if_op.result_bindings[0]);

    // Check then region (base case)
    const then_region = scf_if_op.regions[0];
    try std.testing.expect(then_region.blocks.len == 1);
    try std.testing.expect(then_region.blocks[0].operations.len == 1);
    try std.testing.expectEqualStrings("scf.yield", then_region.blocks[0].operations[0].name);

    // Check else region (recursive case)
    const else_region = scf_if_op.regions[1];
    try std.testing.expect(else_region.blocks.len == 1);
    const else_block = else_region.blocks[0];
    try std.testing.expect(else_block.operations.len == 8);

    // Verify recursive calls
    try std.testing.expectEqualStrings("func.call", else_block.operations[2].name);
    try std.testing.expectEqualStrings("func.call", else_block.operations[5].name);
    try std.testing.expectEqualStrings("arith.addi", else_block.operations[6].name);
    try std.testing.expectEqualStrings("scf.yield", else_block.operations[7].name);
}

test "parser - error on invalid structure" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const allocator = arena.allocator();

    // Not a list
    const bad_source = "42";
    var tok = Tokenizer.init(allocator, bad_source);
    var reader = try Reader.init(allocator, &tok);
    const value = try reader.read();

    var parser = Parser.init(allocator, "");
    const result = parser.parseModule(value);
    try std.testing.expectError(error.ExpectedList, result);
}

test "parser - error on missing name" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const allocator = arena.allocator();

    // Operation without name
    const bad_source = "(mlir (operation (result-bindings [%x])))";
    var tok = Tokenizer.init(allocator, bad_source);
    var reader = try Reader.init(allocator, &tok);
    const value = try reader.read();

    var parser = Parser.init(allocator, "");
    const result = parser.parseModule(value);
    try std.testing.expectError(error.MissingRequiredField, result);
}
