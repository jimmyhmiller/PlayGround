const std = @import("std");
const mlir_lisp = @import("mlir_lisp");
const Tokenizer = mlir_lisp.Tokenizer;
const Reader = mlir_lisp.Reader;
const Parser = mlir_lisp.Parser;
const Printer = mlir_lisp.Printer;

test "parse simple type alias" {
    const source =
        \\(mlir
        \\  (type-alias !my_vec "vector<4xf32>"))
    ;

    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const allocator = arena.allocator();

    var tok = Tokenizer.init(allocator, source);
    var reader = try Reader.init(allocator, &tok);
    const value = try reader.read();

    var parser_obj = Parser.init(allocator, "");
    const module = try parser_obj.parseModule(value);

    // Should have 1 type alias
    try std.testing.expectEqual(@as(usize, 1), module.type_aliases.len);
    try std.testing.expectEqualStrings("!my_vec", module.type_aliases[0].name);
    try std.testing.expectEqualStrings("vector<4xf32>", module.type_aliases[0].definition);

    // Should have 0 operations
    try std.testing.expectEqual(@as(usize, 0), module.operations.len);
}

test "parse multiple type aliases" {
    const source =
        \\(mlir
        \\  (type-alias !my_vec "vector<4xf32>")
        \\  (type-alias !my_tensor "tensor<10x20xf32>")
        \\  (type-alias !my_ptr "!llvm.ptr"))
    ;

    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const allocator = arena.allocator();

    var tok = Tokenizer.init(allocator, source);
    var reader = try Reader.init(allocator, &tok);
    const value = try reader.read();

    var parser_obj = Parser.init(allocator, "");
    const module = try parser_obj.parseModule(value);

    // Should have 3 type aliases
    try std.testing.expectEqual(@as(usize, 3), module.type_aliases.len);

    try std.testing.expectEqualStrings("!my_vec", module.type_aliases[0].name);
    try std.testing.expectEqualStrings("vector<4xf32>", module.type_aliases[0].definition);

    try std.testing.expectEqualStrings("!my_tensor", module.type_aliases[1].name);
    try std.testing.expectEqualStrings("tensor<10x20xf32>", module.type_aliases[1].definition);

    try std.testing.expectEqualStrings("!my_ptr", module.type_aliases[2].name);
    try std.testing.expectEqualStrings("!llvm.ptr", module.type_aliases[2].definition);
}

test "parse type alias with operation" {
    const source =
        \\(mlir
        \\  (type-alias !my_vec "vector<4xf32>")
        \\  (operation
        \\    (name func.func)
        \\    (attributes {
        \\      :sym_name @test
        \\      :function_type (!function (inputs !my_vec) (results !my_vec))
        \\    })
        \\    (regions
        \\      (region
        \\        (block [^entry]
        \\          (arguments [ [%arg0 !my_vec] ])
        \\          (operation
        \\            (name func.return)
        \\            (operands %arg0)))))))
    ;

    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const allocator = arena.allocator();

    var tok = Tokenizer.init(allocator, source);
    var reader = try Reader.init(allocator, &tok);
    const value = try reader.read();

    var parser_obj = Parser.init(allocator, "");
    const module = try parser_obj.parseModule(value);

    // Should have 1 type alias
    try std.testing.expectEqual(@as(usize, 1), module.type_aliases.len);
    try std.testing.expectEqualStrings("!my_vec", module.type_aliases[0].name);

    // Should have 1 operation
    try std.testing.expectEqual(@as(usize, 1), module.operations.len);
    try std.testing.expectEqualStrings("func.func", module.operations[0].name);
}

test "print type alias" {
    const source =
        \\(mlir
        \\  (type-alias !my_vec "vector<4xf32>")
        \\  (type-alias !my_tensor "tensor<10x20xf32>"))
    ;

    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const allocator = arena.allocator();

    // Parse
    var tok = Tokenizer.init(allocator, source);
    var reader = try Reader.init(allocator, &tok);
    const value = try reader.read();
    var parser_obj = Parser.init(allocator, "");
    const module = try parser_obj.parseModule(value);

    // Print
    var printer = Printer.init(allocator);
    defer printer.deinit();
    try printer.printModule(&module);
    const output = printer.getOutput();

    // Verify output contains type aliases
    try std.testing.expect(std.mem.indexOf(u8, output, "(type-alias !my_vec \"vector<4xf32>\")") != null);
    try std.testing.expect(std.mem.indexOf(u8, output, "(type-alias !my_tensor \"tensor<10x20xf32>\")") != null);
}

test "roundtrip type alias" {
    const source =
        \\(mlir
        \\  (type-alias !my_vec "vector<4xf32>")
        \\  (operation
        \\    (name arith.constant)
        \\    (result-bindings [%c0])
        \\    (result-types i32)
        \\    (attributes { :value (: 42 i32) })))
    ;

    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const allocator = arena.allocator();

    // Parse
    var tok = Tokenizer.init(allocator, source);
    var reader = try Reader.init(allocator, &tok);
    const value = try reader.read();
    var parser_obj = Parser.init(allocator, "");
    const module = try parser_obj.parseModule(value);

    // Print
    var printer = Printer.init(allocator);
    defer printer.deinit();
    try printer.printModule(&module);
    const output = printer.getOutput();

    // Parse again
    var tok2 = Tokenizer.init(allocator, output);
    var reader2 = try Reader.init(allocator, &tok2);
    const value2 = try reader2.read();
    var parser_obj2 = Parser.init(allocator, "");
    const module2 = try parser_obj2.parseModule(value2);

    // Should have same number of type aliases
    try std.testing.expectEqual(module.type_aliases.len, module2.type_aliases.len);
    try std.testing.expectEqualStrings(module.type_aliases[0].name, module2.type_aliases[0].name);
    try std.testing.expectEqualStrings(module.type_aliases[0].definition, module2.type_aliases[0].definition);

    // Should have same number of operations
    try std.testing.expectEqual(module.operations.len, module2.operations.len);
}
