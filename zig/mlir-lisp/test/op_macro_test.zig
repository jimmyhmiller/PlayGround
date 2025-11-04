const std = @import("std");
const mlir_lisp = @import("mlir_lisp");
const Reader = mlir_lisp.Reader;
const Value = mlir_lisp.Value;
const ValueType = mlir_lisp.ValueType;
const Tokenizer = mlir_lisp.Tokenizer;
const MacroExpander = mlir_lisp.MacroExpander;
const builtin_macros = mlir_lisp.builtin_macros;

fn readString(allocator: std.mem.Allocator, source: []const u8) !*Value {
    var tok = Tokenizer.init(allocator, source);
    var r = try Reader.init(allocator, &tok);
    return try r.read();
}

test "op macro: simple operation with binding and type" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const allocator = arena.allocator();

    // Test: (op %N (: index) (memref.dim [%B %c1]))
    const input =
        \\(op %N (: index) (memref.dim [%B %c1]))
    ;

    const first_value = try readString(allocator, input);

    var expander = MacroExpander.init(allocator);
    defer expander.deinit();
    try builtin_macros.registerBuiltinMacros(&expander);

    const expanded = try expander.expandAll(first_value);

    // Should expand to:
    // (operation
    //   (name memref.dim)
    //   (result-bindings [%N])
    //   (result-types index)
    //   (operands %B %c1))

    try std.testing.expectEqual(ValueType.list, expanded.type);
    const expanded_list = expanded.data.list;

    // Check "operation"
    try std.testing.expectEqual(ValueType.identifier, expanded_list.at(0).type);
    try std.testing.expectEqualStrings("operation", expanded_list.at(0).data.atom);

    // Check (name memref.dim)
    const name_list = expanded_list.at(1).data.list;
    try std.testing.expectEqualStrings("name", name_list.at(0).data.atom);
    try std.testing.expectEqualStrings("memref.dim", name_list.at(1).data.atom);

    // Check (result-bindings [%N])
    const bindings_list = expanded_list.at(2).data.list;
    try std.testing.expectEqualStrings("result-bindings", bindings_list.at(0).data.atom);
    const bindings_vec = bindings_list.at(1).data.vector;
    try std.testing.expectEqual(@as(usize, 1), bindings_vec.len());
    try std.testing.expectEqual(ValueType.value_id, bindings_vec.at(0).type);
    try std.testing.expectEqualStrings("%N", bindings_vec.at(0).data.atom);

    // Check (result-types index)
    const types_list = expanded_list.at(3).data.list;
    try std.testing.expectEqualStrings("result-types", types_list.at(0).data.atom);
    try std.testing.expectEqualStrings("index", types_list.at(1).data.atom);

    // Check (operands %B %c1)
    const operands_list = expanded_list.at(4).data.list;
    try std.testing.expectEqualStrings("operands", operands_list.at(0).data.atom);
    try std.testing.expectEqual(ValueType.value_id, operands_list.at(1).type);
    try std.testing.expectEqualStrings("%B", operands_list.at(1).data.atom);
    try std.testing.expectEqual(ValueType.value_id, operands_list.at(2).type);
    try std.testing.expectEqualStrings("%c1", operands_list.at(2).data.atom);
}

test "op macro: operation without binding (auto-generated)" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const allocator = arena.allocator();

    // Test: (op (: index) (memref.dim [%B %c1]))
    const input =
        \\(op (: index) (memref.dim [%B %c1]))
    ;

    const first_value = try readString(allocator, input);

    var expander = MacroExpander.init(allocator);
    defer expander.deinit();
    try builtin_macros.registerBuiltinMacros(&expander);

    const expanded = try expander.expandAll(first_value);

    // Should expand to:
    // (operation
    //   (name memref.dim)
    //   (result-types index)
    //   (operands %B %c1))

    try std.testing.expectEqual(ValueType.list, expanded.type);
    const expanded_list = expanded.data.list;

    // Check "operation"
    try std.testing.expectEqualStrings("operation", expanded_list.at(0).data.atom);

    // Check (name memref.dim)
    const name_list = expanded_list.at(1).data.list;
    try std.testing.expectEqualStrings("name", name_list.at(0).data.atom);
    try std.testing.expectEqualStrings("memref.dim", name_list.at(1).data.atom);

    // Check (result-types index) - should be at position 2 (no bindings)
    const types_list = expanded_list.at(2).data.list;
    try std.testing.expectEqualStrings("result-types", types_list.at(0).data.atom);
    try std.testing.expectEqualStrings("index", types_list.at(1).data.atom);

    // Check (operands %B %c1)
    const operands_list = expanded_list.at(3).data.list;
    try std.testing.expectEqualStrings("operands", operands_list.at(0).data.atom);
    try std.testing.expectEqualStrings("%B", operands_list.at(1).data.atom);
    try std.testing.expectEqualStrings("%c1", operands_list.at(2).data.atom);
}

test "op macro: operation without type (void operation)" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const allocator = arena.allocator();

    // Test: (op (memref.store [%value %C %i %j]))
    const input =
        \\(op (memref.store [%value %C %i %j]))
    ;

    
    

    
    

    const first_value = try readString(allocator, input);

    var expander = MacroExpander.init(allocator);
    defer expander.deinit();
    try builtin_macros.registerBuiltinMacros(&expander);

    const expanded = try expander.expandAll(first_value);

    // Should expand to:
    // (operation
    //   (name memref.store)
    //   (operands %value %C %i %j))

    try std.testing.expectEqual(ValueType.list, expanded.type);
    const expanded_list = expanded.data.list;

    // Check "operation"
    try std.testing.expectEqualStrings("operation", expanded_list.at(0).data.atom);

    // Check (name memref.store)
    const name_list = expanded_list.at(1).data.list;
    try std.testing.expectEqualStrings("name", name_list.at(0).data.atom);
    try std.testing.expectEqualStrings("memref.store", name_list.at(1).data.atom);

    // Check (operands %value %C %i %j)
    const operands_list = expanded_list.at(2).data.list;
    try std.testing.expectEqualStrings("operands", operands_list.at(0).data.atom);
    try std.testing.expectEqual(@as(usize, 5), operands_list.len()); // operands + 4 values
    try std.testing.expectEqualStrings("%value", operands_list.at(1).data.atom);
    try std.testing.expectEqualStrings("%C", operands_list.at(2).data.atom);
    try std.testing.expectEqualStrings("%i", operands_list.at(3).data.atom);
    try std.testing.expectEqualStrings("%j", operands_list.at(4).data.atom);
}

test "op macro: operation with explicit regions" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const allocator = arena.allocator();

    // Test: (op %result (: i32) (scf.if [%cond]
    //         (region (block [] (operation (name scf.yield) (operands %x))))
    //         (region (block [] (operation (name scf.yield) (operands %y))))))
    const input =
        \\(op %result (: i32) (scf.if [%cond]
        \\  (region (block [] (operation (name scf.yield) (operands %x))))
        \\  (region (block [] (operation (name scf.yield) (operands %y))))))
    ;

    
    

    
    

    const first_value = try readString(allocator, input);

    var expander = MacroExpander.init(allocator);
    defer expander.deinit();
    try builtin_macros.registerBuiltinMacros(&expander);

    const expanded = try expander.expandAll(first_value);

    // Should expand to:
    // (operation
    //   (name scf.if)
    //   (result-bindings [%result])
    //   (result-types i32)
    //   (operands %cond)
    //   (regions
    //     (region (block [] (operation ...)))
    //     (region (block [] (operation ...)))))

    try std.testing.expectEqual(ValueType.list, expanded.type);
    const expanded_list = expanded.data.list;

    // Check "operation"
    try std.testing.expectEqualStrings("operation", expanded_list.at(0).data.atom);

    // Check (name scf.if)
    const name_list = expanded_list.at(1).data.list;
    try std.testing.expectEqualStrings("name", name_list.at(0).data.atom);
    try std.testing.expectEqualStrings("scf.if", name_list.at(1).data.atom);

    // Check (result-bindings [%result])
    const bindings_list = expanded_list.at(2).data.list;
    try std.testing.expectEqualStrings("result-bindings", bindings_list.at(0).data.atom);

    // Check (result-types i32)
    const types_list = expanded_list.at(3).data.list;
    try std.testing.expectEqualStrings("result-types", types_list.at(0).data.atom);

    // Check (operands %cond)
    const operands_list = expanded_list.at(4).data.list;
    try std.testing.expectEqualStrings("operands", operands_list.at(0).data.atom);

    // Check (regions ...)
    const regions_list = expanded_list.at(5).data.list;
    try std.testing.expectEqualStrings("regions", regions_list.at(0).data.atom);
    try std.testing.expectEqual(@as(usize, 3), regions_list.len()); // "regions" + 2 region forms
}

test "op macro: operation with no operands" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const allocator = arena.allocator();

    // Test: (op %acc_init (: f32) (arith.constant []))
    const input =
        \\(op %acc_init (: f32) (arith.constant []))
    ;

    
    

    
    

    const first_value = try readString(allocator, input);

    var expander = MacroExpander.init(allocator);
    defer expander.deinit();
    try builtin_macros.registerBuiltinMacros(&expander);

    const expanded = try expander.expandAll(first_value);

    // Should expand to:
    // (operation
    //   (name arith.constant)
    //   (result-bindings [%acc_init])
    //   (result-types f32))
    // Note: no operands section since the vector is empty

    try std.testing.expectEqual(ValueType.list, expanded.type);
    const expanded_list = expanded.data.list;

    // Check "operation"
    try std.testing.expectEqualStrings("operation", expanded_list.at(0).data.atom);

    // Check (name arith.constant)
    const name_list = expanded_list.at(1).data.list;
    try std.testing.expectEqualStrings("name", name_list.at(0).data.atom);
    try std.testing.expectEqualStrings("arith.constant", name_list.at(1).data.atom);

    // Check (result-bindings [%acc_init])
    const bindings_list = expanded_list.at(2).data.list;
    try std.testing.expectEqualStrings("result-bindings", bindings_list.at(0).data.atom);

    // Check (result-types f32)
    const types_list = expanded_list.at(3).data.list;
    try std.testing.expectEqualStrings("result-types", types_list.at(0).data.atom);
    try std.testing.expectEqualStrings("f32", types_list.at(1).data.atom);

    // Should only have 4 elements (no operands)
    try std.testing.expectEqual(@as(usize, 4), expanded_list.len());
}

test "op macro: operation with multiple operands" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const allocator = arena.allocator();

    // Test: (op %sum (: i32) (arith.addi [%a %b]))
    const input =
        \\(op %sum (: i32) (arith.addi [%a %b]))
    ;

    
    

    
    

    const first_value = try readString(allocator, input);

    var expander = MacroExpander.init(allocator);
    defer expander.deinit();
    try builtin_macros.registerBuiltinMacros(&expander);

    const expanded = try expander.expandAll(first_value);

    // Should expand to:
    // (operation
    //   (name arith.addi)
    //   (result-bindings [%sum])
    //   (result-types i32)
    //   (operands %a %b))

    try std.testing.expectEqual(ValueType.list, expanded.type);
    const expanded_list = expanded.data.list;

    // Check (operands %a %b)
    const operands_list = expanded_list.at(4).data.list;
    try std.testing.expectEqualStrings("operands", operands_list.at(0).data.atom);
    try std.testing.expectEqual(@as(usize, 3), operands_list.len()); // operands + 2 values
    try std.testing.expectEqualStrings("%a", operands_list.at(1).data.atom);
    try std.testing.expectEqualStrings("%b", operands_list.at(2).data.atom);
}
