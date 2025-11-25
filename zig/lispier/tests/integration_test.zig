const std = @import("std");
const main = @import("main");

test "integration: simple arithmetic" {
    const allocator = std.testing.allocator;

    var compiler = try main.Compiler.init(allocator);
    defer compiler.deinit();

    const source =
        \\(require-dialect arith)
        \\(arith.addi 1 2)
    ;

    var result = try compiler.compile(source);
    defer result.deinit(allocator);

    try std.testing.expect(result.is_valid);
    try std.testing.expectEqual(@as(usize, 2), result.nodes.items.len);
}

test "integration: function with block" {
    const allocator = std.testing.allocator;

    var compiler = try main.Compiler.init(allocator);
    defer compiler.deinit();

    const source =
        \\(require-dialect [func :as f] [arith :as a])
        \\(f/func {:sym_name "add" :function_type (-> [i64 i64] [i64])}
        \\  (do
        \\    (block [(: x i64) (: y i64)]
        \\      (def result (a/addi x y))
        \\      (f/return result))))
    ;

    var result = try compiler.compile(source);
    defer result.deinit(allocator);

    try std.testing.expect(result.is_valid);
}

test "integration: nested let bindings" {
    const allocator = std.testing.allocator;

    var compiler = try main.Compiler.init(allocator);
    defer compiler.deinit();

    const source =
        \\(require-dialect arith)
        \\(let [x 42
        \\      y 10
        \\      sum (arith.addi x y)]
        \\  sum)
    ;

    var result = try compiler.compile(source);
    defer result.deinit(allocator);

    try std.testing.expect(result.is_valid);
}

test "integration: control flow with blocks" {
    const allocator = std.testing.allocator;

    var compiler = try main.Compiler.init(allocator);
    defer compiler.deinit();

    const source =
        \\(require-dialect cf arith)
        \\(do
        \\  (block [(: n i64)]
        \\    (cf.br {:successors [^loop]} n))
        \\  (block ^loop [(: iter i64)]
        \\    (def is_zero (arith.cmpi {:predicate "eq"} iter 0))
        \\    (cf.cond_br {:successors [^done ^continue]
        \\                 :operand_segment_sizes [1 0 1]}
        \\                is_zero iter))
        \\  (block ^continue [(: val i64)]
        \\    (def next (arith.subi val 1))
        \\    (cf.br {:successors [^loop]} next))
        \\  (block ^done
        \\    0))
    ;

    var result = try compiler.compile(source);
    defer result.deinit(allocator);

    try std.testing.expect(result.is_valid);
}

test "integration: type annotations" {
    const allocator = std.testing.allocator;

    var compiler = try main.Compiler.init(allocator);
    defer compiler.deinit();

    const source =
        \\(require-dialect arith)
        \\(def x (: 42 i32))
        \\(def y (: 3.14 f32))
    ;

    var result = try compiler.compile(source);
    defer result.deinit(allocator);

    try std.testing.expect(result.is_valid);
}

test "integration: map attributes" {
    const allocator = std.testing.allocator;

    var compiler = try main.Compiler.init(allocator);
    defer compiler.deinit();

    const source =
        \\(require-dialect arith)
        \\(arith.constant {:value 42})
    ;

    var result = try compiler.compile(source);
    defer result.deinit(allocator);

    try std.testing.expect(result.is_valid);

    // Check that attributes were parsed
    const op_node = result.nodes.items[1];
    try std.testing.expectEqual(main.ast.NodeType.Operation, op_node.node_type);

    const op = op_node.data.operation;
    try std.testing.expect(op.attributes.contains("value"));
}

test "integration: invalid dialect" {
    const allocator = std.testing.allocator;

    var compiler = try main.Compiler.init(allocator);
    defer compiler.deinit();

    const source =
        \\(require-dialect nonexistent)
        \\(nonexistent.op 1 2)
    ;

    var result = try compiler.compile(source);
    defer result.deinit(allocator);

    try std.testing.expect(!result.is_valid);
    try std.testing.expect(result.validation_errors.len > 0);
}

test "integration: invalid operation" {
    const allocator = std.testing.allocator;

    var compiler = try main.Compiler.init(allocator);
    defer compiler.deinit();

    const source =
        \\(require-dialect arith)
        \\(arith.nonexistent 1 2)
    ;

    var result = try compiler.compile(source);
    defer result.deinit(allocator);

    try std.testing.expect(!result.is_valid);
    try std.testing.expect(result.validation_errors.len > 0);
}

test "integration: unknown unqualified operation should fail validation" {
    const allocator = std.testing.allocator;

    var compiler = try main.Compiler.init(allocator);
    defer compiler.deinit();

    // Documentation says operations are validated against loaded dialects,
    // but unqualified ops currently bypass namespace checking entirely.
    const source =
        \\(require-dialect arith)
        \\(totally-made-up-op 1 2)
    ;

    var result = try compiler.compile(source);
    defer result.deinit(allocator);

    try std.testing.expect(!result.is_valid);
    try std.testing.expect(result.validation_errors.len > 0);
}

test "integration: mixed dialect notations" {
    const allocator = std.testing.allocator;

    var compiler = try main.Compiler.init(allocator);
    defer compiler.deinit();

    const source =
        \\(require-dialect arith [func :as f])
        \\(use-dialect memref)
        \\(def buffer (alloc))
        \\(def sum (arith.addi 1 2))
        \\(f/return sum)
    ;

    var result = try compiler.compile(source);
    defer result.deinit(allocator);

    // With proper use-dialect resolution, alloc should resolve to memref
    // This should now succeed!
    try std.testing.expect(result.is_valid);

    // Verify alloc was resolved to memref namespace
    const def_node = result.nodes.items[2]; // (def buffer (alloc))
    try std.testing.expectEqual(main.ast.NodeType.Def, def_node.node_type);
    const binding = def_node.data.binding;
    try std.testing.expectEqual(main.ast.NodeType.Operation, binding.value.node_type);
    const alloc_op = binding.value.data.operation;
    try std.testing.expectEqualStrings("alloc", alloc_op.name);
    try std.testing.expectEqualStrings("memref", alloc_op.namespace.?);
}

test "integration: destructuring" {
    const allocator = std.testing.allocator;

    var compiler = try main.Compiler.init(allocator);
    defer compiler.deinit();

    const source =
        \\(require-dialect arith)
        \\(def [a b] (arith.multi_result))
    ;

    var result = try compiler.compile(source);
    defer result.deinit(allocator);

    // This should parse but may not validate (multi_result doesn't exist)
    try std.testing.expectEqual(@as(usize, 2), result.nodes.items.len);
}

test "integration: block labels and arguments" {
    const allocator = std.testing.allocator;

    var compiler = try main.Compiler.init(allocator);
    defer compiler.deinit();

    const source =
        \\(require-dialect cf)
        \\(do
        \\  (block ^entry
        \\    (cf.br {:successors [^loop]} 0))
        \\  (block ^loop [(: i i64)]
        \\    (cf.br {:successors [^loop]} i)))
    ;

    var result = try compiler.compile(source);
    defer result.deinit(allocator);

    try std.testing.expect(result.is_valid);
}

test "integration: use-dialect with unique operation resolves correctly" {
    const allocator = std.testing.allocator;

    var compiler = try main.Compiler.init(allocator);
    defer compiler.deinit();

    const source =
        \\(use-dialect arith)
        \\(addi 1 2)
    ;

    var result = try compiler.compile(source);
    defer result.deinit(allocator);

    try std.testing.expect(result.is_valid);

    // Check that addi was resolved to arith namespace
    const op_node = result.nodes.items[1];
    try std.testing.expectEqual(main.ast.NodeType.Operation, op_node.node_type);
    const op = op_node.data.operation;
    try std.testing.expectEqualStrings("addi", op.name);
    try std.testing.expect(op.namespace != null);
    try std.testing.expectEqualStrings("arith", op.namespace.?);
}

test "integration: use-dialect with ambiguous operation fails" {
    const allocator = std.testing.allocator;

    var compiler = try main.Compiler.init(allocator);
    defer compiler.deinit();

    // 'constant' exists in both arith and func
    const source =
        \\(use-dialect arith)
        \\(use-dialect func)
        \\(constant 1)
    ;

    // This should fail at the reader level with AmbiguousSymbol error
    const result = compiler.compile(source);
    try std.testing.expectError(main.reader.ReaderError.AmbiguousSymbol, result);
}

test "integration: use-dialect with invalid operation goes to user namespace" {
    const allocator = std.testing.allocator;

    var compiler = try main.Compiler.init(allocator);
    defer compiler.deinit();

    const source =
        \\(use-dialect arith)
        \\(nonexistent_op 1 2)
    ;

    var result = try compiler.compile(source);
    defer result.deinit(allocator);

    // Should compile (resolve to user namespace) but fail validation
    try std.testing.expect(!result.is_valid);
    try std.testing.expect(result.validation_errors.len > 0);

    // Check that it was resolved to user namespace
    const op_node = result.nodes.items[1];
    try std.testing.expectEqual(main.ast.NodeType.Operation, op_node.node_type);
    const op = op_node.data.operation;
    try std.testing.expectEqualStrings("nonexistent_op", op.name);
    try std.testing.expect(op.namespace != null);
    try std.testing.expectEqualStrings("user", op.namespace.?);
}

test "integration: bare symbols get user namespace" {
    const allocator = std.testing.allocator;

    var compiler = try main.Compiler.init(allocator);
    defer compiler.deinit();

    const source =
        \\(my_function 1 2 3)
    ;

    var result = try compiler.compile(source);
    defer result.deinit(allocator);

    // Should parse successfully
    try std.testing.expectEqual(@as(usize, 1), result.nodes.items.len);

    // Check that it was resolved to user namespace
    const op_node = result.nodes.items[0];
    try std.testing.expectEqual(main.ast.NodeType.Operation, op_node.node_type);
    const op = op_node.data.operation;
    try std.testing.expectEqualStrings("my_function", op.name);
    try std.testing.expect(op.namespace != null);
    try std.testing.expectEqualStrings("user", op.namespace.?);
}

test "integration: multiple use-dialect with unique operations" {
    const allocator = std.testing.allocator;

    var compiler = try main.Compiler.init(allocator);
    defer compiler.deinit();

    const source =
        \\(use-dialect arith)
        \\(use-dialect func)
        \\(addi 1 2)
        \\(return 42)
    ;

    var result = try compiler.compile(source);
    defer result.deinit(allocator);

    try std.testing.expect(result.is_valid);

    // Check that addi resolved to arith
    const addi_node = result.nodes.items[2];
    try std.testing.expectEqual(main.ast.NodeType.Operation, addi_node.node_type);
    const addi_op = addi_node.data.operation;
    try std.testing.expectEqualStrings("addi", addi_op.name);
    try std.testing.expectEqualStrings("arith", addi_op.namespace.?);

    // Check that return resolved to func
    const return_node = result.nodes.items[3];
    try std.testing.expectEqual(main.ast.NodeType.Operation, return_node.node_type);
    const return_op = return_node.data.operation;
    try std.testing.expectEqualStrings("return", return_op.name);
    try std.testing.expectEqualStrings("func", return_op.namespace.?);
}

test "integration: explicit dot notation overrides use-dialect" {
    const allocator = std.testing.allocator;

    var compiler = try main.Compiler.init(allocator);
    defer compiler.deinit();

    const source =
        \\(use-dialect func)
        \\(arith.addi 1 2)
    ;

    var result = try compiler.compile(source);
    defer result.deinit(allocator);

    try std.testing.expect(result.is_valid);

    // Check that arith.addi kept its explicit namespace
    const op_node = result.nodes.items[1];
    try std.testing.expectEqual(main.ast.NodeType.Operation, op_node.node_type);
    const op = op_node.data.operation;
    try std.testing.expectEqualStrings("addi", op.name);
    try std.testing.expectEqualStrings("arith", op.namespace.?);
}

test "integration: complex nested structure with mixed namespaces" {
    const allocator = std.testing.allocator;

    var compiler = try main.Compiler.init(allocator);
    defer compiler.deinit();

    const source =
        \\(require-dialect [func :as f] [cf :as control])
        \\(use-dialect arith)
        \\(f/func {:sym_name "complex" :function_type (-> [i64] [i64])}
        \\  (do
        \\    (block [(: n i64)]
        \\      (def x (addi n 1))
        \\      (def y (muli x 2))
        \\      (def cond (cmpi {:predicate "sgt"} y 10))
        \\      (control/cond_br {:successors [^true ^false]} cond))
        \\    (block ^true
        \\      (f/return y))
        \\    (block ^false
        \\      (f/return 0))))
    ;

    var result = try compiler.compile(source);
    defer result.deinit(allocator);

    try std.testing.expect(result.is_valid);

    // Verify we have the expected number of nodes
    try std.testing.expect(result.nodes.items.len > 0);
}
