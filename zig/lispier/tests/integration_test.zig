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

    // Unqualified ops are rejected by the validator even when use-dialect is present.
    try std.testing.expect(!result.is_valid);
    try std.testing.expect(result.validation_errors.len > 0);
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
