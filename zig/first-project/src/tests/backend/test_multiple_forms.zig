const std = @import("std");
const Reader = @import("../../frontend/reader.zig").Reader;
const TypeChecker = @import("../../backend/type_checker.zig");

test "type checker - multiple top-level forms" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    var allocator = arena.allocator();

    var reader = Reader.init(&allocator);
    var checker = TypeChecker.BidirectionalTypeChecker.init(&allocator);
    defer checker.deinit();

    const code =
        \\(def x (: Int) 42)
        \\(def y (: Int) 100)
        \\(def sum (: (-> [] Int)) (fn [] (+ x y)))
        \\(def result (: Int) (sum))
    ;

    // Read all expressions
    const expressions = try reader.readAllString(code);

    // Type check all expressions
    const typed_exprs = try checker.typeCheckAll(expressions.items);

    // Verify we got 4 typed expressions
    try std.testing.expect(typed_exprs.items.len == 4);

    // Verify all definitions are in the environment
    try std.testing.expect(checker.env.get("x") != null);
    try std.testing.expect(checker.env.get("x").? == .int);

    try std.testing.expect(checker.env.get("y") != null);
    try std.testing.expect(checker.env.get("y").? == .int);

    try std.testing.expect(checker.env.get("sum") != null);
    try std.testing.expect(checker.env.get("sum").? == .function);

    try std.testing.expect(checker.env.get("result") != null);
    try std.testing.expect(checker.env.get("result").? == .int);
}

test "type checker - fully typed multiple forms" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    var allocator = arena.allocator();

    var reader = Reader.init(&allocator);
    var checker = TypeChecker.BidirectionalTypeChecker.init(&allocator);
    defer checker.deinit();

    const code =
        \\(def a (: Int) 10)
        \\(def b (: Int) 20)
        \\(def c (: Int) (+ a b))
    ;

    // Read all expressions
    const expressions = try reader.readAllString(code);

    // Get fully typed ASTs for all expressions
    const typed_asts = try checker.typeCheckAll(expressions.items);

    // Verify we got 3 typed ASTs
    try std.testing.expect(typed_asts.items.len == 3);

    // First two should be definitions with Int type
    try std.testing.expect(typed_asts.items[0].getType() == .int);
    try std.testing.expect(typed_asts.items[1].getType() == .int);

    // Third should be definition with Int type
    const third = typed_asts.items[2];
    try std.testing.expect(third.getType() == .int);
}

test "type checker - forward reference fails" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    var allocator = arena.allocator();

    var reader = Reader.init(&allocator);
    var checker = TypeChecker.BidirectionalTypeChecker.init(&allocator);
    defer checker.deinit();

    const code =
        \\(def x (: Int) y)
        \\(def y (: Int) 42)
    ;

    // Read all expressions
    const expressions = try reader.readAllString(code);

    // This should fail because y is not defined when x tries to use it
    const result = checker.typeCheckAll(expressions.items);
    try std.testing.expect(result == TypeChecker.TypeCheckError.UnboundVariable);
}

test "type checker - complex interdependent definitions" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    var allocator = arena.allocator();

    var reader = Reader.init(&allocator);
    var checker = TypeChecker.BidirectionalTypeChecker.init(&allocator);
    defer checker.deinit();

    const code =
        \\(def pi (: Int) 3)
        \\(def radius (: Int) 10)
        \\(def diameter (: Int) (+ radius radius))
        \\(def circumference (: Int) (+ (+ (+ diameter diameter) diameter) pi))
    ;

    // Read and type check all
    const expressions = try reader.readAllString(code);

    _ = try checker.typeCheckAll(expressions.items);

    // All should succeed and be in environment
    try std.testing.expect(checker.env.get("pi").? == .int);
    try std.testing.expect(checker.env.get("radius").? == .int);
    try std.testing.expect(checker.env.get("diameter").? == .int);
    try std.testing.expect(checker.env.get("circumference").? == .int);
}