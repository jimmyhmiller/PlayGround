const std = @import("std");
const Reader = @import("../frontend/reader.zig").Reader;
const TypeChecker = @import("type_checker.zig");

test "forward references - two-pass basic" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    var allocator = arena.allocator();

    var reader = Reader.init(&allocator);
    var checker = TypeChecker.BidirectionalTypeChecker.init(&allocator);
    defer checker.deinit();

    // This would fail with single-pass but should work with two-pass
    const code =
        \\(def x (: Int) y)
        \\(def y (: Int) 42)
    ;

    const expressions = try reader.readAllString(code);

    // Single-pass should fail
    var single_pass_checker = TypeChecker.BidirectionalTypeChecker.init(&allocator);
    defer single_pass_checker.deinit();
    const single_pass_result = single_pass_checker.typeCheckAll(expressions.items);
    try std.testing.expect(single_pass_result == TypeChecker.TypeCheckError.UnboundVariable);

    // Two-pass should succeed
    const two_pass_result = try checker.typeCheckAllTwoPass(expressions.items);
    try std.testing.expect(two_pass_result.items.len == 2);
    try std.testing.expect(checker.env.get("x").? == .int);
    try std.testing.expect(checker.env.get("y").? == .int);
}

test "forward references - function calling forward function" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    var allocator = arena.allocator();

    var reader = Reader.init(&allocator);
    var checker = TypeChecker.BidirectionalTypeChecker.init(&allocator);
    defer checker.deinit();

    const code =
        \\(def f (: (-> [Int] Int)) (fn [x] (g x)))
        \\(def g (: (-> [Int] Int)) (fn [x] (+ x 1)))
    ;

    const expressions = try reader.readAllString(code);
    const typed = try checker.typeCheckAllTwoPass(expressions.items);

    try std.testing.expect(typed.items.len == 2);
    try std.testing.expect(checker.env.get("f").? == .function);
    try std.testing.expect(checker.env.get("g").? == .function);
}

test "forward references - mutual recursion" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    var allocator = arena.allocator();

    var reader = Reader.init(&allocator);
    var checker = TypeChecker.BidirectionalTypeChecker.init(&allocator);
    defer checker.deinit();

    // Simplified mutual recursion (without proper conditional logic)
    const code =
        \\(def even (: (-> [Int] Int)) (fn [n] (odd n)))
        \\(def odd (: (-> [Int] Int)) (fn [n] (even n)))
    ;

    const expressions = try reader.readAllString(code);
    const typed = try checker.typeCheckAllTwoPass(expressions.items);

    try std.testing.expect(typed.items.len == 2);
    try std.testing.expect(checker.env.get("even").? == .function);
    try std.testing.expect(checker.env.get("odd").? == .function);
}

test "forward references - complex dependency chain" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    var allocator = arena.allocator();

    var reader = Reader.init(&allocator);
    var checker = TypeChecker.BidirectionalTypeChecker.init(&allocator);
    defer checker.deinit();

    const code =
        \\(def a (: I32) b)
        \\(def b (: I32) c)
        \\(def c (: I32) d)
        \\(def d (: I32) 42)
    ;

    const expressions = try reader.readAllString(code);
    const typed = try checker.typeCheckAllTwoPass(expressions.items);

    try std.testing.expect(typed.items.len == 4);
    try std.testing.expect(checker.env.get("a").? == .i32);
    try std.testing.expect(checker.env.get("b").? == .i32);
    try std.testing.expect(checker.env.get("c").? == .i32);
    try std.testing.expect(checker.env.get("d").? == .i32);
}

test "forward references - mixed with arithmetic" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    var allocator = arena.allocator();

    var reader = Reader.init(&allocator);
    var checker = TypeChecker.BidirectionalTypeChecker.init(&allocator);
    defer checker.deinit();

    const code =
        \\(def result (: U64) (* base multiplier))
        \\(def base (: U64) 10)
        \\(def multiplier (: U64) 5)
    ;

    const expressions = try reader.readAllString(code);
    const typed = try checker.typeCheckAllTwoPass(expressions.items);

    try std.testing.expect(typed.items.len == 3);
    try std.testing.expect(checker.env.get("result").? == .u64);
    try std.testing.expect(checker.env.get("base").? == .u64);
    try std.testing.expect(checker.env.get("multiplier").? == .u64);
}

test "forward references - fully typed AST with two-pass" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    var allocator = arena.allocator();

    var reader = Reader.init(&allocator);
    var checker = TypeChecker.BidirectionalTypeChecker.init(&allocator);
    defer checker.deinit();

    const code =
        \\(def x (: I16) y)
        \\(def y (: I16) 100)
    ;

    const expressions = try reader.readAllString(code);
    const typed_asts = try checker.typeCheckAllTwoPass(expressions.items);

    try std.testing.expect(typed_asts.items.len == 2);

    // First definition should reference 'y' which now has type I16
    const first_def = typed_asts.items[0];
    try std.testing.expect(first_def.getType() == .i16);

    // Second definition is just the literal value
    const second_def = typed_asts.items[1];
    try std.testing.expect(second_def.getType() == .i16);
}

test "forward references - type mismatch still caught" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    var allocator = arena.allocator();

    var reader = Reader.init(&allocator);
    var checker = TypeChecker.BidirectionalTypeChecker.init(&allocator);
    defer checker.deinit();

    // Forward reference with wrong type should still fail
    const code =
        \\(def x (: Int) y)
        \\(def y (: String) "hello")
    ;

    const expressions = try reader.readAllString(code);
    const result = checker.typeCheckAllTwoPass(expressions.items);

    // Should fail with type mismatch (Int expected, String provided)
    try std.testing.expect(result == TypeChecker.TypeCheckError.TypeMismatch);
}

test "forward references - incomplete definition chain" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    var allocator = arena.allocator();

    var reader = Reader.init(&allocator);
    var checker = TypeChecker.BidirectionalTypeChecker.init(&allocator);
    defer checker.deinit();

    // Reference to completely undefined variable should still fail
    const code =
        \\(def x (: Int) y)
        \\(def z (: Int) 42)
    ;

    const expressions = try reader.readAllString(code);
    const result = checker.typeCheckAllTwoPass(expressions.items);

    // Should fail because 'y' is never defined
    try std.testing.expect(result == TypeChecker.TypeCheckError.UnboundVariable);
}