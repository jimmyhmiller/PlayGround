const std = @import("std");
const Reader = @import("../../frontend/reader.zig").Reader;
const TypeChecker = @import("../../backend/type_checker.zig");

test "type checker - multiple independent definitions" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    var allocator = arena.allocator();

    var reader = Reader.init(&allocator);
    var checker = TypeChecker.BidirectionalTypeChecker.init(&allocator);
    defer checker.deinit();

    // First definition
    const def1 = "(def x (: Int) 42)";
    const expr1 = try reader.readString(def1);
    _ = try checker.typeCheck(expr1);

    // Check x is in environment with correct type
    const x_type = checker.env.get("x");
    try std.testing.expect(x_type != null);
    try std.testing.expect(x_type.? == .int);

    // Second definition
    const def2 = "(def y (: String) \"hello\")";
    const expr2 = try reader.readString(def2);
    _ = try checker.typeCheck(expr2);

    // Check both are in environment
    try std.testing.expect(checker.env.get("x") != null);
    try std.testing.expect(checker.env.get("x").? == .int);
    try std.testing.expect(checker.env.get("y") != null);
    try std.testing.expect(checker.env.get("y").? == .string);
}

test "type checker - definition referencing previous definition" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    var allocator = arena.allocator();

    var reader = Reader.init(&allocator);
    var checker = TypeChecker.BidirectionalTypeChecker.init(&allocator);
    defer checker.deinit();

    // Define x
    const def1 = "(def x (: Int) 42)";
    const expr1 = try reader.readString(def1);
    _ = try checker.typeCheck(expr1);

    // Define y that uses x
    const def2 = "(def y (: Int) (+ x 1))";
    const expr2 = try reader.readString(def2);
    const typed2 = try checker.typeCheck(expr2);

    // Should succeed and have correct type
    try std.testing.expect(typed2.type == .int);
    try std.testing.expect(checker.env.get("y") != null);
    try std.testing.expect(checker.env.get("y").? == .int);
}

test "type checker - function calling another function" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    var allocator = arena.allocator();

    var reader = Reader.init(&allocator);
    var checker = TypeChecker.BidirectionalTypeChecker.init(&allocator);
    defer checker.deinit();

    // Define inc function
    const def1 = "(def inc (: (-> [Int] Int)) (fn [x] (+ x 1)))";
    const expr1 = try reader.readString(def1);
    _ = try checker.typeCheck(expr1);

    // Define double_inc that calls inc twice
    const def2 = "(def double_inc (: (-> [Int] Int)) (fn [x] (inc (inc x))))";
    const expr2 = try reader.readString(def2);
    const typed2 = try checker.typeCheck(expr2);

    // Should succeed
    try std.testing.expect(typed2.type == .function);
    try std.testing.expect(checker.env.get("double_inc") != null);
}

test "type checker - redefining a variable" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    var allocator = arena.allocator();

    var reader = Reader.init(&allocator);
    var checker = TypeChecker.BidirectionalTypeChecker.init(&allocator);
    defer checker.deinit();

    // First definition of x as Int
    const def1 = "(def x (: Int) 42)";
    const expr1 = try reader.readString(def1);
    _ = try checker.typeCheck(expr1);
    try std.testing.expect(checker.env.get("x").? == .int);

    // Redefine x as String
    const def2 = "(def x (: String) \"hello\")";
    const expr2 = try reader.readString(def2);
    _ = try checker.typeCheck(expr2);

    // x should now have String type
    try std.testing.expect(checker.env.get("x").? == .string);
}

test "type checker - mutual recursion (should fail without special handling)" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    var allocator = arena.allocator();

    var reader = Reader.init(&allocator);
    var checker = TypeChecker.BidirectionalTypeChecker.init(&allocator);
    defer checker.deinit();

    // Try to define even which calls odd (not yet defined)
    const def1 = "(def even (: (-> [Int] Bool)) (fn [n] (if (= n 0) true (odd (- n 1)))))";
    const expr1 = try reader.readString(def1);
    const result = checker.typeCheck(expr1);

    // This should fail because 'odd' is not defined yet
    try std.testing.expect(result == TypeChecker.TypeCheckError.UnboundVariable);
}

test "type checker - multiple definitions in sequence" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    var allocator = arena.allocator();

    var reader = Reader.init(&allocator);
    var checker = TypeChecker.BidirectionalTypeChecker.init(&allocator);
    defer checker.deinit();

    // Define a series of values and functions
    const defs = [_][]const u8{
        "(def a (: Int) 10)",
        "(def b (: Int) 20)",
        "(def add_ab (: (-> [] Int)) (fn [] (+ a b)))",
        "(def c (: Int) (add_ab))",
    };

    for (defs) |def| {
        const expr = try reader.readString(def);
        _ = try checker.typeCheck(expr);
    }

    // Check all are defined with correct types
    try std.testing.expect(checker.env.get("a").? == .int);
    try std.testing.expect(checker.env.get("b").? == .int);
    try std.testing.expect(checker.env.get("add_ab").? == .function);
    try std.testing.expect(checker.env.get("c").? == .int);
}