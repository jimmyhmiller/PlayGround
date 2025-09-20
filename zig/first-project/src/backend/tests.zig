const std = @import("std");
const Reader = @import("../frontend/reader.zig").Reader;
const TypeChecker = @import("type_checker.zig");
const Value = @import("../value.zig").Value;

const BidirectionalTypeChecker = TypeChecker.BidirectionalTypeChecker;
const Type = TypeChecker.Type;
const TypedExpression = TypeChecker.TypedExpression;

test "bidirectional type checker - basic types" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    var allocator = arena.allocator();

    var reader = Reader.init(&allocator);
    var checker = BidirectionalTypeChecker.init(&allocator);
    defer checker.deinit();

    // Test integer synthesis
    {
        const expr = try reader.readString("42");
        const typed = try checker.synthesize(expr);
        try std.testing.expect(typed.type == .int);
    }

    // Test string synthesis
    {
        const expr = try reader.readString("\"hello\"");
        const typed = try checker.synthesize(expr);
        try std.testing.expect(typed.type == .string);
    }

    // Test float synthesis
    {
        const expr = try reader.readString("3.14");
        const typed = try checker.synthesize(expr);
        try std.testing.expect(typed.type == .float);
    }

    // Test nil synthesis
    {
        const expr = try reader.readString("nil");
        const typed = try checker.synthesize(expr);
        try std.testing.expect(typed.type == .nil);
    }
}

test "bidirectional type checker - function definition with type annotation" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    var allocator = arena.allocator();

    var reader = Reader.init(&allocator);
    var checker = BidirectionalTypeChecker.init(&allocator);
    defer checker.deinit();

    // Test function definition: (def f (: (-> [Int] Int)) (fn [x] (+ x 1)))
    const code = "(def f (: (-> [Int] Int)) (fn [x] (+ x 1)))";
    const expr = try reader.readString(code);
    const typed = try checker.typeCheck(expr);

    // Should successfully type check as a function type
    try std.testing.expect(typed.getType() == .function);
    // For function type, we need to access the pointer to FunctionType
    switch (typed.getType()) {
        .function => |func_type| {
            try std.testing.expect(func_type.param_types.len == 1);
            try std.testing.expect(func_type.param_types[0] == .int);
            try std.testing.expect(func_type.return_type == .int);
        },
        else => return error.TestFailed,
    }
}

test "bidirectional type checker - simple arithmetic" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    var allocator = arena.allocator();

    var reader = Reader.init(&allocator);
    var checker = BidirectionalTypeChecker.init(&allocator);
    defer checker.deinit();

    // Test arithmetic: (+ 1 2)
    {
        const expr = try reader.readString("(+ 1 2)");
        const typed = try checker.synthesize(expr);
        try std.testing.expect(typed.type == .int);
    }

    // Test nested arithmetic: (+ 1 (+ 2 3))
    {
        const expr = try reader.readString("(+ 1 (+ 2 3))");
        const typed = try checker.synthesize(expr);
        try std.testing.expect(typed.type == .int);
    }
}

test "bidirectional type checker - vectors" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    var allocator = arena.allocator();

    var reader = Reader.init(&allocator);
    var checker = BidirectionalTypeChecker.init(&allocator);
    defer checker.deinit();

    // Test homogeneous vector
    const expr = try reader.readString("[1 2 3]");
    const typed = try checker.synthesize(expr);
    try std.testing.expect(typed.type == .vector);
}

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
        \\(def a (: Int) b)
        \\(def b (: Int) c)
        \\(def c (: Int) d)
        \\(def d (: Int) 42)
    ;

    const expressions = try reader.readAllString(code);
    const typed = try checker.typeCheckAllTwoPass(expressions.items);

    try std.testing.expect(typed.items.len == 4);
    try std.testing.expect(checker.env.get("a").? == .int);
    try std.testing.expect(checker.env.get("b").? == .int);
    try std.testing.expect(checker.env.get("c").? == .int);
    try std.testing.expect(checker.env.get("d").? == .int);
}

test "forward references - mutual recursion with type mismatch (expected to fail)" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    var allocator = arena.allocator();

    var reader = Reader.init(&allocator);
    var checker = TypeChecker.BidirectionalTypeChecker.init(&allocator);
    defer checker.deinit();

    // Mutual recursion with incompatible types - should fail
    const code =
        \\(def evenCheck (: (-> [Int] String)) (fn [n] (oddCheck n)))
        \\(def oddCheck (: (-> [Int] Int)) (fn [n] (evenCheck n)))
    ;

    const expressions = try reader.readAllString(code);

    // This should fail because evenCheck returns String but oddCheck expects Int from evenCheck call
    const result = checker.typeCheckAllTwoPass(expressions.items);
    try std.testing.expectError(TypeChecker.TypeCheckError.TypeMismatch, result);
}
