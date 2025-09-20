const std = @import("std");
const Reader = @import("../../frontend/reader.zig").Reader;
const TypeChecker = @import("../../backend/type_checker.zig");
const Value = @import("../value.zig").Value;

test "fully typed AST - all subexpressions have types" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    var allocator = arena.allocator();

    var reader = Reader.init(&allocator);
    var checker = TypeChecker.BidirectionalTypeChecker.init(&allocator);
    defer checker.deinit();

    // Test nested arithmetic: (+ 1 (+ 2 3))
    const code = "(+ 1 (+ 2 3))";
    const expr = try reader.readString(code);
    const typed = try checker.typeCheck(expr);

    // The top-level expression should have type Int
    try std.testing.expect(typed.getType() == .int);
    try std.testing.expect(typed.* == .list);

    // Check that all subexpressions are typed
    const list = typed.list;
    try std.testing.expect(list.type == .int);
    try std.testing.expect(list.elements.len == 2); // Two arguments to +

    // First argument: 1
    const first_arg = list.elements[0];
    try std.testing.expect(first_arg.* == .int);
    try std.testing.expect(first_arg.int.value == 1);
    try std.testing.expect(first_arg.int.type == .int);

    // Second argument: (+ 2 3)
    const second_arg = list.elements[1];
    try std.testing.expect(second_arg.* == .list);
    try std.testing.expect(second_arg.list.type == .int);
    try std.testing.expect(second_arg.list.elements.len == 2);

    // Check nested elements: 2 and 3
    const nested_first = second_arg.list.elements[0];
    try std.testing.expect(nested_first.* == .int);
    try std.testing.expect(nested_first.int.value == 2);
    try std.testing.expect(nested_first.int.type == .int);

    const nested_second = second_arg.list.elements[1];
    try std.testing.expect(nested_second.* == .int);
    try std.testing.expect(nested_second.int.value == 3);
    try std.testing.expect(nested_second.int.type == .int);
}

test "fully typed AST - vector elements all typed" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    var allocator = arena.allocator();

    var reader = Reader.init(&allocator);
    var checker = TypeChecker.BidirectionalTypeChecker.init(&allocator);
    defer checker.deinit();

    // Test vector: [1 2 3]
    const code = "[1 2 3]";
    const expr = try reader.readString(code);
    const typed = try checker.typeCheck(expr);

    // The vector should have type [Int]
    try std.testing.expect(typed.getType() == .vector);
    try std.testing.expect(typed.* == .vector);

    const vec = typed.vector;
    try std.testing.expect(vec.type == .vector);
    try std.testing.expect(vec.type.vector.* == .int);
    try std.testing.expect(vec.elements.len == 3);

    // Check each element is properly typed
    for (vec.elements, 0..) |elem, i| {
        try std.testing.expect(elem.* == .int);
        try std.testing.expect(elem.int.value == @as(i64, @intCast(i + 1)));
        try std.testing.expect(elem.int.type == .int);
    }
}

test "fully typed AST - heterogeneous vector fails" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    var allocator = arena.allocator();

    var reader = Reader.init(&allocator);
    var checker = TypeChecker.BidirectionalTypeChecker.init(&allocator);
    defer checker.deinit();

    // Test heterogeneous vector: [1 "hello" 3]
    const code = "[1 \"hello\" 3]";
    const expr = try reader.readString(code);
    const result = checker.synthesizeTyped(expr);

    // This should fail with TypeMismatch
    try std.testing.expect(result == TypeChecker.TypeCheckError.TypeMismatch);
}

test "fully typed AST - deeply nested expression" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    var allocator = arena.allocator();

    var reader = Reader.init(&allocator);
    var checker = TypeChecker.BidirectionalTypeChecker.init(&allocator);
    defer checker.deinit();

    // Test deeply nested: (+ (+ (+ 1 2) 3) 4)
    const code = "(+ (+ (+ 1 2) 3) 4)";
    const expr = try reader.readString(code);
    const typed = try checker.typeCheck(expr);

    // Top level should be list with type int
    try std.testing.expect(typed.* == .list);
    try std.testing.expect(typed.list.type == .int);

    // First argument is another list
    const first_arg = typed.list.elements[0];
    try std.testing.expect(first_arg.* == .list);
    try std.testing.expect(first_arg.list.type == .int);

    // That list's first argument is yet another list
    const inner_first = first_arg.list.elements[0];
    try std.testing.expect(inner_first.* == .list);
    try std.testing.expect(inner_first.list.type == .int);

    // The innermost list has two int arguments
    const innermost_first = inner_first.list.elements[0];
    try std.testing.expect(innermost_first.* == .int);
    try std.testing.expect(innermost_first.int.value == 1);
    try std.testing.expect(innermost_first.int.type == .int);

    const innermost_second = inner_first.list.elements[1];
    try std.testing.expect(innermost_second.* == .int);
    try std.testing.expect(innermost_second.int.value == 2);
    try std.testing.expect(innermost_second.int.type == .int);
}