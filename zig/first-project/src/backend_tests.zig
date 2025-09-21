const std = @import("std");
const Reader = @import("reader.zig").Reader;
const TypeChecker = @import("type_checker.zig");
const Value = @import("value.zig").Value;

const BidirectionalTypeChecker = TypeChecker.BidirectionalTypeChecker;
const Type = TypeChecker.Type;
const TypedExpression = TypeChecker.TypedExpression;

test "bidirectional type checker - basic types" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    var allocator = arena.allocator();

    var reader = Reader.init(&allocator);
    var checker = BidirectionalTypeChecker.init(allocator);
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
    var checker = BidirectionalTypeChecker.init(allocator);
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
    var checker = BidirectionalTypeChecker.init(allocator);
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
    var checker = BidirectionalTypeChecker.init(allocator);
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
    var checker = TypeChecker.BidirectionalTypeChecker.init(allocator);
    defer checker.deinit();

    // This would fail with single-pass but should work with two-pass
    const code =
        \\(def x (: Int) y)
        \\(def y (: Int) 42)
    ;

    const expressions = try reader.readAllString(code);

    // Single-pass should fail
    var single_pass_checker = TypeChecker.BidirectionalTypeChecker.init(allocator);
    defer single_pass_checker.deinit();
    const single_pass_result = single_pass_checker.typeCheckAll(expressions.items);
    try std.testing.expect(single_pass_result == TypeChecker.TypeCheckError.UnboundVariable);

    // Two-pass should succeed
    const two_pass_report = try checker.typeCheckAllTwoPass(expressions.items);
    try std.testing.expect(two_pass_report.errors.items.len == 0);
    try std.testing.expect(two_pass_report.typed.items.len == 2);
    try std.testing.expect(checker.env.get("x").? == .int);
    try std.testing.expect(checker.env.get("y").? == .int);
}

test "forward references - function calling forward function" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    var allocator = arena.allocator();

    var reader = Reader.init(&allocator);
    var checker = TypeChecker.BidirectionalTypeChecker.init(allocator);
    defer checker.deinit();

    const code =
        \\(def f (: (-> [Int] Int)) (fn [x] (g x)))
        \\(def g (: (-> [Int] Int)) (fn [x] (+ x 1)))
    ;

    const expressions = try reader.readAllString(code);
    const typed_report = try checker.typeCheckAllTwoPass(expressions.items);

    try std.testing.expect(typed_report.errors.items.len == 0);
    try std.testing.expect(typed_report.typed.items.len == 2);
    try std.testing.expect(checker.env.get("f").? == .function);
    try std.testing.expect(checker.env.get("g").? == .function);
}

test "forward references - mutual recursion" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    var allocator = arena.allocator();

    var reader = Reader.init(&allocator);
    var checker = TypeChecker.BidirectionalTypeChecker.init(allocator);
    defer checker.deinit();

    // Simplified mutual recursion (without proper conditional logic)
    const code =
        \\(def even (: (-> [Int] Int)) (fn [n] (odd n)))
        \\(def odd (: (-> [Int] Int)) (fn [n] (even n)))
    ;

    const expressions = try reader.readAllString(code);
    const typed_report = try checker.typeCheckAllTwoPass(expressions.items);

    try std.testing.expect(typed_report.errors.items.len == 0);
    try std.testing.expect(typed_report.typed.items.len == 2);
    try std.testing.expect(checker.env.get("even").? == .function);
    try std.testing.expect(checker.env.get("odd").? == .function);
}

test "forward references - complex dependency chain" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    var allocator = arena.allocator();

    var reader = Reader.init(&allocator);
    var checker = TypeChecker.BidirectionalTypeChecker.init(allocator);
    defer checker.deinit();

    const code =
        \\(def a (: Int) b)
        \\(def b (: Int) c)
        \\(def c (: Int) d)
        \\(def d (: Int) 42)
    ;

    const expressions = try reader.readAllString(code);
    const typed_report = try checker.typeCheckAllTwoPass(expressions.items);

    try std.testing.expect(typed_report.errors.items.len == 0);
    try std.testing.expect(typed_report.typed.items.len == 4);
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
    var checker = TypeChecker.BidirectionalTypeChecker.init(allocator);
    defer checker.deinit();

    // Mutual recursion with incompatible types - should fail
    const code =
        \\(def evenCheck (: (-> [Int] String)) (fn [n] (oddCheck n)))
        \\(def oddCheck (: (-> [Int] Int)) (fn [n] (evenCheck n)))
    ;

    const expressions = try reader.readAllString(code);

    // This should fail because evenCheck returns String but oddCheck expects Int from evenCheck call
    const report = try checker.typeCheckAllTwoPass(expressions.items);
    try std.testing.expect(report.errors.items.len == 2);
    try std.testing.expect(report.errors.items[0].err == TypeChecker.TypeCheckError.TypeMismatch);
    try std.testing.expect(report.errors.items[1].err == TypeChecker.TypeCheckError.TypeMismatch);
}

test "struct definition" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    var allocator = arena.allocator();

    var reader = Reader.init(&allocator);
    var checker = TypeChecker.BidirectionalTypeChecker.init(allocator);
    defer checker.deinit();

    // Test struct definition: (def Point (Struct [x Int] [y Int]))
    const code = "(def Point (Struct [x Int] [y Int]))";
    const expr = try reader.readString(code);
    const typed = try checker.typeCheck(expr);

    // Should successfully type check - struct type definitions return type_value with Type type
    try std.testing.expect(typed.getType() == .type_type);

    // Check that the struct type is in the environment
    const point_type = checker.env.get("Point").?;
    try std.testing.expect(point_type == .struct_type);

    // Check struct details
    const struct_type = point_type.struct_type;
    try std.testing.expect(std.mem.eql(u8, struct_type.name, "Point"));
    try std.testing.expect(struct_type.fields.len == 2);
    try std.testing.expect(std.mem.eql(u8, struct_type.fields[0].name, "x"));
    try std.testing.expect(struct_type.fields[0].field_type == .int);
    try std.testing.expect(std.mem.eql(u8, struct_type.fields[1].name, "y"));
    try std.testing.expect(struct_type.fields[1].field_type == .int);
}

test "structs as function arguments and return types" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    var allocator = arena.allocator();

    var reader = Reader.init(&allocator);
    var checker = TypeChecker.BidirectionalTypeChecker.init(allocator);
    defer checker.deinit();

    // First define a struct
    const struct_def = "(def Point (Struct [x Int] [y Int]))";
    const struct_expr = try reader.readString(struct_def);
    _ = try checker.typeCheck(struct_expr);

    // Test function that takes struct as parameter and returns struct
    const func_code = "(def movePoint (: (-> [Point] Point)) (fn [p] p))";
    const func_expr = try reader.readString(func_code);
    const func_typed = try checker.typeCheck(func_expr);

    // Should successfully type check as a function type
    try std.testing.expect(func_typed.getType() == .function);

    // Check function type details
    const func_type = func_typed.getType().function;
    try std.testing.expect(func_type.param_types.len == 1);
    try std.testing.expect(func_type.param_types[0] == .struct_type);
    try std.testing.expect(func_type.return_type == .struct_type);

    // Verify the parameter and return types are the Point struct
    const param_struct = func_type.param_types[0].struct_type;
    const return_struct = func_type.return_type.struct_type;
    try std.testing.expect(std.mem.eql(u8, param_struct.name, "Point"));
    try std.testing.expect(std.mem.eql(u8, return_struct.name, "Point"));
}

test "multiple structs in function signatures" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    var allocator = arena.allocator();

    var reader = Reader.init(&allocator);
    var checker = TypeChecker.BidirectionalTypeChecker.init(allocator);
    defer checker.deinit();

    // Define two different struct types
    const struct_defs =
        \\(def Point (Struct [x Int] [y Int]))
        \\(def Color (Struct [r Int] [g Int] [b Int]))
    ;

    const expressions = try reader.readAllString(struct_defs);
    for (expressions.items) |expr| {
        _ = try checker.typeCheck(expr);
    }

    // Test function that takes two different structs and returns one
    const func_code = "(def colorPoint (: (-> [Point Color] Point)) (fn [p c] p))";
    const func_expr = try reader.readString(func_code);
    const func_typed = try checker.typeCheck(func_expr);

    // Should successfully type check as a function type
    try std.testing.expect(func_typed.getType() == .function);

    // Check function type details
    const func_type = func_typed.getType().function;
    try std.testing.expect(func_type.param_types.len == 2);
    try std.testing.expect(func_type.param_types[0] == .struct_type);
    try std.testing.expect(func_type.param_types[1] == .struct_type);
    try std.testing.expect(func_type.return_type == .struct_type);

    // Verify the parameter types are correct structs
    const point_param = func_type.param_types[0].struct_type;
    const color_param = func_type.param_types[1].struct_type;
    const point_return = func_type.return_type.struct_type;

    try std.testing.expect(std.mem.eql(u8, point_param.name, "Point"));
    try std.testing.expect(point_param.fields.len == 2);

    try std.testing.expect(std.mem.eql(u8, color_param.name, "Color"));
    try std.testing.expect(color_param.fields.len == 3);

    try std.testing.expect(std.mem.eql(u8, point_return.name, "Point"));
}
