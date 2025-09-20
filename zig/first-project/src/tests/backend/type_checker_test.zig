const std = @import("std");
const Reader = @import("../../frontend/reader.zig").Reader;
const TypeChecker = @import("../../backend/type_checker.zig");
const Value = @import("../../value.zig").Value;

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
        try std.testing.expect(typed.getType() == .int);
    }

    // Test string synthesis
    {
        const expr = try reader.readString("\"hello\"");
        const typed = try checker.synthesize(expr);
        try std.testing.expect(typed.getType() == .string);
    }

    // Test float synthesis
    {
        const expr = try reader.readString("3.14");
        const typed = try checker.synthesize(expr);
        try std.testing.expect(typed.getType() == .float);
    }

    // Test nil synthesis
    {
        const expr = try reader.readString("nil");
        const typed = try checker.synthesize(expr);
        try std.testing.expect(typed.getType() == .nil);
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
    try std.testing.expect(typed.type.function.param_types.len == 1);
    try std.testing.expect(typed.type.function.param_types[0] == .int);
    try std.testing.expect(typed.type.function.return_type == .int);
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
        try std.testing.expect(typed.getType() == .int);
    }

    // Test nested arithmetic: (+ 1 (+ 2 3))
    {
        const expr = try reader.readString("(+ 1 (+ 2 3))");
        const typed = try checker.synthesize(expr);
        try std.testing.expect(typed.getType() == .int);
    }
}

test "bidirectional type checker - vectors" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    var allocator = arena.allocator();

    var reader = Reader.init(&allocator);
    var checker = BidirectionalTypeChecker.init(&allocator);
    defer checker.deinit();

    // Test homogeneous vector: [1 2 3]
    {
        const expr = try reader.readString("[1 2 3]");
        const typed = try checker.synthesize(expr);
        try std.testing.expect(typed.getType() == .vector);
        try std.testing.expect(typed.type.vector.* == .int);
    }

    // Test empty vector: []
    {
        const expr = try reader.readString("[]");
        const typed = try checker.synthesize(expr);
        try std.testing.expect(typed.getType() == .vector);
        // Empty vector should have a type variable for element type
    }
}

test "bidirectional type checker - function types parsing" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    var allocator = arena.allocator();

    var reader = Reader.init(&allocator);
    var checker = BidirectionalTypeChecker.init(&allocator);
    defer checker.deinit();

    // Test parsing simple function type annotation
    const type_annotation = try reader.readString("(: (-> [Int] Int))");
    const parsed_type = try checker.parseTypeAnnotation(type_annotation);

    try std.testing.expect(parsed_type == .function);
    try std.testing.expect(parsed_type.function.param_types.len == 1);
    try std.testing.expect(parsed_type.function.param_types[0] == .int);
    try std.testing.expect(parsed_type.function.return_type == .int);
}

test "bidirectional type checker - multi-parameter functions" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    var allocator = arena.allocator();

    var reader = Reader.init(&allocator);
    var checker = BidirectionalTypeChecker.init(&allocator);
    defer checker.deinit();

    // Test function with multiple parameters: (def add (: (-> [Int Int] Int)) (fn [x y] (+ x y)))
    const code = "(def add (: (-> [Int Int] Int)) (fn [x y] (+ x y)))";
    const expr = try reader.readString(code);
    const typed = try checker.typeCheck(expr);

    try std.testing.expect(typed.getType() == .function);
    try std.testing.expect(typed.type.function.param_types.len == 2);
    try std.testing.expect(typed.type.function.param_types[0] == .int);
    try std.testing.expect(typed.type.function.param_types[1] == .int);
    try std.testing.expect(typed.type.function.return_type == .int);
}

test "bidirectional type checker - higher-order functions" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    var allocator = arena.allocator();

    var reader = Reader.init(&allocator);
    var checker = BidirectionalTypeChecker.init(&allocator);
    defer checker.deinit();

    // Test higher-order function type: (-> [(-> [Int] Int)] Int)
    const type_annotation = try reader.readString("(: (-> [(-> [Int] Int)] Int))");
    const parsed_type = try checker.parseTypeAnnotation(type_annotation);

    try std.testing.expect(parsed_type == .function);
    try std.testing.expect(parsed_type.function.param_types.len == 1);
    try std.testing.expect(parsed_type.function.param_types[0] == .function);
    try std.testing.expect(parsed_type.function.return_type == .int);

    // The parameter should be a function from Int to Int
    const param_func = parsed_type.function.param_types[0].function;
    try std.testing.expect(param_func.param_types.len == 1);
    try std.testing.expect(param_func.param_types[0] == .int);
    try std.testing.expect(param_func.return_type == .int);
}

test "bidirectional type checker - variable lookup" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    var allocator = arena.allocator();

    var reader = Reader.init(&allocator);
    var checker = BidirectionalTypeChecker.init(&allocator);
    defer checker.deinit();

    // Add a variable to the environment
    try checker.env.put("x", Type.int);

    // Test variable lookup
    const expr = try reader.readString("x");
    const typed = try checker.synthesize(expr);
    try std.testing.expect(typed.getType() == .int);
}

test "bidirectional type checker - error cases" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    var allocator = arena.allocator();

    var reader = Reader.init(&allocator);
    var checker = BidirectionalTypeChecker.init(&allocator);
    defer checker.deinit();

    // Test unbound variable
    {
        const expr = try reader.readString("unbound_var");
        const result = checker.synthesize(expr);
        try std.testing.expect(result == TypeChecker.TypeCheckError.UnboundVariable);
    }

    // Test invalid type annotation
    {
        const expr = try reader.readString("(def f (: InvalidType) 42)");
        const result = checker.typeCheck(expr);
        try std.testing.expect(result == TypeChecker.TypeCheckError.InvalidTypeAnnotation);
    }
}

test "bidirectional type checker - checking mode" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    var allocator = arena.allocator();

    var reader = Reader.init(&allocator);
    var checker = BidirectionalTypeChecker.init(&allocator);
    defer checker.deinit();

    // Test checking an integer against Int type
    {
        const expr = try reader.readString("42");
        const typed = try checker.check(expr, Type.int);
        try std.testing.expect(typed.getType() == .int);
    }

    // Test checking a function against function type
    {
        const func_type = try TypeChecker.createFunctionType(&allocator, &[_]TypeChecker.Type{TypeChecker.Type.int}, TypeChecker.Type.int);
        const expr = try reader.readString("(fn [x] (+ x 1))");
        const typed = try checker.check(expr, func_type);
        try std.testing.expect(typed.getType() == .function);
    }
}

test "bidirectional type checker - type equality" {
    // Test basic type equality
    try std.testing.expect(TypeChecker.BidirectionalTypeChecker.typesEqual(TypeChecker.Type.int, TypeChecker.Type.int));
    try std.testing.expect(!TypeChecker.BidirectionalTypeChecker.typesEqual(TypeChecker.Type.int, TypeChecker.Type.string));

    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    var allocator = arena.allocator();

    // Test function type equality
    const func1 = try TypeChecker.createFunctionType(&allocator, &[_]TypeChecker.Type{TypeChecker.Type.int}, TypeChecker.Type.int);
    const func2 = try TypeChecker.createFunctionType(&allocator, &[_]TypeChecker.Type{TypeChecker.Type.int}, TypeChecker.Type.int);
    const func3 = try TypeChecker.createFunctionType(&allocator, &[_]TypeChecker.Type{TypeChecker.Type.string}, TypeChecker.Type.int);

    try std.testing.expect(TypeChecker.BidirectionalTypeChecker.typesEqual(func1, func2));
    try std.testing.expect(!TypeChecker.BidirectionalTypeChecker.typesEqual(func1, func3));
}

test "bidirectional type checker - complex example" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    var allocator = arena.allocator();

    var reader = Reader.init(&allocator);
    var checker = BidirectionalTypeChecker.init(&allocator);
    defer checker.deinit();

    // Test the main example from the prompt
    const code = "(def f (: (-> [Int] Int)) (fn [x] (+ x 1)))";
    const expr = try reader.readString(code);

    // This should successfully type check
    const typed = try checker.typeCheck(expr);
    try std.testing.expect(typed.getType() == .function);

    // Verify that 'f' is now in the environment with correct type
    const f_type = checker.env.get("f");
    try std.testing.expect(f_type != null);
    try std.testing.expect(f_type.? == .function);
    try std.testing.expect(f_type.?.function.param_types.len == 1);
    try std.testing.expect(f_type.?.function.param_types[0] == .int);
    try std.testing.expect(f_type.?.function.return_type == .int);
}