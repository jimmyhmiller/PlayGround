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

test "bidirectional type checker - if expression" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    var allocator = arena.allocator();

    var reader = Reader.init(&allocator);
    var checker = BidirectionalTypeChecker.init(allocator);
    defer checker.deinit();

    // Valid if expression
    const code = "(def result (: Int) (if (< 1 2) 42 0))";
    const expr = try reader.readString(code);
    _ = try checker.typeCheck(expr);
    try std.testing.expect(checker.env.get("result").? == .int);

    // Branch type mismatch should fail
    const bad_code = "(def bad (: Int) (if (< 1 2) 1 \"nope\"))";
    const bad_expr = try reader.readString(bad_code);
    try std.testing.expectError(TypeChecker.TypeCheckError.TypeMismatch, checker.typeCheck(bad_expr));
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

    // Test struct definition: (def Point (: Type) (Struct [x Int] [y Int]))
    const code = "(def Point (: Type) (Struct [x Int] [y Int]))";
    const expr = try reader.readString(code);
    const typed = try checker.typeCheck(expr);

    // Should successfully type check - struct type definitions return type_value with Type type
    try std.testing.expect(typed.getType() == .type_type);

    // Check that the struct type is in the environment (Point has type Type)
    const point_type = checker.env.get("Point").?;
    try std.testing.expect(point_type == .type_type);

    // Check struct details from type_defs
    const actual_struct_type = checker.type_defs.get("Point").?;
    try std.testing.expect(actual_struct_type == .struct_type);
    const struct_type = actual_struct_type.struct_type;
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
    const struct_def = "(def Point (: Type) (Struct [x Int] [y Int]))";
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
        \\(def Point (: Type) (Struct [x Int] [y Int]))
        \\(def Color (: Type) (Struct [r Int] [g Int] [b Int]))
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

test "enum definition" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    var allocator = arena.allocator();

    var reader = Reader.init(&allocator);
    var checker = TypeChecker.BidirectionalTypeChecker.init(allocator);
    defer checker.deinit();

    const code = "(def Color (: Type) (Enum Red Blue Green))";
    const expr = try reader.readString(code);
    const typed = try checker.typeCheck(expr);

    try std.testing.expect(typed.getType() == .type_type);

    const color_type = checker.type_defs.get("Color").?;
    try std.testing.expect(color_type == .enum_type);

    const enum_type = color_type.enum_type;
    try std.testing.expect(std.mem.eql(u8, enum_type.name, "Color"));
    try std.testing.expect(enum_type.variants.len == 3);
    try std.testing.expect(std.mem.eql(u8, enum_type.variants[0].name, "Red"));
    try std.testing.expect(std.mem.eql(u8, enum_type.variants[1].name, "Blue"));
    try std.testing.expect(std.mem.eql(u8, enum_type.variants[2].name, "Green"));

    const red_type = checker.env.get("Color/Red").?;
    try std.testing.expect(red_type == .enum_type);
    try std.testing.expect(TypeChecker.BidirectionalTypeChecker.typesEqual(color_type, red_type));

    const blue_type = checker.env.get("Color/Blue").?;
    try std.testing.expect(TypeChecker.BidirectionalTypeChecker.typesEqual(color_type, blue_type));

    const green_type = checker.env.get("Color/Green").?;
    try std.testing.expect(TypeChecker.BidirectionalTypeChecker.typesEqual(color_type, green_type));
}

test "enum values and annotations" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    var allocator = arena.allocator();

    var reader = Reader.init(&allocator);
    var checker = TypeChecker.BidirectionalTypeChecker.init(allocator);
    defer checker.deinit();

    const enum_def = try reader.readString("(def Color (: Type) (Enum Red Blue Green))");
    _ = try checker.typeCheck(enum_def);

    const value_def = try reader.readString("(def favoriteColor (: Color) Color/Blue)");
    const value_typed = try checker.typeCheck(value_def);
    try std.testing.expect(value_typed.getType() == .enum_type);

    const invalid_value = try reader.readString("(def badColor (: Color) Color/Yellow)");
    try std.testing.expectError(TypeChecker.TypeCheckError.UnboundVariable, checker.typeCheck(invalid_value));
}

test "let binding basic" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    var allocator = arena.allocator();

    var reader = Reader.init(&allocator);
    var checker = TypeChecker.BidirectionalTypeChecker.init(allocator);
    defer checker.deinit();

    const code = "(let [x (: Int) 7] (+ x 2))";
    const expr = try reader.readString(code);
    const typed = try checker.typeCheck(expr);

    try std.testing.expect(typed.getType() == .int);
    try std.testing.expect(checker.env.get("x") == null);
}

test "let binding multiple" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    var allocator = arena.allocator();

    var reader = Reader.init(&allocator);
    var checker = TypeChecker.BidirectionalTypeChecker.init(allocator);
    defer checker.deinit();

    const code = "(let [x (: Int) 7 y (: Int) (+ x 3)] (+ x y))";
    const expr = try reader.readString(code);
    const typed = try checker.typeCheck(expr);

    try std.testing.expect(typed.getType() == .int);
}

test "let binding type mismatch" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    var allocator = arena.allocator();

    var reader = Reader.init(&allocator);
    var checker = TypeChecker.BidirectionalTypeChecker.init(allocator);
    defer checker.deinit();

    const code = "(let [x (: Int) \"oops\"] x)";
    const expr = try reader.readString(code);
    try std.testing.expectError(TypeChecker.TypeCheckError.TypeMismatch, checker.typeCheck(expr));
}

test "let binding nested" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    var allocator = arena.allocator();

    var reader = Reader.init(&allocator);
    var checker = TypeChecker.BidirectionalTypeChecker.init(allocator);
    defer checker.deinit();

    const code = "(let [x (: Int) 1] (let [y (: Int) (+ x 2)] (+ x y)))";
    const expr = try reader.readString(code);
    const typed = try checker.typeCheck(expr);

    try std.testing.expect(typed.getType() == .int);
    try std.testing.expect(checker.env.get("y") == null);
}

test "let binding shadow different type" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    var allocator = arena.allocator();

    var reader = Reader.init(&allocator);
    var checker = TypeChecker.BidirectionalTypeChecker.init(allocator);
    defer checker.deinit();

    const code = "(let [x (: Int) 1] (let [x (: String) \"hi\"] x))";
    const expr = try reader.readString(code);
    const typed = try checker.typeCheck(expr);

    try std.testing.expect(typed.getType() == .string);
}

test "let binding shadow incompatible" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    var allocator = arena.allocator();

    var reader = Reader.init(&allocator);
    var checker = TypeChecker.BidirectionalTypeChecker.init(allocator);
    defer checker.deinit();

    const code = "(let [x (: Int) 1] (let [x (: Int) \"nope\"] x))";
    const expr = try reader.readString(code);
    try std.testing.expectError(TypeChecker.TypeCheckError.TypeMismatch, checker.typeCheck(expr));
}

// ============================================================================
// POINTER TESTS
// ============================================================================

test "pointer - basic allocation and dereference" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    var allocator = arena.allocator();

    var reader = Reader.init(&allocator);
    var checker = TypeChecker.BidirectionalTypeChecker.init(allocator);
    defer checker.deinit();

    const code =
        \\(def x (: (Pointer Int)) (allocate Int 42))
        \\(dereference x)
    ;
    var expressions = try reader.readAllString(code);
    defer expressions.deinit(allocator);

    var report = try checker.typeCheckAllTwoPass(expressions.items);
    defer report.typed.deinit(allocator);
    defer report.errors.deinit(allocator);

    try std.testing.expect(report.errors.items.len == 0);
    // Last expression is (dereference x), which should return Int
    const last_typed = report.typed.items[report.typed.items.len - 1];
    try std.testing.expect(last_typed.getType() == .int);
}

test "pointer - write and read" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    var allocator = arena.allocator();

    var reader = Reader.init(&allocator);
    var checker = TypeChecker.BidirectionalTypeChecker.init(allocator);
    defer checker.deinit();

    const code =
        \\(def p (: (Pointer Int)) (allocate Int 10))
        \\(pointer-write! p 99)
        \\(dereference p)
    ;
    var expressions = try reader.readAllString(code);
    defer expressions.deinit(allocator);

    var report = try checker.typeCheckAllTwoPass(expressions.items);
    defer report.typed.deinit(allocator);
    defer report.errors.deinit(allocator);

    try std.testing.expect(report.errors.items.len == 0);
    const last_typed = report.typed.items[report.typed.items.len - 1];
    try std.testing.expect(last_typed.getType() == .int);
}

test "pointer - address-of operation" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    var allocator = arena.allocator();

    var reader = Reader.init(&allocator);
    var checker = TypeChecker.BidirectionalTypeChecker.init(allocator);
    defer checker.deinit();

    const code =
        \\(def get-address (: (-> [Int] (Pointer Int))) (fn [x] (address-of x)))
        \\(def test-val (: Int) 42)
        \\(get-address test-val)
    ;
    var expressions = try reader.readAllString(code);
    defer expressions.deinit(allocator);

    var report = try checker.typeCheckAllTwoPass(expressions.items);
    defer report.typed.deinit(allocator);
    defer report.errors.deinit(allocator);

    try std.testing.expect(report.errors.items.len == 0);
    const last_typed = report.typed.items[report.typed.items.len - 1];
    try std.testing.expect(last_typed.getType() == .pointer);
}

test "pointer - pointer-null" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    var allocator = arena.allocator();

    var reader = Reader.init(&allocator);
    var checker = TypeChecker.BidirectionalTypeChecker.init(allocator);
    defer checker.deinit();

    const code =
        \\(def p (: (Pointer Int)) pointer-null)
        \\p
    ;
    var expressions = try reader.readAllString(code);
    defer expressions.deinit(allocator);

    var report = try checker.typeCheckAllTwoPass(expressions.items);
    defer report.typed.deinit(allocator);
    defer report.errors.deinit(allocator);

    try std.testing.expect(report.errors.items.len == 0);
    const last_typed = report.typed.items[report.typed.items.len - 1];
    try std.testing.expect(last_typed.getType() == .pointer);
}

test "pointer - struct field read" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    var allocator = arena.allocator();

    var reader = Reader.init(&allocator);
    var checker = TypeChecker.BidirectionalTypeChecker.init(allocator);
    defer checker.deinit();

    const code =
        \\(def Point (: Type) (Struct [x Int] [y Int]))
        \\(def p (: (Pointer Point)) (allocate Point (Point 10 20)))
        \\(pointer-field-read p x)
    ;
    var expressions = try reader.readAllString(code);
    defer expressions.deinit(allocator);

    var report = try checker.typeCheckAllTwoPass(expressions.items);
    defer report.typed.deinit(allocator);
    defer report.errors.deinit(allocator);

    try std.testing.expect(report.errors.items.len == 0);
    const last_typed = report.typed.items[report.typed.items.len - 1];
    try std.testing.expect(last_typed.getType() == .int);
}

test "pointer - struct field write" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    var allocator = arena.allocator();

    var reader = Reader.init(&allocator);
    var checker = TypeChecker.BidirectionalTypeChecker.init(allocator);
    defer checker.deinit();

    const code =
        \\(def Point (: Type) (Struct [x Int] [y Int]))
        \\(def p (: (Pointer Point)) (allocate Point (Point 0 0)))
        \\(pointer-field-write! p x 42)
        \\(pointer-field-read p x)
    ;
    var expressions = try reader.readAllString(code);
    defer expressions.deinit(allocator);

    var report = try checker.typeCheckAllTwoPass(expressions.items);
    defer report.typed.deinit(allocator);
    defer report.errors.deinit(allocator);

    try std.testing.expect(report.errors.items.len == 0);
    const last_typed = report.typed.items[report.typed.items.len - 1];
    try std.testing.expect(last_typed.getType() == .int);
}

test "pointer - equality comparison" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    var allocator = arena.allocator();

    var reader = Reader.init(&allocator);
    var checker = TypeChecker.BidirectionalTypeChecker.init(allocator);
    defer checker.deinit();

    const code =
        \\(def p1 (: (Pointer Int)) (allocate Int 42))
        \\(def p2 (: (Pointer Int)) p1)
        \\(pointer-equal? p1 p2)
    ;
    var expressions = try reader.readAllString(code);
    defer expressions.deinit(allocator);

    var report = try checker.typeCheckAllTwoPass(expressions.items);
    defer report.typed.deinit(allocator);
    defer report.errors.deinit(allocator);

    try std.testing.expect(report.errors.items.len == 0);
    const last_typed = report.typed.items[report.typed.items.len - 1];
    try std.testing.expect(last_typed.getType() == .bool);
}

test "pointer - deallocate" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    var allocator = arena.allocator();

    var reader = Reader.init(&allocator);
    var checker = TypeChecker.BidirectionalTypeChecker.init(allocator);
    defer checker.deinit();

    const code =
        \\(def p (: (Pointer Int)) (allocate Int 42))
        \\(deallocate p)
    ;
    var expressions = try reader.readAllString(code);
    defer expressions.deinit(allocator);

    var report = try checker.typeCheckAllTwoPass(expressions.items);
    defer report.typed.deinit(allocator);
    defer report.errors.deinit(allocator);

    try std.testing.expect(report.errors.items.len == 0);
    const last_typed = report.typed.items[report.typed.items.len - 1];
    try std.testing.expect(last_typed.getType() == .nil);
}

test "pointer - nested pointers" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    var allocator = arena.allocator();

    var reader = Reader.init(&allocator);
    var checker = TypeChecker.BidirectionalTypeChecker.init(allocator);
    defer checker.deinit();

    const code =
        \\(def p (: (Pointer (Pointer Int))) (allocate (Pointer Int) (allocate Int 42)))
        \\(dereference (dereference p))
    ;
    var expressions = try reader.readAllString(code);
    defer expressions.deinit(allocator);

    var report = try checker.typeCheckAllTwoPass(expressions.items);
    defer report.typed.deinit(allocator);
    defer report.errors.deinit(allocator);

    try std.testing.expect(report.errors.items.len == 0);
    const last_typed = report.typed.items[report.typed.items.len - 1];
    try std.testing.expect(last_typed.getType() == .int);
}

test "pointer - function returning pointer" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    var allocator = arena.allocator();

    var reader = Reader.init(&allocator);
    var checker = TypeChecker.BidirectionalTypeChecker.init(allocator);
    defer checker.deinit();

    const code =
        \\(def make-pointer (: (-> [Int] (Pointer Int))) (fn [x] (allocate Int x)))
        \\(def p (: (Pointer Int)) (make-pointer 100))
        \\(dereference p)
    ;
    var expressions = try reader.readAllString(code);
    defer expressions.deinit(allocator);

    var report = try checker.typeCheckAllTwoPass(expressions.items);
    defer report.typed.deinit(allocator);
    defer report.errors.deinit(allocator);

    try std.testing.expect(report.errors.items.len == 0);
    const last_typed = report.typed.items[report.typed.items.len - 1];
    try std.testing.expect(last_typed.getType() == .int);
}

test "pointer - type checking catches non-pointer dereference" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    var allocator = arena.allocator();

    var reader = Reader.init(&allocator);
    var checker = TypeChecker.BidirectionalTypeChecker.init(allocator);
    defer checker.deinit();

    const code = "(dereference 42)";
    const expr = try reader.readString(code);

    // Should fail type checking - can't dereference an Int
    try std.testing.expectError(TypeChecker.TypeCheckError.TypeMismatch, checker.typeCheck(expr));
}

test "pointer - type checking catches wrong pointer type assignment" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    var allocator = arena.allocator();

    var reader = Reader.init(&allocator);
    var checker = TypeChecker.BidirectionalTypeChecker.init(allocator);
    defer checker.deinit();

    const code =
        \\(def p (: (Pointer Int)) (allocate Int 42))
        \\(pointer-write! p "string")
    ;
    var expressions = try reader.readAllString(code);
    defer expressions.deinit(allocator);

    var report = try checker.typeCheckAllTwoPass(expressions.items);
    defer report.typed.deinit(allocator);
    defer report.errors.deinit(allocator);

    // Should have errors - can't write String to Pointer Int
    try std.testing.expect(report.errors.items.len > 0);
}
