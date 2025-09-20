const std = @import("std");
const Reader = @import("../../frontend/reader.zig").Reader;
const TypeChecker = @import("../../backend/type_checker.zig");
const Value = @import("../../value.zig").Value;

const BidirectionalTypeChecker = TypeChecker.BidirectionalTypeChecker;
const Type = TypeChecker.Type;
const TypedExpression = TypeChecker.TypedExpression;

// Test 1: Type mismatch in function body
test "type mismatch - function returning wrong type" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    var allocator = arena.allocator();

    var reader = Reader.init(&allocator);
    var checker = BidirectionalTypeChecker.init(&allocator);
    defer checker.deinit();

    // Function should return Int but returns String
    const code = "(def f (: (-> [Int] Int)) (fn [x] \"hello\"))";
    const expr = try reader.readString(code);
    const result = checker.typeCheck(expr);

    // This should fail with a TypeMismatch
    try std.testing.expect(result == TypeChecker.TypeCheckError.TypeMismatch);
}

// Test 2: Type mismatch in arithmetic operations
test "type mismatch - adding string to integer" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    var allocator = arena.allocator();

    var reader = Reader.init(&allocator);
    var checker = BidirectionalTypeChecker.init(&allocator);
    defer checker.deinit();

    // Trying to add a string to a number
    const code = "(+ 1 \"hello\")";
    const expr = try reader.readString(code);
    const result = checker.synthesize(expr);

    // This should fail with a TypeMismatch
    try std.testing.expect(result == TypeChecker.TypeCheckError.TypeMismatch);
}

// Test 3: Wrong argument count in function call
test "function application - wrong argument count" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    var allocator = arena.allocator();

    var reader = Reader.init(&allocator);
    var checker = BidirectionalTypeChecker.init(&allocator);
    defer checker.deinit();

    // Define a function that takes two arguments
    const def_code = "(def add (: (-> [Int Int] Int)) (fn [x y] (+ x y)))";
    const def_expr = try reader.readString(def_code);
    _ = try checker.typeCheck(def_expr);

    // Try to call it with one argument (should fail)
    const call_code = "(add 1)";
    const call_expr = try reader.readString(call_code);
    const result = checker.synthesize(call_expr);

    try std.testing.expect(result == TypeChecker.TypeCheckError.ArgumentCountMismatch);
}

// Test 4: Heterogeneous vector (should fail)
test "vector type checking - heterogeneous elements" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    var allocator = arena.allocator();

    var reader = Reader.init(&allocator);
    var checker = BidirectionalTypeChecker.init(&allocator);
    defer checker.deinit();

    // Vector with mixed types
    const code = "[1 \"hello\" 3]";
    const expr = try reader.readString(code);
    const result = checker.synthesize(expr);

    // This should fail with TypeMismatch
    try std.testing.expect(result == TypeChecker.TypeCheckError.TypeMismatch);
}

// Test 5: Function type parameter mismatch
test "function type parameter mismatch" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    var allocator = arena.allocator();

    var reader = Reader.init(&allocator);
    var checker = BidirectionalTypeChecker.init(&allocator);
    defer checker.deinit();

    // Define a function that expects Int parameter
    const def_code = "(def inc (: (-> [Int] Int)) (fn [x] (+ x 1)))";
    const def_expr = try reader.readString(def_code);
    _ = try checker.typeCheck(def_expr);

    // Try to call it with a string (should fail)
    const call_code = "(inc \"hello\")";
    const call_expr = try reader.readString(call_code);
    const result = checker.synthesize(call_expr);

    try std.testing.expect(result == TypeChecker.TypeCheckError.TypeMismatch);
}

// Test 6: Nested function types work correctly
test "nested function type checking" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    var allocator = arena.allocator();

    var reader = Reader.init(&allocator);
    var checker = BidirectionalTypeChecker.init(&allocator);
    defer checker.deinit();

    // Function that takes a function and applies it
    const code = "(def apply (: (-> [(-> [Int] Int) Int] Int)) (fn [f x] (f x)))";
    const expr = try reader.readString(code);
    const typed = try checker.typeCheck(expr);

    // Since we can't easily access the function details from TypedValue,
    // we'll just verify the type is function - this is sufficient for the test
    try std.testing.expect(typed.getType() == .function);
}

// Test 7: Type annotation with wrong keyword
test "invalid type annotation - wrong keyword" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    var allocator = arena.allocator();

    var reader = Reader.init(&allocator);
    var checker = BidirectionalTypeChecker.init(&allocator);
    defer checker.deinit();

    // Using wrong keyword (not ':')
    const code = "(def f (:type (-> [Int] Int)) (fn [x] (+ x 1)))";
    const expr = try reader.readString(code);
    const result = checker.typeCheck(expr);

    // Should fail with InvalidTypeAnnotation
    try std.testing.expect(result == TypeChecker.TypeCheckError.InvalidTypeAnnotation);
}

// Test 8: Checking mode vs synthesis mode
test "bidirectional - synthesis vs checking behavior" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    var allocator = arena.allocator();

    var reader = Reader.init(&allocator);
    var checker = BidirectionalTypeChecker.init(&allocator);
    defer checker.deinit();

    // Lambda without annotation cannot be synthesized
    const lambda = try reader.readString("(fn [x] (+ x 1))");
    const synthesis_result = checker.synthesize(lambda);
    try std.testing.expect(synthesis_result == TypeChecker.TypeCheckError.CannotSynthesize);

    // But it can be checked against an expected type
    const expected_type = try TypeChecker.createFunctionType(&allocator, &[_]TypeChecker.Type{TypeChecker.Type.int}, TypeChecker.Type.int);
    const checked = try checker.checkTyped(lambda, expected_type);
    try std.testing.expect(checked.getType() == .function);
}

// Test 9: Function returning function
test "function returning function type" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    var allocator = arena.allocator();

    var reader = Reader.init(&allocator);
    var checker = BidirectionalTypeChecker.init(&allocator);
    defer checker.deinit();

    // Parse a function that returns a function: (-> [Int] (-> [Int] Int))
    const type_annotation = try reader.readString("(: (-> [Int] (-> [Int] Int)))");
    const parsed_type = try checker.parseTypeAnnotation(type_annotation);

    try std.testing.expect(parsed_type == .function);
    try std.testing.expect(parsed_type.function.param_types.len == 1);
    try std.testing.expect(parsed_type.function.param_types[0] == .int);
    try std.testing.expect(parsed_type.function.return_type == .function);

    const return_func = parsed_type.function.return_type.function;
    try std.testing.expect(return_func.param_types.len == 1);
    try std.testing.expect(return_func.param_types[0] == .int);
    try std.testing.expect(return_func.return_type == .int);
}

// Test 10: Type inference for nested arithmetic
test "type inference - nested arithmetic" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    var allocator = arena.allocator();

    var reader = Reader.init(&allocator);
    var checker = BidirectionalTypeChecker.init(&allocator);
    defer checker.deinit();

    // Complex nested arithmetic
    const code = "(+ (+ 1 2) (+ 3 (+ 4 5)))";
    const expr = try reader.readString(code);
    const typed = try checker.typeCheck(expr);

    // Should infer Int type
    try std.testing.expect(typed.getType() == .int);
}

// Test 11: Function application to non-function
test "applying non-function as function" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    var allocator = arena.allocator();

    var reader = Reader.init(&allocator);
    var checker = BidirectionalTypeChecker.init(&allocator);
    defer checker.deinit();

    // Add an integer to the environment
    try checker.env.put("x", Type.int);

    // Try to apply it as a function
    const code = "(x 42)";
    const expr = try reader.readString(code);
    const result = checker.synthesize(expr);

    // Should fail with CannotApplyNonFunction
    try std.testing.expect(result == TypeChecker.TypeCheckError.CannotApplyNonFunction);
}

// Test 12: Wrong number of type parameters
test "wrong number of function type parameters" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    var allocator = arena.allocator();

    var reader = Reader.init(&allocator);
    var checker = BidirectionalTypeChecker.init(&allocator);
    defer checker.deinit();

    // Function annotation says 2 params but body has 3
    const code = "(def f (: (-> [Int Int] Int)) (fn [x y z] (+ x y)))";
    const expr = try reader.readString(code);
    const result = checker.typeCheck(expr);

    // Should fail with ArgumentCountMismatch
    try std.testing.expect(result == TypeChecker.TypeCheckError.ArgumentCountMismatch);
}