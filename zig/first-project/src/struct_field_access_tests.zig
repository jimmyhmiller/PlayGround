const std = @import("std");
const Reader = @import("reader.zig").Reader;
const TypeChecker = @import("type_checker.zig");

// ============================================================================
// STRUCT FIELD ACCESS TESTS (. syntax)
// ============================================================================

test "struct field access - basic field read" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    var allocator = arena.allocator();

    var reader = Reader.init(&allocator);
    var checker = TypeChecker.BidirectionalTypeChecker.init(allocator);
    defer checker.deinit();

    const code =
        \\(def Point (: Type) (Struct [x Int] [y Int]))
        \\(def p (: Point) (Point 10 20))
        \\(. p x)
    ;
    const read_result = try reader.readAllString(code);
    var expressions = read_result.values;
    defer expressions.deinit(allocator);

    var report = try checker.typeCheckAllTwoPass(expressions.items);
    defer report.typed.deinit(allocator);
    defer report.errors.deinit(allocator);

    try std.testing.expect(report.errors.items.len == 0);
    const last_typed = report.typed.items[report.typed.items.len - 1];
    try std.testing.expect(last_typed.getType() == .int);
}

test "struct field access - multiple fields" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    var allocator = arena.allocator();

    var reader = Reader.init(&allocator);
    var checker = TypeChecker.BidirectionalTypeChecker.init(allocator);
    defer checker.deinit();

    const code =
        \\(def Point (: Type) (Struct [x Int] [y Int]))
        \\(def p (: Point) (Point 5 7))
        \\(def x-val (: Int) (. p x))
        \\(def y-val (: Int) (. p y))
    ;
    const read_result = try reader.readAllString(code);
    var expressions = read_result.values;
    defer expressions.deinit(allocator);

    var report = try checker.typeCheckAllTwoPass(expressions.items);
    defer report.typed.deinit(allocator);
    defer report.errors.deinit(allocator);

    try std.testing.expect(report.errors.items.len == 0);
    try std.testing.expect(checker.env.get("x-val").? == .int);
    try std.testing.expect(checker.env.get("y-val").? == .int);
}

test "struct field access - in arithmetic expression" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    var allocator = arena.allocator();

    var reader = Reader.init(&allocator);
    var checker = TypeChecker.BidirectionalTypeChecker.init(allocator);
    defer checker.deinit();

    const code =
        \\(def Point (: Type) (Struct [x Int] [y Int]))
        \\(def p (: Point) (Point 3 4))
        \\(+ (. p x) (. p y))
    ;
    const read_result = try reader.readAllString(code);
    var expressions = read_result.values;
    defer expressions.deinit(allocator);

    var report = try checker.typeCheckAllTwoPass(expressions.items);
    defer report.typed.deinit(allocator);
    defer report.errors.deinit(allocator);

    try std.testing.expect(report.errors.items.len == 0);
    const last_typed = report.typed.items[report.typed.items.len - 1];
    try std.testing.expect(last_typed.getType() == .int);
}

test "struct field access - nested structs" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    var allocator = arena.allocator();

    var reader = Reader.init(&allocator);
    var checker = TypeChecker.BidirectionalTypeChecker.init(allocator);
    defer checker.deinit();

    const code =
        \\(def Point (: Type) (Struct [x Int] [y Int]))
        \\(def Rect (: Type) (Struct [topLeft Point] [width Int]))
        \\(def r (: Rect) (Rect (Point 0 0) 100))
        \\(. (. r topLeft) x)
    ;
    const read_result = try reader.readAllString(code);
    var expressions = read_result.values;
    defer expressions.deinit(allocator);

    var report = try checker.typeCheckAllTwoPass(expressions.items);
    defer report.typed.deinit(allocator);
    defer report.errors.deinit(allocator);

    try std.testing.expect(report.errors.items.len == 0);
    const last_typed = report.typed.items[report.typed.items.len - 1];
    try std.testing.expect(last_typed.getType() == .int);
}

test "struct field access - with let binding" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    var allocator = arena.allocator();

    var reader = Reader.init(&allocator);
    var checker = TypeChecker.BidirectionalTypeChecker.init(allocator);
    defer checker.deinit();

    const code =
        \\(def Point (: Type) (Struct [x Int] [y Int]))
        \\(let [p (: Point) (Point 8 15)] (. p x))
    ;
    const read_result = try reader.readAllString(code);
    var expressions = read_result.values;
    defer expressions.deinit(allocator);

    var report = try checker.typeCheckAllTwoPass(expressions.items);
    defer report.typed.deinit(allocator);
    defer report.errors.deinit(allocator);

    try std.testing.expect(report.errors.items.len == 0);
    const last_typed = report.typed.items[report.typed.items.len - 1];
    try std.testing.expect(last_typed.getType() == .int);
}

test "struct field access - mixed types" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    var allocator = arena.allocator();

    var reader = Reader.init(&allocator);
    var checker = TypeChecker.BidirectionalTypeChecker.init(allocator);
    defer checker.deinit();

    const code =
        \\(def Person (: Type) (Struct [name String] [age Int]))
        \\(def alice (: Person) (Person "Alice" 30))
        \\(def name-val (: String) (. alice name))
        \\(def age-val (: Int) (. alice age))
    ;
    const read_result = try reader.readAllString(code);
    var expressions = read_result.values;
    defer expressions.deinit(allocator);

    var report = try checker.typeCheckAllTwoPass(expressions.items);
    defer report.typed.deinit(allocator);
    defer report.errors.deinit(allocator);

    try std.testing.expect(report.errors.items.len == 0);
    try std.testing.expect(checker.env.get("name-val").? == .string);
    try std.testing.expect(checker.env.get("age-val").? == .int);
}

test "struct field access - in function" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    var allocator = arena.allocator();

    var reader = Reader.init(&allocator);
    var checker = TypeChecker.BidirectionalTypeChecker.init(allocator);
    defer checker.deinit();

    const code =
        \\(def Point (: Type) (Struct [x Int] [y Int]))
        \\(def getX (: (-> [Point] Int)) (fn [p] (. p x)))
        \\(def point (: Point) (Point 42 99))
        \\(getX point)
    ;
    const read_result = try reader.readAllString(code);
    var expressions = read_result.values;
    defer expressions.deinit(allocator);

    var report = try checker.typeCheckAllTwoPass(expressions.items);
    defer report.typed.deinit(allocator);
    defer report.errors.deinit(allocator);

    try std.testing.expect(report.errors.items.len == 0);
    const last_typed = report.typed.items[report.typed.items.len - 1];
    try std.testing.expect(last_typed.getType() == .int);
}

// ============================================================================
// NEGATIVE TESTS
// ============================================================================

test "struct field access - invalid field name" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    var allocator = arena.allocator();

    var reader = Reader.init(&allocator);
    var checker = TypeChecker.BidirectionalTypeChecker.init(allocator);
    defer checker.deinit();

    const code =
        \\(def Point (: Type) (Struct [x Int] [y Int]))
        \\(def p (: Point) (Point 1 2))
        \\(. p z)
    ;
    const read_result = try reader.readAllString(code);
    var expressions = read_result.values;
    defer expressions.deinit(allocator);

    var report = try checker.typeCheckAllTwoPass(expressions.items);
    defer report.typed.deinit(allocator);
    defer report.errors.deinit(allocator);

    // Should have error - field 'z' doesn't exist
    try std.testing.expect(report.errors.items.len > 0);
}

test "struct field access - on non-struct type" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    var allocator = arena.allocator();

    var reader = Reader.init(&allocator);
    var checker = TypeChecker.BidirectionalTypeChecker.init(allocator);
    defer checker.deinit();

    const code =
        \\(def x (: Int) 42)
        \\(. x foo)
    ;
    const read_result = try reader.readAllString(code);
    var expressions = read_result.values;
    defer expressions.deinit(allocator);

    var report = try checker.typeCheckAllTwoPass(expressions.items);
    defer report.typed.deinit(allocator);
    defer report.errors.deinit(allocator);

    // Should have error - can't access field on Int
    try std.testing.expect(report.errors.items.len > 0);
}

test "struct field access - on string type" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    var allocator = arena.allocator();

    var reader = Reader.init(&allocator);
    var checker = TypeChecker.BidirectionalTypeChecker.init(allocator);
    defer checker.deinit();

    const code =
        \\(def s (: String) "hello")
        \\(. s length)
    ;
    const read_result = try reader.readAllString(code);
    var expressions = read_result.values;
    defer expressions.deinit(allocator);

    var report = try checker.typeCheckAllTwoPass(expressions.items);
    defer report.typed.deinit(allocator);
    defer report.errors.deinit(allocator);

    // Should have error - can't access field on String
    try std.testing.expect(report.errors.items.len > 0);
}

// ============================================================================
// STRUCT FIELD MUTATION TESTS (set! with field access)
// ============================================================================

test "set! struct field - basic field write" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    var allocator = arena.allocator();

    var reader = Reader.init(&allocator);
    var checker = TypeChecker.BidirectionalTypeChecker.init(allocator);
    defer checker.deinit();

    const code =
        \\(def Point (: Type) (Struct [x Int] [y Int]))
        \\(def p (: Point) (Point 10 20))
        \\(set! (. p x) 30)
    ;
    const read_result = try reader.readAllString(code);
    var expressions = read_result.values;
    defer expressions.deinit(allocator);

    var report = try checker.typeCheckAllTwoPass(expressions.items);
    defer report.typed.deinit(allocator);
    defer report.errors.deinit(allocator);

    try std.testing.expect(report.errors.items.len == 0);
    const last_typed = report.typed.items[report.typed.items.len - 1];
    try std.testing.expect(last_typed.getType() == .void);
}

test "set! struct field - multiple field writes" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    var allocator = arena.allocator();

    var reader = Reader.init(&allocator);
    var checker = TypeChecker.BidirectionalTypeChecker.init(allocator);
    defer checker.deinit();

    const code =
        \\(def Point (: Type) (Struct [x Int] [y Int]))
        \\(def p (: Point) (Point 5 7))
        \\(set! (. p x) 100)
        \\(set! (. p y) 200)
    ;
    const read_result = try reader.readAllString(code);
    var expressions = read_result.values;
    defer expressions.deinit(allocator);

    var report = try checker.typeCheckAllTwoPass(expressions.items);
    defer report.typed.deinit(allocator);
    defer report.errors.deinit(allocator);

    try std.testing.expect(report.errors.items.len == 0);
}

test "set! struct field - type mismatch error" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    var allocator = arena.allocator();

    var reader = Reader.init(&allocator);
    var checker = TypeChecker.BidirectionalTypeChecker.init(allocator);
    defer checker.deinit();

    const code =
        \\(def Point (: Type) (Struct [x Int] [y Int]))
        \\(def p (: Point) (Point 10 20))
        \\(set! (. p x) "not an int")
    ;
    const read_result = try reader.readAllString(code);
    var expressions = read_result.values;
    defer expressions.deinit(allocator);

    var report = try checker.typeCheckAllTwoPass(expressions.items);
    defer report.typed.deinit(allocator);
    defer report.errors.deinit(allocator);

    // Should have type mismatch error
    try std.testing.expect(report.errors.items.len > 0);
}

test "set! struct field - nested struct field" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    var allocator = arena.allocator();

    var reader = Reader.init(&allocator);
    var checker = TypeChecker.BidirectionalTypeChecker.init(allocator);
    defer checker.deinit();

    const code =
        \\(def Point (: Type) (Struct [x Int] [y Int]))
        \\(def Line (: Type) (Struct [start Point] [end Point]))
        \\(def line (: Line) (Line (Point 0 0) (Point 10 10)))
        \\(set! (. (. line start) x) 5)
    ;
    const read_result = try reader.readAllString(code);
    var expressions = read_result.values;
    defer expressions.deinit(allocator);

    var report = try checker.typeCheckAllTwoPass(expressions.items);
    defer report.typed.deinit(allocator);
    defer report.errors.deinit(allocator);

    try std.testing.expect(report.errors.items.len == 0);
}

test "set! struct field - nonexistent field error" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    var allocator = arena.allocator();

    var reader = Reader.init(&allocator);
    var checker = TypeChecker.BidirectionalTypeChecker.init(allocator);
    defer checker.deinit();

    const code =
        \\(def Point (: Type) (Struct [x Int] [y Int]))
        \\(def p (: Point) (Point 10 20))
        \\(set! (. p z) 30)
    ;
    const read_result = try reader.readAllString(code);
    var expressions = read_result.values;
    defer expressions.deinit(allocator);

    var report = try checker.typeCheckAllTwoPass(expressions.items);
    defer report.typed.deinit(allocator);
    defer report.errors.deinit(allocator);

    // Should have error - field 'z' doesn't exist
    try std.testing.expect(report.errors.items.len > 0);
}

test "set! struct field - in do block" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    var allocator = arena.allocator();

    var reader = Reader.init(&allocator);
    var checker = TypeChecker.BidirectionalTypeChecker.init(allocator);
    defer checker.deinit();

    const code =
        \\(def Point (: Type) (Struct [x Int] [y Int]))
        \\(def p (: Point) (Point 10 20))
        \\(do
        \\  (set! (. p x) 30)
        \\  (set! (. p y) 40)
        \\  (+ (. p x) (. p y)))
    ;
    const read_result = try reader.readAllString(code);
    var expressions = read_result.values;
    defer expressions.deinit(allocator);

    var report = try checker.typeCheckAllTwoPass(expressions.items);
    defer report.typed.deinit(allocator);
    defer report.errors.deinit(allocator);

    try std.testing.expect(report.errors.items.len == 0);
    const last_typed = report.typed.items[report.typed.items.len - 1];
    try std.testing.expect(last_typed.getType() == .int);
}
