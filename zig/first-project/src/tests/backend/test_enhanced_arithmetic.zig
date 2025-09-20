const std = @import("std");
const Reader = @import("../../frontend/reader.zig").Reader;
const TypeChecker = @import("../../backend/type_checker.zig");

test "enhanced arithmetic - specific numeric types" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    var allocator = arena.allocator();

    var reader = Reader.init(&allocator);
    var checker = TypeChecker.BidirectionalTypeChecker.init(&allocator);
    defer checker.deinit();

    // Test U8 arithmetic
    const code1 = "(def x (: U8) 255)";
    const expr1 = try reader.readString(code1);
    const typed1 = try checker.typeCheck(expr1);
    try std.testing.expect(typed1.getType() == .u8);

    // Test I32 arithmetic
    const code2 = "(def y (: I32) -42)";
    const expr2 = try reader.readString(code2);
    const typed2 = try checker.typeCheck(expr2);
    try std.testing.expect(typed2.getType() == .i32);

    // Test F64 arithmetic
    const code3 = "(def z (: F64) 3.14159)";
    const expr3 = try reader.readString(code3);
    const typed3 = try checker.typeCheck(expr3);
    try std.testing.expect(typed3.getType() == .f64);
}

test "enhanced arithmetic - all operators with integers" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    var allocator = arena.allocator();

    var reader = Reader.init(&allocator);
    var checker = TypeChecker.BidirectionalTypeChecker.init(&allocator);
    defer checker.deinit();

    // Set up variables
    try checker.env.put("a", TypeChecker.Type.u32);
    try checker.env.put("b", TypeChecker.Type.u32);

    // Test addition
    const add_code = "(+ a b)";
    const add_expr = try reader.readString(add_code);
    const add_typed = try checker.typeCheck(add_expr);
    try std.testing.expect(add_typed.getType() == .u32);

    // Test subtraction
    const sub_code = "(- a b)";
    const sub_expr = try reader.readString(sub_code);
    const sub_typed = try checker.typeCheck(sub_expr);
    try std.testing.expect(sub_typed.getType() == .u32);

    // Test multiplication
    const mul_code = "(* a b)";
    const mul_expr = try reader.readString(mul_code);
    const mul_typed = try checker.typeCheck(mul_expr);
    try std.testing.expect(mul_typed.getType() == .u32);

    // Test modulo
    const mod_code = "(% a b)";
    const mod_expr = try reader.readString(mod_code);
    const mod_typed = try checker.typeCheck(mod_expr);
    try std.testing.expect(mod_typed.getType() == .u32);
}

test "enhanced arithmetic - division type promotion" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    var allocator = arena.allocator();

    var reader = Reader.init(&allocator);
    var checker = TypeChecker.BidirectionalTypeChecker.init(&allocator);
    defer checker.deinit();

    // Set up integer variables
    try checker.env.put("x", TypeChecker.Type.i32);
    try checker.env.put("y", TypeChecker.Type.i32);

    // Integer division should produce F64
    const div_code = "(/ x y)";
    const div_expr = try reader.readString(div_code);
    const div_typed = try checker.typeCheck(div_expr);
    try std.testing.expect(div_typed.getType() == .f64);

    // Set up float variables
    try checker.env.put("a", TypeChecker.Type.f32);
    try checker.env.put("b", TypeChecker.Type.f32);

    // Float division should stay float
    const fdiv_code = "(/ a b)";
    const fdiv_expr = try reader.readString(fdiv_code);
    const fdiv_typed = try checker.typeCheck(fdiv_expr);
    try std.testing.expect(fdiv_typed.getType() == .f32);
}

test "enhanced arithmetic - type mismatches" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    var allocator = arena.allocator();

    var reader = Reader.init(&allocator);
    var checker = TypeChecker.BidirectionalTypeChecker.init(&allocator);
    defer checker.deinit();

    // Mixed types should fail
    try checker.env.put("int_val", TypeChecker.Type.i32);
    try checker.env.put("float_val", TypeChecker.Type.f32);

    const mixed_code = "(+ int_val float_val)";
    const mixed_expr = try reader.readString(mixed_code);
    const mixed_result = checker.synthesize(mixed_expr);
    try std.testing.expect(mixed_result == TypeChecker.TypeCheckError.TypeMismatch);

    // Modulo with float should fail
    const mod_float_code = "(% float_val float_val)";
    const mod_float_expr = try reader.readString(mod_float_code);
    const mod_float_result = checker.synthesize(mod_float_expr);
    try std.testing.expect(mod_float_result == TypeChecker.TypeCheckError.TypeMismatch);

    // Non-numeric types should fail
    try checker.env.put("str_val", TypeChecker.Type.string);
    const non_numeric_code = "(+ str_val int_val)";
    const non_numeric_expr = try reader.readString(non_numeric_code);
    const non_numeric_result = checker.synthesize(non_numeric_expr);
    try std.testing.expect(non_numeric_result == TypeChecker.TypeCheckError.TypeMismatch);
}

test "enhanced arithmetic - complex expressions" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    var allocator = arena.allocator();

    var reader = Reader.init(&allocator);
    var checker = TypeChecker.BidirectionalTypeChecker.init(&allocator);
    defer checker.deinit();

    // Complex nested arithmetic with consistent types
    const code =
        \\(def a (: I64) 10)
        \\(def b (: I64) 20)
        \\(def c (: I64) 30)
        \\(def result (: I64) (+ a (* b c)))
    ;

    const expressions = try reader.readAllString(code);
    const typed = try checker.typeCheckAll(expressions.items);

    // All should succeed
    try std.testing.expect(typed.items.len == 4);
    try std.testing.expect(checker.env.get("a").? == .i64);
    try std.testing.expect(checker.env.get("b").? == .i64);
    try std.testing.expect(checker.env.get("c").? == .i64);
    try std.testing.expect(checker.env.get("result").? == .i64);
}

test "enhanced arithmetic - fully typed AST with operators" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    var allocator = arena.allocator();

    var reader = Reader.init(&allocator);
    var checker = TypeChecker.BidirectionalTypeChecker.init(&allocator);
    defer checker.deinit();

    const code =
        \\(def x (: U16) 100)
        \\(def y (: U16) 50)
        \\(- x y)
    ;

    const expressions = try reader.readAllString(code);
    const typed_asts = try checker.typeCheckAll(expressions.items);

    // Third expression should be subtraction with U16 operands
    const sub_expr = typed_asts.items[2];
    try std.testing.expect(sub_expr.getType() == .u16);
    try std.testing.expect(sub_expr.* == .list);

    const list = sub_expr.list;
    try std.testing.expect(list.elements.len == 2);

    // Both operands should have U16 type
    try std.testing.expect(list.elements[0].* == .symbol);
    try std.testing.expect(list.elements[0].symbol.getType() == .u16);
    try std.testing.expect(list.elements[1].* == .symbol);
    try std.testing.expect(list.elements[1].symbol.getType() == .u16);
}