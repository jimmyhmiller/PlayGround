const std = @import("std");
const reader = @import("reader.zig");
const type_checker = @import("type_checker.zig");
const simple_c_compiler = @import("simple_c_compiler.zig");

const Reader = reader.Reader;
const BidirectionalTypeChecker = type_checker.BidirectionalTypeChecker;
const SimpleCCompiler = simple_c_compiler.SimpleCCompiler;

// ============================================================================
// Bitwise AND Tests
// ============================================================================

test "bitwise - AND basic" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    var allocator = arena.allocator();

    var r = Reader.init(&allocator);
    var checker = BidirectionalTypeChecker.init(allocator);
    defer checker.deinit();

    const expr = try r.readString("(bitwise-and 12 10)");
    const typed = try checker.synthesizeTyped(expr);
    try std.testing.expect(typed.getType() == .int);
}

test "bitwise - AND with U32" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    var allocator = arena.allocator();

    var r = Reader.init(&allocator);
    var checker = BidirectionalTypeChecker.init(allocator);
    defer checker.deinit();

    const expr = try r.readString("(bitwise-and (: U32 15) (: U32 7))");
    const typed = try checker.synthesizeTyped(expr);
    try std.testing.expect(typed.getType() == .u32);
}

// ============================================================================
// Bitwise OR Tests
// ============================================================================

test "bitwise - OR basic" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    var allocator = arena.allocator();

    var r = Reader.init(&allocator);
    var checker = BidirectionalTypeChecker.init(allocator);
    defer checker.deinit();

    const expr = try r.readString("(bitwise-or 8 4)");
    const typed = try checker.synthesizeTyped(expr);
    try std.testing.expect(typed.getType() == .int);
}

test "bitwise - OR with I32" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    var allocator = arena.allocator();

    var r = Reader.init(&allocator);
    var checker = BidirectionalTypeChecker.init(allocator);
    defer checker.deinit();

    const expr = try r.readString("(bitwise-or (: I32 16) (: I32 32))");
    const typed = try checker.synthesizeTyped(expr);
    try std.testing.expect(typed.getType() == .i32);
}

// ============================================================================
// Bitwise XOR Tests
// ============================================================================

test "bitwise - XOR basic" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    var allocator = arena.allocator();

    var r = Reader.init(&allocator);
    var checker = BidirectionalTypeChecker.init(allocator);
    defer checker.deinit();

    const expr = try r.readString("(bitwise-xor 15 7)");
    const typed = try checker.synthesizeTyped(expr);
    try std.testing.expect(typed.getType() == .int);
}

test "bitwise - XOR with U64" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    var allocator = arena.allocator();

    var r = Reader.init(&allocator);
    var checker = BidirectionalTypeChecker.init(allocator);
    defer checker.deinit();

    const expr = try r.readString("(bitwise-xor (: U64 255) (: U64 128))");
    const typed = try checker.synthesizeTyped(expr);
    try std.testing.expect(typed.getType() == .u64);
}

// ============================================================================
// Bitwise NOT Tests
// ============================================================================

test "bitwise - NOT basic" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    var allocator = arena.allocator();

    var r = Reader.init(&allocator);
    var checker = BidirectionalTypeChecker.init(allocator);
    defer checker.deinit();

    const expr = try r.readString("(bitwise-not 5)");
    const typed = try checker.synthesizeTyped(expr);
    try std.testing.expect(typed.getType() == .int);
}

test "bitwise - NOT with U8" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    var allocator = arena.allocator();

    var r = Reader.init(&allocator);
    var checker = BidirectionalTypeChecker.init(allocator);
    defer checker.deinit();

    const expr = try r.readString("(bitwise-not (: U8 255))");
    const typed = try checker.synthesizeTyped(expr);
    try std.testing.expect(typed.getType() == .u8);
}

test "bitwise - NOT nested" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    var allocator = arena.allocator();

    var r = Reader.init(&allocator);
    var checker = BidirectionalTypeChecker.init(allocator);
    defer checker.deinit();

    const expr = try r.readString("(bitwise-not (bitwise-not 42))");
    const typed = try checker.synthesizeTyped(expr);
    try std.testing.expect(typed.getType() == .int);
}

// ============================================================================
// Bitwise Shift Left Tests
// ============================================================================

test "bitwise - SHL basic" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    var allocator = arena.allocator();

    var r = Reader.init(&allocator);
    var checker = BidirectionalTypeChecker.init(allocator);
    defer checker.deinit();

    const expr = try r.readString("(bitwise-shl 1 3)");
    const typed = try checker.synthesizeTyped(expr);
    try std.testing.expect(typed.getType() == .int);
}

test "bitwise - SHL with U32" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    var allocator = arena.allocator();

    var r = Reader.init(&allocator);
    var checker = BidirectionalTypeChecker.init(allocator);
    defer checker.deinit();

    const expr = try r.readString("(bitwise-shl (: U32 5) 2)");
    const typed = try checker.synthesizeTyped(expr);
    try std.testing.expect(typed.getType() == .u32);
}

// ============================================================================
// Bitwise Shift Right Tests
// ============================================================================

test "bitwise - SHR basic" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    var allocator = arena.allocator();

    var r = Reader.init(&allocator);
    var checker = BidirectionalTypeChecker.init(allocator);
    defer checker.deinit();

    const expr = try r.readString("(bitwise-shr 16 2)");
    const typed = try checker.synthesizeTyped(expr);
    try std.testing.expect(typed.getType() == .int);
}

test "bitwise - SHR with I64" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    var allocator = arena.allocator();

    var r = Reader.init(&allocator);
    var checker = BidirectionalTypeChecker.init(allocator);
    defer checker.deinit();

    const expr = try r.readString("(bitwise-shr (: I64 128) 3)");
    const typed = try checker.synthesizeTyped(expr);
    try std.testing.expect(typed.getType() == .i64);
}

// ============================================================================
// Nested and Combined Tests
// ============================================================================

test "bitwise - nested AND and OR" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    var allocator = arena.allocator();

    var r = Reader.init(&allocator);
    var checker = BidirectionalTypeChecker.init(allocator);
    defer checker.deinit();

    const expr = try r.readString("(bitwise-or (bitwise-and 12 10) (bitwise-and 5 3))");
    const typed = try checker.synthesizeTyped(expr);
    try std.testing.expect(typed.getType() == .int);
}

test "bitwise - XOR with shift" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    var allocator = arena.allocator();

    var r = Reader.init(&allocator);
    var checker = BidirectionalTypeChecker.init(allocator);
    defer checker.deinit();

    const expr = try r.readString("(bitwise-xor (bitwise-shl 1 4) (bitwise-shr 32 2))");
    const typed = try checker.synthesizeTyped(expr);
    try std.testing.expect(typed.getType() == .int);
}

test "bitwise - complex expression" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    var allocator = arena.allocator();

    var r = Reader.init(&allocator);
    var checker = BidirectionalTypeChecker.init(allocator);
    defer checker.deinit();

    const expr = try r.readString("(bitwise-and (bitwise-or 8 4) (bitwise-not (bitwise-shl 1 2)))");
    const typed = try checker.synthesizeTyped(expr);
    try std.testing.expect(typed.getType() == .int);
}

// ============================================================================
// Negative Tests (Should Fail Type Checking)
// ============================================================================

test "bitwise - AND with float fails" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    var allocator = arena.allocator();

    var r = Reader.init(&allocator);
    var checker = BidirectionalTypeChecker.init(allocator);
    defer checker.deinit();

    const expr = try r.readString("(bitwise-and 12.5 10)");
    const result = checker.synthesizeTyped(expr);
    try std.testing.expectError(error.TypeMismatch, result);
}

test "bitwise - OR with string fails" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    var allocator = arena.allocator();

    var r = Reader.init(&allocator);
    var checker = BidirectionalTypeChecker.init(allocator);
    defer checker.deinit();

    const expr = try r.readString("(bitwise-or 8 \"hello\")");
    const result = checker.synthesizeTyped(expr);
    try std.testing.expectError(error.TypeMismatch, result);
}

test "bitwise - NOT with boolean fails" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    var allocator = arena.allocator();

    var r = Reader.init(&allocator);
    var checker = BidirectionalTypeChecker.init(allocator);
    defer checker.deinit();

    const expr = try r.readString("(bitwise-not true)");
    const result = checker.synthesizeTyped(expr);
    try std.testing.expectError(error.TypeMismatch, result);
}

test "bitwise - SHL with float fails" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    var allocator = arena.allocator();

    var r = Reader.init(&allocator);
    var checker = BidirectionalTypeChecker.init(allocator);
    defer checker.deinit();

    const expr = try r.readString("(bitwise-shl 1.5 2)");
    const result = checker.synthesizeTyped(expr);
    try std.testing.expectError(error.TypeMismatch, result);
}
