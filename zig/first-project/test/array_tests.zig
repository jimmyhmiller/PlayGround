const std = @import("std");
const Reader = @import("reader.zig").Reader;
const TypeChecker = @import("type_checker.zig");
const SimpleCCompiler = @import("simple_c_compiler.zig").SimpleCCompiler;

// ============================================================================
// ARRAY TESTS - Type Checking and Code Generation
// ============================================================================

// Type Checking Tests

test "array - basic array type parsing" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    var allocator = arena.allocator();

    var reader = Reader.init(&allocator);
    var checker = TypeChecker.BidirectionalTypeChecker.init(allocator);
    defer checker.deinit();

    const code = "(def arr (: (Array Int 10)) (array Int 10))";
    const read_result = try reader.readAllString(code);
    var expressions = read_result.values;
    defer expressions.deinit(allocator);

    var report = try checker.typeCheckAllTwoPass(expressions.items);
    defer report.typed.deinit(allocator);
    defer report.errors.deinit(allocator);

    try std.testing.expect(report.errors.items.len == 0);
    const arr_type = checker.env.get("arr").?;
    try std.testing.expect(arr_type == .array);
    try std.testing.expect(arr_type.array.size == 10);
    try std.testing.expect(arr_type.array.element_type == .int);
}

test "array - initialized array" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    var allocator = arena.allocator();

    var reader = Reader.init(&allocator);
    var checker = TypeChecker.BidirectionalTypeChecker.init(allocator);
    defer checker.deinit();

    const code = "(def arr (: (Array Float 5)) (array Float 5 0.0))";
    const read_result = try reader.readAllString(code);
    var expressions = read_result.values;
    defer expressions.deinit(allocator);

    var report = try checker.typeCheckAllTwoPass(expressions.items);
    defer report.typed.deinit(allocator);
    defer report.errors.deinit(allocator);

    try std.testing.expect(report.errors.items.len == 0);
    const arr_type = checker.env.get("arr").?;
    try std.testing.expect(arr_type == .array);
    try std.testing.expect(arr_type.array.size == 5);
    try std.testing.expect(arr_type.array.element_type == .float);
}

test "array - array-ref type checking" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    var allocator = arena.allocator();

    var reader = Reader.init(&allocator);
    var checker = TypeChecker.BidirectionalTypeChecker.init(allocator);
    defer checker.deinit();

    const code =
        \\(def arr (: (Array Int 10)) (array Int 10))
        \\(array-ref arr 5)
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

test "array - array-set! type checking" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    var allocator = arena.allocator();

    var reader = Reader.init(&allocator);
    var checker = TypeChecker.BidirectionalTypeChecker.init(allocator);
    defer checker.deinit();

    const code =
        \\(def arr (: (Array Int 10)) (array Int 10))
        \\(array-set! arr 5 42)
    ;
    const read_result = try reader.readAllString(code);
    var expressions = read_result.values;
    defer expressions.deinit(allocator);

    var report = try checker.typeCheckAllTwoPass(expressions.items);
    defer report.typed.deinit(allocator);
    defer report.errors.deinit(allocator);

    try std.testing.expect(report.errors.items.len == 0);
    const last_typed = report.typed.items[report.typed.items.len - 1];
    try std.testing.expect(last_typed.getType() == .nil);
}

test "array - array-length type checking" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    var allocator = arena.allocator();

    var reader = Reader.init(&allocator);
    var checker = TypeChecker.BidirectionalTypeChecker.init(allocator);
    defer checker.deinit();

    const code =
        \\(def arr (: (Array Int 10)) (array Int 10))
        \\(array-length arr)
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

test "array - multidimensional arrays" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    var allocator = arena.allocator();

    var reader = Reader.init(&allocator);
    var checker = TypeChecker.BidirectionalTypeChecker.init(allocator);
    defer checker.deinit();

    const code = "(def matrix (: (Array (Array Float 3) 2)) (array (Array Float 3) 2))";
    const read_result = try reader.readAllString(code);
    var expressions = read_result.values;
    defer expressions.deinit(allocator);

    var report = try checker.typeCheckAllTwoPass(expressions.items);
    defer report.typed.deinit(allocator);
    defer report.errors.deinit(allocator);

    try std.testing.expect(report.errors.items.len == 0);
    const matrix_type = checker.env.get("matrix").?;
    try std.testing.expect(matrix_type == .array);
    try std.testing.expect(matrix_type.array.size == 2);
    try std.testing.expect(matrix_type.array.element_type == .array);
    try std.testing.expect(matrix_type.array.element_type.array.size == 3);
    try std.testing.expect(matrix_type.array.element_type.array.element_type == .float);
}

test "array - nested array-ref for multidimensional arrays" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    var allocator = arena.allocator();

    var reader = Reader.init(&allocator);
    var checker = TypeChecker.BidirectionalTypeChecker.init(allocator);
    defer checker.deinit();

    const code =
        \\(def matrix (: (Array (Array Float 3) 2)) (array (Array Float 3) 2))
        \\(array-ref (array-ref matrix 1) 2)
    ;
    const read_result = try reader.readAllString(code);
    var expressions = read_result.values;
    defer expressions.deinit(allocator);

    var report = try checker.typeCheckAllTwoPass(expressions.items);
    defer report.typed.deinit(allocator);
    defer report.errors.deinit(allocator);

    try std.testing.expect(report.errors.items.len == 0);
    const last_typed = report.typed.items[report.typed.items.len - 1];
    try std.testing.expect(last_typed.getType() == .float);
}

test "array - nested array-set! for multidimensional arrays" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    var allocator = arena.allocator();

    var reader = Reader.init(&allocator);
    var checker = TypeChecker.BidirectionalTypeChecker.init(allocator);
    defer checker.deinit();

    const code =
        \\(def matrix (: (Array (Array Int 3) 2)) (array (Array Int 3) 2))
        \\(array-set! (array-ref matrix 0) 1 42)
    ;
    const read_result = try reader.readAllString(code);
    var expressions = read_result.values;
    defer expressions.deinit(allocator);

    var report = try checker.typeCheckAllTwoPass(expressions.items);
    defer report.typed.deinit(allocator);
    defer report.errors.deinit(allocator);

    try std.testing.expect(report.errors.items.len == 0);
    const last_typed = report.typed.items[report.typed.items.len - 1];
    try std.testing.expect(last_typed.getType() == .nil);
}

test "array - multidimensional array code generation" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    var allocator = arena.allocator();

    var compiler = SimpleCCompiler.init(&allocator);
    defer compiler.deinit();

    const source =
        \\(ns test)
        \\(def matrix (: (Array (Array Int 3) 2)) (array (Array Int 3) 2))
        \\(array-set! (array-ref matrix 0) 1 42)
        \\(array-ref (array-ref matrix 0) 1)
    ;

    const output = try compiler.compileString(source, .executable);
    defer allocator.free(output);

    // Check for nested array indexing
    try std.testing.expect(std.mem.indexOf(u8, output, "[0][1] = 42") != null);
    try std.testing.expect(std.mem.indexOf(u8, output, "[0][1]") != null);
}

test "array - arrays with specific sized types" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    var allocator = arena.allocator();

    var reader = Reader.init(&allocator);
    var checker = TypeChecker.BidirectionalTypeChecker.init(allocator);
    defer checker.deinit();

    const code = "(def arr (: (Array U32 10)) (array U32 10 0))";
    const read_result = try reader.readAllString(code);
    var expressions = read_result.values;
    defer expressions.deinit(allocator);

    var report = try checker.typeCheckAllTwoPass(expressions.items);
    defer report.typed.deinit(allocator);
    defer report.errors.deinit(allocator);

    try std.testing.expect(report.errors.items.len == 0);
    const arr_type = checker.env.get("arr").?;
    try std.testing.expect(arr_type == .array);
    try std.testing.expect(arr_type.array.element_type == .u32);
}

test "array - type error on wrong element type" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    var allocator = arena.allocator();

    var reader = Reader.init(&allocator);
    var checker = TypeChecker.BidirectionalTypeChecker.init(allocator);
    defer checker.deinit();

    const code =
        \\(def arr (: (Array Int 10)) (array Int 10))
        \\(array-set! arr 0 3.14)
    ;
    const read_result = try reader.readAllString(code);
    var expressions = read_result.values;
    defer expressions.deinit(allocator);

    var report = try checker.typeCheckAllTwoPass(expressions.items);
    defer report.typed.deinit(allocator);
    defer report.errors.deinit(allocator);

    // Should have type error - trying to assign Float to Int array
    try std.testing.expect(report.errors.items.len > 0);
}

// Code Generation Tests

test "array - basic array code generation" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    var allocator = arena.allocator();

    var compiler = SimpleCCompiler.init(&allocator);
    defer compiler.deinit();

    const source =
        \\(ns test)
        \\(def arr (: (Array Int 5)) (array Int 5))
    ;

    const output = try compiler.compileString(source, .executable);
    defer allocator.free(output);

    // Check that array declaration appears in output
    try std.testing.expect(std.mem.indexOf(u8, output, "long long") != null);
    try std.testing.expect(std.mem.indexOf(u8, output, "[5]") != null);
}

test "array - initialized array code generation" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    var allocator = arena.allocator();

    var compiler = SimpleCCompiler.init(&allocator);
    defer compiler.deinit();

    const source =
        \\(ns test)
        \\(def arr (: (Array Float 3)) (array Float 3 1.5))
    ;

    const output = try compiler.compileString(source, .executable);
    defer allocator.free(output);

    // Check that initialized array appears
    try std.testing.expect(std.mem.indexOf(u8, output, "double") != null);
    try std.testing.expect(std.mem.indexOf(u8, output, "[3]") != null);
}

test "array - array-ref code generation" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    var allocator = arena.allocator();

    var compiler = SimpleCCompiler.init(&allocator);
    defer compiler.deinit();

    const source =
        \\(ns test)
        \\(def arr (: (Array Int 5)) (array Int 5 0))
        \\(array-ref arr 2)
    ;

    const output = try compiler.compileString(source, .executable);
    defer allocator.free(output);

    // Check that array indexing appears
    try std.testing.expect(std.mem.indexOf(u8, output, "[2]") != null);
}

test "array - array-set! code generation" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    var allocator = arena.allocator();

    var compiler = SimpleCCompiler.init(&allocator);
    defer compiler.deinit();

    const source =
        \\(ns test)
        \\(def arr (: (Array Int 5)) (array Int 5))
        \\(array-set! arr 0 42)
    ;

    const output = try compiler.compileString(source, .executable);
    defer allocator.free(output);

    // Check that array assignment appears
    try std.testing.expect(std.mem.indexOf(u8, output, "[0] = 42") != null);
}

test "array - array-length code generation" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    var allocator = arena.allocator();

    var compiler = SimpleCCompiler.init(&allocator);
    defer compiler.deinit();

    const source =
        \\(ns test)
        \\(def arr (: (Array Int 10)) (array Int 10))
        \\(array-length arr)
    ;

    const output = try compiler.compileString(source, .executable);
    defer allocator.free(output);

    // Check that length becomes compile-time constant
    try std.testing.expect(std.mem.indexOf(u8, output, "10") != null);
}

test "array - full example with iteration" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    var allocator = arena.allocator();

    var compiler = SimpleCCompiler.init(&allocator);
    defer compiler.deinit();

    const source =
        \\(ns test)
        \\(def scores (: (Array Int 5)) (array Int 5 0))
        \\(array-set! scores 0 85)
        \\(array-set! scores 1 92)
        \\(array-ref scores 0)
    ;

    const output = try compiler.compileString(source, .executable);
    defer allocator.free(output);

    // Verify array operations are present
    try std.testing.expect(std.mem.indexOf(u8, output, "[5]") != null);
    try std.testing.expect(std.mem.indexOf(u8, output, "[0] = 85") != null);
    try std.testing.expect(std.mem.indexOf(u8, output, "[1] = 92") != null);
}

test "array - arrays in structs" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    var allocator = arena.allocator();

    var reader = Reader.init(&allocator);
    var checker = TypeChecker.BidirectionalTypeChecker.init(allocator);
    defer checker.deinit();

    const code =
        \\(def Buffer (: Type) (Struct [data (Array Int 10)] [size Int]))
        \\(def buf (: Buffer) (Buffer (array Int 10 0) 10))
    ;
    const read_result = try reader.readAllString(code);
    var expressions = read_result.values;
    defer expressions.deinit(allocator);

    var report = try checker.typeCheckAllTwoPass(expressions.items);
    defer report.typed.deinit(allocator);
    defer report.errors.deinit(allocator);

    try std.testing.expect(report.errors.items.len == 0);
    const buffer_type = checker.type_defs.get("Buffer").?;
    try std.testing.expect(buffer_type == .struct_type);
    const fields = buffer_type.struct_type.fields;
    try std.testing.expect(fields[0].field_type == .array);
    try std.testing.expect(fields[0].field_type.array.size == 10);
}

// Heap Array Tests (Phase 4)

test "array - allocate-array type checking" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    var allocator = arena.allocator();

    var reader = Reader.init(&allocator);
    var checker = TypeChecker.BidirectionalTypeChecker.init(allocator);
    defer checker.deinit();

    const code = "(allocate-array Float 100)";
    const read_result = try reader.readAllString(code);
    var expressions = read_result.values;
    defer expressions.deinit(allocator);

    var report = try checker.typeCheckAllTwoPass(expressions.items);
    defer report.typed.deinit(allocator);
    defer report.errors.deinit(allocator);

    try std.testing.expect(report.errors.items.len == 0);
    const last_typed = report.typed.items[report.typed.items.len - 1];
    try std.testing.expect(last_typed.getType() == .pointer);
    try std.testing.expect(last_typed.getType().pointer.* == .float);
}

test "array - allocate-array with initialization" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    var allocator = arena.allocator();

    var reader = Reader.init(&allocator);
    var checker = TypeChecker.BidirectionalTypeChecker.init(allocator);
    defer checker.deinit();

    const code = "(allocate-array Int 1000 0)";
    const read_result = try reader.readAllString(code);
    var expressions = read_result.values;
    defer expressions.deinit(allocator);

    var report = try checker.typeCheckAllTwoPass(expressions.items);
    defer report.typed.deinit(allocator);
    defer report.errors.deinit(allocator);

    try std.testing.expect(report.errors.items.len == 0);
    const last_typed = report.typed.items[report.typed.items.len - 1];
    try std.testing.expect(last_typed.getType() == .pointer);
    try std.testing.expect(last_typed.getType().pointer.* == .int);
}

test "array - pointer-index-read type checking" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    var allocator = arena.allocator();

    var reader = Reader.init(&allocator);
    var checker = TypeChecker.BidirectionalTypeChecker.init(allocator);
    defer checker.deinit();

    const code =
        \\(def buffer (: (Pointer Float)) (allocate-array Float 100))
        \\(pointer-index-read buffer 50)
    ;
    const read_result = try reader.readAllString(code);
    var expressions = read_result.values;
    defer expressions.deinit(allocator);

    var report = try checker.typeCheckAllTwoPass(expressions.items);
    defer report.typed.deinit(allocator);
    defer report.errors.deinit(allocator);

    try std.testing.expect(report.errors.items.len == 0);
    const last_typed = report.typed.items[report.typed.items.len - 1];
    try std.testing.expect(last_typed.getType() == .float);
}

test "array - pointer-index-write! type checking" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    var allocator = arena.allocator();

    var reader = Reader.init(&allocator);
    var checker = TypeChecker.BidirectionalTypeChecker.init(allocator);
    defer checker.deinit();

    const code =
        \\(def buffer (: (Pointer Int)) (allocate-array Int 100))
        \\(pointer-index-write! buffer 0 42)
    ;
    const read_result = try reader.readAllString(code);
    var expressions = read_result.values;
    defer expressions.deinit(allocator);

    var report = try checker.typeCheckAllTwoPass(expressions.items);
    defer report.typed.deinit(allocator);
    defer report.errors.deinit(allocator);

    try std.testing.expect(report.errors.items.len == 0);
    const last_typed = report.typed.items[report.typed.items.len - 1];
    try std.testing.expect(last_typed.getType() == .nil);
}

test "array - deallocate-array type checking" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    var allocator = arena.allocator();

    var reader = Reader.init(&allocator);
    var checker = TypeChecker.BidirectionalTypeChecker.init(allocator);
    defer checker.deinit();

    const code =
        \\(def buffer (: (Pointer Float)) (allocate-array Float 100))
        \\(deallocate-array buffer)
    ;
    const read_result = try reader.readAllString(code);
    var expressions = read_result.values;
    defer expressions.deinit(allocator);

    var report = try checker.typeCheckAllTwoPass(expressions.items);
    defer report.typed.deinit(allocator);
    defer report.errors.deinit(allocator);

    try std.testing.expect(report.errors.items.len == 0);
    const last_typed = report.typed.items[report.typed.items.len - 1];
    try std.testing.expect(last_typed.getType() == .nil);
}

test "array - array-ptr type checking" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    var allocator = arena.allocator();

    var reader = Reader.init(&allocator);
    var checker = TypeChecker.BidirectionalTypeChecker.init(allocator);
    defer checker.deinit();

    const code =
        \\(def arr (: (Array Int 10)) (array Int 10))
        \\(array-ptr arr 5)
    ;
    const read_result = try reader.readAllString(code);
    var expressions = read_result.values;
    defer expressions.deinit(allocator);

    var report = try checker.typeCheckAllTwoPass(expressions.items);
    defer report.typed.deinit(allocator);
    defer report.errors.deinit(allocator);

    try std.testing.expect(report.errors.items.len == 0);
    const last_typed = report.typed.items[report.typed.items.len - 1];
    try std.testing.expect(last_typed.getType() == .pointer);
    try std.testing.expect(last_typed.getType().pointer.* == .int);
}

test "array - heap array code generation" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    var allocator = arena.allocator();

    var compiler = SimpleCCompiler.init(&allocator);
    defer compiler.deinit();

    const source =
        \\(ns test)
        \\(def buffer (: (Pointer Float)) (allocate-array Float 100 0.0))
    ;

    const output = try compiler.compileString(source, .executable);
    defer allocator.free(output);

    // Check for malloc and initialization
    try std.testing.expect(std.mem.indexOf(u8, output, "malloc") != null);
    try std.testing.expect(std.mem.indexOf(u8, output, "sizeof(double)") != null or std.mem.indexOf(u8, output, "sizeof(float)") != null);
}

test "array - pointer indexing code generation" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    var allocator = arena.allocator();

    var compiler = SimpleCCompiler.init(&allocator);
    defer compiler.deinit();

    const source =
        \\(ns test)
        \\(def buffer (: (Pointer Int)) (allocate-array Int 10 0))
        \\(pointer-index-write! buffer 5 42)
        \\(pointer-index-read buffer 5)
    ;

    const output = try compiler.compileString(source, .executable);
    defer allocator.free(output);

    // Check for array indexing
    try std.testing.expect(std.mem.indexOf(u8, output, "[5] = 42") != null);
    try std.testing.expect(std.mem.indexOf(u8, output, "[5]") != null);
}

test "array - passing arrays to functions" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    var allocator = arena.allocator();

    var reader = Reader.init(&allocator);
    var checker = TypeChecker.BidirectionalTypeChecker.init(allocator);
    defer checker.deinit();

    const code =
        \\(def sum-array (: (-> [(Array Int 3)] Int))
        \\  (fn [arr]
        \\    (+ (array-ref arr 0) (+ (array-ref arr 1) (array-ref arr 2)))))
        \\(def nums (: (Array Int 3)) (array Int 3 0))
        \\(array-set! nums 0 10)
        \\(array-set! nums 1 20)
        \\(array-set! nums 2 30)
        \\(sum-array nums)
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

test "array - returning arrays from functions" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    var allocator = arena.allocator();

    var reader = Reader.init(&allocator);
    var checker = TypeChecker.BidirectionalTypeChecker.init(allocator);
    defer checker.deinit();

    const code =
        \\(def make-array (: (-> [] (Array Int 3)))
        \\  (fn []
        \\    (let [arr (: (Array Int 3)) (array Int 3 0)]
        \\      (array-set! arr 0 1)
        \\      (array-set! arr 1 2)
        \\      (array-set! arr 2 3)
        \\      arr)))
        \\(def result (: (Array Int 3)) (make-array))
    ;
    const read_result = try reader.readAllString(code);
    var expressions = read_result.values;
    defer expressions.deinit(allocator);

    var report = try checker.typeCheckAllTwoPass(expressions.items);
    defer report.typed.deinit(allocator);
    defer report.errors.deinit(allocator);

    try std.testing.expect(report.errors.items.len == 0);
    const result_type = checker.env.get("result").?;
    try std.testing.expect(result_type == .array);
    try std.testing.expect(result_type.array.size == 3);
    try std.testing.expect(result_type.array.element_type == .int);
}

test "array - array-ptr integration with pointer operations" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    var allocator = arena.allocator();

    var reader = Reader.init(&allocator);
    var checker = TypeChecker.BidirectionalTypeChecker.init(allocator);
    defer checker.deinit();

    const code =
        \\(def arr (: (Array Float 10)) (array Float 10 0.0))
        \\(def ptr (: (Pointer Float)) (array-ptr arr 5))
        \\(pointer-write! ptr 3.14)
        \\(array-ref arr 5)
    ;
    const read_result = try reader.readAllString(code);
    var expressions = read_result.values;
    defer expressions.deinit(allocator);

    var report = try checker.typeCheckAllTwoPass(expressions.items);
    defer report.typed.deinit(allocator);
    defer report.errors.deinit(allocator);

    try std.testing.expect(report.errors.items.len == 0);
    const last_typed = report.typed.items[report.typed.items.len - 1];
    try std.testing.expect(last_typed.getType() == .float);
}

test "array - 3D arrays (nested)" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    var allocator = arena.allocator();

    var reader = Reader.init(&allocator);
    var checker = TypeChecker.BidirectionalTypeChecker.init(allocator);
    defer checker.deinit();

    const code =
        \\(def tensor (: (Array (Array (Array Int 3) 4) 2))
        \\  (array (Array (Array Int 3) 4) 2))
        \\(array-ref (array-ref (array-ref tensor 1) 2) 1)
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
