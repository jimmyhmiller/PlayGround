const std = @import("std");
const mlir_lisp = @import("mlir_lisp");
const reader = mlir_lisp.reader;
const parser = mlir_lisp.parser;
const builder = mlir_lisp.builder;
const mlir = mlir_lisp.mlir;
const c_api = mlir_lisp.c_api;
const c_structs = mlir_lisp.c_structs;
const collection_types = mlir_lisp.collection_types;

test "CVectorLayout - struct field offsets match MLIR expectations" {
    // Verify that the C struct layout matches what MLIR expects
    const CVectorLayout = c_structs.CVectorLayout;

    // Create a dummy layout
    var layout = CVectorLayout.empty(*reader.Value);
    layout.len = 42;
    layout.capacity = 100;
    layout.elem_size = 8;

    // Field 0 (data): offset 0
    // Field 1 (len): offset 8 (after ptr)
    // Field 2 (capacity): offset 16
    // Field 3 (elem_size): offset 24

    const len_offset = @offsetOf(CVectorLayout, "len");
    const capacity_offset = @offsetOf(CVectorLayout, "capacity");
    const elem_size_offset = @offsetOf(CVectorLayout, "elem_size");

    try std.testing.expectEqual(@as(usize, 8), len_offset);
    try std.testing.expectEqual(@as(usize, 16), capacity_offset);
    try std.testing.expectEqual(@as(usize, 24), elem_size_offset);
}

test "C API - vector to layout conversion" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const allocator = arena.allocator();

    // Create a PersistentVector with some Values
    var vec = mlir_lisp.vector.PersistentVector(*reader.Value).init(allocator, null);
    defer vec.deinit();

    // Create some test values
    const val1 = try allocator.create(reader.Value);
    defer allocator.destroy(val1);
    val1.* = reader.Value{ .type = .number, .data = .{ .atom = "1" } };

    const val2 = try allocator.create(reader.Value);
    defer allocator.destroy(val2);
    val2.* = reader.Value{ .type = .number, .data = .{ .atom = "2" } };

    var vec2 = try vec.push(val1);
    defer vec2.deinit();

    var vec3 = try vec2.push(val2);
    defer vec3.deinit();

    // Convert to CVectorLayout
    const layout_ptr = try c_structs.vectorToCLayoutAlloc(*reader.Value, allocator, vec3);
    defer c_structs.destroyCVectorLayout(allocator, layout_ptr);

    // Verify the layout
    try std.testing.expectEqual(@as(usize, 2), layout_ptr.len);
    try std.testing.expect(layout_ptr.data != null);
    try std.testing.expectEqual(@sizeOf(*reader.Value), layout_ptr.elem_size);
}

test "MLIR collection types - create vector layout type" {
    var ctx = try mlir.Context.create();
    defer ctx.destroy();

    // Need to register LLVM dialect for !llvm.struct types
    ctx.registerAllDialects();
    try ctx.getOrLoadDialect("llvm");

    const vec_type = try collection_types.createVectorLayoutType(&ctx);
    try std.testing.expect(!mlir.c.mlirTypeIsNull(vec_type));

    // Should be a struct type with 4 fields
    // We can't easily verify the exact structure without more MLIR API,
    // but we can verify it's not null and parses correctly
}

test "MLIR collection types - field enum indices" {
    // Verify field indices match GEP expectations
    try std.testing.expectEqual(@as(i32, 0), @intFromEnum(collection_types.VectorLayoutField.data));
    try std.testing.expectEqual(@as(i32, 1), @intFromEnum(collection_types.VectorLayoutField.len));
    try std.testing.expectEqual(@as(i32, 2), @intFromEnum(collection_types.VectorLayoutField.capacity));
    try std.testing.expectEqual(@as(i32, 3), @intFromEnum(collection_types.VectorLayoutField.elem_size));
}

test "Code generation - load vector length" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const allocator = arena.allocator();

    const code = try collection_types.generateLoadVectorLen(allocator, "%my_vec", "%len");
    defer allocator.free(code);

    // Should contain GEP with field index 1 (len field)
    try std.testing.expect(std.mem.indexOf(u8, code, "llvm.getelementptr") != null);
    try std.testing.expect(std.mem.indexOf(u8, code, "1 i32") != null);
    try std.testing.expect(std.mem.indexOf(u8, code, "llvm.load") != null);
    try std.testing.expect(std.mem.indexOf(u8, code, "%len") != null);
}

test "End-to-end - generate struct field access code" {
    const allocator = std.testing.allocator;

    // Generate code to load vector length
    const len_code = try collection_types.generateLoadVectorLen(allocator, "%vec_ptr", "%len");
    defer allocator.free(len_code);

    // Verify it contains the expected operations
    try std.testing.expect(std.mem.indexOf(u8, len_code, "llvm.getelementptr") != null);
    try std.testing.expect(std.mem.indexOf(u8, len_code, "1 i32") != null); // Field 1 = len
    try std.testing.expect(std.mem.indexOf(u8, len_code, "llvm.load") != null);
    try std.testing.expect(std.mem.indexOf(u8, len_code, "%len") != null);

    // Generate code to load data pointer
    const data_code = try collection_types.generateLoadVectorData(allocator, "%vec_ptr", "%data");
    defer allocator.free(data_code);

    try std.testing.expect(std.mem.indexOf(u8, data_code, "0 i32") != null); // Field 0 = data
}

test "Documentation example usage is valid" {
    // Just verify that the example_usage string is non-empty and contains expected keywords
    const example = collection_types.example_usage;
    try std.testing.expect(example.len > 0);
    try std.testing.expect(std.mem.indexOf(u8, example, "CVectorLayout") != null);
    try std.testing.expect(std.mem.indexOf(u8, example, "llvm.getelementptr") != null);
    try std.testing.expect(std.mem.indexOf(u8, example, "llvm.load") != null);
}
