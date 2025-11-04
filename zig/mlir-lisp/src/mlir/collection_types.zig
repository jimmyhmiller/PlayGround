const std = @import("std");
const mlir = @import("c.zig");
const c_structs = @import("../collections/c_structs.zig");

/// Helper functions for working with CVectorLayout and CMapLayout in MLIR

/// Create the MLIR struct type for CVectorLayout
/// Layout: !llvm.struct<(ptr, i64, i64, i64)>
/// Fields: [data_ptr, len, capacity, elem_size]
pub fn createVectorLayoutType(ctx: *mlir.Context) !mlir.MlirType {
    const type_str = "!llvm.struct<(ptr, i64, i64, i64)>";
    return mlir.Type.parse(ctx, type_str);
}

/// Create the MLIR struct type for CMapLayout
/// Layout: !llvm.struct<(ptr, i64, i64, i64)>
/// Fields: [entries_ptr, len, capacity, entry_size]
pub fn createMapLayoutType(ctx: *mlir.Context) !mlir.MlirType {
    const type_str = "!llvm.struct<(ptr, i64, i64, i64)>";
    return mlir.Type.parse(ctx, type_str);
}

/// Field indices for CVectorLayout struct
pub const VectorLayoutField = enum(i32) {
    data = 0,
    len = 1,
    capacity = 2,
    elem_size = 3,
};

/// Field indices for CMapLayout struct
pub const MapLayoutField = enum(i32) {
    entries = 0,
    len = 1,
    capacity = 2,
    entry_size = 3,
};

/// Helper to create a GEP operation for accessing CVectorLayout fields
/// This returns the MLIR code to emit, not the actual operation
pub fn buildVectorLayoutGEP(
    allocator: std.mem.Allocator,
    struct_ptr: []const u8,
    field: VectorLayoutField,
) ![]const u8 {
    const field_idx = @intFromEnum(field);
    return std.fmt.allocPrint(allocator,
        "(llvm.getelementptr [{0s} (constant (: {1d} i32)) : !llvm.ptr])",
        .{ struct_ptr, field_idx }
    );
}

/// Helper to create a GEP operation for accessing CMapLayout fields
pub fn buildMapLayoutGEP(
    allocator: std.mem.Allocator,
    struct_ptr: []const u8,
    field: MapLayoutField,
) ![]const u8 {
    const field_idx = @intFromEnum(field);
    return std.fmt.allocPrint(allocator,
        "(llvm.getelementptr [{0s} (constant (: {1d} i32)) : !llvm.ptr])",
        .{ struct_ptr, field_idx }
    );
}

/// Generate lisp code to load the length from a CVectorLayout pointer
pub fn generateLoadVectorLen(allocator: std.mem.Allocator, struct_ptr: []const u8, result_name: []const u8) ![]const u8 {
    return std.fmt.allocPrint(allocator,
        \\(op {0s}_gep (: !llvm.ptr) (llvm.getelementptr [{1s} (constant (: 1 i32)) : !llvm.ptr]))
        \\(op {0s} (: i64) (llvm.load [{0s}_gep]))
    , .{ result_name, struct_ptr });
}

/// Generate lisp code to load the data pointer from a CVectorLayout
pub fn generateLoadVectorData(allocator: std.mem.Allocator, struct_ptr: []const u8, result_name: []const u8) ![]const u8 {
    return std.fmt.allocPrint(allocator,
        \\(op {0s}_gep (: !llvm.ptr) (llvm.getelementptr [{1s} (constant (: 0 i32)) : !llvm.ptr]))
        \\(op {0s} (: !llvm.ptr) (llvm.load [{0s}_gep]))
    , .{ result_name, struct_ptr });
}

/// Generate lisp code to load the element size from a CVectorLayout
pub fn generateLoadVectorElemSize(allocator: std.mem.Allocator, struct_ptr: []const u8, result_name: []const u8) ![]const u8 {
    return std.fmt.allocPrint(allocator,
        \\(op {0s}_gep (: !llvm.ptr) (llvm.getelementptr [{1s} (constant (: 3 i32)) : !llvm.ptr]))
        \\(op {0s} (: i64) (llvm.load [{0s}_gep]))
    , .{ result_name, struct_ptr });
}

/// Generate lisp code to store a length value into a CVectorLayout
pub fn generateStoreVectorLen(allocator: std.mem.Allocator, struct_ptr: []const u8, len_value: []const u8) ![]const u8 {
    return std.fmt.allocPrint(allocator,
        \\(op %len_gep (: !llvm.ptr) (llvm.getelementptr [{0s} (constant (: 1 i32)) : !llvm.ptr]))
        \\(op (llvm.store [{1s} %len_gep]))
    , .{ struct_ptr, len_value });
}

/// Generate lisp code to access an element in the vector data array
/// Returns code that loads element at index from the data pointer
pub fn generateVectorElementAccess(
    allocator: std.mem.Allocator,
    struct_ptr: []const u8,
    index: []const u8,
    elem_type: []const u8,
    result_name: []const u8,
) ![]const u8 {
    return std.fmt.allocPrint(allocator,
        \\(op %data_gep (: !llvm.ptr) (llvm.getelementptr [{0s} (constant (: 0 i32)) : !llvm.ptr]))
        \\(op %data_ptr (: !llvm.ptr) (llvm.load [%data_gep]))
        \\(op %elem_gep (: !llvm.ptr) (llvm.getelementptr [%data_ptr {1s} : !llvm.ptr<{2s}>]))
        \\(op {3s} (: {2s}) (llvm.load [%elem_gep]))
    , .{ struct_ptr, index, elem_type, result_name });
}

// ============================================================================
// Documentation Examples
// ============================================================================

/// Example usage documentation
pub const example_usage =
    \\// Working with CVectorLayout in MLIR
    \\
    \\// 1. Create a pointer to CVectorLayout (typically passed as function argument)
    \\// %vec_ptr : !llvm.ptr   (pointer to CVectorLayout)
    \\
    \\// 2. Access the length field (offset 1 in struct)
    \\(op %len_gep (: !llvm.ptr) (llvm.getelementptr [%vec_ptr (constant (: 1 i32)) : !llvm.ptr]))
    \\(op %len (: i64) (llvm.load [%len_gep]))
    \\
    \\// 3. Access the data pointer (offset 0 in struct)
    \\(op %data_gep (: !llvm.ptr) (llvm.getelementptr [%vec_ptr (constant (: 0 i32)) : !llvm.ptr]))
    \\(op %data_ptr (: !llvm.ptr) (llvm.load [%data_gep]))
    \\
    \\// 4. Access element at index (e.g., index 2)
    \\(op %elem_gep (: !llvm.ptr) (llvm.getelementptr [%data_ptr (constant (: 2 i64)) : !llvm.ptr]))
    \\(op %elem (: !llvm.ptr) (llvm.load [%elem_gep]))  // Assuming ptr type elements
    \\
    \\// 5. Full example: sum vector lengths
    \\(defn get_vector_length [(: %vec_ptr !llvm.ptr)] i64
    \\  ;; GEP to len field (offset 1)
    \\  (op %len_gep (: !llvm.ptr) (llvm.getelementptr [%vec_ptr (constant (: 1 i32)) : !llvm.ptr]))
    \\  ;; Load the length
    \\  (op %len (: i64) (llvm.load [%len_gep]))
    \\  ;; Return it
    \\  (return %len))
;

// ============================================================================
// Tests
// ============================================================================

test "createVectorLayoutType" {
    var ctx = try mlir.Context.create();
    defer ctx.destroy();

    const vec_type = try createVectorLayoutType(&ctx);
    try std.testing.expect(!mlir.c.mlirTypeIsNull(vec_type));
}

test "createMapLayoutType" {
    var ctx = try mlir.Context.create();
    defer ctx.destroy();

    const map_type = try createMapLayoutType(&ctx);
    try std.testing.expect(!mlir.c.mlirTypeIsNull(map_type));
}

test "VectorLayoutField enum values" {
    try std.testing.expectEqual(@as(i32, 0), @intFromEnum(VectorLayoutField.data));
    try std.testing.expectEqual(@as(i32, 1), @intFromEnum(VectorLayoutField.len));
    try std.testing.expectEqual(@as(i32, 2), @intFromEnum(VectorLayoutField.capacity));
    try std.testing.expectEqual(@as(i32, 3), @intFromEnum(VectorLayoutField.elem_size));
}

test "generateLoadVectorLen" {
    const allocator = std.testing.allocator;

    const code = try generateLoadVectorLen(allocator, "%vec_ptr", "%len");
    defer allocator.free(code);

    // Should generate GEP + load for field 1
    try std.testing.expect(std.mem.indexOf(u8, code, "llvm.getelementptr") != null);
    try std.testing.expect(std.mem.indexOf(u8, code, "1 i32") != null);
    try std.testing.expect(std.mem.indexOf(u8, code, "llvm.load") != null);
}

test "generateLoadVectorData" {
    const allocator = std.testing.allocator;

    const code = try generateLoadVectorData(allocator, "%vec_ptr", "%data");
    defer allocator.free(code);

    // Should generate GEP + load for field 0
    try std.testing.expect(std.mem.indexOf(u8, code, "llvm.getelementptr") != null);
    try std.testing.expect(std.mem.indexOf(u8, code, "0 i32") != null);
    try std.testing.expect(std.mem.indexOf(u8, code, "llvm.load") != null);
}

test "generateVectorElementAccess" {
    const allocator = std.testing.allocator;

    const code = try generateVectorElementAccess(
        allocator,
        "%vec_ptr",
        "%idx",
        "!llvm.ptr",
        "%elem"
    );
    defer allocator.free(code);

    // Should generate: data GEP + load, then element GEP + load
    try std.testing.expect(std.mem.indexOf(u8, code, "llvm.getelementptr") != null);
    try std.testing.expect(std.mem.indexOf(u8, code, "llvm.load") != null);
}
