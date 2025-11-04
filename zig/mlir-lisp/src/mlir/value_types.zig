const std = @import("std");
const c_value_layout = @import("../reader/c_value_layout.zig");

/// MLIR type helpers for CValueLayout struct manipulation
///
/// This module provides utilities for working with CValueLayout structs in MLIR code.
/// It generates MLIR type declarations and provides helpers for field access via GEP.
///
/// CValueLayout struct layout in MLIR:
/// !llvm.struct<(i8, [7 x i8], ptr, i64, i64, i64, ptr, ptr)>
///
/// Fields:
/// 0. type_tag: i8 - ValueType enum tag
/// 1. _padding: [7 x i8] - Alignment padding
/// 2. data_ptr: ptr - Universal pointer field
/// 3. data_len: i64 - Universal length field
/// 4. data_capacity: i64 - Capacity field (collections)
/// 5. data_elem_size: i64 - Element size (collections)
/// 6. extra_ptr1: ptr - First extra pointer
/// 7. extra_ptr2: ptr - Second extra pointer

/// Field offsets in CValueLayout struct
/// Use these for type-safe GEP operations
pub const ValueLayoutField = enum(u32) {
    type_tag = 0,
    padding = 1,
    data_ptr = 2,
    data_len = 3,
    data_capacity = 4,
    data_elem_size = 5,
    extra_ptr1 = 6,
    extra_ptr2 = 7,

    pub fn offset(self: ValueLayoutField) u32 {
        return @intFromEnum(self);
    }
};

/// Create the MLIR type declaration for CValueLayout
/// Returns: !llvm.struct<(i8, [7 x i8], ptr, i64, i64, i64, ptr, ptr)>
pub fn createValueLayoutType() []const u8 {
    return "!llvm.struct<(i8, [7 x i8], ptr, i64, i64, i64, ptr, ptr)>";
}

/// Generate MLIR code to load the type_tag field from a CValueLayout pointer
/// Returns code like:
///   %c0 = llvm.mlir.constant(0 : i32) : i32
///   %type_tag_ptr = llvm.getelementptr %struct_ptr[%c0, 0] : (!llvm.ptr, i32) -> !llvm.ptr
///   %type_tag = llvm.load %type_tag_ptr : !llvm.ptr -> i8
pub fn generateLoadValueType(
    allocator: std.mem.Allocator,
    struct_ptr: []const u8,
    result_name: []const u8,
) ![]u8 {
    const field_idx = ValueLayoutField.type_tag.offset();
    return std.fmt.allocPrint(
        allocator,
        \\  %c{d} = llvm.mlir.constant({d} : i32) : i32
        \\  %{s}_ptr = llvm.getelementptr {s}[%c{d}, {d}] : (!llvm.ptr, i32) -> !llvm.ptr
        \\  %{s} = llvm.load %{s}_ptr : !llvm.ptr -> i8
        \\
    ,
        .{ field_idx, field_idx, result_name, struct_ptr, field_idx, field_idx, result_name, result_name },
    );
}

/// Generate MLIR code to load the data_ptr field from a CValueLayout pointer
/// Returns code like:
///   %c2 = llvm.mlir.constant(2 : i32) : i32
///   %data_ptr_field = llvm.getelementptr %struct_ptr[%c2, 2] : (!llvm.ptr, i32) -> !llvm.ptr
///   %data_ptr = llvm.load %data_ptr_field : !llvm.ptr -> !llvm.ptr
pub fn generateLoadDataPtr(
    allocator: std.mem.Allocator,
    struct_ptr: []const u8,
    result_name: []const u8,
) ![]u8 {
    const field_idx = ValueLayoutField.data_ptr.offset();
    return std.fmt.allocPrint(
        allocator,
        \\  %c{d} = llvm.mlir.constant({d} : i32) : i32
        \\  %{s}_field = llvm.getelementptr {s}[%c{d}, {d}] : (!llvm.ptr, i32) -> !llvm.ptr
        \\  %{s} = llvm.load %{s}_field : !llvm.ptr -> !llvm.ptr
        \\
    ,
        .{ field_idx, field_idx, result_name, struct_ptr, field_idx, field_idx, result_name, result_name },
    );
}

/// Generate MLIR code to load the data_len field from a CValueLayout pointer
/// Returns code like:
///   %c3 = llvm.mlir.constant(3 : i32) : i32
///   %data_len_field = llvm.getelementptr %struct_ptr[%c3, 3] : (!llvm.ptr, i32) -> !llvm.ptr
///   %data_len = llvm.load %data_len_field : !llvm.ptr -> i64
pub fn generateLoadDataLen(
    allocator: std.mem.Allocator,
    struct_ptr: []const u8,
    result_name: []const u8,
) ![]u8 {
    const field_idx = ValueLayoutField.data_len.offset();
    return std.fmt.allocPrint(
        allocator,
        \\  %c{d} = llvm.mlir.constant({d} : i32) : i32
        \\  %{s}_field = llvm.getelementptr {s}[%c{d}, {d}] : (!llvm.ptr, i32) -> !llvm.ptr
        \\  %{s} = llvm.load %{s}_field : !llvm.ptr -> i64
        \\
    ,
        .{ field_idx, field_idx, result_name, struct_ptr, field_idx, field_idx, result_name, result_name },
    );
}

/// Generate MLIR code to load the data_capacity field from a CValueLayout pointer
pub fn generateLoadDataCapacity(
    allocator: std.mem.Allocator,
    struct_ptr: []const u8,
    result_name: []const u8,
) ![]u8 {
    const field_idx = ValueLayoutField.data_capacity.offset();
    return std.fmt.allocPrint(
        allocator,
        \\  %c{d} = llvm.mlir.constant({d} : i32) : i32
        \\  %{s}_field = llvm.getelementptr {s}[%c{d}, {d}] : (!llvm.ptr, i32) -> !llvm.ptr
        \\  %{s} = llvm.load %{s}_field : !llvm.ptr -> i64
        \\
    ,
        .{ field_idx, field_idx, result_name, struct_ptr, field_idx, field_idx, result_name, result_name },
    );
}

/// Generate MLIR code to load the data_elem_size field from a CValueLayout pointer
pub fn generateLoadElemSize(
    allocator: std.mem.Allocator,
    struct_ptr: []const u8,
    result_name: []const u8,
) ![]u8 {
    const field_idx = ValueLayoutField.data_elem_size.offset();
    return std.fmt.allocPrint(
        allocator,
        \\  %c{d} = llvm.mlir.constant({d} : i32) : i32
        \\  %{s}_field = llvm.getelementptr {s}[%c{d}, {d}] : (!llvm.ptr, i32) -> !llvm.ptr
        \\  %{s} = llvm.load %{s}_field : !llvm.ptr -> i64
        \\
    ,
        .{ field_idx, field_idx, result_name, struct_ptr, field_idx, field_idx, result_name, result_name },
    );
}

/// Generate MLIR code to load the extra_ptr1 field from a CValueLayout pointer
pub fn generateLoadExtraPtr1(
    allocator: std.mem.Allocator,
    struct_ptr: []const u8,
    result_name: []const u8,
) ![]u8 {
    const field_idx = ValueLayoutField.extra_ptr1.offset();
    return std.fmt.allocPrint(
        allocator,
        \\  %c{d} = llvm.mlir.constant({d} : i32) : i32
        \\  %{s}_field = llvm.getelementptr {s}[%c{d}, {d}] : (!llvm.ptr, i32) -> !llvm.ptr
        \\  %{s} = llvm.load %{s}_field : !llvm.ptr -> !llvm.ptr
        \\
    ,
        .{ field_idx, field_idx, result_name, struct_ptr, field_idx, field_idx, result_name, result_name },
    );
}

/// Generate MLIR code to load the extra_ptr2 field from a CValueLayout pointer
pub fn generateLoadExtraPtr2(
    allocator: std.mem.Allocator,
    struct_ptr: []const u8,
    result_name: []const u8,
) ![]u8 {
    const field_idx = ValueLayoutField.extra_ptr2.offset();
    return std.fmt.allocPrint(
        allocator,
        \\  %c{d} = llvm.mlir.constant({d} : i32) : i32
        \\  %{s}_field = llvm.getelementptr {s}[%c{d}, {d}] : (!llvm.ptr, i32) -> !llvm.ptr
        \\  %{s} = llvm.load %{s}_field : !llvm.ptr -> !llvm.ptr
        \\
    ,
        .{ field_idx, field_idx, result_name, struct_ptr, field_idx, field_idx, result_name, result_name },
    );
}

/// Example usage documentation
pub const example_usage =
    \\// Example: Extract atom string data from a CValueLayout
    \\//
    \\// Given a CValueLayout pointer, extract the string data for atom types
    \\// (identifier, number, string, etc.)
    \\
    \\func.func @get_atom_string(%value_layout_ptr: !llvm.ptr) -> !llvm.ptr {
    \\  // Get the data_ptr field (field index 2)
    \\  %c2 = llvm.mlir.constant(2 : i32) : i32
    \\  %data_ptr_field = llvm.getelementptr %value_layout_ptr[%c2, 2] : (!llvm.ptr, i32) -> !llvm.ptr
    \\  %data_ptr = llvm.load %data_ptr_field : !llvm.ptr -> !llvm.ptr
    \\
    \\  // Get the data_len field (field index 3)
    \\  %c3 = llvm.mlir.constant(3 : i32) : i32
    \\  %data_len_field = llvm.getelementptr %value_layout_ptr[%c3, 3] : (!llvm.ptr, i32) -> !llvm.ptr
    \\  %data_len = llvm.load %data_len_field : !llvm.ptr -> i64
    \\
    \\  // Now you have both the pointer and length for the string
    \\  // You can use these for string operations, comparisons, etc.
    \\
    \\  func.return %data_ptr : !llvm.ptr
    \\}
    \\
    \\// Example: Check value type tag
    \\//
    \\// Extract and compare the type_tag field to determine what kind of value
    \\// this CValueLayout represents
    \\
    \\func.func @is_identifier(%value_layout_ptr: !llvm.ptr) -> i1 {
    \\  // Get the type_tag field (field index 0)
    \\  %c0 = llvm.mlir.constant(0 : i32) : i32
    \\  %type_tag_field = llvm.getelementptr %value_layout_ptr[%c0, 0] : (!llvm.ptr, i32) -> !llvm.ptr
    \\  %type_tag = llvm.load %type_tag_field : !llvm.ptr -> i8
    \\
    \\  // ValueType.identifier enum value (you'd need to know this value)
    \\  %identifier_tag = llvm.mlir.constant(0 : i8) : i8  // Example value
    \\
    \\  // Compare
    \\  %is_id = llvm.icmp "eq" %type_tag, %identifier_tag : i8
    \\
    \\  func.return %is_id : i1
    \\}
    \\
    \\// Example: Access collection data
    \\//
    \\// For list/vector/map types, extract the element array and length
    \\
    \\func.func @get_collection_info(%value_layout_ptr: !llvm.ptr) -> !llvm.ptr {
    \\  // Get data_ptr (element array)
    \\  %c2 = llvm.mlir.constant(2 : i32) : i32
    \\  %data_ptr_field = llvm.getelementptr %value_layout_ptr[%c2, 2] : (!llvm.ptr, i32) -> !llvm.ptr
    \\  %elements = llvm.load %data_ptr_field : !llvm.ptr -> !llvm.ptr
    \\
    \\  // Get data_len (number of elements)
    \\  %c3 = llvm.mlir.constant(3 : i32) : i32
    \\  %len_field = llvm.getelementptr %value_layout_ptr[%c3, 3] : (!llvm.ptr, i32) -> !llvm.ptr
    \\  %num_elements = llvm.load %len_field : !llvm.ptr -> i64
    \\
    \\  // Get data_elem_size (size of each element)
    \\  %c5 = llvm.mlir.constant(5 : i32) : i32
    \\  %elem_size_field = llvm.getelementptr %value_layout_ptr[%c5, 5] : (!llvm.ptr, i32) -> !llvm.ptr
    \\  %elem_size = llvm.load %elem_size_field : !llvm.ptr -> i64
    \\
    \\  // Now you can iterate over elements, etc.
    \\
    \\  func.return %elements : !llvm.ptr
    \\}
    \\
    \\// Example: Access complex type fields
    \\//
    \\// For types like has_type (typed literal), access the extra pointer fields
    \\
    \\func.func @get_typed_value(%value_layout_ptr: !llvm.ptr) -> !llvm.ptr {
    \\  // For has_type: extra_ptr1 points to the value
    \\  %c6 = llvm.mlir.constant(6 : i32) : i32
    \\  %value_field = llvm.getelementptr %value_layout_ptr[%c6, 6] : (!llvm.ptr, i32) -> !llvm.ptr
    \\  %value_ptr = llvm.load %value_field : !llvm.ptr -> !llvm.ptr
    \\
    \\  // extra_ptr2 points to the type_expr
    \\  %c7 = llvm.mlir.constant(7 : i32) : i32
    \\  %type_field = llvm.getelementptr %value_layout_ptr[%c7, 7] : (!llvm.ptr, i32) -> !llvm.ptr
    \\  %type_ptr = llvm.load %type_field : !llvm.ptr -> !llvm.ptr
    \\
    \\  func.return %value_ptr : !llvm.ptr
    \\}
;

// ============================================================================
// Tests
// ============================================================================

test "ValueLayoutField - offsets" {
    try std.testing.expectEqual(@as(u32, 0), ValueLayoutField.type_tag.offset());
    try std.testing.expectEqual(@as(u32, 1), ValueLayoutField.padding.offset());
    try std.testing.expectEqual(@as(u32, 2), ValueLayoutField.data_ptr.offset());
    try std.testing.expectEqual(@as(u32, 3), ValueLayoutField.data_len.offset());
    try std.testing.expectEqual(@as(u32, 4), ValueLayoutField.data_capacity.offset());
    try std.testing.expectEqual(@as(u32, 5), ValueLayoutField.data_elem_size.offset());
    try std.testing.expectEqual(@as(u32, 6), ValueLayoutField.extra_ptr1.offset());
    try std.testing.expectEqual(@as(u32, 7), ValueLayoutField.extra_ptr2.offset());
}

test "createValueLayoutType - returns correct MLIR type" {
    const ty = createValueLayoutType();
    try std.testing.expectEqualStrings("!llvm.struct<(i8, [7 x i8], ptr, i64, i64, i64, ptr, ptr)>", ty);
}

test "generateLoadValueType - generates correct MLIR" {
    const allocator = std.testing.allocator;
    const code = try generateLoadValueType(allocator, "%my_struct", "my_type");
    defer allocator.free(code);

    try std.testing.expect(std.mem.indexOf(u8, code, "%c0 = llvm.mlir.constant(0 : i32)") != null);
    try std.testing.expect(std.mem.indexOf(u8, code, "llvm.getelementptr %my_struct") != null);
    try std.testing.expect(std.mem.indexOf(u8, code, "%my_type = llvm.load") != null);
    try std.testing.expect(std.mem.indexOf(u8, code, "-> i8") != null);
}

test "generateLoadDataPtr - generates correct MLIR" {
    const allocator = std.testing.allocator;
    const code = try generateLoadDataPtr(allocator, "%my_struct", "my_data");
    defer allocator.free(code);

    try std.testing.expect(std.mem.indexOf(u8, code, "%c2 = llvm.mlir.constant(2 : i32)") != null);
    try std.testing.expect(std.mem.indexOf(u8, code, "llvm.getelementptr %my_struct") != null);
    try std.testing.expect(std.mem.indexOf(u8, code, "%my_data = llvm.load") != null);
    try std.testing.expect(std.mem.indexOf(u8, code, "-> !llvm.ptr") != null);
}

test "generateLoadDataLen - generates correct MLIR" {
    const allocator = std.testing.allocator;
    const code = try generateLoadDataLen(allocator, "%my_struct", "my_len");
    defer allocator.free(code);

    try std.testing.expect(std.mem.indexOf(u8, code, "%c3 = llvm.mlir.constant(3 : i32)") != null);
    try std.testing.expect(std.mem.indexOf(u8, code, "llvm.getelementptr %my_struct") != null);
    try std.testing.expect(std.mem.indexOf(u8, code, "%my_len = llvm.load") != null);
    try std.testing.expect(std.mem.indexOf(u8, code, "-> i64") != null);
}

test "generateLoadExtraPtr1 - generates correct MLIR" {
    const allocator = std.testing.allocator;
    const code = try generateLoadExtraPtr1(allocator, "%my_struct", "my_ptr1");
    defer allocator.free(code);

    try std.testing.expect(std.mem.indexOf(u8, code, "%c6 = llvm.mlir.constant(6 : i32)") != null);
    try std.testing.expect(std.mem.indexOf(u8, code, "llvm.getelementptr %my_struct") != null);
    try std.testing.expect(std.mem.indexOf(u8, code, "%my_ptr1 = llvm.load") != null);
    try std.testing.expect(std.mem.indexOf(u8, code, "-> !llvm.ptr") != null);
}

test "generateLoadExtraPtr2 - generates correct MLIR" {
    const allocator = std.testing.allocator;
    const code = try generateLoadExtraPtr2(allocator, "%my_struct", "my_ptr2");
    defer allocator.free(code);

    try std.testing.expect(std.mem.indexOf(u8, code, "%c7 = llvm.mlir.constant(7 : i32)") != null);
    try std.testing.expect(std.mem.indexOf(u8, code, "llvm.getelementptr %my_struct") != null);
    try std.testing.expect(std.mem.indexOf(u8, code, "%my_ptr2 = llvm.load") != null);
    try std.testing.expect(std.mem.indexOf(u8, code, "-> !llvm.ptr") != null);
}
