const std = @import("std");
const mlir_lisp = @import("mlir_lisp");
const reader = mlir_lisp.reader;
const c_value_layout = mlir_lisp.c_value_layout;
const value_types = mlir_lisp.value_types;
const vector = mlir_lisp.vector;

// ============================================================================
// CValueLayout struct tests
// ============================================================================

test "CValueLayout - size and alignment" {
    try std.testing.expectEqual(@as(usize, 56), @sizeOf(c_value_layout.CValueLayout));
    try std.testing.expectEqual(@as(usize, 8), @alignOf(c_value_layout.CValueLayout));
}

test "CValueLayout - field offsets" {
    try std.testing.expectEqual(@as(usize, 0), @offsetOf(c_value_layout.CValueLayout, "type_tag"));
    try std.testing.expectEqual(@as(usize, 1), @offsetOf(c_value_layout.CValueLayout, "_padding"));
    try std.testing.expectEqual(@as(usize, 8), @offsetOf(c_value_layout.CValueLayout, "data_ptr"));
    try std.testing.expectEqual(@as(usize, 16), @offsetOf(c_value_layout.CValueLayout, "data_len"));
    try std.testing.expectEqual(@as(usize, 24), @offsetOf(c_value_layout.CValueLayout, "data_capacity"));
    try std.testing.expectEqual(@as(usize, 32), @offsetOf(c_value_layout.CValueLayout, "data_elem_size"));
    try std.testing.expectEqual(@as(usize, 40), @offsetOf(c_value_layout.CValueLayout, "extra_ptr1"));
    try std.testing.expectEqual(@as(usize, 48), @offsetOf(c_value_layout.CValueLayout, "extra_ptr2"));
}

test "CValueLayout - empty layout creation" {
    const layout = c_value_layout.CValueLayout.empty(.identifier);
    try std.testing.expectEqual(@as(u8, @intFromEnum(reader.ValueType.identifier)), layout.type_tag);
    try std.testing.expectEqual(@as(?[*]u8, null), layout.data_ptr);
    try std.testing.expectEqual(@as(usize, 0), layout.data_len);
    try std.testing.expectEqual(@as(usize, 0), layout.data_capacity);
    try std.testing.expectEqual(@as(usize, 0), layout.data_elem_size);
    try std.testing.expectEqual(@as(?[*]u8, null), layout.extra_ptr1);
    try std.testing.expectEqual(@as(?[*]u8, null), layout.extra_ptr2);
}

test "CValueLayout - isAtom helper" {
    const id_layout = c_value_layout.CValueLayout.empty(.identifier);
    const num_layout = c_value_layout.CValueLayout.empty(.number);
    const str_layout = c_value_layout.CValueLayout.empty(.string);
    const list_layout = c_value_layout.CValueLayout.empty(.list);

    try std.testing.expect(id_layout.isAtom());
    try std.testing.expect(num_layout.isAtom());
    try std.testing.expect(str_layout.isAtom());
    try std.testing.expect(!list_layout.isAtom());
}

test "CValueLayout - isCollection helper" {
    const list_layout = c_value_layout.CValueLayout.empty(.list);
    const vec_layout = c_value_layout.CValueLayout.empty(.vector);
    const map_layout = c_value_layout.CValueLayout.empty(.map);
    const id_layout = c_value_layout.CValueLayout.empty(.identifier);

    try std.testing.expect(list_layout.isCollection());
    try std.testing.expect(vec_layout.isCollection());
    try std.testing.expect(map_layout.isCollection());
    try std.testing.expect(!id_layout.isCollection());
}

// ============================================================================
// Conversion tests
// ============================================================================

test "valueToCLayout - identifier" {
    const allocator = std.testing.allocator;

    var value = reader.Value{
        .type = .identifier,
        .data = .{ .atom = "test_name" },
    };

    const layout = try c_value_layout.valueToCLayout(allocator, &value);

    try std.testing.expectEqual(@as(u8, @intFromEnum(reader.ValueType.identifier)), layout.type_tag);
    try std.testing.expect(layout.data_ptr != null);
    try std.testing.expectEqual(@as(usize, 9), layout.data_len);
    try std.testing.expect(layout.isAtom());
    try std.testing.expect(!layout.isCollection());

    // Verify the string content
    const str_slice = layout.data_ptr.?[0..layout.data_len];
    try std.testing.expectEqualStrings("test_name", str_slice);
}

test "valueToCLayout - number" {
    const allocator = std.testing.allocator;

    var value = reader.Value{
        .type = .number,
        .data = .{ .atom = "42" },
    };

    const layout = try c_value_layout.valueToCLayout(allocator, &value);

    try std.testing.expectEqual(@as(u8, @intFromEnum(reader.ValueType.number)), layout.type_tag);
    try std.testing.expect(layout.data_ptr != null);
    try std.testing.expectEqual(@as(usize, 2), layout.data_len);

    const str_slice = layout.data_ptr.?[0..layout.data_len];
    try std.testing.expectEqualStrings("42", str_slice);
}

test "valueToCLayout - string" {
    const allocator = std.testing.allocator;

    var value = reader.Value{
        .type = .string,
        .data = .{ .atom = "hello world" },
    };

    const layout = try c_value_layout.valueToCLayout(allocator, &value);

    try std.testing.expectEqual(@as(u8, @intFromEnum(reader.ValueType.string)), layout.type_tag);
    try std.testing.expect(layout.data_ptr != null);
    try std.testing.expectEqual(@as(usize, 11), layout.data_len);

    const str_slice = layout.data_ptr.?[0..layout.data_len];
    try std.testing.expectEqualStrings("hello world", str_slice);
}

test "valueToCLayout - boolean literals" {
    const allocator = std.testing.allocator;

    var true_val = reader.Value{
        .type = .true_lit,
        .data = .{ .atom = "" },
    };

    var false_val = reader.Value{
        .type = .false_lit,
        .data = .{ .atom = "" },
    };

    const true_layout = try c_value_layout.valueToCLayout(allocator, &true_val);
    const false_layout = try c_value_layout.valueToCLayout(allocator, &false_val);

    try std.testing.expectEqual(@as(u8, @intFromEnum(reader.ValueType.true_lit)), true_layout.type_tag);
    try std.testing.expectEqual(@as(u8, @intFromEnum(reader.ValueType.false_lit)), false_layout.type_tag);

    // Booleans don't use the data fields
    try std.testing.expectEqual(@as(?[*]u8, null), true_layout.data_ptr);
    try std.testing.expectEqual(@as(?[*]u8, null), false_layout.data_ptr);
}

test "valueToCLayout - list" {
    const allocator = std.testing.allocator;

    // Create a simple list with one element
    const elem = try allocator.create(reader.Value);
    defer allocator.destroy(elem);
    elem.* = reader.Value{
        .type = .number,
        .data = .{ .atom = "42" },
    };

    var vec = vector.PersistentVector(*reader.Value).init(allocator, null);
    defer vec.deinit();

    var vec2 = try vec.push(elem);
    defer vec2.deinit();

    var value = reader.Value{
        .type = .list,
        .data = .{ .list = vec2 },
    };

    const layout = try c_value_layout.valueToCLayout(allocator, &value);

    try std.testing.expectEqual(@as(u8, @intFromEnum(reader.ValueType.list)), layout.type_tag);
    try std.testing.expect(layout.data_ptr != null);
    try std.testing.expectEqual(@as(usize, 1), layout.data_len);
    try std.testing.expectEqual(@as(usize, 1), layout.data_capacity);
    try std.testing.expectEqual(@as(usize, @sizeOf(*reader.Value)), layout.data_elem_size);
    try std.testing.expect(layout.isCollection());
}

test "valueToCLayout - vector" {
    const allocator = std.testing.allocator;

    // Create a vector with two elements
    const elem1 = try allocator.create(reader.Value);
    defer allocator.destroy(elem1);
    elem1.* = reader.Value{
        .type = .number,
        .data = .{ .atom = "1" },
    };

    const elem2 = try allocator.create(reader.Value);
    defer allocator.destroy(elem2);
    elem2.* = reader.Value{
        .type = .number,
        .data = .{ .atom = "2" },
    };

    var vec = vector.PersistentVector(*reader.Value).init(allocator, null);
    defer vec.deinit();

    var vec2 = try vec.push(elem1);
    defer vec2.deinit();

    var vec3 = try vec2.push(elem2);
    defer vec3.deinit();

    var value = reader.Value{
        .type = .vector,
        .data = .{ .vector = vec3 },
    };

    const layout = try c_value_layout.valueToCLayout(allocator, &value);

    try std.testing.expectEqual(@as(u8, @intFromEnum(reader.ValueType.vector)), layout.type_tag);
    try std.testing.expect(layout.data_ptr != null);
    try std.testing.expectEqual(@as(usize, 2), layout.data_len);
    try std.testing.expectEqual(@as(usize, @sizeOf(*reader.Value)), layout.data_elem_size);
}

test "cLayoutToValue - identifier round-trip" {
    const allocator = std.testing.allocator;

    var orig_value = reader.Value{
        .type = .identifier,
        .data = .{ .atom = "test_id" },
    };

    const layout = try c_value_layout.valueToCLayout(allocator, &orig_value);
    const new_value = try c_value_layout.cLayoutToValue(allocator, layout);
    defer allocator.destroy(new_value);

    try std.testing.expectEqual(reader.ValueType.identifier, new_value.type);
    try std.testing.expectEqualStrings("test_id", new_value.data.atom);
}

test "cLayoutToValue - number round-trip" {
    const allocator = std.testing.allocator;

    var orig_value = reader.Value{
        .type = .number,
        .data = .{ .atom = "3.14" },
    };

    const layout = try c_value_layout.valueToCLayout(allocator, &orig_value);
    const new_value = try c_value_layout.cLayoutToValue(allocator, layout);
    defer allocator.destroy(new_value);

    try std.testing.expectEqual(reader.ValueType.number, new_value.type);
    try std.testing.expectEqualStrings("3.14", new_value.data.atom);
}

test "cLayoutToValue - boolean round-trip" {
    const allocator = std.testing.allocator;

    var orig_value = reader.Value{
        .type = .true_lit,
        .data = .{ .atom = "" },
    };

    const layout = try c_value_layout.valueToCLayout(allocator, &orig_value);
    const new_value = try c_value_layout.cLayoutToValue(allocator, layout);
    defer allocator.destroy(new_value);

    try std.testing.expectEqual(reader.ValueType.true_lit, new_value.type);
}

test "cLayoutToValue - list round-trip" {
    const allocator = std.testing.allocator;

    // Create a simple list with one element
    const elem = try allocator.create(reader.Value);
    defer allocator.destroy(elem);
    elem.* = reader.Value{
        .type = .number,
        .data = .{ .atom = "99" },
    };

    var vec = vector.PersistentVector(*reader.Value).init(allocator, null);
    defer vec.deinit();

    var vec2 = try vec.push(elem);
    defer vec2.deinit();

    var orig_value = reader.Value{
        .type = .list,
        .data = .{ .list = vec2 },
    };

    const layout = try c_value_layout.valueToCLayout(allocator, &orig_value);
    const new_value = try c_value_layout.cLayoutToValue(allocator, layout);
    defer {
        if (new_value.data.list.buf) |buf| {
            allocator.free(buf);
        }
        allocator.destroy(new_value);
    }

    try std.testing.expectEqual(reader.ValueType.list, new_value.type);
    try std.testing.expect(new_value.data.list.buf != null);
    try std.testing.expectEqual(@as(usize, 1), new_value.data.list.buf.?.len);
}

// ============================================================================
// MLIR value type helper tests
// ============================================================================

test "ValueLayoutField - offsets" {
    try std.testing.expectEqual(@as(u32, 0), value_types.ValueLayoutField.type_tag.offset());
    try std.testing.expectEqual(@as(u32, 1), value_types.ValueLayoutField.padding.offset());
    try std.testing.expectEqual(@as(u32, 2), value_types.ValueLayoutField.data_ptr.offset());
    try std.testing.expectEqual(@as(u32, 3), value_types.ValueLayoutField.data_len.offset());
    try std.testing.expectEqual(@as(u32, 4), value_types.ValueLayoutField.data_capacity.offset());
    try std.testing.expectEqual(@as(u32, 5), value_types.ValueLayoutField.data_elem_size.offset());
    try std.testing.expectEqual(@as(u32, 6), value_types.ValueLayoutField.extra_ptr1.offset());
    try std.testing.expectEqual(@as(u32, 7), value_types.ValueLayoutField.extra_ptr2.offset());
}

test "createValueLayoutType - returns correct MLIR type" {
    const ty = value_types.createValueLayoutType();
    try std.testing.expectEqualStrings("!llvm.struct<(i8, [7 x i8], ptr, i64, i64, i64, ptr, ptr)>", ty);
}

test "generateLoadValueType - generates correct MLIR" {
    const allocator = std.testing.allocator;
    const code = try value_types.generateLoadValueType(allocator, "%my_value", "my_type");
    defer allocator.free(code);

    try std.testing.expect(std.mem.indexOf(u8, code, "%c0 = llvm.mlir.constant(0 : i32)") != null);
    try std.testing.expect(std.mem.indexOf(u8, code, "llvm.getelementptr %my_value") != null);
    try std.testing.expect(std.mem.indexOf(u8, code, "%my_type = llvm.load") != null);
    try std.testing.expect(std.mem.indexOf(u8, code, "-> i8") != null);
}

test "generateLoadDataPtr - generates correct MLIR" {
    const allocator = std.testing.allocator;
    const code = try value_types.generateLoadDataPtr(allocator, "%my_value", "my_data");
    defer allocator.free(code);

    try std.testing.expect(std.mem.indexOf(u8, code, "%c2 = llvm.mlir.constant(2 : i32)") != null);
    try std.testing.expect(std.mem.indexOf(u8, code, "llvm.getelementptr %my_value") != null);
    try std.testing.expect(std.mem.indexOf(u8, code, "%my_data = llvm.load") != null);
    try std.testing.expect(std.mem.indexOf(u8, code, "-> !llvm.ptr") != null);
}

test "generateLoadDataLen - generates correct MLIR" {
    const allocator = std.testing.allocator;
    const code = try value_types.generateLoadDataLen(allocator, "%my_value", "my_len");
    defer allocator.free(code);

    try std.testing.expect(std.mem.indexOf(u8, code, "%c3 = llvm.mlir.constant(3 : i32)") != null);
    try std.testing.expect(std.mem.indexOf(u8, code, "llvm.getelementptr %my_value") != null);
    try std.testing.expect(std.mem.indexOf(u8, code, "%my_len = llvm.load") != null);
    try std.testing.expect(std.mem.indexOf(u8, code, "-> i64") != null);
}

test "generateLoadDataCapacity - generates correct MLIR" {
    const allocator = std.testing.allocator;
    const code = try value_types.generateLoadDataCapacity(allocator, "%my_value", "my_cap");
    defer allocator.free(code);

    try std.testing.expect(std.mem.indexOf(u8, code, "%c4 = llvm.mlir.constant(4 : i32)") != null);
    try std.testing.expect(std.mem.indexOf(u8, code, "%my_cap = llvm.load") != null);
    try std.testing.expect(std.mem.indexOf(u8, code, "-> i64") != null);
}

test "generateLoadElemSize - generates correct MLIR" {
    const allocator = std.testing.allocator;
    const code = try value_types.generateLoadElemSize(allocator, "%my_value", "my_size");
    defer allocator.free(code);

    try std.testing.expect(std.mem.indexOf(u8, code, "%c5 = llvm.mlir.constant(5 : i32)") != null);
    try std.testing.expect(std.mem.indexOf(u8, code, "%my_size = llvm.load") != null);
    try std.testing.expect(std.mem.indexOf(u8, code, "-> i64") != null);
}

test "generateLoadExtraPtr1 - generates correct MLIR" {
    const allocator = std.testing.allocator;
    const code = try value_types.generateLoadExtraPtr1(allocator, "%my_value", "my_ptr1");
    defer allocator.free(code);

    try std.testing.expect(std.mem.indexOf(u8, code, "%c6 = llvm.mlir.constant(6 : i32)") != null);
    try std.testing.expect(std.mem.indexOf(u8, code, "%my_ptr1 = llvm.load") != null);
    try std.testing.expect(std.mem.indexOf(u8, code, "-> !llvm.ptr") != null);
}

test "generateLoadExtraPtr2 - generates correct MLIR" {
    const allocator = std.testing.allocator;
    const code = try value_types.generateLoadExtraPtr2(allocator, "%my_value", "my_ptr2");
    defer allocator.free(code);

    try std.testing.expect(std.mem.indexOf(u8, code, "%c7 = llvm.mlir.constant(7 : i32)") != null);
    try std.testing.expect(std.mem.indexOf(u8, code, "%my_ptr2 = llvm.load") != null);
    try std.testing.expect(std.mem.indexOf(u8, code, "-> !llvm.ptr") != null);
}
