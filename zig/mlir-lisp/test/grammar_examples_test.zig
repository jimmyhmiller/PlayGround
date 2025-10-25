const std = @import("std");
const mlir_lisp = @import("mlir_lisp");
const Tokenizer = mlir_lisp.Tokenizer;
const Reader = mlir_lisp.Reader;
const Value = mlir_lisp.Value;
const ValueType = mlir_lisp.ValueType;

// Example 1 — simple constant
const example1 =
    \\(mlir
    \\  (operation
    \\    (name arith.constant)
    \\    (result-bindings [%c0])
    \\    (result-types !i32)
    \\    (attributes { :value (#int 42) })
    \\    (location (#unknown))))
;

test "grammar example 1 - simple constant" {
    const allocator = std.testing.allocator;

    var tok = Tokenizer.init(allocator, example1);
    var reader = try Reader.init(allocator, &tok);
    var value = try reader.read();
    defer {
        value.deinit(allocator);
        allocator.destroy(value);
    }

    // Should be a list starting with 'mlir'
    try std.testing.expect(value.type == .list);
    const list = value.data.list;
    try std.testing.expect(list.len() == 2);

    // First element should be 'mlir'
    const first = list.at(0);
    try std.testing.expect(first.type == .identifier);
    try std.testing.expectEqualStrings("mlir", first.data.atom);

    // Second element should be the operation
    const operation = list.at(1);
    try std.testing.expect(operation.type == .list);
    const op_list = operation.data.list;
    try std.testing.expect(op_list.len() == 6); // operation + 5 sections

    // Check operation keyword
    try std.testing.expect(op_list.at(0).type == .identifier);
    try std.testing.expectEqualStrings("operation", op_list.at(0).data.atom);

    // Check name section
    const name_section = op_list.at(1);
    try std.testing.expect(name_section.type == .list);
    try std.testing.expect(name_section.data.list.len() == 2);
    try std.testing.expectEqualStrings("name", name_section.data.list.at(0).data.atom);
    try std.testing.expectEqualStrings("arith.constant", name_section.data.list.at(1).data.atom);

    // Check result-bindings section
    const bindings_section = op_list.at(2);
    try std.testing.expect(bindings_section.type == .list);
    try std.testing.expectEqualStrings("result-bindings", bindings_section.data.list.at(0).data.atom);
    const bindings_vec = bindings_section.data.list.at(1);
    try std.testing.expect(bindings_vec.type == .vector);
    try std.testing.expect(bindings_vec.data.vector.len() == 1);
    try std.testing.expectEqualStrings("%c0", bindings_vec.data.vector.at(0).data.atom);

    // Check result-types section
    const types_section = op_list.at(3);
    try std.testing.expect(types_section.type == .list);
    try std.testing.expectEqualStrings("result-types", types_section.data.list.at(0).data.atom);
    const type_expr = types_section.data.list.at(1);
    try std.testing.expect(type_expr.type == .type_expr);
    try std.testing.expectEqualStrings("i32", type_expr.data.type_expr.data.atom);

    // Check attributes section
    const attrs_section = op_list.at(4);
    try std.testing.expect(attrs_section.type == .list);
    try std.testing.expectEqualStrings("attributes", attrs_section.data.list.at(0).data.atom);
    const attrs_map = attrs_section.data.list.at(1);
    try std.testing.expect(attrs_map.type == .map);
    try std.testing.expect(attrs_map.data.map.len() == 2); // key-value pair
    try std.testing.expectEqualStrings(":value", attrs_map.data.map.at(0).data.atom);
    // The value is (#int 42) which is a list containing an attr_expr
    const attr_value = attrs_map.data.map.at(1);
    try std.testing.expect(attr_value.type == .list);
    try std.testing.expect(attr_value.data.list.at(0).type == .attr_expr);

    // Check location section
    const loc_section = op_list.at(5);
    try std.testing.expect(loc_section.type == .list);
    try std.testing.expectEqualStrings("location", loc_section.data.list.at(0).data.atom);
}

// Example 2 — add inside a function
const example2 =
    \\(mlir
    \\  (operation
    \\    (name func.func)
    \\    (attributes {
    \\      :sym  (#sym @add)
    \\      :type (!function (inputs !i32 !i32) (results !i32))
    \\      :visibility :public
    \\    })
    \\    (regions
    \\      (region
    \\        (block [^entry]
    \\          (arguments [ [%x !i32] [%y !i32] ])
    \\          (operation
    \\            (name arith.addi)
    \\            (result-bindings [%sum])
    \\            (operands %x %y)
    \\            (result-types !i32))
    \\          (operation
    \\            (name func.return)
    \\            (operands %sum)))))))
;

test "grammar example 2 - add inside a function" {
    const allocator = std.testing.allocator;

    var tok = Tokenizer.init(allocator, example2);
    var reader = try Reader.init(allocator, &tok);
    var value = try reader.read();
    defer {
        value.deinit(allocator);
        allocator.destroy(value);
    }

    // Should be a list starting with 'mlir'
    try std.testing.expect(value.type == .list);
    const list = value.data.list;
    try std.testing.expect(list.len() == 2);

    // Get the operation
    const operation = list.at(1);
    try std.testing.expect(operation.type == .list);
    const op_list = operation.data.list;

    // Find the attributes section
    var attrs_section: ?*Value = null;
    for (op_list.slice()) |section| {
        if (section.type == .list and section.data.list.len() > 0) {
            const first = section.data.list.at(0);
            if (first.type == .identifier and std.mem.eql(u8, first.data.atom, "attributes")) {
                attrs_section = section;
                break;
            }
        }
    }
    try std.testing.expect(attrs_section != null);

    // Check attributes map
    const attrs_map = attrs_section.?.data.list.at(1);
    try std.testing.expect(attrs_map.type == .map);
    // Should have 3 key-value pairs = 6 elements
    try std.testing.expect(attrs_map.data.map.len() == 6);

    // Find the regions section
    var regions_section: ?*Value = null;
    for (op_list.slice()) |section| {
        if (section.type == .list and section.data.list.len() > 0) {
            const first = section.data.list.at(0);
            if (first.type == .identifier and std.mem.eql(u8, first.data.atom, "regions")) {
                regions_section = section;
                break;
            }
        }
    }
    try std.testing.expect(regions_section != null);

    // Check region structure
    const region_list = regions_section.?.data.list.at(1);
    try std.testing.expect(region_list.type == .list);
    try std.testing.expectEqualStrings("region", region_list.data.list.at(0).data.atom);

    // Check block
    const block = region_list.data.list.at(1);
    try std.testing.expect(block.type == .list);
    try std.testing.expectEqualStrings("block", block.data.list.at(0).data.atom);

    // Check block label
    const label_vec = block.data.list.at(1);
    try std.testing.expect(label_vec.type == .vector);
    try std.testing.expect(label_vec.data.vector.len() == 1);
    try std.testing.expectEqualStrings("^entry", label_vec.data.vector.at(0).data.atom);

    // Check arguments section
    const args_section = block.data.list.at(2);
    try std.testing.expect(args_section.type == .list);
    try std.testing.expectEqualStrings("arguments", args_section.data.list.at(0).data.atom);
    const args_vec = args_section.data.list.at(1);
    try std.testing.expect(args_vec.type == .vector);
    try std.testing.expect(args_vec.data.vector.len() == 2); // Two argument pairs

    // Check first argument pair
    const arg1 = args_vec.data.vector.at(0);
    try std.testing.expect(arg1.type == .vector);
    try std.testing.expect(arg1.data.vector.len() == 2);
    try std.testing.expectEqualStrings("%x", arg1.data.vector.at(0).data.atom);
    try std.testing.expect(arg1.data.vector.at(1).type == .type_expr);
}

// Example 3 — function call with two constants
const example3 =
    \\(mlir
    \\  (operation
    \\    (name func.func)
    \\    (attributes {
    \\      :sym (#sym @main)
    \\      :type (!function (inputs) (results !i32))
    \\    })
    \\    (regions
    \\      (region
    \\        (block [^entry]
    \\          (arguments [])
    \\          (operation
    \\            (name arith.constant)
    \\            (result-bindings [%a])
    \\            (result-types !i32)
    \\            (attributes { :value (#int 1) }))
    \\          (operation
    \\            (name arith.constant)
    \\            (result-bindings [%b])
    \\            (result-types !i32)
    \\            (attributes { :value (#int 2) }))
    \\          (operation
    \\            (name func.call)
    \\            (result-bindings [%r])
    \\            (result-types !i32)
    \\            (operands %a %b)
    \\            (attributes { :callee (#flat-symbol @add) }))
    \\          (operation
    \\            (name func.return)
    \\            (operands %r)))))))
;

test "grammar example 3 - function call with constants" {
    const allocator = std.testing.allocator;

    var tok = Tokenizer.init(allocator, example3);
    var reader = try Reader.init(allocator, &tok);
    var value = try reader.read();
    defer {
        value.deinit(allocator);
        allocator.destroy(value);
    }

    // Should be a list starting with 'mlir'
    try std.testing.expect(value.type == .list);
    const list = value.data.list;
    try std.testing.expect(list.len() == 2);

    // Get the operation
    const operation = list.at(1);
    try std.testing.expect(operation.type == .list);

    // Just verify it parses correctly
    try std.testing.expect(operation.data.list.at(0).type == .identifier);
    try std.testing.expectEqualStrings("operation", operation.data.list.at(0).data.atom);
}

// Example 4 — control flow with multiple blocks
const example4 =
    \\(mlir
    \\  (operation
    \\    (name func.func)
    \\    (attributes {
    \\      :sym (#sym @branchy)
    \\      :type (!function (inputs !i1 !i32 !i32) (results !i32))
    \\    })
    \\    (regions
    \\      (region
    \\        (block [^entry]
    \\          (arguments [ [%cond !i1] [%x !i32] [%y !i32] ])
    \\          (operation
    \\            (name cf.cond_br)
    \\            (operands %cond)
    \\            (successors
    \\              (successor ^then (%x))
    \\              (successor ^else (%y)))))
    \\        (block [^then]
    \\          (arguments [ [%t !i32] ])
    \\          (operation (name func.return) (operands %t)))
    \\        (block [^else]
    \\          (arguments [ [%e !i32] ])
    \\          (operation (name func.return) (operands %e)))))))
;

test "grammar example 4 - control flow with multiple blocks" {
    const allocator = std.testing.allocator;

    var tok = Tokenizer.init(allocator, example4);
    var reader = try Reader.init(allocator, &tok);
    var value = try reader.read();
    defer {
        value.deinit(allocator);
        allocator.destroy(value);
    }

    // Should be a list starting with 'mlir'
    try std.testing.expect(value.type == .list);
    const list = value.data.list;
    try std.testing.expect(list.len() == 2);

    // Get the operation
    const operation = list.at(1);
    try std.testing.expect(operation.type == .list);

    // Find the regions section
    var regions_section: ?*Value = null;
    for (operation.data.list.slice()) |section| {
        if (section.type == .list and section.data.list.len() > 0) {
            const first = section.data.list.at(0);
            if (first.type == .identifier and std.mem.eql(u8, first.data.atom, "regions")) {
                regions_section = section;
                break;
            }
        }
    }
    try std.testing.expect(regions_section != null);

    // Check region
    const region_list = regions_section.?.data.list.at(1);
    try std.testing.expect(region_list.type == .list);
    try std.testing.expectEqualStrings("region", region_list.data.list.at(0).data.atom);

    // Should have 3 blocks (entry, then, else)
    var block_count: usize = 0;
    for (region_list.data.list.slice()) |item| {
        if (item.type == .list and item.data.list.len() > 0) {
            const first = item.data.list.at(0);
            if (first.type == .identifier and std.mem.eql(u8, first.data.atom, "block")) {
                block_count += 1;
            }
        }
    }
    try std.testing.expect(block_count == 3);

    // Find the first block and check successors
    const first_block = region_list.data.list.at(1);
    try std.testing.expect(first_block.type == .list);
    try std.testing.expectEqualStrings("block", first_block.data.list.at(0).data.atom);

    // Find the successors section in the first block's operation
    var found_successors = false;
    for (first_block.data.list.slice()) |item| {
        if (item.type == .list and item.data.list.len() > 0) {
            for (item.data.list.slice()) |subitem| {
                if (subitem.type == .list and subitem.data.list.len() > 0) {
                    const first = subitem.data.list.at(0);
                    if (first.type == .identifier and std.mem.eql(u8, first.data.atom, "successors")) {
                        found_successors = true;
                        // Check that it has successor entries
                        try std.testing.expect(subitem.data.list.len() >= 2);
                        const succ1 = subitem.data.list.at(1);
                        try std.testing.expect(succ1.type == .list);
                        try std.testing.expectEqualStrings("successor", succ1.data.list.at(0).data.atom);
                        try std.testing.expectEqualStrings("^then", succ1.data.list.at(1).data.atom);
                        break;
                    }
                }
            }
        }
    }
    try std.testing.expect(found_successors);
}

test "all grammar examples parse successfully" {
    const allocator = std.testing.allocator;

    const examples = [_][]const u8{
        example1,
        example2,
        example3,
        example4,
    };

    for (examples, 0..) |example, i| {
        var tok = Tokenizer.init(allocator, example);
        var reader = try Reader.init(allocator, &tok);
        var value = try reader.read();
        defer {
            value.deinit(allocator);
            allocator.destroy(value);
        }

        // All examples should parse to a list starting with 'mlir'
        try std.testing.expect(value.type == .list);
        const list = value.data.list;
        try std.testing.expect(list.len() == 2);
        const first = list.at(0);
        try std.testing.expect(first.type == .identifier);
        try std.testing.expectEqualStrings("mlir", first.data.atom);

        std.debug.print("Example {} parsed successfully\n", .{i + 1});
    }
}
