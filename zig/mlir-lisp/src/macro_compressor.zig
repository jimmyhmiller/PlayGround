/// Macro Compressor - Converts verbose MLIR operation structures to compact macro forms
/// This is the inverse of macro expansion, transforming:
///   (operation (name arith.addi) ...) → (+ (: type) a b)
///   (operation (name arith.constant) ...) → (constant %x (: value type))
///   etc.

const std = @import("std");
const reader = @import("reader.zig");
const Value = reader.Value;
const ValueType = reader.ValueType;
const vector = @import("collections/vector.zig");
const PersistentVector = vector.PersistentVector;

// ============================================================================
// Helper Functions
// ============================================================================

/// Create an identifier Value
fn createIdentifier(allocator: std.mem.Allocator, name: []const u8) !*Value {
    const value = try allocator.create(Value);
    value.* = Value{
        .type = .identifier,
        .data = .{ .atom = name },
    };
    return value;
}

/// Create a list Value from a vector
fn createList(allocator: std.mem.Allocator, vec: PersistentVector(*Value)) !*Value {
    const value = try allocator.create(Value);
    value.* = Value{
        .type = .list,
        .data = .{ .list = vec },
    };
    return value;
}

/// Create a vector Value
fn createVector(allocator: std.mem.Allocator, vec: PersistentVector(*Value)) !*Value {
    const value = try allocator.create(Value);
    value.* = Value{
        .type = .vector,
        .data = .{ .vector = vec },
    };
    return value;
}

/// Create a has_type Value (: value type)
fn createHasType(allocator: std.mem.Allocator, val: *Value, type_expr: *Value) !*Value {
    const value = try allocator.create(Value);
    value.* = Value{
        .type = .has_type,
        .data = .{ .has_type = .{
            .value = val,
            .type_expr = type_expr,
        } },
    };
    return value;
}

/// Check if a value is a list starting with a specific identifier
fn isListStartingWith(value: *const Value, identifier: []const u8) bool {
    if (value.type != .list) return false;
    const list = value.data.list;
    if (list.len() < 1) return false;
    const first = list.at(0);
    return first.type == .identifier and std.mem.eql(u8, first.data.atom, identifier);
}

/// Get the value from a list element (name value)
fn getNamedValue(list: PersistentVector(*Value), name: []const u8) ?*Value {
    for (list.slice()) |elem| {
        if (elem.type != .list) continue;
        const inner_list = elem.data.list;
        if (inner_list.len() < 2) continue;
        const first = inner_list.at(0);
        if (first.type != .identifier) continue;
        if (std.mem.eql(u8, first.data.atom, name)) {
            return inner_list.at(1);
        }
    }
    return null;
}

/// Get all elements from a list element (name elem1 elem2 ...)
fn getNamedElements(list: PersistentVector(*Value), name: []const u8, allocator: std.mem.Allocator) !?std.ArrayList(*Value) {
    for (list.slice()) |elem| {
        if (elem.type != .list) continue;
        const inner_list = elem.data.list;
        if (inner_list.len() < 1) continue;
        const first = inner_list.at(0);
        if (first.type != .identifier) continue;
        if (std.mem.eql(u8, first.data.atom, name)) {
            var result = std.ArrayList(*Value){};
            var i: usize = 1;
            while (i < inner_list.len()) : (i += 1) {
                try result.append(allocator, inner_list.at(i));
            }
            return result;
        }
    }
    return null;
}

// ============================================================================
// Pattern Matchers
// ============================================================================

/// Check if operation is arith.addi
fn isArithAdd(op_list: PersistentVector(*Value)) bool {
    if (getNamedValue(op_list, "name")) |name_val| {
        return name_val.type == .identifier and std.mem.eql(u8, name_val.data.atom, "arith.addi");
    }
    return false;
}

/// Check if operation is arith.muli
fn isArithMul(op_list: PersistentVector(*Value)) bool {
    if (getNamedValue(op_list, "name")) |name_val| {
        return name_val.type == .identifier and std.mem.eql(u8, name_val.data.atom, "arith.muli");
    }
    return false;
}

/// Check if operation is arith.constant
fn isArithConstant(op_list: PersistentVector(*Value)) bool {
    if (getNamedValue(op_list, "name")) |name_val| {
        return name_val.type == .identifier and std.mem.eql(u8, name_val.data.atom, "arith.constant");
    }
    return false;
}

/// Check if operation is func.return
fn isFuncReturn(op_list: PersistentVector(*Value)) bool {
    if (getNamedValue(op_list, "name")) |name_val| {
        return name_val.type == .identifier and std.mem.eql(u8, name_val.data.atom, "func.return");
    }
    return false;
}

/// Check if operation is func.func
fn isFuncFunc(op_list: PersistentVector(*Value)) bool {
    if (getNamedValue(op_list, "name")) |name_val| {
        return name_val.type == .identifier and std.mem.eql(u8, name_val.data.atom, "func.func");
    }
    return false;
}

/// Check if operation is func.call
fn isFuncCall(op_list: PersistentVector(*Value)) bool {
    if (getNamedValue(op_list, "name")) |name_val| {
        return name_val.type == .identifier and std.mem.eql(u8, name_val.data.atom, "func.call");
    }
    return false;
}

// ============================================================================
// Compression Functions
// ============================================================================

/// Compress arith.addi to + macro
/// (operation (name arith.addi) (result-types type) (operands a b)) → (+ (: type) a b)
fn compressArithAdd(allocator: std.mem.Allocator, op_list: PersistentVector(*Value)) !*Value {
    // Get result type
    const result_type = getNamedValue(op_list, "result-types") orelse return error.MissingResultTypes;

    // Get operands
    var operands_list = try getNamedElements(op_list, "operands", allocator) orelse return error.MissingOperands;
    defer operands_list.deinit(allocator);

    if (operands_list.items.len != 2) return error.InvalidOperandCount;

    // Build (: type)
    var type_vec = PersistentVector(*Value).init(allocator, null);
    type_vec = try type_vec.push(try createIdentifier(allocator, ":"));
    type_vec = try type_vec.push(result_type);
    const type_expr = try createList(allocator, type_vec);

    // Build (+ (: type) a b)
    var result_vec = PersistentVector(*Value).init(allocator, null);
    result_vec = try result_vec.push(try createIdentifier(allocator, "+"));
    result_vec = try result_vec.push(type_expr);
    result_vec = try result_vec.push(operands_list.items[0]);
    result_vec = try result_vec.push(operands_list.items[1]);

    return try createList(allocator, result_vec);
}

/// Compress arith.muli to * macro
/// (operation (name arith.muli) (result-types type) (operands a b)) → (* (: type) a b)
fn compressArithMul(allocator: std.mem.Allocator, op_list: PersistentVector(*Value)) !*Value {
    // Get result type
    const result_type = getNamedValue(op_list, "result-types") orelse return error.MissingResultTypes;

    // Get operands
    var operands_list = try getNamedElements(op_list, "operands", allocator) orelse return error.MissingOperands;
    defer operands_list.deinit(allocator);

    if (operands_list.items.len != 2) return error.InvalidOperandCount;

    // Build (: type)
    var type_vec = PersistentVector(*Value).init(allocator, null);
    type_vec = try type_vec.push(try createIdentifier(allocator, ":"));
    type_vec = try type_vec.push(result_type);
    const type_expr = try createList(allocator, type_vec);

    // Build (* (: type) a b)
    var result_vec = PersistentVector(*Value).init(allocator, null);
    result_vec = try result_vec.push(try createIdentifier(allocator, "*"));
    result_vec = try result_vec.push(type_expr);
    result_vec = try result_vec.push(operands_list.items[0]);
    result_vec = try result_vec.push(operands_list.items[1]);

    return try createList(allocator, result_vec);
}

/// Compress arith.constant to constant macro
/// (operation (name arith.constant) [result-bindings] (result-types type) (attributes {:value (: val type)}))
/// → (constant [%x] (: val type))
fn compressArithConstant(allocator: std.mem.Allocator, op_list: PersistentVector(*Value)) !*Value {
    // Get attributes map
    const attrs_val = getNamedValue(op_list, "attributes") orelse return error.MissingAttributes;
    if (attrs_val.type != .map) return error.InvalidAttributes;

    // Find :value in the attributes map
    const attrs_map = attrs_val.data.map;
    var typed_value: ?*Value = null;
    var i: usize = 0;
    while (i < attrs_map.len()) : (i += 2) {
        const key = attrs_map.at(i);
        if (key.type == .keyword and std.mem.eql(u8, key.data.atom, ":value")) {
            if (i + 1 < attrs_map.len()) {
                typed_value = attrs_map.at(i + 1);
            }
            break;
        }
    }
    const value_expr = typed_value orelse return error.MissingValueAttribute;

    // Check for optional result binding
    const bindings_val = getNamedValue(op_list, "result-bindings");
    var binding_name: ?*Value = null;
    if (bindings_val) |bindings| {
        if (bindings.type == .vector) {
            const bindings_vec = bindings.data.vector;
            if (bindings_vec.len() > 0) {
                binding_name = bindings_vec.at(0);
            }
        }
    }

    // Build result: (constant [%x] (: val type))
    var result_vec = PersistentVector(*Value).init(allocator, null);
    result_vec = try result_vec.push(try createIdentifier(allocator, "constant"));
    if (binding_name) |name| {
        result_vec = try result_vec.push(name);
    }
    result_vec = try result_vec.push(value_expr);

    return try createList(allocator, result_vec);
}

/// Compress func.return to return macro
/// (operation (name func.return) (operands x)) → (return x)
fn compressFuncReturn(allocator: std.mem.Allocator, op_list: PersistentVector(*Value)) !*Value {
    // Get operands
    var operands_list = try getNamedElements(op_list, "operands", allocator) orelse return error.MissingOperands;
    defer operands_list.deinit(allocator);

    if (operands_list.items.len != 1) return error.InvalidOperandCount;

    // Build (return x)
    var result_vec = PersistentVector(*Value).init(allocator, null);
    result_vec = try result_vec.push(try createIdentifier(allocator, "return"));
    result_vec = try result_vec.push(operands_list.items[0]);

    return try createList(allocator, result_vec);
}

/// Compress func.call to call macro
/// (operation (name func.call) [result-bindings] [result-types] (operands ...) (attributes {:callee @name}))
/// → (call [%result] @name args... ret-type)
fn compressFuncCall(allocator: std.mem.Allocator, op_list: PersistentVector(*Value)) !*Value {
    // Get callee from attributes
    const attrs_val = getNamedValue(op_list, "attributes") orelse return error.MissingAttributes;
    if (attrs_val.type != .map) return error.InvalidAttributes;

    const attrs_map = attrs_val.data.map;
    var callee: ?*Value = null;
    var i: usize = 0;
    while (i < attrs_map.len()) : (i += 2) {
        const key = attrs_map.at(i);
        if (key.type == .keyword and std.mem.eql(u8, key.data.atom, ":callee")) {
            if (i + 1 < attrs_map.len()) {
                callee = attrs_map.at(i + 1);
            }
            break;
        }
    }
    const callee_name = callee orelse return error.MissingCalleeAttribute;

    // Get operands
    var operands_list = try getNamedElements(op_list, "operands", allocator) orelse blk: {
        const empty_list = std.ArrayList(*Value){};
        break :blk empty_list;
    };
    defer operands_list.deinit(allocator);

    // Get result type
    const result_type = getNamedValue(op_list, "result-types");

    // Get optional result binding
    const bindings_val = getNamedValue(op_list, "result-bindings");
    var binding_name: ?*Value = null;
    if (bindings_val) |bindings| {
        if (bindings.type == .vector) {
            const bindings_vec = bindings.data.vector;
            if (bindings_vec.len() > 0) {
                binding_name = bindings_vec.at(0);
            }
        }
    }

    // Build (call [%result] @name args... ret-type)
    var result_vec = PersistentVector(*Value).init(allocator, null);
    result_vec = try result_vec.push(try createIdentifier(allocator, "call"));
    if (binding_name) |name| {
        result_vec = try result_vec.push(name);
    }
    result_vec = try result_vec.push(callee_name);
    for (operands_list.items) |operand| {
        result_vec = try result_vec.push(operand);
    }
    // The call macro ALWAYS needs a return type as the last argument
    // If there's no result type, use () for void
    if (result_type) |rtype| {
        result_vec = try result_vec.push(rtype);
    } else {
        // Create empty list () for void return
        const void_type = try createList(allocator, PersistentVector(*Value).init(allocator, null));
        result_vec = try result_vec.push(void_type);
    }

    return try createList(allocator, result_vec);
}

/// Compress generic operation to op macro
/// (operation (name X) [result-bindings] [result-types] [operands] [attributes] [regions])
/// → (op [%binding] [(: type)] (X [attrs] [operands] regions...))
fn compressGenericOp(allocator: std.mem.Allocator, op_list: PersistentVector(*Value)) !*Value {
    // Get operation name
    const name_val = getNamedValue(op_list, "name") orelse return error.MissingOperationName;

    // Get optional components
    const bindings_val = getNamedValue(op_list, "result-bindings");
    var binding_name: ?*Value = null;
    if (bindings_val) |bindings| {
        if (bindings.type == .vector) {
            const bindings_vec = bindings.data.vector;
            if (bindings_vec.len() > 0) {
                binding_name = bindings_vec.at(0);
            }
        }
    }

    const result_type = getNamedValue(op_list, "result-types");
    const attrs_val = getNamedValue(op_list, "attributes");

    var operands_list = try getNamedElements(op_list, "operands", allocator) orelse blk: {
        const empty_list = std.ArrayList(*Value){};
        break :blk empty_list;
    };
    defer operands_list.deinit(allocator);

    var regions_list = try getNamedElements(op_list, "regions", allocator) orelse blk: {
        const empty_list = std.ArrayList(*Value){};
        break :blk empty_list;
    };
    defer regions_list.deinit(allocator);

    // Build operation call: (op-name [attrs] [operands] regions...)
    var op_call_vec = PersistentVector(*Value).init(allocator, null);
    op_call_vec = try op_call_vec.push(name_val);

    // Add attributes if present
    if (attrs_val) |attrs| {
        op_call_vec = try op_call_vec.push(attrs);
    }

    // Add operands as vector
    if (operands_list.items.len > 0) {
        var operands_vec = PersistentVector(*Value).init(allocator, null);
        for (operands_list.items) |operand| {
            operands_vec = try operands_vec.push(operand);
        }
        const operands_vector = try createVector(allocator, operands_vec);
        op_call_vec = try op_call_vec.push(operands_vector);
    }

    // Add regions
    for (regions_list.items) |region| {
        op_call_vec = try op_call_vec.push(region);
    }

    const op_call = try createList(allocator, op_call_vec);

    // Build final result: (op [%binding] [(: type)] op-call)
    var result_vec = PersistentVector(*Value).init(allocator, null);
    result_vec = try result_vec.push(try createIdentifier(allocator, "op"));

    if (binding_name) |name| {
        result_vec = try result_vec.push(name);
    }

    if (result_type) |rtype| {
        // Create (: type) form
        var type_vec = PersistentVector(*Value).init(allocator, null);
        type_vec = try type_vec.push(try createIdentifier(allocator, ":"));
        type_vec = try type_vec.push(rtype);
        const type_expr = try createList(allocator, type_vec);
        result_vec = try result_vec.push(type_expr);
    }

    result_vec = try result_vec.push(op_call);

    return try createList(allocator, result_vec);
}

/// Try to compress an operation value to its macro form
/// Returns the compressed form if successful, or null if it should stay as-is
fn tryCompressOperation(allocator: std.mem.Allocator, value: *Value) !?*Value {
    // Must be a list starting with "operation"
    if (!isListStartingWith(value, "operation")) return null;

    const op_list = value.data.list;

    // Try specific macro compressions first
    if (isArithAdd(op_list)) {
        return compressArithAdd(allocator, op_list) catch null;
    }

    if (isArithMul(op_list)) {
        return compressArithMul(allocator, op_list) catch null;
    }

    if (isArithConstant(op_list)) {
        return compressArithConstant(allocator, op_list) catch null;
    }

    if (isFuncReturn(op_list)) {
        return compressFuncReturn(allocator, op_list) catch null;
    }

    if (isFuncCall(op_list)) {
        return compressFuncCall(allocator, op_list) catch null;
    }

    // Skip func.func - it's too complex and rarely benefits from compression
    if (isFuncFunc(op_list)) {
        return null;
    }

    // Fall back to generic op macro
    return compressGenericOp(allocator, op_list) catch null;
}

/// Recursively compress macros in a value tree
pub fn compressMacros(allocator: std.mem.Allocator, value: *Value) error{OutOfMemory}!*Value {
    // Try to compress this value if it's an operation
    if (try tryCompressOperation(allocator, value)) |compressed| {
        // Recursively compress the result
        return compressMacros(allocator, compressed);
    }

    // If not an operation (or compression failed), recursively process children
    switch (value.type) {
        .list => {
            var new_vec = PersistentVector(*Value).init(allocator, null);
            for (value.data.list.slice()) |elem| {
                const compressed_elem = try compressMacros(allocator, elem);
                new_vec = try new_vec.push(compressed_elem);
            }
            return try createList(allocator, new_vec);
        },
        .vector => {
            var new_vec = PersistentVector(*Value).init(allocator, null);
            for (value.data.vector.slice()) |elem| {
                const compressed_elem = try compressMacros(allocator, elem);
                new_vec = try new_vec.push(compressed_elem);
            }
            return try createVector(allocator, new_vec);
        },
        .map => {
            var new_vec = PersistentVector(*Value).init(allocator, null);
            for (value.data.map.slice()) |elem| {
                const compressed_elem = try compressMacros(allocator, elem);
                new_vec = try new_vec.push(compressed_elem);
            }
            const map_val = try allocator.create(Value);
            map_val.* = Value{
                .type = .map,
                .data = .{ .map = new_vec },
            };
            return map_val;
        },
        .has_type => {
            const compressed_value = try compressMacros(allocator, value.data.has_type.value);
            const compressed_type = try compressMacros(allocator, value.data.has_type.type_expr);
            return try createHasType(allocator, compressed_value, compressed_type);
        },
        .attr_expr => {
            const compressed_inner = try compressMacros(allocator, value.data.attr_expr);
            const result = try allocator.create(Value);
            result.* = Value{
                .type = .attr_expr,
                .data = .{ .attr_expr = compressed_inner },
            };
            return result;
        },
        .function_type => {
            var new_inputs = PersistentVector(*Value).init(allocator, null);
            for (value.data.function_type.inputs.slice()) |elem| {
                const compressed_elem = try compressMacros(allocator, elem);
                new_inputs = try new_inputs.push(compressed_elem);
            }
            var new_results = PersistentVector(*Value).init(allocator, null);
            for (value.data.function_type.results.slice()) |elem| {
                const compressed_elem = try compressMacros(allocator, elem);
                new_results = try new_results.push(compressed_elem);
            }
            const result = try allocator.create(Value);
            result.* = Value{
                .type = .function_type,
                .data = .{ .function_type = .{
                    .inputs = new_inputs,
                    .results = new_results,
                } },
            };
            return result;
        },
        // Atoms don't need processing
        else => return value,
    }
}
