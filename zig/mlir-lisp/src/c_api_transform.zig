// Transformation functions for converting high-level syntax to MLIR operations
// Uses Zig APIs internally but exports C-compatible interface

const std = @import("std");
const reader = @import("reader.zig");
const Value = reader.Value;
const ValueType = reader.ValueType;
const vector = @import("collections/vector.zig");
const PersistentVector = vector.PersistentVector;

/// Helper to create an identifier Value
fn createIdentifier(allocator: std.mem.Allocator, name: []const u8) !*Value {
    const value = try allocator.create(Value);
    value.* = Value{
        .type = .identifier,
        .data = .{ .atom = name },
    };
    return value;
}

/// Helper to create a value_id Value (%name)
fn createValueId(allocator: std.mem.Allocator, name: []const u8) !*Value {
    const value = try allocator.create(Value);
    value.* = Value{
        .type = .value_id,
        .data = .{ .atom = name },
    };
    return value;
}

/// Helper to create a keyword Value
fn createKeyword(allocator: std.mem.Allocator, name: []const u8) !*Value {
    const value = try allocator.create(Value);
    value.* = Value{
        .type = .keyword,
        .data = .{ .atom = name },
    };
    return value;
}

/// Helper to create a list Value from a vector
fn createList(allocator: std.mem.Allocator, vec: PersistentVector(*Value)) !*Value {
    const value = try allocator.create(Value);
    value.* = Value{
        .type = .list,
        .data = .{ .list = vec },
    };
    return value;
}

/// Helper to create a vector Value
fn createVector(allocator: std.mem.Allocator, vec: PersistentVector(*Value)) !*Value {
    const value = try allocator.create(Value);
    value.* = Value{
        .type = .vector,
        .data = .{ .vector = vec },
    };
    return value;
}

/// Helper to create a map Value
fn createMap(allocator: std.mem.Allocator, vec: PersistentVector(*Value)) !*Value {
    const value = try allocator.create(Value);
    value.* = Value{
        .type = .map,
        .data = .{ .map = vec },
    };
    return value;
}

// Transform (@test i64) => (operation (name func.call) ...)
// Takes args-only (without the "call" identifier) for use as a macro
// Returns null on error (C-compatible calling convention)
pub export fn transformCallToOperation(allocator_ptr: ?*anyopaque, call_args: ?*anyopaque) ?*anyopaque {
    if (allocator_ptr == null or call_args == null) return null;

    const alloc_ptr: *std.mem.Allocator = @ptrCast(@alignCast(allocator_ptr));
    const allocator = alloc_ptr.*;
    const args_value: *Value = @ptrCast(@alignCast(call_args));

    // Expect a list containing (@test i64)
    if (args_value.type != .list) return null;
    const args_vec = args_value.data.list;

    if (args_vec.len() < 2) return null; // Need at least (<symbol> <type>)

    // Extract: @test, i64
    const callee_symbol = args_vec.at(0);
    const return_type = args_vec.at(1);

    // Build result: (operation (name func.call) (result-bindings [%result0]) (result-types i64) (attributes { :callee @test }))

    const result = transformCallToOperationInner(allocator, callee_symbol, return_type) catch return null;
    return @ptrCast(result);
}

fn transformCallToOperationInner(
    allocator: std.mem.Allocator,
    callee_symbol: *Value,
    return_type: *Value,
) !*Value {
    // Create "operation" identifier
    const operation_ident = try createIdentifier(allocator, "operation");

    // Create (name func.call)
    var name_vec = PersistentVector(*Value).init(allocator, null);
    name_vec = try name_vec.push(try createIdentifier(allocator, "name"));
    name_vec = try name_vec.push(try createIdentifier(allocator, "func.call"));
    const name_clause = try createList(allocator, name_vec);

    // Create (result-bindings [%result0])
    var gensym_vec = PersistentVector(*Value).init(allocator, null);
    gensym_vec = try gensym_vec.push(try createValueId(allocator, "%result0"));
    const bindings_vector = try createVector(allocator, gensym_vec);

    var bindings_vec = PersistentVector(*Value).init(allocator, null);
    bindings_vec = try bindings_vec.push(try createIdentifier(allocator, "result-bindings"));
    bindings_vec = try bindings_vec.push(bindings_vector);
    const bindings_clause = try createList(allocator, bindings_vec);

    // Create (result-types i64)
    var types_vec = PersistentVector(*Value).init(allocator, null);
    types_vec = try types_vec.push(try createIdentifier(allocator, "result-types"));
    types_vec = try types_vec.push(return_type);
    const types_clause = try createList(allocator, types_vec);

    // Create (attributes { :callee @test })
    // Map is a flat vector of key-value pairs
    var map_vec = PersistentVector(*Value).init(allocator, null);
    map_vec = try map_vec.push(try createKeyword(allocator, ":callee"));
    map_vec = try map_vec.push(callee_symbol);
    const attributes_map = try createMap(allocator, map_vec);

    var attrs_vec = PersistentVector(*Value).init(allocator, null);
    attrs_vec = try attrs_vec.push(try createIdentifier(allocator, "attributes"));
    attrs_vec = try attrs_vec.push(attributes_map);
    const attrs_clause = try createList(allocator, attrs_vec);

    // Build the final operation list: (operation <name> <bindings> <types> <attrs>)
    var op_vec = PersistentVector(*Value).init(allocator, null);
    op_vec = try op_vec.push(operation_ident);
    op_vec = try op_vec.push(name_clause);
    op_vec = try op_vec.push(bindings_clause);
    op_vec = try op_vec.push(types_clause);
    op_vec = try op_vec.push(attrs_clause);

    return try createList(allocator, op_vec);
}

// Example usage:
// The macro system will call transformCallToOperation with:
//   - allocator: pointer to std.mem.Allocator
//   - call_args: *Value list containing (@func_name return_type)
// The function will return a *Value representing the full operation syntax
