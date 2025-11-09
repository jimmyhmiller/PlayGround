/// Built-in macros for common operations
/// These macros expand high-level constructs into MLIR operations

const std = @import("std");
const macro_expander = @import("macro_expander.zig");
const MacroExpander = macro_expander.MacroExpander;
const reader = @import("reader.zig");
const Value = reader.Value;
const ValueType = reader.ValueType;
const vector = @import("collections/vector.zig");
const PersistentVector = vector.PersistentVector;
const linked_list = @import("collections/linked_list.zig");
const PersistentLinkedList = linked_list.PersistentLinkedList;
const c_api_macro = @import("c_api_macro.zig");
const c_api_transform = @import("c_api_transform.zig");

// Zig wrapper for C export function - bridges calling conventions
fn callTransformWrapper(allocator: ?*anyopaque, value: ?*anyopaque) ?*anyopaque {
    return c_api_transform.transformCallToOperation(allocator, value);
}

/// Register all built-in macros with the expander
pub fn registerBuiltinMacros(expander: *MacroExpander) !void {
    // Register 'call' macro using the generic wrapper
    // Transforms (call @func type) into full operation syntax
    try expander.registerMacro(
        "call",
        c_api_macro.wrapCTransformAsMacro(&callTransformWrapper),
    );

    // Register arithmetic macros
    try expander.registerMacro("+", addMacro);
    try expander.registerMacro("*", mulMacro);
    try expander.registerMacro("constant", constantMacro);

    // Register control flow macros
    try expander.registerMacro("return", returnMacro);

    // Register function definition macros
    try expander.registerMacro("defn", defnMacro);

    // Register general operation macro
    try expander.registerMacro("op", opMacro);
}

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

/// Create a keyword Value
fn createKeyword(allocator: std.mem.Allocator, name: []const u8) !*Value {
    const value = try allocator.create(Value);
    value.* = Value{
        .type = .keyword,
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

/// Convert linked list to vector
fn linkedListToVector(
    allocator: std.mem.Allocator,
    list: *const PersistentLinkedList(*Value),
) !PersistentVector(*Value) {
    var vec = PersistentVector(*Value).init(allocator, null);
    var iter = list.iterator();
    while (iter.next()) |value| {
        vec = try vec.push(value);
    }
    return vec;
}

// ============================================================================
// Macro Implementations
// ============================================================================

/// if macro: (if condition then-branch else-branch)
/// Expands to: (scf.if condition (region [(block [then-branch])]) (region [(block [else-branch])]))
fn ifMacro(
    allocator: std.mem.Allocator,
    args: *const PersistentLinkedList(*Value),
) !*Value {
    // Validate: need exactly 3 args (condition, then, else)
    if (args.len() != 3) {
        std.debug.print("if macro requires 3 arguments (condition then else), got {}\n", .{args.len()});
        return error.InvalidMacroArgs;
    }

    // Extract arguments
    var iter = args.iterator();
    const condition = iter.next() orelse return error.InvalidMacroArgs;
    const then_branch = iter.next() orelse return error.InvalidMacroArgs;
    const else_branch = iter.next() orelse return error.InvalidMacroArgs;

    // Build the scf.if operation
    // (scf.if condition
    //   (region [(block [then-branch])])
    //   (region [(block [else-branch])]))

    var result_vec = PersistentVector(*Value).init(allocator, null);

    // Add scf.if identifier
    result_vec = try result_vec.push(try createIdentifier(allocator, "scf.if"));

    // Add condition
    result_vec = try result_vec.push(condition);

    // Build then region: (region [(block [then-branch])])
    var then_block_vec = PersistentVector(*Value).init(allocator, null);
    then_block_vec = try then_block_vec.push(then_branch);
    const then_block = try createVector(allocator, then_block_vec);

    var then_blocks_vec = PersistentVector(*Value).init(allocator, null);
    then_blocks_vec = try then_blocks_vec.push(then_block);
    const then_blocks = try createVector(allocator, then_blocks_vec);

    var then_region_vec = PersistentVector(*Value).init(allocator, null);
    then_region_vec = try then_region_vec.push(try createIdentifier(allocator, "region"));
    then_region_vec = try then_region_vec.push(then_blocks);
    const then_region = try createList(allocator, then_region_vec);

    result_vec = try result_vec.push(then_region);

    // Build else region: (region [(block [else-branch])])
    var else_block_vec = PersistentVector(*Value).init(allocator, null);
    else_block_vec = try else_block_vec.push(else_branch);
    const else_block = try createVector(allocator, else_block_vec);

    var else_blocks_vec = PersistentVector(*Value).init(allocator, null);
    else_blocks_vec = try else_blocks_vec.push(else_block);
    const else_blocks = try createVector(allocator, else_blocks_vec);

    var else_region_vec = PersistentVector(*Value).init(allocator, null);
    else_region_vec = try else_region_vec.push(try createIdentifier(allocator, "region"));
    else_region_vec = try else_region_vec.push(else_blocks);
    const else_region = try createList(allocator, else_region_vec);

    result_vec = try result_vec.push(else_region);

    return try createList(allocator, result_vec);
}

/// when macro: (when condition body)
/// Expands to: (scf.if condition (region [(block [body])]) (region [(block [])]))
fn whenMacro(
    allocator: std.mem.Allocator,
    args: *const PersistentLinkedList(*Value),
) !*Value {
    // Validate: need exactly 2 args (condition, body)
    if (args.len() != 2) {
        std.debug.print("when macro requires 2 arguments (condition body), got {}\n", .{args.len()});
        return error.InvalidMacroArgs;
    }

    // Extract arguments
    var iter = args.iterator();
    const condition = iter.next() orelse return error.InvalidMacroArgs;
    const body = iter.next() orelse return error.InvalidMacroArgs;

    // Build the scf.if operation with empty else branch
    var result_vec = PersistentVector(*Value).init(allocator, null);

    // Add scf.if identifier
    result_vec = try result_vec.push(try createIdentifier(allocator, "scf.if"));

    // Add condition
    result_vec = try result_vec.push(condition);

    // Build then region: (region [(block [body])])
    var then_block_vec = PersistentVector(*Value).init(allocator, null);
    then_block_vec = try then_block_vec.push(body);
    const then_block = try createVector(allocator, then_block_vec);

    var then_blocks_vec = PersistentVector(*Value).init(allocator, null);
    then_blocks_vec = try then_blocks_vec.push(then_block);
    const then_blocks = try createVector(allocator, then_blocks_vec);

    var then_region_vec = PersistentVector(*Value).init(allocator, null);
    then_region_vec = try then_region_vec.push(try createIdentifier(allocator, "region"));
    then_region_vec = try then_region_vec.push(then_blocks);
    const then_region = try createList(allocator, then_region_vec);

    result_vec = try result_vec.push(then_region);

    // Build empty else region: (region [(block [])])
    const empty_block_vec = PersistentVector(*Value).init(allocator, null);
    const empty_block = try createVector(allocator, empty_block_vec);

    var else_blocks_vec = PersistentVector(*Value).init(allocator, null);
    else_blocks_vec = try else_blocks_vec.push(empty_block);
    const else_blocks = try createVector(allocator, else_blocks_vec);

    var else_region_vec = PersistentVector(*Value).init(allocator, null);
    else_region_vec = try else_region_vec.push(try createIdentifier(allocator, "region"));
    else_region_vec = try else_region_vec.push(else_blocks);
    const else_region = try createList(allocator, else_region_vec);

    result_vec = try result_vec.push(else_region);

    return try createList(allocator, result_vec);
}

/// unless macro: (unless condition body)
/// Expands to: (scf.if condition (region [(block [])]) (region [(block [body])]))
fn unlessMacro(
    allocator: std.mem.Allocator,
    args: *const PersistentLinkedList(*Value),
) !*Value {
    // Validate: need exactly 2 args (condition, body)
    if (args.len() != 2) {
        std.debug.print("unless macro requires 2 arguments (condition body), got {}\n", .{args.len()});
        return error.InvalidMacroArgs;
    }

    // Extract arguments
    var iter = args.iterator();
    const condition = iter.next() orelse return error.InvalidMacroArgs;
    const body = iter.next() orelse return error.InvalidMacroArgs;

    // Build the scf.if operation with swapped branches
    var result_vec = PersistentVector(*Value).init(allocator, null);

    // Add scf.if identifier
    result_vec = try result_vec.push(try createIdentifier(allocator, "scf.if"));

    // Add condition
    result_vec = try result_vec.push(condition);

    // Build empty then region: (region [(block [])])
    const empty_block_vec = PersistentVector(*Value).init(allocator, null);
    const empty_block = try createVector(allocator, empty_block_vec);

    var then_blocks_vec = PersistentVector(*Value).init(allocator, null);
    then_blocks_vec = try then_blocks_vec.push(empty_block);
    const then_blocks = try createVector(allocator, then_blocks_vec);

    var then_region_vec = PersistentVector(*Value).init(allocator, null);
    then_region_vec = try then_region_vec.push(try createIdentifier(allocator, "region"));
    then_region_vec = try then_region_vec.push(then_blocks);
    const then_region = try createList(allocator, then_region_vec);

    result_vec = try result_vec.push(then_region);

    // Build else region with body: (region [(block [body])])
    var else_block_vec = PersistentVector(*Value).init(allocator, null);
    else_block_vec = try else_block_vec.push(body);
    const else_block = try createVector(allocator, else_block_vec);

    var else_blocks_vec = PersistentVector(*Value).init(allocator, null);
    else_blocks_vec = try else_blocks_vec.push(else_block);
    const else_blocks = try createVector(allocator, else_blocks_vec);

    var else_region_vec = PersistentVector(*Value).init(allocator, null);
    else_region_vec = try else_region_vec.push(try createIdentifier(allocator, "region"));
    else_region_vec = try else_region_vec.push(else_blocks);
    const else_region = try createList(allocator, else_region_vec);

    result_vec = try result_vec.push(else_region);

    return try createList(allocator, result_vec);
}

/// + macro: (+ (: type) operand1 operand2)
/// Expands to: (operation (name arith.addi) (result-types type) (operands operand1 operand2))
fn addMacro(
    allocator: std.mem.Allocator,
    args: *const PersistentLinkedList(*Value),
) !*Value {
    // Validate: need exactly 3 args ((: type), operand1, operand2)
    if (args.len() != 3) {
        std.debug.print("+ macro requires 3 arguments ((: type) operand1 operand2), got {}\n", .{args.len()});
        return error.InvalidMacroArgs;
    }

    // Extract arguments
    var iter = args.iterator();
    const type_expr = iter.next() orelse return error.InvalidMacroArgs;
    const operand1 = iter.next() orelse return error.InvalidMacroArgs;
    const operand2 = iter.next() orelse return error.InvalidMacroArgs;

    // Extract type from (: type) expression
    // Should be a list with 2 elements: : and the type
    if (type_expr.type != .list) {
        std.debug.print("+ macro first argument must be (: type)\n", .{});
        return error.InvalidMacroArgs;
    }

    const type_list = type_expr.data.list;
    if (type_list.len() != 2) {
        std.debug.print("+ macro first argument must be (: type), got {} elements\n", .{type_list.len()});
        return error.InvalidMacroArgs;
    }

    const colon = type_list.at(0);
    if (colon.type != .identifier or !std.mem.eql(u8, colon.data.atom, ":")) {
        std.debug.print("+ macro first argument must start with :\n", .{});
        return error.InvalidMacroArgs;
    }

    const result_type = type_list.at(1);

    // Build operation structure:
    // (operation
    //   (name arith.addi)
    //   (result-types type)
    //   (operands operand1 operand2))

    var op_vec = PersistentVector(*Value).init(allocator, null);

    // Add "operation" identifier
    op_vec = try op_vec.push(try createIdentifier(allocator, "operation"));

    // Add (name arith.addi)
    var name_vec = PersistentVector(*Value).init(allocator, null);
    name_vec = try name_vec.push(try createIdentifier(allocator, "name"));
    name_vec = try name_vec.push(try createIdentifier(allocator, "arith.addi"));
    op_vec = try op_vec.push(try createList(allocator, name_vec));

    // Add (result-types type)
    var types_vec = PersistentVector(*Value).init(allocator, null);
    types_vec = try types_vec.push(try createIdentifier(allocator, "result-types"));
    types_vec = try types_vec.push(result_type);
    op_vec = try op_vec.push(try createList(allocator, types_vec));

    // Add (operands operand1 operand2)
    var operands_vec = PersistentVector(*Value).init(allocator, null);
    operands_vec = try operands_vec.push(try createIdentifier(allocator, "operands"));
    operands_vec = try operands_vec.push(operand1);
    operands_vec = try operands_vec.push(operand2);
    op_vec = try op_vec.push(try createList(allocator, operands_vec));

    return try createList(allocator, op_vec);
}

/// * macro: (* (: type) operand1 operand2)
/// Expands to: (operation (name arith.muli) (result-types type) (operands operand1 operand2))
fn mulMacro(
    allocator: std.mem.Allocator,
    args: *const PersistentLinkedList(*Value),
) !*Value {
    // Validate: need exactly 3 args ((: type), operand1, operand2)
    if (args.len() != 3) {
        std.debug.print("* macro requires 3 arguments ((: type) operand1 operand2), got {}\n", .{args.len()});
        return error.InvalidMacroArgs;
    }

    // Extract arguments
    var iter = args.iterator();
    const type_expr = iter.next() orelse return error.InvalidMacroArgs;
    const operand1 = iter.next() orelse return error.InvalidMacroArgs;
    const operand2 = iter.next() orelse return error.InvalidMacroArgs;

    // Extract type from (: type) expression
    // Should be a list with 2 elements: : and the type
    if (type_expr.type != .list) {
        std.debug.print("* macro first argument must be (: type)\n", .{});
        return error.InvalidMacroArgs;
    }

    const type_list = type_expr.data.list;
    if (type_list.len() != 2) {
        std.debug.print("* macro first argument must be (: type), got {} elements\n", .{type_list.len()});
        return error.InvalidMacroArgs;
    }

    const colon = type_list.at(0);
    if (colon.type != .identifier or !std.mem.eql(u8, colon.data.atom, ":")) {
        std.debug.print("* macro first argument must start with :\n", .{});
        return error.InvalidMacroArgs;
    }

    const result_type = type_list.at(1);

    // Build operation structure:
    // (operation
    //   (name arith.muli)
    //   (result-types type)
    //   (operands operand1 operand2))

    var op_vec = PersistentVector(*Value).init(allocator, null);

    // Add "operation" identifier
    op_vec = try op_vec.push(try createIdentifier(allocator, "operation"));

    // Add (name arith.muli)
    var name_vec = PersistentVector(*Value).init(allocator, null);
    name_vec = try name_vec.push(try createIdentifier(allocator, "name"));
    name_vec = try name_vec.push(try createIdentifier(allocator, "arith.muli"));
    op_vec = try op_vec.push(try createList(allocator, name_vec));

    // Add (result-types type)
    var types_vec = PersistentVector(*Value).init(allocator, null);
    types_vec = try types_vec.push(try createIdentifier(allocator, "result-types"));
    types_vec = try types_vec.push(result_type);
    op_vec = try op_vec.push(try createList(allocator, types_vec));

    // Add (operands operand1 operand2)
    var operands_vec = PersistentVector(*Value).init(allocator, null);
    operands_vec = try operands_vec.push(try createIdentifier(allocator, "operands"));
    operands_vec = try operands_vec.push(operand1);
    operands_vec = try operands_vec.push(operand2);
    op_vec = try op_vec.push(try createList(allocator, operands_vec));

    return try createList(allocator, op_vec);
}

/// return macro: (return [operand])
/// Expands to: (operation (name func.return) [(operands operand)])
/// Supports both (return value) and (return) for void functions
fn returnMacro(
    allocator: std.mem.Allocator,
    args: *const PersistentLinkedList(*Value),
) !*Value {
    // Validate: need 0 or 1 arg
    if (args.len() > 1) {
        std.debug.print("return macro requires 0 or 1 argument (operand), got {}\n", .{args.len()});
        return error.InvalidMacroArgs;
    }

    // Extract the operand if present
    var iter = args.iterator();
    const operand = iter.next();

    // Build operation structure:
    // (operation
    //   (name func.return)
    //   [(operands operand)])

    var op_vec = PersistentVector(*Value).init(allocator, null);

    // Add "operation" identifier
    op_vec = try op_vec.push(try createIdentifier(allocator, "operation"));

    // Add (name func.return)
    var name_vec = PersistentVector(*Value).init(allocator, null);
    name_vec = try name_vec.push(try createIdentifier(allocator, "name"));
    name_vec = try name_vec.push(try createIdentifier(allocator, "func.return"));
    op_vec = try op_vec.push(try createList(allocator, name_vec));

    // Add (operands operand) only if operand is present
    if (operand) |op| {
        var operands_vec = PersistentVector(*Value).init(allocator, null);
        operands_vec = try operands_vec.push(try createIdentifier(allocator, "operands"));
        operands_vec = try operands_vec.push(op);
        op_vec = try op_vec.push(try createList(allocator, operands_vec));
    }

    return try createList(allocator, op_vec);
}

/// constant macro: (constant (: value type)) or (constant %name (: value type))
/// Expands to: (operation (name arith.constant) [result-bindings] (result-types type) (attributes { :value (: value type) }))
fn constantMacro(
    allocator: std.mem.Allocator,
    args: *const PersistentLinkedList(*Value),
) !*Value {
    // Validate: need 1 or 2 args
    if (args.len() < 1 or args.len() > 2) {
        std.debug.print("constant macro requires 1 or 2 arguments, got {}\n", .{args.len()});
        return error.InvalidMacroArgs;
    }

    var iter = args.iterator();
    const first_arg = iter.next() orelse return error.InvalidMacroArgs;

    var binding_name: ?*Value = null;
    var typed_value: *Value = undefined;

    // Check if first arg is a value_id (binding form)
    if (first_arg.type == .value_id) {
        // Form: (constant %name (: value type))
        if (args.len() != 2) {
            std.debug.print("constant macro with binding requires 2 arguments (%name (: value type)), got {}\n", .{args.len()});
            return error.InvalidMacroArgs;
        }
        binding_name = first_arg;
        typed_value = iter.next() orelse return error.InvalidMacroArgs;

        // Validate second arg is has_type
        if (typed_value.type != .has_type) {
            std.debug.print("constant macro second argument must be (: value type)\n", .{});
            return error.InvalidMacroArgs;
        }
    } else if (first_arg.type == .has_type) {
        // Form: (constant (: value type))
        if (args.len() != 1) {
            std.debug.print("constant macro without binding requires 1 argument ((: value type)), got {}\n", .{args.len()});
            return error.InvalidMacroArgs;
        }
        typed_value = first_arg;
    } else {
        std.debug.print("constant macro first argument must be %name or (: value type)\n", .{});
        return error.InvalidMacroArgs;
    }

    // Extract the type from has_type
    const value_type = typed_value.data.has_type.type_expr;

    // Build operation structure:
    // (operation
    //   (name arith.constant)
    //   [(result-bindings [%name])]  ; optional
    //   (result-types type)
    //   (attributes { :value (: value type) }))

    var op_vec = PersistentVector(*Value).init(allocator, null);

    // Add "operation" identifier
    op_vec = try op_vec.push(try createIdentifier(allocator, "operation"));

    // Add (name arith.constant)
    var name_vec = PersistentVector(*Value).init(allocator, null);
    name_vec = try name_vec.push(try createIdentifier(allocator, "name"));
    name_vec = try name_vec.push(try createIdentifier(allocator, "arith.constant"));
    op_vec = try op_vec.push(try createList(allocator, name_vec));

    // Add (result-bindings [%name]) if binding provided
    if (binding_name) |name| {
        var gensym_vec = PersistentVector(*Value).init(allocator, null);
        gensym_vec = try gensym_vec.push(name);
        const bindings_vector = try createVector(allocator, gensym_vec);

        var bindings_vec = PersistentVector(*Value).init(allocator, null);
        bindings_vec = try bindings_vec.push(try createIdentifier(allocator, "result-bindings"));
        bindings_vec = try bindings_vec.push(bindings_vector);
        op_vec = try op_vec.push(try createList(allocator, bindings_vec));
    }

    // Add (result-types type)
    var types_vec = PersistentVector(*Value).init(allocator, null);
    types_vec = try types_vec.push(try createIdentifier(allocator, "result-types"));
    types_vec = try types_vec.push(value_type);
    op_vec = try op_vec.push(try createList(allocator, types_vec));

    // Add (attributes { :value (: value type) })
    // Build the map: { :value (: value type) }
    var attrs_map = PersistentVector(*Value).init(allocator, null);
    attrs_map = try attrs_map.push(try createKeyword(allocator, ":value"));
    attrs_map = try attrs_map.push(typed_value);

    const attrs_map_val = try allocator.create(Value);
    attrs_map_val.* = Value{
        .type = .map,
        .data = .{ .map = attrs_map },
    };

    var attributes_vec = PersistentVector(*Value).init(allocator, null);
    attributes_vec = try attributes_vec.push(try createIdentifier(allocator, "attributes"));
    attributes_vec = try attributes_vec.push(attrs_map_val);
    op_vec = try op_vec.push(try createList(allocator, attributes_vec));

    return try createList(allocator, op_vec);
}

/// op macro: General-purpose operation macro
/// Forms:
///   (op %N (: index) (memref.dim [%B %c1]))           - with binding, type, and operands
///   (op (: index) (memref.dim [%B %c1]))              - with type and operands
///   (op (memref.store [%value %C %i %j]))             - with operands only
///   (op (memref.dim {attrs} [%B %c1]))                - with attributes and operands
///   (op (memref.dim {attrs}))                         - with attributes, no operands
///   (op (memref.dim))                                 - no attributes, no operands
///   (op %result (: i32) (scf.if %cond) (region ...) (region ...)) - with regions
/// Expands to: (operation (name ...) [result-bindings] [result-types] (operands ...) [regions])
fn opMacro(
    allocator: std.mem.Allocator,
    args: *const PersistentLinkedList(*Value),
) !*Value {
    // Need at least 1 argument (the operation call)
    if (args.len() < 1) {
        std.debug.print("op macro requires at least 1 argument, got {}\n", .{args.len()});
        return error.InvalidMacroArgs;
    }

    var iter = args.iterator();
    var current_arg = iter.next() orelse return error.InvalidMacroArgs;

    // Parse optional binding: %N
    var binding_name: ?*Value = null;
    if (current_arg.type == .value_id) {
        binding_name = current_arg;
        current_arg = iter.next() orelse {
            std.debug.print("op macro: expected more arguments after binding\n", .{});
            return error.InvalidMacroArgs;
        };
    }

    // Parse optional type annotation: (: type)
    var result_type: ?*Value = null;
    if (current_arg.type == .has_type) {
        // has_type form: (: value type) -> extract type
        result_type = current_arg.data.has_type.type_expr;
        current_arg = iter.next() orelse {
            std.debug.print("op macro: expected operation call after type annotation\n", .{});
            return error.InvalidMacroArgs;
        };
    } else if (current_arg.type == .list) {
        // Check if it's a list starting with ":"
        const list_vec = current_arg.data.list;
        if (list_vec.len() >= 2) {
            const first_elem = list_vec.at(0);
            if (first_elem.type == .identifier and std.mem.eql(u8, first_elem.data.atom, ":")) {
                // It's a (: type) form
                result_type = list_vec.at(1);
                current_arg = iter.next() orelse {
                    std.debug.print("op macro: expected operation call after type annotation\n", .{});
                    return error.InvalidMacroArgs;
                };
            }
        }
    }

    // Parse operation call: (op-name [operands...])
    if (current_arg.type != .list) {
        std.debug.print("op macro: operation call must be a list, got {}\n", .{current_arg.type});
        return error.InvalidMacroArgs;
    }

    const op_call = current_arg.data.list;
    if (op_call.len() < 1) {
        std.debug.print("op macro: operation call must have at least operation name\n", .{});
        return error.InvalidMacroArgs;
    }

    // Extract operation name
    const op_name = op_call.at(0);
    if (op_name.type != .identifier) {
        std.debug.print("op macro: operation name must be an identifier\n", .{});
        return error.InvalidMacroArgs;
    }

    // Check for optional attribute map at index 1
    var attr_map: ?*Value = null;
    var operands_index: usize = 1;

    if (op_call.len() >= 2) {
        const second_elem = op_call.at(1);
        if (second_elem.type == .map) {
            // Second element is an attribute map
            attr_map = second_elem;
            operands_index = 2;
        }
    }

    // Extract operands - optional vector [operands...] at operands_index
    var operands_vec = PersistentVector(*Value).init(allocator, null);
    var regions_start = operands_index;

    if (op_call.len() > operands_index) {
        const operands_arg = op_call.at(operands_index);
        if (operands_arg.type == .vector) {
            // Copy operands from the vector
            for (operands_arg.data.vector.slice()) |operand| {
                operands_vec = try operands_vec.push(operand);
            }
            // Regions start after the operands vector
            regions_start = operands_index + 1;
        }
        // If not a vector, treat it as the start of regions (operands are optional)
    }

    // Extract regions from the operation call (after operands, if any)
    var regions = std.ArrayList(*Value){};
    defer regions.deinit(allocator);
    if (op_call.len() > regions_start) {
        var i: usize = regions_start;
        while (i < op_call.len()) : (i += 1) {
            try regions.append(allocator, op_call.at(i));
        }
    }

    // Build operation structure
    var op_vec = PersistentVector(*Value).init(allocator, null);

    // Add "operation" identifier
    op_vec = try op_vec.push(try createIdentifier(allocator, "operation"));

    // Add (name operation-name)
    var name_vec = PersistentVector(*Value).init(allocator, null);
    name_vec = try name_vec.push(try createIdentifier(allocator, "name"));
    name_vec = try name_vec.push(op_name);
    op_vec = try op_vec.push(try createList(allocator, name_vec));

    // Add (result-bindings [%name]) if binding provided
    if (binding_name) |name| {
        var bindings_vec_inner = PersistentVector(*Value).init(allocator, null);
        bindings_vec_inner = try bindings_vec_inner.push(name);
        const bindings_vector = try createVector(allocator, bindings_vec_inner);

        var bindings_vec = PersistentVector(*Value).init(allocator, null);
        bindings_vec = try bindings_vec.push(try createIdentifier(allocator, "result-bindings"));
        bindings_vec = try bindings_vec.push(bindings_vector);
        op_vec = try op_vec.push(try createList(allocator, bindings_vec));
    }

    // Add (result-types type) if type provided
    if (result_type) |rtype| {
        var types_vec = PersistentVector(*Value).init(allocator, null);
        types_vec = try types_vec.push(try createIdentifier(allocator, "result-types"));
        types_vec = try types_vec.push(rtype);
        op_vec = try op_vec.push(try createList(allocator, types_vec));
    }

    // Add (operands ...) if there are operands
    if (operands_vec.len() > 0) {
        var ops_vec = PersistentVector(*Value).init(allocator, null);
        ops_vec = try ops_vec.push(try createIdentifier(allocator, "operands"));
        for (operands_vec.slice()) |operand| {
            ops_vec = try ops_vec.push(operand);
        }
        op_vec = try op_vec.push(try createList(allocator, ops_vec));
    }

    // Add (attributes {...}) if attributes were provided
    if (attr_map) |attrs| {
        var attrs_vec = PersistentVector(*Value).init(allocator, null);
        attrs_vec = try attrs_vec.push(try createIdentifier(allocator, "attributes"));
        attrs_vec = try attrs_vec.push(attrs);
        op_vec = try op_vec.push(try createList(allocator, attrs_vec));
    }

    // Handle regions - they should be (region ...) forms from the operation call
    if (regions.items.len > 0) {
        // Wrap all regions in a (regions ...) form
        var regions_vec = PersistentVector(*Value).init(allocator, null);
        regions_vec = try regions_vec.push(try createIdentifier(allocator, "regions"));
        for (regions.items) |region| {
            regions_vec = try regions_vec.push(region);
        }
        op_vec = try op_vec.push(try createList(allocator, regions_vec));
    }

    return try createList(allocator, op_vec);
}

/// defn macro: (defn name [(: arg1 type1) (: arg2 type2) ...] return_type body...)
/// Expands to: (operation (name func.func) (attributes ...) (regions ...))
fn defnMacro(
    allocator: std.mem.Allocator,
    args: *const PersistentLinkedList(*Value),
) !*Value {
    // Validate: need at least 4 args (name, args-vector, return-type, body...)
    if (args.len() < 4) {
        std.debug.print("defn macro requires at least 4 arguments (name args return-type body...), got {}\n", .{args.len()});
        return error.InvalidMacroArgs;
    }

    // Extract arguments
    var iter = args.iterator();
    const name_val = iter.next() orelse return error.InvalidMacroArgs;
    const args_vec_val = iter.next() orelse return error.InvalidMacroArgs;
    const return_type_val = iter.next() orelse return error.InvalidMacroArgs;

    // Validate function name is an identifier
    if (name_val.type != .identifier) {
        std.debug.print("defn macro first argument must be a function name\n", .{});
        return error.InvalidMacroArgs;
    }
    const func_name = name_val.data.atom;

    // Validate args is a vector
    if (args_vec_val.type != .vector) {
        std.debug.print("defn macro second argument must be a vector of arguments\n", .{});
        return error.InvalidMacroArgs;
    }
    const args_vec = args_vec_val.data.vector;

    // Parse argument list to extract names and types
    var arg_names = std.ArrayList(*Value){};
    defer arg_names.deinit(allocator);
    var arg_types = std.ArrayList(*Value){};
    defer arg_types.deinit(allocator);

    for (args_vec.slice()) |arg| {
        // Each arg should be (: name type)
        if (arg.type != .has_type) {
            std.debug.print("defn macro arguments must be (: name type)\n", .{});
            return error.InvalidMacroArgs;
        }

        // has_type has .value and .type_expr fields
        const param_name = arg.data.has_type.value;
        const param_type = arg.data.has_type.type_expr;

        // param_name should be a value_id (e.g., %a, %b)
        if (param_name.type != .value_id) {
            std.debug.print("defn macro parameter name must be a value_id (e.g., %a)\n", .{});
            return error.InvalidMacroArgs;
        }

        // Use the value_id directly
        try arg_names.append(allocator, param_name);
        try arg_types.append(allocator, param_type);
    }

    // Collect body expressions (remaining arguments)
    var body_exprs = std.ArrayList(*Value){};
    defer body_exprs.deinit(allocator);
    while (iter.next()) |expr| {
        try body_exprs.append(allocator, expr);
    }

    // Build function type: direct .function_type value with inputs and results
    var inputs_vec = PersistentVector(*Value).init(allocator, null);
    for (arg_types.items) |arg_type| {
        inputs_vec = try inputs_vec.push(arg_type);
    }

    var results_vec = PersistentVector(*Value).init(allocator, null);
    results_vec = try results_vec.push(return_type_val);

    const func_type = try allocator.create(Value);
    func_type.* = Value{
        .type = .function_type,
        .data = .{ .function_type = .{
            .inputs = inputs_vec,
            .results = results_vec,
        } },
    };

    // Build symbol name: @func_name
    const symbol_name = try std.fmt.allocPrint(allocator, "@{s}", .{func_name});
    const symbol = try allocator.create(Value);
    symbol.* = Value{
        .type = .symbol,
        .data = .{ .atom = symbol_name },
    };

    // Build attributes: { :sym_name @name :function_type (!function ...) }
    var attrs_map = PersistentVector(*Value).init(allocator, null);
    // :sym_name
    attrs_map = try attrs_map.push(try createKeyword(allocator, ":sym_name"));
    attrs_map = try attrs_map.push(symbol);
    // :function_type
    attrs_map = try attrs_map.push(try createKeyword(allocator, ":function_type"));
    attrs_map = try attrs_map.push(func_type);

    const attrs_map_val = try allocator.create(Value);
    attrs_map_val.* = Value{
        .type = .map,
        .data = .{ .map = attrs_map },
    };

    var attrs_vec = PersistentVector(*Value).init(allocator, null);
    attrs_vec = try attrs_vec.push(try createIdentifier(allocator, "attributes"));
    attrs_vec = try attrs_vec.push(attrs_map_val);
    const attributes = try createList(allocator, attrs_vec);

    // Build block arguments: [(: %arg1 type1) (: %arg2 type2) ...]
    var block_args_vec = PersistentVector(*Value).init(allocator, null);
    for (arg_names.items, arg_types.items) |name, type_val| {
        const has_type_val = try createHasType(allocator, name, type_val);
        block_args_vec = try block_args_vec.push(has_type_val);
    }
    const block_args = try createVector(allocator, block_args_vec);

    // Build (arguments [...])
    var arguments_vec = PersistentVector(*Value).init(allocator, null);
    arguments_vec = try arguments_vec.push(try createIdentifier(allocator, "arguments"));
    arguments_vec = try arguments_vec.push(block_args);
    const arguments = try createList(allocator, arguments_vec);

    // Build block: (block [^entry] (arguments ...) body...)
    var block_vec = PersistentVector(*Value).init(allocator, null);
    block_vec = try block_vec.push(try createIdentifier(allocator, "block"));
    
    // Add block label [^entry]
    var label_vec = PersistentVector(*Value).init(allocator, null);
    const entry_label = try allocator.create(Value);
    entry_label.* = Value{
        .type = .block_id,
        .data = .{ .atom = "^entry" },
    };
    label_vec = try label_vec.push(entry_label);
    block_vec = try block_vec.push(try createVector(allocator, label_vec));
    
    // Add arguments
    block_vec = try block_vec.push(arguments);
    
    // Add body expressions
    for (body_exprs.items) |body_expr| {
        block_vec = try block_vec.push(body_expr);
    }
    const block = try createList(allocator, block_vec);

    // Build region: (region block)
    var region_vec = PersistentVector(*Value).init(allocator, null);
    region_vec = try region_vec.push(try createIdentifier(allocator, "region"));
    region_vec = try region_vec.push(block);
    const region = try createList(allocator, region_vec);

    // Build regions: (regions region)
    var regions_vec = PersistentVector(*Value).init(allocator, null);
    regions_vec = try regions_vec.push(try createIdentifier(allocator, "regions"));
    regions_vec = try regions_vec.push(region);
    const regions = try createList(allocator, regions_vec);

    // Build complete operation
    var op_vec = PersistentVector(*Value).init(allocator, null);
    op_vec = try op_vec.push(try createIdentifier(allocator, "operation"));
    
    // (name func.func)
    var name_vec = PersistentVector(*Value).init(allocator, null);
    name_vec = try name_vec.push(try createIdentifier(allocator, "name"));
    name_vec = try name_vec.push(try createIdentifier(allocator, "func.func"));
    op_vec = try op_vec.push(try createList(allocator, name_vec));
    
    // Add attributes and regions
    op_vec = try op_vec.push(attributes);
    op_vec = try op_vec.push(regions);

    return try createList(allocator, op_vec);
}
