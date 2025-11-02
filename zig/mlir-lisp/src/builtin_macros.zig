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

/// Register all built-in macros with the expander
pub fn registerBuiltinMacros(expander: *MacroExpander) !void {
    // No built-in macros registered yet
    // The if/when/unless macros below don't generate proper operation syntax
    // and need to be redesigned to work with the parser
    _ = expander;
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
