const std = @import("std");
const reader = @import("reader.zig");
const Value = reader.Value;
const ValueType = reader.ValueType;
const vector = @import("collections/vector.zig");
const PersistentVector = vector.PersistentVector;

/// Result of flattening a single operand
const FlattenResult = struct {
    /// Hoisted operations (0 or more) - these should be inserted before the parent operation
    hoisted_operations: []*Value,
    /// The result value ID to use in place of the nested operation
    value_id: *Value,
};

/// Operation flattener - converts nested operations to flat SSA form
pub const OperationFlattener = struct {
    allocator: std.mem.Allocator,
    gensym_counter: usize,

    pub fn init(allocator: std.mem.Allocator) OperationFlattener {
        return .{
            .allocator = allocator,
            .gensym_counter = 0,
        };
    }

    /// Generate a unique SSA value binding
    pub fn gensym(self: *OperationFlattener) ![]const u8 {
        const counter = self.gensym_counter;
        self.gensym_counter += 1;
        return try std.fmt.allocPrint(
            self.allocator,
            "%result_G{d}",
            .{counter},
        );
    }

    /// Create a value ID Value from a string
    fn makeValueId(self: *OperationFlattener, name: []const u8) !*Value {
        const value = try self.allocator.create(Value);
        value.* = Value{
            .type = .value_id,
            .data = .{ .atom = name },
        };
        return value;
    }

    /// Create an identifier Value from a string
    fn makeIdentifier(self: *OperationFlattener, name: []const u8) !*Value {
        const value = try self.allocator.create(Value);
        value.* = Value{
            .type = .identifier,
            .data = .{ .atom = name },
        };
        return value;
    }

    /// Flatten a complete module - entry point
    pub fn flattenModule(self: *OperationFlattener, module_value: *Value) anyerror!*Value {
        // Module should be a list
        if (module_value.type != .list) return module_value;

        const list = module_value.data.list;
        if (list.len() == 0) return module_value;

        // Check if first element is an identifier
        const first = list.at(0);
        if (first.type != .identifier) {
            // This is a list of operations (e.g., multiple top-level defns)
            // Flatten each operation individually
            var new_list = PersistentVector(*Value).init(self.allocator, null);

            for (list.slice()) |child| {
                const flattened = try self.flattenValue(child);
                new_list = try new_list.push(flattened);
            }

            const new_value = try self.allocator.create(Value);
            new_value.* = Value{
                .type = .list,
                .data = .{ .list = new_list },
            };
            return new_value;
        }

        // Check if first element is "module"
        if (std.mem.eql(u8, first.data.atom, "module")) {
            // Flatten all children recursively
            var new_list = PersistentVector(*Value).init(self.allocator, null);
            new_list = try new_list.push(first); // Keep "module" identifier

            for (list.slice()[1..]) |child| {
                const flattened = try self.flattenValue(child);
                new_list = try new_list.push(flattened);
            }

            const new_value = try self.allocator.create(Value);
            new_value.* = Value{
                .type = .list,
                .data = .{ .list = new_list },
            };
            return new_value;
        }

        // Not a module wrapper, flatten the value directly (could be a block or operation)
        return try self.flattenValue(module_value);
    }

    /// Flatten any Value recursively, looking for blocks to process
    fn flattenValue(self: *OperationFlattener, value: *Value) anyerror!*Value {
        if (value.type != .list) return value;

        const list = value.data.list;
        if (list.len() == 0) return value;

        const first = list.at(0);
        if (first.type != .identifier) return value;

        const name = first.data.atom;

        // If this is a block, flatten its operations
        if (std.mem.eql(u8, name, "block")) {
            return try self.flattenBlock(value);
        }

        // Otherwise, recursively flatten children
        var new_list = PersistentVector(*Value).init(self.allocator, null);
        for (list.slice()) |child| {
            const flattened = try self.flattenValue(child);
            new_list = try new_list.push(flattened);
        }

        const new_value = try self.allocator.create(Value);
        new_value.* = Value{
            .type = .list,
            .data = .{ .list = new_list },
        };
        return new_value;
    }

    /// Flatten a block - this is where the main flattening logic happens
    fn flattenBlock(self: *OperationFlattener, block_value: *Value) anyerror!*Value {
        const list = block_value.data.list;
        var new_list = PersistentVector(*Value).init(self.allocator, null);

        // Collect all operations in the block
        var operations = std.ArrayList(*Value){};
        defer operations.deinit(self.allocator);

        var i: usize = 0;
        while (i < list.len()) : (i += 1) {
            const child = list.at(i);

            // Check if this is an operation
            if (try self.isOperation(child)) {
                try operations.append(self.allocator, child);
            } else {
                // Not an operation (e.g., "block", "arguments" section), keep as is for now
                new_list = try new_list.push(child);
            }
        }

        // Flatten all operations and collect the results
        const flattened_ops = try self.flattenOperations(operations.items);
        defer self.allocator.free(flattened_ops);

        // Add flattened operations to the new list
        for (flattened_ops) |op| {
            new_list = try new_list.push(op);
        }

        const new_value = try self.allocator.create(Value);
        new_value.* = Value{
            .type = .list,
            .data = .{ .list = new_list },
        };
        return new_value;
    }

    /// Check if a Value is an operation (starts with "operation", "declare", or terse op like "func.return")
    fn isOperation(self: *OperationFlattener, value: *Value) !bool {
        _ = self;
        if (value.type != .list) return false;
        const list = value.data.list;
        if (list.len() == 0) return false;
        const first = list.at(0);
        if (first.type != .identifier) return false;

        const name = first.data.atom;

        // Check for verbose operation syntax
        if (std.mem.eql(u8, name, "operation")) return true;

        // Check for declare syntax
        if (std.mem.eql(u8, name, "declare")) return true;

        // Check for terse operation syntax (contains a dot, like "func.return", "arith.constant")
        for (name) |c| {
            if (c == '.') return true;
        }

        return false;
    }

    /// Flatten a list of operations (main algorithm)
    fn flattenOperations(self: *OperationFlattener, operations: []*Value) anyerror![]*Value {
        var result = std.ArrayList(*Value){};

        for (operations) |op| {
            // Flatten this operation and collect hoisted ops
            const flattened = try self.flattenOperation(op);
            defer self.allocator.free(flattened.hoisted_operations);

            // Add hoisted operations first
            for (flattened.hoisted_operations) |hoisted| {
                try result.append(self.allocator, hoisted);
            }

            // Add the main operation
            try result.append(self.allocator, flattened.operation);
        }

        return try result.toOwnedSlice(self.allocator);
    }

    /// Result of flattening a single operation
    const OperationFlattenResult = struct {
        hoisted_operations: []*Value,
        operation: *Value,
    };

    /// Flatten a single operation - handles nested operations in operands
    fn flattenOperation(self: *OperationFlattener, op_value: *Value) anyerror!OperationFlattenResult {
        const list = op_value.data.list;

        // Check if this is a declare form: (declare NAME VALUE)
        // Declare should NOT be flattened - the parser handles it
        if (list.len() > 0) {
            const first = list.at(0);
            if (first.type == .identifier and std.mem.eql(u8, first.data.atom, "declare")) {
                // Return declare as-is, don't flatten
                return OperationFlattenResult{
                    .hoisted_operations = &[_]*Value{},
                    .operation = op_value,
                };
            }
        }

        // Check if this is a terse operation (first element contains a dot)
        if (list.len() > 0) {
            const first = list.at(0);
            if (first.type == .identifier) {
                const name = first.data.atom;
                // Check for dot in name (terse operation like "arith.addi")
                for (name) |c| {
                    if (c == '.') {
                        // This is a terse operation, convert to verbose first
                        const verbose_op = try self.terseToVerbose(op_value);
                        // Now flatten the verbose operation
                        return try self.flattenOperation(verbose_op);
                    }
                }
            }
        }

        var new_list = PersistentVector(*Value).init(self.allocator, null);
        var all_hoisted = std.ArrayList(*Value){};

        for (list.slice()) |section| {
            if (section.type == .list and section.data.list.len() > 0) {
                const first = section.data.list.at(0);
                if (first.type == .identifier) {
                    const name = first.data.atom;

                    if (std.mem.eql(u8, name, "operands")) {
                        // Flatten operands section
                        const result = try self.flattenOperandsSection(section);
                        defer self.allocator.free(result.hoisted_operations);

                        // Collect hoisted operations
                        for (result.hoisted_operations) |hoisted| {
                            try all_hoisted.append(self.allocator, hoisted);
                        }

                        // Add flattened operands section
                        new_list = try new_list.push(result.section);
                        continue;
                    } else if (std.mem.eql(u8, name, "regions")) {
                        // Recursively flatten regions
                        const flattened = try self.flattenValue(section);
                        new_list = try new_list.push(flattened);
                        continue;
                    }
                }
            }

            // Keep section as is
            new_list = try new_list.push(section);
        }

        const new_op = try self.allocator.create(Value);
        new_op.* = Value{
            .type = .list,
            .data = .{ .list = new_list },
        };

        return OperationFlattenResult{
            .hoisted_operations = try all_hoisted.toOwnedSlice(self.allocator),
            .operation = new_op,
        };
    }

    /// Result of flattening an operands section
    const OperandsSectionResult = struct {
        hoisted_operations: []*Value,
        section: *Value,
    };

    /// Flatten the operands section - this is where nested operations are extracted
    fn flattenOperandsSection(self: *OperationFlattener, operands_value: *Value) anyerror!OperandsSectionResult {
        const list = operands_value.data.list;
        var new_list = PersistentVector(*Value).init(self.allocator, null);
        var all_hoisted = std.ArrayList(*Value){};

        // First element is "operands"
        new_list = try new_list.push(list.at(0));

        // Process each operand
        for (list.slice()[1..]) |operand| {
            const result = try self.flattenOperand(operand);
            defer self.allocator.free(result.hoisted_operations);

            // Collect hoisted operations
            for (result.hoisted_operations) |hoisted| {
                try all_hoisted.append(self.allocator, hoisted);
            }

            // Use the resulting value ID
            new_list = try new_list.push(result.value_id);
        }

        const new_section = try self.allocator.create(Value);
        new_section.* = Value{
            .type = .list,
            .data = .{ .list = new_list },
        };

        return OperandsSectionResult{
            .hoisted_operations = try all_hoisted.toOwnedSlice(self.allocator),
            .section = new_section,
        };
    }

    /// Flatten a single operand - returns the value ID to use and any hoisted operations
    fn flattenOperand(self: *OperationFlattener, operand: *Value) anyerror!FlattenResult {
        // Check if this is a type annotation (: expr type)
        if (operand.type == .has_type) {
            // This is a type annotation - flatten the expression inside
            const expr = operand.data.has_type.value;
            const type_expr = operand.data.has_type.type_expr;

            // Recursively flatten the expression
            const result = try self.flattenOperand(expr);

            // If the expression had no hoisted operations, pass through unchanged
            if (result.hoisted_operations.len == 0) {
                return FlattenResult{
                    .hoisted_operations = &[_]*Value{},
                    .value_id = operand,
                };
            }

            // Expression was flattened - we need to add the type to the hoisted operation
            // The last hoisted operation is the one that produces the result we care about
            if (result.hoisted_operations.len > 0) {
                const last_op = result.hoisted_operations[result.hoisted_operations.len - 1];
                // Add result-types section to the last hoisted operation
                const op_with_type = try self.addResultTypes(last_op, type_expr);
                // Replace the last operation with the typed version
                result.hoisted_operations[result.hoisted_operations.len - 1] = op_with_type;
            }

            return result;
        }

        // If it's not a list, it's already a value ID or other atom - pass through
        if (operand.type != .list) {
            return FlattenResult{
                .hoisted_operations = &[_]*Value{},
                .value_id = operand,
            };
        }

        const list = operand.data.list;
        if (list.len() == 0) {
            return FlattenResult{
                .hoisted_operations = &[_]*Value{},
                .value_id = operand,
            };
        }

        // Check if this is a nested operation
        const first = list.at(0);
        if (first.type == .identifier and std.mem.eql(u8, first.data.atom, "operation")) {
            // This is a nested operation - flatten it recursively
            return try self.flattenNestedOperation(operand);
        }

        // Check if this is a terse operation (contains a dot in the name)
        if (first.type == .identifier) {
            const name = first.data.atom;
            for (name) |c| {
                if (c == '.') {
                    // This is a terse operation - treat it like a nested operation
                    return try self.flattenTerseOperation(operand);
                }
            }
        }

        // Not a nested operation, pass through
        return FlattenResult{
            .hoisted_operations = &[_]*Value{},
            .value_id = operand,
        };
    }

    /// Flatten a nested operation - extracts/generates result binding and returns hoisted ops
    fn flattenNestedOperation(self: *OperationFlattener, op_value: *Value) anyerror!FlattenResult {
        // First, recursively flatten this operation's operands
        const op_result = try self.flattenOperation(op_value);
        defer self.allocator.free(op_result.hoisted_operations);

        // Collect all hoisted operations (including from nested operands)
        var all_hoisted = std.ArrayList(*Value){};
        for (op_result.hoisted_operations) |hoisted| {
            try all_hoisted.append(self.allocator, hoisted);
        }

        // Now process result bindings for this operation
        const list = op_result.operation.data.list;
        var result_value_id: ?*Value = null;

        // Find or generate result-bindings
        for (list.slice()) |section| {
            if (section.type == .list and section.data.list.len() > 0) {
                const first = section.data.list.at(0);
                if (first.type == .identifier and std.mem.eql(u8, first.data.atom, "result-bindings")) {
                    // Found result-bindings - extract first binding
                    const bindings_list = section.data.list;
                    if (bindings_list.len() > 1) {
                        const bindings_container = bindings_list.at(1);
                        if (bindings_container.type == .vector) {
                            const bindings_vec = bindings_container.data.vector;
                            if (bindings_vec.len() > 0) {
                                result_value_id = bindings_vec.at(0);
                            }
                        }
                    }
                    break;
                }
            }
        }

        // If no result bindings found, need to generate and add them
        if (result_value_id == null) {
            const generated_name = try self.gensym();
            result_value_id = try self.makeValueId(generated_name);

            // Add result-bindings section to the operation
            const op_with_bindings = try self.addResultBindings(op_result.operation, result_value_id.?);

            // Add this operation to hoisted operations
            try all_hoisted.append(self.allocator, op_with_bindings);
        } else {
            // Already has bindings, add as is
            try all_hoisted.append(self.allocator, op_result.operation);
        }

        return FlattenResult{
            .hoisted_operations = try all_hoisted.toOwnedSlice(self.allocator),
            .value_id = result_value_id.?,
        };
    }

    /// Convert a terse operation to verbose format
    /// (arith.addi {:attr val} %a %b) => (operation (name arith.addi) (attributes ...) (operands %a %b))
    fn terseToVerbose(self: *OperationFlattener, terse_op: *Value) anyerror!*Value {
        const list = terse_op.data.list;
        if (list.len() == 0) {
            return terse_op;
        }

        const first = list.at(0);
        if (first.type != .identifier) {
            return terse_op;
        }

        const op_name = first.data.atom;

        // Convert terse operation to verbose format
        var verbose_list = PersistentVector(*Value).init(self.allocator, null);

        // Add "operation" identifier
        verbose_list = try verbose_list.push(try self.makeIdentifier("operation"));

        // Add name section: (name op.name)
        var name_list = PersistentVector(*Value).init(self.allocator, null);
        name_list = try name_list.push(try self.makeIdentifier("name"));
        name_list = try name_list.push(try self.makeIdentifier(op_name));
        const name_section = try self.allocator.create(Value);
        name_section.* = Value{
            .type = .list,
            .data = .{ .list = name_list },
        };
        verbose_list = try verbose_list.push(name_section);

        // Add operands section: (operands ...)
        var operands_list = PersistentVector(*Value).init(self.allocator, null);
        operands_list = try operands_list.push(try self.makeIdentifier("operands"));

        // Process operands (skip attributes if present)
        var start_idx: usize = 1;
        if (list.len() > 1) {
            const second = list.at(1);
            if (second.type == .map) {
                // Has attributes, add them
                var attrs_list = PersistentVector(*Value).init(self.allocator, null);
                attrs_list = try attrs_list.push(try self.makeIdentifier("attributes"));
                attrs_list = try attrs_list.push(second);
                const attrs_section = try self.allocator.create(Value);
                attrs_section.* = Value{
                    .type = .list,
                    .data = .{ .list = attrs_list },
                };
                verbose_list = try verbose_list.push(attrs_section);
                start_idx = 2;
            }
        }

        // Add all operands
        for (list.slice()[start_idx..]) |operand| {
            operands_list = try operands_list.push(operand);
        }

        const operands_section = try self.allocator.create(Value);
        operands_section.* = Value{
            .type = .list,
            .data = .{ .list = operands_list },
        };
        verbose_list = try verbose_list.push(operands_section);

        // Create verbose operation value
        const verbose_op = try self.allocator.create(Value);
        verbose_op.* = Value{
            .type = .list,
            .data = .{ .list = verbose_list },
        };

        return verbose_op;
    }

    /// Flatten a terse operation (e.g., (arith.addi %a %b))
    /// Converts it to verbose format and then flattens it
    fn flattenTerseOperation(self: *OperationFlattener, terse_op: *Value) anyerror!FlattenResult {
        const verbose_op = try self.terseToVerbose(terse_op);
        // Now flatten the verbose operation
        return try self.flattenNestedOperation(verbose_op);
    }

    /// Add result-types section to an operation
    fn addResultTypes(self: *OperationFlattener, op_value: *Value, type_value: *Value) !*Value {
        const list = op_value.data.list;
        var new_list = PersistentVector(*Value).init(self.allocator, null);

        // Add "operation" identifier first
        new_list = try new_list.push(list.at(0));

        // Create result-types section: (result-types TYPE)
        var types_list = PersistentVector(*Value).init(self.allocator, null);
        types_list = try types_list.push(try self.makeIdentifier("result-types"));
        types_list = try types_list.push(type_value);

        const types_section = try self.allocator.create(Value);
        types_section.* = Value{
            .type = .list,
            .data = .{ .list = types_list },
        };

        // Add result-types section
        new_list = try new_list.push(types_section);

        // Add rest of the sections
        for (list.slice()[1..]) |section| {
            new_list = try new_list.push(section);
        }

        const new_op = try self.allocator.create(Value);
        new_op.* = Value{
            .type = .list,
            .data = .{ .list = new_list },
        };
        return new_op;
    }

    /// Add result-bindings section to an operation
    fn addResultBindings(self: *OperationFlattener, op_value: *Value, binding: *Value) !*Value {
        const list = op_value.data.list;
        var new_list = PersistentVector(*Value).init(self.allocator, null);

        // Add "operation" identifier first
        new_list = try new_list.push(list.at(0));

        // Create result-bindings section
        var bindings_list = PersistentVector(*Value).init(self.allocator, null);
        bindings_list = try bindings_list.push(try self.makeIdentifier("result-bindings"));

        // Create vector with the binding
        var bindings_vec = PersistentVector(*Value).init(self.allocator, null);
        bindings_vec = try bindings_vec.push(binding);

        const vec_value = try self.allocator.create(Value);
        vec_value.* = Value{
            .type = .vector,
            .data = .{ .vector = bindings_vec },
        };

        bindings_list = try bindings_list.push(vec_value);

        const bindings_section = try self.allocator.create(Value);
        bindings_section.* = Value{
            .type = .list,
            .data = .{ .list = bindings_list },
        };

        // Add result-bindings section
        new_list = try new_list.push(bindings_section);

        // Add rest of the sections
        for (list.slice()[1..]) |section| {
            new_list = try new_list.push(section);
        }

        const new_op = try self.allocator.create(Value);
        new_op.* = Value{
            .type = .list,
            .data = .{ .list = new_list },
        };
        return new_op;
    }
};

test "operation flattener basic init" {
    const allocator = std.testing.allocator;

    var flattener = OperationFlattener.init(allocator);

    const sym1 = try flattener.gensym();
    defer allocator.free(sym1);
    const sym2 = try flattener.gensym();
    defer allocator.free(sym2);

    try std.testing.expectEqualStrings("%result_G0", sym1);
    try std.testing.expectEqualStrings("%result_G1", sym2);
}
