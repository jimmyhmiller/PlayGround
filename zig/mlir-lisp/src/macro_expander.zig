const std = @import("std");
const reader = @import("reader.zig");
const Value = reader.Value;
const ValueType = reader.ValueType;
const vector = @import("collections/vector.zig");
const PersistentVector = vector.PersistentVector;
const linked_list = @import("collections/linked_list.zig");
const PersistentLinkedList = linked_list.PersistentLinkedList;

/// Macro function signature: takes list args (without head), returns expanded form
pub const MacroFn = *const fn (
    allocator: std.mem.Allocator,
    args: *const PersistentLinkedList(*Value),
) anyerror!*Value;

pub const MacroExpander = struct {
    allocator: std.mem.Allocator,
    registry: std.StringHashMap(MacroFn),
    gensym_counter: usize,
    max_iterations: usize,

    pub fn init(allocator: std.mem.Allocator) MacroExpander {
        return MacroExpander{
            .allocator = allocator,
            .registry = std.StringHashMap(MacroFn).init(allocator),
            .gensym_counter = 0,
            .max_iterations = 100, // Default max iterations to prevent infinite loops
        };
    }

    pub fn deinit(self: *MacroExpander) void {
        self.registry.deinit();
    }

    /// Register a macro by name
    pub fn registerMacro(self: *MacroExpander, name: []const u8, func: MacroFn) !void {
        try self.registry.put(name, func);
    }

    /// Generate a unique symbol with the given prefix
    pub fn gensym(self: *MacroExpander, prefix: []const u8) ![]const u8 {
        const counter = self.gensym_counter;
        self.gensym_counter += 1;

        // Generate: prefix_G1234 (G for generated)
        return try std.fmt.allocPrint(
            self.allocator,
            "{s}_G{d}",
            .{ prefix, counter },
        );
    }

    /// Create a gensym'd identifier Value
    pub fn makeGensymValue(self: *MacroExpander, prefix: []const u8) !*Value {
        const name = try self.gensym(prefix);
        const value = try self.allocator.create(Value);
        value.* = Value{
            .type = .identifier,
            .data = .{ .atom = name },
        };
        return value;
    }

    /// Expand all macros iteratively until stable
    pub fn expandAll(self: *MacroExpander, value: *Value) !*Value {
        var current = value;
        var iteration: usize = 0;

        while (iteration < self.max_iterations) : (iteration += 1) {
            const expanded = try self.expandOnce(current);

            // Check if anything changed (pointer equality works for persistent data)
            if (expanded == current) {
                return current; // Stable, no more expansions
            }

            current = expanded;
        }

        return error.MacroExpansionLimit; // Hit max iterations
    }

    /// Perform one pass of macro expansion
    pub fn expandOnce(self: *MacroExpander, value: *Value) !*Value {
        return switch (value.type) {
            .list => try self.expandList(value),
            .vector => try self.expandVector(value),
            .map => try self.expandMap(value),
            .attr_expr => try self.expandAttrExpr(value),
            .has_type => try self.expandHasType(value),
            else => value, // Atoms don't expand
        };
    }

    /// Expand a list - check if first element is a macro
    fn expandList(self: *MacroExpander, value: *Value) anyerror!*Value {
        const list = value.data.list; // PersistentVector(*Value)
        if (list.len() == 0) return value;

        const first = list.at(0);
        if (first.type != .identifier and first.type != .symbol) {
            // Not a potential macro call, just recursively expand children
            return try self.expandChildren(value);
        }

        const name = first.data.atom;
        if (self.registry.get(name)) |macro_fn| {
            // Convert tail to linked list
            const args_list = try self.vectorTailToLinkedList(list);

            // Call macro function
            const expanded = try macro_fn(self.allocator, args_list);

            // Recursively expand the result (macros can expand to macros)
            return try self.expandOnce(expanded);
        }

        // Not a macro, expand children
        return try self.expandChildren(value);
    }

    /// Expand a vector - recursively expand all elements
    fn expandVector(self: *MacroExpander, value: *Value) anyerror!*Value {
        const vec = value.data.vector;
        if (vec.len() == 0) return value;

        var new_vec = PersistentVector(*Value).init(self.allocator, null);
        var changed = false;

        for (vec.slice()) |elem| {
            const expanded = try self.expandOnce(elem);
            new_vec = try new_vec.push(expanded);
            if (expanded != elem) {
                changed = true;
            }
        }

        if (!changed) {
            return value; // No changes, return original
        }

        // Create new vector value
        const new_value = try self.allocator.create(Value);
        new_value.* = Value{
            .type = .vector,
            .data = .{ .vector = new_vec },
        };
        return new_value;
    }

    /// Expand a map - recursively expand all values (keys are usually atoms)
    fn expandMap(self: *MacroExpander, value: *Value) anyerror!*Value {
        const map = value.data.map;
        if (map.len() == 0) return value;

        var new_map = PersistentVector(*Value).init(self.allocator, null);
        var changed = false;

        // Map is flat list of k,v pairs
        var i: usize = 0;
        while (i < map.len()) : (i += 2) {
            const key = map.at(i);
            const val = if (i + 1 < map.len()) map.at(i + 1) else break;

            const expanded_key = try self.expandOnce(key);
            const expanded_val = try self.expandOnce(val);

            new_map = try new_map.push(expanded_key);
            new_map = try new_map.push(expanded_val);

            if (expanded_key != key or expanded_val != val) {
                changed = true;
            }
        }

        if (!changed) {
            return value;
        }

        const new_value = try self.allocator.create(Value);
        new_value.* = Value{
            .type = .map,
            .data = .{ .map = new_map },
        };
        return new_value;
    }

    /// Expand attribute expression
    fn expandAttrExpr(self: *MacroExpander, value: *Value) anyerror!*Value {
        const inner = value.data.attr_expr;
        const expanded_inner = try self.expandOnce(inner);

        if (expanded_inner == inner) {
            return value; // No change
        }

        // Create new attr_expr with expanded inner
        const new_value = try self.allocator.create(Value);
        new_value.* = Value{
            .type = .attr_expr,
            .data = .{ .attr_expr = expanded_inner },
        };
        return new_value;
    }

    /// Expand has_type expression
    fn expandHasType(self: *MacroExpander, value: *Value) anyerror!*Value {
        const has_type = value.data.has_type;
        const expanded_value = try self.expandOnce(has_type.value);
        const expanded_type = try self.expandOnce(has_type.type_expr);

        if (expanded_value == has_type.value and expanded_type == has_type.type_expr) {
            return value; // No change
        }

        // Create new has_type with expanded components
        const new_value = try self.allocator.create(Value);
        new_value.* = Value{
            .type = .has_type,
            .data = .{
                .has_type = .{
                    .value = expanded_value,
                    .type_expr = expanded_type,
                },
            },
        };
        return new_value;
    }

    /// Recursively expand all children in a collection
    fn expandChildren(self: *MacroExpander, value: *Value) anyerror!*Value {
        return switch (value.type) {
            .list => {
                const list = value.data.list;
                if (list.len() == 0) return value;

                var new_list = PersistentVector(*Value).init(self.allocator, null);
                var changed = false;

                for (list.slice()) |elem| {
                    const expanded = try self.expandOnce(elem);
                    new_list = try new_list.push(expanded);
                    if (expanded != elem) {
                        changed = true;
                    }
                }

                if (!changed) {
                    return value;
                }

                const new_value = try self.allocator.create(Value);
                new_value.* = Value{
                    .type = .list,
                    .data = .{ .list = new_list },
                };
                return new_value;
            },
            .vector => try self.expandVector(value),
            .map => try self.expandMap(value),
            else => value,
        };
    }

    /// Convert PersistentVector tail to PersistentLinkedList for macro API
    fn vectorTailToLinkedList(
        self: *MacroExpander,
        vec: PersistentVector(*Value),
    ) !*const PersistentLinkedList(*Value) {
        const slice = vec.slice();
        var list = try PersistentLinkedList(*Value).empty(self.allocator);

        // Build list in reverse (since we cons from the front)
        // Skip index 0 (the macro name)
        if (slice.len <= 1) {
            return list; // Empty args
        }

        var i = slice.len;
        while (i > 1) : (i -= 1) {
            list = try PersistentLinkedList(*Value).cons(self.allocator, slice[i - 1], list);
        }

        return list;
    }
};

test "macro expander basic" {
    const allocator = std.testing.allocator;

    var expander = MacroExpander.init(allocator);
    defer expander.deinit();

    // Create a simple value
    const value = try allocator.create(Value);
    value.* = Value{
        .type = .identifier,
        .data = .{ .atom = "test" },
    };
    defer allocator.destroy(value);

    // Expand should return the same value (no macro registered)
    const expanded = try expander.expandAll(value);
    try std.testing.expect(expanded == value);
}

test "gensym generates unique symbols" {
    const allocator = std.testing.allocator;

    var expander = MacroExpander.init(allocator);
    defer expander.deinit();

    const sym1 = try expander.gensym("temp");
    defer allocator.free(sym1);
    const sym2 = try expander.gensym("temp");
    defer allocator.free(sym2);

    // Should be different
    try std.testing.expect(!std.mem.eql(u8, sym1, sym2));

    // Should have format temp_G0, temp_G1
    try std.testing.expectEqualStrings("temp_G0", sym1);
    try std.testing.expectEqualStrings("temp_G1", sym2);
}
