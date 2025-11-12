const std = @import("std");
const reader = @import("reader.zig");
const tokenizer = @import("tokenizer.zig");
const vector = @import("collections/vector.zig");
const Value = reader.Value;
const Reader = reader.Reader;
const Tokenizer = tokenizer.Tokenizer;
const PersistentVector = vector.PersistentVector;

/// Inlines all (declare var-name expr) forms by replacing variable references with their expressions
pub const Inliner = struct {
    allocator: std.mem.Allocator,
    bindings: std.StringHashMap(*Value),

    pub fn init(allocator: std.mem.Allocator) Inliner {
        return .{
            .allocator = allocator,
            .bindings = std.StringHashMap(*Value).init(allocator),
        };
    }

    pub fn deinit(self: *Inliner) void {
        self.bindings.deinit();
    }

    /// Inline all declares in the given value and return a new value with substitutions
    /// Returns null if this value should be skipped (e.g., it's a declare form)
    pub fn inlineValue(self: *Inliner, value: *Value) !?*Value {
        switch (value.type) {
            .list => {
                const list = value.data.list;
                const list_slice = list.slice();

                // Check if this is a declare form: (declare var-name expr)
                if (list.len() >= 3) {
                    if (list_slice[0].type == .identifier and std.mem.eql(u8, list_slice[0].data.atom, "declare")) {
                        // This is a declare - record the binding and return null to signal "skip this"
                        if (list_slice[1].type == .identifier) {
                            const var_name = list_slice[1].data.atom;
                            const expr_opt = try self.inlineValue(list_slice[2]); // Recursively inline the expression
                            const expr = expr_opt orelse return error.InvalidDeclare; // Declare's value can't be null
                            try self.bindings.put(var_name, expr);

                            // Return null to signal this should be skipped
                            return null;
                        }
                    }
                }

                // Not a declare - recursively inline all elements
                var new_list = std.ArrayList(*Value){};
                defer new_list.deinit(self.allocator);

                for (list_slice) |elem| {
                    const inlined = try self.inlineValue(elem);
                    // Skip null values (which come from declares)
                    if (inlined) |val| {
                        try new_list.append(self.allocator, val);
                    }
                }

                const result = try self.allocator.create(Value);
                const owned_slice = try new_list.toOwnedSlice(self.allocator);
                const vec = PersistentVector(*Value).init(self.allocator, owned_slice);
                result.* = Value{
                    .type = .list,
                    .data = .{ .list = vec },
                };
                return result;
            },
            .vector => {
                const vec = value.data.vector;
                var new_vec = std.ArrayList(*Value){};
                defer new_vec.deinit(self.allocator);

                for (vec.slice()) |elem| {
                    const inlined = try self.inlineValue(elem);
                    if (inlined) |val| {
                        try new_vec.append(self.allocator, val);
                    }
                }

                const result = try self.allocator.create(Value);
                const owned_slice = try new_vec.toOwnedSlice(self.allocator);
                const new_pvec = PersistentVector(*Value).init(self.allocator, owned_slice);
                result.* = Value{
                    .type = .vector,
                    .data = .{ .vector = new_pvec },
                };
                return result;
            },
            .identifier, .value_id => {
                // Check if this is a variable reference (value_id starts with %)
                const name = value.data.atom;
                if (name.len > 1 and name[0] == '%') {
                    const var_name = name[1..]; // Remove the %
                    if (self.bindings.get(var_name)) |expr| {
                        // Return a copy of the expression
                        return try self.copyValue(expr);
                    }
                }

                // Not a variable reference or not bound - return as-is
                return try self.copyValue(value);
            },
            .has_type => {
                // Process both the value and the type expression
                const has_type = value.data.has_type;
                const inlined_value_opt = try self.inlineValue(has_type.value);
                const inlined_value = inlined_value_opt orelse return error.InvalidHasType;
                const inlined_type_opt = try self.inlineValue(has_type.type_expr);
                const inlined_type = inlined_type_opt orelse return error.InvalidHasType;

                const result = try self.allocator.create(Value);
                result.* = Value{
                    .type = .has_type,
                    .data = .{ .has_type = .{
                        .value = inlined_value,
                        .type_expr = inlined_type,
                    }},
                };
                return result;
            },
            .attr_expr => {
                // Process the inner expression
                const attr = value.data.attr_expr;
                const inlined_opt = try self.inlineValue(attr);
                const inlined = inlined_opt orelse return error.InvalidAttr;

                const result = try self.allocator.create(Value);
                result.* = Value{
                    .type = .attr_expr,
                    .data = .{ .attr_expr = inlined },
                };
                return result;
            },
            .function_type => {
                // Process inputs and results
                const func_type = value.data.function_type;

                var new_inputs = std.ArrayList(*Value){};
                defer new_inputs.deinit(self.allocator);
                for (func_type.inputs.slice()) |elem| {
                    const inlined = try self.inlineValue(elem);
                    if (inlined) |val| {
                        try new_inputs.append(self.allocator, val);
                    }
                }
                const inputs_slice = try new_inputs.toOwnedSlice(self.allocator);
                const inputs_vec = PersistentVector(*Value).init(self.allocator, inputs_slice);

                var new_results = std.ArrayList(*Value){};
                defer new_results.deinit(self.allocator);
                for (func_type.results.slice()) |elem| {
                    const inlined = try self.inlineValue(elem);
                    if (inlined) |val| {
                        try new_results.append(self.allocator, val);
                    }
                }
                const results_slice = try new_results.toOwnedSlice(self.allocator);
                const results_vec = PersistentVector(*Value).init(self.allocator, results_slice);

                const result = try self.allocator.create(Value);
                result.* = Value{
                    .type = .function_type,
                    .data = .{ .function_type = .{
                        .inputs = inputs_vec,
                        .results = results_vec,
                    }},
                };
                return result;
            },
            else => {
                // For atoms, strings, numbers, etc. - just copy
                return try self.copyValue(value);
            },
        }
    }

    /// Deep copy a value
    fn copyValue(self: *Inliner, value: *Value) !*Value {
        const result = try self.allocator.create(Value);
        switch (value.type) {
            .list => {
                const list = value.data.list;
                var new_list = std.ArrayList(*Value){};
                defer new_list.deinit(self.allocator);

                for (list.slice()) |elem| {
                    const copied = try self.copyValue(elem);
                    try new_list.append(self.allocator, copied);
                }

                const owned_slice = try new_list.toOwnedSlice(self.allocator);
                const vec = PersistentVector(*Value).init(self.allocator, owned_slice);
                result.* = Value{
                    .type = .list,
                    .data = .{ .list = vec },
                };
            },
            .vector => {
                const vec = value.data.vector;
                var new_vec = std.ArrayList(*Value){};
                defer new_vec.deinit(self.allocator);

                for (vec.slice()) |elem| {
                    const copied = try self.copyValue(elem);
                    try new_vec.append(self.allocator, copied);
                }

                const owned_slice = try new_vec.toOwnedSlice(self.allocator);
                const new_pvec = PersistentVector(*Value).init(self.allocator, owned_slice);
                result.* = Value{
                    .type = .vector,
                    .data = .{ .vector = new_pvec },
                };
            },
            .map => {
                const map = value.data.map;
                var new_map = std.ArrayList(*Value){};
                defer new_map.deinit(self.allocator);

                for (map.slice()) |elem| {
                    const copied = try self.copyValue(elem);
                    try new_map.append(self.allocator, copied);
                }

                const owned_slice = try new_map.toOwnedSlice(self.allocator);
                const new_pmap = PersistentVector(*Value).init(self.allocator, owned_slice);
                result.* = Value{
                    .type = .map,
                    .data = .{ .map = new_pmap },
                };
            },
            .function_type => {
                const func_type = value.data.function_type;

                var new_inputs = std.ArrayList(*Value){};
                defer new_inputs.deinit(self.allocator);
                for (func_type.inputs.slice()) |elem| {
                    const copied = try self.copyValue(elem);
                    try new_inputs.append(self.allocator, copied);
                }
                const inputs_slice = try new_inputs.toOwnedSlice(self.allocator);
                const inputs_vec = PersistentVector(*Value).init(self.allocator, inputs_slice);

                var new_results = std.ArrayList(*Value){};
                defer new_results.deinit(self.allocator);
                for (func_type.results.slice()) |elem| {
                    const copied = try self.copyValue(elem);
                    try new_results.append(self.allocator, copied);
                }
                const results_slice = try new_results.toOwnedSlice(self.allocator);
                const results_vec = PersistentVector(*Value).init(self.allocator, results_slice);

                result.* = Value{
                    .type = .function_type,
                    .data = .{ .function_type = .{
                        .inputs = inputs_vec,
                        .results = results_vec,
                    }},
                };
            },
            .attr_expr => {
                const attr = value.data.attr_expr;
                const copied_attr = try self.copyValue(attr);
                result.* = Value{
                    .type = .attr_expr,
                    .data = .{ .attr_expr = copied_attr },
                };
            },
            .has_type => {
                const has_type = value.data.has_type;
                const copied_value = try self.copyValue(has_type.value);
                const copied_type = try self.copyValue(has_type.type_expr);
                result.* = Value{
                    .type = .has_type,
                    .data = .{ .has_type = .{
                        .value = copied_value,
                        .type_expr = copied_type,
                    }},
                };
            },
            .type => {
                const type_str = value.data.type;
                const copied_type = try self.allocator.dupe(u8, type_str);
                result.* = Value{
                    .type = .type,
                    .data = .{ .type = copied_type },
                };
            },
            // For atoms and simple types (identifier, number, string, etc.)
            else => {
                const atom = value.data.atom;
                const copied_atom = try self.allocator.dupe(u8, atom);
                result.* = Value{
                    .type = value.type,
                    .data = .{ .atom = copied_atom },
                };
            },
        }
        return result;
    }
};

pub fn main() !void {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    const gpa_allocator = gpa.allocator();

    // Use arena allocator to automatically free all memory at once
    var arena = std.heap.ArenaAllocator.init(gpa_allocator);
    defer arena.deinit();
    const allocator = arena.allocator();

    const args = try std.process.argsAlloc(gpa_allocator);
    defer std.process.argsFree(gpa_allocator, args);

    if (args.len < 2) {
        std.debug.print("Usage: {s} <input-file> [output-file]\n", .{args[0]});
        std.debug.print("  If output-file is not specified, prints to stdout\n", .{});
        return;
    }

    const input_file = args[1];
    const output_file = if (args.len >= 3) args[2] else null;

    // Read input file
    const source = try std.fs.cwd().readFileAlloc(allocator, input_file, 1024 * 1024);
    defer allocator.free(source);

    // Parse the source
    var tok = Tokenizer.init(allocator, source);
    var r = try Reader.init(allocator, &tok);
    const value = try r.read();

    // Inline all declares
    var inliner = Inliner.init(allocator);
    defer inliner.deinit();

    const inlined_opt = try inliner.inlineValue(value);
    const inlined = inlined_opt orelse {
        std.debug.print("Error: top-level value was skipped\n", .{});
        return error.InvalidInput;
    };

    // Print the result
    var buffer = std.ArrayList(u8){};
    defer buffer.deinit(allocator);

    try inlined.print(buffer.writer(allocator));

    if (output_file) |out_path| {
        try std.fs.cwd().writeFile(.{ .sub_path = out_path, .data = buffer.items });
    } else {
        std.debug.print("{s}", .{buffer.items});
    }
}
