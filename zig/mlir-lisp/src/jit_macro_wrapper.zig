const std = @import("std");
const MacroExpander = @import("macro_expander.zig").MacroExpander;
const MacroFn = @import("macro_expander.zig").MacroFn;
const Value = @import("reader.zig").Value;
const PersistentLinkedList = @import("collections/linked_list.zig").PersistentLinkedList;
const PersistentVector = @import("collections/vector.zig").PersistentVector;

/// Type signature for JIT-compiled macro functions
/// These take a single pointer to a Value (containing the args list)
/// and return a pointer to a Value (the expanded result)
pub const JitMacroFn = *const fn (?*anyopaque) callconv(.c) ?*anyopaque;

/// Context to hold the JIT function pointer
const WrapperContext = struct {
    jit_fn: JitMacroFn,
};

// Global storage for wrapper contexts (needed because we can't capture runtime values in comptime structs)
var wrapper_contexts = std.ArrayList(WrapperContext){};

/// Convert Value to CValueLayout
fn convertValueToCValueLayout(alloc: std.mem.Allocator, value: *const Value) !*@import("reader/c_value_layout.zig").CValueLayout {
    const CValueLayout = @import("reader/c_value_layout.zig").CValueLayout;

    const layout = try alloc.create(CValueLayout);
    errdefer alloc.destroy(layout);

    switch (value.type) {
        .list, .vector, .map => {
            // For collections, convert the PersistentVector to a flat array of CValueLayout pointers
            const vec = switch (value.type) {
                .list => value.data.list,
                .vector => value.data.vector,
                .map => value.data.map,
                else => unreachable,
            };

            // Allocate array for element pointers
            const elem_layouts = try alloc.alloc(*CValueLayout, vec.len());

            // Recursively convert each element
            for (0..vec.len()) |i| {
                const elem_value = vec.at(i);
                elem_layouts[i] = try convertValueToCValueLayout(alloc, elem_value);
            }

            layout.* = CValueLayout{
                .type_tag = @intFromEnum(value.type),
                ._padding = [_]u8{0} ** 7,
                .data_ptr = @ptrCast(elem_layouts.ptr),
                .data_len = elem_layouts.len,
                .data_capacity = elem_layouts.len,
                .data_elem_size = @sizeOf(*CValueLayout),
                .extra_ptr1 = null,
                .extra_ptr2 = null,
            };
        },
        .identifier, .number, .string, .value_id, .block_id, .symbol, .keyword, .type => {
            // For atoms, the data is a string
            const str = value.data.atom;

            layout.* = CValueLayout{
                .type_tag = @intFromEnum(value.type),
                ._padding = [_]u8{0} ** 7,
                .data_ptr = @constCast(@ptrCast(str.ptr)),
                .data_len = str.len,
                .data_capacity = 0,
                .data_elem_size = 0,
                .extra_ptr1 = null,
                .extra_ptr2 = null,
            };
        },
        else => {
            return error.UnsupportedValueType;
        },
    }

    return layout;
}

/// Convert CValueLayout to Value
fn convertCValueLayoutToValue(alloc: std.mem.Allocator, layout: *const @import("reader/c_value_layout.zig").CValueLayout) !*Value {
    const ValueType = @import("reader.zig").ValueType;

    // Validate type_tag before converting
    if (layout.type_tag > @intFromEnum(ValueType.has_type)) {
        return error.InvalidTypeTag;
    }

    const value_type: ValueType = @enumFromInt(layout.type_tag);

    const value = try alloc.create(Value);
    errdefer alloc.destroy(value);

    switch (value_type) {
        .list, .vector, .map => {
            // For collections, convert the flat array to PersistentVector
            // The array contains CValueLayout* pointers, not Value* pointers!
            const CValueLayout = @import("reader/c_value_layout.zig").CValueLayout;
            const elem_layout_ptrs: [*]*CValueLayout = @ptrCast(@alignCast(layout.data_ptr));
            var vec = PersistentVector(*Value).init(alloc, null);

            for (0..layout.data_len) |i| {
                // Recursively convert each element
                const elem_value = try convertCValueLayoutToValue(alloc, elem_layout_ptrs[i]);
                vec = try vec.push(elem_value);
            }

            value.* = Value{
                .type = value_type,
                .data = switch (value_type) {
                    .list => .{ .list = vec },
                    .vector => .{ .vector = vec },
                    .map => .{ .map = vec },
                    else => unreachable,
                },
            };
        },
        .identifier, .number, .string, .value_id, .block_id, .symbol, .keyword, .type => {
            // For atoms, the data_ptr points to the string
            const str_ptr: [*]const u8 = @ptrCast(layout.data_ptr);
            const str = str_ptr[0..layout.data_len];

            value.* = Value{
                .type = value_type,
                .data = .{ .atom = str },
            };
        },
        else => {
            return error.UnsupportedValueType;
        },
    }

    return value;
}

/// Generate a unique wrapper function for each JIT function
fn makeWrapper(comptime index: usize) type {
    return struct {
        fn call(
            alloc: std.mem.Allocator,
            args: *const PersistentLinkedList(*Value),
        ) anyerror!*Value {
            // Retrieve the JIT function from global context
            const jit_fn_ptr = wrapper_contexts.items[index].jit_fn;

            // Convert linked list to CValueLayout
            // The JIT function expects a CValueLayout* pointing to a list
            const CValueLayout = @import("reader/c_value_layout.zig").CValueLayout;

            // Collect all arguments from linked list and convert each to CValueLayout
            var args_layout_list = std.ArrayList(*CValueLayout){};
            defer args_layout_list.deinit(alloc);

            var iter = args.iterator();
            while (iter.next()) |value| {
                const value_layout = try convertValueToCValueLayout(alloc, value);
                try args_layout_list.append(alloc, value_layout);
            }

            const args_layout_slice = try args_layout_list.toOwnedSlice(alloc);
            defer alloc.free(args_layout_slice);

            // Create a CValueLayout struct containing the list of CValueLayout pointers
            const args_layout = try alloc.create(CValueLayout);
            errdefer alloc.destroy(args_layout);

            args_layout.* = CValueLayout{
                .type_tag = 9, // list (fixed: was 10, should be 9)
                ._padding = [_]u8{0} ** 7,
                .data_ptr = @ptrCast(args_layout_slice.ptr),
                .data_len = args_layout_slice.len,
                .data_capacity = args_layout_slice.len,
                .data_elem_size = @sizeOf(*CValueLayout),
                .extra_ptr1 = null,
                .extra_ptr2 = null,
            };

            // Call the JIT-compiled function
            // Pass the CValueLayout* as an opaque pointer
            const result_ptr = jit_fn_ptr(@ptrCast(args_layout)) orelse {
                return error.MacroExpansionFailed;
            };

            // Cast the result back to a CValueLayout pointer, then convert to Value
            const result_layout: *CValueLayout = @ptrCast(@alignCast(result_ptr));

            // Convert CValueLayout to Value
            const result = try convertCValueLayoutToValue(alloc, result_layout);

            return result;
        }
    };
}

/// Wraps a JIT-compiled macro function to match the MacroFn signature
pub fn wrapJitMacro(
    _: std.mem.Allocator,
    jit_fn: JitMacroFn,
) MacroFn {
    // Store the jit_fn in our global context list
    const ctx = WrapperContext{ .jit_fn = jit_fn };
    wrapper_contexts.append(std.heap.page_allocator, ctx) catch @panic("Failed to store wrapper context");
    const ctx_index = wrapper_contexts.items.len - 1;

    // Generate and return a unique wrapper based on the index
    // We use a switch to generate a finite number of wrappers at comptime
    return switch (ctx_index) {
        0 => &makeWrapper(0).call,
        1 => &makeWrapper(1).call,
        2 => &makeWrapper(2).call,
        3 => &makeWrapper(3).call,
        4 => &makeWrapper(4).call,
        5 => &makeWrapper(5).call,
        6 => &makeWrapper(6).call,
        7 => &makeWrapper(7).call,
        8 => &makeWrapper(8).call,
        9 => &makeWrapper(9).call,
        else => @panic("Too many JIT macros registered (max 10)"),
    };
}

/// Helper to register a JIT-compiled macro
/// This wraps the JIT function and registers it with the macro expander
pub fn registerJitMacro(
    expander: *MacroExpander,
    macro_name: []const u8,
    jit_fn: JitMacroFn,
) !void {
    const wrapper = wrapJitMacro(expander.allocator, jit_fn);
    try expander.registerMacro(macro_name, wrapper);
}
