const std = @import("std");
const macro_expander = @import("macro_expander.zig");
const MacroExpander = macro_expander.MacroExpander;
const MacroFn = macro_expander.MacroFn;
const reader = @import("reader.zig");
const Value = reader.Value;
const vector = @import("collections/vector.zig");
const PersistentVector = vector.PersistentVector;
const linked_list = @import("collections/linked_list.zig");
const PersistentLinkedList = linked_list.PersistentLinkedList;
const c_api = @import("collections/c_api.zig");

// Opaque types for C API
pub const CMacroExpander = opaque {};
pub const CValue = opaque {};
pub const CLinkedListValue = opaque {};
pub const CLinkedListIterator = opaque {};

// Helper to convert opaque allocator pointer to std.mem.Allocator
fn getAllocator(allocator: ?*anyopaque) ?std.mem.Allocator {
    if (allocator == null) return null;
    const alloc_ptr: *std.mem.Allocator = @ptrCast(@alignCast(allocator));
    return alloc_ptr.*;
}

// C-compatible macro function type (opaque pointers)
// Note: Using standard calling convention since .C is not available in this Zig version
pub const CMacroFn = *const fn (
    allocator: ?*anyopaque, // Allocator
    args: ?*anyopaque, // *const PersistentLinkedList(*Value)
) ?*anyopaque; // Returns *Value or null

// C-compatible transformation function type - takes a Value (usually a list) and returns transformed Value
pub const CTransformFn = *const fn (
    allocator: ?*anyopaque, // Allocator
    value: ?*anyopaque, // *Value - typically a list containing the arguments
) ?*anyopaque; // Returns transformed *Value or null

// Global registry for C macro functions
// Maps function pointer address to the actual function
var c_macro_registry = std.AutoHashMap(usize, CMacroFn).init(std.heap.c_allocator);
var registry_mutex = std.Thread.Mutex{};

// ============================================================================
// Generic Wrapper Infrastructure
// ============================================================================

/// Wraps a C transformation function to work as a MacroFn
/// The C function receives a Value list containing only the arguments (no macro name)
/// and returns a transformed Value
pub fn wrapCTransformAsMacro(comptime c_fn: CTransformFn) MacroFn {
    const Wrapper = struct {
        fn call(
            allocator: std.mem.Allocator,
            args: *const PersistentLinkedList(*Value),
        ) anyerror!*Value {
            // 1. Convert Zig allocator to C allocator pointer
            const alloc_ptr = try allocator.create(std.mem.Allocator);
            errdefer allocator.destroy(alloc_ptr);
            alloc_ptr.* = allocator;
            const c_allocator: ?*anyopaque = @ptrCast(alloc_ptr);
            defer allocator.destroy(alloc_ptr);

            // 2. Convert PersistentLinkedList to a Value list
            // Build a vector from the linked list args
            var args_vec = PersistentVector(*Value).init(allocator, null);
            var iter = args.iterator();
            while (iter.next()) |value| {
                args_vec = try args_vec.push(value);
            }

            // Create a Value list from the vector
            const args_value = try allocator.create(Value);
            args_value.* = Value{
                .type = .list,
                .data = .{ .list = args_vec },
            };
            const args_opaque: ?*anyopaque = @ptrCast(args_value);

            // 3. Call the C transformation function
            const result_opaque = c_fn(c_allocator, args_opaque) orelse {
                std.debug.print("ERROR: C transformation function returned null\n", .{});
                std.debug.print("  Input value type: {s}\n", .{@tagName(args_value.type)});

                // Try to print some debug info about the input
                if (args_value.type == .list) {
                    const vec = args_value.data.list;
                    const slice = vec.slice();
                    std.debug.print("  Input is a list with {} elements\n", .{slice.len});
                    for (slice, 0..) |elem, i| {
                        std.debug.print("    [{}] type={s}", .{ i, @tagName(elem.type) });
                        if (elem.type == .identifier or elem.type == .number or
                            elem.type == .string or elem.type == .true_lit or
                            elem.type == .false_lit or elem.type == .keyword or
                            elem.type == .symbol or elem.type == .value_id or
                            elem.type == .block_id) {
                            std.debug.print(" value=\"{s}\"", .{elem.data.atom});
                        }
                        std.debug.print("\n", .{});
                    }
                }

                return error.TransformationFailed;
            };

            // 4. Convert result back to *Value
            const result: *Value = @ptrCast(@alignCast(result_opaque));
            return result;
        }
    };

    return &Wrapper.call;
}

// ============================================================================
// MacroExpander Lifecycle
// ============================================================================

/// Create a new macro expander
pub export fn macro_expander_create(allocator: ?*anyopaque) ?*CMacroExpander {
    const alloc = getAllocator(allocator) orelse return null;
    const expander = alloc.create(MacroExpander) catch return null;
    expander.* = MacroExpander.init(alloc);
    return @ptrCast(expander);
}

/// Destroy a macro expander
pub export fn macro_expander_destroy(expander: ?*anyopaque) void {
    if (expander == null) return;
    const e: *MacroExpander = @ptrCast(@alignCast(expander));
    const alloc = e.allocator;
    e.deinit();
    alloc.destroy(e);
}

/// Set maximum iterations for expansion (default is 100)
pub export fn macro_expander_set_max_iterations(
    expander: ?*anyopaque,
    max_iter: usize,
) void {
    const e: *MacroExpander = @ptrCast(@alignCast(expander orelse return));
    e.max_iterations = max_iter;
}

/// Get current maximum iterations
pub export fn macro_expander_get_max_iterations(expander: ?*anyopaque) usize {
    const e: *MacroExpander = @ptrCast(@alignCast(expander orelse return 0));
    return e.max_iterations;
}

// ============================================================================
// Macro Registration
// ============================================================================

/// Register a C-compatible macro function
/// Note: Pass the function pointer as usize to avoid calling convention issues
pub export fn macro_expander_register(
    expander: ?*anyopaque,
    name: [*:0]const u8,
    func_addr: usize,
) c_int {
    const e: *MacroExpander = @ptrCast(@alignCast(expander orelse return -1));
    const name_slice = std.mem.span(name);

    // Cast the address back to function pointer
    const func: CMacroFn = @ptrFromInt(func_addr);

    // Store C function in global registry
    registry_mutex.lock();
    defer registry_mutex.unlock();
    c_macro_registry.put(func_addr, func) catch return -1;

    // For now, we'll use a simpler approach: store the address in the name
    // This is a limitation that will need to be addressed
    const wrapper_name = std.fmt.allocPrint(e.allocator, "{s}__{x}", .{ name_slice, func_addr }) catch return -1;
    defer e.allocator.free(wrapper_name);

    // Create a custom wrapper - this is a workaround
    // In practice, macros written in the language will need to use a different mechanism
    const stub_wrapper = struct {
        fn call(
            allocator: std.mem.Allocator,
            args: *const PersistentLinkedList(*Value),
        ) anyerror!*Value {
            _ = allocator;
            _ = args;
            return error.CMacroNotSupported;
        }
    }.call;

    e.registerMacro(wrapper_name, &stub_wrapper) catch return -1;
    return 0;
}

// ============================================================================
// Expansion API
// ============================================================================

/// Expand all macros iteratively until stable
pub export fn macro_expander_expand_all(
    expander: ?*anyopaque,
    value: ?*anyopaque,
) ?*anyopaque {
    const e: *MacroExpander = @ptrCast(@alignCast(expander orelse return null));
    const v: *Value = @ptrCast(@alignCast(value orelse return null));

    const expanded = e.expandAll(v) catch return null;
    return @ptrCast(expanded);
}

/// Perform one pass of macro expansion
pub export fn macro_expander_expand_once(
    expander: ?*anyopaque,
    value: ?*anyopaque,
) ?*anyopaque {
    const e: *MacroExpander = @ptrCast(@alignCast(expander orelse return null));
    const v: *Value = @ptrCast(@alignCast(value orelse return null));

    const expanded = e.expandOnce(v) catch return null;
    return @ptrCast(expanded);
}

// ============================================================================
// Gensym API
// ============================================================================

/// Generate a unique symbol with the given prefix
/// Returns a null-terminated C string
pub export fn macro_expander_gensym(
    expander: ?*anyopaque,
    prefix: [*:0]const u8,
) [*:0]const u8 {
    const e: *MacroExpander = @ptrCast(@alignCast(expander orelse return ""));
    const prefix_slice = std.mem.span(prefix);

    const sym = e.gensym(prefix_slice) catch return "";

    // Need to make null-terminated for C
    const c_str = e.allocator.allocSentinel(u8, sym.len, 0) catch return "";
    @memcpy(c_str, sym);
    return c_str;
}

/// Create a gensym'd identifier Value
pub export fn macro_expander_make_gensym_value(
    expander: ?*anyopaque,
    prefix: [*:0]const u8,
) ?*anyopaque {
    const e: *MacroExpander = @ptrCast(@alignCast(expander orelse return null));
    const prefix_slice = std.mem.span(prefix);

    const value = e.makeGensymValue(prefix_slice) catch return null;
    return @ptrCast(value);
}

// ============================================================================
// Macro Argument Access Helpers
// ============================================================================

/// Get the length of the argument list
pub export fn macro_args_len(args: ?*anyopaque) usize {
    const list: *const PersistentLinkedList(*Value) =
        @ptrCast(@alignCast(args orelse return 0));
    return list.len();
}

/// Get the nth argument (0-indexed)
pub export fn macro_args_nth(args: ?*anyopaque, index: usize) ?*anyopaque {
    const list: *const PersistentLinkedList(*Value) =
        @ptrCast(@alignCast(args orelse return null));

    var iter = list.iterator();
    var i: usize = 0;
    while (iter.next()) |value| {
        if (i == index) return @ptrCast(@constCast(value));
        i += 1;
    }
    return null;
}

/// Create an iterator for the argument list
pub export fn macro_args_iterator(args: ?*anyopaque) ?*anyopaque {
    const list: *const PersistentLinkedList(*Value) =
        @ptrCast(@alignCast(args orelse return null));

    const iter = list.allocator.create(
        PersistentLinkedList(*Value).Iterator,
    ) catch return null;
    iter.* = list.iterator();
    return @ptrCast(iter);
}

/// Get the next value from an iterator
pub export fn macro_iterator_next(iter: ?*anyopaque) ?*anyopaque {
    const it: *PersistentLinkedList(*Value).Iterator =
        @ptrCast(@alignCast(iter orelse return null));

    const value = it.next() orelse return null;
    return @ptrCast(@constCast(value));
}

/// Peek at the next value without advancing the iterator
pub export fn macro_iterator_peek(iter: ?*anyopaque) ?*anyopaque {
    const it: *PersistentLinkedList(*Value).Iterator =
        @ptrCast(@alignCast(iter orelse return null));

    const value = it.peek() orelse return null;
    return @ptrCast(@constCast(value));
}

/// Destroy an iterator
pub export fn macro_iterator_destroy(
    allocator: ?*anyopaque,
    iter: ?*anyopaque,
) void {
    const alloc = getAllocator(allocator) orelse return;
    if (iter == null) return;
    const typed: *PersistentLinkedList(*Value).Iterator =
        @ptrCast(@alignCast(iter));
    alloc.destroy(typed);
}

// ============================================================================
// Linked List Construction (for building macro return values)
// ============================================================================

/// Create an empty linked list
pub export fn linked_list_value_create(allocator: ?*anyopaque) ?*anyopaque {
    const alloc = getAllocator(allocator) orelse return null;
    const list = PersistentLinkedList(*Value).empty(alloc) catch return null;
    return @ptrCast(@constCast(list));
}

/// Cons: prepend value to list
pub export fn linked_list_value_cons(
    allocator: ?*anyopaque,
    value: ?*anyopaque,
    list: ?*anyopaque,
) ?*anyopaque {
    const alloc = getAllocator(allocator) orelse return null;
    const v: *Value = @ptrCast(@alignCast(value orelse return null));
    const l: *const PersistentLinkedList(*Value) =
        @ptrCast(@alignCast(list orelse return null));

    const new_list = PersistentLinkedList(*Value).cons(alloc, v, l) catch return null;
    return @ptrCast(@constCast(new_list));
}

/// Push: append value to end of list (less efficient than cons)
pub export fn linked_list_value_push(
    allocator: ?*anyopaque,
    value: ?*anyopaque,
    list: ?*anyopaque,
) ?*anyopaque {
    const alloc = getAllocator(allocator) orelse return null;
    const v: *Value = @ptrCast(@alignCast(value orelse return null));
    const l: *const PersistentLinkedList(*Value) =
        @ptrCast(@alignCast(list orelse return null));

    const new_list = l.push(alloc, v) catch return null;
    return @ptrCast(@constCast(new_list));
}

/// Pop: remove first element from list
pub export fn linked_list_value_pop(list: ?*anyopaque) ?*anyopaque {
    const l: *const PersistentLinkedList(*Value) =
        @ptrCast(@alignCast(list orelse return null));

    const new_list = l.pop();
    return @ptrCast(@constCast(new_list));
}

/// Check if list is empty
pub export fn linked_list_value_is_empty(list: ?*anyopaque) bool {
    const l: *const PersistentLinkedList(*Value) =
        @ptrCast(@alignCast(list orelse return true));
    return l.isEmpty();
}

/// Get length of linked list
pub export fn linked_list_value_len(list: ?*anyopaque) usize {
    const l: *const PersistentLinkedList(*Value) =
        @ptrCast(@alignCast(list orelse return 0));
    return l.len();
}

/// Convert linked list to Value (as a list node)
pub export fn linked_list_to_value_list(
    allocator: ?*anyopaque,
    list: ?*anyopaque,
) ?*anyopaque {
    const alloc = getAllocator(allocator) orelse return null;
    const l: *const PersistentLinkedList(*Value) =
        @ptrCast(@alignCast(list orelse return null));

    // Convert linked list to PersistentVector for Value.data.list
    var vec = PersistentVector(*Value).init(alloc, null);
    var iter = l.iterator();
    while (iter.next()) |value| {
        vec = vec.push(value) catch return null;
    }

    const value = alloc.create(Value) catch return null;
    value.* = Value{
        .type = .list,
        .data = .{ .list = vec },
    };
    return @ptrCast(value);
}

/// Convert Value list to linked list (for passing to macros)
pub export fn value_list_to_linked_list(
    allocator: ?*anyopaque,
    value: ?*anyopaque,
) ?*anyopaque {
    const alloc = getAllocator(allocator) orelse return null;
    const v: *Value = @ptrCast(@alignCast(value orelse return null));

    if (v.type != .list) return null;

    const list_vec = v.data.list;
    var list = PersistentLinkedList(*Value).empty(alloc) catch return null;

    // Build in reverse
    const slice = list_vec.slice();
    var i = slice.len;
    while (i > 0) : (i -= 1) {
        list = PersistentLinkedList(*Value).cons(alloc, slice[i - 1], list) catch return null;
    }

    return @ptrCast(@constCast(list));
}
