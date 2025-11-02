// Test file demonstrating C API usage for transforming function calls
// This file uses only C-compatible types and the C API

// External C API declarations - these would be provided by the compiled library
extern fn allocator_create_c() ?*anyopaque;
extern fn allocator_destroy(allocator: ?*anyopaque) void;

extern fn vector_value_create(allocator: ?*anyopaque) ?*anyopaque;
extern fn vector_value_destroy(allocator: ?*anyopaque, vec: ?*anyopaque) void;
extern fn vector_value_push(allocator: ?*anyopaque, vec: ?*anyopaque, value: ?*anyopaque) ?*anyopaque;
extern fn vector_value_at(vec: ?*const anyopaque, index: usize) ?*anyopaque;
extern fn vector_value_len(vec: ?*const anyopaque) usize;

extern fn map_str_value_create(allocator: ?*anyopaque) ?*anyopaque;
extern fn map_str_value_destroy(allocator: ?*anyopaque, m: ?*anyopaque) void;
extern fn map_str_value_set(allocator: ?*anyopaque, m: ?*anyopaque, key: [*:0]const u8, value: ?*anyopaque) ?*anyopaque;

extern fn value_get_type(val: ?*const anyopaque) c_int;
extern fn value_get_atom(allocator: ?*anyopaque, val: ?*const anyopaque) ?[*:0]const u8;
extern fn value_get_list(allocator: ?*anyopaque, val: ?*const anyopaque) ?*anyopaque;
extern fn value_free_atom(allocator: ?*anyopaque, atom: ?[*:0]const u8) void;

extern fn value_create_identifier(allocator: ?*anyopaque, atom: [*:0]const u8) ?*anyopaque;
extern fn value_create_symbol(allocator: ?*anyopaque, atom: [*:0]const u8) ?*anyopaque;
extern fn value_create_keyword(allocator: ?*anyopaque, atom: [*:0]const u8) ?*anyopaque;
extern fn value_create_list(allocator: ?*anyopaque, vec: ?*anyopaque) ?*anyopaque;
extern fn value_create_map(allocator: ?*anyopaque, vec: ?*anyopaque) ?*anyopaque;
extern fn value_create_type_expr(allocator: ?*anyopaque, inner: ?*anyopaque) ?*anyopaque;
extern fn value_destroy(allocator: ?*anyopaque, val: ?*anyopaque) void;

// C-compatible enum matching CValueType
const CValueType = enum(c_int) {
    identifier = 0,
    number = 1,
    string = 2,
    value_id = 3,
    block_id = 4,
    symbol = 5,
    keyword = 6,
    true_lit = 7,
    false_lit = 8,
    list = 9,
    vector = 10,
    map = 11,
    type_expr = 12,
    attr_expr = 13,
    has_type = 14,
};

// Transform (call @test i64) => (operation (name func.call) ...)
// Returns null on error (C-compatible calling convention)
pub export fn transformCallToOperation(allocator: ?*anyopaque, call_expr: ?*anyopaque) ?*anyopaque {
    if (allocator == null or call_expr == null) return null;

    // Get the list elements from (call @test i64)
    const call_list = value_get_list(allocator, call_expr) orelse return null;
    defer vector_value_destroy(allocator, call_list);

    const list_len = vector_value_len(call_list);
    if (list_len < 3) return null; // Need at least (call <symbol> <type>)

    // Extract: call, @test, i64
    const call_ident = vector_value_at(call_list, 0); // "call"
    const callee_symbol = vector_value_at(call_list, 1); // "@test"
    const return_type = vector_value_at(call_list, 2); // "i64"

    // Verify first element is "call"
    const call_atom = value_get_atom(allocator, call_ident);
    defer value_free_atom(allocator, call_atom);
    // In real code we'd check if it equals "call"

    // Build result: (operation (name func.call) (result-bindings [%gensym]) (result-types i64) (attributes { :callee @test }))

    // Create "operation" identifier
    const operation_ident = value_create_identifier(allocator, "operation") orelse return null;

    // Create (name func.call)
    const name_ident = value_create_identifier(allocator, "name") orelse return null;
    const func_call_ident = value_create_identifier(allocator, "func.call") orelse return null;

    const name_list_vec = vector_value_create(allocator) orelse return null;
    const name_vec_1 = vector_value_push(allocator, name_list_vec, name_ident) orelse return null;
    vector_value_destroy(allocator, name_list_vec);
    const name_vec_2 = vector_value_push(allocator, name_vec_1, func_call_ident) orelse return null;
    vector_value_destroy(allocator, name_vec_1);

    const name_clause = value_create_list(allocator, name_vec_2) orelse return null;
    // Note: name_vec_2 is now owned by name_clause

    // Create (result-bindings [%some_gensym])
    const result_bindings_ident = value_create_identifier(allocator, "result-bindings") orelse return null;
    const gensym_value_id = value_create_identifier(allocator, "%result0") orelse return null; // Using %result0 as gensym

    const gensym_vec = vector_value_create(allocator) orelse return null;
    const gensym_vec_1 = vector_value_push(allocator, gensym_vec, gensym_value_id) orelse return null;
    vector_value_destroy(allocator, gensym_vec);

    // Note: In the real API, we'd create a vector value type, but for now using list
    const bindings_vector = value_create_list(allocator, gensym_vec_1) orelse return null;

    const bindings_vec = vector_value_create(allocator) orelse return null;
    const bindings_vec_1 = vector_value_push(allocator, bindings_vec, result_bindings_ident) orelse return null;
    vector_value_destroy(allocator, bindings_vec);
    const bindings_vec_2 = vector_value_push(allocator, bindings_vec_1, bindings_vector) orelse return null;
    vector_value_destroy(allocator, bindings_vec_1);

    const bindings_clause = value_create_list(allocator, bindings_vec_2) orelse return null;

    // Create (result-types i64)
    const result_types_ident = value_create_identifier(allocator, "result-types") orelse return null;

    // Create type expression i64 from the return_type
    const type_expr = value_create_type_expr(allocator, return_type) orelse return null;

    const types_vec = vector_value_create(allocator) orelse return null;
    const types_vec_1 = vector_value_push(allocator, types_vec, result_types_ident) orelse return null;
    vector_value_destroy(allocator, types_vec);
    const types_vec_2 = vector_value_push(allocator, types_vec_1, type_expr) orelse return null;
    vector_value_destroy(allocator, types_vec_1);

    const types_clause = value_create_list(allocator, types_vec_2) orelse return null;

    // Create (attributes { :callee @test })
    const attributes_ident = value_create_identifier(allocator, "attributes") orelse return null;

    // Create map { :callee @test } - map is a flat vector of key-value pairs
    const callee_keyword = value_create_keyword(allocator, ":callee") orelse return null;

    const map_vec = vector_value_create(allocator) orelse return null;
    const map_vec_1 = vector_value_push(allocator, map_vec, callee_keyword) orelse return null;
    vector_value_destroy(allocator, map_vec);
    const map_vec_2 = vector_value_push(allocator, map_vec_1, callee_symbol) orelse return null;
    vector_value_destroy(allocator, map_vec_1);

    const attributes_map = value_create_map(allocator, map_vec_2) orelse return null;

    const attrs_vec = vector_value_create(allocator) orelse return null;
    const attrs_vec_1 = vector_value_push(allocator, attrs_vec, attributes_ident) orelse return null;
    vector_value_destroy(allocator, attrs_vec);
    const attrs_vec_2 = vector_value_push(allocator, attrs_vec_1, attributes_map) orelse return null;
    vector_value_destroy(allocator, attrs_vec_1);

    const attrs_clause = value_create_list(allocator, attrs_vec_2) orelse return null;

    // Build the final operation list: (operation <name> <bindings> <types> <attrs>)
    const op_vec = vector_value_create(allocator) orelse return null;
    const op_vec_1 = vector_value_push(allocator, op_vec, operation_ident) orelse return null;
    vector_value_destroy(allocator, op_vec);
    const op_vec_2 = vector_value_push(allocator, op_vec_1, name_clause) orelse return null;
    vector_value_destroy(allocator, op_vec_1);
    const op_vec_3 = vector_value_push(allocator, op_vec_2, bindings_clause) orelse return null;
    vector_value_destroy(allocator, op_vec_2);
    const op_vec_4 = vector_value_push(allocator, op_vec_3, types_clause) orelse return null;
    vector_value_destroy(allocator, op_vec_3);
    const op_vec_5 = vector_value_push(allocator, op_vec_4, attrs_clause) orelse return null;
    vector_value_destroy(allocator, op_vec_4);

    const operation = value_create_list(allocator, op_vec_5) orelse return null;

    return operation;
}

// Example usage demonstrating the transformation
// This would be called from C code like:
//
//   void* allocator = allocator_create_c();
//   void* call_expr = ...; // (call @test i64)
//   void* operation = transformCallToOperation(allocator, call_expr);
//   value_destroy(allocator, operation);
//   allocator_destroy(allocator);
//
export fn exampleTransformCallToOperation() ?*anyopaque {
    // Create allocator
    const allocator = allocator_create_c() orelse return null;

    // Build input: (call @test i64)
    const call_ident = value_create_identifier(allocator, "call") orelse return null;
    const test_symbol = value_create_symbol(allocator, "@test") orelse return null;
    const i64_ident = value_create_identifier(allocator, "i64") orelse return null;

    const input_vec = vector_value_create(allocator) orelse return null;
    const input_vec_1 = vector_value_push(allocator, input_vec, call_ident) orelse return null;
    vector_value_destroy(allocator, input_vec);
    const input_vec_2 = vector_value_push(allocator, input_vec_1, test_symbol) orelse return null;
    vector_value_destroy(allocator, input_vec_1);
    const input_vec_3 = vector_value_push(allocator, input_vec_2, i64_ident) orelse return null;
    vector_value_destroy(allocator, input_vec_2);

    const call_expr = value_create_list(allocator, input_vec_3) orelse return null;

    // Transform
    const operation = transformCallToOperation(allocator, call_expr) orelse return null;

    // Clean up input
    value_destroy(allocator, call_expr);

    // Return the operation (caller must destroy it)
    // Note: caller must also destroy allocator when done
    return operation;
}
