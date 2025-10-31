/// Runtime symbol registration for JIT execution
/// This module registers all C API functions and common C library functions
/// with the MLIR ExecutionEngine so that JIT'd code can call them.

const std = @import("std");
const Executor = @import("executor.zig").Executor;
const c_api = @import("collections/c_api.zig");

/// Register all runtime symbols with the execution engine
/// This includes:
/// - Common C library functions (printf, malloc, free)
/// - All allocator management functions from c_api.zig
/// - All vector manipulation functions
/// - All map manipulation functions
/// - All Value manipulation functions
pub fn registerAllRuntimeSymbols(executor: *Executor) void {
    // Register common C library functions
    executor.registerSymbol("printf", @ptrCast(@constCast(&std.c.printf)));
    executor.registerSymbol("malloc", @ptrCast(@constCast(&std.c.malloc)));
    executor.registerSymbol("free", @ptrCast(@constCast(&std.c.free)));

    // Allocator Management
    executor.registerSymbol("allocator_set_global_arena", @ptrCast(@constCast(&c_api.allocator_set_global_arena)));
    executor.registerSymbol("allocator_create_c", @ptrCast(@constCast(&c_api.allocator_create_c)));
    executor.registerSymbol("allocator_destroy", @ptrCast(@constCast(&c_api.allocator_destroy)));

    // Vector API (Value pointers)
    executor.registerSymbol("vector_value_create", @ptrCast(@constCast(&c_api.vector_value_create)));
    executor.registerSymbol("vector_value_destroy", @ptrCast(@constCast(&c_api.vector_value_destroy)));
    executor.registerSymbol("vector_value_push", @ptrCast(@constCast(&c_api.vector_value_push)));
    executor.registerSymbol("vector_value_pop", @ptrCast(@constCast(&c_api.vector_value_pop)));
    executor.registerSymbol("vector_value_at", @ptrCast(@constCast(&c_api.vector_value_at)));
    executor.registerSymbol("vector_value_len", @ptrCast(@constCast(&c_api.vector_value_len)));
    executor.registerSymbol("vector_value_is_empty", @ptrCast(@constCast(&c_api.vector_value_is_empty)));

    // Map API (string -> Value)
    executor.registerSymbol("map_str_value_create", @ptrCast(@constCast(&c_api.map_str_value_create)));
    executor.registerSymbol("map_str_value_destroy", @ptrCast(@constCast(&c_api.map_str_value_destroy)));
    executor.registerSymbol("map_str_value_get", @ptrCast(@constCast(&c_api.map_str_value_get)));
    executor.registerSymbol("map_str_value_set", @ptrCast(@constCast(&c_api.map_str_value_set)));

    // Value API - Type queries and data extraction
    executor.registerSymbol("value_get_type", @ptrCast(@constCast(&c_api.value_get_type)));
    executor.registerSymbol("value_get_atom", @ptrCast(@constCast(&c_api.value_get_atom)));
    executor.registerSymbol("value_get_list", @ptrCast(@constCast(&c_api.value_get_list)));
    executor.registerSymbol("value_get_vector", @ptrCast(@constCast(&c_api.value_get_vector)));
    executor.registerSymbol("value_get_map", @ptrCast(@constCast(&c_api.value_get_map)));
    executor.registerSymbol("value_get_type_string", @ptrCast(@constCast(&c_api.value_get_type_string)));
    executor.registerSymbol("value_get_attr_expr", @ptrCast(@constCast(&c_api.value_get_attr_expr)));
    executor.registerSymbol("value_get_has_type_value", @ptrCast(@constCast(&c_api.value_get_has_type_value)));
    executor.registerSymbol("value_get_has_type_type_expr", @ptrCast(@constCast(&c_api.value_get_has_type_type_expr)));

    // Value API - Creation functions for atoms
    executor.registerSymbol("value_create_identifier", @ptrCast(@constCast(&c_api.value_create_identifier)));
    executor.registerSymbol("value_create_number", @ptrCast(@constCast(&c_api.value_create_number)));
    executor.registerSymbol("value_create_string", @ptrCast(@constCast(&c_api.value_create_string)));
    executor.registerSymbol("value_create_value_id", @ptrCast(@constCast(&c_api.value_create_value_id)));
    executor.registerSymbol("value_create_block_id", @ptrCast(@constCast(&c_api.value_create_block_id)));
    executor.registerSymbol("value_create_symbol", @ptrCast(@constCast(&c_api.value_create_symbol)));
    executor.registerSymbol("value_create_keyword", @ptrCast(@constCast(&c_api.value_create_keyword)));
    executor.registerSymbol("value_create_true", @ptrCast(@constCast(&c_api.value_create_true)));
    executor.registerSymbol("value_create_false", @ptrCast(@constCast(&c_api.value_create_false)));

    // Value API - Creation functions for collections
    executor.registerSymbol("value_create_list", @ptrCast(@constCast(&c_api.value_create_list)));
    executor.registerSymbol("value_create_vector", @ptrCast(@constCast(&c_api.value_create_vector)));
    executor.registerSymbol("value_create_map", @ptrCast(@constCast(&c_api.value_create_map)));

    // Value API - Creation functions for special expressions
    executor.registerSymbol("value_create_type", @ptrCast(@constCast(&c_api.value_create_type)));
    executor.registerSymbol("value_create_attr_expr", @ptrCast(@constCast(&c_api.value_create_attr_expr)));
    executor.registerSymbol("value_create_type_expr", @ptrCast(@constCast(&c_api.value_create_type_expr)));
    executor.registerSymbol("value_create_has_type", @ptrCast(@constCast(&c_api.value_create_has_type)));

    // Value API - Memory management
    executor.registerSymbol("value_destroy", @ptrCast(@constCast(&c_api.value_destroy)));
    executor.registerSymbol("value_free_atom", @ptrCast(@constCast(&c_api.value_free_atom)));
}
