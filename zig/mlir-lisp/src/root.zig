//! By convention, root.zig is the root source file when making a library.
const std = @import("std");

// Export the tokenizer module
pub const tokenizer = @import("tokenizer.zig");
pub const Tokenizer = tokenizer.Tokenizer;
pub const Token = tokenizer.Token;
pub const TokenType = tokenizer.TokenType;

// Export the reader module
pub const reader = @import("reader.zig");
pub const Reader = reader.Reader;
pub const Value = reader.Value;
pub const ValueType = reader.ValueType;

// Export the parser module
pub const parser = @import("parser.zig");
pub const Parser = parser.Parser;
pub const MlirModule = parser.MlirModule;
pub const TypeAlias = parser.TypeAlias;
pub const AttributeAlias = parser.AttributeAlias;
pub const Operation = parser.Operation;
pub const Region = parser.Region;
pub const Block = parser.Block;
pub const Argument = parser.Argument;
pub const Successor = parser.Successor;
pub const Attribute = parser.Attribute;
pub const TypeExpr = parser.TypeExpr;
pub const AttrExpr = parser.AttrExpr;

// Export the collections modules
pub const vector = @import("collections/vector.zig");
pub const PersistentVector = vector.PersistentVector;
pub const PersistentLinkedList = @import("collections/linked_list.zig").PersistentLinkedList;
pub const PersistentMap = @import("collections/map.zig").PersistentMap;
pub const PersistentMapWithEq = @import("collections/map.zig").PersistentMapWithEq;

// Export the C API for collections
pub const c_api = @import("collections/c_api.zig");

// Export C-compatible struct layouts
pub const c_structs = @import("collections/c_structs.zig");
pub const c_value_layout = @import("reader/c_value_layout.zig");

// Export MLIR collection type helpers
pub const collection_types = @import("mlir/collection_types.zig");
pub const value_types = @import("mlir/value_types.zig");

// Export the C API transform module
pub const c_api_transform = @import("c_api_transform.zig");

// Export the macro system modules
pub const macro_expander = @import("macro_expander.zig");
pub const MacroExpander = macro_expander.MacroExpander;
pub const MacroFn = macro_expander.MacroFn;
pub const c_api_macro = @import("c_api_macro.zig");
pub const builtin_macros = @import("builtin_macros.zig");

// Export the operation flattener module
pub const OperationFlattener = @import("operation_flattener.zig").OperationFlattener;

// Export the MLIR module
pub const mlir = @import("mlir/c.zig");

// Export the builder module
pub const builder = @import("builder.zig");
pub const Builder = builder.Builder;

// Export the printer module
pub const Printer = @import("printer.zig").Printer;

// Export the executor module
pub const Executor = @import("executor.zig").Executor;
pub const ExecutorConfig = @import("executor.zig").ExecutorConfig;
pub const OptLevel = @import("executor.zig").OptLevel;

// Export the runtime symbols registration module
pub const runtime_symbols = @import("runtime_symbols.zig");

// Export the function call helper module
pub const function_call_helper = @import("function_call_helper.zig");
pub const FunctionSignature = function_call_helper.FunctionSignature;
pub const FunctionArg = function_call_helper.FunctionArg;
pub const RuntimeType = function_call_helper.RuntimeType;

// Export the REPL module
pub const Repl = @import("repl.zig");

pub fn bufferedPrint() !void {
    // Just print to debug output for simplicity
    std.debug.print("Run `zig build test` to run the tests.\n", .{});
}

pub fn add(a: i32, b: i32) i32 {
    return a + b;
}

test "basic add functionality" {
    try std.testing.expect(add(3, 7) == 10);
}
