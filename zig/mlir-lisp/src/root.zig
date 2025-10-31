//! By convention, root.zig is the root source file when making a library.
const std = @import("std");

// Export the tokenizer module
pub const Tokenizer = @import("tokenizer.zig").Tokenizer;
pub const Token = @import("tokenizer.zig").Token;
pub const TokenType = @import("tokenizer.zig").TokenType;

// Export the reader module
pub const Reader = @import("reader.zig").Reader;
pub const Value = @import("reader.zig").Value;
pub const ValueType = @import("reader.zig").ValueType;

// Export the parser module
pub const Parser = @import("parser.zig").Parser;
pub const MlirModule = @import("parser.zig").MlirModule;
pub const TypeAlias = @import("parser.zig").TypeAlias;
pub const AttributeAlias = @import("parser.zig").AttributeAlias;
pub const Operation = @import("parser.zig").Operation;
pub const Region = @import("parser.zig").Region;
pub const Block = @import("parser.zig").Block;
pub const Argument = @import("parser.zig").Argument;
pub const Successor = @import("parser.zig").Successor;
pub const Attribute = @import("parser.zig").Attribute;
pub const TypeExpr = @import("parser.zig").TypeExpr;
pub const AttrExpr = @import("parser.zig").AttrExpr;

// Export the collections modules
pub const PersistentVector = @import("collections/vector.zig").PersistentVector;
pub const PersistentLinkedList = @import("collections/linked_list.zig").PersistentLinkedList;
pub const PersistentMap = @import("collections/map.zig").PersistentMap;
pub const PersistentMapWithEq = @import("collections/map.zig").PersistentMapWithEq;

// Export the C API for collections
pub const c_api = @import("collections/c_api.zig");

// Export the MLIR module
pub const mlir = @import("mlir/c.zig");

// Export the builder module
pub const Builder = @import("builder.zig").Builder;

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
