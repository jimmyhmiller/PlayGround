//! MLIR Parser Library
//! Provides lexer, parser, printer, and AST for parsing and printing MLIR code
const std = @import("std");

// Export main modules
pub const lexer = @import("lexer.zig");
pub const parser = @import("parser.zig");
pub const printer = @import("printer.zig");
pub const lisp_printer = @import("lisp_printer.zig");
pub const ast = @import("ast.zig");

// Convenience exports
pub const Lexer = lexer.Lexer;
pub const Parser = parser.Parser;
pub const Printer = printer.Printer;
pub const LispPrinter = lisp_printer.LispPrinter;
pub const Token = lexer.Token;
pub const TokenType = lexer.TokenType;

// AST types
pub const Module = ast.Module;
pub const Operation = ast.Operation;
pub const Block = ast.Block;
pub const Region = ast.Region;
pub const Type = ast.Type;

/// Parse MLIR source code into an AST
/// Caller must call deinit() on the returned module
pub fn parse(allocator: std.mem.Allocator, source: []const u8) !Module {
    var lex = Lexer.init(source);
    var p = try Parser.init(allocator, &lex);
    defer p.deinit();
    return try p.parseModule();
}

/// Print MLIR module to string
/// Caller must free the returned string
pub fn print(allocator: std.mem.Allocator, module: Module) ![]u8 {
    return try printer.moduleToString(allocator, module);
}

/// Format MLIR module to a writer
pub fn format(module: Module, writer: std.io.AnyWriter) !void {
    var p = Printer.init(writer);
    try p.printModule(module);
}

/// Print MLIR module to Lisp S-expression string
/// Caller must free the returned string
pub fn printLisp(allocator: std.mem.Allocator, module: Module) ![]u8 {
    var arena = std.heap.ArenaAllocator.init(allocator);
    defer arena.deinit();
    const arena_allocator = arena.allocator();

    var output: std.ArrayList(u8) = .empty;

    var p = LispPrinter.init(output.writer(arena_allocator).any());
    try p.printModule(module);

    // Duplicate the result with the parent allocator so it survives arena.deinit()
    const result = try allocator.dupe(u8, output.items);
    return result;
}

/// Format MLIR module to Lisp S-expression to a writer
pub fn formatLisp(module: Module, writer: std.io.AnyWriter) !void {
    var p = LispPrinter.init(writer);
    try p.printModule(module);
}

test "parse simple types" {
    const source = "i32";
    var lex = Lexer.init(source);
    var p = try Parser.init(std.testing.allocator, &lex);
    defer p.deinit();

    var type_result = try p.parseType();
    defer type_result.deinit(std.testing.allocator);

    try std.testing.expect(type_result == .builtin);
    try std.testing.expect(type_result.builtin == .integer);
}
