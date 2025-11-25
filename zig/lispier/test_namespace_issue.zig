const std = @import("std");
const tokenizer = @import("src/tokenizer.zig");
const reader = @import("src/reader.zig");

pub fn main() !void {
    const allocator = std.heap.page_allocator;
    
    // Test 1: Dot notation WITHOUT require-dialect
    std.debug.print("\n=== Test 1: arith.addi without require-dialect ===\n", .{});
    {
        var tok = tokenizer.Tokenizer.init(allocator, "(arith.addi 1 2)");
        var tokens = try tok.tokenize();
        defer tokens.deinit(allocator);
        
        var rdr = reader.Reader.init(allocator, tokens.items);
        defer rdr.deinit();
        
        var values = try rdr.read();
        defer {
            for (values.items) |v| v.deinit();
            values.deinit(allocator);
        }
        
        const list = values.items[0];
        const sym = list.data.list.items[0];
        std.debug.print("Symbol name: {s}\n", .{sym.data.symbol.name});
        std.debug.print("Has namespace: {}\n", .{sym.data.symbol.namespace != null});
        if (sym.data.symbol.namespace) |ns| {
            std.debug.print("Namespace name: {s}\n", .{ns.name});
        }
        std.debug.print("Uses dot: {}\n", .{sym.data.symbol.uses_dot});
    }
    
    // Test 2: Dot notation WITH require-dialect
    std.debug.print("\n=== Test 2: arith.addi WITH require-dialect ===\n", .{});
    {
        var tok = tokenizer.Tokenizer.init(allocator, "(require-dialect arith) (arith.addi 1 2)");
        var tokens = try tok.tokenize();
        defer tokens.deinit(allocator);
        
        var rdr = reader.Reader.init(allocator, tokens.items);
        defer rdr.deinit();
        
        var values = try rdr.read();
        defer {
            for (values.items) |v| v.deinit();
            values.deinit(allocator);
        }
        
        const list = values.items[1];
        const sym = list.data.list.items[0];
        std.debug.print("Symbol name: {s}\n", .{sym.data.symbol.name});
        std.debug.print("Has namespace: {}\n", .{sym.data.symbol.namespace != null});
        if (sym.data.symbol.namespace) |ns| {
            std.debug.print("Namespace name: {s}\n", .{ns.name});
        }
        std.debug.print("Uses dot: {}\n", .{sym.data.symbol.uses_dot});
    }
}
