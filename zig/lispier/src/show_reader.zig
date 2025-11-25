const std = @import("std");
const main_module = @import("main.zig");
const reader_types = main_module.reader_types;
const tokenizer = main_module.tokenizer;
const reader = main_module.reader;
const mlir_integration = main_module.mlir_integration;

pub fn main() !void {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();

    const args = try std.process.argsAlloc(allocator);
    defer std.process.argsFree(allocator, args);

    if (args.len < 2) {
        std.debug.print("Usage: {s} <source-code>\n", .{args[0]});
        std.debug.print("Example: {s} '(arith.addi 1 2)'\n", .{args[0]});
        return;
    }

    const source = args[1];

    // Create dialect registry for operation validation
    var dialect_registry = try mlir_integration.DialectRegistry.init(allocator);
    defer dialect_registry.deinit();

    // Tokenize
    var tok = tokenizer.Tokenizer.init(allocator, source);
    var tokens = try tok.tokenize();
    defer tokens.deinit(allocator);

    std.debug.print("=== TOKENS ===\n", .{});
    for (tokens.items, 0..) |token, i| {
        std.debug.print("{d}: {s} at line {d}, col {d}", .{
            i,
            @tagName(token.type),
            token.line,
            token.column,
        });
        if (token.lexeme.len > 0) {
            std.debug.print(" = \"{s}\"", .{token.lexeme});
        }
        std.debug.print("\n", .{});
    }

    // Read (with dialect registry for operation validation)
    var rdr = reader.Reader.initWithRegistry(allocator, tokens.items, &dialect_registry);
    defer rdr.deinit();

    var values = try rdr.read();
    defer {
        for (values.items) |v| {
            v.deinit();
        }
        values.deinit(allocator);
    }

    std.debug.print("\n=== READER OUTPUT ===\n", .{});
    for (values.items, 0..) |value, i| {
        std.debug.print("{d}: ", .{i});
        try printValue(value, 0);
        std.debug.print("\n", .{});
    }
}

fn printValue(value: *reader_types.Value, indent: usize) !void {
    const spaces = "                                                  ";
    const indent_str = spaces[0..@min(indent * 2, spaces.len)];

    switch (value.type) {
        .Number => std.debug.print("{d}", .{value.data.number}),
        .String => std.debug.print("\"{s}\"", .{value.data.string}),
        .Symbol => {
            const sym = value.data.symbol;
            if (sym.namespace) |ns| {
                if (sym.uses_dot) {
                    std.debug.print("{s}.{s}", .{ ns.name, sym.name });
                } else if (sym.uses_alias) {
                    std.debug.print("{s}/{s}", .{ ns.name, sym.name });
                } else {
                    // Has namespace but no explicit notation (from use-dialect or default user namespace)
                    // Display with dot notation to show the resolved namespace
                    std.debug.print("{s}.{s}", .{ ns.name, sym.name });
                }
            } else {
                // Shouldn't happen anymore since we always have at least the user namespace
                std.debug.print("{s}", .{sym.name});
            }
        },
        .Keyword => std.debug.print(":{s}", .{value.data.keyword}),
        .Boolean => std.debug.print("{}", .{value.data.boolean}),
        .Nil => std.debug.print("nil", .{}),
        .List => {
            std.debug.print("(", .{});
            for (value.data.list.items, 0..) |item, i| {
                if (i > 0) std.debug.print(" ", .{});
                try printValue(item, indent + 1);
            }
            std.debug.print(")", .{});
        },
        .Vector => {
            std.debug.print("[", .{});
            for (value.data.vector.items, 0..) |item, i| {
                if (i > 0) std.debug.print(" ", .{});
                try printValue(item, indent + 1);
            }
            std.debug.print("]", .{});
        },
        .Map => {
            std.debug.print("{{\n", .{});
            var it = value.data.map.iterator();
            while (it.next()) |entry| {
                std.debug.print("{s}  :{s} ", .{ indent_str, entry.key_ptr.* });
                try printValue(entry.value_ptr.*, indent + 1);
                std.debug.print("\n", .{});
            }
            std.debug.print("{s}}}", .{indent_str});
        },
    }
}
