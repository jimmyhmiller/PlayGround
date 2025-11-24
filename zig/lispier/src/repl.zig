const std = @import("std");
const lispier = @import("main.zig");

pub fn main() !void {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();

    const stdout_file = std.io.getStdOut();
    const stdin_file = std.io.getStdIn();
    const stdout = stdout_file.writer();
    const stdin = stdin_file.reader();

    try stdout.print("Lispier REPL v0.1.0\n", .{});
    try stdout.print("Type expressions or :quit to exit\n\n", .{});

    var compiler = try lispier.Compiler.init(allocator);
    defer compiler.deinit();

    var line_buffer: [4096]u8 = undefined;

    while (true) {
        try stdout.print("> ", .{});

        const line = (try stdin.readUntilDelimiterOrEof(&line_buffer, '\n')) orelse break;

        if (std.mem.eql(u8, line, ":quit") or std.mem.eql(u8, line, ":q")) {
            break;
        }

        if (std.mem.eql(u8, line, ":help")) {
            try printHelp(stdout);
            continue;
        }

        if (std.mem.startsWith(u8, line, ":load-dialect ")) {
            const dialect_name = std.mem.trim(u8, line[14..], " \t");
            compiler.dialect_registry.loadDialect(dialect_name) catch |err| {
                try stdout.print("Error loading dialect: {}\n", .{err});
                continue;
            };
            try stdout.print("Loaded dialect: {s}\n", .{dialect_name});
            continue;
        }

        if (std.mem.eql(u8, line, ":dialects")) {
            const dialects = try compiler.dialect_registry.listLoadedDialects();
            defer {
                for (dialects.items) |d| {
                    allocator.free(d);
                }
                dialects.deinit(allocator);
            }

            try stdout.print("Loaded dialects:\n", .{});
            for (dialects.items) |d| {
                try stdout.print("  - {s}\n", .{d});
            }
            continue;
        }

        if (std.mem.startsWith(u8, line, ":ops ")) {
            const dialect_name = std.mem.trim(u8, line[5..], " \t");
            const ops = compiler.dialect_registry.enumerateOperations(dialect_name) catch |err| {
                try stdout.print("Error enumerating operations: {}\n", .{err});
                continue;
            };
            defer {
                for (ops.items) |op| {
                    allocator.free(op);
                }
                ops.deinit(allocator);
            }

            try stdout.print("Operations in {s}:\n", .{dialect_name});
            for (ops.items) |op| {
                try stdout.print("  - {s}\n", .{op});
            }
            continue;
        }

        if (line.len == 0) {
            continue;
        }

        var result = compiler.compile(line) catch |err| {
            try stdout.print("Compilation error: {}\n", .{err});
            continue;
        };
        defer result.deinit(allocator);

        if (result.is_valid) {
            try stdout.print("✓ Compiled successfully\n", .{});
            try stdout.print("  Tokens: {}\n", .{result.tokens.items.len});
            try stdout.print("  Values: {}\n", .{result.values.items.len});
            try stdout.print("  AST Nodes: {}\n", .{result.nodes.items.len});

            // Print AST summary
            for (result.nodes.items, 0..) |node, i| {
                try stdout.print("  Node {}: {s}\n", .{ i, @tagName(node.node_type) });
                if (node.node_type == .Operation) {
                    const op = node.data.operation;
                    const qualified = try op.getQualifiedName(allocator);
                    defer allocator.free(qualified);
                    try stdout.print("    Operation: {s}\n", .{qualified});
                    try stdout.print("    Operands: {}\n", .{op.operands.items.len});
                    try stdout.print("    Regions: {}\n", .{op.regions.items.len});
                }
            }
        } else {
            try stdout.print("✗ Validation failed:\n", .{});
            for (result.validation_errors) |err| {
                try stdout.print("  - {s}\n", .{err.message});
            }
        }
        try stdout.print("\n", .{});
    }

    try stdout.print("Goodbye!\n", .{});
}

fn printHelp(writer: anytype) !void {
    try writer.print(
        \\Available commands:
        \\  :help               - Show this help
        \\  :quit, :q           - Exit the REPL
        \\  :load-dialect NAME  - Load an MLIR dialect
        \\  :dialects           - List loaded dialects
        \\  :ops DIALECT        - List operations in a dialect
        \\
        \\Examples:
        \\  (require-dialect arith)
        \\  (arith.addi 1 2)
        \\  (def x 42)
        \\
    , .{});
}
