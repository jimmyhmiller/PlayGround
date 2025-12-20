const std = @import("std");
const main_module = @import("main.zig");
const ast = main_module.ast;
const tokenizer = main_module.tokenizer;
const reader = main_module.reader;
const parser = main_module.parser;
const mlir_integration = main_module.mlir_integration;

pub fn main() !void {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();

    const args = try std.process.argsAlloc(allocator);
    defer std.process.argsFree(allocator, args);

    if (args.len < 2) {
        std.debug.print("Usage: {s} <source-code>\n", .{args[0]});
        std.debug.print("Example: {s} '(require-dialect arith) (def %x (arith.addi 1 2))'\n", .{args[0]});
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

    // Parse
    var psr = parser.Parser.init(allocator);
    var nodes = try psr.parse(values);
    defer {
        for (nodes.items) |n| {
            n.deinit();
        }
        nodes.deinit(allocator);
    }

    std.debug.print("=== AST ===\n", .{});
    for (nodes.items, 0..) |node, i| {
        std.debug.print("\n[{d}] ", .{i});
        try printNode(node, 0);
        std.debug.print("\n", .{});
    }
}

fn printIndent(indent: usize) void {
    var i: usize = 0;
    while (i < indent) : (i += 1) {
        std.debug.print("  ", .{});
    }
}

fn printNode(node: *ast.Node, indent: usize) std.mem.Allocator.Error!void {
    printIndent(indent);

    switch (node.node_type) {
        .Module => std.debug.print("Module\n", .{}),

        .Operation => {
            const op = node.data.operation;
            if (op.namespace) |ns| {
                std.debug.print("Operation: {s}.{s}\n", .{ ns, op.name });
            } else {
                std.debug.print("Operation: {s}\n", .{op.name});
            }

            if (op.operands.items.len > 0) {
                printIndent(indent + 1);
                std.debug.print("Operands ({d}):\n", .{op.operands.items.len});
                for (op.operands.items) |operand| {
                    try printNode(operand, indent + 2);
                }
            }

            if (op.result_types.items.len > 0) {
                printIndent(indent + 1);
                std.debug.print("Result Types:\n", .{});
                for (op.result_types.items) |t| {
                    printIndent(indent + 2);
                    std.debug.print("- {s}\n", .{t.name});
                }
            }

            if (op.attributes.count() > 0) {
                printIndent(indent + 1);
                std.debug.print("Attributes:\n", .{});
                var it = op.attributes.iterator();
                while (it.next()) |entry| {
                    printIndent(indent + 2);
                    std.debug.print("{s}: ", .{entry.key_ptr.*});
                    try printAttributeValue(entry.value_ptr.*, indent + 3);
                    std.debug.print("\n", .{});
                }
            }

            if (op.regions.items.len > 0) {
                printIndent(indent + 1);
                std.debug.print("Regions ({d}):\n", .{op.regions.items.len});
                for (op.regions.items) |region| {
                    try printRegion(region, indent + 2);
                }
            }
        },

        .Region => {
            std.debug.print("Region\n", .{});
            try printRegion(node.data.region, indent + 1);
        },

        .Block => {
            std.debug.print("Block\n", .{});
            try printBlock(node.data.block, indent + 1);
        },

        .Def => {
            const binding = node.data.binding;
            std.debug.print("Def:\n", .{});
            printIndent(indent + 1);
            std.debug.print("Names: ", .{});
            for (binding.names.items, 0..) |name, i| {
                if (i > 0) std.debug.print(", ", .{});
                std.debug.print("{s}", .{name});
            }
            std.debug.print("\n", .{});
            printIndent(indent + 1);
            std.debug.print("Value:\n", .{});
            try printNode(binding.value, indent + 2);
        },

        .Let => {
            const let_expr = node.data.let_expr;
            std.debug.print("Let:\n", .{});
            printIndent(indent + 1);
            std.debug.print("Bindings ({d}):\n", .{let_expr.bindings.items.len});
            for (let_expr.bindings.items) |binding| {
                printIndent(indent + 2);
                std.debug.print("Names: ", .{});
                for (binding.names.items, 0..) |name, i| {
                    if (i > 0) std.debug.print(", ", .{});
                    std.debug.print("{s}", .{name});
                }
                std.debug.print("\n", .{});
                printIndent(indent + 2);
                std.debug.print("Value:\n", .{});
                try printNode(binding.value, indent + 3);
            }
            printIndent(indent + 1);
            std.debug.print("Body ({d}):\n", .{let_expr.body.items.len});
            for (let_expr.body.items) |body_node| {
                try printNode(body_node, indent + 2);
            }
        },

        .TypeAnnotation => {
            const ta = node.data.type_annotation;
            std.debug.print("TypeAnnotation: {s}\n", .{ta.typ.name});
            printIndent(indent + 1);
            std.debug.print("Value:\n", .{});
            try printNode(ta.value, indent + 2);
        },

        .FunctionType => {
            std.debug.print("FunctionType\n", .{});
        },

        .Literal => {
            const value = node.data.literal;
            std.debug.print("Literal: ", .{});
            switch (value.type) {
                .Number => std.debug.print("{d}\n", .{value.data.number}),
                .String => std.debug.print("\"{s}\"\n", .{value.data.string}),
                .Symbol => {
                    const sym = value.data.symbol;
                    if (sym.namespace) |ns| {
                        if (sym.uses_dot) {
                            std.debug.print("{s}.{s}\n", .{ ns.name, sym.name });
                        } else if (sym.uses_alias) {
                            std.debug.print("{s}/{s}\n", .{ ns.name, sym.name });
                        } else {
                            std.debug.print("{s}\n", .{sym.name});
                        }
                    } else {
                        std.debug.print("{s}\n", .{sym.name});
                    }
                },
                .Keyword => std.debug.print(":{s}\n", .{value.data.keyword}),
                .Boolean => std.debug.print("{}\n", .{value.data.boolean}),
                .Nil => std.debug.print("nil\n", .{}),
                else => std.debug.print("<complex>\n", .{}),
            }
        },
    }
}

fn printRegion(region: *ast.Region, indent: usize) std.mem.Allocator.Error!void {
    printIndent(indent);
    std.debug.print("Region with {d} block(s):\n", .{region.blocks.items.len});
    for (region.blocks.items) |block| {
        try printBlock(block, indent + 1);
    }
}

fn printBlock(block: *ast.Block, indent: usize) std.mem.Allocator.Error!void {
    printIndent(indent);
    if (block.label) |label| {
        std.debug.print("Block ^{s}:\n", .{label});
    } else {
        std.debug.print("Block:\n", .{});
    }

    if (block.arguments.items.len > 0) {
        printIndent(indent + 1);
        std.debug.print("Arguments:\n", .{});
        for (block.arguments.items) |arg| {
            printIndent(indent + 2);
            std.debug.print("{s}", .{arg.name});
            if (arg.type) |t| {
                std.debug.print(": {s}", .{t.name});
            }
            std.debug.print("\n", .{});
        }
    }

    if (block.operations.items.len > 0) {
        printIndent(indent + 1);
        std.debug.print("Operations ({d}):\n", .{block.operations.items.len});
        for (block.operations.items) |op| {
            try printNode(op, indent + 2);
        }
    }
}

fn printAttributeValue(value: ast.AttributeValue, indent: usize) std.mem.Allocator.Error!void {
    switch (value) {
        .string => |s| std.debug.print("\"{s}\"", .{s}),
        .number => |n| std.debug.print("{d}", .{n}),
        .boolean => |b| std.debug.print("{}", .{b}),
        .array => |arr| {
            std.debug.print("[\n", .{});
            for (arr.items) |item| {
                printIndent(indent);
                try printAttributeValue(item, indent + 1);
                std.debug.print("\n", .{});
            }
            printIndent(indent - 1);
            std.debug.print("]", .{});
        },
        .type => |t| std.debug.print("Type({s})", .{t.name}),
        .function_type => std.debug.print("FunctionType", .{}),
        .typed_number => |tn| std.debug.print("{d} : {s}", .{ tn.value, tn.typ.name }),
    }
}
