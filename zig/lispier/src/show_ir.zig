const std = @import("std");
const main_module = @import("main");

const tokenizer = main_module.tokenizer;
const reader = main_module.reader;
const parser = main_module.parser;
const mlir_integration = main_module.mlir_integration;
const ir_generator = main_module.ir_generator;

pub fn main() !void {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();

    // Get source from command line
    var args = std.process.args();
    _ = args.skip(); // Skip program name

    const source = args.next() orelse {
        std.debug.print("Usage: show-ir '<source>'\n", .{});
        std.debug.print("Example: show-ir '(require-dialect arith) (def x (arith.addi (: 1 i32) (: 2 i32)))'\n", .{});
        return;
    };

    std.debug.print("=== SOURCE ===\n{s}\n\n", .{source});

    // Tokenize
    var tok = tokenizer.Tokenizer.init(allocator, source);
    var tokens = try tok.tokenize();
    defer tokens.deinit(allocator);

    // Create dialect registry
    var registry = try mlir_integration.DialectRegistry.init(allocator);
    defer registry.deinit();

    // Read
    var rdr = reader.Reader.initWithRegistry(allocator, tokens.items, &registry);
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

    // Generate IR
    var gen = try ir_generator.IRGenerator.init(allocator, registry.ctx);
    defer gen.deinit();

    _ = try gen.generate(nodes.items);

    std.debug.print("=== GENERATED MLIR ===\n", .{});
    gen.printModule();

    // Verify
    std.debug.print("\n=== VERIFICATION ===\n", .{});
    if (gen.verify()) {
        std.debug.print("Module verified successfully!\n", .{});
    } else {
        std.debug.print("Module FAILED verification!\n", .{});
    }
}
