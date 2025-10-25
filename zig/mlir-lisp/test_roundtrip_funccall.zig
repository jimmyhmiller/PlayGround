const std = @import("std");
const mlir_lisp = @import("mlir_lisp");
const Tokenizer = mlir_lisp.Tokenizer;
const Reader = mlir_lisp.Reader;
const Parser = mlir_lisp.Parser;
const Printer = mlir_lisp.Printer;

pub fn main() !void {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();

    const input = "(operation (name func.call) (result-bindings [%result]) (result-types !i64) (attributes { :callee @test }))";

    std.debug.print("=== Original input ===\n{s}\n\n", .{input});

    // Parse
    var tok = Tokenizer.init(allocator, input);
    var r = try Reader.init(allocator, &tok);
    var value = try r.read();
    defer {
        value.deinit(allocator);
        allocator.destroy(value);
    }

    std.debug.print("=== After reader ===\n", .{});
    std.debug.print("Value type: {s}\n\n", .{@tagName(value.type)});

    var p = Parser.init(allocator);
    var operation = try p.parseOperation(value);
    defer operation.deinit(allocator);

    std.debug.print("=== After parser ===\n", .{});
    std.debug.print("Operation name: {s}\n", .{operation.name});
    std.debug.print("Attributes count: {}\n\n", .{operation.attributes.len});

    // Print
    var pr = Printer.init(allocator);
    defer pr.deinit();
    try pr.printOperation(&operation);

    const output = pr.getOutput();
    std.debug.print("=== Round-tripped output ===\n{s}\n\n", .{output});

    // Parse again to verify round-trip
    var tok2 = Tokenizer.init(allocator, output);
    var r2 = try Reader.init(allocator, &tok2);
    var value2 = try r2.read();
    defer {
        value2.deinit(allocator);
        allocator.destroy(value2);
    }

    var p2 = Parser.init(allocator);
    var operation2 = try p2.parseOperation(value2);
    defer operation2.deinit(allocator);

    std.debug.print("=== Second parse successful ===\n", .{});
    std.debug.print("Operation name: {s}\n", .{operation2.name});
    std.debug.print("Attributes count: {}\n\n", .{operation2.attributes.len});

    if (!std.mem.eql(u8, operation.name, operation2.name)) {
        std.debug.print("ERROR: Operation names don't match!\n", .{});
        return error.RoundTripFailed;
    }

    std.debug.print("✓ Round-trip successful!\n", .{});
}
