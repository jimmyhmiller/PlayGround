const std = @import("std");
const mlir_lisp = @import("mlir_lisp");

pub fn main() !void {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();

    const input =
        \\(mlir
        \\  (operation
        \\    (name arith.constant)
        \\    (result-bindings [%c0])
        \\    (result-types i32)
        \\    (attributes { :value (#int 42) })))
    ;

    std.debug.print("=== Input ===\n{s}\n\n", .{input});

    // Parse
    var tok = mlir_lisp.Tokenizer.init(allocator, input);
    var r = try mlir_lisp.Reader.init(allocator, &tok);
    var value = try r.read();
    defer {
        value.deinit(allocator);
        allocator.destroy(value);
    }

    var p = mlir_lisp.Parser.init(allocator);
    var module = try p.parseModule(value);
    defer module.deinit();

    // Print
    var pr = mlir_lisp.Printer.init(allocator);
    defer pr.deinit();
    try pr.printModule(&module);

    const output = pr.getOutput();

    std.debug.print("=== Output ===\n{s}\n\n", .{output});

    // Parse the output again to verify round-trip
    var tok2 = mlir_lisp.Tokenizer.init(allocator, output);
    var r2 = try mlir_lisp.Reader.init(allocator, &tok2);
    var value2 = try r2.read();
    defer {
        value2.deinit(allocator);
        allocator.destroy(value2);
    }

    var p2 = mlir_lisp.Parser.init(allocator);
    var module2 = try p2.parseModule(value2);
    defer module2.deinit();

    std.debug.print("✓ Round-trip successful!\n", .{});
    std.debug.print("  Operation name: {s}\n", .{module2.operations[0].name});
    std.debug.print("  Result binding: {s}\n", .{module2.operations[0].result_bindings[0]});
}
