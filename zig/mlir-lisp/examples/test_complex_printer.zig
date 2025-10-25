const std = @import("std");
const mlir_lisp = @import("mlir_lisp");

pub fn main() !void {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();

    const input =
        \\(mlir
        \\  (operation
        \\    (name func.func)
        \\    (attributes {
        \\      :sym (#sym @add)
        \\      :type (!function (inputs !i32 !i32) (results !i32))
        \\    })
        \\    (regions
        \\      (region
        \\        (block [^entry]
        \\          (arguments [ [%x !i32] [%y !i32] ])
        \\          (operation
        \\            (name arith.addi)
        \\            (result-bindings [%sum])
        \\            (operands %x %y)
        \\            (result-types !i32))
        \\          (operation
        \\            (name func.return)
        \\            (operands %sum)))))))
    ;

    std.debug.print("=== Input (Complex Function) ===\n{s}\n\n", .{input});

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

    std.debug.print("Parsed module with {} operation(s)\n", .{module.operations.len});

    const func = module.operations[0];
    std.debug.print("  Function: {s}\n", .{func.name});
    std.debug.print("  Attributes: {}\n", .{func.attributes.len});
    std.debug.print("  Regions: {}\n", .{func.regions.len});
    std.debug.print("  Blocks in region: {}\n", .{func.regions[0].blocks.len});
    std.debug.print("  Operations in block: {}\n\n", .{func.regions[0].blocks[0].operations.len});

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

    const func2 = module2.operations[0];

    std.debug.print("✓ Round-trip successful!\n", .{});
    std.debug.print("  Function: {s}\n", .{func2.name});
    std.debug.print("  Regions: {}\n", .{func2.regions.len});
    std.debug.print("  Block arguments: {}\n", .{func2.regions[0].blocks[0].arguments.len});
    std.debug.print("  Block operations: {}\n", .{func2.regions[0].blocks[0].operations.len});
    std.debug.print("  First op: {s}\n", .{func2.regions[0].blocks[0].operations[0].name});
    std.debug.print("  Second op: {s}\n", .{func2.regions[0].blocks[0].operations[1].name});
}
