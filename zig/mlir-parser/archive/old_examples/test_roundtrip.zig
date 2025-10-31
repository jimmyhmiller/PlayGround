//! Simple demonstration of roundtrip parsing and printing
//! This shows how to parse MLIR code, print it back out, and parse again

const std = @import("std");
const mlir = @import("mlir-parser");

pub fn main() !void {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();

    // Example MLIR source (generic format)
    const source =
        \\%0 = "arith.constant"() <{value = 42 : i32}> : () -> i32
        \\%1 = "arith.constant"() <{value = 13 : i32}> : () -> i32
        \\%2 = "arith.addi"(%0, %1) : (i32, i32) -> i32
    ;

    std.debug.print("=== Original Source ===\n{s}\n\n", .{source});

    // First parse
    std.debug.print("=== Parsing (first time) ===\n", .{});
    var module1 = try mlir.parse(allocator, source);
    defer module1.deinit();
    std.debug.print("Parsed {} operations\n\n", .{module1.operations.len});

    // Print back to string
    std.debug.print("=== Printing (first time) ===\n", .{});
    const printed1 = try mlir.print(allocator, module1);
    defer allocator.free(printed1);
    std.debug.print("{s}\n\n", .{printed1});

    // Second parse
    std.debug.print("=== Parsing (second time) ===\n", .{});
    var module2 = try mlir.parse(allocator, printed1);
    defer module2.deinit();
    std.debug.print("Parsed {} operations\n\n", .{module2.operations.len});

    // Print again
    std.debug.print("=== Printing (second time) ===\n", .{});
    const printed2 = try mlir.print(allocator, module2);
    defer allocator.free(printed2);
    std.debug.print("{s}\n\n", .{printed2});

    // Verify roundtrip stability
    std.debug.print("=== Roundtrip Verification ===\n", .{});
    if (std.mem.eql(u8, printed1, printed2)) {
        std.debug.print("✓ Roundtrip successful! Output is stable.\n", .{});
    } else {
        std.debug.print("✗ Roundtrip failed! Outputs differ.\n", .{});
        std.debug.print("First output:  {s}\n", .{printed1});
        std.debug.print("Second output: {s}\n", .{printed2});
        return error.RoundtripFailed;
    }
}
