const std = @import("std");
const mlir = @import("src/root.zig");

pub fn main() !void {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();

    const stdout = std.io.getStdOut().writer();

    // Test cases to verify roundtrip
    const test_cases = [_][]const u8{
        "%0 = \"arith.constant\"() <{value = 42 : i32}> : () -> i32",
        "%0 = \"arith.constant\"() <{value = 1 : i1}> : () -> i1",
        "%0:2 = \"test.op\"() : () -> (i32, i32)",
        "%1 = \"test.op\"(%0#1) : (i32) -> i32",
        "%0 = \"test.op\"() : () -> tensor<4x8xf32>",
        "%0 = \"test.op\"() : () -> vector<4x8xf32>",
        "%0 = \"test.op\"() : () -> memref<10x20xf32>",
        "%0 = \"test.op\"() : () -> complex<f64>",
        "%0 = \"test.op\"() : () -> tuple<i32, f32>",
        "%0 = \"test.op\"() : () -> !llvm.ptr",
    };

    var passed: usize = 0;
    var failed: usize = 0;

    for (test_cases, 0..) |test_case, i| {
        try stdout.print("\n[Test {d}] Testing: {s}\n", .{ i + 1, test_case });

        // Parse -> Print -> Parse -> Print
        var module1 = mlir.parse(allocator, test_case) catch |err| {
            try stdout.print("  ❌ FAILED: Parse 1 error: {}\n", .{err});
            failed += 1;
            continue;
        };
        defer module1.deinit();

        const printed1 = mlir.print(allocator, module1) catch |err| {
            try stdout.print("  ❌ FAILED: Print 1 error: {}\n", .{err});
            failed += 1;
            continue;
        };
        defer allocator.free(printed1);

        try stdout.print("  Printed 1: {s}\n", .{printed1});

        var module2 = mlir.parse(allocator, printed1) catch |err| {
            try stdout.print("  ❌ FAILED: Parse 2 error: {}\n", .{err});
            failed += 1;
            continue;
        };
        defer module2.deinit();

        const printed2 = mlir.print(allocator, module2) catch |err| {
            try stdout.print("  ❌ FAILED: Print 2 error: {}\n", .{err});
            failed += 1;
            continue;
        };
        defer allocator.free(printed2);

        try stdout.print("  Printed 2: {s}\n", .{printed2});

        // Compare
        if (std.mem.eql(u8, printed1, printed2)) {
            try stdout.print("  ✅ PASSED\n", .{});
            passed += 1;
        } else {
            try stdout.print("  ❌ FAILED: Outputs differ\n", .{});
            try stdout.print("    First:  {s}\n", .{printed1});
            try stdout.print("    Second: {s}\n", .{printed2});
            failed += 1;
        }
    }

    try stdout.print("\n\n=== SUMMARY ===\n", .{});
    try stdout.print("Passed: {d}/{d}\n", .{ passed, test_cases.len });
    try stdout.print("Failed: {d}/{d}\n", .{ failed, test_cases.len });

    if (failed > 0) {
        std.process.exit(1);
    }
}
