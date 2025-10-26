//! Example Validation Test
//! Validates all MLIR example files in test_data/examples/ using mlir-opt

const std = @import("std");
const testing = std.testing;

test "all example files are valid MLIR" {
    const allocator = testing.allocator;

    // Open examples directory
    var examples_dir = try std.fs.cwd().openDir("test_data/examples", .{ .iterate = true });
    defer examples_dir.close();

    var iterator = examples_dir.iterate();

    var total: usize = 0;
    var valid: usize = 0;
    var invalid: usize = 0;
    var errors: std.ArrayList(u8) = .empty;
    defer errors.deinit(allocator);

    // Iterate through all .mlir files
    while (try iterator.next()) |entry| {
        if (entry.kind != .file) continue;
        if (!std.mem.endsWith(u8, entry.name, ".mlir")) continue;

        total += 1;

        // Build full path
        const path = try std.fmt.allocPrint(allocator, "test_data/examples/{s}", .{entry.name});
        defer allocator.free(path);

        // Run mlir-opt on the file
        const result = std.process.Child.run(.{
            .allocator = allocator,
            .argv = &[_][]const u8{
                "mlir-opt",
                "--allow-unregistered-dialect",
                "--mlir-print-op-generic",
                path,
            },
        }) catch |err| {
            try errors.writer(allocator).print("Failed to run mlir-opt on {s}: {}\n", .{ entry.name, err });
            invalid += 1;
            continue;
        };
        defer allocator.free(result.stdout);
        defer allocator.free(result.stderr);

        // Check exit code
        switch (result.term) {
            .Exited => |code| {
                if (code == 0) {
                    valid += 1;
                } else {
                    invalid += 1;
                    try errors.writer(allocator).print("✗ {s}:\n{s}\n", .{ entry.name, result.stderr });
                }
            },
            else => {
                invalid += 1;
                try errors.writer(allocator).print("✗ {s}: abnormal termination\n", .{entry.name});
            },
        }
    }

    // Report results
    if (invalid > 0) {
        std.debug.print("\n=== MLIR Validation Results ===\n", .{});
        std.debug.print("Total files: {}\n", .{total});
        std.debug.print("Valid: {}\n", .{valid});
        std.debug.print("Invalid: {}\n", .{invalid});
        std.debug.print("\n=== Errors ===\n{s}\n", .{errors.items});
        return error.InvalidMLIRExamples;
    }

    // All files valid
    std.debug.print("✓ All {} MLIR example files are valid\n", .{total});
}
