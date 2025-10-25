/// Test utilities for MLIR validation and testing
///
/// This module provides helper functions for testing MLIR code generation.
/// Tests can explicitly call these utilities when they want to validate
/// that the generated MLIR is correct.
const std = @import("std");
const mlir_lisp = @import("mlir_lisp");
const mlir = mlir_lisp.mlir;

/// Validates MLIR by writing to a temp file and running mlir-opt --verify-diagnostics
///
/// This function:
/// 1. Captures the MLIR module output to a string
/// 2. Writes it to /tmp/test.mlir
/// 3. Runs `mlir-opt --verify-diagnostics` to validate the MLIR
/// 4. Returns an error if validation fails
///
/// Example usage in a test:
/// ```zig
/// const test_utils = @import("test_utils.zig");
///
/// test "my mlir test" {
///     var mlir_module = try builder.buildModule(&parsed_module);
///     defer mlir_module.destroy();
///
///     // Explicitly validate the generated MLIR
///     try test_utils.validateMLIRWithOpt(allocator, mlir_module);
/// }
/// ```
pub fn validateMLIRWithOpt(allocator: std.mem.Allocator, mlir_module: mlir.Module) !void {
    const tmp_path = "/tmp/test.mlir";

    // Write MLIR to temp file
    {
        const file = try std.fs.createFileAbsolute(tmp_path, .{});
        defer file.close();

        // Capture module output to string
        var buffer = std.ArrayList(u8){};
        defer buffer.deinit(allocator);

        // Use mlir's print to string functionality
        const Context = struct {
            buffer: *std.ArrayList(u8),
            allocator: std.mem.Allocator,
        };
        var ctx = Context{
            .buffer = &buffer,
            .allocator = allocator,
        };

        const callback = struct {
            fn write(str: mlir.c.MlirStringRef, user_data: ?*anyopaque) callconv(.c) void {
                const context = @as(*Context, @ptrCast(@alignCast(user_data)));
                const slice = str.data[0..str.length];
                context.buffer.appendSlice(context.allocator, slice) catch unreachable;
            }
        }.write;

        const op = mlir.c.mlirModuleGetOperation(mlir_module.module);
        mlir.c.mlirOperationPrint(op, &callback, @ptrCast(&ctx));

        try file.writeAll(buffer.items);
    }

    // Run mlir-opt --verify-diagnostics
    const result = try std.process.Child.run(.{
        .allocator = allocator,
        .argv = &[_][]const u8{
            "mlir-opt",
            tmp_path,
            "--verify-diagnostics",
        },
    });
    defer allocator.free(result.stdout);
    defer allocator.free(result.stderr);

    if (result.term != .Exited or result.term.Exited != 0) {
        std.debug.print("\n=== mlir-opt validation FAILED ===\n", .{});
        std.debug.print("stdout: {s}\n", .{result.stdout});
        std.debug.print("stderr: {s}\n", .{result.stderr});
        return error.MLIRValidationFailed;
    }

    std.debug.print("✓ MLIR validation passed\n", .{});
}
