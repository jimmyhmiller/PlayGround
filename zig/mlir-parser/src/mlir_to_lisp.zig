const std = @import("std");
const mlir = @import("mlir_parser");

fn printUsage() !void {
    var stdout_buffer: [4096]u8 = undefined;
    var stdout_writer = std.fs.File.stdout().writer(&stdout_buffer);
    const stdout = &stdout_writer.interface;

    try stdout.writeAll(
        \\mlir-to-lisp - Convert MLIR to Lisp S-expressions
        \\
        \\USAGE:
        \\    mlir-to-lisp [OPTIONS] <file.mlir>
        \\
        \\ARGS:
        \\    <file.mlir>    Input MLIR file to convert
        \\
        \\OPTIONS:
        \\    -h, --help     Print this help message
        \\
        \\DESCRIPTION:
        \\    Converts MLIR (Multi-Level Intermediate Representation) code to
        \\    Lisp S-expression format. The input file will be automatically
        \\    converted to generic MLIR format using mlir-opt before parsing.
        \\
        \\EXAMPLES:
        \\    mlir-to-lisp example.mlir
        \\    mlir-to-lisp input.mlir > output.mlisp
        \\
        \\REQUIREMENTS:
        \\    mlir-opt must be installed and available in PATH
        \\
    );
    try stdout.flush();
}

pub fn main() !void {
    var arena = std.heap.ArenaAllocator.init(std.heap.page_allocator);
    defer arena.deinit();
    const allocator = arena.allocator();

    // Read from file argument
    const args = try std.process.argsAlloc(allocator);
    defer std.process.argsFree(allocator, args);

    if (args.len < 2) {
        try printUsage();
        return error.MissingArgument;
    }

    const input_file = args[1];

    // Check for --help or -h
    if (std.mem.eql(u8, input_file, "--help") or std.mem.eql(u8, input_file, "-h")) {
        try printUsage();
        return;
    }

    // First, convert to generic format using mlir-opt
    // This ensures we can parse any MLIR format (custom or generic)
    const result = try std.process.Child.run(.{
        .allocator = allocator,
        .argv = &[_][]const u8{
            "mlir-opt",
            "--mlir-print-op-generic",
            "--allow-unregistered-dialect",
            input_file,
        },
        .max_output_bytes = 100 * 1024 * 1024, // 100MB max
    });
    defer allocator.free(result.stdout);
    defer allocator.free(result.stderr);

    if (result.term.Exited != 0) {
        std.debug.print("Error running mlir-opt:\n{s}\n", .{result.stderr});
        return error.MlirOptFailed;
    }

    const source = result.stdout;

    // Parse MLIR
    var module = try mlir.parse(allocator, source);
    defer module.deinit();

    // Convert to Lisp
    const lisp_output = try mlir.printLisp(allocator, module);
    defer allocator.free(lisp_output);

    // Write to stdout with proper buffering
    var stdout_buffer: [4096]u8 = undefined;
    var stdout_writer = std.fs.File.stdout().writer(&stdout_buffer);
    const stdout = &stdout_writer.interface;

    try stdout.print("{s}\n", .{lisp_output});
    try stdout.flush();
}
