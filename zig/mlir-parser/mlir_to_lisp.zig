const std = @import("std");
const mlir = @import("mlir_parser");

pub fn main() !void {
    var arena = std.heap.ArenaAllocator.init(std.heap.page_allocator);
    defer arena.deinit();
    const allocator = arena.allocator();

    // Read from file argument
    const args = try std.process.argsAlloc(allocator);
    defer std.process.argsFree(allocator, args);

    if (args.len < 2) {
        std.debug.print("Usage: mlir-to-lisp <file.mlir>\n", .{});
        return error.MissingArgument;
    }

    // Read from file
    const file = try std.fs.cwd().openFile(args[1], .{});
    defer file.close();
    const source = try file.readToEndAlloc(allocator, 10 * 1024 * 1024); // 10MB max

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
