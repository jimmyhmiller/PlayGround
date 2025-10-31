const std = @import("std");
const mlir = @import("mlir_parser");

pub fn main() !void {
    var arena = std.heap.ArenaAllocator.init(std.heap.page_allocator);
    defer arena.deinit();
    const allocator = arena.allocator();

    // Use proper Zig 0.15.1 stdout API
    var stdout_buffer: [4096]u8 = undefined;
    var stdout_writer = std.fs.File.stdout().writer(&stdout_buffer);
    const stdout = &stdout_writer.interface;

    // Test the failing cases to see what they print
    const tests = [_]struct { name: []const u8, source: []const u8 }{
        .{
            .name = "Dynamic tensor",
            .source = "%0 = \"test.op\"() : () -> tensor<?x8xf32>",
        },
        .{
            .name = "Scalable vector",
            .source = "%0 = \"test.op\"() : () -> vector<[4]x8xf32>",
        },
        .{
            .name = "Function type nested",
            .source = "%0 = \"test.op\"() : () -> (i32, i32, f32) -> i64",
        },
        .{
            .name = "Dialect type body",
            .source = "%0 = \"test.op\"() : () -> !llvm<ptr<i32>>",
        },
        .{
            .name = "Successor with args",
            .source = "%0 = \"cf.br\"(%arg0) [^bb1(%arg0 : i32)] : (i32) -> ()",
        },
    };

    for (tests) |t| {
        try stdout.print("\n=== {s} ===\n", .{t.name});
        try stdout.print("Input:  {s}\n", .{t.source});

        var module = mlir.parse(allocator, t.source) catch |err| {
            try stdout.print("ERROR: Failed to parse: {}\n", .{err});
            continue;
        };
        defer module.deinit();

        const printed = mlir.print(allocator, module) catch |err| {
            try stdout.print("ERROR: Failed to print: {}\n", .{err});
            continue;
        };
        defer allocator.free(printed);

        try stdout.print("Output: {s}\n", .{printed});

        // Try to parse again
        var module2 = mlir.parse(allocator, printed) catch |err| {
            try stdout.print("ERROR: Failed to parse output: {}\n", .{err});
            continue;
        };
        defer module2.deinit();

        try stdout.print("SUCCESS: Roundtrip worked!\n", .{});
    }

    try stdout.flush();
}
