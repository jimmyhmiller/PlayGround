const std = @import("std");
const main_module = @import("main");
const mlir_integration = main_module.mlir_integration;

pub fn main() !void {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();

    var registry = try mlir_integration.DialectRegistry.init(allocator);
    defer registry.deinit();

    // Load dialects
    const dialects = [_][]const u8{ "arith", "func", "cf", "scf", "memref", "vector" };

    var dialect_ops = std.StringArrayHashMap(std.ArrayList([]const u8)).init(allocator);
    defer {
        var it = dialect_ops.iterator();
        while (it.next()) |entry| {
            for (entry.value_ptr.items) |op| {
                allocator.free(op);
            }
            entry.value_ptr.deinit(allocator);
        }
        dialect_ops.deinit();
    }

    // Get operations for each dialect
    for (dialects) |dialect| {
        const ops = try registry.enumerateOperations(dialect);
        try dialect_ops.put(dialect, ops);
        std.debug.print("Loaded {s}: {} ops\n", .{ dialect, ops.items.len });
    }

    // Find overlaps
    std.debug.print("\nLooking for overlapping operation names...\n\n", .{});

    for (dialects, 0..) |d1, i| {
        for (dialects[i+1..]) |d2| {
            const ops1 = dialect_ops.get(d1).?;
            const ops2 = dialect_ops.get(d2).?;

            for (ops1.items) |op1| {
                // Extract just the operation name (after the dot)
                const dot_pos = std.mem.indexOf(u8, op1, ".") orelse continue;
                const op1_name = op1[dot_pos+1..];

                for (ops2.items) |op2| {
                    const dot_pos2 = std.mem.indexOf(u8, op2, ".") orelse continue;
                    const op2_name = op2[dot_pos2+1..];

                    if (std.mem.eql(u8, op1_name, op2_name)) {
                        std.debug.print("OVERLAP: '{s}' in both {s} and {s}\n", .{ op1_name, d1, d2 });
                    }
                }
            }
        }
    }
}
