const std = @import("std");
const reader = @import("src/reader.zig");
const tokenizer = @import("src/tokenizer.zig");

pub fn main() !void {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();

    const args = try std.process.argsAlloc(allocator);
    defer std.process.argsFree(allocator, args);

    if (args.len < 2) {
        std.debug.print("Usage: test_reader_compare <file.lisp>\n", .{});
        return;
    }

    const file_path = args[1];
    const source = try std.fs.cwd().readFileAlloc(allocator, file_path, 1024 * 1024);
    defer allocator.free(source);

    var tok = tokenizer.Tokenizer.init(allocator, source);
    var r = try reader.Reader.init(allocator, &tok);
    var values = try r.readAll();
    defer values.deinit();

    // Navigate to the block in func.func -> regions -> region -> block
    if (values.len() < 1) {
        std.debug.print("No values\n", .{});
        return;
    }

    const mlir_value = values.at(0);
    if (mlir_value.type != .list) {
        std.debug.print("Not a list\n", .{});
        return;
    }

    const mlir_list = mlir_value.data.list;
    if (mlir_list.len() < 2) {
        std.debug.print("mlir list too short\n", .{});
        return;
    }

    const func_op = mlir_list.at(1); // (operation ...)
    if (func_op.type != .list) {
        std.debug.print("func_op not a list\n", .{});
        return;
    }

    const func_list = func_op.data.list;

    // Find regions section
    var regions_value: ?*reader.Value = null;
    var i: usize = 0;
    while (i < func_list.len()) : (i += 1) {
        const item = func_list.at(i);
        if (item.type == .list) {
            const sublist = item.data.list;
            if (sublist.len() > 0) {
                const first = sublist.at(0);
                if (first.type == .identifier and std.mem.eql(u8, first.data.atom, "regions")) {
                    regions_value = item;
                    break;
                }
            }
        }
    }

    if (regions_value == null) {
        std.debug.print("No regions found\n", .{});
        return;
    }

    const regions_list = regions_value.?.data.list;
    if (regions_list.len() < 2) {
        std.debug.print("regions list too short\n", .{});
        return;
    }

    const region_value = regions_list.at(1); // First region
    if (region_value.type != .list) {
        std.debug.print("region not a list\n", .{});
        return;
    }

    const region_list = region_value.data.list;
    if (region_list.len() < 2) {
        std.debug.print("region list too short\n", .{});
        return;
    }

    const block_value = region_list.at(1); // (block ...)
    if (block_value.type != .list) {
        std.debug.print("block not a list\n", .{});
        return;
    }

    const block_list = block_value.data.list;

    std.debug.print("\n=== BLOCK CONTENTS ({d} elements) ===\n", .{block_list.len()});

    i = 0;
    while (i < block_list.len()) : (i += 1) {
        const elem = block_list.at(i);
        if (elem.type == .list) {
            const sublist = elem.data.list;
            if (sublist.len() > 0) {
                const first = sublist.at(0);
                if (first.type == .identifier) {
                    std.debug.print("  [{d}] ({s} ...)\n", .{i, first.data.atom});
                } else {
                    std.debug.print("  [{d}] (list, first elem: {s})\n", .{i, @tagName(first.type)});
                }
            } else {
                std.debug.print("  [{d}] (empty list)\n", .{i});
            }
        } else {
            std.debug.print("  [{d}] {s}\n", .{i, @tagName(elem.type)});
        }
    }
}
