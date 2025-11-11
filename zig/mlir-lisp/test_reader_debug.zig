const std = @import("std");
const reader = @import("src/reader.zig");
const tokenizer = @import("src/tokenizer.zig");

pub fn main() !void {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();

    const file_path = "examples/test_reader_order.lisp";
    const source = try std.fs.cwd().readFileAlloc(allocator, file_path, 1024 * 1024);
    defer allocator.free(source);

    std.debug.print("Source file:\n{s}\n\n", .{source});

    var tok = tokenizer.Tokenizer.init(allocator, source);

    var r = try reader.Reader.init(allocator, &tok);

    var values = try r.readAll();
    defer {
        var idx: usize = 0;
        while (idx < values.len()) : (idx += 1) {
            var v = values.at(idx);
            v.deinit(allocator);
        }
        values.deinit();
    }

    std.debug.print("Reader output ({d} top-level values):\n", .{values.len()});
    var i: usize = 0;
    while (i < values.len()) : (i += 1) {
        const value = values.at(i);
        std.debug.print("\nValue {d}:\n", .{i});
        printValue(value, 0);
    }
}

fn printValue(value: *const reader.Value, indent: usize) void {
    var i: usize = 0;
    while (i < indent) : (i += 1) {
        std.debug.print("  ", .{});
    }

    switch (value.type) {
        .list => {
            const list = value.data.list;
            std.debug.print("(list with {d} elements:\n", .{list.len()});
            var idx: usize = 0;
            while (idx < list.len()) : (idx += 1) {
                var j: usize = 0;
                while (j < indent + 1) : (j += 1) {
                    std.debug.print("  ", .{});
                }
                std.debug.print("[{d}] ", .{idx});
                printValue(list.at(idx), indent + 1);
            }
            i = 0;
            while (i < indent) : (i += 1) {
                std.debug.print("  ", .{});
            }
            std.debug.print(")\n", .{});
        },
        .identifier => {
            std.debug.print("identifier: {s}\n", .{value.data.atom});
        },
        .value_id => {
            std.debug.print("value_id: {s}\n", .{value.data.atom});
        },
        else => {
            std.debug.print("{s}\n", .{@tagName(value.type)});
        },
    }
}
