const std = @import("std");
const reader_mod = @import("reader.zig");
const Compiler = @import("compiler.zig").Compiler;

pub fn main() !void {
    std.debug.print("=== Lisp Reader Demo ===\n", .{});

    var arena = std.heap.ArenaAllocator.init(std.heap.page_allocator);
    defer arena.deinit();
    var allocator = arena.allocator();

    var compiler = Compiler.init(&allocator);

    // Demo: Parse the complex example
    const clojure_code = "(def f (: (-> [Int] Int)) (fn [x] (+ x 1)))";
    std.debug.print("Parsing: {s}\n", .{clojure_code});

    compiler.compileString(clojure_code) catch |err| {
        std.debug.print("Error parsing: {}\n", .{err});
        return;
    };
}
