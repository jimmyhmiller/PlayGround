const std = @import("std");
const mlir_lisp = @import("mlir_lisp");

pub fn main() !void {
    // Prints to stderr, ignoring potential errors.
    std.debug.print("All your {s} are belong to us.\n", .{"codebase"});
    try mlir_lisp.bufferedPrint();
}
