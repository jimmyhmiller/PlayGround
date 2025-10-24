//! By convention, root.zig is the root source file when making a library.
const std = @import("std");

// Export the tokenizer module
pub const Tokenizer = @import("tokenizer.zig").Tokenizer;
pub const Token = @import("tokenizer.zig").Token;
pub const TokenType = @import("tokenizer.zig").TokenType;

pub fn bufferedPrint() !void {
    // Just print to debug output for simplicity
    std.debug.print("Run `zig build test` to run the tests.\n", .{});
}

pub fn add(a: i32, b: i32) i32 {
    return a + b;
}

test "basic add functionality" {
    try std.testing.expect(add(3, 7) == 10);
}
