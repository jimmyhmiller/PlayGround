const std = @import("std");
const tokenizer = @import("src/tokenizer.zig");

pub fn main() !void {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();

    const source = "(...)";
    std.debug.print("Source: {s}\n\n", .{source});

    var tok = tokenizer.Tokenizer.init(allocator, source);

    std.debug.print("Tokens:\n", .{});
    while (true) {
        const token = try tok.next();
        std.debug.print("{s}: '{s}'\n", .{ @tagName(token.type), token.lexeme });
        if (token.type == .eof) break;
    }
}
