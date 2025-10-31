const std = @import("std");
const Tokenizer = @import("src/tokenizer.zig").Tokenizer;

pub fn main() !void {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();

    const source = "(#int 42)";

    var tok = Tokenizer.init(allocator, source);

    while (true) {
        const token = try tok.next();
        std.debug.print("Token: {s} = '{s}'\n", .{ @tagName(token.type), token.lexeme });
        if (token.type == .eof) break;
    }
}
