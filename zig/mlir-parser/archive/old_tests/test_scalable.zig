const std = @import("std");
const lexer_mod = @import("src/lexer.zig");

pub fn main() !void {
    const source = "vector<[4]x8xf32>";
    var lexer = lexer_mod.Lexer{ .source = source };
    
    while (true) {
        const token = lexer.nextToken();
        std.debug.print("{s}: '{s}'\n", .{@tagName(token.type), token.lexeme});
        if (token.type == .eof) break;
    }
}
