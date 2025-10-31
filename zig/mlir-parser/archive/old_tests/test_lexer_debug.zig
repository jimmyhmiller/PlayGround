const std = @import("std");
const Lexer = @import("src/lexer.zig").Lexer;

pub fn main() !void {
    const source = "%0 = arith.constant 42 : i32";
    var lex = Lexer.init(source);
    
    while (true) {
        const token = lex.nextToken();
        std.debug.print("{any}\n", .{token});
        if (token.type == .eof) break;
    }
}
