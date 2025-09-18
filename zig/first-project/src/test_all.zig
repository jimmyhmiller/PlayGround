// This file imports all modules to run all tests together
const std = @import("std");

// Import all modules to include their tests
const value = @import("value.zig");
const lexer = @import("frontend/lexer.zig");
const parser = @import("frontend/parser.zig");
const reader = @import("frontend/reader.zig");

// Import collections
const vector = @import("collections/vector.zig");
const linked_list = @import("collections/linked_list.zig");
const map = @import("collections/map.zig");

// Force test inclusion by referencing test blocks
comptime {
    std.testing.refAllDecls(@This());
    std.testing.refAllDecls(value);
    std.testing.refAllDecls(lexer);
    std.testing.refAllDecls(parser);
    std.testing.refAllDecls(reader);
    std.testing.refAllDecls(vector);
    std.testing.refAllDecls(linked_list);
    std.testing.refAllDecls(map);
}

test "test runner working" {
    try std.testing.expect(true);
}