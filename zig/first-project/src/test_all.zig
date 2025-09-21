// This file imports all module tests to run them together
const std = @import("std");

// Import all module test files
const frontend_tests = @import("frontend_tests.zig");
const backend_tests = @import("backend_tests.zig");
const collections_tests = @import("collections/tests.zig");
const showcase_test = @import("showcase_test.zig");

// Import all modules to include their inline tests
const value = @import("value.zig");
const lexer = @import("lexer.zig");
const parser = @import("parser.zig");
const reader = @import("reader.zig");
const vector = @import("collections/vector.zig");
const linked_list = @import("collections/linked_list.zig");
const map = @import("collections/map.zig");
const type_checker = @import("type_checker.zig");
const compiler = @import("compiler.zig");

// Force test inclusion by referencing test blocks
comptime {
    std.testing.refAllDecls(@This());
    std.testing.refAllDecls(frontend_tests);
    std.testing.refAllDecls(backend_tests);
    std.testing.refAllDecls(collections_tests);
    std.testing.refAllDecls(showcase_test);
    std.testing.refAllDecls(value);
    std.testing.refAllDecls(lexer);
    std.testing.refAllDecls(parser);
    std.testing.refAllDecls(reader);
    std.testing.refAllDecls(vector);
    std.testing.refAllDecls(linked_list);
    std.testing.refAllDecls(map);
    std.testing.refAllDecls(type_checker);
    std.testing.refAllDecls(compiler);
}

test "test runner working" {
    try std.testing.expect(true);
}
