// This file imports all module tests to run them together
const std = @import("std");

// Import all module test files
const frontend_tests = @import("frontend_tests.zig");
const backend_tests = @import("backend_tests.zig");
const collections_tests = @import("collections/tests.zig");
const showcase_test = @import("showcase_test.zig");
const type_checker_comprehensive_tests = @import("type_checker_comprehensive_tests.zig");
const struct_field_access_tests = @import("struct_field_access_tests.zig");
const array_tests = @import("array_tests.zig");
const macro_comprehensive_tests = @import("macro_comprehensive_tests.zig");
const c_api_tests = @import("c_api_tests.zig");
const integration_tests = @import("integration_tests.zig");
const self_referential_struct_tests = @import("self_referential_struct_tests.zig");
const macro_let_tests = @import("macro_let_tests.zig");
const bitwise_tests = @import("bitwise_tests.zig");

// Import all modules to include their inline tests
const value = @import("value.zig");
const macro_expander = @import("macro_expander.zig");
const lexer = @import("lexer.zig");
const parser = @import("parser.zig");
const reader = @import("reader.zig");
const vector = @import("collections/vector.zig");
const linked_list = @import("collections/linked_list.zig");
const map = @import("collections/map.zig");
const type_checker = @import("type_checker.zig");
const simple_c_compiler = @import("simple_c_compiler.zig");
const c_api = @import("c_api.zig");

// Force test inclusion by referencing test blocks
comptime {
    std.testing.refAllDecls(@This());
    std.testing.refAllDecls(frontend_tests);
    std.testing.refAllDecls(backend_tests);
    std.testing.refAllDecls(collections_tests);
    std.testing.refAllDecls(showcase_test);
    std.testing.refAllDecls(type_checker_comprehensive_tests);
    std.testing.refAllDecls(struct_field_access_tests);
    std.testing.refAllDecls(array_tests);
    std.testing.refAllDecls(macro_comprehensive_tests);
    std.testing.refAllDecls(c_api_tests);
    std.testing.refAllDecls(integration_tests);
    std.testing.refAllDecls(self_referential_struct_tests);
    std.testing.refAllDecls(macro_let_tests);
    std.testing.refAllDecls(bitwise_tests);
    std.testing.refAllDecls(value);
    std.testing.refAllDecls(lexer);
    std.testing.refAllDecls(parser);
    std.testing.refAllDecls(reader);
    std.testing.refAllDecls(vector);
    std.testing.refAllDecls(linked_list);
    std.testing.refAllDecls(map);
    std.testing.refAllDecls(type_checker);
    std.testing.refAllDecls(simple_c_compiler);
    std.testing.refAllDecls(macro_expander);
    std.testing.refAllDecls(c_api);
}

test "test runner working" {
    try std.testing.expect(true);
}
