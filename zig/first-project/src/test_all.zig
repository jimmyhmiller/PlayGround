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

// Import backend
const type_checker_test = @import("backend/type_checker_test.zig");
const comprehensive_type_tests = @import("backend/comprehensive_type_tests.zig");
const test_full_annotation = @import("backend/test_full_annotation.zig");
const test_multiple_defs = @import("backend/test_multiple_defs.zig");
const test_multiple_forms = @import("backend/test_multiple_forms.zig");
const test_enhanced_arithmetic = @import("backend/test_enhanced_arithmetic.zig");
const test_forward_refs = @import("backend/test_forward_refs.zig");

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
    std.testing.refAllDecls(type_checker_test);
    std.testing.refAllDecls(comprehensive_type_tests);
    std.testing.refAllDecls(test_full_annotation);
    std.testing.refAllDecls(test_multiple_defs);
    std.testing.refAllDecls(test_multiple_forms);
    std.testing.refAllDecls(test_enhanced_arithmetic);
    std.testing.refAllDecls(test_forward_refs);
}

test "test runner working" {
    try std.testing.expect(true);
}