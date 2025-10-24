const std = @import("std");
const compiler = @import("compiler");

test "can import compiler module" {
    // Verify that we can access the compiler module
    try std.testing.expect(true);
}

// This test file demonstrates the module structure
// The actual comprehensive tests remain in src/ for now
// This follows the pattern from mlir-lisp where test/main_test.zig
// imports the main module and can test it in isolation
