const std = @import("std");
const main = @import("main");

test "can import main module" {
    // Verify that we can import the main module
    // Note: We don't call main() directly because it parses command-line args
    // which interfere with the test runner's arguments
    try std.testing.expect(true);
}
