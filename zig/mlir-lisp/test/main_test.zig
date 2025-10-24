const std = @import("std");
const main = @import("main");

test "can import main and access main function" {
    // Verify that we can call the main function from the main module
    try main.main();
    try std.testing.expect(true);
}
