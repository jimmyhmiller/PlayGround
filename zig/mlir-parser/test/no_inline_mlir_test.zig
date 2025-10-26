//! No Inline MLIR Test
//! Enforces that all MLIR examples must be in external .mlir files,
//! not inline strings in test files.

const std = @import("std");
const testing = std.testing;

/// Check if a line contains an inline MLIR example
fn containsInlineMLIR(line: []const u8) bool {
    // Trim leading whitespace
    const trimmed = std.mem.trimLeft(u8, line, " \t");

    // Skip empty lines and comments
    if (trimmed.len == 0 or std.mem.startsWith(u8, trimmed, "//")) {
        return false;
    }

    // Pattern 1: String assignment with MLIR operation (e.g., const source = "%0 = ")
    if (std.mem.indexOf(u8, line, "const source =") != null or
        std.mem.indexOf(u8, line, "const source=") != null)
    {
        // Check if it contains MLIR-like patterns
        if (std.mem.indexOf(u8, line, "%") != null and
            (std.mem.indexOf(u8, line, " = \"") != null or
            std.mem.indexOf(u8, line, "= \"") != null))
        {
            return true;
        }
    }

    // Pattern 2: Multi-line string with MLIR operation (e.g., \\%0 = "arith.constant")
    if (std.mem.indexOf(u8, trimmed, "\\\\%") != null and
        std.mem.indexOf(u8, trimmed, " = \"") != null)
    {
        // Check for dialect.operation pattern
        if (std.mem.indexOf(u8, trimmed, ".") != null) {
            return true;
        }
    }

    // Pattern 3: String literal with MLIR SSA value (e.g., "%0 = \"test.op\"")
    if (std.mem.startsWith(u8, trimmed, "\"%") and
        std.mem.indexOf(u8, trimmed, " = \"") != null)
    {
        return true;
    }

    // Pattern 4: Quoted MLIR operation name in test (e.g., "arith.constant", "test.op")
    // Only flag if it appears in a string context with operation-like patterns
    if (std.mem.indexOf(u8, line, "\"") != null) {
        // Check for dialect.operation in quotes
        var in_quotes = false;
        var quote_start: usize = 0;
        for (line, 0..) |char, i| {
            if (char == '"') {
                if (in_quotes) {
                    // End of quoted section
                    const quoted = line[quote_start..i];
                    // Check if it looks like an MLIR operation: dialect.name
                    if (std.mem.indexOf(u8, quoted, ".") != null and
                        (std.mem.indexOf(u8, quoted, "%") != null or
                        std.mem.indexOf(u8, quoted, " = ") != null or
                        std.mem.indexOf(u8, quoted, "() ") != null))
                    {
                        return true;
                    }
                    in_quotes = false;
                } else {
                    in_quotes = true;
                    quote_start = i + 1;
                }
            }
        }
    }

    return false;
}

test "no inline MLIR in test files" {
    const allocator = testing.allocator;

    // Files to check
    const test_files = [_][]const u8{
        "test/roundtrip_test.zig",
        "test/basic_test.zig",
        "test/integration_test.zig",
        "test/unsupported_features_test.zig",
    };

    var violations: std.ArrayList(u8) = .empty;
    defer violations.deinit(allocator);

    var has_violations = false;

    for (test_files) |test_file| {
        // Open file
        const file = std.fs.cwd().openFile(test_file, .{}) catch |err| {
            // If file doesn't exist, skip it
            if (err == error.FileNotFound) continue;
            return err;
        };
        defer file.close();

        // Read file content
        const content = try file.readToEndAlloc(allocator, 10 * 1024 * 1024); // 10MB max
        defer allocator.free(content);

        // Check each line
        var line_iterator = std.mem.splitScalar(u8, content, '\n');
        var line_num: usize = 1;
        while (line_iterator.next()) |line| : (line_num += 1) {
            if (containsInlineMLIR(line)) {
                has_violations = true;
                try violations.writer(allocator).print("{s}:{}: {s}\n", .{ test_file, line_num, line });
            }
        }
    }

    if (has_violations) {
        std.debug.print("\n=== Inline MLIR Violations ===\n", .{});
        std.debug.print("Found inline MLIR examples in test files.\n", .{});
        std.debug.print("All MLIR examples must be in test_data/examples/*.mlir files.\n\n", .{});
        std.debug.print("{s}\n", .{violations.items});
        std.debug.print("To fix: Extract these examples to .mlir files and use loadTestFile() instead.\n", .{});
        return error.InlineMLIRDetected;
    }

    std.debug.print("✓ No inline MLIR detected in test files\n", .{});
}
