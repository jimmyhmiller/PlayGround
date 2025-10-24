const std = @import("std");
const fs = std.fs;
const testing = std.testing;

const simple_c_compiler = @import("simple_c_compiler.zig");
const SimpleCCompiler = simple_c_compiler.SimpleCCompiler;

/// Integration test result
const TestResult = struct {
    name: []const u8,
    passed: bool,
    message: []const u8,
};

/// Run a single integration test
fn runIntegrationTest(allocator: std.mem.Allocator, test_dir: []const u8, test_name: []const u8) !TestResult {
    var arena = std.heap.ArenaAllocator.init(allocator);
    defer arena.deinit();
    var arena_alloc = arena.allocator();

    // Construct paths
    const lisp_path = try std.fmt.allocPrint(arena_alloc, "{s}/{s}.lisp", .{ test_dir, test_name });
    const expected_path = try std.fmt.allocPrint(arena_alloc, "{s}/{s}.expected", .{ test_dir, test_name });
    const error_path = try std.fmt.allocPrint(arena_alloc, "{s}/{s}.error", .{ test_dir, test_name });

    // Check if this is a negative test (expects compilation error)
    const is_negative_test = blk: {
        fs.cwd().access(error_path, .{}) catch {
            break :blk false;
        };
        break :blk true;
    };

    // Read the lisp file
    const lisp_source = fs.cwd().readFileAlloc(arena_alloc, lisp_path, 1024 * 1024) catch |err| {
        const msg = try std.fmt.allocPrint(allocator, "Failed to read {s}: {}", .{ lisp_path, err });
        return TestResult{
            .name = try allocator.dupe(u8, test_name),
            .passed = false,
            .message = msg,
        };
    };

    // Try to compile the file
    var compiler = SimpleCCompiler.init(&arena_alloc);
    defer compiler.deinit();
    const compile_result = compiler.compileString(lisp_source, .executable);

    if (is_negative_test) {
        // For negative tests, we expect compilation to fail
        if (compile_result) |_| {
            const msg = try std.fmt.allocPrint(allocator, "Expected compilation error, but compilation succeeded", .{});
            return TestResult{
                .name = try allocator.dupe(u8, test_name),
                .passed = false,
                .message = msg,
            };
        } else |err| {
            // Read expected error pattern
            const expected_error = fs.cwd().readFileAlloc(arena_alloc, error_path, 1024 * 1024) catch |read_err| {
                const msg = try std.fmt.allocPrint(allocator, "Failed to read {s}: {}", .{ error_path, read_err });
                return TestResult{
                    .name = try allocator.dupe(u8, test_name),
                    .passed = false,
                    .message = msg,
                };
            };

            const error_str = try std.fmt.allocPrint(arena_alloc, "{}", .{err});

            // Check if error contains expected pattern (simple substring match)
            const trimmed_expected = std.mem.trim(u8, expected_error, &std.ascii.whitespace);
            if (std.mem.indexOf(u8, error_str, trimmed_expected) != null) {
                const msg = try std.fmt.allocPrint(allocator, "Compilation failed as expected with: {}", .{err});
                return TestResult{
                    .name = try allocator.dupe(u8, test_name),
                    .passed = true,
                    .message = msg,
                };
            } else {
                const msg = try std.fmt.allocPrint(allocator, "Compilation failed, but error doesn't match.\nExpected substring: {s}\nGot: {}", .{ trimmed_expected, err });
                return TestResult{
                    .name = try allocator.dupe(u8, test_name),
                    .passed = false,
                    .message = msg,
                };
            }
        }
    }

    // For positive tests, compilation should succeed
    const c_code = compile_result catch |err| {
        const msg = try std.fmt.allocPrint(allocator, "Compilation failed: {}", .{err});
        return TestResult{
            .name = try allocator.dupe(u8, test_name),
            .passed = false,
            .message = msg,
        };
    };

    // Write C code to temporary file
    var tmp_dir = testing.tmpDir(.{});
    var tmp_dir_path_buf: [std.fs.max_path_bytes]u8 = undefined;
    const tmp_dir_path = try tmp_dir.dir.realpath(".", &tmp_dir_path_buf);

    const c_file_path = try std.fmt.allocPrint(arena_alloc, "{s}/test.c", .{tmp_dir_path});
    const exe_path = try std.fmt.allocPrint(arena_alloc, "{s}/test", .{tmp_dir_path});

    try fs.cwd().writeFile(.{ .sub_path = c_file_path, .data = c_code });

    // Compile the C code
    var compile_args = std.ArrayList([]const u8){};
    defer compile_args.deinit(arena_alloc);

    try compile_args.appendSlice(arena_alloc, &.{ "zig", "cc", c_file_path, "-o", exe_path, "-std=c11", "-Wno-everything" });

    // Propagate compiler flags requested by the source
    for (compiler.compiler_flags.items) |flag| {
        try compile_args.append(arena_alloc, flag);
    }

    // Link against any requested libraries
    for (compiler.linked_libraries.items) |lib| {
        const lib_arg = try std.fmt.allocPrint(arena_alloc, "-l{s}", .{lib});
        try compile_args.append(arena_alloc, lib_arg);
    }

    // Ensure required namespace bundles are linked in
    for (compiler.required_bundles.items) |bundle| {
        try compile_args.append(arena_alloc, bundle);
    }

    var compile_process = std.process.Child.init(compile_args.items, arena_alloc);
    compile_process.stdout_behavior = .Ignore;
    compile_process.stderr_behavior = .Ignore;

    const compile_term = compile_process.spawnAndWait() catch |err| {
        tmp_dir.cleanup();
        const msg = try std.fmt.allocPrint(allocator, "Failed to spawn zig cc: {}", .{err});
        return TestResult{
            .name = try allocator.dupe(u8, test_name),
            .passed = false,
            .message = msg,
        };
    };

    if (compile_term != .Exited or compile_term.Exited != 0) {
        tmp_dir.cleanup();
        const msg = try std.fmt.allocPrint(allocator, "zig cc compilation failed with status: {}", .{compile_term});
        return TestResult{
            .name = try allocator.dupe(u8, test_name),
            .passed = false,
            .message = msg,
        };
    }

    // Run the executable and capture output
    var run_process = std.process.Child.init(&[_][]const u8{exe_path}, arena_alloc);
    run_process.stdout_behavior = .Pipe;
    run_process.stderr_behavior = .Pipe;

    try run_process.spawn();

    const stdout = try run_process.stdout.?.readToEndAlloc(arena_alloc, 1024 * 1024);
    const stderr = try run_process.stderr.?.readToEndAlloc(arena_alloc, 1024 * 1024);

    const run_term = try run_process.wait();

    // Check if the program exited successfully
    if (run_term != .Exited or run_term.Exited != 0) {
        tmp_dir.cleanup();
        const msg = try std.fmt.allocPrint(allocator, "Program execution failed with status: {}\nStderr: {s}", .{ run_term, stderr });
        return TestResult{
            .name = try allocator.dupe(u8, test_name),
            .passed = false,
            .message = msg,
        };
    }

    tmp_dir.cleanup();

    // Read expected output
    const expected_output = fs.cwd().readFileAlloc(arena_alloc, expected_path, 1024 * 1024) catch |err| {
        const msg = try std.fmt.allocPrint(allocator, "Failed to read {s}: {}", .{ expected_path, err });
        return TestResult{
            .name = try allocator.dupe(u8, test_name),
            .passed = false,
            .message = msg,
        };
    };

    // Compare outputs (trim whitespace for comparison)
    const trimmed_expected = std.mem.trim(u8, expected_output, &std.ascii.whitespace);
    const trimmed_actual = std.mem.trim(u8, stdout, &std.ascii.whitespace);

    if (std.mem.eql(u8, trimmed_expected, trimmed_actual)) {
        const msg = try std.fmt.allocPrint(allocator, "Output matches expected", .{});
        return TestResult{
            .name = try allocator.dupe(u8, test_name),
            .passed = true,
            .message = msg,
        };
    } else {
        const msg = try std.fmt.allocPrint(allocator, "Output mismatch.\nExpected:\n{s}\n\nGot:\n{s}\n\nStderr:\n{s}", .{ trimmed_expected, trimmed_actual, stderr });
        return TestResult{
            .name = try allocator.dupe(u8, test_name),
            .passed = false,
            .message = msg,
        };
    }
}

/// Run all integration tests in a directory
pub fn runAllIntegrationTests(allocator: std.mem.Allocator, test_dir: []const u8) !void {
    var dir = try fs.cwd().openDir(test_dir, .{ .iterate = true });
    defer dir.close();

    var it = dir.iterate();

    var test_names = std.ArrayList([]const u8){};
    defer {
        for (test_names.items) |name| {
            allocator.free(name);
        }
        test_names.deinit(allocator);
    }

    // Collect all .lisp files
    while (try it.next()) |entry| {
        if (entry.kind == .file) {
            if (std.mem.endsWith(u8, entry.name, ".lisp")) {
                const name = entry.name[0 .. entry.name.len - 5]; // Remove .lisp extension
                try test_names.append(allocator, try allocator.dupe(u8, name));
            }
        }
    }

    // Sort test names for consistent ordering
    std.mem.sort([]const u8, test_names.items, {}, struct {
        fn lessThan(_: void, a: []const u8, b: []const u8) bool {
            return std.mem.order(u8, a, b) == .lt;
        }
    }.lessThan);

    std.debug.print("\n=== Running Integration Tests ===\n", .{});
    std.debug.print("Test directory: {s}\n", .{test_dir});
    std.debug.print("Found {d} test(s)\n\n", .{test_names.items.len});

    var passed: usize = 0;
    var failed: usize = 0;

    var results = std.ArrayList(TestResult){};
    defer {
        for (results.items) |result| {
            allocator.free(result.name);
            allocator.free(result.message);
        }
        results.deinit(allocator);
    }

    for (test_names.items) |test_name| {
        const result = try runIntegrationTest(allocator, test_dir, test_name);
        try results.append(allocator, result);

        if (result.passed) {
            passed += 1;
            std.debug.print("✓ {s}: PASS\n", .{result.name});
        } else {
            failed += 1;
            std.debug.print("✗ {s}: FAIL\n", .{result.name});
        }
    }

    std.debug.print("\n=== Test Summary ===\n", .{});
    std.debug.print("Total: {d}, Passed: {d}, Failed: {d}\n", .{ test_names.items.len, passed, failed });

    if (failed > 0) {
        std.debug.print("\n=== Failed Tests ===\n", .{});
        for (results.items) |result| {
            if (!result.passed) {
                std.debug.print("\n--- {s} ---\n", .{result.name});
                std.debug.print("{s}\n", .{result.message});
            }
        }
    }

    if (failed > 0) {
        return error.TestsFailed;
    }
}

test "integration tests" {
    const allocator = testing.allocator;

    // Run integration tests if the directory exists
    runAllIntegrationTests(allocator, "tests/integration") catch |err| {
        if (err == error.FileNotFound) {
            std.debug.print("Integration test directory not found, skipping\n", .{});
            return;
        }
        return err;
    };
}
