const std = @import("std");
const testing = std.testing;

const TestCase = struct {
    name: []const u8,
    input: []const u8,
    expected_outputs: []const []const u8,
    expected_errors: []const []const u8 = &[_][]const u8{},
};

fn runReplTest(allocator: std.mem.Allocator, test_case: TestCase) !void {
    std.debug.print("\n=== Running test: {s} ===\n", .{test_case.name});

    // Create a temporary file for input
    var tmp_dir = testing.tmpDir(.{});
    defer tmp_dir.cleanup();

    const input_file = try tmp_dir.dir.createFile("input.txt", .{});
    defer input_file.close();
    try input_file.writeAll(test_case.input);

    // Get the absolute path to the input file
    const tmp_path = try tmp_dir.dir.realpathAlloc(allocator, ".");
    defer allocator.free(tmp_path);
    const input_path = try std.fs.path.join(allocator, &[_][]const u8{ tmp_path, "input.txt" });
    defer allocator.free(input_path);

    // Run the REPL with the input file
    const argv = [_][]const u8{
        "./zig-out/bin/mlir_lisp",
        "--repl",
    };

    var child = std.process.Child.init(&argv, allocator);
    child.stdin_behavior = .Pipe;
    child.stdout_behavior = .Pipe;
    child.stderr_behavior = .Pipe;

    try child.spawn();

    // Write input
    if (child.stdin) |stdin| {
        try stdin.writeAll(test_case.input);
        stdin.close();
        child.stdin = null;
    }

    // Read output - must read before wait() or it can deadlock
    var stdout_bytes: []const u8 = "";
    if (child.stdout) |stdout| {
        stdout_bytes = try stdout.readToEndAlloc(allocator, 1024 * 1024);
    }
    defer if (stdout_bytes.len > 0) allocator.free(stdout_bytes);

    var stderr_bytes: []const u8 = "";
    if (child.stderr) |stderr| {
        stderr_bytes = try stderr.readToEndAlloc(allocator, 1024 * 1024);
    }
    defer if (stderr_bytes.len > 0) allocator.free(stderr_bytes);

    const term = try child.wait();

    const combined_output = try std.fmt.allocPrint(allocator, "{s}{s}", .{ stdout_bytes, stderr_bytes });
    defer allocator.free(combined_output);

    std.debug.print("Output:\n{s}\n", .{combined_output});

    // Check for expected outputs
    for (test_case.expected_outputs) |expected| {
        if (std.mem.indexOf(u8, combined_output, expected) == null) {
            std.debug.print("FAILED: Expected output not found: {s}\n", .{expected});
            return error.TestFailed;
        }
    }

    // Check for expected errors
    for (test_case.expected_errors) |expected_error| {
        if (std.mem.indexOf(u8, combined_output, expected_error) == null) {
            std.debug.print("FAILED: Expected error not found: {s}\n", .{expected_error});
            return error.TestFailed;
        }
    }

    // If we expected errors, the process should have failed
    if (test_case.expected_errors.len > 0) {
        if (term != .Exited or term.Exited == 0) {
            std.debug.print("FAILED: Expected process to fail but it succeeded\n", .{});
            return error.TestFailed;
        }
    }

    std.debug.print("PASSED\n", .{});
}

test "REPL: help command" {
    const test_case = TestCase{
        .name = "help command",
        .input = ":help\n:quit\n",
        .expected_outputs = &[_][]const u8{
            "REPL Commands:",
            ":help",
            ":quit",
            ":mlir",
            ":clear",
        },
    };

    try runReplTest(testing.allocator, test_case);
}

test "REPL: simple constant auto-execution" {
    const test_case = TestCase{
        .name = "simple constant auto-execution",
        .input =
        \\(operation
        \\  (name arith.constant)
        \\  (result-bindings [%x])
        \\  (result-types i32)
        \\  (attributes { :value (: 42 i32) }))
        \\:quit
        \\
        ,
        .expected_outputs = &[_][]const u8{
            "Result:",
        },
    };

    try runReplTest(testing.allocator, test_case);
}

test "REPL: function definition" {
    const test_case = TestCase{
        .name = "function definition",
        .input =
        \\(operation
        \\  (name func.func)
        \\  (attributes {
        \\    :sym_name @test
        \\    :function_type (!function (inputs) (results i64))
        \\  })
        \\  (regions
        \\    (region
        \\      (block [^entry]
        \\        (arguments [])
        \\        (operation
        \\          (name arith.constant)
        \\          (result-bindings [%c42])
        \\          (result-types i64)
        \\          (attributes { :value (: 42 i64) }))
        \\        (operation
        \\          (name func.return)
        \\          (operands %c42))))))
        \\:quit
        \\
        ,
        .expected_outputs = &[_][]const u8{
            "Function defined",
        },
    };

    try runReplTest(testing.allocator, test_case);
}

test "REPL: mlir command shows module" {
    const test_case = TestCase{
        .name = "mlir command shows module",
        .input =
        \\(operation
        \\  (name func.func)
        \\  (attributes {
        \\    :sym_name @test
        \\    :function_type (!function (inputs) (results i64))
        \\  })
        \\  (regions
        \\    (region
        \\      (block [^entry]
        \\        (arguments [])
        \\        (operation
        \\          (name arith.constant)
        \\          (result-bindings [%c42])
        \\          (result-types i64)
        \\          (attributes { :value (: 42 i64) }))
        \\        (operation
        \\          (name func.return)
        \\          (operands %c42))))))
        \\:mlir
        \\:quit
        \\
        ,
        .expected_outputs = &[_][]const u8{
            "Function defined",
            "Current MLIR module:",
            "@test",
        },
    };

    try runReplTest(testing.allocator, test_case);
}

test "REPL: clear command" {
    const test_case = TestCase{
        .name = "clear command",
        .input =
        \\(operation
        \\  (name func.func)
        \\  (attributes {
        \\    :sym_name @test
        \\    :function_type (!function (inputs) (results i64))
        \\  })
        \\  (regions
        \\    (region
        \\      (block [^entry]
        \\        (arguments [])
        \\        (operation
        \\          (name arith.constant)
        \\          (result-bindings [%c42])
        \\          (result-types i64)
        \\          (attributes { :value (: 42 i64) }))
        \\        (operation
        \\          (name func.return)
        \\          (operands %c42))))))
        \\:clear
        \\:mlir
        \\:quit
        \\
        ,
        .expected_outputs = &[_][]const u8{
            "Function defined",
            "Module cleared",
            "No module compiled yet",
        },
    };

    try runReplTest(testing.allocator, test_case);
}

test "REPL: multiple operations accumulate" {
    const test_case = TestCase{
        .name = "multiple operations accumulate",
        .input =
        \\(operation
        \\  (name arith.constant)
        \\  (result-bindings [%x])
        \\  (result-types i32)
        \\  (attributes { :value (: 42 i32) }))
        \\(operation
        \\  (name arith.constant)
        \\  (result-bindings [%y])
        \\  (result-types i64)
        \\  (attributes { :value (: 999 i64) }))
        \\:quit
        \\
        ,
        .expected_outputs = &[_][]const u8{
            "Result:",
            "Result:",
        },
    };

    try runReplTest(testing.allocator, test_case);
}

test "REPL: invalid syntax error" {
    const test_case = TestCase{
        .name = "invalid syntax error",
        .input =
        \\(invalid syntax here
        \\:quit
        \\
        ,
        .expected_outputs = &[_][]const u8{
            "Error:",
        },
    };

    try runReplTest(testing.allocator, test_case);
}

test "REPL: unbalanced parentheses error" {
    const test_case = TestCase{
        .name = "unbalanced parentheses",
        .input =
        \\(operation))
        \\:quit
        \\
        ,
        .expected_outputs = &[_][]const u8{
            "Unbalanced parentheses",
        },
    };

    try runReplTest(testing.allocator, test_case);
}
