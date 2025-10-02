const std = @import("std");
const builtin = @import("builtin");
const SimpleCCompiler = @import("simple_c_compiler.zig").SimpleCCompiler;
const Reader = @import("reader.zig").Reader;

fn runFile(allocator: std.mem.Allocator, filename: []const u8) !void {
    // Use arena for parsing to avoid leaks
    var parse_arena = std.heap.ArenaAllocator.init(allocator);
    defer parse_arena.deinit();
    var parse_allocator = parse_arena.allocator();

    // Read the entire file
    const file = try std.fs.cwd().openFile(filename, .{});
    defer file.close();

    const file_content = try file.readToEndAlloc(parse_allocator, 10 * 1024 * 1024);

    // Parse all expressions
    var r = Reader.init(&parse_allocator);

    const expressions = try r.readAllString(file_content);

    // Now eval each expression one by one like the REPL
    var definitions_map = std.StringHashMap([]const u8).init(allocator);
    defer {
        var iter = definitions_map.valueIterator();
        while (iter.next()) |value| {
            allocator.free(value.*);
        }
        definitions_map.deinit();
    }

    var definitions_order = std.ArrayList([]const u8){};
    defer {
        for (definitions_order.items) |name| {
            allocator.free(name);
        }
        definitions_order.deinit(allocator);
    }

    var counter: usize = 0;

    for (expressions.items) |expr| {
        const expr_str = try r.valueToString(expr);
        // expr_str is allocated with parse_allocator, will be freed with arena

        // Process like the REPL does
        try evalExpression(allocator, expr_str, &definitions_map, &definitions_order, &counter);
    }
}

fn evalExpression(
    allocator: std.mem.Allocator,
    input: []const u8,
    definitions_map: *std.StringHashMap([]const u8),
    definitions_order: *std.ArrayList([]const u8),
    counter: *usize,
) !void {
    // Check if this is a definition and extract the name
    const is_definition = std.mem.startsWith(u8, input, "(def ");
    var def_name: ?[]const u8 = null;
    var existing_key: ?[]const u8 = null;

    if (is_definition) {
        // Extract name from (def name ...)
        var start: usize = 5; // After "(def "
        while (start < input.len and input[start] == ' ') start += 1;
        var end = start;
        while (end < input.len and input[end] != ' ' and input[end] != ')') end += 1;
        if (end > start) {
            def_name = input[start..end];
            // Check if this name already exists
            for (definitions_order.items) |existing_name| {
                if (std.mem.eql(u8, existing_name, def_name.?)) {
                    existing_key = existing_name;
                    break;
                }
            }
        }
    }

    // For redefinitions, temporarily update the map with the new definition
    var old_def_backup: ?[]const u8 = null;
    if (existing_key) |key| {
        old_def_backup = definitions_map.get(key);
        const temp_def = try allocator.dupe(u8, input);
        try definitions_map.put(key, temp_def);
    }

    // Compile to bundle
    var arena = std.heap.ArenaAllocator.init(allocator);
    defer arena.deinit();
    var arena_allocator = arena.allocator();

    // Build the complete source from all current definitions + expression
    var complete_source = std.ArrayList(u8){};
    defer complete_source.deinit(arena_allocator);

    // Add all existing definitions in order
    for (definitions_order.items) |name| {
        if (definitions_map.get(name)) |def| {
            try complete_source.appendSlice(arena_allocator, def);
            try complete_source.appendSlice(arena_allocator, " ");
        }
    }

    // Add current input if it's an expression or a new definition
    if (!is_definition or existing_key == null) {
        try complete_source.appendSlice(arena_allocator, input);
    }

    var compiler = SimpleCCompiler.init(&arena_allocator);
    const c_source = compiler.compileString(complete_source.items, .bundle) catch {
        // Rollback redefinition on error
        if (old_def_backup) |old_def| {
            if (existing_key) |key| {
                // Free the temp def we created
                if (definitions_map.get(key)) |temp_def| {
                    allocator.free(temp_def);
                }
                // Restore old definition
                try definitions_map.put(key, old_def);
            }
        }
        return;
    };

    // Compilation succeeded
    if (def_name) |_| {
        if (existing_key) |_| {
            // Redefinition succeeded - free the old backup
            if (old_def_backup) |old_def| {
                allocator.free(old_def);
            }
            // The new definition is already in the map from the temp update
        } else {
            // New definition - add to both map and order
            const name_copy = try allocator.dupe(u8, def_name.?);
            try definitions_order.append(allocator, name_copy);
            const def_copy = try allocator.dupe(u8, input);
            try definitions_map.put(name_copy, def_copy);
        }
    }

    // Ensure build directory exists
    std.fs.cwd().makeDir("build") catch |err| {
        if (err != error.PathAlreadyExists) return err;
    };

    // Generate unique filenames
    const c_filename = try std.fmt.allocPrint(arena_allocator, "build/repl_{d}.c", .{counter.*});
    const bundle_filename = try std.fmt.allocPrint(arena_allocator, "build/repl_{d}.bundle", .{counter.*});
    counter.* += 1;

    // Write C source
    try std.fs.cwd().writeFile(.{ .sub_path = c_filename, .data = c_source });

    // Compile to bundle
    var cc_child = std.process.Child.init(&.{ "zig", "cc", "-dynamiclib", c_filename, "-o", bundle_filename }, arena_allocator);
    cc_child.stdout_behavior = .Ignore;
    cc_child.stderr_behavior = .Pipe;
    try cc_child.spawn();

    const stderr_data = try cc_child.stderr.?.readToEndAlloc(arena_allocator, 1024 * 1024);
    const cc_term = try cc_child.wait();

    switch (cc_term) {
        .Exited => |code| {
            if (code != 0) {
                std.debug.print("Compilation failed:\n{s}\n", .{stderr_data});
                return error.CompilationFailed;
            }
        },
        else => {
            std.debug.print("Compilation terminated abnormally\n", .{});
            return error.CompilationFailed;
        },
    }

    // Load and execute bundle
    const bundle_real_path = try std.fs.cwd().realpathAlloc(arena_allocator, bundle_filename);
    defer std.fs.cwd().deleteFile(bundle_filename) catch {};

    var lib = std.DynLib.open(bundle_real_path) catch |err| {
        std.debug.print("Failed to load bundle: {s}\n", .{@errorName(err)});
        return;
    };
    defer lib.close();

    const lisp_main = lib.lookup(*const fn () callconv(.c) i64, "lisp_main") orelse {
        std.debug.print("Failed to find lisp_main\n", .{});
        return;
    };

    const result = lisp_main();

    // Only print result for non-definition expressions
    if (!is_definition) {
        std.debug.print("{d}\n", .{result});
    }
}

pub fn main() !void {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    var allocator = gpa.allocator();

    // Check if a file was passed as argument
    var args = try std.process.argsWithAllocator(allocator);
    defer args.deinit();
    _ = args.skip(); // skip program name

    const file_arg = args.next();

    if (file_arg) |filename| {
        // File mode: read and eval the file expression by expression
        return try runFile(allocator, filename);
    }

    var stdin_buffer: [4096]u8 = undefined;
    var stdout_buffer: [4096]u8 = undefined;
    var stdin_wrapper = std.fs.File.stdin().reader(&stdin_buffer);
    var stdout_wrapper = std.fs.File.stdout().writer(&stdout_buffer);
    const stdin: *std.Io.Reader = &stdin_wrapper.interface;
    const stdout: *std.Io.Writer = &stdout_wrapper.interface;

    try stdout.print("Lisp REPL - Type expressions and press Enter\n", .{});
    try stdout.print("Type (exit) to quit\n\n", .{});

    var buffer = std.ArrayList(u8){};
    defer buffer.deinit(allocator);

    // Track definitions by name, so we can replace them
    // Also track insertion order for stable output
    var definitions_map = std.StringHashMap([]const u8).init(allocator);
    var definitions_order = std.ArrayList([]const u8){};
    defer {
        var iter = definitions_map.valueIterator();
        while (iter.next()) |value| {
            allocator.free(value.*);
        }
        for (definitions_order.items) |name| {
            allocator.free(name);
        }
        definitions_map.deinit();
        definitions_order.deinit(allocator);
    }

    var counter: usize = 0;
    var input_buffer = std.ArrayList(u8){};
    defer input_buffer.deinit(allocator);

    while (true) {
        const prompt = if (input_buffer.items.len == 0) "> " else "  ";
        try stdout.print("{s}", .{prompt});
        try stdout.*.flush();

        const line = stdin.*.takeDelimiterExclusive('\n') catch |err| switch (err) {
            error.EndOfStream => break,
            error.StreamTooLong => {
                std.debug.print("Input too long\n", .{});
                continue;
            },
            else => return err,
        };

        // Accumulate input
        if (input_buffer.items.len > 0) {
            try input_buffer.appendSlice(allocator, "\n");
        }
        try input_buffer.appendSlice(allocator, line);

        const trimmed = std.mem.trim(u8, input_buffer.items, &std.ascii.whitespace);

        // Check if we have a complete expression by counting parens
        var paren_count: i32 = 0;
        var in_string = false;
        var escape_next = false;

        for (trimmed) |c| {
            if (escape_next) {
                escape_next = false;
                continue;
            }
            if (c == '\\') {
                escape_next = true;
                continue;
            }
            if (c == '"') {
                in_string = !in_string;
                continue;
            }
            if (!in_string) {
                if (c == '(') paren_count += 1;
                if (c == ')') paren_count -= 1;
            }
        }

        // If we don't have balanced parens yet, continue reading
        if (paren_count > 0) continue;

        const input = trimmed;
        defer input_buffer.clearRetainingCapacity();

        if (input.len == 0) continue;

        // Check for exit command
        if (std.mem.eql(u8, input, "(exit)")) break;

        // Check if this is a definition and extract the name
        const is_definition = std.mem.startsWith(u8, input, "(def ");
        var def_name: ?[]const u8 = null;
        var existing_key: ?[]const u8 = null;

        if (is_definition) {
            // Extract name from (def name ...)
            var start: usize = 5; // After "(def "
            while (start < input.len and input[start] == ' ') start += 1;
            var end = start;
            while (end < input.len and input[end] != ' ' and input[end] != ')') end += 1;
            if (end > start) {
                def_name = input[start..end];
                // Check if this name already exists
                for (definitions_order.items) |existing_name| {
                    if (std.mem.eql(u8, existing_name, def_name.?)) {
                        existing_key = existing_name;
                        break;
                    }
                }
            }
        }

        // For redefinitions, temporarily update the map with the new definition
        var old_def_backup: ?[]const u8 = null;
        if (existing_key) |key| {
            old_def_backup = definitions_map.get(key);
            const temp_def = try allocator.dupe(u8, input);
            try definitions_map.put(key, temp_def);
        }

        // Compile to bundle
        var arena = std.heap.ArenaAllocator.init(allocator);
        defer arena.deinit();
        var arena_allocator = arena.allocator();

        // Build the complete source from all current definitions + expression
        var complete_source = std.ArrayList(u8){};
        defer complete_source.deinit(arena_allocator);

        // Add all existing definitions in order
        for (definitions_order.items) |name| {
            if (definitions_map.get(name)) |def| {
                try complete_source.appendSlice(arena_allocator, def);
                try complete_source.appendSlice(arena_allocator, " ");
            }
        }

        // Add current input if it's an expression or a new definition
        if (!is_definition or existing_key == null) {
            try complete_source.appendSlice(arena_allocator, input);
        }

        var compiler = SimpleCCompiler.init(&arena_allocator);
        const c_source = compiler.compileString(complete_source.items, .bundle) catch {
            // Rollback redefinition on error
            if (old_def_backup) |old_def| {
                if (existing_key) |key| {
                    // Free the temp def we created
                    if (definitions_map.get(key)) |temp_def| {
                        allocator.free(temp_def);
                    }
                    // Restore old definition
                    try definitions_map.put(key, old_def);
                }
            }
            continue;
        };

        // Compilation succeeded
        if (def_name) |_| {
            if (existing_key) |_| {
                // Redefinition succeeded - free the old backup
                if (old_def_backup) |old_def| {
                    allocator.free(old_def);
                }
                // The new definition is already in the map from the temp update
            } else {
                // New definition - add to order list and map
                const name_copy = try allocator.dupe(u8, def_name.?);
                try definitions_order.append(allocator, name_copy);
                const new_def = try allocator.dupe(u8, input);
                try definitions_map.put(name_copy, new_def);
            }
        }

        // Ensure build directory exists
        std.fs.cwd().makeDir("build") catch |err| {
            if (err != error.PathAlreadyExists) return err;
        };

        // Generate unique filenames
        const c_filename = try std.fmt.allocPrint(arena_allocator, "build/repl_{d}.c", .{counter});
        const bundle_filename = try std.fmt.allocPrint(arena_allocator, "build/repl_{d}.bundle", .{counter});
        counter += 1;

        // Write C source
        try std.fs.cwd().writeFile(.{ .sub_path = c_filename, .data = c_source });

        // Compile to bundle
        var cc_child = std.process.Child.init(&.{ "zig", "cc", "-dynamiclib", c_filename, "-o", bundle_filename }, arena_allocator);
        cc_child.stdout_behavior = .Ignore;
        cc_child.stderr_behavior = .Pipe;
        try cc_child.spawn();

        const stderr_data = try cc_child.stderr.?.readToEndAlloc(arena_allocator, 1024 * 1024);
        const cc_term = try cc_child.wait();

        switch (cc_term) {
            .Exited => |code| {
                if (code != 0) {
                    std.debug.print("Build error:\n{s}\n", .{stderr_data});
                    continue;
                }
            },
            else => {
                std.debug.print("Build failed\n", .{});
                continue;
            },
        }

        // Load and execute bundle
        const bundle_real_path = try std.fs.cwd().realpathAlloc(arena_allocator, bundle_filename);
        defer std.fs.cwd().deleteFile(bundle_filename) catch {};

        var lib = std.DynLib.open(bundle_real_path) catch |err| {
            std.debug.print("Failed to load bundle: {s}\n", .{@errorName(err)});
            continue;
        };
        defer lib.close();

        const entry_fn = lib.lookup(*const fn () callconv(.c) void, "lisp_main") orelse {
            std.debug.print("Bundle missing lisp_main entry\n", .{});
            continue;
        };

        @call(.auto, entry_fn, .{});

        // Flush to ensure all output is visible (important for Emacs inferior-lisp mode)
        try stdout.*.flush();
    }

    try stdout.print("Goodbye!\n", .{});
}
