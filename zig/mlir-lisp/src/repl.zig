const std = @import("std");
const mlir_lisp = @import("root.zig");
const mlir = mlir_lisp.mlir;
const Tokenizer = mlir_lisp.Tokenizer;
const Reader = mlir_lisp.Reader;
const Value = mlir_lisp.Value;
const Parser = mlir_lisp.Parser;
const Builder = mlir_lisp.Builder;
const Executor = mlir_lisp.Executor;
const ExecutorConfig = mlir_lisp.ExecutorConfig;
const MlirModule = mlir_lisp.MlirModule;
const Operation = mlir_lisp.Operation;

pub fn run(allocator: std.mem.Allocator) !void {
    std.debug.print("MLIR-Lisp REPL\n", .{});
    std.debug.print("Type :help for commands, :quit to exit\n\n", .{});

    // Initialize MLIR context (persistent across REPL session)
    var ctx = try mlir.Context.create();
    defer ctx.destroy();
    ctx.registerAllDialects();
    mlir.Context.registerAllPasses();

    // Load necessary dialects
    try ctx.getOrLoadDialect("func");
    try ctx.getOrLoadDialect("arith");
    try ctx.getOrLoadDialect("scf");
    try ctx.getOrLoadDialect("cf");

    // Create initial empty module
    var operations_list = std.ArrayList(Operation){};
    defer {
        for (operations_list.items) |*op| {
            op.deinit(allocator);
        }
        operations_list.deinit(allocator);
    }

    // Keep track of Values that need to stay alive (for auto-exec wrappers)
    var alive_values = std.ArrayList(*Value){};
    defer {
        for (alive_values.items) |val| {
            val.deinit(allocator);
            allocator.destroy(val);
        }
        alive_values.deinit(allocator);
    }

    // Keep track of source strings that need to stay alive (for auto-exec wrappers)
    var alive_sources = std.ArrayList([]const u8){};
    defer {
        for (alive_sources.items) |source| {
            allocator.free(source);
        }
        alive_sources.deinit(allocator);
    }

    // Create builder and executor (persistent)
    var builder = Builder.init(allocator, &ctx);
    defer builder.deinit();

    const executor_config = ExecutorConfig{
        .opt_level = .O2,
        .enable_verifier = true,
    };
    var executor = Executor.init(allocator, &ctx, executor_config);
    defer executor.deinit();

    // REPL state
    var mlir_module: ?mlir.Module = null;
    defer if (mlir_module) |*mod| mod.destroy();

    var input_buffer = std.ArrayList(u8){};
    defer input_buffer.deinit(allocator);

    var paren_depth: i32 = 0;
    var brace_depth: i32 = 0;
    var bracket_depth: i32 = 0;

    // Setup stdin reader
    const stdin_file = std.fs.File.stdin();

    // Buffer for reading lines
    var line_buffer = std.ArrayList(u8){};
    defer line_buffer.deinit(allocator);

    while (true) {
        // Show prompt
        if (paren_depth == 0 and brace_depth == 0 and bracket_depth == 0) {
            std.debug.print("mlir-lisp> ", .{});
        } else {
            std.debug.print("        ... ", .{});
        }

        // Read line (or EOF)
        line_buffer.clearRetainingCapacity();
        var buf: [1]u8 = undefined;
        while (true) {
            const n = stdin_file.read(&buf) catch |err| {
                if (err == error.EndOfStream) {
                    if (line_buffer.items.len == 0) {
                        return;  // EOF with no input - exit REPL
                    }
                    break;  // EOF after some input - process what we have
                }
                return err;
            };
            if (n == 0) {  // EOF
                if (line_buffer.items.len == 0) {
                    return;
                }
                break;
            }
            if (buf[0] == '\n') break;
            try line_buffer.append(allocator, buf[0]);
        }
        const line = line_buffer.items;

        // Trim whitespace
        const trimmed = std.mem.trim(u8, line, " \t\r\n");

        // Handle special commands (only when not in multi-line mode)
        if (paren_depth == 0 and trimmed.len > 0 and trimmed[0] == ':') {
            if (std.mem.eql(u8, trimmed, ":quit")) {
                std.debug.print("Goodbye!\n", .{});
                break;
            } else if (std.mem.eql(u8, trimmed, ":help")) {
                try printHelp();
                continue;
            } else if (std.mem.eql(u8, trimmed, ":mlir")) {
                if (mlir_module) |*mod| {
                    std.debug.print("\nCurrent MLIR module:\n", .{});
                    std.debug.print("----------------------------------------\n", .{});
                    mod.print();
                    std.debug.print("----------------------------------------\n\n", .{});
                } else {
                    std.debug.print("No module compiled yet.\n\n", .{});
                }
                continue;
            } else if (std.mem.eql(u8, trimmed, ":clear")) {
                // Reset module
                for (operations_list.items) |*op| {
                    op.deinit(allocator);
                }
                operations_list.clearRetainingCapacity();
                if (mlir_module) |*mod| mod.destroy();
                mlir_module = null;
                std.debug.print("Module cleared.\n\n", .{});
                continue;
            } else {
                std.debug.print("Unknown command: {s}\n", .{trimmed});
                try printHelp();
                continue;
            }
        }

        // Skip empty lines
        if (trimmed.len == 0) {
            continue;
        }

        // Append to input buffer
        if (input_buffer.items.len > 0) {
            try input_buffer.append(allocator, '\n');
        }
        try input_buffer.appendSlice(allocator, trimmed);

        // Track bracket depths
        for (trimmed) |c| {
            if (c == '(') paren_depth += 1;
            if (c == ')') paren_depth -= 1;
            if (c == '{') brace_depth += 1;
            if (c == '}') brace_depth -= 1;
            if (c == '[') bracket_depth += 1;
            if (c == ']') bracket_depth -= 1;
        }

        // If we have a complete expression, process it
        if (paren_depth == 0 and brace_depth == 0 and bracket_depth == 0) {
            processInput(
                allocator,
                input_buffer.items,
                &ctx,
                &operations_list,
                &builder,
                &executor,
                &mlir_module,
                &alive_values,
                &alive_sources,
            ) catch |err| {
                std.debug.print("Error: {}\n\n", .{err});
            };

            // Clear input buffer
            input_buffer.clearRetainingCapacity();
        } else if (paren_depth < 0 or brace_depth < 0 or bracket_depth < 0) {
            std.debug.print("Error: Unbalanced brackets\n\n", .{});
            input_buffer.clearRetainingCapacity();
            paren_depth = 0;
            brace_depth = 0;
            bracket_depth = 0;
        }
    }
}

// Helper function to create a wrapper function that executes a single operation and returns 0
// TODO: Properly return operation results
fn createReplExecWrapper(allocator: std.mem.Allocator, user_op: Operation, parser: *Parser, alive_values: *std.ArrayList(*Value), alive_sources: *std.ArrayList([]const u8)) !Operation {
    // Use Printer to convert the operation to a string
    const Printer = mlir_lisp.Printer;
    var printer = Printer.init(allocator);
    defer printer.deinit();

    try printer.printOperation(&user_op);
    const op_string = try allocator.dupe(u8, printer.getOutput());
    defer allocator.free(op_string);

    // Debug: print the serialized operation
    std.debug.print("\n=== Serialized operation ===\n{s}\n============================\n", .{op_string});

    // Build a wrapper function source with the operation embedded
    const wrapper_source = try std.fmt.allocPrint(allocator,
        \\(operation
        \\  (name func.func)
        \\  (attributes {{
        \\    :sym_name @__repl_exec
        \\    :function_type (!function (inputs) (results !i64))
        \\  }})
        \\  (regions
        \\    (region
        \\      (block
        \\        (arguments [])
        \\        {s}
        \\        (operation
        \\          (name arith.constant)
        \\          (result-bindings [%__result])
        \\          (result-types !i64)
        \\          (attributes {{ :value (: 0 !i64) }}))
        \\        (operation
        \\          (name func.return)
        \\          (operands %__result))))))
    , .{op_string});

    // Debug: print the wrapper source
    std.debug.print("\n=== Wrapper source ===\n{s}\n======================\n", .{wrapper_source});

    // Parse the wrapper
    std.debug.print("DEBUG: About to tokenize wrapper_source (len={})\n", .{wrapper_source.len});
    var tok = Tokenizer.init(allocator, wrapper_source);
    var reader = try Reader.init(allocator, &tok);
    std.debug.print("DEBUG: About to read wrapper value\n", .{});
    const value = try reader.read();
    std.debug.print("DEBUG: Wrapper value read successfully, type={s}\n", .{@tagName(value.type)});

    // Keep the value and source alive - don't deinit them
    try alive_values.append(allocator, value);
    try alive_sources.append(allocator, wrapper_source);

    std.debug.print("DEBUG: About to parse wrapper operation\n", .{});
    const result = try parser.parseOperation(value);
    std.debug.print("DEBUG: Wrapper operation parsed successfully, name={s}\n", .{result.name});
    std.debug.print("DEBUG: Wrapper has {} regions\n", .{result.regions.len});
    if (result.regions.len > 0) {
        std.debug.print("DEBUG: First region has {} blocks\n", .{result.regions[0].blocks.len});
        if (result.regions[0].blocks.len > 0) {
            std.debug.print("DEBUG: First block has {} operations\n", .{result.regions[0].blocks[0].operations.len});
            for (result.regions[0].blocks[0].operations, 0..) |*nested_op, op_idx| {
                std.debug.print("DEBUG: Operation[{}] name={s}, attributes={}\n", .{op_idx, nested_op.name, nested_op.attributes.len});
                for (nested_op.attributes, 0..) |attr, attr_idx| {
                    std.debug.print("DEBUG:   attr[{}] key=\"{s}\"\n", .{attr_idx, attr.key});
                }
            }
        }
    }
    return result;
}

fn processInput(
    allocator: std.mem.Allocator,
    input: []const u8,
    _: *mlir.Context,
    operations_list: *std.ArrayList(Operation),
    builder: *Builder,
    executor: *Executor,
    mlir_module: *?mlir.Module,
    alive_values: *std.ArrayList(*Value),
    alive_sources: *std.ArrayList([]const u8),
) !void {

    // Parse the input
    var tok = Tokenizer.init(allocator, input);
    var reader = try Reader.init(allocator, &tok);
    var value = try reader.read();
    defer {
        value.deinit(allocator);
        allocator.destroy(value);
    }

    // Parse to AST
    var parser = Parser.init(allocator);

    // Check if this is a module definition or a single operation
    if (value.type != .list) {
        std.debug.print("Error: Expected list starting with 'mlir' or 'operation'\n\n", .{});
        return;
    }

    const list = value.data.list;
    if (list.len() == 0) {
        std.debug.print("Error: Empty list\n\n", .{});
        return;
    }

    const first = list.at(0);
    if (first.type != .identifier and first.type != .symbol) {
        std.debug.print("Error: First element must be an identifier or symbol, got {s}\n", .{@tagName(first.type)});
        if (first.type == .keyword or first.type == .string) {
            std.debug.print("  Value: {s}\n", .{first.data.atom});
        }
        std.debug.print("\n", .{});
        return;
    }

    const sym = first.data.atom;
    if (std.mem.eql(u8, sym, "mlir")) {
        // Full module definition - replace current module
        const new_module_ast = try parser.parseModule(value);

        // Clear old operations
        for (operations_list.items) |*op| {
            op.deinit(allocator);
        }
        operations_list.clearRetainingCapacity();

        // Copy new operations
        try operations_list.appendSlice(allocator, new_module_ast.operations);

        // Don't deinit new_module_ast operations since we copied them
        allocator.free(new_module_ast.operations);

        std.debug.print("Module updated.\n", .{});

        // Build and compile the module (but don't auto-execute)
        if (mlir_module.*) |*mod| mod.destroy();

        var module_ast = MlirModule{
            .operations = operations_list.items,
            .allocator = allocator,
        };

        var new_mlir_module = try builder.buildModule(&module_ast);

        // Compile
        executor.compile(&new_mlir_module) catch |err| {
            std.debug.print("Error: Compilation failed: {}\n", .{err});
            new_mlir_module.print();
            new_mlir_module.destroy();
            return err;
        };
        mlir_module.* = new_mlir_module;

        std.debug.print("✓ Compiled successfully\n\n", .{});
    } else if (std.mem.eql(u8, sym, "operation")) {
        // Single operation
        const op = try parser.parseOperation(value);

        // Check if this is a function definition
        const is_func_def = std.mem.eql(u8, op.name, "func.func");

        if (is_func_def) {
            // Function definitions: add to module and compile (no execution)
            try operations_list.append(allocator, op);

            // Build and compile
            if (mlir_module.*) |*mod| mod.destroy();

            var module_ast = MlirModule{
                .operations = operations_list.items,
                .allocator = allocator,
            };

            var new_mlir_module = try builder.buildModule(&module_ast);

            executor.compile(&new_mlir_module) catch |err| {
                std.debug.print("Error: Compilation failed: {}\n", .{err});
                new_mlir_module.print();
                new_mlir_module.destroy();
                // Remove the operation before returning
                if (operations_list.pop()) |removed_op| {
                    var op_copy = removed_op;
                    op_copy.deinit(allocator);
                }
                return err;
            };
            mlir_module.* = new_mlir_module;

            std.debug.print("Function defined\n\n", .{});
        } else {
            // Other operations: wrap in function, execute, show result
            const wrapper = try createReplExecWrapper(allocator, op, &parser, alive_values, alive_sources);

            // DEBUGGING: Don't free the original operation yet - testing if this causes corruption
            // var op_copy = op;
            // op_copy.deinit(allocator);

            try operations_list.append(allocator, wrapper);

            // Build and compile the module with the wrapper
            if (mlir_module.*) |*mod| mod.destroy();

            var module_ast = MlirModule{
                .operations = operations_list.items,
                .allocator = allocator,
            };

            var new_mlir_module = try builder.buildModule(&module_ast);

            executor.compile(&new_mlir_module) catch |err| {
                std.debug.print("Error: Compilation failed: {}\n", .{err});
                new_mlir_module.print();
                new_mlir_module.destroy();
                // Remove the wrapper before returning
                if (operations_list.pop()) |removed_wrapper| {
                    var wrapper_copy = removed_wrapper;
                    wrapper_copy.deinit(allocator);
                }
                return err;
            };
            mlir_module.* = new_mlir_module;

            // Execute the wrapper function
            const exec_fn_ptr = executor.lookup("__repl_exec") orelse {
                std.debug.print("Error: Failed to find __repl_exec function\n\n", .{});
                if (operations_list.pop()) |removed_wrapper| {
                    var wrapper_copy = removed_wrapper;
                    wrapper_copy.deinit(allocator);
                }
                return error.FunctionNotFound;
            };

            const exec_fn: *const fn () callconv(.c) i64 = @ptrCast(@alignCast(exec_fn_ptr));
            const result = exec_fn();
            std.debug.print("Result: {}\n\n", .{result});

            // Remove the temporary wrapper function from the operations list
            if (operations_list.pop()) |removed_wrapper| {
                var wrapper_copy = removed_wrapper;
                wrapper_copy.deinit(allocator);
            }

            // Rebuild module without the wrapper to keep clean state
            if (mlir_module.*) |*mod| mod.destroy();

            var clean_module_ast = MlirModule{
                .operations = operations_list.items,
                .allocator = allocator,
            };

            var clean_mlir_module = try builder.buildModule(&clean_module_ast);

            executor.compile(&clean_mlir_module) catch |err| {
                std.debug.print("Warning: Failed to recompile clean module: {}\n", .{err});
                clean_mlir_module.destroy();
                mlir_module.* = null;
                return err;
            };
            mlir_module.* = clean_mlir_module;
        }
    } else {
        std.debug.print("Error: Expected 'mlir' or 'operation' at top level\n\n", .{});
        return;
    }
}

fn printHelp() !void {
    std.debug.print(
        \\
        \\REPL Commands:
        \\  :help     - Show this help
        \\  :quit     - Exit the REPL
        \\  :mlir     - Show current MLIR module
        \\  :clear    - Clear the module
        \\
        \\Usage:
        \\  - Enter operations or complete module definitions
        \\  - Multi-line input is supported (closes when parentheses balance)
        \\  - Single (operation ...) forms are auto-executed and show results
        \\  - Use (mlir ...) to define/replace the entire module (no auto-execution)
        \\  - Operations accumulate - you can define functions then call them
        \\
        \\Examples:
        \\
        \\  Auto-execute a constant:
        \\    mlir-lisp> (operation
        \\            ...   (name arith.constant)
        \\            ...   (result-bindings [%x])
        \\            ...   (result-types !i32)
        \\            ...   (attributes {{ :value (: 42 !i32) }}))
        \\    Result: 42
        \\
        \\  Define a function then call it:
        \\    mlir-lisp> (operation (name func.func) ... @add ...)
        \\    Result: 0
        \\    mlir-lisp> (operation (name func.call) ... calling @add ...)
        \\    Result: 15
        \\
        \\  Define a complete module (no auto-execute):
        \\    mlir-lisp> (mlir ...)
        \\    Module updated.
        \\    ✓ Compiled successfully
        \\
        \\
    , .{});
}
