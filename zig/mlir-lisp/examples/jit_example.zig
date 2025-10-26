const std = @import("std");
const mlir_lisp = @import("mlir_lisp");

pub fn main() !void {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();

    std.debug.print("=== MLIR-Lisp JIT Compilation Example ===\n\n", .{});

    // Create an MLIR context and register all dialects and passes
    var ctx = try mlir_lisp.mlir.Context.create();
    defer ctx.destroy();
    ctx.registerAllDialects();
    mlir_lisp.mlir.Context.registerAllPasses();

    // Load the func dialect explicitly (needed for passes)
    try ctx.getOrLoadDialect("func");
    try ctx.getOrLoadDialect("arith");

    // Example 1: Simple addition function
    std.debug.print("Example 1: Simple Addition Function\n", .{});
    std.debug.print("-----------------------------------\n", .{});

    const add_program =
        \\(mlir
        \\  (operation
        \\    (name func.func)
        \\    (attributes {
        \\      :sym_name @add
        \\      :function_type (!function (inputs i32 i32) (results i32))
        \\    })
        \\    (regions
        \\      (region
        \\        (block
        \\          (arguments [[%arg0 i32] [%arg1 i32]])
        \\          (operation
        \\            (name arith.addi)
        \\            (result-bindings [%sum])
        \\            (result-types i32)
        \\            (operands %arg0 %arg1))
        \\          (operation
        \\            (name func.return)
        \\            (operands %sum)))))))
    ;

    std.debug.print("Input program:\n{s}\n\n", .{add_program});

    // Parse the program
    var tok = mlir_lisp.Tokenizer.init(allocator, add_program);
    var r = try mlir_lisp.Reader.init(allocator, &tok);
    var value = try r.read();
    defer {
        value.deinit(allocator);
        allocator.destroy(value);
    }

    // Parse to our AST
    var p = mlir_lisp.Parser.init(allocator);
    var parsed_module = try p.parseModule(value);
    defer parsed_module.deinit();

    // Build MLIR IR
    var builder = mlir_lisp.Builder.init(allocator, &ctx);
    defer builder.deinit();

    var mlir_module = try builder.buildModule(&parsed_module);
    defer mlir_module.destroy();

    std.debug.print("MLIR module created successfully!\n", .{});

    // Print the MLIR before lowering
    std.debug.print("\nMLIR (before lowering):\n", .{});
    mlir_module.print();
    std.debug.print("\n", .{});

    // Create executor and compile
    const executor_config = mlir_lisp.ExecutorConfig{
        .opt_level = .O0,
        .enable_verifier = true,
    };

    var executor = mlir_lisp.Executor.init(allocator, &ctx, executor_config);
    defer executor.deinit();

    std.debug.print("Compiling with JIT (optimization level: O0)...\n", .{});
    executor.compile(&mlir_module) catch |err| {
        std.debug.print("Compilation failed: {}\n", .{err});
        return err;
    };

    std.debug.print("✓ Compilation successful!\n\n", .{});

    // Look up the compiled function
    const add_fn_ptr = executor.lookup("add");
    if (add_fn_ptr == null) {
        std.debug.print("Failed to find 'add' function in JIT'd code\n", .{});
        return error.FunctionNotFound;
    }

    // Cast to function pointer type: fn(i32, i32) -> i32
    const AddFn = *const fn (i32, i32) callconv(.c) i32;
    const add_fn: AddFn = @ptrCast(@alignCast(add_fn_ptr));

    // Test the function with some values
    std.debug.print("Testing JIT'd function:\n", .{});
    const test_cases = [_][2]i32{
        .{ 5, 3 },
        .{ 100, 200 },
        .{ -10, 25 },
        .{ 0, 0 },
    };

    for (test_cases) |test_case| {
        const a = test_case[0];
        const b = test_case[1];
        const result = add_fn(a, b);
        std.debug.print("  add({}, {}) = {}\n", .{ a, b, result });
    }
    std.debug.print("\n", .{});

    // Example 2: Show different optimization levels
    std.debug.print("\nExample 2: Different Optimization Levels\n", .{});
    std.debug.print("----------------------------------------\n", .{});

    for ([_]mlir_lisp.OptLevel{ .O0, .O1, .O2, .O3 }) |opt_level| {
        std.debug.print("Testing optimization level: {s}\n", .{@tagName(opt_level)});

        // Need to rebuild the module for each compilation
        var tok2 = mlir_lisp.Tokenizer.init(allocator, add_program);
        var r2 = try mlir_lisp.Reader.init(allocator, &tok2);
        var value2 = try r2.read();
        defer {
            value2.deinit(allocator);
            allocator.destroy(value2);
        }

        var p2 = mlir_lisp.Parser.init(allocator);
        var parsed_module2 = try p2.parseModule(value2);
        defer parsed_module2.deinit();

        var builder2 = mlir_lisp.Builder.init(allocator, &ctx);
        defer builder2.deinit();

        var mlir_module2 = try builder2.buildModule(&parsed_module2);
        defer mlir_module2.destroy();

        const config = mlir_lisp.ExecutorConfig{
            .opt_level = opt_level,
            .enable_verifier = true,
        };

        var exec = mlir_lisp.Executor.init(allocator, &ctx, config);
        defer exec.deinit();

        exec.compile(&mlir_module2) catch |err| {
            std.debug.print("  ✗ Failed: {}\n", .{err});
            continue;
        };

        std.debug.print("  ✓ Compiled successfully\n", .{});
    }

    std.debug.print("\n=== JIT compilation examples completed! ===\n", .{});
}
