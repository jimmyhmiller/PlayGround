const std = @import("std");
const mlir_lisp = @import("mlir_lisp");

pub fn main() !void {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();

    std.debug.print("=== MLIR-Lisp Fibonacci JIT Compilation Example ===\n\n", .{});

    // Create an MLIR context and register all dialects and passes
    var ctx = try mlir_lisp.mlir.Context.create();
    defer ctx.destroy();
    ctx.registerAllDialects();
    mlir_lisp.mlir.Context.registerAllPasses();

    // Load the necessary dialects explicitly
    try ctx.getOrLoadDialect("func");
    try ctx.getOrLoadDialect("arith");
    try ctx.getOrLoadDialect("scf");
    try ctx.getOrLoadDialect("cf");

    std.debug.print("Loading fibonacci.mlir-lisp...\n", .{});

    // Read the fibonacci program from file
    const fib_file_path = "examples/fibonacci.mlir-lisp";
    const fib_program = std.fs.cwd().readFileAlloc(allocator, fib_file_path, 1024 * 1024) catch |err| {
        std.debug.print("Failed to read {s}: {}\n", .{ fib_file_path, err });
        std.debug.print("Make sure you're running from the project root directory.\n", .{});
        return err;
    };
    defer allocator.free(fib_program);

    std.debug.print("Program loaded ({} bytes)\n\n", .{fib_program.len});

    // Parse the program
    std.debug.print("Parsing...\n", .{});
    var tok = mlir_lisp.Tokenizer.init(allocator, fib_program);
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
    std.debug.print("Building MLIR IR...\n", .{});
    var builder = mlir_lisp.Builder.init(allocator, &ctx);
    defer builder.deinit();

    var mlir_module = try builder.buildModule(&parsed_module);
    defer mlir_module.destroy();

    std.debug.print("✓ MLIR module created successfully!\n", .{});

    // Print the MLIR before lowering
    std.debug.print("\nMLIR (before lowering):\n", .{});
    std.debug.print("----------------------------------------\n", .{});
    mlir_module.print();
    std.debug.print("----------------------------------------\n\n", .{});

    // Extract function signature before compilation
    std.debug.print("Extracting function signature...\n", .{});
    var signature = try mlir_lisp.function_call_helper.extractFunctionSignature(
        allocator,
        &mlir_module,
        "fibonacci",
    );
    defer signature.deinit();

    std.debug.print("Function signature:\n", .{});
    std.debug.print("  Inputs: ", .{});
    for (signature.inputs, 0..) |input, i| {
        if (i > 0) std.debug.print(", ", .{});
        std.debug.print("{s}", .{@tagName(input)});
    }
    std.debug.print("\n  Results: ", .{});
    for (signature.results, 0..) |result, i| {
        if (i > 0) std.debug.print(", ", .{});
        std.debug.print("{s}", .{@tagName(result)});
    }
    std.debug.print("\n\n", .{});

    // Create executor and compile
    const executor_config = mlir_lisp.ExecutorConfig{
        .opt_level = .O2, // Use O2 for better fibonacci performance
        .enable_verifier = true,
    };

    var executor = mlir_lisp.Executor.init(allocator, &ctx, executor_config);
    defer executor.deinit();

    std.debug.print("Compiling with JIT (optimization level: O2)...\n", .{});
    executor.compile(&mlir_module) catch |err| {
        std.debug.print("Compilation failed: {}\n", .{err});
        std.debug.print("\nMLIR after failed lowering:\n", .{});
        mlir_module.print();
        return err;
    };

    std.debug.print("✓ Compilation successful!\n\n", .{});

    // Test the fibonacci function with different inputs
    std.debug.print("Testing JIT'd fibonacci function:\n", .{});
    std.debug.print("----------------------------------\n", .{});

    const test_values = [_]i32{ 0, 1, 2, 3, 5, 8, 10, 15, 20 };

    for (test_values) |n| {
        // Create arguments for the call
        const args = [_]mlir_lisp.FunctionArg{
            mlir_lisp.FunctionArg{ .i32 = n },
        };

        // Call the function dynamically
        const result = try mlir_lisp.function_call_helper.callFunction(
            executor,
            "fibonacci",
            signature,
            &args,
        );

        std.debug.print("  fibonacci({}) = {}\n", .{ n, result.i32 });
    }

    std.debug.print("\n=== Fibonacci JIT compilation completed successfully! ===\n", .{});
}
