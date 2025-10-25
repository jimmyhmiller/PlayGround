const std = @import("std");
const mlir_lisp = @import("mlir_lisp");

pub fn main() !void {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();

    // Parse command line arguments
    const args = try std.process.argsAlloc(allocator);
    defer std.process.argsFree(allocator, args);

    if (args.len < 2) {
        std.debug.print("Usage: {s} <file.mlir-lisp> [function_name] [args...]\n", .{args[0]});
        std.debug.print("\nExamples:\n", .{});
        std.debug.print("  {s} examples/fibonacci.mlir-lisp fibonacci 10\n", .{args[0]});
        std.debug.print("  {s} program.mlir-lisp main\n", .{args[0]});
        std.debug.print("\nIf no function name is provided, 'main' is assumed.\n", .{});
        return error.InvalidArgs;
    }

    const file_path = args[1];
    const function_name = if (args.len > 2) args[2] else "main";

    // Read the program
    const program = std.fs.cwd().readFileAlloc(allocator, file_path, 10 * 1024 * 1024) catch |err| {
        std.debug.print("Error reading file '{s}': {}\n", .{ file_path, err });
        return err;
    };
    defer allocator.free(program);

    std.debug.print("=== MLIR-Lisp JIT Runner ===\n", .{});
    std.debug.print("File: {s}\n", .{file_path});
    std.debug.print("Function: {s}\n\n", .{function_name});

    // Create MLIR context and register dialects
    var ctx = try mlir_lisp.mlir.Context.create();
    defer ctx.destroy();
    ctx.registerAllDialects();
    mlir_lisp.mlir.Context.registerAllPasses();

    // Load common dialects
    try ctx.getOrLoadDialect("func");
    try ctx.getOrLoadDialect("arith");
    try ctx.getOrLoadDialect("scf");
    try ctx.getOrLoadDialect("cf");

    // Parse the program
    std.debug.print("Parsing...\n", .{});
    var tok = mlir_lisp.Tokenizer.init(allocator, program);
    var r = try mlir_lisp.Reader.init(allocator, &tok);
    var value = try r.read();
    defer {
        value.deinit(allocator);
        allocator.destroy(value);
    }

    var p = mlir_lisp.Parser.init(allocator);
    var parsed_module = try p.parseModule(value);
    defer parsed_module.deinit();

    // Build MLIR IR
    std.debug.print("Building MLIR IR...\n", .{});
    var builder = mlir_lisp.Builder.init(allocator, &ctx);
    defer builder.deinit();

    var mlir_module = try builder.buildModule(&parsed_module);
    defer mlir_module.destroy();

    std.debug.print("✓ MLIR module created\n\n", .{});

    // Extract function signature
    std.debug.print("Extracting signature for '{s}'...\n", .{function_name});
    var signature = mlir_lisp.function_call_helper.extractFunctionSignature(
        allocator,
        &mlir_module,
        function_name,
    ) catch |err| {
        std.debug.print("Error: Function '{s}' not found in module\n", .{function_name});
        std.debug.print("\nAvailable functions:\n", .{});
        // Try to list available functions
        const body = mlir_module.getBody();
        var op = mlir_lisp.mlir.Block.getFirstOperation(body);
        while (!mlir_lisp.mlir.c.mlirOperationIsNull(op)) {
            const op_name = mlir_lisp.mlir.Operation.getName(op);
            if (std.mem.eql(u8, op_name, "func.func")) {
                if (mlir_lisp.mlir.Operation.getAttributeByName(op, "sym_name")) |sym_attr| {
                    const sym_ref = if (mlir_lisp.mlir.c.mlirAttributeIsAFlatSymbolRef(sym_attr))
                        mlir_lisp.mlir.c.mlirFlatSymbolRefAttrGetValue(sym_attr)
                    else
                        mlir_lisp.mlir.c.mlirStringAttrGetValue(sym_attr);
                    const sym_name = sym_ref.data[0..sym_ref.length];
                    std.debug.print("  - {s}\n", .{sym_name});
                }
            }
            op = mlir_lisp.mlir.Block.getNextInBlock(op);
        }
        return err;
    };
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

    // Parse function arguments from command line
    if (args.len - 3 != signature.inputs.len) {
        std.debug.print("Error: Function expects {} argument(s), got {}\n", .{ signature.inputs.len, args.len - 3 });
        std.debug.print("Usage: {s} {s} {s}", .{ args[0], file_path, function_name });
        for (signature.inputs) |input_type| {
            std.debug.print(" <{s}>", .{@tagName(input_type)});
        }
        std.debug.print("\n", .{});
        return error.InvalidArgs;
    }

    // Parse arguments based on signature
    var fn_args = std.ArrayList(mlir_lisp.FunctionArg).init(allocator);
    defer fn_args.deinit();

    for (signature.inputs, 0..) |input_type, i| {
        const arg_str = args[3 + i];
        const arg = try parseArgument(input_type, arg_str);
        try fn_args.append(arg);
    }

    // Create executor and compile
    const executor_config = mlir_lisp.ExecutorConfig{
        .opt_level = .O2,
        .enable_verifier = true,
    };

    var executor = mlir_lisp.Executor.init(allocator, &ctx, executor_config);
    defer executor.deinit();

    std.debug.print("Compiling with JIT (O2)...\n", .{});
    try executor.compile(&mlir_module);
    std.debug.print("✓ Compilation successful\n\n", .{});

    // Call the function
    std.debug.print("Executing {s}(", .{function_name});
    for (fn_args.items, 0..) |arg, i| {
        if (i > 0) std.debug.print(", ", .{});
        printArg(arg);
    }
    std.debug.print(")...\n", .{});

    const result = try mlir_lisp.function_call_helper.callFunction(
        executor,
        function_name,
        signature,
        fn_args.items,
    );

    std.debug.print("\nResult: ", .{});
    printArg(result);
    std.debug.print("\n", .{});
}

fn parseArgument(arg_type: mlir_lisp.RuntimeType, arg_str: []const u8) !mlir_lisp.FunctionArg {
    return switch (arg_type) {
        .i1 => mlir_lisp.FunctionArg{ .i1 = !std.mem.eql(u8, arg_str, "0") },
        .i8 => mlir_lisp.FunctionArg{ .i8 = try std.fmt.parseInt(i8, arg_str, 10) },
        .i16 => mlir_lisp.FunctionArg{ .i16 = try std.fmt.parseInt(i16, arg_str, 10) },
        .i32 => mlir_lisp.FunctionArg{ .i32 = try std.fmt.parseInt(i32, arg_str, 10) },
        .i64 => mlir_lisp.FunctionArg{ .i64 = try std.fmt.parseInt(i64, arg_str, 10) },
        .f32 => mlir_lisp.FunctionArg{ .f32 = try std.fmt.parseFloat(f32, arg_str) },
        .f64 => mlir_lisp.FunctionArg{ .f64 = try std.fmt.parseFloat(f64, arg_str) },
    };
}

fn printArg(arg: mlir_lisp.FunctionArg) void {
    switch (arg) {
        .i1 => |v| std.debug.print("{}", .{v}),
        .i8 => |v| std.debug.print("{}", .{v}),
        .i16 => |v| std.debug.print("{}", .{v}),
        .i32 => |v| std.debug.print("{}", .{v}),
        .i64 => |v| std.debug.print("{}", .{v}),
        .f32 => |v| std.debug.print("{d}", .{v}),
        .f64 => |v| std.debug.print("{d}", .{v}),
    }
}
