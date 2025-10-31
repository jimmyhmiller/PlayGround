/// Temporary testing binary for JIT compilation with runtime symbol resolution
/// This binary compiles MLIR-Lisp code and calls exampleTransformCallToOperation specifically

const std = @import("std");
const mlir_lisp = @import("mlir_lisp");
const mlir = mlir_lisp.mlir;
const Reader = mlir_lisp.Reader;
const Parser = mlir_lisp.Parser;
const Builder = mlir_lisp.Builder;
const Executor = mlir_lisp.Executor;
const registerAllRuntimeSymbols = mlir_lisp.runtime_symbols.registerAllRuntimeSymbols;

pub fn main() !void {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();

    // Parse command line arguments
    const args = try std.process.argsAlloc(allocator);
    defer std.process.argsFree(allocator, args);

    if (args.len < 2) {
        std.debug.print("Usage: {s} <file.mlir-lisp>\n", .{args[0]});
        std.debug.print("  Compiles the MLIR-Lisp file with runtime symbol resolution\n", .{});
        std.debug.print("  and calls exampleTransformCallToOperation\n", .{});
        return error.MissingArgument;
    }

    const file_path = args[1];

    try runWithRuntimeSymbols(allocator, file_path);
}

fn runWithRuntimeSymbols(backing_allocator: std.mem.Allocator, file_path: []const u8) !void {
    std.debug.print("=== MLIR-Lisp JIT with Runtime Symbol Resolution ===\n\n", .{});
    std.debug.print("Loading file: {s}\n", .{file_path});

    // Create an arena for the entire compilation process
    var arena = std.heap.ArenaAllocator.init(backing_allocator);
    defer arena.deinit();
    const allocator = arena.allocator();

    // Read the file
    const source = try std.fs.cwd().readFileAlloc(allocator, file_path, 10 * 1024 * 1024); // 10MB limit

    std.debug.print("File loaded ({} bytes)\n", .{source.len});

    // Create MLIR context and register dialects
    var ctx = try mlir.Context.create();
    defer ctx.destroy();
    ctx.registerAllDialects();
    mlir.Context.registerAllPasses();

    // Load the necessary dialects
    try ctx.getOrLoadDialect("func");
    try ctx.getOrLoadDialect("llvm");
    try ctx.getOrLoadDialect("arith");
    try ctx.getOrLoadDialect("scf");
    try ctx.getOrLoadDialect("cf");

    // Parse the program
    std.debug.print("Parsing MLIR-Lisp...\n", .{});
    var tok = mlir_lisp.Tokenizer.init(allocator, source);
    var reader = Reader.init(allocator, &tok) catch |err| {
        const pos = tok.getPosition();
        std.debug.print("\nerror: {} at line {}, column {}\n", .{ err, pos.line, pos.column });
        return err;
    };
    const value = reader.read() catch |err| {
        const pos = tok.getPosition();
        std.debug.print("\nerror: {} at line {}, column {}\n", .{ err, pos.line, pos.column });
        return err;
    };

    // Parse to AST
    std.debug.print("Parsing to AST...\n", .{});
    var parser = Parser.init(allocator, source);
    var parsed_module = try parser.parseModule(value);
    defer parsed_module.deinit();

    // Build MLIR IR
    std.debug.print("Building MLIR IR...\n", .{});
    var builder = Builder.init(allocator, &ctx);
    defer builder.deinit();

    var mlir_module = try builder.buildModule(&parsed_module);
    defer mlir_module.destroy();

    std.debug.print("✓ MLIR module created successfully!\n\n", .{});

    // Create executor with optimization
    const executor_config = mlir_lisp.ExecutorConfig{
        .opt_level = .O2,
        .enable_verifier = true,
    };

    var executor = Executor.init(allocator, &ctx, executor_config);
    defer executor.deinit();

    // Register all runtime symbols BEFORE compilation
    std.debug.print("Registering runtime symbols...\n", .{});
    registerAllRuntimeSymbols(&executor);
    std.debug.print("✓ Runtime symbols registered (C stdlib + c_api.zig)\n\n", .{});

    // Compile the module
    std.debug.print("Compiling with JIT (optimization level: O2)...\n", .{});
    try executor.compile(&mlir_module);
    std.debug.print("✓ Compilation successful!\n\n", .{});

    // Look up exampleTransformCallToOperation
    std.debug.print("Looking up exampleTransformCallToOperation...\n", .{});
    const fn_ptr = executor.lookup("exampleTransformCallToOperation");

    if (fn_ptr == null) {
        std.debug.print("ERROR: exampleTransformCallToOperation not found in compiled module\n", .{});
        std.debug.print("Available symbols might be:\n", .{});
        std.debug.print("  - Try checking with different name mangling\n", .{});
        return error.FunctionNotFound;
    }

    std.debug.print("✓ Found exampleTransformCallToOperation\n\n", .{});

    // Set the global arena allocator so JIT'd code uses our arena
    std.debug.print("Setting global arena allocator for JIT'd code...\n", .{});
    const set_arena_fn_ptr = executor.lookup("allocator_set_global_arena");
    if (set_arena_fn_ptr) |set_fn| {
        const set_arena_fn: *const fn (?*anyopaque) callconv(.c) void = @ptrCast(@alignCast(set_fn));
        // Create a pointer to our arena's allocator
        var arena_alloc = arena.allocator();
        set_arena_fn(@ptrCast(&arena_alloc));
        std.debug.print("✓ Global arena set\n\n", .{});
    }

    // Call the function - signature is ptr () according to MLIR-Lisp
    std.debug.print("Calling exampleTransformCallToOperation()...\n", .{});
    const transform_fn: *const fn () callconv(.c) ?*anyopaque = @ptrCast(@alignCast(fn_ptr));
    const result = transform_fn();

    std.debug.print("\n=== Execution Complete ===\n", .{});
    if (result) |ptr| {
        std.debug.print("Result: pointer value = 0x{x}\n\n", .{@intFromPtr(ptr)});

        // Cast to Value and print it using the built-in printer
        const value_ptr: *mlir_lisp.Value = @ptrCast(@alignCast(ptr));
        std.debug.print("Result Value (lisp syntax):\n", .{});

        var buffer = std.ArrayList(u8){};
        defer buffer.deinit(backing_allocator);
        value_ptr.print(buffer.writer(backing_allocator)) catch |err| {
            std.debug.print("Error printing value: {}\n", .{err});
        };
        std.debug.print("{s}\n", .{buffer.items});
    } else {
        std.debug.print("Result: null pointer\n", .{});
        std.debug.print("(Function returned null - may indicate an error)\n", .{});
    }
}
