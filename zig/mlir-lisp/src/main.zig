const std = @import("std");
const mlir_lisp = @import("mlir_lisp");
const mlir = mlir_lisp.mlir;
const Reader = mlir_lisp.Reader;
const Parser = mlir_lisp.Parser;
const Builder = mlir_lisp.Builder;
const Executor = mlir_lisp.Executor;
const Repl = mlir_lisp.Repl;

pub fn main() !void {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();

    // Parse command line arguments
    const args = try std.process.argsAlloc(allocator);
    defer std.process.argsFree(allocator, args);

    // Check for flags
    if (args.len >= 2) {
        if (std.mem.eql(u8, args[1], "--repl")) {
            try Repl.run(allocator);
            return;
        }
    }

    // If a file is provided, run it. Otherwise show usage.
    if (args.len < 2) {
        std.debug.print("MLIR-Lisp Compiler/JIT\n", .{});
        std.debug.print("\nUsage:\n", .{});
        std.debug.print("  {s} <file.lisp>  - JIT compile and run a .lisp file\n", .{args[0]});
        std.debug.print("  {s} --repl       - Start interactive REPL\n", .{args[0]});
        std.debug.print("  {s}              - Run basic MLIR tests\n", .{args[0]});
        std.debug.print("\nRunning basic MLIR tests...\n\n", .{});
        try runBasicTests();
        return;
    }

    const file_path = args[1];
    try runFile(allocator, file_path);
}

fn runFile(allocator: std.mem.Allocator, file_path: []const u8) !void {
    std.debug.print("Loading file: {s}\n", .{file_path});

    // Read the file
    const source = try std.fs.cwd().readFileAlloc(allocator, file_path, 1024 * 1024);
    defer allocator.free(source);

    std.debug.print("File loaded ({} bytes)\n\n", .{source.len});

    // Create MLIR context and register dialects
    var ctx = try mlir.Context.create();
    defer ctx.destroy();
    ctx.registerAllDialects();
    mlir.Context.registerAllPasses();

    // Load the necessary dialects
    try ctx.getOrLoadDialect("func");
    try ctx.getOrLoadDialect("arith");
    try ctx.getOrLoadDialect("scf");
    try ctx.getOrLoadDialect("cf");

    // Parse the program
    std.debug.print("Parsing...\n", .{});
    var tok = mlir_lisp.Tokenizer.init(allocator, source);
    var reader = try Reader.init(allocator, &tok);
    var value = try reader.read();
    defer {
        value.deinit(allocator);
        allocator.destroy(value);
    }

    // Parse to AST
    var parser = Parser.init(allocator);
    var parsed_module = try parser.parseModule(value);
    defer parsed_module.deinit();

    // Build MLIR IR
    std.debug.print("Building MLIR IR...\n", .{});
    var builder = Builder.init(allocator, &ctx);
    defer builder.deinit();

    var mlir_module = try builder.buildModule(&parsed_module);
    defer mlir_module.destroy();

    std.debug.print("✓ MLIR module created successfully!\n", .{});

    // Print the MLIR
    std.debug.print("\nGenerated MLIR:\n", .{});
    std.debug.print("----------------------------------------\n", .{});
    mlir_module.print();
    std.debug.print("----------------------------------------\n\n", .{});

    // Create executor and compile
    const executor_config = mlir_lisp.ExecutorConfig{
        .opt_level = .O2,
        .enable_verifier = true,
    };

    var executor = Executor.init(allocator, &ctx, executor_config);
    defer executor.deinit();

    std.debug.print("Compiling with JIT (optimization level: O2)...\n", .{});
    try executor.compile(&mlir_module);

    std.debug.print("✓ Compilation successful!\n\n", .{});

    // Look up and call main() with fixed signature: () -> i64
    std.debug.print("Executing main()...\n", .{});
    const main_fn_ptr = executor.lookup("main") orelse return error.MainNotFound;
    const main_fn: *const fn () callconv(.c) i64 = @ptrCast(@alignCast(main_fn_ptr));
    const result = main_fn();

    std.debug.print("\nResult: {}\n", .{result});
}

fn runBasicTests() !void {
    // Create an MLIR context
    var ctx = try mlir.Context.create();
    defer ctx.destroy();

    std.debug.print("✓ MLIR context created\n", .{});

    // Create a module
    const loc = mlir.Location.unknown(&ctx);
    var mod = try mlir.Module.create(loc);
    defer mod.destroy();

    std.debug.print("✓ MLIR module created\n", .{});

    // Create some types
    const i32_type = mlir.Type.@"i32"(&ctx);
    const f64_type = mlir.Type.@"f64"(&ctx);
    _ = i32_type;
    _ = f64_type;

    std.debug.print("✓ Created i32 and f64 types\n", .{});
    std.debug.print("\nMLIR integration is working!\n", .{});
}
