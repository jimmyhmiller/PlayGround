const std = @import("std");
const mlir_lisp = @import("mlir_lisp");
const mlir = mlir_lisp.mlir;
const Reader = mlir_lisp.Reader;
const Value = mlir_lisp.Value;
const PersistentVector = mlir_lisp.PersistentVector;
const Parser = mlir_lisp.Parser;
const Builder = mlir_lisp.Builder;
const Executor = mlir_lisp.Executor;
const Repl = mlir_lisp.Repl;
const MacroExpander = mlir_lisp.MacroExpander;
const builtin_macros = mlir_lisp.builtin_macros;

/// Helper function to wrap multiple top-level forms in an implicit list
fn createImplicitList(allocator: std.mem.Allocator, forms: PersistentVector(*Value)) !*Value {
    const value = try allocator.create(Value);
    value.* = Value{
        .type = .list,
        .data = .{ .list = forms },
    };
    return value;
}

pub fn main() !void {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();

    // Parse command line arguments
    const args = try std.process.argsAlloc(allocator);
    defer std.process.argsFree(allocator, args);

    // Parse flags
    var use_generic_format = false;
    var file_path: ?[]const u8 = null;

    var i: usize = 1;
    while (i < args.len) : (i += 1) {
        if (std.mem.eql(u8, args[i], "--repl")) {
            try Repl.run(allocator);
            return;
        } else if (std.mem.eql(u8, args[i], "--generic")) {
            use_generic_format = true;
        } else if (std.mem.eql(u8, args[i], "-g")) {
            use_generic_format = true;
        } else {
            file_path = args[i];
        }
    }

    // If a file is provided, run it. Otherwise show usage.
    if (file_path == null) {
        std.debug.print("MLIR-Lisp Compiler/JIT\n", .{});
        std.debug.print("\nUsage:\n", .{});
        std.debug.print("  {s} [--generic|-g] <file.lisp>  - JIT compile and run a .lisp file\n", .{args[0]});
        std.debug.print("  {s} --repl                       - Start interactive REPL\n", .{args[0]});
        std.debug.print("  {s}                              - Run basic MLIR tests\n", .{args[0]});
        std.debug.print("\nOptions:\n", .{});
        std.debug.print("  --generic, -g  Print MLIR in generic form (shows all attributes)\n", .{});
        std.debug.print("\nRunning basic MLIR tests...\n\n", .{});
        try runBasicTests();
        return;
    }

    try runFile(allocator, file_path.?, use_generic_format);
}

fn printErrorLocation(source: []const u8, line: usize, column: usize) void {
    if (line == 0 or column == 0) return;

    // Find the line in the source
    var current_line: usize = 1;
    var line_start: usize = 0;
    var i: usize = 0;

    while (i < source.len) : (i += 1) {
        if (current_line == line) {
            // Find the end of this line
            var line_end = i;
            while (line_end < source.len and source[line_end] != '\n') : (line_end += 1) {}

            // Print the line
            const line_content = source[line_start..line_end];
            std.debug.print("{s}\n", .{line_content});

            // Print pointer to error location
            var j: usize = 0;
            while (j < column - 1) : (j += 1) {
                std.debug.print(" ", .{});
            }
            std.debug.print("^\n", .{});
            return;
        }

        if (source[i] == '\n') {
            current_line += 1;
            line_start = i + 1;
        }
    }
}

fn runFile(backing_allocator: std.mem.Allocator, file_path: []const u8, use_generic_format: bool) !void {
    std.debug.print("Loading file: {s}\n", .{file_path});

    // Create an arena for the entire compilation process
    var arena = std.heap.ArenaAllocator.init(backing_allocator);
    defer arena.deinit();
    const allocator = arena.allocator();

    // Read the file
    const source = try std.fs.cwd().readFileAlloc(allocator, file_path, 1024 * 1024);

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
    try ctx.getOrLoadDialect("memref");
    try ctx.getOrLoadDialect("affine");
    try ctx.getOrLoadDialect("cf");

    // Parse the program
    std.debug.print("Parsing...\n", .{});
    var tok = mlir_lisp.Tokenizer.init(allocator, source);
    var reader = Reader.init(allocator, &tok) catch |err| {
        const pos = tok.getPosition();
        std.debug.print("\nerror: {} at line {}, column {}\n", .{ err, pos.line, pos.column });
        printErrorLocation(source, pos.line, pos.column);
        return err;
    };
    // Read all top-level forms
    const forms = reader.readAll() catch |err| {
        const pos = tok.getPosition();
        std.debug.print("\nerror: {} at line {}, column {}\n", .{ err, pos.line, pos.column });
        printErrorLocation(source, pos.line, pos.column);
        return err;
    };
    // If there's only one form, use it directly; otherwise wrap in a list
    const value = if (forms.len() == 1)
        forms.at(0)
    else
        try createImplicitList(allocator, forms);
    // No need to manually free - arena handles it

    // Macro expansion
    std.debug.print("Expanding macros...\n", .{});
    var macro_expander = MacroExpander.init(allocator);
    defer macro_expander.deinit();
    try builtin_macros.registerBuiltinMacros(&macro_expander);

    const expanded_value = try macro_expander.expandAll(value);

    // Operation flattening (convert nested operations to flat SSA form)
    std.debug.print("Flattening operations...\n", .{});
    const OperationFlattener = mlir_lisp.OperationFlattener;
    var flattener = OperationFlattener.init(allocator);
    const flattened_value = try flattener.flattenModule(expanded_value);

    // Parse to AST
    var parser = Parser.init(allocator, source);
    var parsed_module = try parser.parseModule(flattened_value);
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
    if (use_generic_format) {
        mlir_module.printGeneric();
    } else {
        mlir_module.print();
    }
    std.debug.print("----------------------------------------\n\n", .{});

    // Check if the module has a main function
    const has_main = blk: {
        for (parsed_module.operations) |*op| {
            if (std.mem.eql(u8, op.name, "func.func")) {
                // Check for sym_name attribute
                for (op.attributes) |*attr| {
                    if (std.mem.eql(u8, attr.key, "sym_name")) {
                        // The value should be a symbol starting with @
                        const val = attr.value.value;
                        if (val.type == .symbol and std.mem.eql(u8, val.data.atom, "@main")) {
                            break :blk true;
                        }
                    }
                }
            }
        }
        break :blk false;
    };

    if (!has_main) {
        std.debug.print("Note: No main() function found - skipping JIT execution\n", .{});
        std.debug.print("(Add a main() function with signature () -> i64 to enable execution)\n", .{});
        return;
    }

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

    // Try to look up main with MLIR's C interface mangling
    var main_fn_ptr = executor.lookup("_mlir_ciface_main");
    if (main_fn_ptr == null) {
        // Try without the prefix
        main_fn_ptr = executor.lookup("main");
    }

    if (main_fn_ptr == null) {
        // This shouldn't happen since we validated main exists, but handle it anyway
        std.debug.print("ERROR: main() function not found after compilation\n", .{});
        return error.MainNotFound;
    }

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
