const std = @import("std");
const mlir_lisp = @import("mlir_lisp");
const mlir = mlir_lisp.mlir;
const Reader = mlir_lisp.Reader;
const Value = mlir_lisp.Value;
const PersistentVector = mlir_lisp.PersistentVector;
const Parser = mlir_lisp.Parser;
const Operation = mlir_lisp.Operation;
const Builder = mlir_lisp.Builder;
const Executor = mlir_lisp.Executor;
const Repl = mlir_lisp.Repl;
const MacroExpander = mlir_lisp.MacroExpander;
const builtin_macros = mlir_lisp.builtin_macros;
const DialectRegistry = mlir_lisp.DialectRegistry;
const metadata_detector = mlir_lisp.metadata_detector;
const gpu_lowering = @import("gpu_lowering.zig");

/// Helper function to recursively check for GPU operations in regions
fn hasGPUOpsInRegions(op: *const Operation) bool {
    for (op.regions) |*region| {
        for (region.blocks) |*block| {
            for (block.operations) |*nested_op| {
                if (std.mem.startsWith(u8, nested_op.name, "gpu.") or
                    std.mem.indexOf(u8, nested_op.name, "gpu.") != null) {
                    return true;
                }
                if (hasGPUOpsInRegions(nested_op)) {
                    return true;
                }
            }
        }
    }
    return false;
}

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
    var transform_file_path: ?[]const u8 = null;

    var i: usize = 1;
    while (i < args.len) : (i += 1) {
        if (std.mem.eql(u8, args[i], "--repl")) {
            try Repl.run(allocator);
            return;
        } else if (std.mem.eql(u8, args[i], "--generic")) {
            use_generic_format = true;
        } else if (std.mem.eql(u8, args[i], "-g")) {
            use_generic_format = true;
        } else if (std.mem.eql(u8, args[i], "--transform") or std.mem.eql(u8, args[i], "-t")) {
            i += 1;
            if (i < args.len) {
                transform_file_path = args[i];
            }
        } else {
            file_path = args[i];
        }
    }

    // If a file is provided, run it. Otherwise show usage.
    if (file_path == null) {
        std.debug.print("MLIR-Lisp Compiler/JIT\n", .{});
        std.debug.print("\nUsage:\n", .{});
        std.debug.print("  {s} [--generic|-g] [-t transform.lisp] <file.lisp>  - JIT compile and run a .lisp file\n", .{args[0]});
        std.debug.print("  {s} --repl                       - Start interactive REPL\n", .{args[0]});
        std.debug.print("  {s}                              - Run basic MLIR tests\n", .{args[0]});
        std.debug.print("\nOptions:\n", .{});
        std.debug.print("  --generic, -g      Print MLIR in generic form (shows all attributes)\n", .{});
        std.debug.print("  --transform, -t    Load transform operations from separate file\n", .{});
        std.debug.print("\nRunning basic MLIR tests...\n\n", .{});
        try runBasicTests();
        return;
    }

    try runFile(allocator, file_path.?, transform_file_path, use_generic_format);
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

fn runFile(backing_allocator: std.mem.Allocator, file_path: []const u8, transform_file_path: ?[]const u8, use_generic_format: bool) !void {
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
    try ctx.getOrLoadDialect("irdl");
    try ctx.getOrLoadDialect("transform");

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
    var macro_expander = MacroExpander.init(allocator);
    defer macro_expander.deinit();
    try builtin_macros.registerBuiltinMacros(&macro_expander);

    const expanded_value = try macro_expander.expandAll(value);

    const OperationFlattener = mlir_lisp.OperationFlattener;
    var flattener = OperationFlattener.init(allocator);
    const flattened_value = try flattener.flattenModule(expanded_value);

    // Parse to AST
    var parser = Parser.init(allocator, source);
    var parsed_module = try parser.parseModule(flattened_value);
    defer parsed_module.deinit();

    // ========== METADATA-BASED COMPILATION: IRDL + TRANSFORM SUPPORT ==========

    std.debug.print("Building MLIR IR...\n", .{});
    var builder = Builder.init(allocator, &ctx);
    defer builder.deinit();

    // Separate metadata modules from application modules
    var metadata_ops = std.ArrayList(*const mlir_lisp.parser.Operation){};
    defer metadata_ops.deinit(allocator);
    var app_ops = std.ArrayList(*const mlir_lisp.parser.Operation){};
    defer app_ops.deinit(allocator);

    for (parsed_module.operations) |*op| {
        if (metadata_detector.isMetadataModule(op)) {
            try metadata_ops.append(allocator, op);
        } else {
            try app_ops.append(allocator, op);
        }
    }

    std.debug.print("Found {} metadata module(s), {} application module(s)\n", .{metadata_ops.items.len, app_ops.items.len});

    // Process metadata modules: IRDL and transforms
    var has_irdl = false;
    var transform_module: mlir.Module = undefined;
    var has_transforms = false;

    for (metadata_ops.items) |meta_op| {
        const meta_type = metadata_detector.detectMetadataType(meta_op);

        switch (meta_type) {
            .irdl => {
                std.debug.print("Processing IRDL metadata module...\n", .{});
                var irdl_module = try builder.buildSingleOperation(meta_op);

                // Load IRDL dialects
                try irdl_module.loadIRDLDialects();
                std.debug.print("  ✓ IRDL dialects loaded into context\n", .{});
                has_irdl = true;

                // NOTE: Don't destroy irdl_module! The context holds references.
            },
            .transform => {
                std.debug.print("Processing transform metadata module...\n", .{});
                transform_module = try builder.buildSingleOperation(meta_op);
                has_transforms = true;

                std.debug.print("  Transform module structure:\n", .{});
                transform_module.print();
                std.debug.print("\n", .{});
            },
            .unknown => {
                std.debug.print("Warning: Metadata module without IRDL or transform operations\n", .{});
            },
        }
    }

    // Handle transforms from separate file (if --transform flag used)
    if (transform_file_path) |transform_path| {
        std.debug.print("Loading transforms from file: {s}\n", .{transform_path});

        const transform_source = try std.fs.cwd().readFileAlloc(allocator, transform_path, 1024 * 1024);
        std.debug.print("  Transform file loaded ({} bytes)\n", .{transform_source.len});

        // Parse and build the transform module
        var transform_tok = mlir_lisp.Tokenizer.init(allocator, transform_source);
        var transform_reader = try Reader.init(allocator, &transform_tok);
        const transform_forms = try transform_reader.readAll();

        const transform_value = if (transform_forms.len() == 1)
            transform_forms.at(0)
        else
            try createImplicitList(allocator, transform_forms);

        var transform_macro_expander = MacroExpander.init(allocator);
        defer transform_macro_expander.deinit();
        try builtin_macros.registerBuiltinMacros(&transform_macro_expander);
        const expanded_transform = try transform_macro_expander.expandAll(transform_value);

        var transform_flattener = mlir_lisp.OperationFlattener.init(allocator);
        const flattened_transform = try transform_flattener.flattenModule(expanded_transform);

        var transform_parser = Parser.init(allocator, transform_source);
        var parsed_transform_module = try transform_parser.parseModule(flattened_transform);
        defer parsed_transform_module.deinit();

        var transform_builder = Builder.init(allocator, &ctx);
        defer transform_builder.deinit();

        // Override any transforms from same file with file transforms
        if (has_transforms) {
            transform_module.destroy();
        }
        transform_module = try transform_builder.buildModule(&parsed_transform_module);
        has_transforms = true;

        std.debug.print("  ✓ Transform module loaded from file\n", .{});
    }

    // Build application code (non-metadata modules)
    std.debug.print("Building application code ({} module(s))...\n", .{app_ops.items.len});
    var mlir_module = try builder.buildFromOperations(app_ops.items);
    defer mlir_module.destroy();

    // If we have transforms, they need to be destroyed later (after execution)
    // The destroy is done conditionally in the executor section

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

    // Check if the module has a main function (recursively search through regions)
    const has_main = blk: {
        // Helper to recursively search for main
        const Searcher = struct {
            fn hasMainInOp(op: *const Operation) bool {
                // Check if this operation is func.func @main
                if (std.mem.eql(u8, op.name, "func.func")) {
                    for (op.attributes) |*attr| {
                        if (std.mem.eql(u8, attr.key, "sym_name")) {
                            const val = attr.value.value;
                            if (val.type == .symbol and std.mem.eql(u8, val.data.atom, "@main")) {
                                return true;
                            }
                        }
                    }
                }

                // Search in regions
                for (op.regions) |*region| {
                    for (region.blocks) |*block| {
                        for (block.operations) |*nested_op| {
                            if (hasMainInOp(nested_op)) {
                                return true;
                            }
                        }
                    }
                }

                return false;
            }
        };

        for (parsed_module.operations) |*op| {
            if (Searcher.hasMainInOp(op)) {
                break :blk true;
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

    // Set transforms if any were found
    if (has_transforms) {
        std.debug.print("  Extracting transform operation...\n", .{});

        // Get the first operation from the transform module body
        const transform_module_op = transform_module.getOperation();
        const body_region = mlir.c.mlirOperationGetRegion(transform_module_op, 0);
        const body_block = mlir.c.mlirRegionGetFirstBlock(body_region);
        const first_op = mlir.c.mlirBlockGetFirstOperation(body_block);

        if (!mlir.c.mlirOperationIsNull(first_op)) {
            const op_name = mlir.Operation.getName(first_op);
            std.debug.print("  Found transform operation: {s}\n", .{op_name});

            // If it's a builtin.module with {:metadata unit} attribute, extract inner operation
            if (std.mem.eql(u8, op_name, "builtin.module")) {
                const attr_name_str = "metadata";
                const attr_name_ref = mlir.c.mlirStringRefCreate(attr_name_str.ptr, attr_name_str.len);
                const attr = mlir.c.mlirOperationGetAttributeByName(first_op, attr_name_ref);

                if (!mlir.c.mlirAttributeIsNull(attr)) {
                    // This is a metadata module, extract the first operation inside it
                    const inner_region = mlir.c.mlirOperationGetRegion(first_op, 0);
                    const inner_block = mlir.c.mlirRegionGetFirstBlock(inner_region);
                    const inner_op = mlir.c.mlirBlockGetFirstOperation(inner_block);

                    if (!mlir.c.mlirOperationIsNull(inner_op)) {
                        std.debug.print("  Using inner operation: {s}\n", .{mlir.Operation.getName(inner_op)});
                        executor.setTransform(inner_op, first_op);
                    }
                } else {
                    // Regular builtin.module (from --transform file), use module and first op
                    executor.setTransform(first_op, transform_module_op);
                }
            } else {
                // Direct transform operation (from --transform file)
                executor.setTransform(first_op, transform_module_op);
            }
        }
    }

    // Check if the module contains GPU operations and apply lowering if needed
    const has_gpu_ops = blk: {
        // Check if any operation name contains "gpu."
        for (parsed_module.operations) |*op| {
            if (std.mem.startsWith(u8, op.name, "gpu.") or
                std.mem.indexOf(u8, op.name, "gpu.") != null) {
                break :blk true;
            }
            // Also check nested operations
            if (hasGPUOpsInRegions(op)) {
                break :blk true;
            }
        }
        break :blk false;
    };

    if (has_gpu_ops) {
        std.debug.print("Detected GPU operations, applying lowering passes...\n", .{});
        const gpu_type = gpu_lowering.detectGPU();
        std.debug.print("Detected GPU type: {s}\n", .{@tagName(gpu_type)});
        try gpu_lowering.applyGPULoweringPasses(allocator, &ctx, &mlir_module, gpu_type);

        std.debug.print("\nLowered MLIR (after GPU passes):\n", .{});
        std.debug.print("----------------------------------------\n", .{});
        mlir_module.print();
        std.debug.print("----------------------------------------\n\n", .{});
    }

    // Check if GPU ops still remain after lowering
    const still_has_gpu = blk: {
        for (parsed_module.operations) |*op| {
            if (std.mem.startsWith(u8, op.name, "gpu.") or
                std.mem.indexOf(u8, op.name, "gpu.") != null) {
                break :blk true;
            }
            if (hasGPUOpsInRegions(op)) {
                break :blk true;
            }
        }
        break :blk false;
    };

    if (still_has_gpu) {
        std.debug.print("Note: GPU operations remain in IR after lowering.\n", .{});
        std.debug.print("      Skipping JIT execution (GPU ops cannot be JIT compiled directly).\n", .{});
        std.debug.print("      The MLIR above shows the successfully lowered GPU code.\n\n", .{});
        return;
    }

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
