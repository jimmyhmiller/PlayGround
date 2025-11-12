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
const macro_compressor = mlir_lisp.macro_compressor;
const vector = mlir_lisp.vector;

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
    var macroify = false;
    var macro_expand = false;

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
        } else if (std.mem.eql(u8, args[i], "--macroify")) {
            macroify = true;
        } else if (std.mem.eql(u8, args[i], "--macro-expand")) {
            macro_expand = true;
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
        std.debug.print("  {s} --macroify <file.lisp>       - Compress verbose operations to macro form (print only)\n", .{args[0]});
        std.debug.print("  {s} --macro-expand <file.lisp>   - Expand macros to verbose form (print only)\n", .{args[0]});
        std.debug.print("  {s}                              - Run basic MLIR tests\n", .{args[0]});
        std.debug.print("\nOptions:\n", .{});
        std.debug.print("  --generic, -g      Print MLIR in generic form (shows all attributes)\n", .{});
        std.debug.print("  --transform, -t    Load transform operations from separate file\n", .{});
        std.debug.print("  --macroify         Compress verbose MLIR operations to compact macros\n", .{});
        std.debug.print("  --macro-expand     Expand macros to verbose MLIR operation form\n", .{});
        std.debug.print("\nRunning basic MLIR tests...\n\n", .{});
        try runBasicTests();
        return;
    }

    // Check for mutually exclusive flags
    if (macroify and macro_expand) {
        std.debug.print("Error: --macroify and --macro-expand cannot be used together\n", .{});
        return error.ConflictingFlags;
    }

    // Handle transformation-only modes
    if (macroify or macro_expand) {
        try runTransformation(allocator, file_path.?, macroify);
        return;
    }

    try runFile(allocator, file_path.?, transform_file_path, use_generic_format);
}

/// Check if a list contains a region (recursively search first level children)
fn containsRegion(list: vector.PersistentVector(*Value)) bool {
    for (list.slice()) |elem| {
        if (elem.type == .list) {
            const inner_list = elem.data.list;
            if (inner_list.len() > 0) {
                const first = inner_list.at(0);
                if (first.type == .identifier and std.mem.eql(u8, first.data.atom, "region")) {
                    return true;
                }
            }
        }
    }
    return false;
}

/// Pretty print a Value with indentation
fn prettyPrintValue(value: *const Value, writer: anytype, indent: usize) error{OutOfMemory}!void {
    const indent_str = "  ";

    switch (value.type) {
        .list => {
            try writer.writeAll("(");
            const list = value.data.list;
            if (list.len() == 0) {
                try writer.writeAll(")");
                return;
            }

            // Check what kind of list this is
            const first = list.at(0);
            const is_operation = first.type == .identifier and std.mem.eql(u8, first.data.atom, "operation");
            const is_region = first.type == .identifier and std.mem.eql(u8, first.data.atom, "region");
            const is_block = first.type == .identifier and std.mem.eql(u8, first.data.atom, "block");
            const is_mlir = first.type == .identifier and std.mem.eql(u8, first.data.atom, "mlir");
            const is_op = first.type == .identifier and std.mem.eql(u8, first.data.atom, "op");
            const is_call = first.type == .identifier and std.mem.eql(u8, first.data.atom, "call");
            const is_constant = first.type == .identifier and std.mem.eql(u8, first.data.atom, "constant");
            const is_defn = first.type == .identifier and std.mem.eql(u8, first.data.atom, "defn");
            const is_return = first.type == .identifier and std.mem.eql(u8, first.data.atom, "return");

            // Determine if we need multiline formatting
            const needs_multiline = blk: {
                // Always multiline for these structural forms
                if (is_operation or is_region or is_block or is_mlir) break :blk true;

                // defn needs special multiline handling
                if (is_defn) break :blk true;

                // Always single line for these macros
                if (is_call or is_constant or is_return) break :blk false;

                // For 'op' macro calls, check if they have regions
                if (is_op) {
                    // If op has regions as children, format multiline
                    if (containsRegion(list)) break :blk true;
                    // Simple op calls with ≤ 4 elements can be single line
                    if (list.len() <= 4) break :blk false;
                    break :blk true;
                }

                // For other lists, use a heuristic
                // Single line if short and no regions
                if (list.len() <= 4 and !containsRegion(list)) break :blk false;

                break :blk true;
            };

            if (needs_multiline) {
                if (is_defn) {
                    // Special formatting for defn: (defn name [args] return-type\n  body...)
                    // Elements: [0]=defn [1]=name [2]=args [3]=return-type [4+]=body
                    for (list.slice(), 0..) |elem, i| {
                        if (i == 0) {
                            // defn keyword
                            try prettyPrintValue(elem, writer, indent);
                        } else if (i <= 3) {
                            // name, args, return-type on same line
                            try writer.writeAll(" ");
                            try prettyPrintValue(elem, writer, indent);
                        } else {
                            // body on new lines
                            try writer.writeAll("\n");
                            for (0..indent + 1) |_| try writer.writeAll(indent_str);
                            try prettyPrintValue(elem, writer, indent + 1);
                        }
                    }
                } else {
                    // Standard multiline formatting
                    for (list.slice(), 0..) |elem, i| {
                        if (i == 0) {
                            try prettyPrintValue(elem, writer, indent);
                        } else {
                            try writer.writeAll("\n");
                            for (0..indent + 1) |_| try writer.writeAll(indent_str);
                            try prettyPrintValue(elem, writer, indent + 1);
                        }
                    }
                }
            } else {
                for (list.slice(), 0..) |elem, i| {
                    if (i > 0) try writer.writeAll(" ");
                    try prettyPrintValue(elem, writer, indent);
                }
            }
            try writer.writeAll(")");
        },
        .vector => {
            try writer.writeAll("[");
            const vec = value.data.vector;
            for (vec.slice(), 0..) |elem, i| {
                if (i > 0) try writer.writeAll(" ");
                try prettyPrintValue(elem, writer, indent);
            }
            try writer.writeAll("]");
        },
        .map => {
            try writer.writeAll("{");
            const map = value.data.map;
            for (map.slice(), 0..) |elem, i| {
                if (i > 0) try writer.writeAll(" ");
                try prettyPrintValue(elem, writer, indent);
            }
            try writer.writeAll("}");
        },
        .has_type => {
            try writer.writeAll("(: ");
            try prettyPrintValue(value.data.has_type.value, writer, indent);
            try writer.writeAll(" ");
            try prettyPrintValue(value.data.has_type.type_expr, writer, indent);
            try writer.writeAll(")");
        },
        .attr_expr => {
            try writer.writeAll("#");
            try prettyPrintValue(value.data.attr_expr, writer, indent);
        },
        .function_type => {
            try writer.writeAll("(!function (inputs");
            const inputs = value.data.function_type.inputs;
            for (inputs.slice()) |input| {
                try writer.writeAll(" ");
                try prettyPrintValue(input, writer, indent);
            }
            try writer.writeAll(") (results");
            const results = value.data.function_type.results;
            for (results.slice()) |result| {
                try writer.writeAll(" ");
                try prettyPrintValue(result, writer, indent);
            }
            try writer.writeAll("))");
        },
        // Atoms
        .identifier, .number, .value_id, .block_id, .symbol => {
            try writer.writeAll(value.data.atom);
        },
        .keyword => {
            try writer.print(":{s}", .{value.keywordToName()});
        },
        .string => {
            // String atoms already include quotes
            try writer.writeAll(value.data.atom);
        },
        .type => {
            try writer.writeAll(value.data.type);
        },
        .true_lit => try writer.writeAll("true"),
        .false_lit => try writer.writeAll("false"),
    }
}

/// Run macro transformation (--macroify or --macro-expand) and print result
fn runTransformation(backing_allocator: std.mem.Allocator, file_path: []const u8, do_macroify: bool) !void {
    // Create an arena for the transformation process
    var arena = std.heap.ArenaAllocator.init(backing_allocator);
    defer arena.deinit();
    const allocator = arena.allocator();

    // Read the file
    const source = try std.fs.cwd().readFileAlloc(allocator, file_path, 1024 * 1024);

    // Parse the program
    var tok = mlir_lisp.Tokenizer.init(allocator, source);
    var reader = Reader.init(allocator, &tok) catch |err| {
        const pos = tok.getPosition();
        std.debug.print("error: {} at line {}, column {}\n", .{ err, pos.line, pos.column });
        printErrorLocation(source, pos.line, pos.column);
        return err;
    };

    // Read all top-level forms
    const forms = reader.readAll() catch |err| {
        const pos = tok.getPosition();
        std.debug.print("error: {} at line {}, column {}\n", .{ err, pos.line, pos.column });
        printErrorLocation(source, pos.line, pos.column);
        return err;
    };

    // If there's only one form, use it directly; otherwise wrap in a list
    const value = if (forms.len() == 1)
        forms.at(0)
    else
        try createImplicitList(allocator, forms);

    const transformed_value = if (do_macroify) blk: {
        // Macroify: compress verbose operations to macro form
        const compressed = try macro_compressor.compressMacros(allocator, value);
        // Unwrap non-metadata modules to remove boilerplate
        const unwrapped = try macro_compressor.unwrapNonMetadataModule(allocator, compressed);
        break :blk unwrapped;
    } else blk: {
        // Macro-expand: expand macros to verbose form
        var macro_expander = MacroExpander.init(allocator);
        defer macro_expander.deinit();
        try builtin_macros.registerBuiltinMacros(&macro_expander);
        break :blk try macro_expander.expandAll(value);
    };

    // Print the result to stdout (not stderr!)
    var buffer = std.ArrayList(u8){};
    defer buffer.deinit(allocator);
    const writer = buffer.writer(allocator);

    // Check if result is a plain list (multiple top-level forms from unwrapping)
    // vs a special form (like (mlir ...), (defn ...), etc.)
    const should_print_as_multiple = blk: {
        if (transformed_value.type != .list) break :blk false;
        const list = transformed_value.data.list;
        if (list.len() == 0) break :blk false;

        // If it starts with a known keyword/form, it's a single form
        const first = list.at(0);
        if (first.type != .identifier) break :blk true;

        // Known form identifiers that indicate single forms
        const known_forms = [_][]const u8{ "mlir", "operation", "defn", "op", "region", "block", "call", "constant", "return", "+", "*" };
        for (known_forms) |form| {
            if (std.mem.eql(u8, first.data.atom, form)) break :blk false;
        }

        // Otherwise, it's likely a list of top-level forms
        break :blk true;
    };

    if (should_print_as_multiple) {
        // Print each element as a separate top-level form
        const list = transformed_value.data.list;
        for (list.slice(), 0..) |elem, i| {
            if (i > 0) try writer.writeAll("\n\n");
            try prettyPrintValue(elem, writer, 0);
        }
        try writer.writeAll("\n");
    } else {
        // Print as a single form
        try prettyPrintValue(transformed_value, writer, 0);
        try writer.writeAll("\n");
    }

    // Write to stdout file descriptor
    const stdout_fd = std.posix.STDOUT_FILENO;
    _ = try std.posix.write(stdout_fd, buffer.items);
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
    ctx.registerAllLLVMTranslations();
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

    // Register type and attribute aliases before building
    try builder.registerTypeAliases(parsed_module.type_aliases);
    try builder.registerAttributeAliases(parsed_module.attribute_aliases);

    // Build application code (non-metadata modules)
    std.debug.print("Building application code ({} module(s))...\n", .{app_ops.items.len});
    var mlir_module = try builder.buildFromOperations(app_ops.items);
    defer mlir_module.destroy();

    // If we have transforms, they need to be destroyed later (after execution)
    // The destroy is done conditionally in the executor section

    std.debug.print("✓ MLIR module created successfully!\n", .{});

    // Print the MLIR first to see what was generated
    std.debug.print("\nGenerated MLIR:\n", .{});
    std.debug.print("----------------------------------------\n", .{});
    if (use_generic_format) {
        mlir_module.printGeneric();
    } else {
        mlir_module.print();
    }
    std.debug.print("----------------------------------------\n\n", .{});

    // Verify the module after printing
    std.debug.print("\nVerifying MLIR module...\n", .{});
    if (!mlir_module.verify()) {
        std.debug.print("ERROR: Module verification failed!\n", .{});
        std.debug.print("The generated MLIR is malformed. This usually means:\n", .{});
        std.debug.print("  - Invalid operation structure or nesting\n", .{});
        std.debug.print("  - Missing required attributes or regions\n", .{});
        std.debug.print("  - Type mismatches\n", .{});
        return error.ModuleVerificationFailed;
    }
    std.debug.print("✓ Module verification passed!\n", .{});

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

    // Create executor - we'll add GPU runtime later if needed
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

    var gpu_type: gpu_lowering.GPUType = .cpu_fallback;
    if (has_gpu_ops) {
        std.debug.print("Detected GPU operations, applying lowering passes...\n", .{});
        gpu_type = gpu_lowering.detectGPU();
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
        const has_rocdl = std.mem.indexOf(u8, @tagName(gpu_type), "amd") != null;
        const has_nvvm = std.mem.indexOf(u8, @tagName(gpu_type), "nvidia") != null;

        if (has_rocdl or has_nvvm) {
            std.debug.print("\n✓ GPU kernel successfully lowered to hardware-specific IR!\n", .{});
            if (has_rocdl) {
                std.debug.print("  - Kernel converted to ROCDL (AMD ROCm dialect)\n", .{});
                std.debug.print("  - Host code lowered to LLVM with GPU runtime calls\n", .{});
            } else {
                std.debug.print("  - Kernel converted to NVVM (NVIDIA CUDA dialect)\n", .{});
                std.debug.print("  - Host code lowered to LLVM with GPU runtime calls\n", .{});
            }
            std.debug.print("\nAttempting JIT execution with GPU runtime...\n", .{});
            // Continue to JIT execution instead of returning
        } else {
            std.debug.print("Note: GPU operations remain in IR after lowering.\n", .{});
            std.debug.print("      Skipping JIT execution (GPU ops cannot be JIT compiled directly).\n", .{});
            std.debug.print("      The MLIR above shows the successfully lowered GPU code.\n\n", .{});
            return;
        }
    }

    // Add GPU runtime library if still has GPU ops after lowering
    if (still_has_gpu) {
        var shared_libs = std.ArrayList([]const u8){};
        defer shared_libs.deinit(allocator);

        // Path to the ROCm runtime wrappers library
        // Try absolute path first, fall back to relative
        const home = std.posix.getenv("HOME") orelse "/home/jimmyhmiller";
        const rocm_runtime_lib = try std.fmt.allocPrint(allocator, "{s}/mlir-lisp-remote/lib/libmlir_rocm_runtime.so", .{home});
        try shared_libs.append(allocator, rocm_runtime_lib);
        std.debug.print("Loading GPU runtime library: {s}\n", .{rocm_runtime_lib});

        // Also load MLIR runner utils for printMemrefF32 and other utilities
        const runner_utils_lib = "/usr/local/lib/libmlir_runner_utils.so";
        try shared_libs.append(allocator, runner_utils_lib);
        std.debug.print("Loading MLIR runner utils library: {s}\n", .{runner_utils_lib});

        // Update executor config with shared libs
        executor.config.shared_lib_paths = try allocator.dupe([]const u8, shared_libs.items);
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
