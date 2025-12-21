const std = @import("std");
const main_module = @import("main");

const tokenizer = main_module.tokenizer;
const reader = main_module.reader;
const parser = main_module.parser;
const mlir_integration = main_module.mlir_integration;
const ir_generator = main_module.ir_generator;
const jit = main_module.jit;
const ast = main_module.ast;

const ReturnType = enum {
    i32,
    i64,
    f32,
    f64,
    void_type,
    unknown,
};

/// Find the return type of the main function from parsed AST
fn findMainReturnType(nodes: []*ast.Node) ReturnType {
    for (nodes) |node| {
        if (node.node_type == .Operation) {
            const op = node.data.operation;
            // Check if this is func.func
            if (std.mem.eql(u8, op.name, "func") and op.namespace != null and
                std.mem.eql(u8, op.namespace.?, "func"))
            {
                // Check if it's the main function
                if (op.attributes.get("sym_name")) |sym_attr| {
                    if (sym_attr == .string and std.mem.eql(u8, sym_attr.string, "main")) {
                        // Get function_type attribute
                        if (op.attributes.get("function_type")) |ft_attr| {
                            if (ft_attr == .function_type) {
                                const ft = ft_attr.function_type;
                                if (ft.return_types.items.len > 0) {
                                    const ret_type_name = ft.return_types.items[0].name;
                                    if (std.mem.eql(u8, ret_type_name, "i32")) return .i32;
                                    if (std.mem.eql(u8, ret_type_name, "i64")) return .i64;
                                    if (std.mem.eql(u8, ret_type_name, "f32")) return .f32;
                                    if (std.mem.eql(u8, ret_type_name, "f64")) return .f64;
                                    return .unknown;
                                }
                                return .void_type;
                            }
                        }
                    }
                }
            }
        }
    }
    return .unknown;
}

pub fn main() !void {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();

    // Get source from command line
    var args = std.process.args();
    _ = args.skip(); // Skip program name

    const arg = args.next() orelse {
        std.debug.print("Usage: run <file.lisp> or run '<source>'\n", .{});
        std.debug.print("Example: run examples/add.lisp\n", .{});
        std.debug.print("Example: run '(require-dialect arith) (arith.addi (: 1 i32) (: 2 i32))'\n", .{});
        return;
    };

    // Check if arg is a file path (ends with .lisp or .mlsp) or inline source
    var source: []const u8 = undefined;
    var source_owned: ?[]u8 = null;
    defer if (source_owned) |s| allocator.free(s);

    if (std.mem.endsWith(u8, arg, ".lisp") or std.mem.endsWith(u8, arg, ".mlsp")) {
        // Read file
        const file = std.fs.cwd().openFile(arg, .{}) catch |err| {
            std.debug.print("Error opening file '{s}': {}\n", .{ arg, err });
            return;
        };
        defer file.close();

        source_owned = file.readToEndAlloc(allocator, 1024 * 1024) catch |err| {
            std.debug.print("Error reading file '{s}': {}\n", .{ arg, err });
            return;
        };
        source = source_owned.?;
        std.debug.print("=== FILE: {s} ===\n{s}\n\n", .{ arg, source });
    } else {
        // Inline source
        source = arg;
        std.debug.print("=== SOURCE ===\n{s}\n\n", .{source});
    }

    // Tokenize
    var tok = tokenizer.Tokenizer.init(allocator, source);
    var tokens = try tok.tokenize();
    defer tokens.deinit(allocator);

    // Create dialect registry
    var registry = try mlir_integration.DialectRegistry.init(allocator);
    defer registry.deinit();

    // Read
    var rdr = reader.Reader.initWithRegistry(allocator, tokens.items, &registry);
    defer rdr.deinit();

    var values = try rdr.read();
    defer {
        for (values.items) |v| {
            v.deinit();
        }
        values.deinit(allocator);
    }

    // Parse
    var psr = parser.Parser.init(allocator);
    var nodes = try psr.parse(values);
    defer {
        for (nodes.items) |n| {
            n.deinit();
        }
        nodes.deinit(allocator);
    }

    // Generate IR
    var gen = try ir_generator.IRGenerator.init(allocator, registry.ctx);
    defer gen.deinit();

    _ = try gen.generate(nodes.items);

    std.debug.print("=== GENERATED MLIR ===\n", .{});
    gen.printModule();

    // Verify before lowering
    std.debug.print("\n=== VERIFICATION ===\n", .{});
    if (!gen.verify()) {
        std.debug.print("Module FAILED verification!\n", .{});
        return;
    }
    std.debug.print("Module verified successfully!\n", .{});

    // JIT compile
    std.debug.print("\n=== JIT COMPILATION ===\n", .{});

    var jit_engine = jit.JIT.init(allocator, registry.ctx, gen.module) catch |err| {
        std.debug.print("JIT compilation failed: {}\n", .{err});
        return;
    };
    defer jit_engine.deinit();
    std.debug.print("JIT compilation successful!\n", .{});

    // Try to invoke main function
    std.debug.print("\n=== EXECUTION ===\n", .{});

    // Try to lookup the function directly
    const main_fn = jit_engine.lookup("main") catch |err| {
        std.debug.print("Failed to lookup main: {}\n", .{err});
        return;
    };

    // Determine return type from AST
    const ret_type = findMainReturnType(nodes.items);

    // Cast to appropriate function pointer and call based on return type
    switch (ret_type) {
        .i32 => {
            const MainFn = *const fn () callconv(.c) i32;
            const func: MainFn = @ptrCast(@alignCast(main_fn));
            const result = func();
            std.debug.print("Result: {d}\n", .{result});
        },
        .i64 => {
            const MainFn = *const fn () callconv(.c) i64;
            const func: MainFn = @ptrCast(@alignCast(main_fn));
            const result = func();
            std.debug.print("Result: {d}\n", .{result});
        },
        .f32 => {
            const MainFn = *const fn () callconv(.c) f32;
            const func: MainFn = @ptrCast(@alignCast(main_fn));
            const result = func();
            std.debug.print("Result: {d:.6}\n", .{result});
        },
        .f64 => {
            const MainFn = *const fn () callconv(.c) f64;
            const func: MainFn = @ptrCast(@alignCast(main_fn));
            const result = func();
            std.debug.print("Result: {d:.6}\n", .{result});
        },
        .void_type => {
            const MainFn = *const fn () callconv(.c) void;
            const func: MainFn = @ptrCast(@alignCast(main_fn));
            func();
            std.debug.print("(void)\n", .{});
        },
        .unknown => {
            // Fall back to i64 for unknown types
            const MainFn = *const fn () callconv(.c) i64;
            const func: MainFn = @ptrCast(@alignCast(main_fn));
            const result = func();
            std.debug.print("Result: {d} (unknown return type)\n", .{result});
        },
    }
}
