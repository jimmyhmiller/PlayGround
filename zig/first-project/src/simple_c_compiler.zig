const std = @import("std");
const builtin = @import("builtin");
const reader = @import("reader.zig");
const value = @import("value.zig");
const type_checker = @import("type_checker.zig");
const macro_expander = @import("macro_expander.zig");
const namespace_manager = @import("namespace_manager.zig");

const Reader = reader.Reader;
const Value = value.Value;
const TypeChecker = type_checker.BidirectionalTypeChecker;
const Type = type_checker.Type;
const TypedValue = type_checker.TypedValue;
const MacroExpander = macro_expander.MacroExpander;
const NamespaceManager = namespace_manager.NamespaceManager;

const IncludeFlags = struct {
    need_stdint: bool = false,
    need_stdbool: bool = false,
    need_stddef: bool = false,
    need_stdlib: bool = false,
    need_stdio: bool = false,
    need_string: bool = false,
};

pub const SimpleCCompiler = struct {
    allocator: *std.mem.Allocator,
    linked_libraries: std.ArrayList([]const u8),
    include_paths: std.ArrayList([]const u8),
    compiler_flags: std.ArrayList([]const u8),
    namespace_manager: NamespaceManager,
    required_bundles: std.ArrayList([]const u8), // Bundle paths needed for linking

    pub const TargetKind = enum {
        executable,
        bundle,
    };

    const NamespaceDef = struct {
        name: []const u8,
        expr: *Value,
        typed: *TypedValue,
        var_type: Type,
        idx: usize,
    };

    const NamespaceContext = struct {
        name: ?[]const u8,
        def_names: *std.StringHashMap(void),
        in_init_function: bool = false,
        init_only_def_names: ?*std.StringHashMap(void) = null, // Defs initialized in init (not main)
        local_bindings: ?*std.StringHashMap(void) = null, // Track let-bound variables in current scope
        requires: ?*std.StringHashMap([]const u8) = null, // Map alias -> namespace_name for qualified name resolution
    };

    const SpecialForm = enum {
        // Control flow
        if_form,
        while_form,
        c_for,
        let_form,
        do_form,

        // Assignment
        set,

        // Arithmetic operators
        add,
        subtract,
        multiply,
        divide,
        modulo,

        // Comparison operators
        less_than,
        greater_than,
        less_equal,
        greater_equal,
        equal,
        not_equal,

        // Logical operators
        and_op,
        or_op,
        not_op,

        // Bitwise operators
        bitwise_and,
        bitwise_or,
        bitwise_xor,
        bitwise_not,
        bitwise_shl,
        bitwise_shr,

        // Memory operations
        allocate,
        deallocate,
        dereference,
        pointer_write,
        pointer_equal,
        address_of,

        // Pointer field operations
        pointer_field_read,
        pointer_field_write,

        // Array operations
        array_ref,
        array_set,
        array_length,
        array_create,
        array_ptr,
        allocate_array,
        deallocate_array,
        pointer_index_read,
        pointer_index_write,

        // Other operations
        cast,
        field_access,
        printf_fn,
        c_str,

        // Unknown/not a special form
        unknown,
    };

    fn identifySpecialForm(op: []const u8) SpecialForm {
        // Use a hash map or simple if-else chain for string matching
        if (std.mem.eql(u8, op, "if")) return .if_form;
        if (std.mem.eql(u8, op, "while")) return .while_form;
        if (std.mem.eql(u8, op, "c-for")) return .c_for;
        if (std.mem.eql(u8, op, "let")) return .let_form;
        if (std.mem.eql(u8, op, "do")) return .do_form;
        if (std.mem.eql(u8, op, "set!")) return .set;

        // Arithmetic
        if (std.mem.eql(u8, op, "+")) return .add;
        if (std.mem.eql(u8, op, "-")) return .subtract;
        if (std.mem.eql(u8, op, "*")) return .multiply;
        if (std.mem.eql(u8, op, "/")) return .divide;
        if (std.mem.eql(u8, op, "%")) return .modulo;

        // Comparison
        if (std.mem.eql(u8, op, "<")) return .less_than;
        if (std.mem.eql(u8, op, ">")) return .greater_than;
        if (std.mem.eql(u8, op, "<=")) return .less_equal;
        if (std.mem.eql(u8, op, ">=")) return .greater_equal;
        if (std.mem.eql(u8, op, "=")) return .equal;
        if (std.mem.eql(u8, op, "!=")) return .not_equal;

        // Logical
        if (std.mem.eql(u8, op, "and")) return .and_op;
        if (std.mem.eql(u8, op, "or")) return .or_op;
        if (std.mem.eql(u8, op, "not")) return .not_op;

        // Bitwise
        if (std.mem.eql(u8, op, "bitwise-and")) return .bitwise_and;
        if (std.mem.eql(u8, op, "bitwise-or")) return .bitwise_or;
        if (std.mem.eql(u8, op, "bitwise-xor")) return .bitwise_xor;
        if (std.mem.eql(u8, op, "bitwise-not")) return .bitwise_not;
        if (std.mem.eql(u8, op, "bitwise-shl")) return .bitwise_shl;
        if (std.mem.eql(u8, op, "bitwise-shr")) return .bitwise_shr;

        // Memory
        if (std.mem.eql(u8, op, "allocate")) return .allocate;
        if (std.mem.eql(u8, op, "deallocate")) return .deallocate;
        if (std.mem.eql(u8, op, "dereference")) return .dereference;
        if (std.mem.eql(u8, op, "pointer-write!")) return .pointer_write;
        if (std.mem.eql(u8, op, "pointer-equal?")) return .pointer_equal;
        if (std.mem.eql(u8, op, "address-of")) return .address_of;

        // Pointer fields
        if (std.mem.eql(u8, op, "pointer-field-read")) return .pointer_field_read;
        if (std.mem.eql(u8, op, "pointer-field-write!")) return .pointer_field_write;

        // Arrays
        if (std.mem.eql(u8, op, "array-ref")) return .array_ref;
        if (std.mem.eql(u8, op, "array-set!")) return .array_set;
        if (std.mem.eql(u8, op, "array-length")) return .array_length;
        if (std.mem.eql(u8, op, "array")) return .array_create;
        if (std.mem.eql(u8, op, "array-ptr")) return .array_ptr;
        if (std.mem.eql(u8, op, "allocate-array")) return .allocate_array;
        if (std.mem.eql(u8, op, "deallocate-array")) return .deallocate_array;
        if (std.mem.eql(u8, op, "pointer-index-read")) return .pointer_index_read;
        if (std.mem.eql(u8, op, "pointer-index-write!")) return .pointer_index_write;

        // Other
        if (std.mem.eql(u8, op, "cast")) return .cast;
        if (std.mem.eql(u8, op, ".")) return .field_access;
        if (std.mem.eql(u8, op, "printf")) return .printf_fn;
        if (std.mem.eql(u8, op, "c-str")) return .c_str;

        return .unknown;
    }

    /// Static callback function for namespace loading
    /// This is called by the type checker when it encounters a require statement
    fn namespaceLoaderCallback(ctx: *anyopaque, namespace_name: []const u8, parent_checker: *TypeChecker) anyerror!void {
        const self: *SimpleCCompiler = @ptrCast(@alignCast(ctx));
        try self.compileNamespaceAndExtractExports(namespace_name, parent_checker);
    }

    pub const Error = error{
        UnsupportedExpression,
        MissingOperand,
        InvalidDefinition,
        InvalidFunction,
        InvalidIfForm,
        UnsupportedType,
        TypeCheckFailed,
        UnboundVariable,
        TypeMismatch,
        CannotSynthesize,
        CannotApplyNonFunction,
        ArgumentCountMismatch,
        InvalidTypeAnnotation,
        UnexpectedToken,
        UnterminatedString,
        UnterminatedList,
        UnterminatedVector,
        UnterminatedMap,
        InvalidNumber,
        OutOfMemory,
        FileNotFound,
        ParseError,
        MacroExpansionError,
    };

    const int_type_name = "long long";
    const int_printf_format = "%lld";

    fn parseSimpleType(self: *SimpleCCompiler, type_name: []const u8) Error!Type {
        _ = self;
        if (std.mem.eql(u8, type_name, "U8")) return Type.u8;
        if (std.mem.eql(u8, type_name, "U16")) return Type.u16;
        if (std.mem.eql(u8, type_name, "U32")) return Type.u32;
        if (std.mem.eql(u8, type_name, "U64")) return Type.u64;
        if (std.mem.eql(u8, type_name, "I8")) return Type.i8;
        if (std.mem.eql(u8, type_name, "I16")) return Type.i16;
        if (std.mem.eql(u8, type_name, "I32")) return Type.i32;
        if (std.mem.eql(u8, type_name, "I64")) return Type.i64;
        if (std.mem.eql(u8, type_name, "CString")) return Type.c_string;
        if (std.mem.eql(u8, type_name, "Void")) return Type.void;
        if (std.mem.eql(u8, type_name, "Int")) return Type.int;
        if (std.mem.eql(u8, type_name, "Float")) return Type.float;
        if (std.mem.eql(u8, type_name, "String")) return Type.string;
        if (std.mem.eql(u8, type_name, "Bool")) return Type.bool;
        return Error.UnsupportedType;
    }

    pub fn init(allocator: *std.mem.Allocator) SimpleCCompiler {
        var ns_manager = NamespaceManager.init(allocator.*);
        // Add default search paths for namespace resolution
        ns_manager.addSearchPath(".") catch {};
        ns_manager.addSearchPath("math") catch {};

        return SimpleCCompiler{
            .allocator = allocator,
            .linked_libraries = std.ArrayList([]const u8){},
            .include_paths = std.ArrayList([]const u8){},
            .compiler_flags = std.ArrayList([]const u8){},
            .namespace_manager = ns_manager,
            .required_bundles = std.ArrayList([]const u8){},
        };
    }

    pub fn deinit(self: *SimpleCCompiler) void {
        self.linked_libraries.deinit(self.allocator.*);
        self.include_paths.deinit(self.allocator.*);
        self.compiler_flags.deinit(self.allocator.*);
        for (self.required_bundles.items) |bundle| {
            self.allocator.*.free(bundle);
        }
        self.required_bundles.deinit(self.allocator.*);
        self.namespace_manager.deinit();
    }

    /// Resolve namespace name to file path
    /// e.g., "math.utils" -> "math/utils.lisp"
    fn resolveNamespaceFile(self: *SimpleCCompiler, namespace_name: []const u8) Error![]const u8 {
        // Convert dots to slashes
        var path_buf = std.ArrayList(u8){};
        defer path_buf.deinit(self.allocator.*);

        for (namespace_name) |c| {
            if (c == '.') {
                try path_buf.append(self.allocator.*, '/');
            } else {
                try path_buf.append(self.allocator.*, c);
            }
        }
        try path_buf.appendSlice(self.allocator.*, ".lisp");

        const relative_path = try path_buf.toOwnedSlice(self.allocator.*);

        // Check if file exists
        std.fs.cwd().access(relative_path, .{}) catch {
            std.debug.print("ERROR: Cannot find namespace file: {s}\n", .{relative_path});
            return Error.FileNotFound;
        };

        return relative_path;
    }

    /// Compile a namespace file and extract its type environment
    /// Used for processing `require` statements
    pub fn compileNamespaceAndExtractExports(
        self: *SimpleCCompiler,
        namespace_name: []const u8,
        parent_checker: *TypeChecker,
    ) Error!void {
        // Check if already compiled and cached
        if (self.namespace_manager.isCompiled(namespace_name)) {
            std.debug.print("Namespace {s} already compiled, using cache\n", .{namespace_name});
            const ns = self.namespace_manager.getCompiledNamespace(namespace_name) orelse {
                return Error.TypeCheckFailed;
            };
            // Re-register exports with parent checker
            try parent_checker.registerNamespaceExports(namespace_name, ns.definitions, ns.type_defs);

            // Add cached bundle to required bundles list
            const bundle_path = try self.allocator.*.dupe(u8, ns.bundle_path);
            try self.required_bundles.append(self.allocator.*, bundle_path);
            return;
        }

        // Check for circular dependencies
        if (self.namespace_manager.isInCompilationStack(namespace_name)) {
            const stack_trace = try self.namespace_manager.getCompilationStackTrace();
            defer self.allocator.*.free(stack_trace);
            std.debug.print("ERROR: Circular dependency detected\n{s}\n  -> {s}\n", .{ stack_trace, namespace_name });
            return Error.TypeCheckFailed;
        }

        try self.namespace_manager.pushCompilationStack(namespace_name);
        defer self.namespace_manager.popCompilationStack();

        std.debug.print("Compiling namespace: {s}\n", .{namespace_name});

        // 1. Resolve file path
        const file_path = try self.resolveNamespaceFile(namespace_name);
        defer self.allocator.*.free(file_path);

        // 2. Read file
        const source = std.fs.cwd().readFileAlloc(
            self.allocator.*,
            file_path,
            10 * 1024 * 1024, // 10MB max
        ) catch |err| {
            std.debug.print("ERROR: Cannot read namespace file {s}: {}\n", .{ file_path, err });
            return Error.FileNotFound;
        };
        defer self.allocator.*.free(source);

        // 3. Parse
        var reader_instance = Reader.init(self.allocator);
        const read_result = reader_instance.readAllString(source) catch {
            return Error.ParseError;
        };
        var expressions = read_result.values;
        var line_numbers = read_result.line_numbers;
        defer expressions.deinit(self.allocator.*);
        defer line_numbers.deinit(self.allocator.*);

        // 4. Expand macros
        var expander = MacroExpander.init(self.allocator.*);
        defer expander.deinit();

        var expanded_expressions = std.ArrayList(*Value){};
        defer expanded_expressions.deinit(self.allocator.*);

        for (expressions.items) |expr| {
            const expanded = expander.expand(expr) catch {
                return Error.MacroExpansionError;
            };
            if (expanded.isMacro()) {
                // Skip macro definitions
                continue;
            }
            try expanded_expressions.append(self.allocator.*, expanded);
        }

        // 5. Type check to extract exports
        var checker = TypeChecker.init(self.allocator.*);
        defer checker.deinit();

        // Set up namespace loader for nested requires
        checker.setNamespaceLoader(self, namespaceLoaderCallback);

        // Use two-pass type checking to properly handle requires and forward references
        _ = checker.typeCheckAllTwoPass(expanded_expressions.items) catch {
            std.debug.print("ERROR: Type checking failed for required namespace {s}\n", .{namespace_name});
            return Error.TypeCheckFailed;
        };

        // 6. Extract and register exports with parent checker
        // Clone the type environments (shallow copy is fine since Types are immutable)
        var exports = type_checker.TypeEnv.init(self.allocator.*);
        var type_defs = type_checker.TypeEnv.init(self.allocator.*);

        // Copy all definitions from checker.env to exports
        var env_iter = checker.env.iterator();
        while (env_iter.next()) |entry| {
            try exports.put(entry.key_ptr.*, entry.value_ptr.*);
        }

        // Copy all type definitions
        var typedef_iter = checker.type_defs.iterator();
        while (typedef_iter.next()) |entry| {
            try type_defs.put(entry.key_ptr.*, entry.value_ptr.*);
        }

        // 7. Register exports with parent checker
        // Note: The alias mapping is registered separately by the require handler
        try parent_checker.registerNamespaceExports(namespace_name, exports, type_defs);

        // 8. Compile namespace to .bundle file
        const bundle_path = try self.compileToBundleFile(namespace_name);
        std.debug.print("Compiled namespace {s} to bundle: {s}\n", .{ namespace_name, bundle_path });

        // Add bundle to required bundles list for linking
        const bundle_path_copy = try self.allocator.*.dupe(u8, bundle_path);
        try self.required_bundles.append(self.allocator.*, bundle_path_copy);

        // 9. Cache the compiled namespace
        var compiled_ns = try namespace_manager.CompiledNamespace.init(self.allocator.*, namespace_name);
        compiled_ns.file_path = try self.allocator.*.dupe(u8, file_path);
        compiled_ns.bundle_path = bundle_path; // Takes ownership
        compiled_ns.definitions = exports;
        compiled_ns.type_defs = type_defs;

        // Copy requires from checker (alias -> namespace_name mappings)
        compiled_ns.requires = std.ArrayList(value.RequireDecl){};
        var requires_iter = checker.requires.iterator();
        while (requires_iter.next()) |entry| {
            const req_decl = value.RequireDecl{
                .namespace = try self.allocator.*.dupe(u8, entry.value_ptr.*),
                .alias = try self.allocator.*.dupe(u8, entry.key_ptr.*),
            };
            try compiled_ns.requires.append(self.allocator.*, req_decl);
        }

        try self.namespace_manager.addCompiledNamespace(compiled_ns);
        std.debug.print("Cached namespace {s}\n", .{namespace_name});
    }

    /// Compile a namespace to a .bundle file
    /// Returns the absolute path to the compiled bundle
    fn compileToBundleFile(
        self: *SimpleCCompiler,
        namespace_name: []const u8,
    ) Error![]const u8 {
        // 1. Resolve and read namespace file
        const file_path = try self.resolveNamespaceFile(namespace_name);
        defer self.allocator.*.free(file_path);

        const source = std.fs.cwd().readFileAlloc(
            self.allocator.*,
            file_path,
            10 * 1024 * 1024, // 10MB max
        ) catch |err| {
            std.debug.print("ERROR: Cannot read namespace file {s}: {}\n", .{ file_path, err });
            return Error.FileNotFound;
        };
        defer self.allocator.*.free(source);

        // 2. Generate C code using compileString
        const c_code = try self.compileString(source, .bundle);
        defer self.allocator.*.free(c_code);

        // Save the required bundles that were collected during compilation
        // We need to link against these when creating this namespace's bundle
        var nested_bundles = std.ArrayList([]const u8){};
        defer nested_bundles.deinit(self.allocator.*);
        for (self.required_bundles.items) |bundle| {
            try nested_bundles.append(self.allocator.*, bundle);
        }

        // 3. Convert namespace name to file path
        var path_buf = std.ArrayList(u8){};
        defer path_buf.deinit(self.allocator.*);

        for (namespace_name) |c| {
            if (c == '.') {
                try path_buf.append(self.allocator.*, '/');
            } else {
                try path_buf.append(self.allocator.*, c);
            }
        }

        const relative_base = try path_buf.toOwnedSlice(self.allocator.*);
        defer self.allocator.*.free(relative_base);

        // 4. Write C file
        const c_filename = try std.fmt.allocPrint(self.allocator.*, "{s}.c", .{relative_base});
        defer self.allocator.*.free(c_filename);

        std.fs.cwd().writeFile(.{ .sub_path = c_filename, .data = c_code }) catch |err| {
            std.debug.print("ERROR: Failed to write C file {s}: {}\n", .{ c_filename, err });
            return Error.FileNotFound; // Reuse existing error
        };

        // 5. Compile to .o
        const obj_filename = try std.fmt.allocPrint(self.allocator.*, "{s}.o", .{relative_base});
        defer self.allocator.*.free(obj_filename);

        var compile_args = std.ArrayList([]const u8){};
        defer compile_args.deinit(self.allocator.*);
        try compile_args.appendSlice(self.allocator.*, &.{ "zig", "cc", "-c", c_filename, "-o", obj_filename });

        var compile_child = std.process.Child.init(compile_args.items, self.allocator.*);
        compile_child.stdin_behavior = .Inherit;
        compile_child.stdout_behavior = .Inherit;
        compile_child.stderr_behavior = .Inherit;
        compile_child.spawn() catch |err| {
            std.debug.print("ERROR: Failed to spawn compiler for namespace {s}: {}\n", .{ namespace_name, err });
            return Error.FileNotFound;
        };
        const compile_term = compile_child.wait() catch |err| {
            std.debug.print("ERROR: Failed to wait for compiler for namespace {s}: {}\n", .{ namespace_name, err });
            return Error.FileNotFound;
        };
        switch (compile_term) {
            .Exited => |code| {
                if (code != 0) {
                    std.debug.print("Compilation failed for namespace {s}\n", .{namespace_name});
                    return Error.TypeCheckFailed;
                }
            },
            else => return Error.TypeCheckFailed,
        }

        // 6. Link to .dylib (for namespaces that can be linked against)
        const bundle_filename = try std.fmt.allocPrint(self.allocator.*, "{s}.dylib", .{relative_base});
        defer self.allocator.*.free(bundle_filename);

        var link_args = std.ArrayList([]const u8){};
        defer link_args.deinit(self.allocator.*);
        try link_args.appendSlice(self.allocator.*, &.{ "zig", "cc", "-dynamiclib", obj_filename, "-o", bundle_filename });

        // Add required bundles that this namespace depends on
        for (nested_bundles.items) |bundle| {
            try link_args.append(self.allocator.*, bundle);
        }

        // Add linked libraries
        for (self.linked_libraries.items) |lib| {
            const lib_arg = try std.fmt.allocPrint(self.allocator.*, "-l{s}", .{lib});
            defer self.allocator.*.free(lib_arg);
            try link_args.append(self.allocator.*, try self.allocator.*.dupe(u8, lib_arg));
        }

        var link_child = std.process.Child.init(link_args.items, self.allocator.*);
        link_child.stdin_behavior = .Inherit;
        link_child.stdout_behavior = .Inherit;
        link_child.stderr_behavior = .Inherit;
        link_child.spawn() catch |err| {
            std.debug.print("ERROR: Failed to spawn linker for namespace {s}: {}\n", .{ namespace_name, err });
            return Error.FileNotFound;
        };
        const link_term = link_child.wait() catch |err| {
            std.debug.print("ERROR: Failed to wait for linker for namespace {s}: {}\n", .{ namespace_name, err });
            return Error.FileNotFound;
        };
        switch (link_term) {
            .Exited => |code| {
                if (code != 0) {
                    std.debug.print("Linking failed for namespace {s}\n", .{namespace_name});
                    return Error.TypeCheckFailed;
                }
            },
            else => return Error.TypeCheckFailed,
        }

        // 7. Return absolute path to bundle
        const bundle_path = std.fs.cwd().realpathAlloc(self.allocator.*, bundle_filename) catch |err| {
            std.debug.print("ERROR: Failed to get realpath for bundle {s}: {}\n", .{ bundle_filename, err });
            return Error.FileNotFound;
        };
        return bundle_path;
    }

    /// Collect all transitive namespace dependencies in topological order
    /// (dependencies before dependents) for proper initialization
    fn collectTransitiveDependencies(
        self: *SimpleCCompiler,
        visited: *std.StringHashMap(void),
        result: *std.ArrayList([]const u8),
    ) Error!void {
        // Iterate over all compiled namespaces
        var ns_iter = self.namespace_manager.compiled_namespaces.iterator();
        while (ns_iter.next()) |entry| {
            const ns_name = entry.key_ptr.*;
            try self.collectTransitiveDependenciesRecursive(ns_name, visited, result);
        }
    }

    /// Recursive helper for collectTransitiveDependencies
    fn collectTransitiveDependenciesRecursive(
        self: *SimpleCCompiler,
        ns_name: []const u8,
        visited: *std.StringHashMap(void),
        result: *std.ArrayList([]const u8),
    ) Error!void {
        // Skip if already visited
        if (visited.contains(ns_name)) {
            return;
        }

        // Mark as visited
        try visited.put(ns_name, {});

        // Get the compiled namespace
        const compiled_ns = self.namespace_manager.getCompiledNamespace(ns_name) orelse {
            return; // Should not happen, but handle gracefully
        };

        // Recursively visit all dependencies first (depth-first)
        for (compiled_ns.requires.items) |req| {
            const dep_ns_name = req.namespace;
            try self.collectTransitiveDependenciesRecursive(dep_ns_name, visited, result);
        }

        // Add this namespace after all its dependencies
        try result.append(self.allocator.*, ns_name);
    }

    pub fn compileString(self: *SimpleCCompiler, source: []const u8, target: TargetKind) Error![]u8 {
        // Clear required bundles from previous compilation
        for (self.required_bundles.items) |bundle| {
            self.allocator.*.free(bundle);
        }
        self.required_bundles.clearRetainingCapacity();

        var reader_instance = Reader.init(self.allocator);
        const read_result = reader_instance.readAllString(source) catch |err| {
            return switch (err) {
                error.UnexpectedToken => Error.UnexpectedToken,
                error.UnterminatedString => Error.UnterminatedString,
                error.UnterminatedList => Error.UnterminatedList,
                error.UnterminatedVector => Error.UnterminatedVector,
                error.UnterminatedMap => Error.UnterminatedMap,
                error.InvalidNumber => Error.InvalidNumber,
                error.OutOfMemory => Error.OutOfMemory,
            };
        };
        var expressions = read_result.values;
        var line_numbers = read_result.line_numbers;
        defer expressions.deinit(self.allocator.*);
        defer line_numbers.deinit(self.allocator.*);

        // MACRO EXPANSION PHASE: Expand all macros before type checking
        var expander = MacroExpander.init(self.allocator.*);
        defer expander.deinit();

        var expanded_expressions = std.ArrayList(*Value){};
        defer expanded_expressions.deinit(self.allocator.*);

        for (expressions.items) |expr| {
            const expanded = expander.expand(expr) catch |err| {
                std.debug.print("Macro expansion error: {s}\n", .{@errorName(err)});
                return Error.UnsupportedExpression;
            };
            // Skip macro definitions - they shouldn't be type-checked or code-generated
            // They're only needed during the expansion phase
            if (!expanded.isMacro()) {
                try expanded_expressions.append(self.allocator.*, expanded);
            }
        }

        var checker = TypeChecker.init(self.allocator.*);
        defer checker.deinit();

        // Set up namespace loader so the type checker can load required namespaces on-demand
        checker.setNamespaceLoader(@ptrCast(self), namespaceLoaderCallback);

        var report = try checker.typeCheckAllTwoPass(expanded_expressions.items);
        defer report.typed.deinit(self.allocator.*);
        defer report.errors.deinit(self.allocator.*);

        if (report.errors.items.len != 0 or report.typed.items.len != expanded_expressions.items.len) {
            if (report.errors.items.len == 0) {
                std.debug.print("Type check failed: expected {d} typed expressions, got {d}\n", .{ expanded_expressions.items.len, report.typed.items.len });

                // Show all expressions with their status
                std.debug.print("\nExpression details:\n", .{});
                for (expanded_expressions.items, 0..) |expr, idx| {
                    const line = if (idx < line_numbers.items.len) line_numbers.items[idx] else 0;
                    const maybe_str = self.formatValue(expr) catch null;
                    if (maybe_str) |expr_str| {
                        defer self.allocator.*.free(expr_str);
                        const status = if (idx < report.typed.items.len) "✓" else "✗";
                        std.debug.print("  {s} Line {d} (expr #{d}): {s}\n", .{ status, line, idx, expr_str });
                    }
                }
                std.debug.print("\nNote: {d} expressions type-checked successfully, {d} failed\n", .{ report.typed.items.len, expanded_expressions.items.len - report.typed.items.len });
            }
            for (report.errors.items) |detail| {
                const line = if (detail.index < line_numbers.items.len) line_numbers.items[detail.index] else 0;
                const maybe_str = self.formatValue(detail.expr) catch null;

                std.debug.print("Type error at line {d} (expr #{d}): {s}", .{ line, detail.index, @errorName(detail.err) });

                if (detail.info) |info| {
                    switch (info) {
                        .unbound => |unbound| {
                            std.debug.print(" - unbound variable '{s}'", .{unbound.name});
                        },
                        .type_mismatch => |mismatch| {
                            std.debug.print("\n  Expected: {any}\n  Actual:   {any}", .{ mismatch.expected, mismatch.actual });
                        },
                    }
                }

                if (maybe_str) |expr_str| {
                    defer self.allocator.*.free(expr_str);
                    std.debug.print("\n  Expression: {s}\n", .{expr_str});
                } else {
                    std.debug.print("\n", .{});
                }
            }
            // Return the first specific error instead of generic TypeCheckFailed
            return report.errors.items[0].err;
        }

        // Collect namespace info and definitions
        var namespace_name: ?[]const u8 = null;
        var namespace_defs = std.ArrayList(NamespaceDef){};
        defer namespace_defs.deinit(self.allocator.*);
        var non_def_exprs = std.ArrayList(struct { expr: *Value, typed: *TypedValue, idx: usize }){};
        defer non_def_exprs.deinit(self.allocator.*);

        // Note: expanded_expressions and report.typed have the same length
        // because we filtered out macros before type checking
        for (expanded_expressions.items, report.typed.items, 0..) |expr, typed_val, expr_idx| {

            // Check for namespace declaration
            if (expr.* == .namespace) {
                namespace_name = expr.namespace.name;
                continue;
            }

            // Check if this is a def or extern declaration
            if (expr.* == .list) {
                var iter = expr.list.iterator();
                if (iter.next()) |head_val| {
                    if (head_val.isSymbol()) {
                        // Skip extern declarations and directives - they're handled in forward decl pass
                        if (std.mem.eql(u8, head_val.symbol, "extern-fn") or
                            std.mem.eql(u8, head_val.symbol, "extern-type") or
                            std.mem.eql(u8, head_val.symbol, "extern-union") or
                            std.mem.eql(u8, head_val.symbol, "extern-struct") or
                            std.mem.eql(u8, head_val.symbol, "extern-var") or
                            std.mem.eql(u8, head_val.symbol, "include-header") or
                            std.mem.eql(u8, head_val.symbol, "link-library") or
                            std.mem.eql(u8, head_val.symbol, "compiler-flag"))
                        {
                            continue;
                        }
                    }
                    if (head_val.isSymbol() and std.mem.eql(u8, head_val.symbol, "def")) {
                        if (iter.next()) |name_val| {
                            if (name_val.isSymbol()) {
                                const def_name = name_val.symbol;
                                // Skip 'main' function - it's a special entry point that shouldn't be in namespace
                                if (std.mem.eql(u8, def_name, "main")) {
                                    continue;
                                }
                                // Skip type definitions (struct/enum) - they're handled differently
                                if (checker.type_defs.get(def_name) == null) {
                                    if (checker.env.get(def_name)) |var_type| {
                                        // Include both regular vars and functions
                                        try namespace_defs.append(self.allocator.*, .{
                                            .name = def_name,
                                            .expr = expr,
                                            .typed = typed_val,
                                            .var_type = var_type,
                                            .idx = expr_idx,
                                        });
                                        continue;
                                    }
                                } else {
                                    // This is a type definition - skip it (don't add to non_def_exprs)
                                    continue;
                                }
                            }
                        }
                    }
                }
            }

            // Not a namespace or def - it's a regular expression
            try non_def_exprs.append(self.allocator.*, .{ .expr = expr, .typed = typed_val, .idx = expr_idx });
        }

        var forward_decls = std.ArrayList(u8){};
        defer forward_decls.deinit(self.allocator.*);
        var prelude = std.ArrayList(u8){};
        defer prelude.deinit(self.allocator.*);
        var body = std.ArrayList(u8){};
        defer body.deinit(self.allocator.*);

        const forward_writer = forward_decls.writer(self.allocator.*);
        const prelude_writer = prelude.writer(self.allocator.*);
        const body_writer = body.writer(self.allocator.*);

        var includes = IncludeFlags{};

        // First pass: emit forward declarations for functions and structs
        for (expanded_expressions.items, report.typed.items) |expr, typed_val| {
            try self.emitForwardDecl(forward_writer, expr, typed_val, &checker, &includes);
        }

        // Emit full struct definitions for all namespaces (including transitive dependencies)
        // Collect all transitive dependencies in topological order
        var visited_decl = std.StringHashMap(void).init(self.allocator.*);
        defer visited_decl.deinit();
        var all_namespaces = std.ArrayList([]const u8){};
        defer all_namespaces.deinit(self.allocator.*);
        try self.collectTransitiveDependencies(&visited_decl, &all_namespaces);

        // Emit declarations for all namespaces
        for (all_namespaces.items) |required_ns_name| {
            const sanitized_ns = try self.sanitizeIdentifier(required_ns_name);
            defer self.allocator.*.free(sanitized_ns);

            try forward_writer.print("// Required namespace: {s}\n", .{required_ns_name});

            // Get the compiled namespace to access its type info
            std.debug.print("DEBUG: Looking up namespace {s} in cache\n", .{required_ns_name});
            if (self.namespace_manager.getCompiledNamespace(required_ns_name)) |compiled_ns| {
                std.debug.print("DEBUG: Found namespace {s} in cache with {d} definitions\n", .{ required_ns_name, compiled_ns.definitions.count() });

                // Emit type definitions first (struct/enum types used by namespace)
                var typedef_iter = compiled_ns.type_defs.iterator();
                while (typedef_iter.next()) |typedef_entry| {
                    const type_name = typedef_entry.key_ptr.*;
                    const type_def = typedef_entry.value_ptr.*;

                    if (type_def == .struct_type) {
                        const st = type_def.struct_type;
                        const sanitized_type_name = try self.sanitizeIdentifier(type_name);
                        defer self.allocator.*.free(sanitized_type_name);

                        try forward_writer.print("typedef struct {{\n", .{});
                        for (st.fields) |field| {
                            const c_field_type = try self.cTypeFor(field.field_type, &includes);
                            try forward_writer.print("    {s} {s};\n", .{ c_field_type, field.name });
                        }
                        try forward_writer.print("}} {s};\n\n", .{sanitized_type_name});
                    } else if (type_def == .enum_type) {
                        const et = type_def.enum_type;
                        const sanitized_type_name = try self.sanitizeIdentifier(type_name);
                        defer self.allocator.*.free(sanitized_type_name);

                        try forward_writer.print("typedef enum {{\n", .{});
                        for (et.variants) |variant| {
                            const sanitized_variant = try self.sanitizeIdentifier(variant.qualified_name.?);
                            defer self.allocator.*.free(sanitized_variant);
                            try forward_writer.print("    {s},\n", .{sanitized_variant});
                        }
                        try forward_writer.print("}} {s};\n\n", .{sanitized_type_name});
                    }
                }

                // Emit full struct definition
                try forward_writer.print("typedef struct {{\n", .{});

                // Emit fields for each exported definition
                var def_iter = compiled_ns.definitions.iterator();
                while (def_iter.next()) |def_entry| {
                    const def_name = def_entry.key_ptr.*;
                    const def_type = def_entry.value_ptr.*;

                    // Skip type definitions (Point, Color, etc.) - they're compile-time only
                    if (def_type == .type_type) {
                        std.debug.print("DEBUG: Skipping type definition '{s}' (compile-time only)\n", .{def_name});
                        continue;
                    }

                    const sanitized_field = try self.sanitizeIdentifier(def_name);
                    defer self.allocator.*.free(sanitized_field);

                    // Special handling for function types
                    if (def_type == .function) {
                        const fn_type = def_type.function;
                        const ret_c_type = try self.cTypeFor(fn_type.return_type, &includes);

                        // Build parameter list
                        var params_list = std.ArrayList([]const u8){};
                        defer params_list.deinit(self.allocator.*);

                        for (fn_type.param_types) |param_type| {
                            const param_c_type = try self.cTypeFor(param_type, &includes);
                            try params_list.append(self.allocator.*, param_c_type);
                        }

                        const params = try std.mem.join(self.allocator.*, ", ", params_list.items);
                        defer self.allocator.*.free(params);

                        // Emit function pointer with name inside: ReturnType (*name)(params);
                        try forward_writer.print("    {s} (*{s})({s});\n", .{ ret_c_type, sanitized_field, params });
                    } else {
                        // Regular field
                        const c_type = self.cTypeFor(def_type, &includes) catch |err| {
                            std.debug.print("ERROR: Cannot convert type for field '{s}': {}\n", .{ def_name, err });
                            std.debug.print("  Type: {any}\n", .{def_type});
                            return err;
                        };
                        try forward_writer.print("    {s} {s};\n", .{ c_type, sanitized_field });
                    }
                }

                try forward_writer.print("}} Namespace_{s};\n\n", .{sanitized_ns});
            } else {
                // Fallback to forward declaration if namespace not found
                std.debug.print("DEBUG: Namespace {s} NOT found in cache, using forward declaration\n", .{required_ns_name});
                try forward_writer.print("typedef struct Namespace_{s} Namespace_{s};\n", .{ sanitized_ns, sanitized_ns });
            }

            try forward_writer.print("extern Namespace_{s} g_{s};\n", .{ sanitized_ns, sanitized_ns });
            try forward_writer.print("void init_namespace_{s}(Namespace_{s}* ns);\n\n", .{ sanitized_ns, sanitized_ns });
        }

        // Create empty context for non-namespace code (used in emitDefinition/emitFunctionDefinition)
        var empty_def_names = std.StringHashMap(void).init(self.allocator.*);
        defer empty_def_names.deinit();
        _ = &empty_def_names; // Mark as used to suppress warning
        var empty_ctx = NamespaceContext{
            .name = null,
            .def_names = &empty_def_names,
        };
        _ = &empty_ctx; // Mark as used (referenced in nested functions)

        // Copy requires map from type checker for qualified name resolution
        var requires_map = std.StringHashMap([]const u8).init(self.allocator.*);
        defer requires_map.deinit();
        var requires_iter = checker.requires.iterator();
        while (requires_iter.next()) |entry| {
            try requires_map.put(entry.key_ptr.*, entry.value_ptr.*);
        }

        // If we have namespace defs, generate the namespace struct
        var namespace_struct = std.ArrayList(u8){};
        defer namespace_struct.deinit(self.allocator.*);
        var namespace_init = std.ArrayList(u8){};
        defer namespace_init.deinit(self.allocator.*);

        // Track which defs are actually successfully initialized in init function
        var actually_initialized_defs = std.StringHashMap(void).init(self.allocator.*);
        defer actually_initialized_defs.deinit();

        if (namespace_defs.items.len > 0) {
            const ns_writer = namespace_struct.writer(self.allocator.*);
            const init_writer = namespace_init.writer(self.allocator.*);

            const ns_name = namespace_name orelse "user";
            const sanitized_ns = try self.sanitizeIdentifier(ns_name);
            defer self.allocator.*.free(sanitized_ns);

            // Generate namespace struct
            try ns_writer.print("typedef struct {{\n", .{});
            for (namespace_defs.items) |def| {
                const sanitized_field = try self.sanitizeIdentifier(def.name);
                defer self.allocator.*.free(sanitized_field);

                if (def.var_type == .function) {
                    // Function pointer: return_type (*name)(param_types...)
                    const fn_type = def.var_type.function;
                    const return_type_str = self.cTypeFor(fn_type.return_type, &includes) catch |err| {
                        if (err == Error.UnsupportedType) {
                            try ns_writer.print("    // unsupported function type for: {s}\n", .{def.name});
                            continue;
                        }
                        return err;
                    };
                    try ns_writer.print("    {s} (*{s})(", .{ return_type_str, sanitized_field });
                    for (fn_type.param_types, 0..) |param_type, i| {
                        if (i > 0) try ns_writer.print(", ", .{});
                        const param_type_str = self.cTypeFor(param_type, &includes) catch |err| {
                            if (err == Error.UnsupportedType) {
                                try ns_writer.print("/* unsupported */", .{});
                                continue;
                            }
                            return err;
                        };
                        try ns_writer.print("{s}", .{param_type_str});
                    }
                    try ns_writer.print(");\n", .{});
                } else {
                    // Regular variable
                    try ns_writer.print("    ", .{});
                    self.emitArrayDecl(ns_writer, sanitized_field, def.var_type, &includes) catch |err| {
                        if (err == Error.UnsupportedType) {
                            try ns_writer.print("// unsupported type for: {s}\n", .{def.name});
                            continue;
                        }
                        return err;
                    };
                    try ns_writer.print(";\n", .{});
                }
            }
            try ns_writer.print("}} Namespace_{s};\n\n", .{sanitized_ns});

            // Generate global namespace instance
            try ns_writer.print("Namespace_{s} g_{s};\n\n", .{ sanitized_ns, sanitized_ns });

            // Generate forward declarations for static functions
            for (namespace_defs.items) |def| {
                if (def.var_type == .function) {
                    const sanitized_field = try self.sanitizeIdentifier(def.name);
                    defer self.allocator.*.free(sanitized_field);

                    const fn_type = def.var_type.function;
                    const return_type_str = self.cTypeFor(fn_type.return_type, &includes) catch continue;

                    try ns_writer.print("static {s} {s}(", .{ return_type_str, sanitized_field });
                    for (fn_type.param_types, 0..) |param_type, i| {
                        if (i > 0) try ns_writer.print(", ", .{});
                        const param_type_str = self.cTypeFor(param_type, &includes) catch continue;
                        try ns_writer.print("{s}", .{param_type_str});
                    }
                    try ns_writer.print(");\n", .{});
                }
            }
            try ns_writer.print("\n", .{});

            // Build def names set for init function context
            var init_def_names = std.StringHashMap(void).init(self.allocator.*);
            defer init_def_names.deinit();
            for (namespace_defs.items) |def| {
                try init_def_names.put(def.name, {});
            }

            // Create init function context (can reference namespace vars as ns->field)
            var init_ctx = NamespaceContext{
                .name = ns_name,
                .def_names = &init_def_names,
                .in_init_function = true,
                .requires = &requires_map,
            };

            // Generate init function signature
            try init_writer.print("void init_namespace_{s}(Namespace_{s}* ns) {{\n", .{ sanitized_ns, sanitized_ns });

            // Find the index of the first non-def expression
            // Defs before this index can be initialized in init()
            // Defs after this index must be initialized in main() to preserve source order
            var first_non_def_idx: ?usize = null;
            if (non_def_exprs.items.len > 0) {
                first_non_def_idx = non_def_exprs.items[0].idx;
                for (non_def_exprs.items) |non_def| {
                    if (first_non_def_idx == null or non_def.idx < first_non_def_idx.?) {
                        first_non_def_idx = non_def.idx;
                    }
                }
            }

            // Emit initialization for defs that appear before any non-def expressions
            for (namespace_defs.items) |def| {
                // Skip defs that appear after non-def expressions - they'll be initialized in main
                if (first_non_def_idx != null and def.idx >= first_non_def_idx.?) {
                    continue;
                }

                const sanitized_field = try self.sanitizeIdentifier(def.name);
                defer self.allocator.*.free(sanitized_field);

                if (def.var_type == .function) {
                    // For functions, just assign the address of the static function
                    try init_writer.print("    ns->{s} = &{s};\n", .{ sanitized_field, sanitized_field });
                    try actually_initialized_defs.put(def.name, {});
                } else if (def.var_type == .array) {
                    // Arrays can't be assigned in C, so we need to copy element by element
                    // For now, generate a loop to initialize the array
                    const array_type = def.var_type.array;

                    // Check if there's an init value
                    const typed_list = def.typed.list;
                    const has_init = typed_list.elements.len == 3; // array op has 3 elements when initialized

                    if (has_init) {
                        try init_writer.print("    for (size_t __i_{s} = 0; __i_{s} < {d}; __i_{s}++) {{\n", .{ sanitized_field, sanitized_field, array_type.size, sanitized_field });
                        try init_writer.print("        ns->{s}[__i_{s}] = ", .{ sanitized_field, sanitized_field });
                        self.writeExpressionTyped(init_writer, typed_list.elements[2], &init_ctx, &includes) catch |err| {
                            switch (err) {
                                Error.UnsupportedExpression => {
                                    try init_writer.print("0;\n    }}\n", .{});
                                    continue;
                                },
                                else => return err,
                            }
                        };
                        try init_writer.print(";\n    }}\n", .{});
                        try actually_initialized_defs.put(def.name, {});
                    }
                    // If no init value, array is left uninitialized (or zero-initialized by C)
                } else {
                    // Get the value expression from the def
                    var iter = def.expr.list.iterator();
                    _ = iter.next(); // skip 'def'
                    _ = iter.next(); // skip name

                    // Skip type annotation if present
                    var maybe_value = iter.next();
                    if (maybe_value) |val| {
                        if (val.isList()) {
                            var val_iter = val.list.iterator();
                            if (val_iter.next()) |first| {
                                // Check if this is a type annotation: (: ...)
                                // The keyword : is stored with empty name
                                if (first.isKeyword() and first.keyword.len == 0) {
                                    maybe_value = iter.next();
                                }
                            }
                        }
                    }

                    if (maybe_value) |value_expr| {
                        _ = value_expr;
                        // Evaluate the expression using typed AST (handles let, do, and all other expressions)
                        try init_writer.print("    ns->{s} = ", .{sanitized_field});

                        self.writeExpressionTyped(init_writer, def.typed, &init_ctx, &includes) catch |err| {
                            switch (err) {
                                Error.UnsupportedExpression, Error.MissingOperand, Error.InvalidIfForm => {
                                    try init_writer.print("0; // unsupported expression\n", .{});
                                    continue;
                                },
                                else => return err,
                            }
                        };
                        try init_writer.print(";\n", .{});
                        try actually_initialized_defs.put(def.name, {});
                    }
                }
            }

            try init_writer.print("}}\n\n", .{});

            // Create function context (can reference namespace vars as g_namespace.field)
            var fn_ctx = NamespaceContext{
                .name = ns_name,
                .def_names = &init_def_names,
                .in_init_function = false,
                .requires = &requires_map,
            };

            // Emit static function definitions
            for (namespace_defs.items) |def| {
                if (def.var_type == .function) {
                    // Extract the function expression from the def (untyped AST)
                    var def_iter = def.expr.list.iterator();
                    _ = def_iter.next(); // skip 'def'
                    _ = def_iter.next(); // skip name
                    var maybe_fn_expr = def_iter.next();
                    if (maybe_fn_expr) |val| {
                        if (val.isList()) {
                            var val_iter = val.list.iterator();
                            if (val_iter.next()) |first| {
                                if (first.isKeyword() and first.keyword.len == 0) {
                                    maybe_fn_expr = def_iter.next();
                                }
                            }
                        }
                    }

                    // Extract the typed function expression from def.typed
                    // def.typed is the typed body from the type checker (not the full def form)
                    // It should be a list: (fn [params...] body...)
                    const fn_typed = def.typed;

                    if (maybe_fn_expr) |fn_expr| {
                        // Use emitFunctionDefinition which handles both untyped and typed AST
                        self.emitFunctionDefinition(init_writer, def.name, fn_expr, fn_typed, def.var_type, &includes, &fn_ctx) catch |err| {
                            std.debug.print("Failed to emit namespace function {s}: {}\n", .{ def.name, err });
                            continue;
                        };
                    }
                }
            }
        }

        // Build a set of namespace def names for quick lookup
        var namespace_def_names = std.StringHashMap(void).init(self.allocator.*);
        defer namespace_def_names.deinit();
        for (namespace_defs.items) |def| {
            try namespace_def_names.put(def.name, {});
        }

        // Build sets for tracking which defs were initialized where
        var init_processed_indices = std.AutoHashMap(usize, void).init(self.allocator.*);
        defer init_processed_indices.deinit();

        var init_only_def_names = std.StringHashMap(void).init(self.allocator.*);
        defer init_only_def_names.deinit();

        // Use actual tracking: only mark defs that were successfully initialized in init
        for (namespace_defs.items) |def| {
            if (actually_initialized_defs.contains(def.name)) {
                try init_processed_indices.put(def.idx, {});
                try init_only_def_names.put(def.name, {});
            }
        }

        // Create namespace context (use "user" as default if no namespace declared)
        const ns_name = namespace_name orelse "user";
        var ns_ctx = NamespaceContext{
            .name = ns_name,
            .def_names = &namespace_def_names,
            .init_only_def_names = &init_only_def_names,
            .requires = &requires_map,
        };

        // Check if user defined a 'main' function
        var has_user_main = false;
        for (expanded_expressions.items) |expr| {
            if (expr.* == .list) {
                var iter = expr.list.iterator();
                if (iter.next()) |head_val| {
                    if (head_val.isSymbol() and std.mem.eql(u8, head_val.symbol, "def")) {
                        if (iter.next()) |name_val| {
                            if (name_val.isSymbol() and std.mem.eql(u8, name_val.symbol, "main")) {
                                has_user_main = true;
                                break;
                            }
                        }
                    }
                }
            }
        }

        // Second pass: emit full definitions and expressions (skip namespace defs and init-processed exprs)
        for (expanded_expressions.items, report.typed.items, 0..) |expr, typed_val, expr_idx| {
            // Skip expressions that were already processed in the init function
            if (namespace_defs.items.len > 0 and init_processed_indices.contains(expr_idx)) {
                continue;
            }
            try self.emitTopLevel(prelude_writer, body_writer, expr, typed_val, &checker, &includes, &ns_ctx);
        }

        var output = std.ArrayList(u8){};
        defer output.deinit(self.allocator.*);

        try output.appendSlice(self.allocator.*, "#include <stdio.h>\n");
        if (includes.need_stdbool) {
            try output.appendSlice(self.allocator.*, "#include <stdbool.h>\n");
        }
        if (includes.need_stdint) {
            try output.appendSlice(self.allocator.*, "#include <stdint.h>\n");
        }
        if (includes.need_stddef) {
            try output.appendSlice(self.allocator.*, "#include <stddef.h>\n");
        }
        if (includes.need_stdlib) {
            try output.appendSlice(self.allocator.*, "#include <stdlib.h>\n");
        }
        if (includes.need_string) {
            try output.appendSlice(self.allocator.*, "#include <string.h>\n");
        }
        try output.appendSlice(self.allocator.*, "\n");

        if (forward_decls.items.len > 0) {
            try output.appendSlice(self.allocator.*, forward_decls.items);
            try output.appendSlice(self.allocator.*, "\n");
        }

        // Add namespace struct and global instance
        if (namespace_struct.items.len > 0) {
            try output.appendSlice(self.allocator.*, namespace_struct.items);
        }

        if (prelude.items.len > 0) {
            try output.appendSlice(self.allocator.*, prelude.items);
            try output.appendSlice(self.allocator.*, "\n");
        }

        // Add namespace init function
        if (namespace_init.items.len > 0) {
            try output.appendSlice(self.allocator.*, namespace_init.items);
        }

        switch (target) {
            .executable => {
                // Only generate wrapper if user hasn't defined main
                if (!has_user_main) {
                    try output.appendSlice(self.allocator.*, "int main() {\n");

                    // NOTE: For executables, all namespace code is compiled in, so we need
                    // to initialize all namespaces. For bundles, required namespaces are
                    // loaded dynamically and initialized by their own lisp_main functions.

                    // Collect all namespaces in topological order (dependencies before dependents)
                    var visited = std.StringHashMap(void).init(self.allocator.*);
                    defer visited.deinit();
                    var sorted_namespaces = std.ArrayList([]const u8){};
                    defer sorted_namespaces.deinit(self.allocator.*);
                    try self.collectTransitiveDependencies(&visited, &sorted_namespaces);

                    // Initialize all namespaces in topological order
                    for (sorted_namespaces.items) |ns_to_init| {
                        const sanitized = try self.sanitizeIdentifier(ns_to_init);
                        defer self.allocator.*.free(sanitized);
                        try output.print(self.allocator.*, "    init_namespace_{s}(&g_{s});\n", .{ sanitized, sanitized });
                    }

                    // Initialize this namespace if present (and not already in the list)
                    if (namespace_defs.items.len > 0) {
                        var already_init = false;
                        for (sorted_namespaces.items) |ns_to_init| {
                            if (std.mem.eql(u8, ns_to_init, ns_name)) {
                                already_init = true;
                                break;
                            }
                        }
                        if (!already_init) {
                            const sanitized_ns = try self.sanitizeIdentifier(ns_name);
                            defer self.allocator.*.free(sanitized_ns);
                            try output.print(self.allocator.*, "    init_namespace_{s}(&g_{s});\n", .{ sanitized_ns, sanitized_ns });
                        }
                    }
                    if (body.items.len > 0) {
                        try output.appendSlice(self.allocator.*, body.items);
                    }
                    try output.appendSlice(self.allocator.*, "    return 0;\n}\n");
                } else {
                    // User defined main, just emit the body (which contains the main function)
                    if (body.items.len > 0) {
                        try output.appendSlice(self.allocator.*, body.items);
                    }
                }
            },
            .bundle => {
                try output.appendSlice(self.allocator.*, "void lisp_main(void) {\n");

                // Initialize all required namespaces first (in topological order)
                // This ensures that transitive dependencies are initialized before they're used
                var visited_bundle = std.StringHashMap(void).init(self.allocator.*);
                defer visited_bundle.deinit();
                var sorted_deps = std.ArrayList([]const u8){};
                defer sorted_deps.deinit(self.allocator.*);
                try self.collectTransitiveDependencies(&visited_bundle, &sorted_deps);

                for (sorted_deps.items) |dep_ns_name| {
                    const sanitized = try self.sanitizeIdentifier(dep_ns_name);
                    defer self.allocator.*.free(sanitized);
                    try output.print(self.allocator.*, "    init_namespace_{s}(&g_{s});\n", .{ sanitized, sanitized });
                }

                // Initialize this namespace if present
                if (namespace_defs.items.len > 0) {
                    const sanitized_ns = try self.sanitizeIdentifier(ns_name);
                    defer self.allocator.*.free(sanitized_ns);
                    try output.print(self.allocator.*, "    init_namespace_{s}(&g_{s});\n", .{ sanitized_ns, sanitized_ns });
                }
                if (body.items.len > 0) {
                    try output.appendSlice(self.allocator.*, body.items);
                }
                try output.appendSlice(self.allocator.*, "}\n");
            },
        }

        return output.toOwnedSlice(self.allocator.*);
    }

    fn emitForwardDecl(self: *SimpleCCompiler, forward_writer: anytype, expr: *Value, typed: *TypedValue, checker: *TypeChecker, includes: *IncludeFlags) Error!void {
        switch (expr.*) {
            .list => |list| {
                var iter = list.iterator();
                const head_val = iter.next() orelse return;
                if (!head_val.isSymbol()) return;

                const head = head_val.symbol;

                // Handle extern declarations
                if (std.mem.eql(u8, head, "extern-fn")) {
                    // extern-fn: function is declared in C header, we just acknowledge it exists
                    // Don't generate any extern declaration - the C header already has it
                    // This avoids conflicts when headers define functions with different signatures
                    _ = typed; // Function type info is only needed for type checking
                    return;
                } else if (std.mem.eql(u8, head, "extern-type")) {
                    // extern-type just creates a type alias or forward declaration
                    const name_val = iter.next() orelse return;
                    if (!name_val.isSymbol()) return;
                    const type_name = name_val.symbol;

                    // For SDL, most types are opaque structs
                    try forward_writer.print("typedef struct {s} {s};\n", .{ type_name, type_name });
                    return;
                } else if (std.mem.eql(u8, head, "extern-union")) {
                    // extern-union creates a union type forward declaration
                    const name_val = iter.next() orelse return;
                    if (!name_val.isSymbol()) return;
                    const type_name = name_val.symbol;

                    // Create union typedef
                    try forward_writer.print("typedef union {s} {s};\n", .{ type_name, type_name });
                    return;
                } else if (std.mem.eql(u8, head, "extern-struct")) {
                    // extern-struct: type is defined in C header, we just acknowledge it exists
                    // Don't generate any typedef - the C header already has it
                    return;
                } else if (std.mem.eql(u8, head, "extern-var")) {
                    // extern-var creates an extern variable declaration
                    const name_val = iter.next() orelse return;
                    if (!name_val.isSymbol()) return;
                    const var_name = name_val.symbol;

                    const type_val = iter.next() orelse return;
                    if (type_val.isSymbol()) {
                        // Parse the type - for now just handle simple types
                        const type_str = self.cTypeFor(try self.parseSimpleType(type_val.symbol), includes) catch return;
                        try forward_writer.print("extern {s} {s};\n", .{ type_str, var_name });
                    }
                    return;
                } else if (std.mem.eql(u8, head, "include-header")) {
                    // Handle include-header directive
                    const header_val = iter.next() orelse return;
                    if (!header_val.isString()) return;
                    const header_name = header_val.string;

                    // Emit the #include directive
                    try forward_writer.print("#include \"{s}\"\n", .{header_name});
                    return;
                } else if (std.mem.eql(u8, head, "link-library")) {
                    // Collect library for linking
                    const lib_val = iter.next() orelse return;
                    if (!lib_val.isString()) return;
                    const lib_name = lib_val.string;

                    // Store library name for later use during compilation
                    try self.linked_libraries.append(self.allocator.*, try self.allocator.*.dupe(u8, lib_name));
                    return;
                } else if (std.mem.eql(u8, head, "compiler-flag")) {
                    // Collect compiler flag
                    const flag_val = iter.next() orelse return;
                    if (!flag_val.isString()) return;
                    const flag = flag_val.string;

                    // Store compiler flag for later use during compilation
                    try self.compiler_flags.append(self.allocator.*, try self.allocator.*.dupe(u8, flag));
                    return;
                } else if (std.mem.eql(u8, head, "def")) {
                    const name_val = iter.next() orelse return;
                    if (!name_val.isSymbol()) return;
                    const name = name_val.symbol;

                    // Check if this is a struct or enum definition by looking it up in type_defs
                    if (checker.type_defs.get(name)) |type_def| {
                        if (type_def == .struct_type) {
                            // This is a struct definition - emit struct declaration
                            const sanitized_name = try self.sanitizeIdentifier(name);
                            defer self.allocator.*.free(sanitized_name);

                            // Check if this is a self-referential struct (contains pointer to itself)
                            var is_self_referential = false;
                            for (type_def.struct_type.fields) |field| {
                                if (field.field_type == .pointer) {
                                    const pointee = field.field_type.pointer.*;
                                    if (pointee == .struct_type) {
                                        if (std.mem.eql(u8, pointee.struct_type.name, name)) {
                                            is_self_referential = true;
                                            break;
                                        }
                                    }
                                }
                            }

                            // If self-referential, emit forward declaration first
                            if (is_self_referential) {
                                try forward_writer.print("typedef struct {s} {s};\n", .{ sanitized_name, sanitized_name });
                            }

                            // Emit struct definition
                            if (is_self_referential) {
                                try forward_writer.print("struct {s} {{\n", .{sanitized_name});
                            } else {
                                try forward_writer.print("typedef struct {{\n", .{});
                            }

                            for (type_def.struct_type.fields) |field| {
                                const sanitized_field = try self.sanitizeIdentifier(field.name);
                                defer self.allocator.*.free(sanitized_field);
                                try forward_writer.print("    ", .{});
                                self.emitArrayDecl(forward_writer, sanitized_field, field.field_type, includes) catch {
                                    try forward_writer.print("// unsupported field type: {s}\n", .{field.name});
                                    continue;
                                };
                                try forward_writer.print(";\n", .{});
                            }

                            if (is_self_referential) {
                                try forward_writer.print("}};\n\n", .{});
                            } else {
                                try forward_writer.print("}} {s};\n\n", .{sanitized_name});
                            }
                            return;
                        } else if (type_def == .enum_type) {
                            // This is an enum definition - emit enum declaration
                            const sanitized_name = try self.sanitizeIdentifier(name);
                            defer self.allocator.*.free(sanitized_name);

                            try forward_writer.print("typedef enum {{\n", .{});
                            for (type_def.enum_type.variants) |variant| {
                                const sanitized_variant = try self.sanitizeIdentifier(variant.qualified_name.?);
                                defer self.allocator.*.free(sanitized_variant);
                                try forward_writer.print("    {s},\n", .{sanitized_variant});
                            }
                            try forward_writer.print("}} {s};\n\n", .{sanitized_name});
                            return;
                        }
                    }

                    // Check if this is a function definition

                    // Skip the type annotation if present
                    var maybe_value = iter.next();
                    if (maybe_value) |val| {
                        if (val.isList()) {
                            var val_iter = val.list.iterator();
                            if (val_iter.next()) |first| {
                                if (first.isSymbol() and std.mem.eql(u8, first.symbol, ":")) {
                                    maybe_value = iter.next();
                                }
                            }
                        }
                    }

                    if (maybe_value) |value_expr| {
                        if (value_expr.isList()) {
                            var fn_iter = value_expr.list.iterator();
                            const maybe_fn = fn_iter.next() orelse return;
                            if (maybe_fn.isSymbol() and std.mem.eql(u8, maybe_fn.symbol, "fn")) {
                                // This is a function definition - emit forward declaration
                                const var_type = checker.env.get(name) orelse return;
                                if (var_type != .function) return;

                                const return_type_str = self.cTypeFor(var_type.function.return_type, includes) catch return;
                                const sanitized_name = try self.sanitizeIdentifier(name);
                                defer self.allocator.*.free(sanitized_name);

                                // Don't emit 'static' for main function
                                const is_main = std.mem.eql(u8, name, "main");
                                if (is_main) {
                                    try forward_writer.print("{s} {s}(", .{ return_type_str, sanitized_name });
                                } else {
                                    try forward_writer.print("static {s} {s}(", .{ return_type_str, sanitized_name });
                                }

                                const params_val = fn_iter.next() orelse return;
                                if (!params_val.isVector()) return;
                                const params_vec = params_val.vector;

                                for (var_type.function.param_types, 0..) |param_type, i| {
                                    if (i > 0) try forward_writer.print(", ", .{});
                                    const param_val = params_vec.at(i);
                                    const sanitized_param = try self.sanitizeIdentifier(param_val.symbol);
                                    defer self.allocator.*.free(sanitized_param);
                                    // Use emitArrayDecl which handles function pointers correctly
                                    self.emitArrayDecl(forward_writer, sanitized_param, param_type, includes) catch return;
                                }

                                try forward_writer.print(");\n", .{});
                            }
                        }
                    }
                }
            },
            else => {},
        }
    }

    fn emitTopLevel(self: *SimpleCCompiler, def_writer: anytype, body_writer: anytype, expr: *Value, typed: *TypedValue, checker: *TypeChecker, includes: *IncludeFlags, ns_ctx: *NamespaceContext) Error!void {
        switch (expr.*) {
            .namespace => |ns| {
                try body_writer.print("    // namespace {s}\n", .{ns.name});
            },
            .require => |req| {
                try body_writer.print("    // require [{s} :as {s}]\n", .{ req.namespace, req.alias });
            },
            .list => |list| {
                var iter = list.iterator();
                const head_val = iter.next() orelse {
                    try self.emitStatement(body_writer, typed, includes, ns_ctx);
                    return;
                };

                if (!head_val.isSymbol()) {
                    try self.emitStatement(body_writer, typed, includes, ns_ctx);
                    return;
                }

                const head = head_val.symbol;

                // Skip extern declarations - they're already emitted in forward declarations
                if (std.mem.eql(u8, head, "extern-fn") or
                    std.mem.eql(u8, head, "extern-type") or
                    std.mem.eql(u8, head, "extern-union") or
                    std.mem.eql(u8, head, "extern-struct") or
                    std.mem.eql(u8, head, "extern-var") or
                    std.mem.eql(u8, head, "include-header") or
                    std.mem.eql(u8, head, "link-library") or
                    std.mem.eql(u8, head, "compiler-flag"))
                {
                    return;
                }

                if (std.mem.eql(u8, head, "def")) {
                    // Check if this def was initialized in namespace init
                    if (iter.next()) |name_val| {
                        if (name_val.isSymbol()) {
                            const def_name = name_val.symbol;
                            const is_namespace_def = ns_ctx.def_names.contains(def_name);
                            const was_init_in_init = if (ns_ctx.init_only_def_names) |init_only|
                                init_only.contains(def_name)
                            else
                                is_namespace_def;

                            if (was_init_in_init) {
                                // Skip - it was fully handled in namespace init
                                return;
                            }

                            if (is_namespace_def) {
                                // This is a namespace def that wasn't initialized in init
                                // Emit it as a namespace field assignment in main
                                const sanitized_name = try self.sanitizeIdentifier(def_name);
                                defer self.allocator.*.free(sanitized_name);
                                const sanitized_ns = try self.sanitizeIdentifier(ns_ctx.name.?);
                                defer self.allocator.*.free(sanitized_ns);

                                // typed is already the body/value of the def (not the whole def form)
                                // Special case: if it's a function definition, just assign the function name
                                const value_type = typed.getType();
                                if (value_type == .function) {
                                    try body_writer.print("    g_{s}.{s} = {s};\n", .{ sanitized_ns, sanitized_name, sanitized_name });
                                    return;
                                }

                                // Special case: if it's an array type, use memcpy
                                if (value_type == .array) {
                                    includes.need_string = true;
                                    try body_writer.print("    memcpy(g_{s}.{s}, ", .{ sanitized_ns, sanitized_name });
                                    self.writeExpressionTyped(body_writer, typed, ns_ctx, includes) catch |err| {
                                        switch (err) {
                                            Error.UnsupportedExpression, Error.MissingOperand, Error.InvalidIfForm => {
                                                try body_writer.print("0, 0); // unsupported\n", .{});
                                                return;
                                            },
                                            else => return err,
                                        }
                                    };
                                    try body_writer.print(", sizeof(g_{s}.{s}));\n", .{ sanitized_ns, sanitized_name });
                                    return;
                                }

                                // For non-function, non-array values, emit the expression
                                try body_writer.print("    g_{s}.{s} = ", .{ sanitized_ns, sanitized_name });
                                self.writeExpressionTyped(body_writer, typed, ns_ctx, includes) catch |err| {
                                    switch (err) {
                                        Error.UnsupportedExpression, Error.MissingOperand, Error.InvalidIfForm => {
                                            try body_writer.print("0; // unsupported\n", .{});
                                            return;
                                        },
                                        else => return err,
                                    }
                                };
                                try body_writer.print(";\n", .{});
                                return;
                            }
                        }
                    }
                    // Not a namespace def - reset iterator and emit as normal standalone def
                    iter = list.iterator();
                    _ = iter.next(); // skip 'def' again
                    try self.emitDefinition(def_writer, expr, typed, checker, includes, ns_ctx);
                    return;
                }

                try self.emitStatement(body_writer, typed, includes, ns_ctx);
            },
            else => {
                try self.emitStatement(body_writer, typed, includes, ns_ctx);
            },
        }
    }

    fn emitDefinition(self: *SimpleCCompiler, def_writer: anytype, list_expr: *Value, typed: *TypedValue, checker: *TypeChecker, includes: *IncludeFlags, ns_ctx: *NamespaceContext) Error!void {
        var iter = list_expr.list.iterator();
        _ = iter.next(); // Skip 'def'

        const name_val = iter.next() orelse return Error.InvalidDefinition;
        if (!name_val.isSymbol()) return Error.InvalidDefinition;
        const name = name_val.symbol;

        // Skip type definitions (struct/enum) - they're already declared in forward_decls
        if (checker.type_defs.get(name)) |_| {
            return;
        }

        var values_buf: [8]*Value = undefined;
        var value_count: usize = 0;
        while (iter.next()) |val| {
            if (value_count >= values_buf.len) {
                const repr = try self.formatValue(list_expr);
                defer self.allocator.*.free(repr);
                try def_writer.print("// unsupported definition: {s}\n", .{repr});
                return;
            }
            values_buf[value_count] = val;
            value_count += 1;
        }

        const value_expr = switch (value_count) {
            1 => values_buf[0],
            2 => values_buf[1],
            else => {
                const repr = try self.formatValue(list_expr);
                defer self.allocator.*.free(repr);
                try def_writer.print("// unsupported definition: {s}\n", .{repr});
                return;
            },
        };
        const var_type = checker.env.get(name) orelse {
            const repr = try self.formatValue(list_expr);
            defer self.allocator.*.free(repr);
            try def_writer.print("// unknown type for definition: {s}\n", .{repr});
            return;
        };

        if (value_expr.isList()) {
            var fn_iter = value_expr.list.iterator();
            const maybe_fn = fn_iter.next() orelse return Error.InvalidFunction;
            if (maybe_fn.isSymbol() and std.mem.eql(u8, maybe_fn.symbol, "fn")) {
                if (var_type != .function) {
                    return Error.UnsupportedType;
                }
                try self.emitFunctionDefinition(def_writer, name, value_expr, typed, var_type, includes, ns_ctx);
                return;
            }
        }

        const c_type = self.cTypeFor(var_type, includes) catch |err| {
            if (err == Error.UnsupportedType) {
                const repr = try self.formatValue(list_expr);
                defer self.allocator.*.free(repr);
                try def_writer.print("// unsupported definition: {s}\n", .{repr});
                return;
            }
            return err;
        };

        const sanitized_var_name = try self.sanitizeIdentifier(name);
        defer self.allocator.*.free(sanitized_var_name);
        try def_writer.print("{s} {s} = ", .{ c_type, sanitized_var_name });

        // Use typed AST for code generation
        self.writeExpressionTyped(def_writer, typed, ns_ctx, includes) catch |err| {
            switch (err) {
                Error.UnsupportedExpression, Error.MissingOperand, Error.InvalidIfForm => {
                    const repr = try self.formatValue(list_expr);
                    defer self.allocator.*.free(repr);
                    try def_writer.print("0; // unsupported definition: {s}\n", .{repr});
                    return;
                },
                else => return err,
            }
        };
        try def_writer.print(";\n", .{});
    }

    fn emitFunctionDefinition(self: *SimpleCCompiler, def_writer: anytype, name: []const u8, fn_expr: *Value, fn_typed: *TypedValue, fn_type: Type, includes: *IncludeFlags, ns_ctx: *NamespaceContext) Error!void {
        if (fn_type != .function) return Error.UnsupportedType;

        const param_types = fn_type.function.param_types;
        var fn_iter = fn_expr.list.iterator();
        _ = fn_iter.next(); // Skip 'fn'
        const params_val = fn_iter.next() orelse return Error.InvalidFunction;
        if (!params_val.isVector()) return Error.InvalidFunction;
        const params_vec = params_val.vector;

        if (param_types.len != params_vec.len()) {
            return Error.InvalidFunction;
        }

        // Extract typed body expressions from fn_typed
        // fn_typed is a list like (fn [params...] body...)
        if (fn_typed.* != .list) return Error.InvalidFunction;
        const typed_list = fn_typed.list;
        if (typed_list.elements.len < 2) return Error.InvalidFunction;

        const typed_body_count = typed_list.elements.len - 2; // Subtract 'fn' symbol and params vector
        if (typed_body_count == 0) return Error.InvalidFunction;

        const return_type_str = self.cTypeFor(fn_type.function.return_type, includes) catch |err| {
            if (err == Error.UnsupportedType) {
                const repr = try self.formatValue(fn_expr);
                defer self.allocator.*.free(repr);
                try def_writer.print("// unsupported function: {s}\n", .{repr});
                return;
            }
            return err;
        };

        if (params_vec.len() > 32) {
            const repr = try self.formatValue(fn_expr);
            defer self.allocator.*.free(repr);
            try def_writer.print("// unsupported function (too many parameters): {s}\n", .{repr});
            return;
        }

        const sanitized_name = try self.sanitizeIdentifier(name);
        defer self.allocator.*.free(sanitized_name);
        // Emit 'main' without 'static' keyword so it can be an entry point
        const is_main = std.mem.eql(u8, name, "main");
        if (is_main) {
            try def_writer.print("{s} {s}(", .{ return_type_str, sanitized_name });
        } else {
            try def_writer.print("static {s} {s}(", .{ return_type_str, sanitized_name });
        }
        var index: usize = 0;
        while (index < params_vec.len()) : (index += 1) {
            const param_val = params_vec.at(index);
            if (!param_val.isSymbol()) return Error.InvalidFunction;
            if (index > 0) {
                try def_writer.print(", ", .{});
            }
            const sanitized_param = try self.sanitizeIdentifier(param_val.symbol);
            defer self.allocator.*.free(sanitized_param);
            // Use emitArrayDecl which handles function pointers correctly
            self.emitArrayDecl(def_writer, sanitized_param, param_types[index], includes) catch |err| {
                if (err == Error.UnsupportedType) {
                    const repr = try self.formatValue(fn_expr);
                    defer self.allocator.*.free(repr);
                    try def_writer.print("/* unsupported */", .{});
                } else {
                    return err;
                }
            };
        }
        try def_writer.writeAll(") {\n");

        // Emit all body expressions except the last using typed AST
        for (typed_list.elements[2 .. typed_list.elements.len - 1]) |typed_stmt| {
            try def_writer.print("    ", .{});
            self.writeExpressionTyped(def_writer, typed_stmt, ns_ctx, includes) catch |err| {
                switch (err) {
                    Error.UnsupportedExpression, Error.MissingOperand, Error.InvalidIfForm => {
                        try def_writer.print("/* unsupported statement */", .{});
                    },
                    else => return err,
                }
            };
            try def_writer.writeAll(";\n");
        }

        // Emit return statement with last expression using typed AST
        // For void functions, just evaluate the expression without returning it
        const last_typed_expr = typed_list.elements[typed_list.elements.len - 1];
        const is_void_function = std.mem.eql(u8, return_type_str, "void");

        // Check if the last expression is a bare nil literal (not a complex expression with nil type)
        // We want to skip bare nil, but still emit complex expressions like (if ...) that have nil type
        const is_bare_nil = last_typed_expr.* == .nil;

        if (is_void_function and is_bare_nil) {
            // For void functions that end with a bare nil literal, just close the function
            // This avoids the "unused expression result" warning for bare `0;`
            try def_writer.writeAll("}\n");
        } else {
            if (!is_void_function) {
                try def_writer.print("    return ", .{});
            } else {
                try def_writer.print("    ", .{});
            }

            self.writeExpressionTyped(def_writer, last_typed_expr, ns_ctx, includes) catch |err| {
                switch (err) {
                    Error.UnsupportedExpression, Error.MissingOperand, Error.InvalidIfForm => {
                        const repr = try self.formatValue(fn_expr);
                        defer self.allocator.*.free(repr);
                        try def_writer.writeAll("0;\n}\n");
                        try def_writer.print("// unsupported function body: {s}\n", .{repr});
                        std.debug.print("ERROR writing return expression: {s}\n", .{@errorName(err)});
                        return;
                    },
                    else => return err,
                }
            };
            try def_writer.writeAll(";\n}\n");
        }
    }

    // Helper to parse type from type annotation expression
    fn parseTypeFromAnnotation(self: *SimpleCCompiler, annotation: *Value) Error!Type {
        // Type annotation is (: Type)
        if (!annotation.isList()) {
            std.debug.print("ERROR: Type annotation is not a list: {s}\n", .{@tagName(annotation.*)});
            return Error.InvalidTypeAnnotation;
        }
        var iter = annotation.list.iterator();
        const first = iter.next() orelse {
            std.debug.print("ERROR: Empty type annotation list\n", .{});
            return Error.InvalidTypeAnnotation;
        };
        if (!first.isKeyword() or first.keyword.len != 0) {
            std.debug.print("ERROR: Type annotation doesn't start with ':' keyword, got: {s}\n", .{@tagName(first.*)});
            return Error.InvalidTypeAnnotation;
        }

        const type_expr = iter.next() orelse {
            std.debug.print("ERROR: Type annotation missing type expression after ':'\n", .{});
            return Error.InvalidTypeAnnotation;
        };

        // Parse basic types
        if (type_expr.isSymbol()) {
            const type_name = type_expr.symbol;
            if (std.mem.eql(u8, type_name, "Int")) return Type.int;
            if (std.mem.eql(u8, type_name, "Float")) return Type.float;
            if (std.mem.eql(u8, type_name, "Bool")) return Type.bool;
            if (std.mem.eql(u8, type_name, "String")) return Type.string;
            if (std.mem.eql(u8, type_name, "I8")) return Type.i8;
            if (std.mem.eql(u8, type_name, "I16")) return Type.i16;
            if (std.mem.eql(u8, type_name, "I32")) return Type.i32;
            if (std.mem.eql(u8, type_name, "I64")) return Type.i64;
            if (std.mem.eql(u8, type_name, "U8")) return Type.u8;
            if (std.mem.eql(u8, type_name, "U16")) return Type.u16;
            if (std.mem.eql(u8, type_name, "U32")) return Type.u32;
            if (std.mem.eql(u8, type_name, "U64")) return Type.u64;
            if (std.mem.eql(u8, type_name, "F32")) return Type.f32;
            if (std.mem.eql(u8, type_name, "F64")) return Type.f64;

            // For unknown type names, treat them as extern types
            // This is a hack - we create an extern type with the given name
            const extern_type_ptr = try self.allocator.*.create(type_checker.ExternType);
            extern_type_ptr.* = .{
                .name = type_name,
                .is_opaque = true, // Assume opaque since we don't have field info
            };
            return Type{ .extern_type = extern_type_ptr };
        }

        // For complex types (like Pointer, Array, etc.), we'd need more parsing
        if (type_expr.isList()) {
            var type_iter = type_expr.list.iterator();
            if (type_iter.next()) |first_type| {
                if (first_type.isSymbol()) {
                    const type_constructor = first_type.symbol;

                    // Handle (Pointer T)
                    if (std.mem.eql(u8, type_constructor, "Pointer")) {
                        const pointee_expr = type_iter.next() orelse {
                            std.debug.print("ERROR: Pointer type missing pointee type\n", .{});
                            return Error.InvalidTypeAnnotation;
                        };
                        const pointee_type = try self.parseTypeFromAnnotation_Simple(pointee_expr);
                        const pointee_ptr = try self.allocator.*.create(Type);
                        pointee_ptr.* = pointee_type;
                        return Type{ .pointer = pointee_ptr };
                    }

                    std.debug.print("ERROR: Unsupported type constructor: {s}\n", .{type_constructor});
                    return Error.InvalidTypeAnnotation;
                }
            }
        }

        std.debug.print("ERROR: Complex type annotation not supported yet: {s}\n", .{@tagName(type_expr.*)});
        return Error.InvalidTypeAnnotation;
    }

    // Simplified type parser for pointee types (no type annotation wrapper)
    fn parseTypeFromAnnotation_Simple(self: *SimpleCCompiler, type_expr: *Value) Error!Type {
        if (type_expr.isSymbol()) {
            const type_name = type_expr.symbol;
            if (std.mem.eql(u8, type_name, "Int")) return Type.int;
            if (std.mem.eql(u8, type_name, "Float")) return Type.float;
            if (std.mem.eql(u8, type_name, "Bool")) return Type.bool;
            if (std.mem.eql(u8, type_name, "String")) return Type.string;
            if (std.mem.eql(u8, type_name, "Nil")) return Type.nil;
            if (std.mem.eql(u8, type_name, "Void")) return Type.void;
            if (std.mem.eql(u8, type_name, "I8")) return Type.i8;
            if (std.mem.eql(u8, type_name, "I16")) return Type.i16;
            if (std.mem.eql(u8, type_name, "I32")) return Type.i32;
            if (std.mem.eql(u8, type_name, "I64")) return Type.i64;
            if (std.mem.eql(u8, type_name, "U8")) return Type.u8;
            if (std.mem.eql(u8, type_name, "U16")) return Type.u16;
            if (std.mem.eql(u8, type_name, "U32")) return Type.u32;
            if (std.mem.eql(u8, type_name, "U64")) return Type.u64;
            if (std.mem.eql(u8, type_name, "F32")) return Type.f32;
            if (std.mem.eql(u8, type_name, "F64")) return Type.f64;

            // For unknown type names, treat them as extern types
            const extern_type_ptr = try self.allocator.*.create(type_checker.ExternType);
            extern_type_ptr.* = .{
                .name = type_name,
                .is_opaque = true,
            };
            return Type{ .extern_type = extern_type_ptr };
        }

        // Handle nested complex types like (Pointer (Pointer T))
        if (type_expr.isList()) {
            var type_iter = type_expr.list.iterator();
            if (type_iter.next()) |first_type| {
                if (first_type.isSymbol()) {
                    const type_constructor = first_type.symbol;

                    // Handle (Pointer T) recursively
                    if (std.mem.eql(u8, type_constructor, "Pointer")) {
                        const pointee_expr = type_iter.next() orelse {
                            std.debug.print("ERROR: Pointer type missing pointee type\n", .{});
                            return Error.InvalidTypeAnnotation;
                        };
                        const pointee_type = try self.parseTypeFromAnnotation_Simple(pointee_expr);
                        const pointee_ptr = try self.allocator.*.create(Type);
                        pointee_ptr.* = pointee_type;
                        return Type{ .pointer = pointee_ptr };
                    }

                    // Handle (-> [param_types...] return_type)
                    if (std.mem.eql(u8, type_constructor, "->")) {
                        const params_expr = type_iter.next() orelse {
                            std.debug.print("ERROR: Function type missing parameter list\n", .{});
                            return Error.InvalidTypeAnnotation;
                        };
                        const return_expr = type_iter.next() orelse {
                            std.debug.print("ERROR: Function type missing return type\n", .{});
                            return Error.InvalidTypeAnnotation;
                        };

                        // Parse parameter types
                        if (!params_expr.isVector()) {
                            std.debug.print("ERROR: Function parameter list must be a vector\n", .{});
                            return Error.InvalidTypeAnnotation;
                        }
                        const params_vec = params_expr.vector;
                        const param_types = try self.allocator.*.alloc(Type, params_vec.len());
                        for (0..params_vec.len()) |i| {
                            param_types[i] = try self.parseTypeFromAnnotation_Simple(params_vec.at(i));
                        }

                        // Parse return type
                        const return_type = try self.parseTypeFromAnnotation_Simple(return_expr);

                        // Create FunctionType on heap (Type.function expects *FunctionType)
                        const fn_type_ptr = try self.allocator.*.create(type_checker.FunctionType);
                        fn_type_ptr.* = type_checker.FunctionType{
                            .param_types = param_types,
                            .return_type = return_type,
                        };
                        return Type{ .function = fn_type_ptr };
                    }

                    std.debug.print("ERROR: Unsupported type constructor in simple parser: {s}\n", .{type_constructor});
                    return Error.InvalidTypeAnnotation;
                }
            }
        }

        std.debug.print("ERROR: Unsupported simple type: {s}\n", .{@tagName(type_expr.*)});
        return Error.InvalidTypeAnnotation;
    }

    fn sanitizeIdentifier(self: *SimpleCCompiler, name: []const u8) ![]u8 {
        // Check if it's a C keyword
        const c_keywords = [_][]const u8{
            "auto",       "break",    "case",     "char",   "const",   "continue",
            "default",    "do",       "double",   "else",   "enum",    "extern",
            "float",      "for",      "goto",     "if",     "inline",  "int",
            "long",       "register", "restrict", "return", "short",   "signed",
            "sizeof",     "static",   "struct",   "switch", "typedef", "union",
            "unsigned",   "void",     "volatile", "while",  "_Bool",   "_Complex",
            "_Imaginary",
        };

        var is_keyword = false;
        for (c_keywords) |kw| {
            if (std.mem.eql(u8, name, kw)) {
                is_keyword = true;
                break;
            }
        }

        if (is_keyword) {
            // Prefix with _ to avoid keyword collision
            var result = try self.allocator.*.alloc(u8, name.len + 1);
            result[0] = '_';
            for (name, 0..) |c, i| {
                result[i + 1] = if (c == '-' or c == '.' or c == '/') '_' else c;
            }
            return result;
        } else {
            var result = try self.allocator.*.alloc(u8, name.len);
            for (name, 0..) |c, i| {
                result[i] = if (c == '-' or c == '.' or c == '/') '_' else c;
            }
            return result;
        }
    }

    fn emitStatement(self: *SimpleCCompiler, body_writer: anytype, typed: *TypedValue, includes: *IncludeFlags, ns_ctx: *NamespaceContext) Error!void {
        // Emit as statement using typed code path
        try body_writer.print("    ", .{});
        self.writeExpressionTyped(body_writer, typed, ns_ctx, includes) catch |err| {
            switch (err) {
                Error.UnsupportedExpression, Error.MissingOperand, Error.InvalidIfForm => {
                    try body_writer.print("/* unsupported */", .{});
                },
                else => return err,
            }
        };
        try body_writer.print(";\n", .{});
    }

    // Helper method to dispatch special forms using a switch statement
    fn dispatchSpecialForm(self: *SimpleCCompiler, writer: anytype, form: SpecialForm, l: anytype, ns_ctx: *NamespaceContext, includes: *IncludeFlags) Error!bool {
        switch (form) {
            .if_form => {
                if (l.elements.len != 4) return Error.UnsupportedExpression;

                // Check if this if expression has nil/void type
                // If so, emit as a statement to avoid "unused expression result" warnings
                const if_type = l.type;
                const is_void_if = if_type == .nil or if_type == .void;

                if (is_void_if) {
                    // Emit as if-else statement wrapped in compound expression
                    // Check if branches are bare nil to avoid emitting unused `0;`
                    const then_is_nil = l.elements[2].* == .nil;
                    const else_is_nil = l.elements[3].* == .nil;

                    try writer.print("({{ if (", .{});
                    try self.writeExpressionTyped(writer, l.elements[1], ns_ctx, includes);
                    try writer.print(") {{ ", .{});
                    if (!then_is_nil) {
                        try self.writeExpressionTyped(writer, l.elements[2], ns_ctx, includes);
                        try writer.print("; ", .{});
                    }
                    try writer.print("}} else {{ ", .{});
                    if (!else_is_nil) {
                        try self.writeExpressionTyped(writer, l.elements[3], ns_ctx, includes);
                        try writer.print("; ", .{});
                    }
                    try writer.print("}} }})", .{});
                } else {
                    // Emit as ternary expression for non-void types
                    try writer.print("(", .{});
                    try self.writeExpressionTyped(writer, l.elements[1], ns_ctx, includes);
                    try writer.print(" ? ", .{});
                    try self.writeExpressionTyped(writer, l.elements[2], ns_ctx, includes);
                    try writer.print(" : ", .{});
                    try self.writeExpressionTyped(writer, l.elements[3], ns_ctx, includes);
                    try writer.print(")", .{});
                }
                return true;
            },
            .while_form => {
                if (l.elements.len < 3) return Error.UnsupportedExpression;
                try writer.print("({{ while (", .{});
                try self.writeExpressionTyped(writer, l.elements[1], ns_ctx, includes);
                try writer.print(") {{ ", .{});
                for (l.elements[2..]) |body_stmt| {
                    try self.writeExpressionTyped(writer, body_stmt, ns_ctx, includes);
                    try writer.print("; ", .{});
                }
                try writer.print("}} }})", .{});
                return true;
            },
            .c_for => {
                if (l.elements.len < 5) return Error.UnsupportedExpression;
                if (l.elements[1].* != .vector) return Error.UnsupportedExpression;
                const init_vec = l.elements[1].vector;
                if (init_vec.elements.len != 3) return Error.UnsupportedExpression;

                const var_sym = init_vec.elements[0];
                const type_val = init_vec.elements[1];
                const init_val = init_vec.elements[2];

                if (var_sym.* != .symbol) return Error.UnsupportedExpression;
                if (type_val.* != .type_value) return Error.UnsupportedExpression;

                const var_name = var_sym.symbol.name;
                const var_type = type_val.type_value.value_type;

                const c_type = try self.cTypeFor(var_type, includes);
                const sanitized = try self.sanitizeIdentifier(var_name);
                defer self.allocator.*.free(sanitized);

                try writer.print("({{ for ({s} {s} = ", .{ c_type, sanitized });
                try self.writeExpressionTyped(writer, init_val, ns_ctx, includes);
                try writer.print("; ", .{});
                try self.writeExpressionTyped(writer, l.elements[2], ns_ctx, includes);
                try writer.print("; ", .{});
                try self.writeExpressionTyped(writer, l.elements[3], ns_ctx, includes);
                try writer.print(") {{ ", .{});

                for (l.elements[4..]) |body_stmt| {
                    try self.writeExpressionTyped(writer, body_stmt, ns_ctx, includes);
                    try writer.print("; ", .{});
                }

                try writer.print("}} }})", .{});
                return true;
            },
            .let_form => {
                if (l.elements.len < 3) return Error.UnsupportedExpression;
                if (l.elements[1].* != .vector) return Error.UnsupportedExpression;

                const bindings_vec = l.elements[1].vector;
                if (bindings_vec.elements.len % 3 != 0) return Error.UnsupportedExpression;
                const binding_count = bindings_vec.elements.len / 3;

                try writer.print("({{ ", .{});

                // Create a map to track local bindings in this let scope
                var local_bindings_map = std.StringHashMap(void).init(self.allocator.*);
                defer local_bindings_map.deinit();

                // Create a new context with local bindings for this let scope
                var let_ctx = ns_ctx.*;
                let_ctx.local_bindings = &local_bindings_map;

                var i: usize = 0;
                while (i < binding_count) : (i += 1) {
                    const name_typed = bindings_vec.elements[i * 3];
                    const type_typed = bindings_vec.elements[i * 3 + 1];
                    const value_typed = bindings_vec.elements[i * 3 + 2];

                    if (name_typed.* != .symbol) return Error.UnsupportedExpression;
                    if (type_typed.* != .type_value) return Error.UnsupportedExpression;

                    const var_name = name_typed.symbol.name;
                    const var_type = type_typed.type_value.value_type;

                    if (var_type == .nil) {
                        try self.writeExpressionTyped(writer, value_typed, &let_ctx, includes);
                        try writer.print("; ", .{});
                        continue;
                    }

                    const c_type = try self.cTypeFor(var_type, includes);
                    const sanitized = try self.sanitizeIdentifier(var_name);
                    defer self.allocator.*.free(sanitized);

                    try writer.print("{s} {s} = ", .{ c_type, sanitized });
                    // Add explicit cast for pointers to avoid const-qualifier warnings
                    if (var_type == .pointer) {
                        try writer.print("({s})", .{c_type});
                    }
                    try self.writeExpressionTyped(writer, value_typed, &let_ctx, includes);
                    try writer.print("; ", .{});

                    // Add this binding to the local bindings map AFTER emitting the init value
                    // This ensures the init value is evaluated in the context before this binding
                    // (supporting sequential let semantics where later bindings can see earlier ones)
                    try local_bindings_map.put(var_name, {});
                }

                for (l.elements[2 .. l.elements.len - 1]) |body_stmt| {
                    try self.writeExpressionTyped(writer, body_stmt, &let_ctx, includes);
                    try writer.print("; ", .{});
                }

                try self.writeExpressionTyped(writer, l.elements[l.elements.len - 1], &let_ctx, includes);
                try writer.print("; }})", .{});
                return true;
            },
            .do_form => {
                if (l.elements.len < 1) return Error.UnsupportedExpression;

                try writer.print("({{ ", .{});

                // Emit all but the last expression as statements (with semicolons)
                for (l.elements[1 .. l.elements.len - 1]) |body_stmt| {
                    try self.writeExpressionTyped(writer, body_stmt, ns_ctx, includes);
                    try writer.print("; ", .{});
                }

                // Emit the final expression as the return value
                if (l.elements.len > 1) {
                    try self.writeExpressionTyped(writer, l.elements[l.elements.len - 1], ns_ctx, includes);
                } else {
                    // Empty do block returns 0 (nil)
                    try writer.print("0", .{});
                }
                try writer.print("; }})", .{});
                return true;
            },
            .set => {
                if (l.elements.len != 3) return Error.UnsupportedExpression;
                try self.writeExpressionTyped(writer, l.elements[1], ns_ctx, includes);
                try writer.print(" = ", .{});
                try self.writeExpressionTyped(writer, l.elements[2], ns_ctx, includes);
                return true;
            },
            .add, .subtract, .multiply, .divide, .modulo => {
                if (l.elements.len < 2) return Error.UnsupportedExpression;
                const op_str = l.elements[0].symbol.name;

                if (form == .subtract and l.elements.len == 2) {
                    try writer.print("(-(", .{});
                    try self.writeExpressionTyped(writer, l.elements[1], ns_ctx, includes);
                    try writer.print("))", .{});
                    return true;
                }

                try writer.print("(", .{});
                try self.writeExpressionTyped(writer, l.elements[1], ns_ctx, includes);
                var i: usize = 2;
                while (i < l.elements.len) : (i += 1) {
                    try writer.print(" {s} ", .{op_str});
                    try self.writeExpressionTyped(writer, l.elements[i], ns_ctx, includes);
                }
                try writer.print(")", .{});
                return true;
            },
            .less_than, .greater_than, .less_equal, .greater_equal, .equal, .not_equal => {
                if (l.elements.len != 3) return Error.UnsupportedExpression;
                const op = l.elements[0].symbol.name;
                const c_op = if (std.mem.eql(u8, op, "=")) "==" else op;
                try writer.print("(", .{});
                try self.writeExpressionTyped(writer, l.elements[1], ns_ctx, includes);
                try writer.print(" {s} ", .{c_op});
                try self.writeExpressionTyped(writer, l.elements[2], ns_ctx, includes);
                try writer.print(")", .{});
                return true;
            },
            .and_op, .or_op => {
                if (l.elements.len != 3) return Error.UnsupportedExpression;
                const c_op = if (form == .and_op) "&&" else "||";
                try writer.print("(", .{});
                try self.writeExpressionTyped(writer, l.elements[1], ns_ctx, includes);
                try writer.print(" {s} ", .{c_op});
                try self.writeExpressionTyped(writer, l.elements[2], ns_ctx, includes);
                try writer.print(")", .{});
                return true;
            },
            .not_op => {
                if (l.elements.len != 2) return Error.UnsupportedExpression;
                try writer.print("(!(", .{});
                try self.writeExpressionTyped(writer, l.elements[1], ns_ctx, includes);
                try writer.print("))", .{});
                return true;
            },
            .bitwise_and, .bitwise_or, .bitwise_xor, .bitwise_shl, .bitwise_shr => {
                if (l.elements.len < 2) return Error.UnsupportedExpression;
                const op_name = l.elements[0].symbol.name;
                const c_op = if (std.mem.eql(u8, op_name, "bitwise-and"))
                    "&"
                else if (std.mem.eql(u8, op_name, "bitwise-or"))
                    "|"
                else if (std.mem.eql(u8, op_name, "bitwise-xor"))
                    "^"
                else if (std.mem.eql(u8, op_name, "bitwise-shl"))
                    "<<"
                else if (std.mem.eql(u8, op_name, "bitwise-shr"))
                    ">>"
                else
                    op_name;

                try writer.print("(", .{});
                try self.writeExpressionTyped(writer, l.elements[1], ns_ctx, includes);
                var i: usize = 2;
                while (i < l.elements.len) : (i += 1) {
                    try writer.print(" {s} ", .{c_op});
                    try self.writeExpressionTyped(writer, l.elements[i], ns_ctx, includes);
                }
                try writer.print(")", .{});
                return true;
            },
            .bitwise_not => {
                if (l.elements.len != 2) return Error.UnsupportedExpression;
                try writer.print("(~(", .{});
                try self.writeExpressionTyped(writer, l.elements[1], ns_ctx, includes);
                try writer.print("))", .{});
                return true;
            },
            .array_ref => {
                if (l.elements.len != 3) return false;
                try self.writeExpressionTyped(writer, l.elements[1], ns_ctx, includes);
                try writer.print("[", .{});
                try self.writeExpressionTyped(writer, l.elements[2], ns_ctx, includes);
                try writer.print("]", .{});
                return true;
            },
            .array_set => {
                if (l.elements.len != 4 or l.type != .nil) return false;

                // Check if the value being assigned is an array type
                const value_type = l.elements[3].getType();
                if (value_type == .array) {
                    // For array assignments, use memcpy since C doesn't allow direct array assignment
                    includes.need_string = true;
                    try writer.print("memcpy(", .{});
                    try self.writeExpressionTyped(writer, l.elements[1], ns_ctx, includes);
                    try writer.print("[", .{});
                    try self.writeExpressionTyped(writer, l.elements[2], ns_ctx, includes);
                    try writer.print("], ", .{});
                    try self.writeExpressionTyped(writer, l.elements[3], ns_ctx, includes);
                    try writer.print(", sizeof(", .{});
                    try self.writeExpressionTyped(writer, l.elements[3], ns_ctx, includes);
                    try writer.print("))", .{});
                    return true;
                }

                // For non-array values, use regular assignment
                try writer.print("(", .{});
                try self.writeExpressionTyped(writer, l.elements[1], ns_ctx, includes);
                try writer.print("[", .{});
                try self.writeExpressionTyped(writer, l.elements[2], ns_ctx, includes);
                try writer.print("] = ", .{});
                try self.writeExpressionTyped(writer, l.elements[3], ns_ctx, includes);
                try writer.print(")", .{});
                return true;
            },
            .array_length => {
                if (l.elements.len != 2) return false;
                const array_type = l.elements[1].getType();
                if (array_type == .array) {
                    try writer.print("{d}", .{array_type.array.size});
                    return true;
                }
                return false;
            },
            .array_create => {
                if (l.type != .array) return false;
                const array_type = l.type.array;
                const elem_c_type = try self.cTypeFor(array_type.element_type, includes);

                if (l.elements.len == 3) {
                    try writer.print("({{ {s} __tmp_arr[{d}]; for (size_t __i = 0; __i < {d}; __i++) __tmp_arr[__i] = ", .{ elem_c_type, array_type.size, array_type.size });
                    try self.writeExpressionTyped(writer, l.elements[2], ns_ctx, includes);
                    try writer.print("; __tmp_arr; }})", .{});
                } else {
                    try writer.print("({{ {s} __tmp_arr[{d}]; __tmp_arr; }})", .{ elem_c_type, array_type.size });
                }
                return true;
            },
            .array_ptr => {
                if (l.elements.len != 3) return false;
                try writer.print("(&", .{});
                try self.writeExpressionTyped(writer, l.elements[1], ns_ctx, includes);
                try writer.print("[", .{});
                try self.writeExpressionTyped(writer, l.elements[2], ns_ctx, includes);
                try writer.print("])", .{});
                return true;
            },
            .allocate_array => {
                if (l.type != .pointer) return false;
                const elem_type = l.type.pointer.*;
                const elem_c_type = try self.cTypeFor(elem_type, includes);
                includes.need_stdlib = true;

                const size_typed = l.elements[2];

                if (l.elements.len == 4) {
                    try writer.print("({{ {s}* __arr = ({s}*)malloc(", .{ elem_c_type, elem_c_type });
                    try self.writeExpressionTyped(writer, size_typed, ns_ctx, includes);
                    try writer.print(" * sizeof({s})); for (size_t __i = 0; __i < ", .{elem_c_type});
                    try self.writeExpressionTyped(writer, size_typed, ns_ctx, includes);
                    try writer.print("; __i++) __arr[__i] = ", .{});
                    try self.writeExpressionTyped(writer, l.elements[3], ns_ctx, includes);
                    try writer.print("; __arr; }})", .{});
                } else {
                    try writer.print("({s}*)malloc(", .{elem_c_type});
                    try self.writeExpressionTyped(writer, size_typed, ns_ctx, includes);
                    try writer.print(" * sizeof({s}))", .{elem_c_type});
                }
                return true;
            },
            .deallocate_array => {
                if (l.elements.len != 2) return false;
                includes.need_stdlib = true;
                try writer.print("free(", .{});
                try self.writeExpressionTyped(writer, l.elements[1], ns_ctx, includes);
                try writer.print(")", .{});
                return true;
            },
            .pointer_index_read => {
                if (l.elements.len != 3) return false;
                try self.writeExpressionTyped(writer, l.elements[1], ns_ctx, includes);
                try writer.print("[", .{});
                try self.writeExpressionTyped(writer, l.elements[2], ns_ctx, includes);
                try writer.print("]", .{});
                return true;
            },
            .pointer_index_write => {
                if (l.elements.len != 4 or l.type != .nil) return false;
                try writer.print("(", .{});
                try self.writeExpressionTyped(writer, l.elements[1], ns_ctx, includes);
                try writer.print("[", .{});
                try self.writeExpressionTyped(writer, l.elements[2], ns_ctx, includes);
                try writer.print("] = ", .{});
                try self.writeExpressionTyped(writer, l.elements[3], ns_ctx, includes);
                try writer.print(")", .{});
                return true;
            },
            .printf_fn => {
                if (l.type != .i32) return false;
                try writer.print("printf(", .{});
                for (l.elements[1..], 0..) |arg, i| {
                    if (i > 0) try writer.print(", ", .{});
                    try self.writeExpressionTyped(writer, arg, ns_ctx, includes);
                }
                try writer.print(")", .{});
                return true;
            },
            .c_str => {
                // c-str just passes through the string literal (element[1])
                if (l.elements.len == 2 and l.type == .c_string) {
                    try self.writeExpressionTyped(writer, l.elements[1], ns_ctx, includes);
                    return true;
                }
                return false;
            },
            .allocate => {
                // allocate: [allocate, type-marker, value?]
                if (l.elements.len < 2 or l.elements.len > 3) return false;
                const ptr_type = l.type;
                if (ptr_type != .pointer) return false;
                const pointee = ptr_type.pointer.*;
                includes.need_stdlib = true;
                try writer.print("({{ ", .{});
                const c_type = try self.cTypeFor(pointee, includes);
                try writer.print("{s}* __tmp_ptr = malloc(sizeof({s})); ", .{ c_type, c_type });

                // Initialize if value provided (element at index 2)
                if (l.elements.len == 3) {
                    try writer.print("*__tmp_ptr = ", .{});
                    try self.writeExpressionTyped(writer, l.elements[2], ns_ctx, includes);
                    try writer.print("; ", .{});
                }

                try writer.print("__tmp_ptr; }})", .{});
                return true;
            },
            .deallocate => {
                // deallocate: 1 element (pointer), result type is nil
                if (l.elements.len != 2) return false;
                if (l.elements[1].getType() != .pointer) return false;
                if (l.type != .nil) return false;
                includes.need_stdlib = true;
                try writer.print("free(", .{});
                try self.writeExpressionTyped(writer, l.elements[1], ns_ctx, includes);
                try writer.print(")", .{});
                return true;
            },
            .dereference => {
                // dereference: [dereference, ptr]
                if (l.elements.len != 2) return false;
                try writer.print("(*", .{});
                try self.writeExpressionTyped(writer, l.elements[1], ns_ctx, includes);
                try writer.print(")", .{});
                return true;
            },
            .cast => {
                // cast: 2 elements (cast marker symbol, value)
                // cast generates: ((TargetType)value)
                if (l.elements.len != 2) return false;
                const c_type = try self.cTypeFor(l.type, includes);
                try writer.print("(({s})", .{c_type});
                try self.writeExpressionTyped(writer, l.elements[1], ns_ctx, includes);
                try writer.print(")", .{});
                return true;
            },
            .field_access => {
                // field access: [., struct-expr, field-symbol]
                if (l.elements.len != 3) return false;
                try self.writeExpressionTyped(writer, l.elements[1], ns_ctx, includes);
                if (l.elements[2].* == .symbol) {
                    try writer.print(".{s}", .{l.elements[2].symbol.name});
                    return true;
                }
                return false;
            },
            .pointer_write => {
                // pointer-write!: [pointer-write!, ptr, value]
                if (l.elements.len != 3) return false;
                try writer.print("(*", .{});
                try self.writeExpressionTyped(writer, l.elements[1], ns_ctx, includes);
                try writer.print(" = ", .{});
                try self.writeExpressionTyped(writer, l.elements[2], ns_ctx, includes);
                try writer.print(")", .{});
                return true;
            },
            .pointer_equal => {
                // pointer-equal?: [pointer-equal?, ptr1, ptr2]
                if (l.elements.len != 3) return false;
                try writer.print("(", .{});
                try self.writeExpressionTyped(writer, l.elements[1], ns_ctx, includes);
                try writer.print(" == ", .{});
                try self.writeExpressionTyped(writer, l.elements[2], ns_ctx, includes);
                try writer.print(")", .{});
                return true;
            },
            .pointer_field_read => {
                // pointer-field-read: [pointer-field-read, ptr, field]
                if (l.elements.len != 3) return false;
                try self.writeExpressionTyped(writer, l.elements[1], ns_ctx, includes);
                try writer.print("->", .{});
                const sanitized = try self.sanitizeIdentifier(l.elements[2].symbol.name);
                defer self.allocator.*.free(sanitized);
                try writer.print("{s}", .{sanitized});
                return true;
            },
            .pointer_field_write => {
                // pointer-field-write!: [pointer-field-write!, ptr, field, value]
                if (l.elements.len != 4) return false;
                try self.writeExpressionTyped(writer, l.elements[1], ns_ctx, includes);
                try writer.print("->", .{});
                const sanitized = try self.sanitizeIdentifier(l.elements[2].symbol.name);
                defer self.allocator.*.free(sanitized);
                try writer.print("{s} = ", .{sanitized});
                try self.writeExpressionTyped(writer, l.elements[3], ns_ctx, includes);
                return true;
            },
            .address_of => {
                // address-of: [address-of, var]
                if (l.elements.len != 2) return false;
                try writer.print("(&", .{});
                try self.writeExpressionTyped(writer, l.elements[1], ns_ctx, includes);
                try writer.print(")", .{});
                return true;
            },
            .unknown => {
                return false;
            },
        }
    }

    fn writeExpressionTyped(self: *SimpleCCompiler, writer: anytype, typed: *TypedValue, ns_ctx: *NamespaceContext, includes: *IncludeFlags) Error!void {
        switch (typed.*) {
            .struct_instance => |si| {
                // Emit C99 compound literal: (TypeName){field1, field2, ...}
                // Handle both regular struct_type and extern_type
                const type_name = switch (si.type) {
                    .struct_type => |st| st.name,
                    .extern_type => |et| et.name,
                    else => return Error.TypeCheckFailed,
                };
                const sanitized_name = try self.sanitizeIdentifier(type_name);
                defer self.allocator.*.free(sanitized_name);

                // Check if any field value is an array type - if so, use special handling
                var has_array_field = false;
                for (si.field_values) |field_val| {
                    if (field_val.getType() == .array) {
                        has_array_field = true;
                        break;
                    }
                }

                if (has_array_field) {
                    // Can't use array variables directly in compound literals
                    // Use compound statement expression with memcpy
                    includes.need_string = true; // For memcpy
                    try writer.print("({{ {s} __tmp_struct; ", .{sanitized_name});

                    // Get struct definition to access field names
                    const struct_def = switch (si.type) {
                        .struct_type => |st| st,
                        else => return Error.UnsupportedType,
                    };

                    // Initialize each field
                    for (si.field_values, 0..) |field_val, i| {
                        const field_type = field_val.getType();
                        const field_name = struct_def.fields[i].name;
                        const sanitized_field = try self.sanitizeIdentifier(field_name);
                        defer self.allocator.*.free(sanitized_field);

                        if (field_type == .array) {
                            // Use memcpy for array fields
                            try writer.print("memcpy(__tmp_struct.{s}, ", .{sanitized_field});
                            try self.writeExpressionTyped(writer, field_val, ns_ctx, includes);
                            try writer.print(", sizeof(__tmp_struct.{s})); ", .{sanitized_field});
                        } else {
                            // Regular assignment for non-array fields
                            try writer.print("__tmp_struct.{s} = ", .{sanitized_field});
                            try self.writeExpressionTyped(writer, field_val, ns_ctx, includes);
                            try writer.print("; ", .{});
                        }
                    }

                    try writer.print("__tmp_struct; }})", .{});
                } else {
                    // No array fields, use regular compound literal
                    try writer.print("({s}){{", .{sanitized_name});
                    for (si.field_values, 0..) |field_val, i| {
                        if (i > 0) try writer.print(", ", .{});
                        try self.writeExpressionTyped(writer, field_val, ns_ctx, includes);
                    }
                    try writer.print("}}", .{});
                }
            },
            .list => |l| {
                // Try special form dispatch first
                if (l.elements.len > 0 and l.elements[0].* == .symbol) {
                    const op = l.elements[0].symbol.name;
                    const form = identifySpecialForm(op);
                    const handled = try self.dispatchSpecialForm(writer, form, l, ns_ctx, includes);
                    if (handled) return;
                }

                // Check if this is a function or function pointer application
                // Function application has type .function or .pointer(.function) for the first element
                if (l.elements.len > 0) {
                    const first_elem = l.elements[0];
                    const first_type = first_elem.getType();
                    const is_function = first_type == .function or
                        (first_type == .pointer and first_type.pointer.* == .function);

                    if (is_function) {
                        // Emit function call
                        // For function pointers, we need: (*fn_ptr)(args...)
                        // For regular functions, we need: fn_name(args...)
                        const is_fn_ptr = first_type == .pointer;

                        if (is_fn_ptr) {
                            try writer.print("(*", .{});
                        }
                        try self.writeExpressionTyped(writer, first_elem, ns_ctx, includes);
                        if (is_fn_ptr) {
                            try writer.print(")", .{});
                        }

                        try writer.print("(", .{});
                        var i: usize = 1;
                        while (i < l.elements.len) : (i += 1) {
                            if (i > 1) try writer.print(", ", .{});
                            try self.writeExpressionTyped(writer, l.elements[i], ns_ctx, includes);
                        }
                        try writer.print(")", .{});
                        return;
                    }
                }

                return Error.UnsupportedExpression;
            },
            .int => |i| try writer.print("{d}", .{i.value}),
            .float => |f| try writer.print("{d}", .{f.value}),
            .string => |s| try writer.print("\"{s}\"", .{s.value}),
            .symbol => |sym| {
                // Check if this is pointer-null
                if (std.mem.eql(u8, sym.name, "pointer-null")) {
                    try writer.print("NULL", .{});
                    return;
                }

                // Check if this is a boolean literal
                if (std.mem.eql(u8, sym.name, "true")) {
                    try writer.print("1", .{});
                    return;
                }
                if (std.mem.eql(u8, sym.name, "false")) {
                    try writer.print("0", .{});
                    return;
                }

                // Check if this is an enum variant (has type enum_type AND qualified name with /)
                // We need to distinguish enum variants (Color/Red) from variables of enum type (pc: Color)
                if (sym.type == .enum_type and std.mem.indexOf(u8, sym.name, "/") != null) {
                    // Enum variants like Color/Red become ENUM_VARIANT in C
                    const sanitized = try self.sanitizeIdentifier(sym.name);
                    defer self.allocator.*.free(sanitized);
                    try writer.print("{s}", .{sanitized});
                    return;
                }

                // Check if this symbol is a local binding (let-bound variable)
                // Local bindings shadow namespace definitions
                const is_local = if (ns_ctx.local_bindings) |local_map|
                    local_map.contains(sym.name)
                else
                    false;

                // Check if this is a qualified name from a required namespace (contains `/`)
                // But not an enum variant (already handled above)
                if (!is_local and sym.type != .enum_type) {
                    if (std.mem.indexOf(u8, sym.name, "/")) |slash_pos| {
                        const alias = sym.name[0..slash_pos];
                        const def_name = sym.name[slash_pos + 1 ..];

                        // Look up the namespace for this alias
                        if (ns_ctx.requires) |requires_map| {
                            if (requires_map.get(alias)) |namespace_name| {
                                // Generate reference to the other namespace's variable
                                const sanitized_ns = try self.sanitizeIdentifier(namespace_name);
                                defer self.allocator.*.free(sanitized_ns);
                                const sanitized_field = try self.sanitizeIdentifier(def_name);
                                defer self.allocator.*.free(sanitized_field);
                                try writer.print("g_{s}.{s}", .{ sanitized_ns, sanitized_field });
                                return;
                            }
                        }
                    }
                }

                // Check if this symbol is in the namespace (and not shadowed by a local binding)
                if (!is_local and ns_ctx.def_names.*.contains(sym.name)) {
                    if (ns_ctx.name) |_| {
                        const sanitized_field = try self.sanitizeIdentifier(sym.name);
                        defer self.allocator.*.free(sanitized_field);

                        if (ns_ctx.in_init_function) {
                            // In init function, use ns->field
                            try writer.print("ns->{s}", .{sanitized_field});
                        } else {
                            // In regular code, use g_namespace.field
                            const sanitized_ns = try self.sanitizeIdentifier(ns_ctx.name.?);
                            defer self.allocator.*.free(sanitized_ns);
                            try writer.print("g_{s}.{s}", .{ sanitized_ns, sanitized_field });
                        }
                        return;
                    }
                }
                const sanitized = try self.sanitizeIdentifier(sym.name);
                defer self.allocator.*.free(sanitized);
                try writer.print("{s}", .{sanitized});
            },
            .nil => |n| {
                // Check if this is pointer-null (has pointer type)
                if (n.type == .pointer) {
                    try writer.print("NULL", .{});
                } else {
                    try writer.print("0", .{});
                }
            },
            else => return Error.UnsupportedExpression,
        }
    }

    // Helper to emit array declarations which have special C syntax: type name[size]
    // For multi-dimensional arrays, we need to emit: type name[size1][size2]...
    fn emitArrayDecl(self: *SimpleCCompiler, writer: anytype, var_name: []const u8, var_type: Type, includes: *IncludeFlags) Error!void {
        if (var_type == .array) {
            // Get the base element type and collect all array dimensions
            var current_type = var_type;
            var dimensions: [16]usize = undefined; // Support up to 16 dimensions
            var dim_count: usize = 0;

            // Traverse nested array types to collect dimensions
            while (current_type == .array and dim_count < 16) {
                dimensions[dim_count] = current_type.array.size;
                dim_count += 1;
                current_type = current_type.array.element_type;
            }

            // Emit the base type
            const base_c_type = try self.cTypeFor(current_type, includes);
            try writer.print("{s} {s}", .{ base_c_type, var_name });

            // Emit all dimensions: [size1][size2]...
            var i: usize = 0;
            while (i < dim_count) : (i += 1) {
                try writer.print("[{d}]", .{dimensions[i]});
            }
        } else if (var_type == .function) {
            // Direct function type (treated as function pointer in C)
            // Function pointer: return_type (*var_name)(params...)
            const fn_type = var_type.function;
            const return_type_str = try self.cTypeFor(fn_type.return_type, includes);
            try writer.print("{s} (*{s})(", .{ return_type_str, var_name });
            for (fn_type.param_types, 0..) |param_type, i| {
                if (i > 0) try writer.print(", ", .{});
                const param_type_str = try self.cTypeFor(param_type, includes);
                try writer.print("{s}", .{param_type_str});
            }
            try writer.print(")", .{});
        } else if (var_type == .pointer) {
            // Check if this is a pointer to a function
            const pointee = var_type.pointer.*;
            if (pointee == .function) {
                // Function pointer: return_type (*var_name)(params...)
                const fn_type = pointee.function;
                const return_type_str = try self.cTypeFor(fn_type.return_type, includes);
                try writer.print("{s} (*{s})(", .{ return_type_str, var_name });
                for (fn_type.param_types, 0..) |param_type, i| {
                    if (i > 0) try writer.print(", ", .{});
                    const param_type_str = try self.cTypeFor(param_type, includes);
                    try writer.print("{s}", .{param_type_str});
                }
                try writer.print(")", .{});
            } else {
                // Regular pointer
                const c_type = try self.cTypeFor(var_type, includes);
                try writer.print("{s} {s}", .{ c_type, var_name });
            }
        } else {
            const c_type = try self.cTypeFor(var_type, includes);
            try writer.print("{s} {s}", .{ c_type, var_name });
        }
    }

    fn cTypeFor(self: *SimpleCCompiler, type_info: Type, includes: *IncludeFlags) Error![]const u8 {
        return switch (type_info) {
            .int => int_type_name,
            .float => "double",
            .string => "const char *",
            .bool => blk: {
                includes.need_stdbool = true;
                break :blk "bool";
            },
            .u8 => blk: {
                includes.need_stdint = true;
                break :blk "uint8_t";
            },
            .u16 => blk: {
                includes.need_stdint = true;
                break :blk "uint16_t";
            },
            .u32 => blk: {
                includes.need_stdint = true;
                break :blk "uint32_t";
            },
            .u64 => blk: {
                includes.need_stdint = true;
                break :blk "uint64_t";
            },
            .usize => blk: {
                includes.need_stddef = true;
                break :blk "size_t";
            },
            .i8 => blk: {
                includes.need_stdint = true;
                break :blk "int8_t";
            },
            .i16 => blk: {
                includes.need_stdint = true;
                break :blk "int16_t";
            },
            .i32 => blk: {
                includes.need_stdint = true;
                break :blk "int32_t";
            },
            .i64 => blk: {
                includes.need_stdint = true;
                break :blk "int64_t";
            },
            .isize => int_type_name,
            .f32 => "float",
            .f64 => "double",
            .struct_type => |st| st.name,
            .enum_type => |et| et.name,
            .function => |fn_type| {
                // Function pointer: ReturnType (*)(Param1Type, Param2Type, ...)
                const ret_c_type = try self.cTypeFor(fn_type.return_type, includes);

                // Build parameter list
                var params_list = std.ArrayList([]const u8){};
                defer params_list.deinit(self.allocator.*);

                for (fn_type.param_types) |param_type| {
                    const param_c_type = try self.cTypeFor(param_type, includes);
                    try params_list.append(self.allocator.*, param_c_type);
                }

                // Join with ", "
                const params = try std.mem.join(self.allocator.*, ", ", params_list.items);
                defer self.allocator.free(params);

                // Format final string
                return try std.fmt.allocPrint(self.allocator.*, "{s} (*)({s})", .{ ret_c_type, params });
            },
            .pointer => |pointee| {
                // Special handling for pointer to function type
                // In Lisp: (Pointer (-> [Args] Return)) represents a function pointer
                // In C: ReturnType (*)(Args) is a function pointer
                if (pointee.* == .function) {
                    const fn_type = pointee.function;
                    const ret_c_type = try self.cTypeFor(fn_type.return_type, includes);

                    // Build parameter list
                    var params_list = std.ArrayList([]const u8){};
                    defer params_list.deinit(self.allocator.*);

                    for (fn_type.param_types) |param_type| {
                        const param_c_type = try self.cTypeFor(param_type, includes);
                        try params_list.append(self.allocator.*, param_c_type);
                    }

                    const params = try std.mem.join(self.allocator.*, ", ", params_list.items);
                    defer self.allocator.free(params);

                    // Function pointer: ReturnType (*)(Params)
                    return try std.fmt.allocPrint(self.allocator.*, "{s} (*)({s})", .{ ret_c_type, params });
                } else {
                    const pointee_c_type = try self.cTypeFor(pointee.*, includes);
                    // Allocate string for "pointee_type*"
                    const ptr_type_str = try std.fmt.allocPrint(self.allocator.*, "{s}*", .{pointee_c_type});
                    return ptr_type_str;
                }
            },
            .c_string => "const char*",
            .void => "void",
            .nil => "void",
            .extern_type => |et| et.name,
            .extern_function => Error.UnsupportedType, // Can't have values of extern function type directly
            .array => Error.UnsupportedType, // Arrays need special declaration syntax, handled separately
            else => Error.UnsupportedType,
        };
    }

    fn formatValue(self: *SimpleCCompiler, expr: *Value) ![]u8 {
        var buf = std.ArrayList(u8){};
        defer buf.deinit(self.allocator.*);
        try expr.format("", .{}, buf.writer(self.allocator.*));
        return buf.toOwnedSlice(self.allocator.*);
    }
};

test "simple c compiler basic program" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();

    var allocator = arena.allocator();
    var compiler = SimpleCCompiler.init(&allocator);

    const source =
        "(ns my.app)\n" ++
        "(def answer (: Int) 41)\n" ++
        "(+ answer 1)";

    const output = try compiler.compileString(source, .executable);

    const expected =
        "#include <stdio.h>\n\n" ++ "typedef struct {\n" ++ "    long long answer;\n" ++ "} Namespace_my_app;\n\n" ++ "Namespace_my_app g_my_app;\n\n\n" ++ "void init_namespace_my_app(Namespace_my_app* ns) {\n" ++ "    ns->answer = 41;\n" ++ "}\n\n" ++ "int main() {\n" ++ "    init_namespace_my_app(&g_my_app);\n" ++ "    // namespace my.app\n" ++ "    (g_my_app.answer + 1);\n" ++ "    return 0;\n" ++ "}\n";

    try std.testing.expectEqualStrings(expected, output);
}

test "simple c compiler fibonacci program with zig cc" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();

    var allocator = arena.allocator();
    var compiler = SimpleCCompiler.init(&allocator);

    const source =
        "(ns demo.core)\n" ++ "(def f0 (: Int) 0)\n" ++ "(def f1 (: Int) 1)\n" ++ "(def fib (: (-> [Int] Int)) (fn [n] (if (< n 2) n (+ (fib (- n 1)) (fib (- n 2))))))\n" ++ "(printf (c-str \"%lld\\n\") (fib 10))";

    const c_source = try compiler.compileString(source, .executable);

    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();

    try tmp.dir.writeFile(.{ .sub_path = "main.c", .data = c_source });

    const exe_name = if (builtin.os.tag == .windows) "program.exe" else "program";

    var cc_child = std.process.Child.init(&.{ "zig", "cc", "main.c", "-o", exe_name }, std.testing.allocator);
    cc_child.cwd_dir = tmp.dir;
    try cc_child.spawn();
    const cc_term = try cc_child.wait();
    switch (cc_term) {
        .Exited => |code| try std.testing.expect(code == 0),
        else => return error.TestUnexpectedResult,
    }

    const exe_path = try tmp.dir.realpathAlloc(std.testing.allocator, exe_name);
    defer std.testing.allocator.free(exe_path);

    var run_child = std.process.Child.init(&.{exe_path}, std.testing.allocator);
    run_child.stdout_behavior = .Pipe;
    try run_child.spawn();

    var stdout_file = run_child.stdout orelse return error.TestUnexpectedResult;
    const output = try stdout_file.readToEndAlloc(std.testing.allocator, 1024);
    defer std.testing.allocator.free(output);
    stdout_file.close();
    run_child.stdout = null;

    const run_term = try run_child.wait();
    switch (run_term) {
        .Exited => |code| try std.testing.expect(code == 0),
        else => return error.TestUnexpectedResult,
    }

    try std.testing.expectEqualStrings("55\n", output);
}

pub fn main() !void {
    var arena = std.heap.ArenaAllocator.init(std.heap.page_allocator);
    defer arena.deinit();
    var allocator = arena.allocator();

    var args = std.process.args();
    defer args.deinit();

    _ = args.next() orelse return;
    const source_path = args.next() orelse {
        std.debug.print("Usage: simple_c_compiler <source-file> [--run]\n", .{});
        return;
    };

    var run_flag = false;
    var bundle_flag = false;
    while (args.next()) |arg| {
        if (std.mem.eql(u8, arg, "--run")) {
            run_flag = true;
        } else if (std.mem.eql(u8, arg, "--bundle")) {
            bundle_flag = true;
        } else {
            std.debug.print("Unknown argument: {s}\n", .{arg});
            return;
        }
    }

    const target_kind: SimpleCCompiler.TargetKind = if (bundle_flag) .bundle else .executable;

    const source = try std.fs.cwd().readFileAlloc(allocator, source_path, std.math.maxInt(usize));
    defer allocator.free(source);

    var compiler = SimpleCCompiler.init(&allocator);
    defer compiler.deinit();
    const c_source = try compiler.compileString(source, target_kind);
    defer allocator.free(c_source);

    const basename = std.fs.path.basename(source_path);
    const stem = blk: {
        if (std.mem.lastIndexOfScalar(u8, basename, '.')) |idx| {
            break :blk basename[0..idx];
        } else {
            break :blk basename;
        }
    };

    const c_filename = try std.fmt.allocPrint(allocator, "{s}.c", .{stem});
    defer allocator.free(c_filename);

    const dir_opt = std.fs.path.dirname(source_path);
    if (dir_opt) |dir_path| {
        var dir = try std.fs.cwd().openDir(dir_path, .{});
        defer dir.close();
        try dir.writeFile(.{ .sub_path = c_filename, .data = c_source });
    } else {
        try std.fs.cwd().writeFile(.{ .sub_path = c_filename, .data = c_source });
    }

    const c_path = if (dir_opt) |dir_path|
        try std.fs.path.join(allocator, &.{ dir_path, c_filename })
    else
        try allocator.dupe(u8, c_filename);
    defer allocator.free(c_path);

    const c_real_path = try std.fs.cwd().realpathAlloc(allocator, c_path);
    defer allocator.free(c_real_path);

    std.debug.print("Generated C file: {s}\n", .{c_real_path});

    if (!run_flag) return;

    // Step 1: Compile to .o file
    const obj_name = try std.fmt.allocPrint(allocator, "{s}.o", .{stem});
    defer allocator.free(obj_name);

    const obj_path = if (dir_opt) |dir_path|
        try std.fs.path.join(allocator, &.{ dir_path, obj_name })
    else
        try allocator.dupe(u8, obj_name);
    defer allocator.free(obj_path);

    var compile_args = std.ArrayList([]const u8){};
    defer compile_args.deinit(allocator);
    try compile_args.appendSlice(allocator, &.{ "zig", "cc", "-c", c_path, "-o", obj_path });

    // Add compiler flags for compilation
    for (compiler.compiler_flags.items) |flag| {
        try compile_args.append(allocator, flag);
    }

    var compile_child = std.process.Child.init(compile_args.items, allocator);
    compile_child.stdin_behavior = .Inherit;
    compile_child.stdout_behavior = .Inherit;
    compile_child.stderr_behavior = .Inherit;
    try compile_child.spawn();
    const compile_term = try compile_child.wait();
    switch (compile_term) {
        .Exited => |code| {
            if (code != 0) {
                std.debug.print("zig cc compilation failed with code {d}\n", .{code});
                return;
            }
        },
        else => {
            std.debug.print("zig cc compilation failed\n", .{});
            return;
        },
    }

    const obj_real_path = try std.fs.cwd().realpathAlloc(allocator, obj_path);
    defer allocator.free(obj_real_path);
    std.debug.print("Compiled object file: {s}\n", .{obj_real_path});

    // Step 2: Link to final binary
    if (bundle_flag) {
        const bundle_name = try std.fmt.allocPrint(allocator, "{s}.bundle", .{stem});
        defer allocator.free(bundle_name);

        const bundle_path = if (dir_opt) |dir_path|
            try std.fs.path.join(allocator, &.{ dir_path, bundle_name })
        else
            try allocator.dupe(u8, bundle_name);
        defer allocator.free(bundle_path);

        // Link command with linked libraries
        var link_args = std.ArrayList([]const u8){};
        defer link_args.deinit(allocator);
        try link_args.appendSlice(allocator, &.{ "zig", "cc", "-dynamiclib", obj_path, "-o", bundle_path });

        // Add compiler flags
        for (compiler.compiler_flags.items) |flag| {
            try link_args.append(allocator, flag);
        }

        // Add linked libraries
        for (compiler.linked_libraries.items) |lib| {
            const lib_arg = try std.fmt.allocPrint(allocator, "-l{s}", .{lib});
            try link_args.append(allocator, lib_arg);
        }

        // Add required namespace bundles
        for (compiler.required_bundles.items) |bundle| {
            try link_args.append(allocator, bundle);
        }

        var link_child = std.process.Child.init(link_args.items, allocator);
        link_child.stdin_behavior = .Inherit;
        link_child.stdout_behavior = .Inherit;
        link_child.stderr_behavior = .Inherit;
        try link_child.spawn();
        const link_term = try link_child.wait();
        switch (link_term) {
            .Exited => |code| {
                if (code != 0) {
                    std.debug.print("zig cc linking failed with code {d}\n", .{code});
                    return;
                }
            },
            else => {
                std.debug.print("zig cc failed to link the bundle\n", .{});
                return;
            },
        }

        const bundle_real_path = try std.fs.cwd().realpathAlloc(allocator, bundle_path);
        defer allocator.free(bundle_real_path);

        std.debug.print("Built bundle: {s}\n", .{bundle_real_path});

        if (!run_flag) return;

        // Load the main bundle - the system's dynamic linker will automatically load required dylibs
        var lib = try std.DynLib.open(bundle_real_path);
        defer lib.close();

        // Call lisp_main for all required namespace dylibs to complete their initialization
        // (init functions do partial initialization, lisp_main completes it)
        for (compiler.required_bundles.items) |bundle| {
            const bundle_abs_path = try std.fs.cwd().realpathAlloc(allocator, bundle);
            defer allocator.free(bundle_abs_path);

            std.debug.print("Completing initialization of required namespace: {s}\n", .{bundle_abs_path});

            // Open the dylib directly to call its lisp_main
            var ns_lib = std.DynLib.open(bundle_abs_path) catch |err| {
                std.debug.print("WARNING: Failed to open required bundle {s}: {}\n", .{ bundle_abs_path, err });
                continue;
            };
            defer ns_lib.close();

            const ns_init_fn = ns_lib.lookup(*const fn () callconv(.c) void, "lisp_main") orelse {
                std.debug.print("WARNING: Required bundle {s} missing lisp_main\n", .{bundle_abs_path});
                continue;
            };
            @call(.auto, ns_init_fn, .{});
        }

        // Run the main bundle's lisp_main
        const entry_fn = lib.lookup(*const fn () callconv(.c) void, "lisp_main") orelse {
            std.debug.print("Bundle missing lisp_main entry\n", .{});
            return;
        };

        @call(.auto, entry_fn, .{});
        return;
    }

    const exe_name = if (builtin.os.tag == .windows)
        try std.fmt.allocPrint(allocator, "{s}.exe", .{stem})
    else
        try allocator.dupe(u8, stem);
    defer allocator.free(exe_name);

    const exe_path = if (dir_opt) |dir_path|
        try std.fs.path.join(allocator, &.{ dir_path, exe_name })
    else
        try allocator.dupe(u8, exe_name);
    defer allocator.free(exe_path);

    // Link command with linked libraries
    var link_args = std.ArrayList([]const u8){};
    defer link_args.deinit(allocator);
    try link_args.appendSlice(allocator, &.{ "zig", "cc", obj_path, "-o", exe_path });

    // Add compiler flags
    for (compiler.compiler_flags.items) |flag| {
        try link_args.append(allocator, flag);
    }

    // Add linked libraries
    for (compiler.linked_libraries.items) |lib| {
        const lib_arg = try std.fmt.allocPrint(allocator, "-l{s}", .{lib});
        try link_args.append(allocator, lib_arg);
    }

    // Add required namespace bundles
    for (compiler.required_bundles.items) |bundle| {
        try link_args.append(allocator, bundle);
    }

    var link_child = std.process.Child.init(link_args.items, allocator);
    link_child.stdin_behavior = .Inherit;
    link_child.stdout_behavior = .Inherit;
    link_child.stderr_behavior = .Inherit;
    try link_child.spawn();
    const link_term = try link_child.wait();
    switch (link_term) {
        .Exited => |code| {
            if (code != 0) {
                std.debug.print("zig cc linking failed with code {d}\n", .{code});
                return;
            }
        },
        else => {
            std.debug.print("zig cc failed to link the program\n", .{});
            return;
        },
    }

    const exe_real_path = try std.fs.cwd().realpathAlloc(allocator, exe_path);
    defer allocator.free(exe_real_path);

    std.debug.print("Built executable: {s}\n", .{exe_real_path});

    if (!run_flag) return;

    var run_child = std.process.Child.init(&.{exe_real_path}, allocator);
    run_child.stdin_behavior = .Inherit;
    run_child.stdout_behavior = .Inherit;
    run_child.stderr_behavior = .Inherit;
    try run_child.spawn();
    const run_term = try run_child.wait();
    switch (run_term) {
        .Exited => |code| {
            if (code != 0) {
                std.debug.print("Program exited with code {d}\n", .{code});
            }
        },
        else => std.debug.print("Program terminated abnormally\n", .{}),
    }
}
