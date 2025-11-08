/// JIT Executor for MLIR
/// Manages lowering passes and JIT compilation of MLIR modules
const std = @import("std");
const mlir = @import("mlir/c.zig");

pub const ExecutorError = error{
    PassManagerCreationFailed,
    ExecutionEngineCreationFailed,
    LoweringFailed,
    InvokeFailed,
    MetadataModuleInJITPipeline,
} || std.mem.Allocator.Error;

pub const OptLevel = enum(u32) {
    O0 = 0,
    O1 = 1,
    O2 = 2,
    O3 = 3,
};

/// Configuration for the executor
pub const ExecutorConfig = struct {
    /// Optimization level (O0, O1, O2, O3)
    opt_level: OptLevel = .O0,

    /// Whether to enable the verifier after passes
    enable_verifier: bool = true,

    /// Shared library paths for external symbols
    shared_lib_paths: []const []const u8 = &.{},

    /// Whether to dump object files for debugging
    enable_object_dump: bool = false,
};

/// JIT executor for MLIR modules
pub const Executor = struct {
    allocator: std.mem.Allocator,
    ctx: *mlir.Context,
    config: ExecutorConfig,
    engine: ?mlir.ExecutionEngine,
    /// Optional transform operation to apply before lowering
    transform_op: ?mlir.MlirOperation,
    /// Optional transform module containing the transform
    transform_module: ?mlir.MlirOperation,

    pub fn init(allocator: std.mem.Allocator, ctx: *mlir.Context, config: ExecutorConfig) Executor {
        return Executor{
            .allocator = allocator,
            .ctx = ctx,
            .config = config,
            .engine = null,
            .transform_op = null,
            .transform_module = null,
        };
    }

    /// Set transform to apply during compilation
    pub fn setTransform(self: *Executor, transform_op: mlir.MlirOperation, transform_module: mlir.MlirOperation) void {
        self.transform_op = transform_op;
        self.transform_module = transform_module;
    }

    pub fn deinit(self: *Executor) void {
        if (self.engine) |*eng| {
            eng.destroy();
        }
    }

    /// Apply transform dialect operation to the module
    pub fn applyTransforms(
        self: *Executor,
        module: *mlir.Module,
    ) !void {
        if (self.transform_op == null or self.transform_module == null) {
            // Nothing to do
            return;
        }

        const transform_op = self.transform_op.?;
        const transform_module = self.transform_module.?;

        std.debug.print("  Applying transform operation...\n", .{});
        std.debug.print("    Transform operation: {s}\n", .{mlir.Operation.getName(transform_op)});
        std.debug.print("    Transform module: {s}\n", .{mlir.Operation.getName(transform_module)});

        // Create transform options
        var transform_options = mlir.TransformOptions.create();
        defer transform_options.destroy();

        // Enable expensive checks if verifier is enabled
        transform_options.enableExpensiveChecks(self.config.enable_verifier);

        std.debug.print("  Module before transform:\n", .{});
        module.print();

        // Apply the transform
        // Note: applyNamedSequence can actually handle different transform types,
        // not just transform.named_sequence. It works with transform.with_pdl_patterns,
        // transform.sequence, and other transform dialect operations.
        const payload = module.getOperation();

        std.debug.print("  Applying transform to payload...\n", .{});
        const result = mlir.Transform.applyNamedSequence(
            payload,
            transform_op,
            transform_module,  // The module containing the transform
            &transform_options,
        );

        if (result) {
            std.debug.print("  Module after transform:\n", .{});
            module.print();
            std.debug.print("  ✓ Transforms applied successfully\n", .{});
        } else |err| {
            std.debug.print("  ✗ Transform application failed: {}\n", .{err});
            return err;
        }
    }

    /// Validate that the module doesn't contain metadata modules
    /// Metadata modules should not be JIT compiled
    fn validateNoMetadata(module: *mlir.Module) !void {
        const module_op = module.getOperation();
        const body_region = mlir.c.mlirOperationGetRegion(module_op, 0);
        const body_block = mlir.c.mlirRegionGetFirstBlock(body_region);

        var current_op = mlir.c.mlirBlockGetFirstOperation(body_block);

        while (!mlir.c.mlirOperationIsNull(current_op)) {
            const op_name = mlir.Operation.getName(current_op);

            // Check if it's a builtin.module with metadata attribute
            if (std.mem.eql(u8, op_name, "builtin.module")) {
                const attr_name_str = "metadata";
                const attr_name_ref = mlir.c.mlirStringRefCreate(attr_name_str.ptr, attr_name_str.len);
                const attr = mlir.c.mlirOperationGetAttributeByName(current_op, attr_name_ref);

                if (!mlir.c.mlirAttributeIsNull(attr)) {
                    std.debug.print("ERROR: Found metadata module in JIT compilation pipeline!\n", .{});
                    std.debug.print("Metadata modules should not be JIT compiled.\n", .{});
                    return error.MetadataModuleInJITPipeline;
                }
            }

            current_op = mlir.c.mlirOperationGetNextInBlock(current_op);
        }
    }

    /// Apply lowering passes to convert high-level MLIR to LLVM IR
    /// This uses a standard lowering pipeline:
    /// 1. Convert func/arith/scf/cf to lower level dialects
    /// 2. Convert everything to LLVM dialect
    /// 3. Optimize if requested
    pub fn lowerToLLVM(self: *Executor, module: *mlir.Module) !void {
        var pm = try mlir.PassManager.create(self.ctx);
        defer pm.destroy();

        pm.enableVerifier(self.config.enable_verifier);

        // Build the lowering pipeline - note func passes must be nested in func.func()
        // Must be null-terminated for C API
        // Added convert-scf-to-cf to support structured control flow (scf.if, scf.while, etc.)
        // Added convert-cf-to-llvm to convert control flow to LLVM
        // Added finalize-memref-to-llvm to convert memref operations to LLVM
        const pipeline: [:0]const u8 = "builtin.module(func.func(convert-scf-to-cf,convert-arith-to-llvm),convert-cf-to-llvm,finalize-memref-to-llvm,convert-func-to-llvm,reconcile-unrealized-casts)";

        try pm.addPipeline(pipeline);

        // Run the passes
        try pm.run(module);
    }

    /// Compile the MLIR module and create an execution engine
    pub fn compile(self: *Executor, module: *mlir.Module) !void {
        // Validate that no metadata modules are present
        try validateNoMetadata(module);

        // First, apply transforms if any were set
        try self.applyTransforms(module);

        // Then, lower to LLVM IR
        try self.lowerToLLVM(module);

        // Create execution engine config
        const engine_config = mlir.ExecutionEngine.Config{
            .opt_level = @intFromEnum(self.config.opt_level),
            .shared_lib_paths = self.config.shared_lib_paths,
            .enable_object_dump = self.config.enable_object_dump,
        };

        // Create the execution engine
        self.engine = try mlir.ExecutionEngine.create(module, engine_config);
    }

    /// Invoke a function by name with packed arguments
    /// The function must have been compiled with llvm.emit_c_interface attribute
    pub fn invokePacked(self: *Executor, name: []const u8, args: ?*anyopaque) !void {
        const eng = self.engine orelse return error.ExecutionEngineNotCreated;
        try eng.invokePacked(name, args);
    }

    /// Look up a function pointer by name
    pub fn lookup(self: *const Executor, name: []const u8) ?*anyopaque {
        const eng = self.engine orelse return null;
        return eng.lookup(name);
    }

    /// Register an external symbol that can be called from JIT'd code
    /// Useful for registering C library functions like printf, malloc, etc.
    pub fn registerSymbol(self: *Executor, name: []const u8, sym: *anyopaque) void {
        if (self.engine) |*eng| {
            eng.registerSymbol(name, sym);
        }
    }

    /// Dump the compiled module to an object file
    pub fn dumpToObjectFile(self: *Executor, filename: []const u8) void {
        if (self.engine) |eng| {
            eng.dumpToObjectFile(filename);
        }
    }
};

/// Helper to register common C library functions
pub fn registerCommonCLibraryFunctions(executor: *Executor) void {
    // Register printf, malloc, free directly from std.c
    // These are already C functions, so we can register them directly
    executor.registerSymbol("printf", @ptrCast(&std.c.printf));
    executor.registerSymbol("malloc", @ptrCast(&std.c.malloc));
    executor.registerSymbol("free", @ptrCast(&std.c.free));
}
