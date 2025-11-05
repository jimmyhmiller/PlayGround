/// JIT Executor for MLIR
/// Manages lowering passes and JIT compilation of MLIR modules
const std = @import("std");
const mlir = @import("mlir/c.zig");

pub const ExecutorError = error{
    PassManagerCreationFailed,
    ExecutionEngineCreationFailed,
    LoweringFailed,
    InvokeFailed,
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
    /// Optional transform operations to apply before lowering
    transform_ops: []const mlir.MlirOperation,

    pub fn init(allocator: std.mem.Allocator, ctx: *mlir.Context, config: ExecutorConfig) Executor {
        return Executor{
            .allocator = allocator,
            .ctx = ctx,
            .config = config,
            .engine = null,
            .transform_ops = &.{},
        };
    }

    /// Set transform operations to apply during compilation
    pub fn setTransforms(self: *Executor, transforms: []const mlir.MlirOperation) void {
        self.transform_ops = transforms;
    }

    pub fn deinit(self: *Executor) void {
        if (self.engine) |*eng| {
            eng.destroy();
        }
    }

    /// Apply transform dialect operations to the module
    /// This takes transform operations (e.g., PDL patterns) and applies them to the module
    /// transformOps: Array of transform operations to apply
    pub fn applyTransforms(
        self: *Executor,
        module: *mlir.Module,
        transformOps: []const mlir.MlirOperation,
    ) !void {
        if (transformOps.len == 0) {
            // Nothing to do
            return;
        }

        std.debug.print("  Applying {} transform operation(s)...\n", .{transformOps.len});

        // Create a temporary module to hold the transform operations
        const location = mlir.Location.unknown(self.ctx);
        var transform_module = try mlir.Module.create(location);
        defer transform_module.destroy();

        // Clone transform operations into the transform module
        const transform_mod_body = transform_module.getBody();
        for (transformOps) |transform_op| {
            // Clone the operation
            const cloned = mlir.c.mlirOperationClone(transform_op);
            mlir.c.mlirBlockAppendOwnedOperation(transform_mod_body, cloned);
        }

        std.debug.print("  Transform module:\n", .{});
        transform_module.print();

        // Get the first transform operation as the root
        if (transformOps.len == 0) return;

        const first_transform = mlir.c.mlirBlockGetFirstOperation(transform_mod_body);
        if (mlir.c.mlirOperationIsNull(first_transform)) {
            return error.NoTransformRoot;
        }

        // Create transform options
        var transform_options = mlir.TransformOptions.create();
        defer transform_options.destroy();

        // Enable expensive checks if verifier is enabled
        transform_options.enableExpensiveChecks(self.config.enable_verifier);

        std.debug.print("  Module before transform:\n", .{});
        module.print();

        // Apply the transform
        const payload = module.getOperation();
        const transform_mod_op = transform_module.getOperation();

        try mlir.Transform.applyNamedSequence(
            payload,
            first_transform,
            transform_mod_op,
            &transform_options,
        );

        std.debug.print("  Module after transform:\n", .{});
        module.print();
        std.debug.print("  ✓ Transforms applied successfully\n", .{});
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
        // First, apply transforms if any were set
        if (self.transform_ops.len > 0) {
            try self.applyTransforms(module, self.transform_ops);
        }

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
