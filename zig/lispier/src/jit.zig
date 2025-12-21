const std = @import("std");
const mlir_integration = @import("mlir_integration.zig");
const c = mlir_integration.c;

pub const JITError = error{
    PassPipelineFailed,
    PassManagerRunFailed,
    ExecutionEngineCreationFailed,
    InvocationFailed,
    FunctionNotFound,
    OutOfMemory,
};

/// JIT compiler for MLIR modules
pub const JIT = struct {
    allocator: std.mem.Allocator,
    ctx: c.MlirContext,
    engine: c.MlirExecutionEngine,

    /// Initialize JIT by lowering the module to LLVM and creating an execution engine
    pub fn init(allocator: std.mem.Allocator, ctx: c.MlirContext, module: c.MlirModule) !JIT {
        // Register all passes and LLVM translations (idempotent)
        c.mlirRegisterAllPasses();
        c.mlirRegisterAllLLVMTranslations(ctx);

        // Lower the module to LLVM dialect
        try lowerToLLVM(ctx, module);

        // Create execution engine
        const engine = c.mlirExecutionEngineCreate(
            module,
            2, // optLevel
            0, // numPaths
            null, // sharedLibPaths
            false, // enableObjectDump
        );

        if (c.mlirExecutionEngineIsNull(engine)) {
            return error.ExecutionEngineCreationFailed;
        }

        return .{
            .allocator = allocator,
            .ctx = ctx,
            .engine = engine,
        };
    }

    pub fn deinit(self: *JIT) void {
        c.mlirExecutionEngineDestroy(self.engine);
    }

    /// Invoke a function by name with packed arguments
    /// The function must have the llvm.emit_c_interface attribute
    pub fn invoke(self: *JIT, name: []const u8, args: []*anyopaque) !void {
        const name_z = try std.fmt.allocPrint(self.allocator, "{s}\x00", .{name});
        defer self.allocator.free(name_z);

        const name_ref = c.MlirStringRef{
            .data = name_z.ptr,
            .length = name_z.len - 1, // Don't include null terminator in length
        };

        const result = c.mlirExecutionEngineInvokePacked(
            self.engine,
            name_ref,
            @ptrCast(args.ptr),
        );

        if (result.value == 0) {
            return error.InvocationFailed;
        }
    }

    /// Lookup a function pointer by name
    pub fn lookup(self: *JIT, name: []const u8) !*anyopaque {
        const name_z = try std.fmt.allocPrint(self.allocator, "{s}\x00", .{name});
        defer self.allocator.free(name_z);

        const name_ref = c.MlirStringRef{
            .data = name_z.ptr,
            .length = name_z.len - 1,
        };

        const ptr = c.mlirExecutionEngineLookup(self.engine, name_ref);
        if (ptr == null) {
            return error.FunctionNotFound;
        }

        return ptr.?;
    }

    /// Register an external symbol with the JIT
    pub fn registerSymbol(self: *JIT, name: []const u8, sym: *anyopaque) !void {
        const name_z = try std.fmt.allocPrint(self.allocator, "{s}\x00", .{name});
        defer self.allocator.free(name_z);

        const name_ref = c.MlirStringRef{
            .data = name_z.ptr,
            .length = name_z.len - 1,
        };

        c.mlirExecutionEngineRegisterSymbol(self.engine, name_ref, sym);
    }
};

/// Lower a module from high-level dialects to LLVM dialect
fn lowerToLLVM(ctx: c.MlirContext, module: c.MlirModule) !void {
    // Create pass manager
    const pm = c.mlirPassManagerCreate(ctx);
    defer c.mlirPassManagerDestroy(pm);

    // Enable verification after each pass
    c.mlirPassManagerEnableVerifier(pm, true);

    // Get the OpPassManager for the module
    const opm = c.mlirPassManagerGetAsOpPassManager(pm);

    // Add the lowering pipeline
    // This pipeline converts high-level dialects to LLVM dialect
    // SCF -> CF first, then everything to LLVM
    const pipeline =
        "convert-scf-to-cf," ++
        "convert-arith-to-llvm," ++
        "convert-index-to-llvm," ++
        "convert-func-to-llvm," ++
        "convert-cf-to-llvm," ++
        "finalize-memref-to-llvm," ++
        "reconcile-unrealized-casts";

    const pipeline_ref = c.MlirStringRef{
        .data = pipeline.ptr,
        .length = pipeline.len,
    };

    // Error callback for pipeline parsing
    const ErrorContext = struct {
        failed: bool = false,

        fn callback(str_ref: c.MlirStringRef, user_data: ?*anyopaque) callconv(.c) void {
            const ctx_ptr: *@This() = @ptrCast(@alignCast(user_data.?));
            ctx_ptr.failed = true;
            // Print error message
            if (str_ref.length > 0) {
                std.debug.print("Pipeline error: {s}\n", .{str_ref.data[0..str_ref.length]});
            }
        }
    };

    var error_ctx = ErrorContext{};
    const parse_result = c.mlirOpPassManagerAddPipeline(
        opm,
        pipeline_ref,
        ErrorContext.callback,
        &error_ctx,
    );

    if (parse_result.value == 0 or error_ctx.failed) {
        return error.PassPipelineFailed;
    }

    // Run the pass manager on the module operation
    const module_op = c.mlirModuleGetOperation(module);
    const run_result = c.mlirPassManagerRunOnOp(pm, module_op);

    if (run_result.value == 0) {
        return error.PassManagerRunFailed;
    }
}

/// Print the lowered LLVM IR for debugging
pub fn printLoweredModule(module: c.MlirModule) void {
    const op = c.mlirModuleGetOperation(module);
    c.mlirOperationPrint(op, printCallback, null);
    std.debug.print("\n", .{});
}

fn printCallback(str_ref: c.MlirStringRef, _: ?*anyopaque) callconv(.c) void {
    if (str_ref.length > 0) {
        std.debug.print("{s}", .{str_ref.data[0..str_ref.length]});
    }
}
