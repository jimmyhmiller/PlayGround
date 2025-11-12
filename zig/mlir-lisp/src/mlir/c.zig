/// MLIR C API bindings
/// This file provides Zig wrappers around the MLIR C API
const std = @import("std");

// Import MLIR C headers
pub const c = @cImport({
    @cInclude("mlir-c/IR.h");
    @cInclude("mlir-c/BuiltinAttributes.h");
    @cInclude("mlir-c/BuiltinTypes.h");
    @cInclude("mlir-c/Diagnostics.h");
    @cInclude("mlir-c/Support.h");
    @cInclude("mlir-c/RegisterEverything.h");
    @cInclude("mlir-c/Pass.h");
    @cInclude("mlir-c/Transforms.h");
    @cInclude("mlir-c/Conversion.h");
    @cInclude("mlir-c/ExecutionEngine.h");
    @cInclude("mlir-c/Interfaces.h");
    @cInclude("mlir-c/Dialect/IRDL.h");
    @cInclude("mlir-c/Dialect/Transform.h");
    @cInclude("mlir-c/Dialect/Transform/Interpreter.h");
});

// Re-export common types for convenience
pub const MlirContext = c.MlirContext;
pub const MlirModule = c.MlirModule;
pub const MlirOperation = c.MlirOperation;
pub const MlirType = c.MlirType;
pub const MlirAttribute = c.MlirAttribute;
pub const MlirLocation = c.MlirLocation;
pub const MlirValue = c.MlirValue;
pub const MlirBlock = c.MlirBlock;
pub const MlirRegion = c.MlirRegion;

/// Reimplementation of mlirLogicalResultIsFailure since it's an inline function
/// that cannot be called through FFI
inline fn mlirLogicalResultIsFailure(result: c.MlirLogicalResult) bool {
    return result.value == 0;
}

/// Context wrapper for MLIR
pub const Context = struct {
    ctx: MlirContext,

    pub fn create() !Context {
        const ctx = c.mlirContextCreate();
        if (c.mlirContextIsNull(ctx)) {
            return error.ContextCreationFailed;
        }
        return Context{ .ctx = ctx };
    }

    pub fn destroy(self: *Context) void {
        c.mlirContextDestroy(self.ctx);
    }

    pub fn getOrLoadDialect(self: *Context, name: []const u8) !void {
        const name_ref = c.mlirStringRefCreate(name.ptr, name.len);
        const dialect = c.mlirContextGetOrLoadDialect(self.ctx, name_ref);
        if (c.mlirDialectIsNull(dialect)) {
            return error.DialectLoadFailed;
        }
    }

    pub fn setAllowUnregisteredDialects(self: *Context, allow: bool) void {
        c.mlirContextSetAllowUnregisteredDialects(self.ctx, allow);
    }

    pub fn registerAllDialects(self: *Context) void {
        const registry = c.mlirDialectRegistryCreate();
        defer c.mlirDialectRegistryDestroy(registry);
        c.mlirRegisterAllDialects(registry);
        c.mlirContextAppendDialectRegistry(self.ctx, registry);
    }

    pub fn registerAllLLVMTranslations(self: *Context) void {
        c.mlirRegisterAllLLVMTranslations(self.ctx);
    }

    pub fn registerAllPasses() void {
        c.mlirRegisterAllPasses();
    }
};

/// Module wrapper
pub const Module = struct {
    module: MlirModule,

    pub fn create(location: c.MlirLocation) !Module {
        const mod = c.mlirModuleCreateEmpty(location);
        if (c.mlirModuleIsNull(mod)) {
            return error.ModuleCreationFailed;
        }
        return Module{ .module = mod };
    }

    pub fn fromOperation(op: MlirOperation) !Module {
        const mod = c.mlirModuleFromOperation(op);
        if (c.mlirModuleIsNull(mod)) {
            return error.InvalidModuleOperation;
        }
        return Module{ .module = mod };
    }

    pub fn destroy(self: *Module) void {
        c.mlirModuleDestroy(self.module);
    }

    pub fn getOperation(self: *Module) MlirOperation {
        return c.mlirModuleGetOperation(self.module);
    }

    pub fn getBody(self: *Module) MlirBlock {
        return c.mlirModuleGetBody(self.module);
    }

    /// Verify that the module is well-formed
    pub fn verify(self: *Module) bool {
        const op = self.getOperation();
        return c.mlirOperationVerify(op);
    }

    pub fn print(self: *Module) void {
        const op = self.getOperation();
        c.mlirOperationPrint(op, &printCallback, null);
    }

    pub fn printGeneric(self: *Module) void {
        const op = self.getOperation();
        const flags = c.mlirOpPrintingFlagsCreate();
        defer c.mlirOpPrintingFlagsDestroy(flags);
        c.mlirOpPrintingFlagsPrintGenericOpForm(flags);
        c.mlirOperationPrintWithFlags(op, flags, &printCallback, null);
    }

    fn printCallback(str: c.MlirStringRef, userData: ?*anyopaque) callconv(.c) void {
        _ = userData;
        const slice = str.data[0..str.length];
        std.debug.print("{s}", .{slice});
    }

    /// Load IRDL dialects defined in this module into the context
    pub fn loadIRDLDialects(self: *Module) !void {
        const result = c.mlirLoadIRDLDialects(self.module);
        if (mlirLogicalResultIsFailure(result)) {
            return error.IRDLLoadFailed;
        }
    }

    /// Walk all operations in module and collect those matching predicate
    /// Caller owns returned slice and must free it
    pub fn collectOperations(
        self: *Module,
        allocator: std.mem.Allocator,
        predicate: *const fn(MlirOperation) bool,
    ) ![]MlirOperation {
        var ops = std.ArrayList(MlirOperation){};
        defer ops.deinit(allocator);

        const mod_op = self.getOperation();
        try walkOperations(mod_op, &ops, allocator, predicate);

        return ops.toOwnedSlice(allocator);
    }

    /// Find all operations with a specific name prefix (e.g., "irdl.", "transform.")
    pub fn collectOperationsByPrefix(
        self: *Module,
        allocator: std.mem.Allocator,
        prefix: []const u8,
    ) ![]MlirOperation {
        const Ctx = struct {
            prefix: []const u8,
            fn matches(op: MlirOperation, ctx_prefix: []const u8) bool {
                const name = Operation.getName(op);
                return std.mem.startsWith(u8, name, ctx_prefix);
            }
        };

        var ops = std.ArrayList(MlirOperation){};
        defer ops.deinit(allocator);

        const mod_op = self.getOperation();
        try walkOperationsWithContext(mod_op, &ops, allocator, prefix, Ctx.matches);

        return ops.toOwnedSlice(allocator);
    }

    /// Find all operations with a specific exact name
    pub fn collectOperationsByName(
        self: *Module,
        allocator: std.mem.Allocator,
        name: []const u8,
    ) ![]MlirOperation {
        const Ctx = struct {
            name: []const u8,
            fn matches(op: MlirOperation, ctx_name: []const u8) bool {
                const op_name = Operation.getName(op);
                return std.mem.eql(u8, op_name, ctx_name);
            }
        };

        var ops = std.ArrayList(MlirOperation){};
        defer ops.deinit(allocator);

        const mod_op = self.getOperation();
        try walkOperationsWithContext(mod_op, &ops, allocator, name, Ctx.matches);

        return ops.toOwnedSlice(allocator);
    }

    /// Helper: recursively walk all operations
    fn walkOperations(
        op: MlirOperation,
        list: *std.ArrayList(MlirOperation),
        allocator: std.mem.Allocator,
        predicate: *const fn(MlirOperation) bool,
    ) !void {
        if (predicate(op)) {
            try list.append(allocator, op);
        }

        const num_regions = Operation.getNumRegions(op);
        var i: usize = 0;
        while (i < num_regions) : (i += 1) {
            const region = Operation.getRegion(op, i);
            if (Region.isNull(region)) continue;

            const first_block = c.mlirRegionGetFirstBlock(region);
            if (Block.isNull(first_block)) continue;

            var block = first_block;
            while (!Block.isNull(block)) {
                var child_op = c.mlirBlockGetFirstOperation(block);
                while (!c.mlirOperationIsNull(child_op)) {
                    try walkOperations(child_op, list, allocator, predicate);
                    child_op = c.mlirOperationGetNextInBlock(child_op);
                }
                block = c.mlirBlockGetNextInRegion(block);
            }
        }
    }

    /// Helper: recursively walk with context
    fn walkOperationsWithContext(
        op: MlirOperation,
        list: *std.ArrayList(MlirOperation),
        allocator: std.mem.Allocator,
        context: []const u8,
        predicate: *const fn(MlirOperation, []const u8) bool,
    ) !void {
        if (predicate(op, context)) {
            try list.append(allocator, op);
        }

        const num_regions = Operation.getNumRegions(op);
        var i: usize = 0;
        while (i < num_regions) : (i += 1) {
            const region = Operation.getRegion(op, i);
            if (Region.isNull(region)) continue;

            const first_block = c.mlirRegionGetFirstBlock(region);
            if (Block.isNull(first_block)) continue;

            var block = first_block;
            while (!Block.isNull(block)) {
                var child_op = c.mlirBlockGetFirstOperation(block);
                while (!c.mlirOperationIsNull(child_op)) {
                    try walkOperationsWithContext(child_op, list, allocator, context, predicate);
                    child_op = c.mlirOperationGetNextInBlock(child_op);
                }
                block = c.mlirBlockGetNextInRegion(block);
            }
        }
    }
};

/// Location wrapper
pub const Location = struct {
    pub fn unknown(ctx: *Context) c.MlirLocation {
        return c.mlirLocationUnknownGet(ctx.ctx);
    }

    pub fn fileLineCol(ctx: *Context, filename: []const u8, line: u32, col: u32) c.MlirLocation {
        const name_ref = c.mlirStringRefCreate(filename.ptr, filename.len);
        return c.mlirLocationFileLineColGet(ctx.ctx, name_ref, line, col);
    }
};

/// Type utilities
pub const Type = struct {
    pub fn parse(ctx: *Context, type_str: []const u8) !MlirType {
        const type_ref = c.mlirStringRefCreate(type_str.ptr, type_str.len);
        const ty = c.mlirTypeParseGet(ctx.ctx, type_ref);
        if (c.mlirTypeIsNull(ty)) {
            return error.TypeParseFailed;
        }
        return ty;
    }

    pub fn @"i32"(ctx: *Context) MlirType {
        return c.mlirIntegerTypeGet(ctx.ctx, 32);
    }

    pub fn @"i64"(ctx: *Context) MlirType {
        return c.mlirIntegerTypeGet(ctx.ctx, 64);
    }

    pub fn @"f32"(ctx: *Context) MlirType {
        return c.mlirF32TypeGet(ctx.ctx);
    }

    pub fn @"f64"(ctx: *Context) MlirType {
        return c.mlirF64TypeGet(ctx.ctx);
    }

    // Type checking utilities
    pub fn isInteger(ty: MlirType) bool {
        return c.mlirTypeIsAInteger(ty);
    }

    pub fn isF32(ty: MlirType) bool {
        return c.mlirTypeIsAF32(ty);
    }

    pub fn isF64(ty: MlirType) bool {
        return c.mlirTypeIsAF64(ty);
    }

    pub fn getIntegerWidth(ty: MlirType) u32 {
        return c.mlirIntegerTypeGetWidth(ty);
    }

    // Function type introspection
    pub fn isFunctionType(ty: MlirType) bool {
        // Check if the type is a function type by trying to get its type ID
        const func_type_id = c.mlirFunctionTypeGetTypeID();
        const ty_id = c.mlirTypeGetTypeID(ty);
        return c.mlirTypeIDEqual(func_type_id, ty_id);
    }

    pub fn getFunctionNumInputs(ty: MlirType) usize {
        return @intCast(c.mlirFunctionTypeGetNumInputs(ty));
    }

    pub fn getFunctionNumResults(ty: MlirType) usize {
        return @intCast(c.mlirFunctionTypeGetNumResults(ty));
    }

    pub fn getFunctionInput(ty: MlirType, index: usize) MlirType {
        return c.mlirFunctionTypeGetInput(ty, @intCast(index));
    }

    pub fn getFunctionResult(ty: MlirType, index: usize) MlirType {
        return c.mlirFunctionTypeGetResult(ty, @intCast(index));
    }
};

/// Attribute utilities
pub const Attribute = struct {
    pub fn parse(ctx: *Context, attr_str: []const u8) !MlirAttribute {
        const attr_ref = c.mlirStringRefCreate(attr_str.ptr, attr_str.len);
        const attr = c.mlirAttributeParseGet(ctx.ctx, attr_ref);
        if (c.mlirAttributeIsNull(attr)) {
            return error.AttributeParseFailed;
        }
        return attr;
    }

    pub fn integer(ty: MlirType, value: i64) MlirAttribute {
        return c.mlirIntegerAttrGet(ty, value);
    }

    pub fn float(ty: MlirType, value: f64) MlirAttribute {
        return c.mlirFloatAttrDoubleGet(ty.context, ty, value);
    }

    pub fn stringGet(ctx: *Context, str: []const u8) MlirAttribute {
        const str_ref = c.mlirStringRefCreate(str.ptr, str.len);
        return c.mlirStringAttrGet(ctx.ctx, str_ref);
    }
};

/// Operation building utilities
pub const Operation = struct {
    pub fn create(
        name: []const u8,
        location: MlirLocation,
        results: []const MlirType,
        operands: []const MlirValue,
        attributes: []const c.MlirNamedAttribute,
        successors: []const MlirBlock,
        regions: []const MlirRegion,
    ) MlirOperation {
        const name_ref = c.mlirStringRefCreate(name.ptr, name.len);
        var state = c.mlirOperationStateGet(name_ref, location);

        c.mlirOperationStateAddResults(&state, @intCast(results.len), results.ptr);
        c.mlirOperationStateAddOperands(&state, @intCast(operands.len), operands.ptr);
        c.mlirOperationStateAddAttributes(&state, @intCast(attributes.len), attributes.ptr);
        c.mlirOperationStateAddSuccessors(&state, @intCast(successors.len), successors.ptr);
        c.mlirOperationStateAddOwnedRegions(&state, @intCast(regions.len), regions.ptr);

        return c.mlirOperationCreate(&state);
    }

    pub fn getResult(op: MlirOperation, index: usize) MlirValue {
        return c.mlirOperationGetResult(op, @intCast(index));
    }

    // Operation introspection
    pub fn getName(op: MlirOperation) []const u8 {
        const ident = c.mlirOperationGetName(op);
        const str_ref = c.mlirIdentifierStr(ident);
        return str_ref.data[0..str_ref.length];
    }

    pub fn getAttributeByName(op: MlirOperation, name: []const u8) ?MlirAttribute {
        const name_ref = c.mlirStringRefCreate(name.ptr, name.len);
        const attr = c.mlirOperationGetAttributeByName(op, name_ref);
        if (c.mlirAttributeIsNull(attr)) {
            return null;
        }
        return attr;
    }

    pub fn getNumRegions(op: MlirOperation) usize {
        return @intCast(c.mlirOperationGetNumRegions(op));
    }

    pub fn getRegion(op: MlirOperation, index: usize) MlirRegion {
        return c.mlirOperationGetRegion(op, @intCast(index));
    }

    pub fn getFirstRegion(op: MlirOperation) MlirRegion {
        return c.mlirOperationGetFirstRegion(op);
    }

    /// Verify that an operation is well-formed
    pub fn verify(op: MlirOperation) bool {
        return c.mlirOperationVerify(op);
    }
};

/// Region utilities
pub const Region = struct {
    pub fn create() MlirRegion {
        return c.mlirRegionCreate();
    }

    pub fn appendBlock(region: MlirRegion, block: MlirBlock) void {
        c.mlirRegionAppendOwnedBlock(region, block);
    }

    pub fn getFirstBlock(region: MlirRegion) MlirBlock {
        return c.mlirRegionGetFirstBlock(region);
    }

    pub fn isNull(region: MlirRegion) bool {
        return c.mlirRegionIsNull(region);
    }
};

/// Block utilities
pub const Block = struct {
    pub fn create(arg_types: []const MlirType, arg_locs: []const MlirLocation) MlirBlock {
        return c.mlirBlockCreate(@intCast(arg_types.len), arg_types.ptr, arg_locs.ptr);
    }

    pub fn appendOperation(block: MlirBlock, operation: MlirOperation) void {
        c.mlirBlockAppendOwnedOperation(block, operation);
    }

    pub fn getArgument(block: MlirBlock, index: usize) MlirValue {
        return c.mlirBlockGetArgument(block, @intCast(index));
    }

    pub fn getFirstOperation(block: MlirBlock) MlirOperation {
        return c.mlirBlockGetFirstOperation(block);
    }

    pub fn getNextInBlock(op: MlirOperation) MlirOperation {
        return c.mlirOperationGetNextInBlock(op);
    }

    pub fn isNull(block: MlirBlock) bool {
        return c.mlirBlockIsNull(block);
    }
};

/// Named attribute helper
pub fn namedAttribute(ctx: *Context, name: []const u8, attr: MlirAttribute) c.MlirNamedAttribute {
    const name_ref = c.mlirStringRefCreate(name.ptr, name.len);
    const identifier = c.mlirIdentifierGet(ctx.ctx, name_ref);
    return c.mlirNamedAttributeGet(identifier, attr);
}

test "mlir - context creation" {
    var ctx = try Context.create();
    defer ctx.destroy();

    // Context should be valid
    try std.testing.expect(!c.mlirContextIsNull(ctx.ctx));
}

test "mlir - load dialect" {
    var ctx = try Context.create();
    defer ctx.destroy();

    // Try to load a builtin dialect (should always work)
    try ctx.getOrLoadDialect("builtin");
}

test "mlir - module creation" {
    var ctx = try Context.create();
    defer ctx.destroy();

    const loc = Location.unknown(&ctx);
    var mod = try Module.create(loc);
    defer mod.destroy();

    // Module should be valid
    try std.testing.expect(!c.mlirModuleIsNull(mod.module));
}

test "mlir - type creation" {
    var ctx = try Context.create();
    defer ctx.destroy();

    const i32_type = Type.@"i32"(&ctx);
    try std.testing.expect(!c.mlirTypeIsNull(i32_type));

    const f64_type = Type.@"f64"(&ctx);
    try std.testing.expect(!c.mlirTypeIsNull(f64_type));
}

/// PassManager wrapper
pub const PassManager = struct {
    pm: c.MlirPassManager,

    pub fn create(ctx: *Context) !PassManager {
        const pm = c.mlirPassManagerCreate(ctx.ctx);
        if (c.mlirPassManagerIsNull(pm)) {
            return error.PassManagerCreationFailed;
        }
        return PassManager{ .pm = pm };
    }

    pub fn createOnOperation(ctx: *Context, anchor_op: []const u8) !PassManager {
        const anchor_ref = c.mlirStringRefCreate(anchor_op.ptr, anchor_op.len);
        const pm = c.mlirPassManagerCreateOnOperation(ctx.ctx, anchor_ref);
        if (c.mlirPassManagerIsNull(pm)) {
            return error.PassManagerCreationFailed;
        }
        return PassManager{ .pm = pm };
    }

    pub fn destroy(self: *PassManager) void {
        c.mlirPassManagerDestroy(self.pm);
    }

    pub fn run(self: *PassManager, module: *Module) !void {
        const op = module.getOperation();
        const result = c.mlirPassManagerRunOnOp(self.pm, op);
        if (mlirLogicalResultIsFailure(result)) {
            return error.PassManagerRunFailed;
        }
    }

    pub fn enableVerifier(self: *PassManager, enable: bool) void {
        c.mlirPassManagerEnableVerifier(self.pm, enable);
    }

    pub fn addPipeline(self: *PassManager, pipeline: [:0]const u8) !void {
        const opm = c.mlirPassManagerGetAsOpPassManager(self.pm);
        const pipeline_ref = c.mlirStringRefCreateFromCString(pipeline.ptr);
        const result = c.mlirParsePassPipeline(opm, pipeline_ref, &errorCallback, null);

        if (mlirLogicalResultIsFailure(result)) {
            return error.PassPipelineAddFailed;
        }
    }

    fn errorCallback(str: c.MlirStringRef, userData: ?*anyopaque) callconv(.c) void {
        _ = userData;
        const slice = str.data[0..str.length];
        std.debug.print("Pass pipeline error: {s}\n", .{slice});
    }
};

/// ExecutionEngine wrapper
pub const ExecutionEngine = struct {
    engine: c.MlirExecutionEngine,

    pub const Config = struct {
        opt_level: u32 = 0,
        shared_lib_paths: []const []const u8 = &.{},
        enable_object_dump: bool = false,
    };

    pub fn create(module: *Module, config: Config) !ExecutionEngine {
        // Convert shared lib paths to MlirStringRef array
        var arena = std.heap.ArenaAllocator.init(std.heap.page_allocator);
        defer arena.deinit();
        const allocator = arena.allocator();

        var lib_refs = std.ArrayList(c.MlirStringRef).initCapacity(allocator, config.shared_lib_paths.len) catch std.ArrayList(c.MlirStringRef){};
        defer lib_refs.deinit(allocator);

        for (config.shared_lib_paths) |path| {
            const ref = c.mlirStringRefCreate(path.ptr, path.len);
            try lib_refs.append(allocator, ref);
        }

        const engine = c.mlirExecutionEngineCreate(
            module.module,
            @intCast(config.opt_level),
            @intCast(lib_refs.items.len),
            if (lib_refs.items.len > 0) lib_refs.items.ptr else null,
            config.enable_object_dump,
        );

        if (c.mlirExecutionEngineIsNull(engine)) {
            return error.ExecutionEngineCreationFailed;
        }

        return ExecutionEngine{ .engine = engine };
    }

    pub fn destroy(self: *ExecutionEngine) void {
        c.mlirExecutionEngineDestroy(self.engine);
    }

    pub fn invokePacked(self: *ExecutionEngine, name: []const u8, args: ?*anyopaque) !void {
        const name_ref = c.mlirStringRefCreate(name.ptr, name.len);
        const result = c.mlirExecutionEngineInvokePacked(self.engine, name_ref, @ptrCast(args));
        if (mlirLogicalResultIsFailure(result)) {
            return error.InvokeFailed;
        }
    }

    pub fn lookup(self: *const ExecutionEngine, name: []const u8) ?*anyopaque {
        const name_ref = c.mlirStringRefCreate(name.ptr, name.len);
        return c.mlirExecutionEngineLookup(self.engine, name_ref);
    }

    pub fn registerSymbol(self: *ExecutionEngine, name: []const u8, sym: *anyopaque) void {
        const name_ref = c.mlirStringRefCreate(name.ptr, name.len);
        c.mlirExecutionEngineRegisterSymbol(self.engine, name_ref, sym);
    }

    pub fn dumpToObjectFile(self: *ExecutionEngine, filename: []const u8) void {
        const name_ref = c.mlirStringRefCreate(filename.ptr, filename.len);
        c.mlirExecutionEngineDumpToObjectFile(self.engine, name_ref);
    }
};

/// TransformOptions wrapper for transform dialect interpreter
pub const TransformOptions = struct {
    options: c.MlirTransformOptions,

    pub fn create() TransformOptions {
        return TransformOptions{
            .options = c.mlirTransformOptionsCreate()
        };
    }

    pub fn destroy(self: *TransformOptions) void {
        c.mlirTransformOptionsDestroy(self.options);
    }

    pub fn enableExpensiveChecks(self: *TransformOptions, enable: bool) void {
        c.mlirTransformOptionsEnableExpensiveChecks(self.options, enable);
    }

    pub fn enforceSingleTopLevelTransformOp(self: *TransformOptions, enable: bool) void {
        c.mlirTransformOptionsEnforceSingleTopLevelTransformOp(self.options, enable);
    }
};

/// Transform API wrapper
pub const Transform = struct {
    /// Apply a transform sequence to a payload operation
    /// payload: The operation to transform
    /// transformRoot: The root transform operation (e.g., transform.sequence)
    /// transformModule: The module containing the transform
    /// options: Transform options
    pub fn applyNamedSequence(
        payload: MlirOperation,
        transformRoot: MlirOperation,
        transformModule: MlirOperation,
        options: *TransformOptions,
    ) !void {
        const result = c.mlirTransformApplyNamedSequence(
            payload,
            transformRoot,
            transformModule,
            options.options,
        );
        if (mlirLogicalResultIsFailure(result)) {
            return error.TransformApplyFailed;
        }
    }

    /// Merge symbols from one module into another
    pub fn mergeSymbolsIntoFromClone(
        target: MlirOperation,
        other: MlirOperation,
    ) !void {
        const result = c.mlirMergeSymbolsIntoFromClone(target, other);
        if (mlirLogicalResultIsFailure(result)) {
            return error.SymbolMergeFailed;
        }
    }
};
