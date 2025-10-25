/// Dynamic function signature extraction and calling utilities for JIT'd MLIR functions
const std = @import("std");
const mlir = @import("mlir/c.zig");

pub const FunctionSignatureError = error{
    FunctionNotFound,
    UnsupportedType,
    InvalidSignature,
    ArgumentCountMismatch,
};

/// Represents a simplified runtime type for function parameters
pub const RuntimeType = enum {
    i1,
    i8,
    i16,
    i32,
    i64,
    f32,
    f64,

    pub fn fromMlirType(ty: mlir.MlirType) !RuntimeType {
        if (mlir.Type.isInteger(ty)) {
            const width = mlir.Type.getIntegerWidth(ty);
            return switch (width) {
                1 => .i1,
                8 => .i8,
                16 => .i16,
                32 => .i32,
                64 => .i64,
                else => error.UnsupportedType,
            };
        } else if (mlir.Type.isF32(ty)) {
            return .f32;
        } else if (mlir.Type.isF64(ty)) {
            return .f64;
        }
        return error.UnsupportedType;
    }
};

/// Represents a function signature extracted from MLIR
pub const FunctionSignature = struct {
    inputs: []RuntimeType,
    results: []RuntimeType,
    allocator: std.mem.Allocator,

    pub fn deinit(self: *FunctionSignature) void {
        self.allocator.free(self.inputs);
        self.allocator.free(self.results);
    }
};

/// Extract function signature from an MLIR module
pub fn extractFunctionSignature(
    allocator: std.mem.Allocator,
    module: *mlir.Module,
    fn_name: []const u8,
) !FunctionSignature {
    // Get the module body
    const body = module.getBody();
    if (mlir.Block.isNull(body)) {
        return error.FunctionNotFound;
    }

    // Iterate through operations in the module body to find our function
    var op = mlir.Block.getFirstOperation(body);
    while (!mlir.c.mlirOperationIsNull(op)) {
        const op_name = mlir.Operation.getName(op);

        // Check if this is a func.func operation
        if (std.mem.eql(u8, op_name, "func.func")) {
            // Get the symbol name attribute
            if (mlir.Operation.getAttributeByName(op, "sym_name")) |sym_attr| {
                // Get string value from symbol attribute
                // For FlatSymbolRef attributes, use mlirFlatSymbolRefAttrGetValue
                // For String attributes, use mlirStringAttrGetValue
                const sym_ref = if (mlir.c.mlirAttributeIsAFlatSymbolRef(sym_attr))
                    mlir.c.mlirFlatSymbolRefAttrGetValue(sym_attr)
                else
                    mlir.c.mlirStringAttrGetValue(sym_attr);

                const sym_name = sym_ref.data[0..sym_ref.length];

                if (std.mem.eql(u8, sym_name, fn_name)) {
                    // Found our function! Extract its type
                    return try extractFunctionTypeFromOp(allocator, op);
                }
            }
        }

        op = mlir.Block.getNextInBlock(op);
    }

    return error.FunctionNotFound;
}

fn extractFunctionTypeFromOp(allocator: std.mem.Allocator, op: mlir.MlirOperation) !FunctionSignature {
    // Get the function_type or type attribute
    const type_attr = mlir.Operation.getAttributeByName(op, "function_type") orelse
        mlir.Operation.getAttributeByName(op, "type") orelse
        return error.InvalidSignature;

    // The attribute itself IS a TypeAttr, so we need to get the type from it
    const func_type = mlir.c.mlirTypeAttrGetValue(type_attr);
    if (!mlir.Type.isFunctionType(func_type)) {
        return error.InvalidSignature;
    }

    // Extract input and result types
    const num_inputs = mlir.Type.getFunctionNumInputs(func_type);
    const num_results = mlir.Type.getFunctionNumResults(func_type);

    var inputs = try allocator.alloc(RuntimeType, num_inputs);
    errdefer allocator.free(inputs);

    var results = try allocator.alloc(RuntimeType, num_results);
    errdefer allocator.free(results);

    for (0..num_inputs) |i| {
        const input_ty = mlir.Type.getFunctionInput(func_type, i);
        inputs[i] = try RuntimeType.fromMlirType(input_ty);
    }

    for (0..num_results) |i| {
        const result_ty = mlir.Type.getFunctionResult(func_type, i);
        results[i] = try RuntimeType.fromMlirType(result_ty);
    }

    return FunctionSignature{
        .inputs = inputs,
        .results = results,
        .allocator = allocator,
    };
}

/// Dynamic argument type for function calls
pub const FunctionArg = union(RuntimeType) {
    i1: bool,
    i8: i8,
    i16: i16,
    i32: i32,
    i64: i64,
    f32: f32,
    f64: f64,

    pub fn getType(self: FunctionArg) RuntimeType {
        return @as(RuntimeType, self);
    }
};

/// Call a JIT'd function with dynamic dispatch based on signature
pub fn callFunction(
    executor: anytype, // Should have a lookup() method
    fn_name: []const u8,
    signature: FunctionSignature,
    args: []const FunctionArg,
) !FunctionArg {
    // Validate argument count
    if (args.len != signature.inputs.len) {
        return error.ArgumentCountMismatch;
    }

    // Validate argument types
    for (args, signature.inputs) |arg, expected_type| {
        if (@intFromEnum(arg.getType()) != @intFromEnum(expected_type)) {
            return error.ArgumentCountMismatch; // Type mismatch
        }
    }

    // Look up the function
    const fn_ptr = executor.lookup(fn_name) orelse return error.FunctionNotFound;

    // Dispatch based on signature
    // For now, we'll handle common cases with comptime dispatch
    // This is a bit verbose but gives us type safety

    if (signature.inputs.len == 1 and signature.results.len == 1) {
        return try call1Arg1Result(fn_ptr, signature, args);
    } else if (signature.inputs.len == 2 and signature.results.len == 1) {
        return try call2Args1Result(fn_ptr, signature, args);
    } else {
        // Add more cases as needed
        return error.UnsupportedType;
    }
}

fn call1Arg1Result(
    fn_ptr: *anyopaque,
    signature: FunctionSignature,
    args: []const FunctionArg,
) !FunctionArg {
    const arg_type = signature.inputs[0];
    const result_type = signature.results[0];

    // Generate the appropriate function call based on types
    switch (arg_type) {
        .i32 => {
            const arg_val = args[0].i32;
            switch (result_type) {
                .i32 => {
                    const FnType = *const fn (i32) callconv(.c) i32;
                    const func: FnType = @ptrCast(@alignCast(fn_ptr));
                    return FunctionArg{ .i32 = func(arg_val) };
                },
                .i64 => {
                    const FnType = *const fn (i32) callconv(.c) i64;
                    const func: FnType = @ptrCast(@alignCast(fn_ptr));
                    return FunctionArg{ .i64 = func(arg_val) };
                },
                else => return error.UnsupportedType,
            }
        },
        .i64 => {
            const arg_val = args[0].i64;
            switch (result_type) {
                .i32 => {
                    const FnType = *const fn (i64) callconv(.c) i32;
                    const func: FnType = @ptrCast(@alignCast(fn_ptr));
                    return FunctionArg{ .i32 = func(arg_val) };
                },
                .i64 => {
                    const FnType = *const fn (i64) callconv(.c) i64;
                    const func: FnType = @ptrCast(@alignCast(fn_ptr));
                    return FunctionArg{ .i64 = func(arg_val) };
                },
                else => return error.UnsupportedType,
            }
        },
        .f32 => {
            const arg_val = args[0].f32;
            switch (result_type) {
                .f32 => {
                    const FnType = *const fn (f32) callconv(.c) f32;
                    const func: FnType = @ptrCast(@alignCast(fn_ptr));
                    return FunctionArg{ .f32 = func(arg_val) };
                },
                .f64 => {
                    const FnType = *const fn (f32) callconv(.c) f64;
                    const func: FnType = @ptrCast(@alignCast(fn_ptr));
                    return FunctionArg{ .f64 = func(arg_val) };
                },
                else => return error.UnsupportedType,
            }
        },
        .f64 => {
            const arg_val = args[0].f64;
            switch (result_type) {
                .f32 => {
                    const FnType = *const fn (f64) callconv(.c) f32;
                    const func: FnType = @ptrCast(@alignCast(fn_ptr));
                    return FunctionArg{ .f32 = func(arg_val) };
                },
                .f64 => {
                    const FnType = *const fn (f64) callconv(.c) f64;
                    const func: FnType = @ptrCast(@alignCast(fn_ptr));
                    return FunctionArg{ .f64 = func(arg_val) };
                },
                else => return error.UnsupportedType,
            }
        },
        else => return error.UnsupportedType,
    }
}

fn call2Args1Result(
    fn_ptr: *anyopaque,
    signature: FunctionSignature,
    args: []const FunctionArg,
) !FunctionArg {
    const arg0_type = signature.inputs[0];
    const arg1_type = signature.inputs[1];
    const result_type = signature.results[0];

    // Handle the most common case: (i32, i32) -> i32
    if (arg0_type == .i32 and arg1_type == .i32 and result_type == .i32) {
        const FnType = *const fn (i32, i32) callconv(.c) i32;
        const func: FnType = @ptrCast(@alignCast(fn_ptr));
        return FunctionArg{ .i32 = func(args[0].i32, args[1].i32) };
    }

    // Add more combinations as needed
    return error.UnsupportedType;
}
