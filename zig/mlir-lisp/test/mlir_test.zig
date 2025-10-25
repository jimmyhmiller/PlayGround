const std = @import("std");
const mlir_lisp = @import("mlir_lisp");
const mlir = mlir_lisp.mlir;

test "mlir - basic context creation" {
    var ctx = try mlir.Context.create();
    defer ctx.destroy();

    // Context should be valid
    try std.testing.expect(!mlir.c.mlirContextIsNull(ctx.ctx));
}

test "mlir - module creation" {
    var ctx = try mlir.Context.create();
    defer ctx.destroy();

    const loc = mlir.Location.unknown(&ctx);
    var mod = try mlir.Module.create(loc);
    defer mod.destroy();

    // Module should be valid
    try std.testing.expect(!mlir.c.mlirModuleIsNull(mod.module));
}

test "mlir - type creation" {
    var ctx = try mlir.Context.create();
    defer ctx.destroy();

    const i32_type = mlir.Type.@"i32"(&ctx);
    try std.testing.expect(!mlir.c.mlirTypeIsNull(i32_type));

    const i64_type = mlir.Type.@"i64"(&ctx);
    try std.testing.expect(!mlir.c.mlirTypeIsNull(i64_type));

    const f32_type = mlir.Type.@"f32"(&ctx);
    try std.testing.expect(!mlir.c.mlirTypeIsNull(f32_type));

    const f64_type = mlir.Type.@"f64"(&ctx);
    try std.testing.expect(!mlir.c.mlirTypeIsNull(f64_type));
}

test "mlir - location creation" {
    var ctx = try mlir.Context.create();
    defer ctx.destroy();

    const unknown_loc = mlir.Location.unknown(&ctx);
    try std.testing.expect(!mlir.c.mlirLocationIsNull(unknown_loc));

    const file_loc = mlir.Location.fileLineCol(&ctx, "test.mlir", 10, 5);
    try std.testing.expect(!mlir.c.mlirLocationIsNull(file_loc));
}
