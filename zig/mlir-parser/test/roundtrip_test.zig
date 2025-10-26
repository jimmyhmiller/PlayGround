//! Roundtrip Tests
//! Verifies that: parse(source) → AST → print(AST) → source' where source ≈ source'
//! Then: parse(source') → AST' where AST ≈ AST'

const std = @import("std");
const testing = std.testing;
const mlir = @import("mlir_parser");

const Lexer = mlir.Lexer;
const Parser = mlir.Parser;
const Printer = mlir.Printer;

/// Helper to parse MLIR source and return the module
fn parseSource(allocator: std.mem.Allocator, source: []const u8) !mlir.Module {
    return try mlir.parse(allocator, source);
}

/// Helper to print module to string
fn printModule(allocator: std.mem.Allocator, module: mlir.Module) ![]u8 {
    return try mlir.print(allocator, module);
}

/// Test roundtrip: parse → print → parse
fn testRoundtrip(allocator: std.mem.Allocator, source: []const u8) !void {
    // First parse
    var module1 = try parseSource(allocator, source);
    defer module1.deinit();

    // Print to string
    const printed = try printModule(allocator, module1);
    defer allocator.free(printed);

    // Second parse
    var module2 = try parseSource(allocator, printed);
    defer module2.deinit();

    // Print again
    const printed2 = try printModule(allocator, module2);
    defer allocator.free(printed2);

    // The two printed versions should be identical
    try testing.expectEqualStrings(printed, printed2);
}

test "roundtrip - simple constant operation" {
    const source = "%0 = \"arith.constant\"() <{value = 42 : i32}> : () -> i32";
    try testRoundtrip(testing.allocator, source);
}

test "roundtrip - simple addition operation" {
    const source =
        \\%0 = "arith.constant"() <{value = 42 : i32}> : () -> i32
        \\%1 = "arith.constant"() <{value = 13 : i32}> : () -> i32
        \\%2 = "arith.addi"(%0, %1) : (i32, i32) -> i32
    ;
    try testRoundtrip(testing.allocator, source);
}

test "roundtrip - integer types" {
    const source = "%0 = \"arith.constant\"() <{value = 1 : i1}> : () -> i1";
    try testRoundtrip(testing.allocator, source);
}

test "roundtrip - signed and unsigned integers" {
    const source = "%0 = \"arith.constant\"() <{value = 1 : si32}> : () -> si32";
    try testRoundtrip(testing.allocator, source);
}

test "roundtrip - float types" {
    const source = "%0 = \"arith.constant\"() <{value = 3.14 : f32}> : () -> f32";
    try testRoundtrip(testing.allocator, source);
}

test "roundtrip - index type" {
    const source = "%0 = \"arith.constant\"() <{value = 0 : index}> : () -> index";
    try testRoundtrip(testing.allocator, source);
}

test "roundtrip - multiple results" {
    const source = "%0:2 = \"test.op\"() : () -> (i32, i32)";
    try testRoundtrip(testing.allocator, source);
}

test "roundtrip - value use with result number" {
    const source = "%1 = \"test.op\"(%0#1) : (i32) -> i32";
    try testRoundtrip(testing.allocator, source);
}

test "roundtrip - tensor type" {
    const source = "%0 = \"test.op\"() : () -> tensor<4x8xf32>";
    try testRoundtrip(testing.allocator, source);
}

test "roundtrip - tensor with dynamic dimensions" {
    const source = "%0 = \"test.op\"() : () -> tensor<?x8xf32>";
    try testRoundtrip(testing.allocator, source);
}

test "roundtrip - memref type" {
    const source = "%0 = \"test.op\"() : () -> memref<10x20xf32>";
    try testRoundtrip(testing.allocator, source);
}

test "roundtrip - vector type" {
    const source = "%0 = \"test.op\"() : () -> vector<4x8xf32>";
    try testRoundtrip(testing.allocator, source);
}

test "roundtrip - vector with scalable dimensions" {
    const source = "%0 = \"test.op\"() : () -> vector<[4]x8xf32>";
    try testRoundtrip(testing.allocator, source);
}

test "roundtrip - complex type" {
    const source = "%0 = \"test.op\"() : () -> complex<f64>";
    try testRoundtrip(testing.allocator, source);
}

test "roundtrip - tuple type" {
    const source = "%0 = \"test.op\"() : () -> tuple<i32, f32>";
    try testRoundtrip(testing.allocator, source);
}

test "roundtrip - empty tuple type" {
    const source = "%0 = \"test.op\"() : () -> tuple<>";
    try testRoundtrip(testing.allocator, source);
}

test "roundtrip - none type" {
    const source = "%0 = \"test.op\"() : () -> none";
    try testRoundtrip(testing.allocator, source);
}

test "roundtrip - function type with multiple inputs" {
    const source = "%0 = \"test.op\"() : () -> (i32, i32, f32) -> i64";
    try testRoundtrip(testing.allocator, source);
}

test "roundtrip - dialect type" {
    const source = "%0 = \"test.op\"() : () -> !llvm.ptr";
    try testRoundtrip(testing.allocator, source);
}

test "roundtrip - dialect type with body" {
    const source = "%0 = \"test.op\"() : () -> !llvm<ptr<i32>>";
    try testRoundtrip(testing.allocator, source);
}

test "roundtrip - type alias" {
    const source = "%0 = \"test.op\"() : () -> !my_alias";
    try testRoundtrip(testing.allocator, source);
}

test "roundtrip - operation with attributes" {
    const source = "%0 = \"test.op\"() {attr1 = 42 : i32, attr2 = true} : () -> i32";
    try testRoundtrip(testing.allocator, source);
}

test "roundtrip - operation with region (entry block only)" {
    const source =
        \\%0 = "test.op"() ({
        \\  %1 = "test.inner"() : () -> i32
        \\}) : () -> i32
    ;
    try testRoundtrip(testing.allocator, source);
}

test "roundtrip - operation with region (labeled blocks)" {
    const source =
        \\%0 = "test.op"() ({
        \\^bb0:
        \\  %1 = "test.inner"() : () -> i32
        \\}) : () -> i32
    ;
    try testRoundtrip(testing.allocator, source);
}

test "roundtrip - operation with successors" {
    const source = "%0 = \"cf.br\"() [^bb1] : () -> ()";
    try testRoundtrip(testing.allocator, source);
}

test "roundtrip - operation with successor arguments" {
    const source = "%0 = \"cf.br\"(%arg0) [^bb1(%arg0 : i32)] : (i32) -> ()";
    try testRoundtrip(testing.allocator, source);
}

test "roundtrip - operation with location" {
    const source = "%0 = \"test.op\"() : () -> i32 loc(unknown)";
    try testRoundtrip(testing.allocator, source);
}

test "roundtrip - type alias definition" {
    const source = "!my_type = i32";
    try testRoundtrip(testing.allocator, source);
}

test "roundtrip - attribute alias definition" {
    const source = "#my_attr = 42 : i32";
    try testRoundtrip(testing.allocator, source);
}

test "roundtrip - module with multiple operations" {
    const source =
        \\%0 = "arith.constant"() <{value = 1 : i32}> : () -> i32
        \\%1 = "arith.constant"() <{value = 2 : i32}> : () -> i32
        \\%2 = "arith.addi"(%0, %1) : (i32, i32) -> i32
    ;
    try testRoundtrip(testing.allocator, source);
}

test "roundtrip - complex nested region" {
    const source =
        \\%0 = "scf.if"(%cond) ({
        \\  %1 = "arith.constant"() <{value = 1 : i32}> : () -> i32
        \\  "scf.yield"(%1) : (i32) -> ()
        \\}, {
        \\  %2 = "arith.constant"() <{value = 2 : i32}> : () -> i32
        \\  "scf.yield"(%2) : (i32) -> ()
        \\}) : (i1) -> i32
    ;
    try testRoundtrip(testing.allocator, source);
}

// Test that we can parse the printed output with mlir-opt
// This test requires mlir-opt to be installed
test "roundtrip - validate with mlir-opt" {
    if (true) return error.SkipZigTest; // Skip by default - requires mlir-opt

    const source = "%0 = \"arith.constant\"() <{value = 42 : i32}> : () -> i32";

    // Parse and print
    var module = try parseSource(testing.allocator, source);
    defer module.deinit();

    const printed = try printModule(testing.allocator, module);
    defer testing.allocator.free(printed);

    // Write to temp file
    const tmp_file = try std.fs.cwd().createFile("test_roundtrip.mlir", .{});
    defer {
        tmp_file.close();
        std.fs.cwd().deleteFile("test_roundtrip.mlir") catch {};
    }

    try tmp_file.writeAll(printed);

    // Validate with mlir-opt
    const result = try std.ChildProcess.exec(.{
        .allocator = testing.allocator,
        .argv = &[_][]const u8{ "mlir-opt", "--verify-diagnostics", "test_roundtrip.mlir" },
    });
    defer {
        testing.allocator.free(result.stdout);
        testing.allocator.free(result.stderr);
    }

    try testing.expectEqual(@as(u32, 0), result.term.Exited);
}
