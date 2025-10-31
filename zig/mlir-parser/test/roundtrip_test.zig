//! Roundtrip Tests
//! Verifies that: parse(source) → AST → print(AST) → source' where source ≈ source'
//! Then: parse(source') → AST' where AST ≈ AST'

const std = @import("std");
const testing = std.testing;
const mlir = @import("mlir_parser");

const Lexer = mlir.Lexer;
const Parser = mlir.Parser;
const Printer = mlir.Printer;

/// Helper to load test file from test_data/examples/
fn loadTestFile(allocator: std.mem.Allocator, filename: []const u8) ![]u8 {
    // Get the directory of this test file
    const dir = std.fs.cwd();

    // Build path: test_data/examples/{filename}
    const path = try std.fmt.allocPrint(allocator, "test_data/examples/{s}", .{filename});
    defer allocator.free(path);

    // Read file
    const file = try dir.openFile(path, .{});
    defer file.close();

    const content = try file.readToEndAlloc(allocator, 1024 * 1024); // 1MB max
    return content;
}

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

/// Test roundtrip by loading from file
fn testRoundtripFromFile(allocator: std.mem.Allocator, filename: []const u8) !void {
    const source = try loadTestFile(allocator, filename);
    defer allocator.free(source);
    try testRoundtrip(allocator, source);
}

test "roundtrip - simple constant operation" {
    var arena = std.heap.ArenaAllocator.init(testing.allocator);
    defer arena.deinit();
    try testRoundtripFromFile(arena.allocator(), "simple_constant.mlir");
}

test "roundtrip - simple addition operation" {
    var arena = std.heap.ArenaAllocator.init(testing.allocator);
    defer arena.deinit();
    try testRoundtripFromFile(arena.allocator(), "simple_addition.mlir");
}

test "roundtrip - integer types" {
    var arena = std.heap.ArenaAllocator.init(testing.allocator);
    defer arena.deinit();
    try testRoundtripFromFile(arena.allocator(), "integer_type_i1.mlir");
}

test "roundtrip - signed and unsigned integers" {
    var arena = std.heap.ArenaAllocator.init(testing.allocator);
    defer arena.deinit();
    try testRoundtripFromFile(arena.allocator(), "integer_type_si32.mlir");
}

test "roundtrip - float types" {
    var arena = std.heap.ArenaAllocator.init(testing.allocator);
    defer arena.deinit();
    try testRoundtripFromFile(arena.allocator(), "float_type_f32.mlir");
}

test "roundtrip - index type" {
    var arena = std.heap.ArenaAllocator.init(testing.allocator);
    defer arena.deinit();
    try testRoundtripFromFile(arena.allocator(), "index_type.mlir");
}

test "roundtrip - multiple results" {
    var arena = std.heap.ArenaAllocator.init(testing.allocator);
    defer arena.deinit();
    try testRoundtripFromFile(arena.allocator(), "multiple_results.mlir");
}

test "roundtrip - value use with result number" {
    var arena = std.heap.ArenaAllocator.init(testing.allocator);
    defer arena.deinit();
    try testRoundtripFromFile(arena.allocator(), "value_use_with_result_number.mlir");
}

test "roundtrip - tensor type" {
    var arena = std.heap.ArenaAllocator.init(testing.allocator);
    defer arena.deinit();
    try testRoundtripFromFile(arena.allocator(), "tensor_type.mlir");
}

test "roundtrip - tensor with dynamic dimensions" {
    var arena = std.heap.ArenaAllocator.init(testing.allocator);
    defer arena.deinit();
    try testRoundtripFromFile(arena.allocator(), "tensor_dynamic_dimensions.mlir");
}

test "roundtrip - memref type" {
    var arena = std.heap.ArenaAllocator.init(testing.allocator);
    defer arena.deinit();
    try testRoundtripFromFile(arena.allocator(), "memref_type.mlir");
}

test "roundtrip - vector type" {
    var arena = std.heap.ArenaAllocator.init(testing.allocator);
    defer arena.deinit();
    try testRoundtripFromFile(arena.allocator(), "vector_type.mlir");
}

test "roundtrip - vector with scalable dimensions" {
    var arena = std.heap.ArenaAllocator.init(testing.allocator);
    defer arena.deinit();
    try testRoundtripFromFile(arena.allocator(), "vector_scalable_dimensions.mlir");
}

test "roundtrip - complex type" {
    var arena = std.heap.ArenaAllocator.init(testing.allocator);
    defer arena.deinit();
    try testRoundtripFromFile(arena.allocator(), "complex_type.mlir");
}

test "roundtrip - tuple type" {
    var arena = std.heap.ArenaAllocator.init(testing.allocator);
    defer arena.deinit();
    try testRoundtripFromFile(arena.allocator(), "tuple_type.mlir");
}

test "roundtrip - empty tuple type" {
    var arena = std.heap.ArenaAllocator.init(testing.allocator);
    defer arena.deinit();
    try testRoundtripFromFile(arena.allocator(), "empty_tuple_type.mlir");
}

test "roundtrip - none type" {
    var arena = std.heap.ArenaAllocator.init(testing.allocator);
    defer arena.deinit();
    try testRoundtripFromFile(arena.allocator(), "none_type.mlir");
}

test "roundtrip - dialect type" {
    var arena = std.heap.ArenaAllocator.init(testing.allocator);
    defer arena.deinit();
    try testRoundtripFromFile(arena.allocator(), "dialect_type.mlir");
}

test "roundtrip - dialect type with body" {
    var arena = std.heap.ArenaAllocator.init(testing.allocator);
    defer arena.deinit();
    try testRoundtripFromFile(arena.allocator(), "dialect_type_with_body.mlir");
}

test "roundtrip - type alias" {
    var arena = std.heap.ArenaAllocator.init(testing.allocator);
    defer arena.deinit();
    try testRoundtripFromFile(arena.allocator(), "type_alias.mlir");
}

test "roundtrip - operation with attributes" {
    var arena = std.heap.ArenaAllocator.init(testing.allocator);
    defer arena.deinit();
    try testRoundtripFromFile(arena.allocator(), "operation_with_attributes.mlir");
}

test "roundtrip - operation with region (entry block only)" {
    var arena = std.heap.ArenaAllocator.init(testing.allocator);
    defer arena.deinit();
    try testRoundtripFromFile(arena.allocator(), "operation_with_region_entry_block.mlir");
}

test "roundtrip - operation with region (labeled blocks)" {
    var arena = std.heap.ArenaAllocator.init(testing.allocator);
    defer arena.deinit();
    try testRoundtripFromFile(arena.allocator(), "operation_with_region_labeled_blocks.mlir");
}

test "roundtrip - operation with successors" {
    var arena = std.heap.ArenaAllocator.init(testing.allocator);
    defer arena.deinit();
    try testRoundtripFromFile(arena.allocator(), "operation_with_successors.mlir");
}

test "roundtrip - operation with successor arguments" {
    var arena = std.heap.ArenaAllocator.init(testing.allocator);
    defer arena.deinit();
    try testRoundtripFromFile(arena.allocator(), "operation_with_successor_arguments.mlir");
}

test "roundtrip - operation with location" {
    var arena = std.heap.ArenaAllocator.init(testing.allocator);
    defer arena.deinit();
    try testRoundtripFromFile(arena.allocator(), "operation_with_location.mlir");
}

test "roundtrip - type alias definition" {
    var arena = std.heap.ArenaAllocator.init(testing.allocator);
    defer arena.deinit();
    try testRoundtripFromFile(arena.allocator(), "type_alias_definition.mlir");
}

test "roundtrip - attribute alias definition" {
    var arena = std.heap.ArenaAllocator.init(testing.allocator);
    defer arena.deinit();
    try testRoundtripFromFile(arena.allocator(), "attribute_alias_definition.mlir");
}

test "roundtrip - module with multiple operations" {
    var arena = std.heap.ArenaAllocator.init(testing.allocator);
    defer arena.deinit();
    try testRoundtripFromFile(arena.allocator(), "module_with_multiple_operations.mlir");
}

test "roundtrip - complex nested region" {
    var arena = std.heap.ArenaAllocator.init(testing.allocator);
    defer arena.deinit();
    try testRoundtripFromFile(arena.allocator(), "complex_nested_region.mlir");
}

// Auto-discover and test all .mlir files in test_data/examples/
test "roundtrip - all example files" {
    var arena = std.heap.ArenaAllocator.init(testing.allocator);
    defer arena.deinit();
    const allocator = arena.allocator();

    // Open the test_data/examples directory
    const dir = try std.fs.cwd().openDir("test_data/examples", .{ .iterate = true });
    var iter = dir.iterate();

    var tested_count: usize = 0;

    while (try iter.next()) |entry| {
        // Only test .mlir files
        if (entry.kind != .file) continue;
        if (!std.mem.endsWith(u8, entry.name, ".mlir")) continue;

        // ALL files MUST be in generic format and pass roundtrip test
        // Failing tests show us what features need to be implemented

        // Run roundtrip test on this file
        std.debug.print("Testing roundtrip for: {s}\n", .{entry.name});
        testRoundtripFromFile(allocator, entry.name) catch |err| {
            std.debug.print("FAILED: {s} - Error: {}\n", .{ entry.name, err });
            return err;
        };
        tested_count += 1;
    }

    std.debug.print("Roundtrip testing complete: {d} files tested\n", .{tested_count});
}
