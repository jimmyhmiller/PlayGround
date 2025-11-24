const std = @import("std");

pub const tokenizer = @import("tokenizer.zig");
pub const reader_types = @import("reader_types.zig");
pub const reader = @import("reader.zig");
pub const ast = @import("ast.zig");
pub const parser = @import("parser.zig");
pub const mlir_integration = @import("mlir_integration.zig");

/// Main API for parsing Lispier code
pub const Compiler = struct {
    allocator: std.mem.Allocator,
    dialect_registry: mlir_integration.DialectRegistry,

    pub fn init(allocator: std.mem.Allocator) !Compiler {
        return .{
            .allocator = allocator,
            .dialect_registry = try mlir_integration.DialectRegistry.init(allocator),
        };
    }

    pub fn deinit(self: *Compiler) void {
        self.dialect_registry.deinit();
    }

    /// Compile source code to AST
    pub fn compile(self: *Compiler, source: []const u8) !CompileResult {
        // Tokenize
        var tok = tokenizer.Tokenizer.init(self.allocator, source);
        var tokens = try tok.tokenize();
        errdefer tokens.deinit(self.allocator);

        // Read
        var rdr = reader.Reader.init(self.allocator, tokens.items);
        defer rdr.deinit();

        var values = try rdr.read();
        errdefer {
            for (values.items) |v| {
                v.deinit();
            }
            values.deinit(self.allocator);
        }

        // Parse
        var psr = parser.Parser.init(self.allocator);
        var nodes = try psr.parse(values);
        errdefer {
            for (nodes.items) |n| {
                n.deinit();
            }
            nodes.deinit(self.allocator);
        }

        // Validate
        var validator = mlir_integration.ASTValidator.init(self.allocator, &self.dialect_registry);
        defer validator.deinit();

        var all_valid = true;
        for (nodes.items) |node| {
            const valid = try validator.validate(node);
            if (!valid) {
                all_valid = false;
            }
        }

        return .{
            .tokens = tokens,
            .values = values,
            .nodes = nodes,
            .validation_errors = try self.allocator.dupe(
                mlir_integration.ASTValidator.ErrorInfo,
                validator.getErrors(),
            ),
            .is_valid = all_valid,
        };
    }

    pub const CompileResult = struct {
        tokens: std.ArrayList(tokenizer.Token),
        values: std.ArrayList(*reader_types.Value),
        nodes: std.ArrayList(*ast.Node),
        validation_errors: []const mlir_integration.ASTValidator.ErrorInfo,
        is_valid: bool,

        pub fn deinit(self: *CompileResult, allocator: std.mem.Allocator) void {
            self.tokens.deinit(allocator);

            // Free nodes first
            for (self.nodes.items) |n| {
                n.deinit();
            }
            self.nodes.deinit(allocator);

            // Then free values (nodes don't own them)
            for (self.values.items) |v| {
                v.deinit();
            }
            self.values.deinit(allocator);

            allocator.free(self.validation_errors);
        }
    };
};

test "compile simple expression" {
    const allocator = std.testing.allocator;

    var compiler = try Compiler.init(allocator);
    defer compiler.deinit();

    var result = try compiler.compile("(require-dialect arith) (arith.addi 1 2)");
    defer result.deinit(allocator);

    try std.testing.expect(result.is_valid);
    try std.testing.expectEqual(@as(usize, 0), result.validation_errors.len);
}

test "compile with invalid operation" {
    const allocator = std.testing.allocator;

    var compiler = try Compiler.init(allocator);
    defer compiler.deinit();

    var result = try compiler.compile("(require-dialect arith) (arith.invalid_op 1 2)");
    defer result.deinit(allocator);

    try std.testing.expect(!result.is_valid);
    try std.testing.expect(result.validation_errors.len > 0);
}

test {
    _ = tokenizer;
    _ = reader_types;
    _ = reader;
    _ = ast;
    _ = parser;
    _ = mlir_integration;
}
