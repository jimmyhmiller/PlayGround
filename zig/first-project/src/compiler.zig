const std = @import("std");
const Reader = @import("reader.zig").Reader;
const Value = @import("value.zig").Value;

pub const Compiler = struct {
    allocator: *std.mem.Allocator,

    pub fn init(allocator: *std.mem.Allocator) Compiler {
        return Compiler{
            .allocator = allocator,
        };
    }

    pub fn compileString(self: *Compiler, source: []const u8) !void {
        var reader = Reader.init(self.allocator);
        const results = try reader.readAllString(source);
        for (results.items) |expr| {
            try self.compileExpression(expr);
        }
    }

    pub fn compileExpression(self: *Compiler, expr: *Value) !void {
        switch (expr.*) {
            .list => {
                // For demonstration, just print the list
                std.debug.print("Compiling list expression:\n", .{});
                var iter = expr.list.iterator();
                while (iter.next()) |entry| {
                    try self.compileExpression(entry);
                }
            },
            .symbol => {
                if (std.mem.eql(u8, expr.symbol, "def")) {
                    std.debug.print("Defining a new variable or function\n", .{});
                } else if (std.mem.eql(u8, expr.symbol, "fn")) {
                    std.debug.print("Defining a new function\n", .{});
                } else if (std.mem.eql(u8, expr.symbol, "->")) {
                    std.debug.print("Defining a function type\n", .{});
                } else {
                    std.debug.print("Compiling symbol: {s}\n", .{expr.symbol});
                }
            },
            .int => {
                std.debug.print("Compiling integer: {d}\n", .{expr.int});
            },
            .string => {
                std.debug.print("Compiling string: {s}\n", .{expr.string});
            },
            .keyword => {
                std.debug.print("Compiling keyword: :{s}\n", .{expr.keyword});
            },
            .float => {
                std.debug.print("Compiling float: {d}\n", .{expr.float});
            },
            .vector => {
                std.debug.print("Compiling vector:\n", .{});
                var iter = expr.vector.iterator();
                while (iter.next()) |entry| {
                    try self.compileExpression(entry);
                }
            },
            .map => {
                std.debug.print("Compiling map:\n", .{});
                var iter = expr.map.iterator();
                while (iter.next()) |entry| {
                    try self.compileExpression(entry.key);
                    try self.compileExpression(entry.value);
                }
            },
            .nil => {
                std.debug.print("Compiling nil\n", .{});
            },
            // .boolean => {
            //     std.debug.print("Compiling boolean: {s}\n", .{expr.boolean});
            // },
        }
    }
};
