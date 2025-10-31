const std = @import("std");
const mlir_lisp = @import("mlir_lisp");
const Reader = mlir_lisp.Reader;
const Value = mlir_lisp.Value;
const Tokenizer = mlir_lisp.Tokenizer;

// Test that we can round-trip: parse -> print -> parse and get the same result
test "round-trip: simple atoms" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const allocator = arena.allocator();

    const test_cases = [_][]const u8{
        "hello",
        "42",
        "\"hello world\"",
        "%x",
        "^entry",
        "@main",
        ":value",
        "true",
        "false",
    };

    for (test_cases) |source| {
        // Parse
        var tok = Tokenizer.init(allocator, source);
        var rdr = try Reader.init(allocator, &tok);
        const value1 = try rdr.read();

        // Print
        var buf = std.ArrayList(u8){};
        try value1.print(buf.writer(allocator));

        // Parse again
        var tok2 = Tokenizer.init(allocator, buf.items);
        var rdr2 = try Reader.init(allocator, &tok2);
        const value2 = try rdr2.read();

        // Print again and compare
        var buf2 = std.ArrayList(u8){};
        try value2.print(buf2.writer(allocator));

        try std.testing.expectEqualStrings(buf.items, buf2.items);
    }
}

test "round-trip: collections" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const allocator = arena.allocator();

    const test_cases = [_][]const u8{
        "()",
        "(1 2 3)",
        "(1 (2 3) 4)",
        "[]",
        "[%x %y]",
        "{}",
        "{:value 42 :name \"test\"}",
    };

    for (test_cases) |source| {
        // Parse
        var tok = Tokenizer.init(allocator, source);
        var rdr = try Reader.init(allocator, &tok);
        const value1 = try rdr.read();

        // Print
        var buf = std.ArrayList(u8){};
        try value1.print(buf.writer(allocator));

        // Parse again
        var tok2 = Tokenizer.init(allocator, buf.items);
        var rdr2 = try Reader.init(allocator, &tok2);
        const value2 = try rdr2.read();

        // Print again and compare
        var buf2 = std.ArrayList(u8){};
        try value2.print(buf2.writer(allocator));

        try std.testing.expectEqualStrings(buf.items, buf2.items);
    }
}

test "round-trip: types and attributes" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const allocator = arena.allocator();

    const test_cases = [_][]const u8{
        "!llvm.ptr",
        "!llvm.array<10 x i8>",
        "#arith.overflow<none>",
    };

    for (test_cases) |source| {
        // Parse
        var tok = Tokenizer.init(allocator, source);
        var rdr = try Reader.init(allocator, &tok);
        const value1 = try rdr.read();

        // Print
        var buf = std.ArrayList(u8){};
        try value1.print(buf.writer(allocator));

        // Parse again
        var tok2 = Tokenizer.init(allocator, buf.items);
        var rdr2 = try Reader.init(allocator, &tok2);
        const value2 = try rdr2.read();

        // Print again and compare
        var buf2 = std.ArrayList(u8){};
        try value2.print(buf2.writer(allocator));

        try std.testing.expectEqualStrings(buf.items, buf2.items);
    }
}

test "round-trip: typed literals" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const allocator = arena.allocator();

    const test_cases = [_][]const u8{
        "(: 42 !llvm.i32)",
        "(: 42 i32)",
    };

    for (test_cases) |source| {
        // Parse
        var tok = Tokenizer.init(allocator, source);
        var rdr = try Reader.init(allocator, &tok);
        const value1 = try rdr.read();

        // Print
        var buf = std.ArrayList(u8){};
        try value1.print(buf.writer(allocator));

        // Parse again
        var tok2 = Tokenizer.init(allocator, buf.items);
        var rdr2 = try Reader.init(allocator, &tok2);
        const value2 = try rdr2.read();

        // Print again and compare
        var buf2 = std.ArrayList(u8){};
        try value2.print(buf2.writer(allocator));

        try std.testing.expectEqualStrings(buf.items, buf2.items);
    }
}

test "round-trip: function types" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const allocator = arena.allocator();

    const test_cases = [_][]const u8{
        "(!function (inputs i32 i32) (results i32))",
        "(!function (inputs) (results i32))",
        "(!function (inputs i32) (results))",
    };

    for (test_cases) |source| {
        // Parse
        var tok = Tokenizer.init(allocator, source);
        var rdr = try Reader.init(allocator, &tok);
        const value1 = try rdr.read();

        // Print
        var buf = std.ArrayList(u8){};
        try value1.print(buf.writer(allocator));

        // Parse again
        var tok2 = Tokenizer.init(allocator, buf.items);
        var rdr2 = try Reader.init(allocator, &tok2);
        const value2 = try rdr2.read();

        // Print again and compare
        var buf2 = std.ArrayList(u8){};
        try value2.print(buf2.writer(allocator));

        try std.testing.expectEqualStrings(buf.items, buf2.items);
    }
}

test "round-trip: complex nested expression" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const allocator = arena.allocator();

    const source =
        \\(operation (name arith.constant) (result-bindings [%c0]) (result-types i32) (attributes {:value #(int 42)}))
    ;

    // Parse
    var tok = Tokenizer.init(allocator, source);
    var rdr = try Reader.init(allocator, &tok);
    const value1 = try rdr.read();

    // Print
    var buf = std.ArrayList(u8){};
    try value1.print(buf.writer(allocator));

    // Parse again
    var tok2 = Tokenizer.init(allocator, buf.items);
    var rdr2 = try Reader.init(allocator, &tok2);
    const value2 = try rdr2.read();

    // Print again and compare
    var buf2 = std.ArrayList(u8){};
    try value2.print(buf2.writer(allocator));

    try std.testing.expectEqualStrings(buf.items, buf2.items);

    // Verify we can still parse it a third time (stable)
    var tok3 = Tokenizer.init(allocator, buf2.items);
    var rdr3 = try Reader.init(allocator, &tok3);
    const value3 = try rdr3.read();

    var buf3 = std.ArrayList(u8){};
    try value3.print(buf3.writer(allocator));

    try std.testing.expectEqualStrings(buf2.items, buf3.items);
}

test "round-trip: multiple expressions" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const allocator = arena.allocator();

    const source = "1 2 (3 4) [5 6] {:key \"value\"}";

    // Parse all
    var tok = Tokenizer.init(allocator, source);
    var rdr = try Reader.init(allocator, &tok);
    const values1 = try rdr.readAll();

    // Print all with spaces between
    var buf = std.ArrayList(u8){};
    const values1_slice = values1.slice();
    for (values1_slice, 0..) |val, i| {
        if (i > 0) try buf.writer(allocator).writeAll(" ");
        try val.print(buf.writer(allocator));
    }

    // Parse again
    var tok2 = Tokenizer.init(allocator, buf.items);
    var rdr2 = try Reader.init(allocator, &tok2);
    const values2 = try rdr2.readAll();

    // Print again and compare
    var buf2 = std.ArrayList(u8){};
    const values2_slice = values2.slice();
    for (values2_slice, 0..) |val, i| {
        if (i > 0) try buf2.writer(allocator).writeAll(" ");
        try val.print(buf2.writer(allocator));
    }

    try std.testing.expectEqualStrings(buf.items, buf2.items);
}
