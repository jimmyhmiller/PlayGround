const std = @import("std");
const parser = @import("parser.zig");
const value = @import("value.zig");

const Parser = parser.Parser;
const Value = value.Value;

pub const Reader = struct {
    allocator: *std.mem.Allocator,

    pub fn init(allocator: *std.mem.Allocator) Reader {
        return Reader{
            .allocator = allocator,
        };
    }

    pub fn readString(self: *Reader, source: []const u8) !*Value {
        var p = Parser.init(self.allocator, source);
        return try p.parse();
    }

    pub const ReadAllResult = struct {
        values: std.ArrayList(*Value),
        line_numbers: std.ArrayList(u32),
    };

    pub fn readAllString(self: *Reader, source: []const u8) !ReadAllResult {
        var p = Parser.init(self.allocator, source);
        var results = std.ArrayList(*Value){};
        var line_numbers = std.ArrayList(u32){};
        try p.parseAll(self.allocator.*, &results, &line_numbers);
        return ReadAllResult{
            .values = results,
            .line_numbers = line_numbers,
        };
    }

    pub fn valueToString(self: *Reader, val: *Value) ![]u8 {
        var buf: [2048]u8 = undefined;
        var stream = std.io.fixedBufferStream(&buf);
        try val.format("", .{}, stream.writer());
        return try self.allocator.dupe(u8, buf[0..stream.pos]);
    }

    pub fn printValue(_: *Reader, val: *Value) void {
        // Use a simple format string approach with debug print
        var buf: [1024]u8 = undefined;
        var stream = std.io.fixedBufferStream(&buf);
        val.format("", .{}, stream.writer()) catch {};
        std.debug.print("{s}\n", .{buf[0..stream.pos]});
    }
};

test "reader complex example" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    var allocator = arena.allocator();
    var reader = Reader.init(&allocator);

    const clojure_code = "(defn my-func [{:keys [a b c]}] (+ a b c))";
    const result = try reader.readString(clojure_code);

    // Should be a list
    try std.testing.expect(result.isList());

    // The list should have 4 elements: defn, my-func, parameter vector, and body
    var current: ?*const @TypeOf(result.list.*) = result.list;
    var count: usize = 0;
    while (current != null and !current.?.isEmpty()) {
        count += 1;
        current = current.?.next;
    }
    try std.testing.expect(count == 4);

    // First element should be 'defn'
    try std.testing.expect(result.list.value.?.isSymbol());
    try std.testing.expect(std.mem.eql(u8, result.list.value.?.symbol, "defn"));

    // Second element should be 'my-func'
    const second = result.list.next.?.value.?;
    try std.testing.expect(second.isSymbol());
    try std.testing.expect(std.mem.eql(u8, second.symbol, "my-func"));

    // Third element should be a vector (parameters)
    const third = result.list.next.?.next.?.value.?;
    try std.testing.expect(third.isVector());

    // Fourth element should be a list (body)
    const fourth = result.list.next.?.next.?.next.?.value.?;
    try std.testing.expect(fourth.isList());
}

test "reader multiple expressions" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    var allocator = arena.allocator();
    var reader = Reader.init(&allocator);

    const code = "42 :hello \"world\" (+ 1 2) [a b c]";
    const read_result = try reader.readAllString(code);
    const results = read_result.values;

    try std.testing.expect(results.items.len == 5);

    try std.testing.expect(results.items[0].isInt());
    try std.testing.expect(results.items[0].int == 42);

    try std.testing.expect(results.items[1].isKeyword());
    try std.testing.expect(std.mem.eql(u8, results.items[1].keyword, "hello"));

    try std.testing.expect(results.items[2].isString());
    try std.testing.expect(std.mem.eql(u8, results.items[2].string, "world"));

    try std.testing.expect(results.items[3].isList());
    try std.testing.expect(results.items[4].isVector());
}

test "reader nested structures" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    var allocator = arena.allocator();
    var reader = Reader.init(&allocator);

    const code = "{:name \"John\" :age 30 :hobbies [\"reading\" \"coding\"]}";
    const result = try reader.readString(code);

    try std.testing.expect(result.isMap());

    // Map should contain the key-value pairs
    const name_key = try value.createKeyword(arena.allocator(), "name");
    const name_val = result.map.get(name_key);
    try std.testing.expect(name_val != null);
    try std.testing.expect(name_val.?.isString());
}

test "reader quote" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    var allocator = arena.allocator();
    var reader = Reader.init(&allocator);

    const code = "'hello";
    const result = try reader.readString(code);

    // Should be (quote hello)
    try std.testing.expect(result.isList());
    try std.testing.expect(result.list.len() == 2);

    // First element should be 'quote'
    try std.testing.expect(result.list.value.?.isSymbol());
    try std.testing.expect(std.mem.eql(u8, result.list.value.?.symbol, "quote"));

    // Second element should be 'hello'
    const quoted = result.list.next.?.value.?;
    try std.testing.expect(quoted.isSymbol());
    try std.testing.expect(std.mem.eql(u8, quoted.symbol, "hello"));
}

test "reader round-trip" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();

    // Test cases for round-trip (read -> print -> read should be equivalent)
    const test_cases = [_][]const u8{
        "42",
        "3.14",
        "hello",
        ":keyword",
        "\"string\"",
        "nil",
        "()",
        "(+ 1 2)",
        "[1 2 3]",
        "{:a 1 :b 2}",
        "(defn test [x] (+ x 1))",
        "[[:nested [\"vector\"]] {:with {:nested :map}}]",
        "(ns my.namespace)",
    };

    for (test_cases) |original| {
        // Create a new arena for each test case
        var test_arena = std.heap.ArenaAllocator.init(std.testing.allocator);
        defer test_arena.deinit();
        var allocator = test_arena.allocator();
        var reader = Reader.init(&allocator);

        // Read the original
        const parsed = try reader.readString(original);

        // Convert back to string
        const printed = try reader.valueToString(parsed);

        // Store the type and value info before creating new reader
        const original_tag = std.meta.activeTag(parsed.*);
        var expected_int: i64 = undefined;
        var expected_float: f64 = undefined;
        var expected_symbol: []u8 = undefined;
        var expected_keyword: []u8 = undefined;
        var expected_string: []u8 = undefined;

        switch (parsed.*) {
            .int => |i| expected_int = i,
            .float => |f| expected_float = f,
            .symbol => |s| expected_symbol = try std.testing.allocator.dupe(u8, s),
            .keyword => |k| expected_keyword = try std.testing.allocator.dupe(u8, k),
            .string => |str| expected_string = try std.testing.allocator.dupe(u8, str),
            .nil => {},
            else => {},
        }

        // Copy the printed string to avoid it being freed
        const printed_copy = try std.testing.allocator.dupe(u8, printed);
        defer std.testing.allocator.free(printed_copy);

        // Create a new arena and reader for reparsing
        var reparse_arena = std.heap.ArenaAllocator.init(std.testing.allocator);
        defer reparse_arena.deinit();
        var reparse_allocator = reparse_arena.allocator();
        var reparse_reader = Reader.init(&reparse_allocator);

        // Read the printed version
        const reparsed = try reparse_reader.readString(printed_copy);

        // The values should be equal
        const reparsed_tag = std.meta.activeTag(reparsed.*);
        try std.testing.expect(original_tag == reparsed_tag);

        // Basic value checks for some types
        switch (original_tag) {
            .int => try std.testing.expect(reparsed.int == expected_int),
            .float => try std.testing.expect(reparsed.float == expected_float),
            .symbol => {
                defer std.testing.allocator.free(expected_symbol);
                try std.testing.expect(std.mem.eql(u8, reparsed.symbol, expected_symbol));
            },
            .keyword => {
                defer std.testing.allocator.free(expected_keyword);
                try std.testing.expect(std.mem.eql(u8, reparsed.keyword, expected_keyword));
            },
            .string => {
                defer std.testing.allocator.free(expected_string);
                try std.testing.expect(std.mem.eql(u8, reparsed.string, expected_string));
            },
            .nil => {},
            else => {}, // For complex types, just check the tag for now
        }
    }
}
