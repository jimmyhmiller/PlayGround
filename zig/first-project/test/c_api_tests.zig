const std = @import("std");
const c_api = @import("c_api.zig");

test "c_api: reader create and destroy" {
    const reader = c_api.reader_create();
    try std.testing.expect(reader != null);
    c_api.reader_destroy(reader);
}

test "c_api: reader read simple int" {
    const reader = c_api.reader_create();
    defer c_api.reader_destroy(reader);

    const val = c_api.reader_read_string(reader, "42");
    try std.testing.expect(val != null);
    try std.testing.expect(c_api.value_is_int(val));
    try std.testing.expect(c_api.value_get_int(val) == 42);
}

test "c_api: reader read symbol" {
    const reader = c_api.reader_create();
    defer c_api.reader_destroy(reader);

    const val = c_api.reader_read_string(reader, "hello");
    try std.testing.expect(val != null);
    try std.testing.expect(c_api.value_is_symbol(val));

    const sym = c_api.value_get_symbol(val);
    try std.testing.expect(sym != null);
    const sym_slice = std.mem.span(sym.?);
    try std.testing.expect(std.mem.eql(u8, sym_slice, "hello"));
}

test "c_api: reader read keyword" {
    const reader = c_api.reader_create();
    defer c_api.reader_destroy(reader);

    const val = c_api.reader_read_string(reader, ":world");
    try std.testing.expect(val != null);
    try std.testing.expect(c_api.value_is_keyword(val));

    const kw = c_api.value_get_keyword(val);
    try std.testing.expect(kw != null);
    const kw_slice = std.mem.span(kw.?);
    try std.testing.expect(std.mem.eql(u8, kw_slice, "world"));
}

test "c_api: reader read string" {
    const reader = c_api.reader_create();
    defer c_api.reader_destroy(reader);

    const val = c_api.reader_read_string(reader, "\"test\"");
    try std.testing.expect(val != null);
    try std.testing.expect(c_api.value_is_string(val));

    const str = c_api.value_get_string(val);
    try std.testing.expect(str != null);
    const str_slice = std.mem.span(str.?);
    try std.testing.expect(std.mem.eql(u8, str_slice, "test"));
}

test "c_api: reader read float" {
    const reader = c_api.reader_create();
    defer c_api.reader_destroy(reader);

    const val = c_api.reader_read_string(reader, "3.14");
    try std.testing.expect(val != null);
    try std.testing.expect(c_api.value_is_float(val));
    try std.testing.expect(c_api.value_get_float(val) == 3.14);
}

test "c_api: reader read nil" {
    const reader = c_api.reader_create();
    defer c_api.reader_destroy(reader);

    const val = c_api.reader_read_string(reader, "nil");
    try std.testing.expect(val != null);
    try std.testing.expect(c_api.value_is_nil(val));
}

test "c_api: list operations" {
    const reader = c_api.reader_create();
    defer c_api.reader_destroy(reader);

    const val = c_api.reader_read_string(reader, "(1 2 3)");
    try std.testing.expect(val != null);
    try std.testing.expect(c_api.value_is_list(val));
    try std.testing.expect(!c_api.list_is_empty(val));
    try std.testing.expect(c_api.list_length(val) == 3);

    // First element
    const first = c_api.list_first(val);
    try std.testing.expect(first != null);
    try std.testing.expect(c_api.value_is_int(first));
    try std.testing.expect(c_api.value_get_int(first) == 1);

    // Rest
    const rest = c_api.list_rest(val);
    try std.testing.expect(rest != null);
    try std.testing.expect(c_api.value_is_list(rest));
    try std.testing.expect(c_api.list_length(rest) == 2);

    // Nth element
    const second = c_api.list_nth(val, 1);
    try std.testing.expect(second != null);
    try std.testing.expect(c_api.value_get_int(second) == 2);

    const third = c_api.list_nth(val, 2);
    try std.testing.expect(third != null);
    try std.testing.expect(c_api.value_get_int(third) == 3);
}

test "c_api: list empty and cons" {
    const empty = c_api.list_empty();
    try std.testing.expect(empty != null);
    try std.testing.expect(c_api.value_is_list(empty));
    try std.testing.expect(c_api.list_is_empty(empty));
    try std.testing.expect(c_api.list_length(empty) == 0);

    // Cons a value
    const val = c_api.value_create_int(42);
    const list = c_api.list_cons(val, empty);
    try std.testing.expect(list != null);
    try std.testing.expect(c_api.list_length(list) == 1);

    const first = c_api.list_first(list);
    try std.testing.expect(c_api.value_get_int(first) == 42);
}

test "c_api: vector operations" {
    const reader = c_api.reader_create();
    defer c_api.reader_destroy(reader);

    const val = c_api.reader_read_string(reader, "[1 2 3]");
    try std.testing.expect(val != null);
    try std.testing.expect(c_api.value_is_vector(val));
    try std.testing.expect(!c_api.vector_is_empty(val));
    try std.testing.expect(c_api.vector_length(val) == 3);

    // Access elements
    const first = c_api.vector_nth(val, 0);
    try std.testing.expect(first != null);
    try std.testing.expect(c_api.value_get_int(first) == 1);

    const second = c_api.vector_nth(val, 1);
    try std.testing.expect(second != null);
    try std.testing.expect(c_api.value_get_int(second) == 2);

    const third = c_api.vector_nth(val, 2);
    try std.testing.expect(third != null);
    try std.testing.expect(c_api.value_get_int(third) == 3);
}

test "c_api: vector empty and push" {
    const empty = c_api.vector_empty();
    try std.testing.expect(empty != null);
    try std.testing.expect(c_api.value_is_vector(empty));
    try std.testing.expect(c_api.vector_is_empty(empty));
    try std.testing.expect(c_api.vector_length(empty) == 0);

    // Push a value
    const val = c_api.value_create_int(42);
    const vec1 = c_api.vector_push(empty, val);
    try std.testing.expect(vec1 != null);
    try std.testing.expect(c_api.vector_length(vec1) == 1);

    const first = c_api.vector_nth(vec1, 0);
    try std.testing.expect(c_api.value_get_int(first) == 42);

    // Push another value
    const val2 = c_api.value_create_int(100);
    const vec2 = c_api.vector_push(vec1, val2);
    try std.testing.expect(c_api.vector_length(vec2) == 2);

    // Pop a value
    const vec3 = c_api.vector_pop(vec2);
    try std.testing.expect(c_api.vector_length(vec3) == 1);
}

test "c_api: map operations" {
    const reader = c_api.reader_create();
    defer c_api.reader_destroy(reader);

    const val = c_api.reader_read_string(reader, "{:a 1 :b 2}");
    try std.testing.expect(val != null);
    try std.testing.expect(c_api.value_is_map(val));
    try std.testing.expect(c_api.map_count(val) == 2);

    // Create keys to look up
    const key_a = c_api.value_create_keyword("a");
    const key_b = c_api.value_create_keyword("b");

    // Get values
    const val_a = c_api.map_get(val, key_a);
    try std.testing.expect(val_a != null);
    try std.testing.expect(c_api.value_get_int(val_a) == 1);

    const val_b = c_api.map_get(val, key_b);
    try std.testing.expect(val_b != null);
    try std.testing.expect(c_api.value_get_int(val_b) == 2);

    // Non-existent key
    const key_c = c_api.value_create_keyword("c");
    const val_c = c_api.map_get(val, key_c);
    try std.testing.expect(val_c == null);
}

test "c_api: map empty and set" {
    const empty = c_api.map_empty();
    try std.testing.expect(empty != null);
    try std.testing.expect(c_api.value_is_map(empty));
    try std.testing.expect(c_api.map_count(empty) == 0);

    // Set a value
    const key = c_api.value_create_keyword("test");
    const val = c_api.value_create_int(42);
    const map1 = c_api.map_set(empty, key, val);
    try std.testing.expect(map1 != null);
    try std.testing.expect(c_api.map_count(map1) == 1);

    // Get the value back
    const retrieved = c_api.map_get(map1, key);
    try std.testing.expect(retrieved != null);
    try std.testing.expect(c_api.value_get_int(retrieved) == 42);
}

test "c_api: map iterator" {
    const reader = c_api.reader_create();
    defer c_api.reader_destroy(reader);

    const val = c_api.reader_read_string(reader, "{:x 10 :y 20}");
    try std.testing.expect(val != null);

    const iter = c_api.map_iterator_create(val);
    try std.testing.expect(iter != null);
    defer c_api.map_iterator_destroy(iter);

    var count: usize = 0;
    var key: ?*c_api.Value = null;
    var value: ?*c_api.Value = null;

    while (c_api.map_iterator_next(iter, &key, &value)) {
        try std.testing.expect(key != null);
        try std.testing.expect(value != null);
        try std.testing.expect(c_api.value_is_keyword(key));
        try std.testing.expect(c_api.value_is_int(value));
        count += 1;
    }

    try std.testing.expect(count == 2);
}

test "c_api: value creation" {
    // Int
    const int_val = c_api.value_create_int(123);
    try std.testing.expect(int_val != null);
    try std.testing.expect(c_api.value_is_int(int_val));
    try std.testing.expect(c_api.value_get_int(int_val) == 123);

    // Float
    const float_val = c_api.value_create_float(4.56);
    try std.testing.expect(float_val != null);
    try std.testing.expect(c_api.value_is_float(float_val));
    try std.testing.expect(c_api.value_get_float(float_val) == 4.56);

    // String
    const str_val = c_api.value_create_string("hello");
    try std.testing.expect(str_val != null);
    try std.testing.expect(c_api.value_is_string(str_val));

    // Symbol
    const sym_val = c_api.value_create_symbol("foo");
    try std.testing.expect(sym_val != null);
    try std.testing.expect(c_api.value_is_symbol(sym_val));

    // Keyword
    const kw_val = c_api.value_create_keyword("bar");
    try std.testing.expect(kw_val != null);
    try std.testing.expect(c_api.value_is_keyword(kw_val));

    // Nil
    const nil_val = c_api.value_create_nil();
    try std.testing.expect(nil_val != null);
    try std.testing.expect(c_api.value_is_nil(nil_val));
}

test "c_api: value to string" {
    const int_val = c_api.value_create_int(42);
    var buf: [256]u8 = undefined;
    const len = c_api.value_to_string(int_val, &buf, buf.len);
    try std.testing.expect(len > 0);
    try std.testing.expect(std.mem.eql(u8, buf[0..@intCast(len)], "42"));

    const reader = c_api.reader_create();
    defer c_api.reader_destroy(reader);

    const list_val = c_api.reader_read_string(reader, "(+ 1 2)");
    const len2 = c_api.value_to_string(list_val, &buf, buf.len);
    try std.testing.expect(len2 > 0);
    try std.testing.expect(std.mem.eql(u8, buf[0..@intCast(len2)], "(+ 1 2)"));
}

test "c_api: complex nested structure" {
    const reader = c_api.reader_create();
    defer c_api.reader_destroy(reader);

    const val = c_api.reader_read_string(reader, "(defn add [x y] (+ x y))");
    try std.testing.expect(val != null);
    try std.testing.expect(c_api.value_is_list(val));
    try std.testing.expect(c_api.list_length(val) == 4);

    // First element should be symbol "defn"
    const first = c_api.list_first(val);
    try std.testing.expect(c_api.value_is_symbol(first));

    // Third element should be a vector [x y]
    const params = c_api.list_nth(val, 2);
    try std.testing.expect(params != null);
    try std.testing.expect(c_api.value_is_vector(params));
    try std.testing.expect(c_api.vector_length(params) == 2);

    // Fourth element should be a list (+ x y)
    const body = c_api.list_nth(val, 3);
    try std.testing.expect(body != null);
    try std.testing.expect(c_api.value_is_list(body));
}
