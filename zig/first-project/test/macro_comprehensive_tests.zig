const std = @import("std");
const SimpleCCompiler = @import("simple_c_compiler.zig").SimpleCCompiler;

test "macro integration - simple identity macro" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    var allocator = arena.allocator();

    const source =
        \\(defmacro identity [x] x)
        \\(def result (: Int) (identity 42))
        \\result
    ;

    var compiler = SimpleCCompiler.init(&allocator);
    defer compiler.deinit();

    const c_code = try compiler.compileString(source, .executable);
    _ = c_code; // Just verify it compiles without error
}

test "macro integration - unless macro" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    var allocator = arena.allocator();

    const source =
        \\(defmacro unless [condition body]
        \\  (if condition 0 body))
        \\(def result1 (: Int) (unless false 42))
        \\(def result2 (: Int) (unless true 99))
        \\(+ result1 result2)
    ;

    var compiler = SimpleCCompiler.init(&allocator);
    defer compiler.deinit();

    const c_code = try compiler.compileString(source, .executable);
    _ = c_code;
}

test "macro integration - multiple macros" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    var allocator = arena.allocator();

    const source =
        \\(defmacro identity [x] x)
        \\(defmacro double [x] (+ x x))
        \\(def a (: Int) (identity 5))
        \\(def b (: Int) (double 10))
        \\(+ a b)
    ;

    var compiler = SimpleCCompiler.init(&allocator);
    defer compiler.deinit();

    const c_code = try compiler.compileString(source, .executable);
    _ = c_code;
}

test "macro integration - macro using another macro" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    var allocator = arena.allocator();

    const source =
        \\(defmacro double [x] (+ x x))
        \\(defmacro quadruple [x] (double (double x)))
        \\(def result (: Int) (quadruple 3))
        \\result
    ;

    var compiler = SimpleCCompiler.init(&allocator);
    defer compiler.deinit();

    const c_code = try compiler.compileString(source, .executable);
    _ = c_code;
}

test "macro integration - nested expansion" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    var allocator = arena.allocator();

    const source =
        \\(defmacro add1 [x] (+ x 1))
        \\(def result (: Int) (add1 (add1 (add1 10))))
        \\result
    ;

    var compiler = SimpleCCompiler.init(&allocator);
    defer compiler.deinit();

    const c_code = try compiler.compileString(source, .executable);
    _ = c_code;
}

test "macro integration - macro with arithmetic" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    var allocator = arena.allocator();

    const source =
        \\(defmacro square [x] (* x x))
        \\(def result (: Int) (square 7))
        \\result
    ;

    var compiler = SimpleCCompiler.init(&allocator);
    defer compiler.deinit();

    const c_code = try compiler.compileString(source, .executable);
    _ = c_code;
}

test "macroexpand - single step expansion" {
    const MacroExpander = @import("macro_expander.zig").MacroExpander;
    const Reader = @import("reader.zig").Reader;

    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const allocator = arena.allocator();

    var expander = MacroExpander.init(allocator);
    defer expander.deinit();

    // Define a simple macro
    const macro_source = "(defmacro add1 [x] (+ x 1))";
    var alloc_ptr = allocator;
    var macro_reader = Reader.init(&alloc_ptr);
    const macro_expr = try macro_reader.readString(macro_source);
    _ = try expander.expand(macro_expr);

    // Test expandOne on a macro call
    const call_source = "(add1 5)";
    var call_reader = Reader.init(&alloc_ptr);
    const call_expr = try call_reader.readString(call_source);

    const expanded = try expander.expandOne(call_expr);

    // Should expand to (+ 5 1) but not evaluate further
    try std.testing.expect(expanded.isList());
    const list = expanded.list;
    try std.testing.expect(!list.isEmpty());
    const first = list.value.?;
    try std.testing.expect(first.isSymbol());
    try std.testing.expectEqualStrings("+", first.symbol);
}

test "macroexpand - non-macro returns unchanged" {
    const MacroExpander = @import("macro_expander.zig").MacroExpander;
    const Reader = @import("reader.zig").Reader;

    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const allocator = arena.allocator();

    var expander = MacroExpander.init(allocator);
    defer expander.deinit();

    // Test expandOne on non-macro
    const source = "(+ 1 2)";
    var alloc_ptr = allocator;
    var reader = Reader.init(&alloc_ptr);
    const expr = try reader.readString(source);

    const expanded = try expander.expandOne(expr);

    // Should return unchanged
    try std.testing.expect(expanded == expr);
}

test "macroexpand-all - recursive expansion" {
    const MacroExpander = @import("macro_expander.zig").MacroExpander;
    const Reader = @import("reader.zig").Reader;

    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const allocator = arena.allocator();

    var expander = MacroExpander.init(allocator);
    defer expander.deinit();

    var alloc_ptr = allocator;

    // Define two macros that use each other
    const macro1_source = "(defmacro add1 [x] (+ x 1))";
    var macro1_reader = Reader.init(&alloc_ptr);
    const macro1_expr = try macro1_reader.readString(macro1_source);
    _ = try expander.expand(macro1_expr);

    const macro2_source = "(defmacro add2 [x] (add1 (add1 x)))";
    var macro2_reader = Reader.init(&alloc_ptr);
    const macro2_expr = try macro2_reader.readString(macro2_source);
    _ = try expander.expand(macro2_expr);

    // Test expandAll on nested macro call
    const call_source = "(add2 5)";
    var call_reader = Reader.init(&alloc_ptr);
    const call_expr = try call_reader.readString(call_source);

    const expanded = try expander.expandAll(call_expr);

    // Should fully expand to (+ (+ 5 1) 1)
    try std.testing.expect(expanded.isList());
    const outer_list = expanded.list;
    const outer_first = outer_list.value.?;
    try std.testing.expect(outer_first.isSymbol());
    try std.testing.expectEqualStrings("+", outer_first.symbol);

    // Check inner expression is also (+ ...)
    const second_elem = outer_list.next.?.value.?;
    try std.testing.expect(second_elem.isList());
    const inner_list = second_elem.list;
    const inner_first = inner_list.value.?;
    try std.testing.expect(inner_first.isSymbol());
    try std.testing.expectEqualStrings("+", inner_first.symbol);
}

test "macroexpand vs macroexpand-all - difference" {
    const MacroExpander = @import("macro_expander.zig").MacroExpander;
    const Reader = @import("reader.zig").Reader;

    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const allocator = arena.allocator();

    var expander = MacroExpander.init(allocator);
    defer expander.deinit();

    var alloc_ptr = allocator;

    // Define nested macros
    const macro1_source = "(defmacro inner [x] (+ x 1))";
    var macro1_reader = Reader.init(&alloc_ptr);
    const macro1_expr = try macro1_reader.readString(macro1_source);
    _ = try expander.expand(macro1_expr);

    const macro2_source = "(defmacro outer [x] (inner x))";
    var macro2_reader = Reader.init(&alloc_ptr);
    const macro2_expr = try macro2_reader.readString(macro2_source);
    _ = try expander.expand(macro2_expr);

    // Test call
    const call_source = "(outer 10)";
    var call_reader = Reader.init(&alloc_ptr);
    const call_expr = try call_reader.readString(call_source);

    // expandOne should expand to (inner 10)
    const expanded_once = try expander.expandOne(call_expr);
    try std.testing.expect(expanded_once.isList());
    const once_first = expanded_once.list.value.?;
    try std.testing.expect(once_first.isSymbol());
    try std.testing.expectEqualStrings("inner", once_first.symbol);

    // expandAll should expand to (+ 10 1)
    var call_reader2 = Reader.init(&alloc_ptr);
    const call_expr2 = try call_reader2.readString(call_source);
    const expanded_all = try expander.expandAll(call_expr2);
    try std.testing.expect(expanded_all.isList());
    const all_first = expanded_all.list.value.?;
    try std.testing.expect(all_first.isSymbol());
    try std.testing.expectEqualStrings("+", all_first.symbol);
}
