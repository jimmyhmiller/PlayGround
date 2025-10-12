const std = @import("std");
const SimpleCCompiler = @import("simple_c_compiler.zig").SimpleCCompiler;

test "macro with expansion-time let - simple binding" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    var allocator = arena.allocator();

    const source =
        \\(defmacro simple-let [x]
        \\  (let [y x]
        \\    `(+ ~y 1)))
        \\(def result (: Int) (simple-let 5))
        \\result
    ;

    var compiler = SimpleCCompiler.init(&allocator);
    defer compiler.deinit();

    const c_code = try compiler.compileString(source, .executable);
    _ = c_code;
    // Macro should expand to (+ 5 1), not (let [y 5] (+ y 1))
}

test "macro with expansion-time let - gensym binding" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    var allocator = arena.allocator();

    const source =
        \\(defmacro with-temp [val expr]
        \\  (let [tmp (gensym "temp")]
        \\    `(let [~tmp (: Int) ~val] ~expr)))
        \\
        \\(def increment (: (-> [Int] Int))
        \\  (fn [x] (+ x 1)))
        \\
        \\(def result (: Int)
        \\  (with-temp 42 (increment 10)))
        \\result
    ;

    var compiler = SimpleCCompiler.init(&allocator);
    defer compiler.deinit();

    const c_code = try compiler.compileString(source, .executable);
    _ = c_code;
    // The tmp binding should be evaluated at expansion time,
    // generating a unique symbol that's used in the runtime let
}

test "macro with expansion-time let - multiple bindings" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    var allocator = arena.allocator();

    const source =
        \\(defmacro double-add [a b]
        \\  (let [x a]
        \\    (let [y b]
        \\      `(+ ~x ~y ~x ~y))))
        \\
        \\(def result (: Int) (double-add 10 20))
        \\result
    ;

    var compiler = SimpleCCompiler.init(&allocator);
    defer compiler.deinit();

    const c_code = try compiler.compileString(source, .executable);
    _ = c_code;
    // Should expand to (+ 10 20 10 20)
}

test "macro with expansion-time let - sequential bindings in one let" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    var allocator = arena.allocator();

    const source =
        \\(defmacro seq-let [x]
        \\  (let [a x b (gensym "var")]
        \\    `(let [~b (: Int) ~a] (+ ~b 1))))
        \\
        \\(def result (: Int) (seq-let 99))
        \\result
    ;

    var compiler = SimpleCCompiler.init(&allocator);
    defer compiler.deinit();

    const c_code = try compiler.compileString(source, .executable);
    _ = c_code;
}

test "macro with expansion-time let - nested lets" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    var allocator = arena.allocator();

    const source =
        \\(defmacro nested-let [x]
        \\  (let [outer x]
        \\    (let [inner (gensym)]
        \\      `(+ ~outer 1))))
        \\
        \\(def result (: Int) (nested-let 7))
        \\result
    ;

    var compiler = SimpleCCompiler.init(&allocator);
    defer compiler.deinit();

    const c_code = try compiler.compileString(source, .executable);
    _ = c_code;
}

test "macro with expansion-time let - complex macro with gensym" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    var allocator = arena.allocator();

    const source =
        \\(defmacro swap [a b]
        \\  (let [tmp (gensym "swap")]
        \\    `(let [~tmp ~a]
        \\       (set! ~a ~b)
        \\       (set! ~b ~tmp))))
        \\
        \\(def result (: Int) 42)
        \\result
    ;

    var compiler = SimpleCCompiler.init(&allocator);
    defer compiler.deinit();

    const c_code = try compiler.compileString(source, .executable);
    _ = c_code;
    // The macro defines a swap operation that uses expansion-time let
    // to generate a unique temporary variable name
}

test "macro with expansion-time let - using let binding in quote" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    var allocator = arena.allocator();

    const source =
        \\(defmacro make-adder [n]
        \\  (let [num n]
        \\    `(fn [x] (+ x ~num))))
        \\
        \\(def add5 (: (-> [Int] Int)) (make-adder 5))
        \\(add5 10)
    ;

    var compiler = SimpleCCompiler.init(&allocator);
    defer compiler.deinit();

    const c_code = try compiler.compileString(source, .executable);
    _ = c_code;
}

test "macro with expansion-time let - computation in binding" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    var allocator = arena.allocator();

    const source =
        \\(defmacro double-it [x]
        \\  (let [sym1 (gensym "a") sym2 (gensym "b")]
        \\    `(+ ~x ~x)))
        \\
        \\(def result (: Int) (double-it 21))
        \\result
    ;

    var compiler = SimpleCCompiler.init(&allocator);
    defer compiler.deinit();

    const c_code = try compiler.compileString(source, .executable);
    _ = c_code;
}
