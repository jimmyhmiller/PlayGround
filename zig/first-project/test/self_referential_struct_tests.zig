const std = @import("std");
const SimpleCCompiler = @import("simple_c_compiler.zig").SimpleCCompiler;

test "self-referential struct - basic linked list node" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    var allocator = arena.allocator();

    const source =
        \\(def Node (: Type) (Struct [value Int] [next (Pointer Node)]))
        \\(def create-node (: (-> [Int] Node)) (fn [val] (Node val pointer-null)))
        \\create-node
    ;

    var compiler = SimpleCCompiler.init(&allocator);
    defer compiler.deinit();

    const c_code = try compiler.compileString(source, .executable);
    _ = c_code;
    // If we get here without error, the test passes
}

test "self-referential struct - binary tree node" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    var allocator = arena.allocator();

    const source =
        \\(def TreeNode (: Type) (Struct [value Int] [left (Pointer TreeNode)] [right (Pointer TreeNode)]))
        \\(def create-leaf (: (-> [Int] TreeNode)) (fn [val] (TreeNode val pointer-null pointer-null)))
        \\create-leaf
    ;

    var compiler = SimpleCCompiler.init(&allocator);
    defer compiler.deinit();

    const c_code = try compiler.compileString(source, .executable);
    _ = c_code;
}

test "self-referential struct - linked list with operations" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    var allocator = arena.allocator();

    const source =
        \\(def Node (: Type) (Struct [data Int] [next (Pointer Node)]))
        \\
        \\(def make-node (: (-> [Int (Pointer Node)] Node))
        \\  (fn [value next-ptr]
        \\    (Node value next-ptr)))
        \\
        \\(def get-data (: (-> [(Pointer Node)] Int))
        \\  (fn [node-ptr]
        \\    (if (pointer-equal? node-ptr pointer-null)
        \\      0
        \\      (. (dereference node-ptr) data))))
        \\
        \\(get-data pointer-null)
    ;

    var compiler = SimpleCCompiler.init(&allocator);
    defer compiler.deinit();

    const c_code = try compiler.compileString(source, .executable);
    _ = c_code;
}

test "self-referential struct - doubly linked list" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    var allocator = arena.allocator();

    const source =
        \\(def DNode (: Type) (Struct [value Int] [prev (Pointer DNode)] [next (Pointer DNode)]))
        \\(def make-dnode (: (-> [Int] DNode)) (fn [val] (DNode val pointer-null pointer-null)))
        \\make-dnode
    ;

    var compiler = SimpleCCompiler.init(&allocator);
    defer compiler.deinit();

    const c_code = try compiler.compileString(source, .executable);
    _ = c_code;
}

test "self-referential struct - complex nested struct" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    var allocator = arena.allocator();

    const source =
        \\(def Graph (: Type) (Struct [id Int] [neighbors (Pointer Graph)]))
        \\(def Data (: Type) (Struct [value Int] [graph Graph]))
        \\
        \\(def make-data (: (-> [Int Graph] Data))
        \\  (fn [val g]
        \\    (Data val g)))
        \\
        \\make-data
    ;

    var compiler = SimpleCCompiler.init(&allocator);
    defer compiler.deinit();

    const c_code = try compiler.compileString(source, .executable);
    _ = c_code;
}
