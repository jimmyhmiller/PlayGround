mod ast;
mod codegen;
mod runtime;

use inkwell::context::Context;
use inkwell::execution_engine::JitFunction;
use inkwell::OptimizationLevel;
use std::ffi::c_void;

use ast::{Expr, FunctionDef, Parameter, Program, Stmt, StructDef, StructField, Type};
use codegen::Compiler;
use runtime::{Heap, ObjectHeader, Thread};
use runtime::object::ObjectMeta;

/// Build the binary trees benchmark program as an AST
fn build_binary_trees_program() -> Program {
    // struct TreeNode { left: TreeNode, right: TreeNode }
    let tree_node = StructDef::new(
        "TreeNode",
        vec![
            StructField {
                name: "left".to_string(),
                typ: Type::Struct("TreeNode".to_string()),
            },
            StructField {
                name: "right".to_string(),
                typ: Type::Struct("TreeNode".to_string()),
            },
        ],
    );

    // fn bottomUpTree(depth: i64) -> TreeNode
    // Creates a binary tree of the given depth
    let bottom_up_tree = FunctionDef::new(
        "bottomUpTree",
        vec![Parameter::new("depth", Type::I64)],
        Type::Struct("TreeNode".to_string()),
        vec![
            // let node: TreeNode = new TreeNode
            Stmt::let_decl("node", Type::Struct("TreeNode".to_string()), Some(Expr::new_struct("TreeNode"))),
            // if depth > 0
            Stmt::if_stmt(
                Expr::gt(Expr::var("depth"), Expr::int(0)),
                vec![
                    // node.left = bottomUpTree(depth - 1)
                    Stmt::field_set(
                        Expr::var("node"),
                        "TreeNode",
                        "left",
                        Expr::call("bottomUpTree", vec![Expr::sub(Expr::var("depth"), Expr::int(1))]),
                    ),
                    // node.right = bottomUpTree(depth - 1)
                    Stmt::field_set(
                        Expr::var("node"),
                        "TreeNode",
                        "right",
                        Expr::call("bottomUpTree", vec![Expr::sub(Expr::var("depth"), Expr::int(1))]),
                    ),
                ],
                vec![],
            ),
            // return node
            Stmt::ret(Some(Expr::var("node"))),
        ],
    );

    // fn itemCheck(node: TreeNode) -> i64
    // Counts nodes in the tree
    let item_check = FunctionDef::new(
        "itemCheck",
        vec![Parameter::new("node", Type::Struct("TreeNode".to_string()))],
        Type::I64,
        vec![
            // let left: TreeNode = node.left
            Stmt::let_decl(
                "left",
                Type::Struct("TreeNode".to_string()),
                Some(Expr::field_get(Expr::var("node"), "TreeNode", "left")),
            ),
            // if left == null
            Stmt::if_stmt(
                Expr::eq(Expr::var("left"), Expr::Null),
                vec![
                    // return 1
                    Stmt::ret(Some(Expr::int(1))),
                ],
                vec![],
            ),
            // return 1 + itemCheck(left) + itemCheck(node.right)
            Stmt::ret(Some(Expr::add(
                Expr::int(1),
                Expr::add(
                    Expr::call("itemCheck", vec![Expr::var("left")]),
                    Expr::call(
                        "itemCheck",
                        vec![Expr::field_get(Expr::var("node"), "TreeNode", "right")],
                    ),
                ),
            ))),
        ],
    );

    // fn work(iterations: i64, depth: i64) -> i64
    // Create many trees and sum their checks
    let work = FunctionDef::new(
        "work",
        vec![
            Parameter::new("iterations", Type::I64),
            Parameter::new("depth", Type::I64),
        ],
        Type::I64,
        vec![
            // let check: i64 = 0
            Stmt::let_decl("check", Type::I64, Some(Expr::int(0))),
            // let i: i64 = 0
            Stmt::let_decl("i", Type::I64, Some(Expr::int(0))),
            // while i < iterations
            Stmt::while_loop(
                Expr::lt(Expr::var("i"), Expr::var("iterations")),
                vec![
                    // let tree: TreeNode = bottomUpTree(depth)
                    Stmt::let_decl(
                        "tree",
                        Type::Struct("TreeNode".to_string()),
                        Some(Expr::call("bottomUpTree", vec![Expr::var("depth")])),
                    ),
                    // check = check + itemCheck(tree)
                    Stmt::assign(
                        "check",
                        Expr::add(
                            Expr::var("check"),
                            Expr::call("itemCheck", vec![Expr::var("tree")]),
                        ),
                    ),
                    // i = i + 1
                    Stmt::assign("i", Expr::add(Expr::var("i"), Expr::int(1))),
                ],
            ),
            // return check
            Stmt::ret(Some(Expr::var("check"))),
        ],
    );

    // fn main() -> i64
    // Entry point
    let main_fn = FunctionDef::new(
        "main",
        vec![],
        Type::I64,
        vec![
            // let maxDepth: i64 = 10
            Stmt::let_decl("maxDepth", Type::I64, Some(Expr::int(10))),
            // let stretchDepth: i64 = maxDepth + 1
            Stmt::let_decl(
                "stretchDepth",
                Type::I64,
                Some(Expr::add(Expr::var("maxDepth"), Expr::int(1))),
            ),
            // Build stretch tree
            Stmt::let_decl(
                "stretchTree",
                Type::Struct("TreeNode".to_string()),
                Some(Expr::call("bottomUpTree", vec![Expr::var("stretchDepth")])),
            ),
            // let stretchCheck: i64 = itemCheck(stretchTree)
            Stmt::let_decl(
                "stretchCheck",
                Type::I64,
                Some(Expr::call("itemCheck", vec![Expr::var("stretchTree")])),
            ),
            // Build long-lived tree
            Stmt::let_decl(
                "longLivedTree",
                Type::Struct("TreeNode".to_string()),
                Some(Expr::call("bottomUpTree", vec![Expr::var("maxDepth")])),
            ),
            // let depth: i64 = 4
            Stmt::let_decl("depth", Type::I64, Some(Expr::int(4))),
            // let total: i64 = 0
            Stmt::let_decl("total", Type::I64, Some(Expr::int(0))),
            // while depth <= maxDepth
            Stmt::while_loop(
                Expr::le(Expr::var("depth"), Expr::var("maxDepth")),
                vec![
                    // let iterations: i64 = 1 << (maxDepth - depth + 4)
                    // We don't have shift, so compute via repeated multiplication
                    // iterations = 16 * 2^(maxDepth - depth) = 16384 >> depth (when maxDepth=10)
                    // Actually, let's just use a simpler formula for demo:
                    // iterations = 64 (fixed for simplicity, or we could add shift operator)
                    Stmt::let_decl("iterations", Type::I64, Some(Expr::int(64))),
                    // let checksum: i64 = work(iterations, depth)
                    Stmt::let_decl(
                        "checksum",
                        Type::I64,
                        Some(Expr::call(
                            "work",
                            vec![Expr::var("iterations"), Expr::var("depth")],
                        )),
                    ),
                    // total = total + checksum
                    Stmt::assign(
                        "total",
                        Expr::add(Expr::var("total"), Expr::var("checksum")),
                    ),
                    // depth = depth + 2
                    Stmt::assign("depth", Expr::add(Expr::var("depth"), Expr::int(2))),
                ],
            ),
            // let longLivedCheck: i64 = itemCheck(longLivedTree)
            Stmt::let_decl(
                "longLivedCheck",
                Type::I64,
                Some(Expr::call("itemCheck", vec![Expr::var("longLivedTree")])),
            ),
            // return total + longLivedCheck + stretchCheck
            Stmt::ret(Some(Expr::add(
                Expr::var("total"),
                Expr::add(Expr::var("longLivedCheck"), Expr::var("stretchCheck")),
            ))),
        ],
    );

    Program::new(
        vec![tree_node],
        vec![bottom_up_tree, item_check, work, main_fn],
    )
}

// FFI functions that the generated code calls

use std::sync::atomic::{AtomicUsize, Ordering};

static ALLOCATION_COUNT: AtomicUsize = AtomicUsize::new(0);
static TRIGGER_AT: AtomicUsize = AtomicUsize::new(100); // Trigger stack dump at 100th allocation

#[no_mangle]
pub extern "C" fn gc_pollcheck_slow(thread: *mut Thread, _origin: *const c_void) {
    // This is where your GC would hook in
    let thread = unsafe { &mut *thread };

    // Clear the check flag
    thread.state = 0;

    println!("\n=== GC Safepoint Reached ===");
    println!("Stack walking to find all GC roots:");
    thread.dump_stack();

    let roots = thread.collect_roots();
    println!("Total roots found: {}", roots.len());

    // In a real GC, you would:
    // 1. Mark all roots from the stack
    // 2. Trace the object graph
    // 3. Sweep unmarked objects
    println!("=== End Safepoint ===\n");
}

#[no_mangle]
pub extern "C" fn gc_allocate(
    thread: *mut Thread,
    meta: *const ObjectMeta,
    payload_size: u64,
) -> *mut ObjectHeader {
    let thread = unsafe { &mut *thread };
    let heap = unsafe { &mut *(thread.heap as *mut Heap) };

    let count = ALLOCATION_COUNT.fetch_add(1, Ordering::SeqCst);

    // Trigger a pollcheck at certain allocation counts to demonstrate stack walking
    if count == TRIGGER_AT.load(Ordering::SeqCst) {
        // Set the check requested flag - next pollcheck will trigger
        thread.state = runtime::thread::THREAD_STATE_CHECK_REQUESTED;
        println!("\n*** Triggering GC safepoint after {} allocations ***", count);
    }

    heap.allocate(meta, payload_size as usize)
}

#[no_mangle]
pub extern "C" fn gc_allocate_array(
    thread: *mut Thread,
    meta: *const ObjectMeta,
    length: u64,
) -> *mut ObjectHeader {
    let thread = unsafe { &mut *thread };
    let heap = unsafe { &mut *(thread.heap as *mut Heap) };
    heap.allocate_array(meta, length as u32)
}

fn main() {
    println!("Building binary trees program...");
    let program = build_binary_trees_program();

    println!("Compiling to LLVM IR...");
    let context = Context::create();
    let compiler = Compiler::new(&context, program);
    compiler.compile();

    println!("\n=== Generated LLVM IR ===\n");
    compiler.print_ir();

    println!("\n=== Running with JIT ===\n");

    // Create execution engine
    let execution_engine = compiler
        .module
        .create_jit_execution_engine(OptimizationLevel::None)
        .unwrap();

    // Add symbol mappings for runtime functions
    execution_engine.add_global_mapping(
        &compiler.runtime.pollcheck_slow,
        gc_pollcheck_slow as usize,
    );
    execution_engine.add_global_mapping(
        &compiler.runtime.allocate,
        gc_allocate as usize,
    );
    execution_engine.add_global_mapping(
        &compiler.runtime.allocate_array,
        gc_allocate_array as usize,
    );

    // Set up runtime
    let mut heap = Heap::new();
    let heap_ptr = &mut *heap as *mut Heap as *mut c_void;
    let mut thread = Thread::new(heap_ptr);

    // Get the main function
    type MainFn = unsafe extern "C" fn(*mut Thread) -> i64;
    let main_fn: JitFunction<MainFn> = unsafe {
        execution_engine.get_function("main").unwrap()
    };

    // Run it!
    println!("Calling main()...");
    let result = unsafe { main_fn.call(&mut thread) };

    println!("\nResult: {}", result);
    println!("Heap allocated {} bytes across {} objects",
             heap.bytes_allocated(),
             heap.objects().len());

    // Demonstrate stack walking one more time
    println!("\n=== Final stack state ===");
    thread.dump_stack();

    // Show we can enumerate all roots
    println!("\n=== All roots from stack ===");
    let roots = thread.collect_roots();
    println!("Found {} roots on the stack", roots.len());
    for (i, root) in roots.iter().enumerate() {
        println!("  root[{}]: {:p}", i, *root);
    }
}
