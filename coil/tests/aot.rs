//! Ahead-of-time compilation: source → object → linked native executable.
//! No JIT is involved anywhere in this path.

mod common;
use common::{build_and_run, unique_path};

#[test]
fn aot_arithmetic() {
    assert_eq!(build_and_run("(defn main [] (-> :i64) (iadd 40 2))"), 42);
}

#[test]
fn aot_shim_trampoline_natively_linked() {
    // A calling convention LLVM can't express, compiled to a native object and
    // resolved by the real assembler + linker.
    assert_eq!(build_and_run(include_str!("../examples/shim.coil")), 42);
}

#[test]
fn aot_heap_allocation_links_libc() {
    // malloc/free resolved at link time against libc.
    assert_eq!(build_and_run(include_str!("../examples/allocation.coil")), 42);
}

#[test]
fn aot_expands_all_macros_then_compiles() {
    // The whole point: macros expand at compile time (a tree-walking interpreter,
    // no JIT), then the result AOT-compiles to a native binary.
    let src = r#"
        (defn when [(c Code) & (body Code)] (-> Code) `(if ~c (do ~@body) 0))
        (defn inc [(x Code)] (-> Code) `(iadd ~x 1))
        (defn pow-form [(x Code) (n i64)] (-> Code)
          (if (icmp-eq n 0) `1 `(imul ~x ~(pow-form x (isub n 1)))))
        (defn pow [(x Code) (n Code)] (-> Code) (pow-form x (code-int n)))
        (defn main [] (-> :i64)
          (when (icmp-eq 1 1)
            (iadd (pow 2 5) (inc 9))))  ; 32 + 10 = 42
    "#;
    assert_eq!(build_and_run(src), 42);
}

#[test]
fn aot_object_file_is_emitted() {
    let obj = unique_path("obj").with_extension("o");
    coil::compile_to_object("(defn main [] (-> :i64) 0)", &obj).expect("compile_to_object");
    let meta = std::fs::metadata(&obj).expect("object exists");
    assert!(meta.len() > 0, "object file should be non-empty");
    let _ = std::fs::remove_file(&obj);
}
