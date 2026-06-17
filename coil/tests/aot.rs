//! Ahead-of-time compilation: source → object → linked native executable.
//! These run the produced binary and check its process exit code (low 8 bits of
//! the i64 `main` returns), so they prove the *real* output path, not the JIT.

use std::path::PathBuf;
use std::process::Command;
use std::sync::atomic::{AtomicU32, Ordering};

static N: AtomicU32 = AtomicU32::new(0);

fn unique_exe(tag: &str) -> PathBuf {
    let n = N.fetch_add(1, Ordering::Relaxed);
    std::env::temp_dir().join(format!("coil_aot_{}_{}_{}", tag, std::process::id(), n))
}

fn build_and_run(src: &str, tag: &str) -> i32 {
    let exe = unique_exe(tag);
    coil::build_executable(src, &exe).expect("build_executable");
    let status = Command::new(&exe).status().expect("run executable");
    let _ = std::fs::remove_file(&exe);
    status.code().expect("exit code")
}

#[test]
fn aot_arithmetic() {
    let src = "(defn main [] (-> :i64) (iadd 40 2))";
    assert_eq!(build_and_run(src, "arith"), 42);
}

#[test]
fn aot_shim_trampoline_natively_linked() {
    // The hard case: a calling convention LLVM can't express, compiled to a
    // native object and resolved by the real assembler + linker.
    let src = include_str!("../examples/shim.coil");
    assert_eq!(build_and_run(src, "shim"), 42);
}

#[test]
fn aot_heap_allocation_links_libc() {
    // malloc/free are resolved at link time against libc.
    let src = include_str!("../examples/allocation.coil");
    assert_eq!(build_and_run(src, "alloc"), 42);
}

#[test]
fn aot_object_file_is_emitted() {
    let obj = unique_exe("obj").with_extension("o");
    coil::compile_to_object("(defn main [] (-> :i64) 0)", &obj).expect("compile_to_object");
    let meta = std::fs::metadata(&obj).expect("object exists");
    assert!(meta.len() > 0, "object file should be non-empty");
    let _ = std::fs::remove_file(&obj);
}
