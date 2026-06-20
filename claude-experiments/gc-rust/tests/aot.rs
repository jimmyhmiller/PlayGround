//! End-to-end AOT tests: `gcr build` must produce a standalone native binary
//! that runs against the statically-linked GC runtime and exits with the
//! program's `i64` result truncated to the process exit code (`& 0xFF`).
//!
//! We drive the public `gcrust::codegen::build_executable` path directly (the
//! same one `gcr build` uses), link, run the binary, and assert its exit code.

use std::path::{Path, PathBuf};
use std::process::Command;

use gcrust::codegen::build_executable;
use gcrust::lexer::lex;
use gcrust::lower::lower_program;
use gcrust::parser::parse_module;
use gcrust::resolve::resolve_module;

/// Ensure the runtime staticlib (`libgcrust_rt.a`) exists so the AOT link can
/// find it. It is a `staticlib` artifact of the `gcrust-rt` crate, which a bare
/// `cargo test` of the main package does not build automatically — so build it
/// here and point the linker at it via `$GCRUST_RUNTIME_LIB`.
fn ensure_runtime_lib() -> PathBuf {
    let manifest = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let status = Command::new(env!("CARGO"))
        .args(["build", "-p", "gcrust-rt"])
        .current_dir(&manifest)
        .status()
        .expect("failed to run cargo build -p gcrust-rt");
    assert!(status.success(), "building gcrust-rt staticlib failed");
    let lib = manifest.join("target").join("debug").join("libgcrust_rt.a");
    assert!(lib.exists(), "libgcrust_rt.a not found at {}", lib.display());
    lib
}

/// Compile `src` to a native executable at `out`, returning nothing on success.
fn build(src: &str, out: &Path, runtime_lib: &Path) {
    // Safety: tests in this file run serially w.r.t. this var because each sets
    // it to the same path before building.
    unsafe {
        std::env::set_var("GCRUST_RUNTIME_LIB", runtime_lib);
    }
    let module = parse_module(&lex(src).unwrap()).unwrap();
    let resolved = resolve_module(module).unwrap();
    let prog = lower_program(&resolved.globals).unwrap();
    build_executable(&prog, out, &[]).expect("build_executable failed");
}

/// Run an executable and return its process exit code.
fn run_exit_code(bin: &Path) -> i32 {
    let status = Command::new(bin).status().expect("failed to run AOT binary");
    status.code().expect("AOT binary terminated by signal")
}

fn tmp(name: &str) -> PathBuf {
    let mut p = std::env::temp_dir();
    p.push(format!("gcrust_aot_test_{}_{}", std::process::id(), name));
    p
}

#[test]
fn aot_fib_exit_code() {
    let lib = ensure_runtime_lib();
    let out = tmp("fib");
    build(include_str!("../examples/fib.gcr"), &out, &lib);
    // fib(32) = 2178309; the process exit code is the low byte.
    assert_eq!(run_exit_code(&out), 2178309 & 0xFF);
    let _ = std::fs::remove_file(&out);
}

#[test]
fn aot_heap_struct_and_enum() {
    let lib = ensure_runtime_lib();
    let out = tmp("shapes");
    // Struct + enum-with-payload + match + GC heap allocation.
    build(include_str!("../examples/shapes.gcr"), &out, &lib);
    // 3*3*3 + 4*5 = 47.
    assert_eq!(run_exit_code(&out), 47);
    let _ = std::fs::remove_file(&out);
}

#[test]
fn aot_binary_trees_gc_under_load() {
    let lib = ensure_runtime_lib();
    let out = tmp("binary_trees");
    build(include_str!("../examples/binary_trees.gcr"), &out, &lib);
    // checksum 5242840; low byte = 216. Proves the GC runs under load in the
    // AOT-linked binary without crashing.
    assert_eq!(run_exit_code(&out), 5242840 & 0xFF);
    let _ = std::fs::remove_file(&out);
}
