//! Threading tests (M2): real OS-thread spawn/join sharing the GC heap.
//!
//! The strongest correctness check is `multi_thread_alloc_under_stress`, which
//! runs three concurrently-allocating threads with `--gc-stress` (collect on
//! every allocation) — exercising stop-the-world coordination, cross-thread root
//! scanning, and relocation under maximal contention. See `docs/threads.md`.

use std::path::{Path, PathBuf};
use std::process::Command;

use gcrust::codegen::{build_executable, jit_run_i64, jit_run_i64_gc};
use gcrust::compile::parse_with_prelude;
use gcrust::lower::lower_program;
use gcrust::resolve::resolve_module;

fn prog_of(src: &str) -> gcrust::core::CoreProgram {
    let module = parse_with_prelude(src).unwrap();
    let resolved = resolve_module(module).unwrap();
    lower_program(&resolved.globals).unwrap()
}

#[test]
fn spawn_join_returns_result() {
    let src = r#"
        fn fib(n: i64) -> i64 { if n < 2 { n } else { fib(n - 1) + fib(n - 2) } }
        fn main() -> i64 {
            let t = Thread::spawn(|| fib(30));
            let local = fib(28);
            local + t.join()
        }
    "#;
    // fib(28)=317811, fib(30)=832040 -> 1149851
    assert_eq!(jit_run_i64(&prog_of(src)).unwrap(), 1149851);
}

#[test]
fn multi_thread_alloc_correct() {
    let src = include_str!("../examples/threads.gcr");
    assert_eq!(jit_run_i64(&prog_of(src)).unwrap(), 37492500);
}

#[test]
fn multi_thread_alloc_under_stress() {
    // Collect on every allocation across three concurrent mutators.
    let src = include_str!("../examples/threads.gcr");
    assert_eq!(jit_run_i64_gc(&prog_of(src), true).unwrap(), 37492500);
}

// Regression: a thread blocked in `join` must transition to BLOCKED so a GC
// triggered by the child can proceed. Before the fix this DEADLOCKED under
// stress (main parked in libc never reaches a safepoint). The child allocates
// heavily AFTER main has already entered join.
const JOIN_DEADLOCK_SRC: &str = r#"
    fn alloc_heavy(n: i64) -> i64 {
        let v: Vec<i64> = vec_new();
        let mut acc = v;
        let mut i = 0;
        while i < n { acc = vec_push(acc, i); i = i + 1; }
        vec_len(acc)
    }
    fn main() -> i64 {
        let t = Thread::spawn(|| alloc_heavy(20000));
        t.join()
    }
"#;

#[test]
fn join_does_not_deadlock_when_child_triggers_gc() {
    assert_eq!(jit_run_i64(&prog_of(JOIN_DEADLOCK_SRC)).unwrap(), 20000);
}

#[test]
fn join_does_not_deadlock_under_stress() {
    // The strongest form: child collects on every allocation while main blocks
    // in join. Deadlocked before the BLOCKED-state transition fix.
    assert_eq!(jit_run_i64_gc(&prog_of(JOIN_DEADLOCK_SRC), true).unwrap(), 20000);
}

// Regression: main holds a live Vec ACROSS join; a GC during the join (from the
// child) relocates it. main must publish its frame so the collector scans+moves
// the Vec, and read correct data on resume — not a dangling from-space pointer.
const JOIN_LIVE_ROOTS_SRC: &str = r#"
    fn alloc_heavy(n: i64) -> i64 {
        let v: Vec<i64> = vec_new();
        let mut acc = v;
        let mut i = 0;
        while i < n { acc = vec_push(acc, i); i = i + 1; }
        vec_len(acc)
    }
    fn main() -> i64 {
        let mine: Vec<i64> = vec_new();
        let mut m = mine;
        let mut k = 0;
        while k < 100 { m = vec_push(m, k * 2); k = k + 1; }
        let t = Thread::spawn(|| alloc_heavy(20000));
        let child = t.join();
        let mut s = 0;
        let mut j = 0;
        while j < vec_len(m) { s = s + vec_get(m, j); j = j + 1; }
        s + child
    }
"#;

#[test]
fn live_roots_survive_relocation_across_join() {
    // sum(m) = 2*(0..99) = 9900; child = 20000 -> 29900.
    assert_eq!(jit_run_i64_gc(&prog_of(JOIN_LIVE_ROOTS_SRC), true).unwrap(), 29900);
}

// Nested/concurrent spawns: drives concurrent env-handoff (globals.add/get/set)
// from multiple threads at once.
const NESTED_SRC: &str = r#"
    fn work(n: i64) -> i64 {
        let v: Vec<i64> = vec_new();
        let mut acc = v;
        let mut i = 0;
        while i < n { acc = vec_push(acc, i); i = i + 1; }
        vec_len(acc)
    }
    fn spawner(_x: i64) -> i64 {
        let a = Thread::spawn(|| work(2000));
        let b = Thread::spawn(|| work(2000));
        work(2000) + a.join() + b.join()
    }
    fn main() -> i64 {
        let s1 = Thread::spawn(|| spawner(0));
        let s2 = Thread::spawn(|| spawner(0));
        let s3 = Thread::spawn(|| spawner(0));
        s1.join() + s2.join() + s3.join()
    }
"#;

// Regression: a spawned closure that CAPTURES a GC pointer used to segfault —
// the closure's code-pointer offset was computed from the static (placeholder,
// ptr_fields=0) layout, so for a capturing closure (ptr_fields>0) the spawn read
// the code pointer from the wrong slot and jumped to garbage. Capture-free
// closures (all the other thread tests) happened to work. Fixed by recovering
// the code pointer from the env header's type_id at runtime.
// A spawned closure capturing an IMMUTABLE value struct (#[value]) is allowed —
// it's deeply immutable hence Sync. (This is the safe way to share read-only data
// across threads; capturing a mutable reference struct is correctly rejected, see
// the Send/Sync tests in shared_mem.rs.) Also the regression for the closure
// code-pointer bug: this closure captures a non-scalar value into a spawn.
#[test]
fn spawn_closure_capturing_value_struct() {
    let src = r#"
        #[value] struct Pt { x: i64, y: i64 }
        fn main() -> i64 {
            let p = Pt { x: 40, y: 2 };
            let t = Thread::spawn(|| p.x + p.y);
            t.join()
        }
    "#;
    assert_eq!(jit_run_i64(&prog_of(src)).unwrap(), 42);
}

#[test]
fn spawn_closure_capturing_value_struct_under_stress() {
    let src = r#"
        #[value] struct Pt { x: i64, y: i64 }
        fn use_it(p: Pt, iters: i64) -> i64 {
            let mut s = 0; let mut i = 0;
            while i < iters { s = s + p.x + p.y; i = i + 1; }
            s
        }
        fn main() -> i64 {
            let p = Pt { x: 1, y: 1 };
            let t = Thread::spawn(|| use_it(p, 2000));
            let l = use_it(p, 2000);
            l + t.join()
        }
    "#;
    assert_eq!(jit_run_i64_gc(&prog_of(src), true).unwrap(), 8000);
}

// Generic Thread<T>: a thread can return ANY type, not just i64. The result
// travels back via the handle's cell (Future model), kept alive by the GC.
#[test]
fn thread_returns_vec() {
    let src = r#"
        fn build(n: i64) -> Vec<i64> {
            let v: Vec<i64> = vec_new();
            let mut acc = v; let mut i = 0;
            while i < n { acc = vec_push(acc, i); i = i + 1; }
            acc
        }
        fn main() -> i64 {
            let t = Thread::spawn(|| build(100));
            let r = t.join();
            let mut s = 0; let mut j = 0;
            while j < vec_len(r) { s = s + vec_get(r, j); j = j + 1; }
            s
        }
    "#;
    assert_eq!(jit_run_i64(&prog_of(src)).unwrap(), 4950);
}

#[test]
fn thread_returns_vec_under_stress() {
    // The Vec result can be relocated in the finish->join window; the handle
    // keeps it rooted. Collect on every allocation.
    let src = r#"
        fn build(n: i64) -> Vec<i64> {
            let v: Vec<i64> = vec_new();
            let mut acc = v; let mut i = 0;
            while i < n { acc = vec_push(acc, i); i = i + 1; }
            acc
        }
        fn main() -> i64 {
            let t = Thread::spawn(|| build(80));
            let r = t.join();
            let mut s = 0; let mut j = 0;
            while j < vec_len(r) { s = s + vec_get(r, j); j = j + 1; }
            s
        }
    "#;
    assert_eq!(jit_run_i64_gc(&prog_of(src), true).unwrap(), 3160);
}

#[test]
fn thread_returns_struct() {
    let src = r#"
        struct Pair { a: i64, b: i64 }
        fn mk(x: i64) -> Pair { Pair { a: x, b: x * 2 } }
        fn main() -> i64 {
            let t1 = Thread::spawn(|| mk(10));
            let t2 = Thread::spawn(|| 7);
            let p = t1.join();
            p.a + p.b + t2.join()
        }
    "#;
    assert_eq!(jit_run_i64(&prog_of(src)).unwrap(), 37);
}

#[test]
fn nested_concurrent_spawns() {
    assert_eq!(jit_run_i64(&prog_of(NESTED_SRC)).unwrap(), 18000);
}

// --- AOT: the spawned threads run in a standalone binary too ---

fn ensure_runtime_lib() -> PathBuf {
    let manifest = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let status = Command::new(env!("CARGO"))
        .args(["build", "-p", "gcrust-rt"])
        .current_dir(&manifest)
        .status()
        .expect("failed to run cargo build -p gcrust-rt");
    assert!(status.success(), "building gcrust-rt staticlib failed");
    manifest.join("target").join("debug").join("libgcrust_rt.a")
}

fn run_exit_code(bin: &Path) -> i32 {
    Command::new(bin).status().expect("failed to run AOT binary")
        .code().expect("AOT binary terminated by signal")
}

#[test]
fn aot_spawn_join() {
    let lib = ensure_runtime_lib();
    unsafe { std::env::set_var("GCRUST_RUNTIME_LIB", &lib); }
    let mut out = std::env::temp_dir();
    out.push(format!("gcrust_threads_test_{}", std::process::id()));

    let src = include_str!("../examples/threads.gcr");
    build_executable(&prog_of(src), &out, &[]).expect("build_executable failed");
    // 37492500 & 0xFF = 0x... low byte
    assert_eq!(run_exit_code(&out), 37492500 & 0xFF);
    let _ = std::fs::remove_file(&out);
}
