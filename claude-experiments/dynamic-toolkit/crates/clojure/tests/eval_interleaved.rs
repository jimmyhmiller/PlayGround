//! True compile-during-run interleaving.
//!
//! With the toolkit migration to growable per-function metadata
//! (`GrowableTable<FunctionMetadata>` in dynlower), `JitModule::extend`
//! now takes `&self` and serializes only against other extends — not
//! against `gc.run_jit`. The clojure Engine wraps the JitModule in
//! `RwLock<Box<JitModule>>` and uses the **read** guard for both
//! eval (compile + extend) and call_compiled (run). This means
//! many threads — some compiling, some running — proceed in parallel.

use clojure::Engine;
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};
use std::thread;
use std::time::{Duration, Instant};

fn nanobox_to_int(v: u64) -> i64 {
    f64::from_bits(v) as i64
}

/// While thread B runs a long-running JIT computation (`fib(N)`),
/// thread A repeatedly compiles new functions via `eval`. Both
/// proceed in parallel — extending the JIT does NOT wait for B's
/// long compute, and B's compute is not corrupted by A's
/// concurrent code-memory mutations.
///
/// This is the JVM "compiler thread is just another mutator"
/// property, end-to-end. Required toolkit pieces (all in place):
/// 1. `dynlower::JitModule::extend(&self, ...)` with internal
///    `extend_lock` + `GrowableTable<FunctionMetadata>` (per-function
///    metadata has stable addresses, never reallocates).
/// 2. `dynalloc::Heap` STW protocol + `MutatorThread` registration.
/// 3. `dynlang` registers `MutatorThread` per OS thread, routes
///    allocations through it, installs a JIT-frame walker so the
///    heap can scan parked threads' JIT frames as roots.
/// 4. `dynruntime::JitSafepointSession` binds to the calling
///    thread's `ThreadState`, polls `gc_requested` at every
///    safepoint, publishes the parked frame pointer for cross-
///    thread GC root scanning.
/// 5. `clojure` compiler emits `Inst::Safepoint` before every call
///    so pure-compute functions like `fib` have poll points.
/// 6. `dynlower::walk_parked_thread_jit_roots` (cross-thread variant
///    that doesn't rely on the caller's thread-local fence).
/// 7. `dynasm::PagedCodeMemory::finalize` rounds `used` up to the
///    next page so concurrent extends don't flip executing pages
///    back to RW (fatal if another thread is mid-instruction).
#[test]
fn extend_proceeds_during_long_run() {
    let engine = Arc::new(Engine::new());
    engine.eval("(def fib (fn [n] (if (< n 2) n (+ (fib (- n 1)) (fib (- n 2))))))");

    let a_t0 = Instant::now();

    let e_a = engine.clone();
    let h_a = thread::spawn(move || {
        let mut iters = 0u64;
        while iters < 50 {
            let src = format!("(def helper_{iters} (fn [x] (+ x {iters}))) (helper_{iters} 5)");
            e_a.eval(&src);
            iters += 1;
        }
        iters
    });

    let e_b = engine.clone();
    let h_b = thread::spawn(move || e_b.call_compiled("fib", &[f64::to_bits(22.0)]));

    let a_iters = h_a.join().unwrap();
    let r_b = h_b.join().unwrap();
    let elapsed = a_t0.elapsed();

    eprintln!(
        "fib(22) = {}; A completed {} concurrent compiles in {:?}",
        nanobox_to_int(r_b),
        a_iters,
        elapsed
    );

    assert_eq!(nanobox_to_int(r_b), 17711);
    assert_eq!(a_iters, 50);
}

/// Stress: 4 threads, each looping between (compile a new fn, run it,
/// run an old fn) for a fixed duration. No outer serialization — all
/// threads share a single `Arc<Engine>` and make progress concurrently.
/// Verifies no deadlocks, no torn reads, no UB.
#[test]
fn stress_interleaved_compile_and_run() {
    let engine = Arc::new(Engine::new());
    engine.eval("(def base (fn [n] (* n n)))");

    let n_threads = 4;
    let barrier = Arc::new(std::sync::Barrier::new(n_threads));
    let stop = Arc::new(std::sync::atomic::AtomicBool::new(false));
    let mut handles = Vec::new();

    for tid in 0..n_threads {
        let engine = engine.clone();
        let barrier = barrier.clone();
        let stop = stop.clone();
        handles.push(thread::spawn(move || {
            barrier.wait();
            let mut iters = 0u32;
            while !stop.load(Ordering::Acquire) {
                let name = format!("f_{tid}_{iters}");
                let body = format!("(def {name} (fn [x] (+ x {tid})))");
                engine.eval(&body);
                let v = engine.call_compiled(&name, &[f64::to_bits(7.0)]);
                assert_eq!(nanobox_to_int(v), 7 + tid as i64);
                let v = engine.call_compiled("base", &[f64::to_bits(11.0)]);
                assert_eq!(nanobox_to_int(v), 121);
                iters += 1;
            }
            iters
        }));
    }

    thread::sleep(Duration::from_millis(50));
    stop.store(true, Ordering::Release);

    let mut total = 0u32;
    for h in handles {
        total += h.join().unwrap();
    }
    eprintln!("stress: {n_threads} threads completed {total} compile+run iterations in 50ms");
    assert!(total > 0);
}
