//! Phase 5: the JIT runs on the concurrent tier's OS threads. Programs are
//! compiled once and executed by worker threads over one shared `Shared`
//! runtime (shared heap, shared world) — the JIT counterpart of the
//! interpreter's `Shared::run_threads`. This respects the LLVM/Miri boundary:
//! `livetype-core` never links LLVM; this crate owns the threads and the
//! compiled code and calls into core's thread-safe `Shared`.

use livetype::*;
use std::sync::Arc;

/// Compile `source` through the frontend and freeze it for concurrent JIT.
fn shared(source: &str) -> (std::sync::Arc<Shared>, Session) {
    let mut s = Session::new();
    s.eval(source).expect("compile");
    // `from_runtime` consumes the runtime, so hand back a fresh Session only for
    // its symbol table (id lookups) — rebuilt from the same source.
    let mut ids = Session::new();
    ids.eval(source).unwrap();
    (Shared::from_runtime(s.runtime), ids)
}

#[test]
fn jit_loop_runs_across_threads() {
    // A real loop (Branch/Jump/AddI64/LtI64), no FFI — summing 0..n.
    let (sh, ids) = shared(
        "fn sum(n: i64) -> i64 {
            let i = 0;
            let acc = 0;
            while i < n {
                acc = acc + i;
                i = i + 1;
            }
            acc
        }",
    );
    let sum = ids.fn_id("sum").unwrap();
    let outcomes = run_jit_threads(&sh, vec![(sum, vec![Value::I64(10)]); 4]).unwrap();
    for outcome in outcomes {
        assert_eq!(outcome, Outcome::Complete(Value::I64(45)));
    }
}

#[test]
fn jit_call_and_shared_heap_alloc_across_threads() {
    // Exercises Call, New (allocating on the SHARED heap concurrently), and
    // GetField from JIT-compiled code on four threads at once.
    let (sh, ids) = shared(
        "struct Box { v: i64 }
         fn dbl(x: i64) -> i64 { x * 2 }
         fn make(n: i64) -> i64 { let b = Box { v: n }; dbl(b.v) }",
    );
    let make = ids.fn_id("make").unwrap();
    let outcomes = run_jit_threads(&sh, vec![(make, vec![Value::I64(21)]); 4]).unwrap();
    for outcome in outcomes {
        assert_eq!(outcome, Outcome::Complete(Value::I64(42)));
    }
    // Four threads each allocated one Box on the shared heap.
    assert_eq!(sh.object_count(), 4);
}

#[test]
fn stop_the_world_gc_pauses_jit_threads() {
    // JIT workers churn allocations while another thread fires stop-the-world
    // collections. If the JIT driver did not hit safepoints, `request_gc` would
    // deadlock waiting for the workers to park — so completion proves the JIT
    // tier participates in the preemptive collector.
    let (sh, ids) = shared(
        "struct Box { v: i64 }
         fn churn(n: i64) -> i64 {
            let i = 0;
            while i < n {
                let b = Box { v: i };
                i = i + 1;
            }
            0
         }",
    );
    let churn = ids.fn_id("churn").unwrap();

    let collector = {
        let sh = Arc::clone(&sh);
        std::thread::spawn(move || {
            for _ in 0..30 {
                sh.request_gc();
            }
        })
    };
    let outcomes = run_jit_threads(&sh, vec![(churn, vec![Value::I64(400)]); 3]).unwrap();
    collector.join().unwrap();

    for outcome in outcomes {
        assert_eq!(outcome, Outcome::Complete(Value::I64(0)));
    }
    assert!(sh.collections() > 0, "at least one collection ran mid-flight");
}
