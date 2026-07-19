//! The JIT on OS threads. Worker threads execute compiled `step` functions
//! over the one shared runtime (shared heap, shared world) — the same engine
//! loop as everywhere else, with an always-JIT configuration. This respects
//! the LLVM/Miri boundary: `livetype-core` never links LLVM; this crate
//! supplies the compiler through `TierSource`.

use livetype::*;
use std::sync::Arc;
use std::time::Duration;

/// Compile `source` through the frontend onto an always-JIT engine.
fn engine_for(source: &str) -> (Arc<Engine>, Session) {
    let mut s = Session::with_engine(jit_engine(0));
    s.eval(source).expect("compile");
    let engine = Arc::clone(&s.engine);
    (engine, s)
}

#[test]
fn jit_loop_runs_across_threads() {
    // A real loop (Branch/Jump/AddI64/LtI64), no FFI — summing 0..n.
    let (engine, ids) = engine_for(
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
    let outcomes = engine.run_threads(vec![(sum, vec![Value::I64(10)]); 4]);
    for outcome in outcomes {
        assert_eq!(outcome, Outcome::Complete(Value::I64(45)));
    }
}

#[test]
fn jit_call_and_shared_heap_alloc_across_threads() {
    // Exercises Call, New (allocating on the SHARED heap concurrently), and
    // GetField from JIT-compiled code on four threads at once.
    let (engine, ids) = engine_for(
        "struct Box { v: i64 }
         fn dbl(x: i64) -> i64 { x * 2 }
         fn make(n: i64) -> i64 { let b = Box { v: n }; dbl(b.v) }",
    );
    let make = ids.fn_id("make").unwrap();
    let outcomes = engine.run_threads(vec![(make, vec![Value::I64(21)]); 4]);
    for outcome in outcomes {
        assert_eq!(outcome, Outcome::Complete(Value::I64(42)));
    }
    // Four threads each allocated one Box on the shared heap.
    assert_eq!(engine.shared().object_count(), 4);
}

#[test]
fn stop_the_world_gc_pauses_jit_threads() {
    // JIT workers churn allocations while another thread fires stop-the-world
    // collections. If the engine did not hit safepoints, `request_gc` would
    // deadlock waiting for the workers to park — so completion proves the JIT
    // configuration participates in the preemptive collector.
    let (engine, ids) = engine_for(
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
        let sh = Arc::clone(engine.shared());
        std::thread::spawn(move || {
            for _ in 0..30 {
                sh.request_gc();
            }
        })
    };
    let outcomes = engine.run_threads(vec![(churn, vec![Value::I64(400)]); 3]);
    collector.join().unwrap();

    for outcome in outcomes {
        assert_eq!(outcome, Outcome::Complete(Value::I64(0)));
    }
    assert!(engine.shared().collections() > 0, "at least one collection ran mid-flight");
}

/// A single-instruction `tick()` returning the constant `n` at `version`.
fn tick_returning(id: DefId, version: u64, n: i64) -> Function {
    Function {
        id,
        version: Version(version),
        name: "tick".into(),
        params: vec![],
        result: Type::I64,
        registers: 1,
        code: vec![
            Instruction::Const { dst: 0, value: Value::I64(n) },
            Instruction::Return { value: 0 },
        ],
    }
}

/// A deliberately ill-typed `tick()` (declares i64, returns a bool) — installs
/// Broken, so a running caller traps on its next call.
fn tick_broken(id: DefId, version: u64) -> Function {
    Function {
        id,
        version: Version(version),
        name: "tick".into(),
        params: vec![],
        result: Type::I64,
        registers: 1,
        code: vec![
            Instruction::Const { dst: 0, value: Value::Bool(true) },
            Instruction::Return { value: 0 },
        ],
    }
}

fn wait_until(shared: &Shared, pred: impl Fn(&[Value]) -> bool) {
    for _ in 0..200_000 {
        if pred(&shared.output()) {
            return;
        }
        std::thread::sleep(Duration::from_micros(50));
    }
    panic!("timed out waiting for a condition on the output");
}

#[test]
fn live_edit_a_program_running_on_a_jit_thread() {
    // A worker runs a tight loop calling `tick()` on a JIT thread. From another
    // thread we hot-swap `tick` (1 -> 2) while it runs — the source recompiles
    // on demand and the worker picks up the new version — then install a
    // breaking edit to stop it. Deterministic in outcome: we actively wait for
    // each observable transition rather than sleep-and-hope.
    let (engine, ids) = engine_for(
        "fn tick() -> i64 { 1 }
         fn worker() -> i64 { while 0 < 1 { emit(tick()); } 0 }",
    );
    let tick = ids.fn_id("tick").unwrap();
    let worker = ids.fn_id("worker").unwrap();

    let editor = {
        let engine = Arc::clone(&engine);
        std::thread::spawn(move || {
            // Wait until the worker is emitting v1, then hot-swap to v2.
            wait_until(engine.shared(), |o| o.iter().any(|v| *v == Value::I64(1)));
            engine.install_function(tick_returning(tick, 2, 2)).unwrap();
            // Wait until the swapped-in v2 is observed, then break `tick` so the
            // worker's next call traps and the run ends.
            wait_until(engine.shared(), |o| o.iter().any(|v| *v == Value::I64(2)));
            engine.install_function(tick_broken(tick, 3)).unwrap();
        })
    };

    let outcomes = engine.run_threads(vec![(worker, vec![])]);
    editor.join().unwrap();

    // The breaking edit stopped the loop with a BrokenFunction trap.
    assert!(
        matches!(outcomes[0], Outcome::Paused(Condition::BrokenFunction { .. })),
        "got {:?}",
        outcomes[0]
    );
    // The running worker emitted the old version, then the hot-swapped one, with
    // no 1 ever appearing after the first 2 (the swap is monotonic).
    let out = engine.output();
    assert!(out.contains(&Value::I64(1)), "never ran v1");
    assert!(out.contains(&Value::I64(2)), "never picked up v2");
    let first_two = out.iter().position(|v| *v == Value::I64(2)).unwrap();
    assert!(
        out[first_two..].iter().all(|v| *v == Value::I64(2)),
        "a v1 appeared after the swap to v2"
    );
}
