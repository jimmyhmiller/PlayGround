//! Phase 4: live editing a program *while it runs across threads*. A worker
//! runs on its own OS thread over the shared runtime; between two of its calls
//! we hot-swap the function it calls, from another thread, and the worker picks
//! up the new version on its next call — no restart. The handshake is done with
//! the runtime's own message passing, so the interleaving is deterministic (no
//! sleeps-and-hope).

use livetype_core::*;
use std::sync::Arc;
use std::time::Duration;

const TICK: DefId = 1000;
const WORKER: DefId = 2000;

/// `tick()` returning the constant `n`.
fn tick(version: u64, n: i64) -> Function {
    Function {
        id: TICK,
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

/// A worker that, twice, waits for a go-signal then emits `tick()`. The two
/// `Recv`s let the driving thread place a hot edit *between* the two `tick`
/// calls with no timing guesswork.
fn worker() -> Function {
    Function {
        id: WORKER,
        version: Version(1),
        name: "worker".into(),
        params: vec![],
        result: Type::I64,
        registers: 5,
        code: vec![
            Instruction::Recv { dst: 0, ty: Type::I64 },          // gate 1
            Instruction::Call { dst: 1, function: TICK, args: vec![] },
            Instruction::Emit { value: 1 },
            Instruction::Recv { dst: 2, ty: Type::I64 },          // gate 2
            Instruction::Call { dst: 3, function: TICK, args: vec![] },
            Instruction::Emit { value: 3 },
            Instruction::Const { dst: 4, value: Value::I64(0) },
            Instruction::Return { value: 4 },
        ],
    }
}

/// Block until the shared runtime has emitted at least `n` values, or fail
/// loudly rather than hang.
fn wait_output(shared: &Shared, n: usize) {
    for _ in 0..100_000 {
        if shared.output().len() >= n {
            return;
        }
        std::thread::sleep(Duration::from_micros(50));
    }
    panic!("timed out waiting for {n} outputs (got {})", shared.output().len());
}

#[test]
fn function_hot_swapped_while_a_worker_thread_runs() {
    let mut rt = Runtime::default();
    rt.install_function(tick(1, 1)).unwrap();
    rt.install_function(worker()).unwrap();

    let shared = Shared::from_runtime(rt);
    shared.ensure_mailbox(0); // the worker's mailbox, before it can be sent to

    // Launch the worker on its own OS thread. It blocks on its first `Recv`.
    let worker_shared = Arc::clone(&shared);
    let handle = std::thread::spawn(move || worker_shared.run_actor(0, WORKER, vec![]));

    // Gate 1: the worker calls tick@v1 and emits 1.
    assert!(shared.send_to(0, Value::I64(0)));
    wait_output(&shared, 1);

    // Hot edit from THIS thread while the worker thread is alive: tick → v2.
    shared.install_function(tick(2, 2)).unwrap();

    // Gate 2: the worker's next call re-resolves tick and now emits 2.
    assert!(shared.send_to(0, Value::I64(0)));
    wait_output(&shared, 2);

    let outcome = handle.join().unwrap();
    assert_eq!(outcome, Outcome::Complete(Value::I64(0)));
    assert_eq!(
        shared.output(),
        vec![Value::I64(1), Value::I64(2)],
        "the running worker emitted the old version, then the hot-swapped one"
    );
}
