//! Phase 7: auto-tiering. Functions start interpreted; a hot one is promoted to
//! JIT mid-run, with interpreted and JIT frames coexisting in one actor. The
//! result is identical to running everything interpreted — promotion is an
//! optimization, never a behavior change.

use livetype::*;

const SUM_TO_100: i64 = 4950; // 0 + 1 + ... + 99

fn program() -> Session {
    let mut s = Session::new();
    s.eval(
        "fn add(a: i64, b: i64) -> i64 { a + b }
         fn main() -> i64 {
            let s = 0;
            let i = 0;
            while i < 100 {
                s = add(s, i);
                i = i + 1;
            }
            s
         }",
    )
    .unwrap();
    s
}

#[test]
fn hot_function_is_promoted_and_result_is_unchanged() {
    // Reference: everything interpreted.
    let mut interp = program();
    let interp_result = interp.call("main", vec![]).unwrap();
    assert_eq!(interp_result, Value::I64(SUM_TO_100));

    // Tiered: `add` is called 100 times, so with a threshold of 10 it is
    // promoted to JIT partway through the loop.
    let s = program();
    let add = s.fn_id("add").unwrap();
    let main = s.fn_id("main").unwrap();
    let mut tiered = Tiered::new(s.runtime, 10);
    let tiered_result = tiered.run(main, vec![]);

    assert_eq!(
        tiered_result,
        Outcome::Complete(Value::I64(SUM_TO_100)),
        "tiered result must match the interpreter"
    );
    assert!(
        tiered.is_hot(add, Version(1)),
        "`add` was called 100x with threshold 10 — it must have been promoted"
    );
}

#[test]
fn tiering_preserves_ffi_and_globals() {
    // A hot function that calls a foreign fn and reads a global still produces
    // the right answer once promoted (the JIT frame calls the same externs).
    let mut s = Session::new();
    s.eval("foreign fn dbl(n: i64) -> i64;").unwrap();
    s.register_foreign(
        "dbl",
        Box::new(|args| match args[0] {
            Value::I64(n) => Value::I64(n * 2),
            _ => panic!("dbl"),
        }),
    )
    .unwrap();
    s.eval(
        "letonce base = 3;
         fn step(x: i64) -> i64 { dbl(x) }
         fn main() -> i64 {
            let acc = base;
            let i = 0;
            while i < 50 {
                acc = step(acc);
                i = i + 1;
            }
            i
         }",
    )
    .unwrap();
    let step_id = s.fn_id("step").unwrap();
    let main = s.fn_id("main").unwrap();
    let mut tiered = Tiered::new(s.runtime, 5);
    let result = tiered.run(main, vec![]);
    assert_eq!(result, Outcome::Complete(Value::I64(50)));
    assert!(tiered.is_hot(step_id, Version(1)), "`step` (50 calls) must be promoted");
}
