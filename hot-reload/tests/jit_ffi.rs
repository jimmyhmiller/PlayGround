//! The JIT runs FFI and globals. `RawSlot` tag-encodes a `Foreign` handle
//! (kind in the tag's high bits, pointer in the payload), and the JIT lowers
//! `CallForeign`/`LoadGlobal` to externs, so a foreign program runs identically
//! interpreted and compiled, single-threaded or across worker threads.

use livetype::*;
use std::sync::{Arc, Mutex};

/// Build the `dbl` + `base` program with `dbl` bound, on an always-JIT engine.
fn doubling_session() -> Session {
    let mut s = Session::with_engine(jit_engine(0));
    s.eval("foreign fn dbl(n: i64) -> i64;").unwrap();
    s.register_foreign(
        "dbl",
        Box::new(|args| match args[0] {
            Value::I64(n) => Value::I64(n * 2),
            _ => panic!("dbl expects i64"),
        }),
    )
    .unwrap();
    s.eval("letonce base = 21; fn compute() -> i64 { dbl(base) }")
        .unwrap();
    s
}

#[test]
fn jit_runs_ffi_and_globals_single_threaded() {
    let mut s = doubling_session();
    assert_eq!(s.call("compute", vec![]).unwrap(), Value::I64(42));
}

#[test]
fn jit_runs_ffi_and_globals_across_threads() {
    let s = doubling_session();
    let compute = s.fn_id("compute").unwrap();
    let outcomes = s.engine.run_threads(vec![(compute, vec![]); 4]);
    for outcome in outcomes {
        assert_eq!(outcome, Outcome::Complete(Value::I64(42)));
    }
}

#[test]
fn foreign_handle_round_trips_through_jit_slots() {
    // A `letonce` opens a native "window" (a Foreign handle); a function loads
    // that global and passes the handle to `draw`. This exercises the Foreign
    // tag-encoding surviving a LoadGlobal → CallForeign-argument round trip
    // through the JIT's two-word slots.
    let native = Arc::new(Mutex::new(Vec::<(u64, i64)>::new()));
    let mut s = Session::with_engine(jit_engine(0));
    s.eval(
        "foreign type Window;
         foreign fn open_window() -> Window;
         foreign fn draw(w: Window, n: i64) -> ();",
    )
    .unwrap();
    let kind = s.foreign_kind("Window").unwrap();
    s.register_foreign(
        "open_window",
        Box::new(move |_| Value::Foreign { kind, ptr: 7 }),
    )
    .unwrap();
    {
        let native = Arc::clone(&native);
        s.register_foreign(
            "draw",
            Box::new(move |args| {
                let (Value::Foreign { ptr, .. }, Value::I64(n)) = (&args[0], &args[1]) else {
                    panic!("draw args");
                };
                native.lock().unwrap().push((*ptr, *n));
                Value::Unit
            }),
        )
        .unwrap();
    }
    s.eval("letonce win = open_window(); fn frame() -> i64 { draw(win, 99); 0 }")
        .unwrap();

    assert_eq!(s.call("frame", vec![]).unwrap(), Value::I64(0));
    // The handle (ptr 7) and the value (99) arrived intact through the JIT.
    assert_eq!(*native.lock().unwrap(), vec![(7, 99)]);
}
