//! FFI slice: foreign handles, native calls, `letonce` globals, and the
//! native → managed trampoline with late binding.

use livetype_core::{Condition, Session, Value};
use std::sync::{Arc, Mutex};

/// A shared native-side log so tests can observe what the foreign functions did.
#[derive(Default)]
struct Native {
    opened: u64,
    next_handle: u64,
    drawn: Vec<i64>,
}

/// Declare the interface and bind working `open_window` / `draw` natives.
fn bind(session: &mut Session, native: &Arc<Mutex<Native>>) {
    session
        .eval(
            "foreign type Window;
             foreign fn open_window() -> Window;
             foreign fn draw(w: Window, n: i64) -> ();",
        )
        .unwrap();
    let kind = session.foreign_kind("Window").unwrap();
    let n = Arc::clone(native);
    session
        .register_foreign(
            "open_window",
            Box::new(move |_| {
                let mut g = n.lock().unwrap();
                g.opened += 1;
                g.next_handle += 1;
                Value::Foreign { kind, ptr: g.next_handle }
            }),
        )
        .unwrap();
    let n = Arc::clone(native);
    session
        .register_foreign(
            "draw",
            Box::new(move |args| {
                let Value::I64(v) = args[1] else { panic!("draw arg 2 not i64") };
                n.lock().unwrap().drawn.push(v);
                Value::Unit
            }),
        )
        .unwrap();
}

#[test]
fn letonce_survives_edits_and_callback_late_binds() {
    let native = Arc::new(Mutex::new(Native::default()));
    let mut s = Session::new();
    bind(&mut s, &native);

    // The `letonce` opens the window once; on_frame draws 42.
    s.eval(
        "letonce win = open_window();
         fn on_frame() -> i64 { draw(win, 42); 42 }",
    )
    .unwrap();

    assert_eq!(s.call("on_frame", vec![]).unwrap(), Value::I64(42));
    assert_eq!(s.call("on_frame", vec![]).unwrap(), Value::I64(42));

    // Live edit: same trampoline, new code. The window is NOT reopened.
    s.eval("fn on_frame() -> i64 { draw(win, 84); 42 }").unwrap();
    assert_eq!(s.call("on_frame", vec![]).unwrap(), Value::I64(42));

    let g = native.lock().unwrap();
    assert_eq!(g.opened, 1, "letonce must run its initializer exactly once");
    assert_eq!(g.drawn, vec![42, 42, 84], "callback rebinds to current code");
}

#[test]
fn a_global_reference_is_a_gc_root() {
    // A `letonce` holding a struct reference keeps it live across a collection.
    let mut s = Session::new();
    s.eval(
        "struct Cell { v: i64 }
         letonce c = Cell { v: 7 };
         fn read() -> i64 { c.v }",
    )
    .unwrap();
    let before = s.runtime.heap.len();
    assert_eq!(before, 1);
    let freed = s.runtime.collect_garbage();
    assert_eq!(freed, 0, "the global's object must be rooted");
    assert_eq!(s.call("read", vec![]).unwrap(), Value::I64(7));
}

#[test]
fn unregistered_foreign_fn_traps_clearly() {
    // Declared but never bound: a call must trap with a clear condition, not do
    // something silent.
    let mut s = Session::new();
    s.eval(
        "foreign type Window;
         foreign fn open_window() -> Window;
         foreign fn draw(w: Window, n: i64) -> ();",
    )
    .unwrap();
    let kind = s.foreign_kind("Window").unwrap();
    // Bind open_window only; leave draw unbound.
    s.register_foreign(
        "open_window",
        Box::new(move |_| Value::Foreign { kind, ptr: 1 }),
    )
    .unwrap();
    s.eval(
        "letonce win = open_window();
         fn on_frame() -> i64 { draw(win, 1); 0 }",
    )
    .unwrap();

    match s.call("on_frame", vec![]) {
        Err(Condition::RuntimeTypeError { message, .. }) => {
            assert!(message.contains("registered implementation"), "got: {message}");
        }
        other => panic!("expected a clear trap, got {other:?}"),
    }
}

#[test]
fn native_returning_the_wrong_type_traps_at_the_boundary() {
    // The native → managed return is a use-boundary: a native impl that lies
    // about its return type traps rather than poisoning the caller.
    let mut s = Session::new();
    s.eval("foreign fn bogus() -> i64;").unwrap();
    // Declared to return i64, but hands back a Unit.
    s.register_foreign("bogus", Box::new(|_| Value::Unit)).unwrap();
    s.eval("fn go() -> i64 { bogus() }").unwrap();
    match s.call("go", vec![]) {
        Err(Condition::RuntimeTypeError { message, .. }) => {
            assert!(message.contains("foreign result"), "got: {message}");
        }
        other => panic!("expected a boundary trap, got {other:?}"),
    }
}
