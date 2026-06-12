//! Hot reload (spec §8): name-keyed function table, atomic swap at an
//! instruction boundary, atom state survives, in-flight frames finish on old
//! code while new calls resolve to new code.

use funct::{Cause, Funct, RunResult, StopWhen, Value};

fn int(i: i64) -> Value {
    Value::Int(i)
}

#[test]
fn redefining_a_fn_swaps_it_for_all_callers() {
    let mut vm = Funct::new();
    vm.eval("fn greet() = \"v1\"\nfn use_it() = greet() + \"!\"")
        .unwrap();
    assert_eq!(vm.eval("use_it()").unwrap(), Value::str("v1!"));
    // reload just `greet` — `use_it` is untouched but picks up the new code
    vm.eval("fn greet() = \"v2\"").unwrap();
    assert_eq!(vm.eval("use_it()").unwrap(), Value::str("v2!"));
}

#[test]
fn closures_see_reloaded_functions() {
    let mut vm = Funct::new();
    vm.eval("fn f(x) = x + 1\nlet stored = y => f(y) * 10")
        .unwrap();
    assert_eq!(vm.eval("stored(1)").unwrap(), int(20));
    vm.eval("fn f(x) = x + 5").unwrap();
    // `stored` was created before the reload, but resolves f by name->id
    assert_eq!(vm.eval("stored(1)").unwrap(), int(60));
}

#[test]
fn atom_state_survives_reload() {
    let mut vm = Funct::new();
    vm.eval("let counter = atom(0)\nfn bump() = swap!(counter, n => n + 1)")
        .unwrap();
    vm.eval("bump()\nbump()").unwrap();
    assert_eq!(vm.eval("@counter").unwrap(), int(2));
    // reload bump with different behavior; the atom keeps its value
    vm.eval("fn bump() = swap!(counter, n => n + 10)").unwrap();
    vm.eval("bump()").unwrap();
    assert_eq!(vm.eval("@counter").unwrap(), int(12));
}

#[test]
fn reload_while_paused_new_calls_use_new_code() {
    let mut vm = Funct::new();
    vm.eval("let log = atom([])").unwrap();
    vm.load("fn g() = 1\nfn work() {\n for i in 0..4 {\n  swap!(log, xs => push(xs, g()))\n }\n @log\n}")
        .unwrap();
    let mut st = vm.start("work", vec![]).unwrap();

    // run in small fuel slices until exactly 2 items are logged
    let logged_len = |vm: &Funct| -> usize {
        match vm.global("log").unwrap() {
            Value::Atom(a) => match &*a.value.read() {
                Value::List(items) => items.len(),
                _ => 0,
            },
            _ => 0,
        }
    };
    while logged_len(&vm) < 2 {
        match vm.run(&mut st, StopWhen::Fuel(5)) {
            RunResult::Paused(Cause::FuelExhausted) => {}
            other => panic!("finished too early: {:?}", other),
        }
    }

    // hot swap g between two steps — the paused state is at an instruction
    // boundary, so this is always safe (spec §8.3)
    vm.eval("fn g() = 2").unwrap();

    match vm.run(&mut st, StopWhen::Never) {
        RunResult::Done(v) => {
            let items: Vec<_> = match v {
                Value::List(items) => items.iter().cloned().collect(),
                other => panic!("expected list, got {:?}", other),
            };
            // first iterations used old g, later ones the new g
            assert_eq!(items.len(), 4);
            assert!(items.starts_with(&[int(1), int(1)]), "items: {:?}", items);
            assert!(items.ends_with(&[int(2), int(2)]), "items: {:?}", items);
        }
        other => panic!("unexpected: {:?}", other),
    }
}

#[test]
fn in_flight_frame_finishes_on_old_code() {
    let mut vm = Funct::new();
    // `slow` computes its result across many instructions; pause inside it,
    // redefine it, and the in-flight call still returns the OLD result
    vm.load("fn slow() {\n let mut t = 0\n for i in 1..=10 { t = t + i }\n t\n}")
        .unwrap();
    let mut st = vm.start("slow", vec![]).unwrap();
    match vm.run(&mut st, StopWhen::Fuel(20)) {
        RunResult::Paused(Cause::FuelExhausted) => {}
        other => panic!("unexpected: {:?}", other),
    }
    vm.eval("fn slow() = 999").unwrap();
    match vm.run(&mut st, StopWhen::Never) {
        RunResult::Done(v) => assert_eq!(v, int(55), "in-flight call must finish on old code"),
        other => panic!("unexpected: {:?}", other),
    }
    // but a fresh call uses the new definition
    assert_eq!(vm.eval("slow()").unwrap(), int(999));
}

#[test]
fn reload_preserves_other_definitions() {
    let mut vm = Funct::new();
    vm.eval("fn a() = 1\nfn b() = 2").unwrap();
    vm.eval("fn a() = 10").unwrap();
    assert_eq!(vm.eval("a() + b()").unwrap(), int(12));
}

#[test]
fn top_level_lets_can_be_re_evaluated() {
    let mut vm = Funct::new();
    vm.eval("let config = { debug: false }").unwrap();
    vm.eval("let config = { debug: true }").unwrap();
    assert_eq!(vm.eval("config.debug").unwrap(), Value::Bool(true));
}
