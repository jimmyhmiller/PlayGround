//! Thread-safety: values are Arc/Mutex-backed, so the engine and paused
//! VmStates are Send (+Sync) and can move across threads / live in
//! thread-safe containers (e.g. Bevy resources).

use funct::{Funct, Value};

fn assert_send<T: Send>(_: &T) {}
fn assert_sync<T: Sync>(_: &T) {}

#[test]
fn engine_and_values_are_send_sync() {
    let mut vm = Funct::new();
    vm.eval("fn f(x) = x * 2\nlet a = atom([1, 2])").unwrap();
    assert_send(&vm);
    assert_sync(&vm);
    let v = vm.eval("{ x: [1, atom(2)] }").unwrap();
    assert_send(&v);
    assert_sync(&v);
}

#[test]
fn engine_moves_across_threads() {
    let mut vm = Funct::new();
    vm.register1("host_inc", |x: i64| x + 1);
    vm.eval("let state = atom(0)\nfn bump() = swap!(state, n => host_inc(n))").unwrap();
    vm.call("bump", vec![]).unwrap();

    // move the whole engine (with live atoms and natives) to another thread
    let handle = std::thread::spawn(move || {
        let mut vm = vm;
        vm.call("bump", vec![]).unwrap();
        vm.call("bump", vec![]).unwrap();
        vm.eval("@state").unwrap()
    });
    assert_eq!(handle.join().unwrap(), Value::Int(3));
}

#[test]
fn paused_state_moves_across_threads() {
    let mut vm = Funct::new();
    vm.load("fn go(n, acc) = if n == 0 { acc } else { go(n - 1, acc + n) }").unwrap();
    let mut st = vm.start("go", vec![Value::Int(100), Value::Int(0)]).unwrap();
    match vm.run(&mut st, funct::StopWhen::Fuel(200)) {
        funct::RunResult::Paused(_) => {}
        other => panic!("unexpected: {:?}", other),
    }
    // both the engine and the paused VmState cross the thread boundary
    let handle = std::thread::spawn(move || {
        let mut vm = vm;
        let mut st = st;
        match vm.run(&mut st, funct::StopWhen::Never) {
            funct::RunResult::Done(v) => v,
            other => panic!("unexpected: {:?}", other),
        }
    });
    assert_eq!(handle.join().unwrap(), Value::Int(5050));
}
