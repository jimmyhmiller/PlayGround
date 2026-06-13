//! Serializing paused VM state to disk and resuming — including in a fresh
//! engine ("new process" simulation). Spec §6.2 + §7.

use funct::{Cause, Funct, RunResult, StopWhen, Value};

fn int(i: i64) -> Value {
    Value::Int(i)
}

#[test]
fn save_and_resume_in_same_engine() {
    let mut vm = Funct::new();
    vm.load("fn go(n, acc) = if n == 0 { acc } else { go(n - 1, acc + n) }")
        .unwrap();
    let mut st = vm.start("go", vec![int(200), int(0)]).unwrap();
    match vm.run(&mut st, StopWhen::Fuel(500)) {
        RunResult::Paused(Cause::FuelExhausted) => {}
        other => panic!("unexpected: {:?}", other),
    }
    let json = vm.save_state(&st).unwrap();
    let mut restored = vm.restore_state(&json).unwrap();
    match vm.run(&mut restored, StopWhen::Never) {
        RunResult::Done(v) => assert_eq!(v, int(20100)),
        other => panic!("unexpected: {:?}", other),
    }
}

#[test]
fn save_to_disk_and_resume_in_fresh_engine() {
    // engine 1: run halfway, save to an actual file
    let mut vm1 = Funct::new();
    vm1.load("fn go(n, acc) = if n == 0 { acc } else { go(n - 1, acc + n) }")
        .unwrap();
    let mut st = vm1.start("go", vec![int(300), int(0)]).unwrap();
    match vm1.run(&mut st, StopWhen::Fuel(700)) {
        RunResult::Paused(Cause::FuelExhausted) => {}
        other => panic!("unexpected: {:?}", other),
    }
    let json = vm1.save_state(&st).unwrap();
    let path = std::env::temp_dir().join("funct_state_test.json");
    std::fs::write(&path, &json).unwrap();
    drop(vm1);

    // engine 2: fresh process simulation — same natives (Funct::new), restore
    let loaded = std::fs::read_to_string(&path).unwrap();
    std::fs::remove_file(&path).ok();
    let mut vm2 = Funct::new();
    let mut st2 = vm2.restore_state(&loaded).unwrap();
    match vm2.run(&mut st2, StopWhen::Never) {
        RunResult::Done(v) => assert_eq!(v, int(45150)),
        other => panic!("unexpected: {:?}", other),
    }
}

#[test]
fn mutable_locals_survive_serialization() {
    // `let mut` locals live in Cells — the snapshot must preserve them and
    // their sharing with closures.
    let mut vm = Funct::new();
    vm.load(
        "fn work(n) {\n let mut total = 0\n let mut i = 0\n while i < n {\n  i = i + 1\n  total = total + i\n }\n total\n}",
    )
    .unwrap();
    let mut st = vm.start("work", vec![int(100)]).unwrap();
    // pause mid-loop several times, snapshotting and restoring each time
    let mut json = None;
    for _ in 0..3 {
        match vm.run(&mut st, StopWhen::Fuel(400)) {
            RunResult::Paused(Cause::FuelExhausted) => {
                json = Some(vm.save_state(&st).unwrap());
                st = vm.restore_state(json.as_ref().unwrap()).unwrap();
            }
            RunResult::Done(v) => {
                assert_eq!(v, int(5050));
                return;
            }
            other => panic!("unexpected: {:?}", other),
        }
    }
    let mut vm2 = Funct::new();
    let mut st2 = vm2.restore_state(&json.unwrap()).unwrap();
    match vm2.run(&mut st2, StopWhen::Never) {
        RunResult::Done(v) => assert_eq!(v, int(5050)),
        other => panic!("unexpected: {:?}", other),
    }
}

#[test]
fn closure_cell_sharing_survives_round_trip() {
    let mut vm = Funct::new();
    vm.eval(
        "fn make() {\n let mut n = 0\n let inc = () => { n = n + 1; n }\n let get = () => n\n (inc, get)\n}\nlet (inc, get) = make()",
    )
    .unwrap();
    vm.call("inc", vec![]).unwrap();
    let st = funct::VmState {
        frames: vec![],
        stack: vec![],
        status: funct::Status::Done(Value::Unit),
    };
    let json = vm.save_state(&st).unwrap();

    let mut vm2 = Funct::new();
    vm2.restore_state(&json).unwrap();
    // the two closures still share one cell
    assert_eq!(vm2.call("inc", vec![]).unwrap(), int(2));
    assert_eq!(vm2.call("inc", vec![]).unwrap(), int(3));
    assert_eq!(vm2.call("get", vec![]).unwrap(), int(3));
}

#[test]
fn atoms_survive_with_shared_identity() {
    let mut vm = Funct::new();
    vm.eval("let a = atom(5)\nlet pair = (a, a)").unwrap();
    let st = funct::VmState {
        frames: vec![],
        stack: vec![],
        status: funct::Status::Done(Value::Unit),
    };
    let json = vm.save_state(&st).unwrap();

    let mut vm2 = Funct::new();
    vm2.restore_state(&json).unwrap();
    // mutating through one tuple slot is visible through the other -> same atom
    assert_eq!(
        vm2.eval("reset!(pair[0], 99)\n@(pair[1])").unwrap(),
        int(99)
    );
    // and through the original binding
    assert_eq!(vm2.eval("@a").unwrap(), int(99));
}

#[test]
fn paused_state_with_atom_mutation_resumes_correctly() {
    let mut vm = Funct::new();
    vm.eval("let log = atom([])").unwrap();
    vm.load("fn work() {\n for i in 1..=6 {\n  swap!(log, xs => push(xs, i))\n }\n @log\n}")
        .unwrap();
    let mut st = vm.start("work", vec![]).unwrap();
    // run until some (but not all) items are logged
    let mut json = None;
    for fuel in [60u64, 60, 60] {
        match vm.run(&mut st, StopWhen::Fuel(fuel)) {
            RunResult::Paused(Cause::FuelExhausted) => {
                json = Some(vm.save_state(&st).unwrap());
            }
            RunResult::Done(_) => break,
            other => panic!("unexpected: {:?}", other),
        }
    }
    let json = json.expect("should have paused at least once");

    // resume the saved snapshot in a fresh engine: the atom's partial state
    // plus the in-flight loop position must both be in the snapshot
    let mut vm2 = Funct::new();
    let mut st2 = vm2.restore_state(&json).unwrap();
    match vm2.run(&mut st2, StopWhen::Never) {
        RunResult::Done(v) => assert_eq!(v, vm2.eval("[1, 2, 3, 4, 5, 6]").unwrap()),
        other => panic!("unexpected: {:?}", other),
    }
    assert_eq!(vm2.eval("@log").unwrap(), v_list_1_to_6());
}

fn v_list_1_to_6() -> Value {
    Value::list((1..=6).map(Value::Int).collect())
}

#[test]
fn watchers_survive_serialization() {
    let mut vm = Funct::new();
    vm.eval("let a = atom(0)\nlet count = atom(0)\nwatch(a, \"w\", (old, new) => swap!(count, n => n + 1))")
        .unwrap();
    let st = funct::VmState {
        frames: vec![],
        stack: vec![],
        status: funct::Status::Done(Value::Unit),
    };
    let json = vm.save_state(&st).unwrap();

    let mut vm2 = Funct::new();
    vm2.restore_state(&json).unwrap();
    vm2.eval("swap!(a, x => x + 1)\nswap!(a, x => x + 1)")
        .unwrap();
    assert_eq!(vm2.eval("@count").unwrap(), int(2));
}

#[test]
fn native_value_in_state_fails_loudly() {
    struct Handle;
    let mut vm = Funct::new();
    vm.register_type::<Handle>("Handle")
        .ctor0("make_handle", || Handle);
    vm.eval("let h = make_handle()").unwrap();
    let st = funct::VmState {
        frames: vec![],
        stack: vec![],
        status: funct::Status::Done(Value::Unit),
    };
    let err = vm.save_state(&st).unwrap_err();
    assert!(
        err.msg.contains("cannot serialize native host value") && err.msg.contains("Handle"),
        "{}",
        err.msg
    );
}

#[test]
fn restore_without_required_native_fails_loudly() {
    let mut vm = Funct::new();
    vm.register1("custom_native", |x: i64| x * 2);
    vm.eval("let f = custom_native").unwrap();
    let st = funct::VmState {
        frames: vec![],
        stack: vec![],
        status: funct::Status::Done(Value::Unit),
    };
    let json = vm.save_state(&st).unwrap();

    let mut vm2 = Funct::new(); // does NOT register custom_native
    match vm2.restore_state(&json) {
        Err(e) => {
            let msg = e.to_string();
            assert!(msg.contains("custom_native"), "{}", msg);
        }
        Ok(_) => panic!("restore should fail without the native registered"),
    }

    // registering it first makes restore work
    let mut vm3 = Funct::new();
    vm3.register1("custom_native", |x: i64| x * 2);
    vm3.restore_state(&json).unwrap();
    assert_eq!(vm3.eval("f(21)").unwrap(), int(42));
}

// ---------- §7 capture_atoms / restore_atoms ----------

#[test]
fn capture_and_restore_atoms() {
    let program =
        "let counter = atom(0)\nlet name = atom(\"x\")\nfn bump() = swap!(counter, n => n + 1)";

    let mut vm = Funct::new();
    vm.eval(program).unwrap();
    vm.eval("bump()\nbump()\nreset!(name, \"saved\")").unwrap();
    let snapshot = vm.capture_atoms().unwrap();
    drop(vm);

    // "reopen the editor": fresh engine, re-eval the same program (atoms are
    // recreated with the same deterministic ids), then restore state into them
    let mut vm2 = Funct::new();
    vm2.eval(program).unwrap();
    assert_eq!(vm2.eval("@counter").unwrap(), int(0)); // fresh
    vm2.restore_atoms(&snapshot).unwrap();
    assert_eq!(vm2.eval("@counter").unwrap(), int(2));
    assert_eq!(vm2.eval("@name").unwrap(), Value::str("saved"));
    // and the program keeps working against restored state
    vm2.eval("bump()").unwrap();
    assert_eq!(vm2.eval("@counter").unwrap(), int(3));
}

#[test]
fn restore_atoms_mismatch_fails_loudly() {
    let mut vm = Funct::new();
    vm.eval("let a = atom(1)\nlet b = atom(2)").unwrap();
    let snapshot = vm.capture_atoms().unwrap();

    // different program shape: only one atom
    let mut vm2 = Funct::new();
    vm2.eval("let a = atom(1)").unwrap();
    let err = vm2.restore_atoms(&snapshot).unwrap_err();
    assert!(err.msg.contains("no live atom"), "{}", err.msg);
}

#[test]
fn done_and_faulted_states_round_trip() {
    let mut vm = Funct::new();
    let mut st = vm.eval_resumable("40 + 2").unwrap();
    match vm.run(&mut st, StopWhen::Never) {
        RunResult::Done(v) => assert_eq!(v, int(42)),
        other => panic!("unexpected: {:?}", other),
    }
    let json = vm.save_state(&st).unwrap();
    let mut vm2 = Funct::new();
    let st2 = vm2.restore_state(&json).unwrap();
    match st2.status {
        funct::Status::Done(v) => assert_eq!(v, int(42)),
        _ => panic!("expected Done status after restore"),
    }
}
