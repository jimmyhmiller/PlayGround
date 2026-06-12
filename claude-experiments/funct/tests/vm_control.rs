//! The reified-VM control surface: step, fuel, pause/resume, snapshots
//! (time travel), breakpoints, inspection. Spec §6.

use funct::{Cause, Funct, RunResult, StepResult, StopWhen, Value, VmState};
use std::collections::HashSet;

fn int(i: i64) -> Value {
    Value::Int(i)
}

#[test]
fn single_stepping_to_completion() {
    let mut vm = Funct::new();
    let mut st = vm.eval_resumable("1 + 2 * 3").unwrap();
    let mut steps = 0;
    loop {
        match vm.step(&mut st) {
            StepResult::Running => {
                steps += 1;
                assert!(steps < 1000, "runaway");
            }
            StepResult::Done(v) => {
                assert_eq!(v, int(7));
                break;
            }
            StepResult::Faulted(f) => panic!("fault: {}", f),
        }
    }
    assert!(
        steps >= 4,
        "should take several instructions, took {}",
        steps
    );
    // stepping a finished state stays Done
    assert_eq!(vm.step(&mut st), StepResult::Done(int(7)));
}

#[test]
fn fuel_pauses_and_resumes() {
    let mut vm = Funct::new();
    vm.load("fn go(n, acc) = if n == 0 { acc } else { go(n - 1, acc + n) }")
        .unwrap();
    let mut st = vm.start("go", vec![int(1000), int(0)]).unwrap();
    let mut pauses = 0;
    let result = loop {
        match vm.run(&mut st, StopWhen::Fuel(100)) {
            RunResult::Paused(Cause::FuelExhausted) => {
                pauses += 1;
                assert!(st.is_running());
                assert!(pauses < 10000, "never finished");
            }
            RunResult::Done(v) => break v,
            other => panic!("unexpected: {:?}", other),
        }
    };
    assert_eq!(result, int(500500));
    assert!(pauses > 10, "expected many fuel pauses, got {}", pauses);
}

#[test]
fn inspect_state_while_paused() {
    let mut vm = Funct::new();
    vm.load("fn work(n) {\n let mut total = 0\n for i in 1..=n { total = total + i }\n total\n}")
        .unwrap();
    let mut st = vm.start("work", vec![int(100)]).unwrap();
    // run a bit, then look inside
    match vm.run(&mut st, StopWhen::Fuel(50)) {
        RunResult::Paused(Cause::FuelExhausted) => {}
        other => panic!("unexpected: {:?}", other),
    }
    assert_eq!(st.depth(), 1);
    let frame = st.frames.last().unwrap();
    assert_eq!(frame.proto.name, "work");
    assert_eq!(frame.locals[0], int(100)); // the argument
    assert!(st.current_line().is_some());
    // and finish
    match vm.run(&mut st, StopWhen::Never) {
        RunResult::Done(v) => assert_eq!(v, int(5050)),
        other => panic!("unexpected: {:?}", other),
    }
}

#[test]
fn snapshot_time_travel() {
    let mut vm = Funct::new();
    vm.load("fn go(n, acc) = if n == 0 { acc } else { go(n - 1, acc + n) }")
        .unwrap();
    let mut st = vm.start("go", vec![int(500), int(0)]).unwrap();
    // run halfway
    match vm.run(&mut st, StopWhen::Fuel(1000)) {
        RunResult::Paused(Cause::FuelExhausted) => {}
        other => panic!("unexpected: {:?}", other),
    }
    // snapshot = plain clone of plain data
    let snapshot: VmState = st.clone();
    // finish the original
    let r1 = match vm.run(&mut st, StopWhen::Never) {
        RunResult::Done(v) => v,
        other => panic!("unexpected: {:?}", other),
    };
    // rewind: run the snapshot to completion too
    let mut replay = snapshot;
    let r2 = match vm.run(&mut replay, StopWhen::Never) {
        RunResult::Done(v) => v,
        other => panic!("unexpected: {:?}", other),
    };
    assert_eq!(r1, r2);
    assert_eq!(r1, int(125250));
}

#[test]
fn breakpoints() {
    let mut vm = Funct::new();
    let src = "fn f(x) {\n let a = x + 1\n let b = a * 2\n b + 3\n}"; // lines 1-5
    vm.load(src).unwrap();
    let mut st = vm.start("f", vec![int(10)]).unwrap();
    let bps: HashSet<u32> = [3].into_iter().collect(); // `let b = a * 2`
    match vm.run(&mut st, StopWhen::Breakpoints(bps.clone())) {
        RunResult::Paused(Cause::Breakpoint(line)) => assert_eq!(line, 3),
        other => panic!("unexpected: {:?}", other),
    }
    // `a` is already bound at the breakpoint
    let frame = st.frames.last().unwrap();
    assert!(
        frame.locals.contains(&int(11)),
        "locals: {:?}",
        frame.locals
    );
    // resuming doesn't re-trigger the same breakpoint
    match vm.run(&mut st, StopWhen::Breakpoints(bps)) {
        RunResult::Done(v) => assert_eq!(v, int(25)),
        other => panic!("unexpected: {:?}", other),
    }
}

#[test]
fn next_line_stepping() {
    let mut vm = Funct::new();
    vm.load("fn f() {\n let a = 1\n let b = 2\n a + b\n}")
        .unwrap();
    let mut st = vm.start("f", vec![]).unwrap();
    let mut lines = Vec::new();
    loop {
        match vm.run(&mut st, StopWhen::NextLine) {
            RunResult::Paused(Cause::NextLine(l)) => {
                lines.push(l);
                assert!(lines.len() < 100);
            }
            RunResult::Done(v) => {
                assert_eq!(v, int(3));
                break;
            }
            other => panic!("unexpected: {:?}", other),
        }
    }
    // we should have visited each source line in order
    assert!(lines.windows(2).all(|w| w[0] <= w[1]), "lines: {:?}", lines);
    assert!(
        lines.contains(&3) && lines.contains(&4),
        "lines: {:?}",
        lines
    );
}

#[test]
fn fault_state_is_preserved() {
    let mut vm = Funct::new();
    vm.load("fn f() = 1 / 0").unwrap();
    let mut st = vm.start("f", vec![]).unwrap();
    match vm.run(&mut st, StopWhen::Never) {
        RunResult::Faulted(f) => {
            assert!(f.msg.contains("division by zero"));
            assert!(
                f.at.as_deref().unwrap_or("").contains("f:1"),
                "at: {:?}",
                f.at
            );
        }
        other => panic!("unexpected: {:?}", other),
    }
    // stepping a faulted state keeps returning the fault
    match vm.step(&mut st) {
        StepResult::Faulted(f) => assert!(f.msg.contains("division by zero")),
        other => panic!("unexpected: {:?}", other),
    }
}

#[test]
fn suspension_across_calls() {
    // CALL pushes a frame in the same step loop — no host recursion — so we
    // can pause inside deeply nested script calls.
    let mut vm = Funct::new();
    vm.load("fn f(n) = if n == 0 { 0 } else { g(n) }\nfn g(n) = f(n - 1) + 1")
        .unwrap();
    let mut st = vm.start("f", vec![int(50)]).unwrap();
    // pause somewhere in the middle
    match vm.run(&mut st, StopWhen::Fuel(200)) {
        RunResult::Paused(Cause::FuelExhausted) => {}
        other => panic!("unexpected: {:?}", other),
    }
    assert!(
        st.depth() > 3,
        "should be paused deep in nested calls, depth {}",
        st.depth()
    );
    match vm.run(&mut st, StopWhen::Never) {
        RunResult::Done(v) => assert_eq!(v, int(50)),
        other => panic!("unexpected: {:?}", other),
    }
}

#[test]
fn host_can_abandon_a_run() {
    // "cancel" = just stop calling step() and drop the state
    let mut vm = Funct::new();
    vm.load("fn forever(n) = forever(n + 1)").unwrap();
    let mut st = vm.start("forever", vec![int(0)]).unwrap();
    match vm.run(&mut st, StopWhen::Fuel(10_000)) {
        RunResult::Paused(Cause::FuelExhausted) => {}
        other => panic!("infinite loop should pause, got: {:?}", other),
    }
    drop(st); // and the engine is still fine:
    assert_eq!(vm.eval("1 + 1").unwrap(), int(2));
}

// ---------- epoch / time budgets ----------

const SPIN: &str = "fn spin() {\n let mut n = 0\n while true { n = n + 1 }\n n\n}";

#[test]
fn host_driven_epoch_pauses_and_resumes() {
    // The host advances the epoch itself; no timer thread, fully deterministic.
    let mut vm = Funct::new();
    vm.load(SPIN).unwrap();
    let mut st = vm.start("spin", vec![]).unwrap();

    // Arm a deadline one tick ahead, then trip it by bumping the epoch.
    vm.set_deadline(vm.epoch_now() + 1);
    vm.bump_epoch();
    assert_eq!(
        vm.run(&mut st, StopWhen::Epoch),
        RunResult::Paused(Cause::DeadlineReached)
    );
    assert!(st.is_running(), "an infinite loop is not finished");

    // The paused state still resumes (e.g. under a gas budget).
    assert_eq!(
        vm.run(&mut st, StopWhen::Fuel(100)),
        RunResult::Paused(Cause::FuelExhausted)
    );
}

#[test]
fn epoch_handle_bumped_from_another_thread() {
    let mut vm = Funct::new();
    vm.load(SPIN).unwrap();
    let mut st = vm.start("spin", vec![]).unwrap();

    let epoch = vm.epoch();
    vm.set_deadline(vm.epoch_now() + 5);
    // A separate thread drives the counter past the deadline.
    let bumper = std::thread::spawn(move || {
        for _ in 0..5 {
            std::thread::sleep(std::time::Duration::from_millis(1));
            epoch.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        }
    });
    assert_eq!(
        vm.run(&mut st, StopWhen::Epoch),
        RunResult::Paused(Cause::DeadlineReached)
    );
    bumper.join().unwrap();
}

#[test]
fn wall_clock_deadline_interrupts_a_spin() {
    // The opt-in ticker turns a Duration into an epoch target.
    let mut vm = Funct::new();
    vm.load(SPIN).unwrap();
    let mut st = vm.start("spin", vec![]).unwrap();
    let t0 = std::time::Instant::now();
    assert_eq!(
        vm.run(
            &mut st,
            StopWhen::Deadline(std::time::Duration::from_millis(25))
        ),
        RunResult::Paused(Cause::DeadlineReached)
    );
    let elapsed = t0.elapsed();
    assert!(st.is_running());
    assert!(
        elapsed.as_millis() < 2000,
        "took far too long: {:?}",
        elapsed
    );
}

#[test]
fn budget_fuel_trips_before_deadline() {
    let mut vm = Funct::new();
    vm.load(SPIN).unwrap();
    let mut st = vm.start("spin", vec![]).unwrap();
    // Tiny gas, huge time budget → gas wins.
    assert_eq!(
        vm.run(
            &mut st,
            StopWhen::Budget {
                fuel: Some(50),
                deadline: Some(std::time::Duration::from_secs(60)),
            }
        ),
        RunResult::Paused(Cause::FuelExhausted)
    );
}

#[test]
fn budget_deadline_trips_before_fuel() {
    let mut vm = Funct::new();
    vm.load(SPIN).unwrap();
    let mut st = vm.start("spin", vec![]).unwrap();
    // Effectively unlimited gas, short time budget → the clock wins.
    assert_eq!(
        vm.run(
            &mut st,
            StopWhen::Budget {
                fuel: Some(u64::MAX),
                deadline: Some(std::time::Duration::from_millis(25)),
            }
        ),
        RunResult::Paused(Cause::DeadlineReached)
    );
}

#[test]
fn budget_without_deadline_runs_to_completion() {
    // Budget with no time limit is just gas; a finite program completes.
    let mut vm = Funct::new();
    vm.load("fn go(n, acc) = if n == 0 { acc } else { go(n - 1, acc + n) }")
        .unwrap();
    let mut st = vm.start("go", vec![int(1000), int(0)]).unwrap();
    loop {
        match vm.run(
            &mut st,
            StopWhen::Budget {
                fuel: Some(100),
                deadline: None,
            },
        ) {
            RunResult::Paused(Cause::FuelExhausted) => assert!(st.is_running()),
            RunResult::Done(v) => {
                assert_eq!(v, int(500500));
                break;
            }
            other => panic!("unexpected: {:?}", other),
        }
    }
}
