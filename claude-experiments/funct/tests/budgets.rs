//! Time/instruction budgets: the epoch counter, wall-clock deadlines, and the
//! combinable `Budget`, plus how they interact with snapshots and resumption.

use funct::{Cause, Funct, RunResult, StopWhen, Value, VmState};
use std::sync::atomic::Ordering;
use std::time::Duration;

const SPIN: &str = "fn spin() {\n let mut n = 0\n while true { n = n + 1 }\n n\n}";
// a long but FINITE loop, for deterministic pause→resume→complete tests
const COUNT: &str = "fn count(n) {\n let mut i = 0\n while i < n { i = i + 1 }\n i\n}";

fn spin_vm() -> (Funct, VmState) {
    let mut vm = Funct::new();
    vm.load(SPIN).unwrap();
    let st = vm.start("spin", vec![]).unwrap();
    (vm, st)
}

#[test]
fn budget_with_no_limits_runs_to_completion() {
    // Budget { None, None } is "no limit at all" — a finite program finishes.
    let mut vm = Funct::new();
    vm.load(COUNT).unwrap();
    let mut st = vm.start("count", vec![Value::Int(5000)]).unwrap();
    assert_eq!(
        vm.run(
            &mut st,
            StopWhen::Budget {
                fuel: None,
                deadline: None,
            },
        ),
        RunResult::Done(Value::Int(5000))
    );
}

#[test]
fn epoch_deadline_already_passed_pauses_immediately() {
    let (mut vm, mut st) = spin_vm();
    vm.set_deadline(vm.epoch_now()); // deadline == now → already met
    assert_eq!(
        vm.run(&mut st, StopWhen::Epoch),
        RunResult::Paused(Cause::DeadlineReached)
    );
}

#[test]
fn unreached_epoch_deadline_lets_a_finite_program_finish() {
    // Deadline set far in the future and never bumped → never trips.
    let mut vm = Funct::new();
    vm.load(COUNT).unwrap();
    let mut st = vm.start("count", vec![Value::Int(10_000)]).unwrap();
    vm.set_deadline(u64::MAX);
    assert_eq!(
        vm.run(&mut st, StopWhen::Epoch),
        RunResult::Done(Value::Int(10_000))
    );
}

#[test]
fn bump_epoch_is_monotonic_and_returned() {
    let vm = Funct::new();
    let base = vm.epoch_now();
    assert_eq!(vm.bump_epoch(), base + 1);
    assert_eq!(vm.bump_epoch(), base + 2);
    assert_eq!(vm.epoch_now(), base + 2);
    // the shared handle observes the same counter
    assert_eq!(vm.epoch().load(Ordering::Relaxed), base + 2);
}

#[test]
fn clear_deadline_disarms_a_previously_set_one() {
    let mut vm = Funct::new();
    vm.load(COUNT).unwrap();
    let mut st = vm.start("count", vec![Value::Int(2000)]).unwrap();
    vm.set_deadline(vm.epoch_now()); // would pause immediately...
    vm.clear_deadline(); // ...but we disarm it
    assert_eq!(
        vm.run(&mut st, StopWhen::Epoch),
        RunResult::Done(Value::Int(2000))
    );
}

#[test]
fn deadline_paused_state_snapshots_and_resumes_to_completion() {
    // The headline interaction: a time/epoch interrupt yields the same plain
    // VmState, so it round-trips through save/restore and finishes elsewhere.
    let mut vm = Funct::new();
    vm.load(COUNT).unwrap();
    let mut st = vm.start("count", vec![Value::Int(300_000)]).unwrap();

    // host-driven epoch (deterministic): arm + trip, pausing mid-loop
    vm.set_deadline(vm.epoch_now() + 1);
    vm.bump_epoch();
    assert_eq!(
        vm.run(&mut st, StopWhen::Epoch),
        RunResult::Paused(Cause::DeadlineReached)
    );
    assert!(st.is_running(), "should pause partway, not finish");

    let json = vm.save_state(&st).expect("a mid-loop state serializes");

    // restore into a fresh engine and run to completion
    let mut vm2 = Funct::new();
    vm2.load(COUNT).unwrap();
    let mut st2 = vm2.restore_state(&json).unwrap();
    assert_eq!(
        vm2.run(&mut st2, StopWhen::Never),
        RunResult::Done(Value::Int(300_000))
    );
}

#[test]
fn deadline_can_be_reused_across_runs() {
    // Calling Deadline twice must reuse the one ticker, not deadlock or leak.
    let (mut vm, mut st) = spin_vm();
    assert_eq!(
        vm.run(&mut st, StopWhen::Deadline(Duration::from_millis(15))),
        RunResult::Paused(Cause::DeadlineReached)
    );
    assert_eq!(
        vm.run(&mut st, StopWhen::Deadline(Duration::from_millis(15))),
        RunResult::Paused(Cause::DeadlineReached)
    );
    assert!(st.is_running());
}

#[test]
fn fuel_zero_pauses_before_any_progress() {
    let (mut vm, mut st) = spin_vm();
    assert_eq!(
        vm.run(&mut st, StopWhen::Fuel(0)),
        RunResult::Paused(Cause::FuelExhausted)
    );
}

#[test]
fn many_engines_with_tickers_clean_up() {
    // Each Deadline run starts a ticker; dropping the engine must stop its
    // thread. Churn a bunch and rely on Ticker's Drop (no thread-leak / hang).
    for _ in 0..16 {
        let (mut vm, mut st) = spin_vm();
        let _ = vm.run(&mut st, StopWhen::Deadline(Duration::from_millis(2)));
        // vm dropped here → ticker thread is signalled and joined
    }
}

#[test]
fn a_deadline_cannot_interrupt_inside_an_atomic_native_call() {
    // Safe points are between instructions, so a native call runs to its end
    // even if the deadline elapses mid-call. With a single ~30ms native and a
    // 5ms budget, the pause can only land AFTER the native returns — proving it
    // wasn't cut short.
    let mut vm = Funct::new();
    vm.register_raw("slow", |_vm, _args| {
        std::thread::sleep(Duration::from_millis(30));
        Ok(Value::Int(1))
    });
    vm.load("fn work() {\n let mut t = 0\n while true { t = t + slow() }\n t\n}")
        .unwrap();
    let mut st = vm.start("work", vec![]).unwrap();

    let t0 = std::time::Instant::now();
    let r = vm.run(&mut st, StopWhen::Deadline(Duration::from_millis(5)));
    let elapsed = t0.elapsed();

    assert_eq!(r, RunResult::Paused(Cause::DeadlineReached));
    assert!(
        elapsed >= Duration::from_millis(25),
        "native was interrupted mid-call (only {:?} elapsed)",
        elapsed
    );
}
