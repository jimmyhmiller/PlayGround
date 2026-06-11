//! Adversarial / corner-case coverage: things likely to be wrong in a young
//! compiler+VM (tail calls through control flow, ?, scoping, pause points in
//! awkward states, etc.)

use funct::{Cause, Funct, RunResult, StopWhen, Value};

fn eval(src: &str) -> Value {
    let mut vm = Funct::new();
    vm.eval(src).unwrap_or_else(|e| panic!("eval failed: {}\nsource:\n{}", e, src))
}

fn int(i: i64) -> Value {
    Value::Int(i)
}

#[test]
fn tail_call_through_match_arm() {
    assert_eq!(
        eval("fn go(n) = match n {\n 0 => \"done\",\n _ => go(n - 1)\n}\ngo(500000)"),
        Value::str("done")
    );
}

#[test]
fn mutual_tail_recursion() {
    assert_eq!(
        eval("fn even(n) = if n == 0 { true } else { odd(n - 1) }\nfn odd(n) = if n == 0 { false } else { even(n - 1) }\neven(400000)"),
        Value::Bool(true)
    );
}

#[test]
fn question_mark_inside_larger_expression() {
    assert_eq!(
        eval("fn get() = Ok(40)\nfn f() = Ok(get()? + 2)\nf()"),
        Value::ok(int(42))
    );
}

#[test]
fn question_mark_chained() {
    let src = r#"
fn a() = Ok(1)
fn b() = Ok(2)
fn f() = Ok(a()? + b()?)
f()
"#;
    assert_eq!(eval(src), Value::ok(int(3)));
}

#[test]
fn match_arms_separated_by_newlines_only() {
    assert_eq!(
        eval("match 2 {\n 1 => \"a\"\n 2 => \"b\"\n _ => \"c\"\n}"),
        Value::str("b")
    );
}

#[test]
fn negative_literals_in_patterns() {
    assert_eq!(
        eval("match -3 {\n -3 => \"neg three\",\n _ => \"other\"\n}"),
        Value::str("neg three")
    );
    assert_eq!(
        eval("match -5 {\n -10..0 => \"small neg\",\n _ => \"other\"\n}"),
        Value::str("small neg")
    );
}

#[test]
fn guard_calls_a_function() {
    assert_eq!(
        eval("fn big(x) = x > 10\nmatch 20 {\n n if big(n) => \"big\",\n _ => \"small\"\n}"),
        Value::str("big")
    );
}

#[test]
fn guard_failure_falls_through_with_subject_intact() {
    assert_eq!(
        eval("match (1, 2) {\n (a, b) if a > b => \"gt\",\n (a, b) if a < b => \"lt\",\n _ => \"eq\"\n}"),
        Value::str("lt")
    );
}

#[test]
fn closures_in_loop_capture_current_value() {
    let src = r#"
fn make() {
    let mut fns = []
    for i in [1, 2, 3] {
        fns = push(fns, () => i * 10)
    }
    fns
}
make() |> map(f => f())
"#;
    assert_eq!(eval(src), eval("[10, 20, 30]"));
}

#[test]
fn let_mut_inside_loop_is_fresh_each_iteration() {
    let src = r#"
fn f() {
    let mut out = []
    for i in [1, 2] {
        let mut x = 0
        x = x + i
        out = push(out, x)
    }
    out
}
f()
"#;
    assert_eq!(eval(src), eval("[1, 2]"));
}

#[test]
fn shadowing_inside_blocks_restores_outer() {
    assert_eq!(
        eval("fn f() {\n let x = 1\n let y = {\n  let x = 100\n  x\n }\n x + y\n}\nf()"),
        int(101)
    );
}

#[test]
fn nested_closures_capture_through_two_levels() {
    assert_eq!(
        eval("fn f(a) = b => c => a + b + c\nf(1)(2)(3)"),
        int(6)
    );
}

#[test]
fn nested_mut_capture_through_two_levels() {
    let src = r#"
fn make() {
    let mut n = 0
    () => () => { n = n + 100; n }
}
let f = make()
f()()
f()()
"#;
    assert_eq!(eval(src), int(200));
}

#[test]
fn pause_with_values_on_operand_stack_serializes() {
    // pause in the middle of evaluating `1000 + fib(...)` so the operand
    // stack and several frames are non-trivial, then round-trip
    let mut vm = Funct::new();
    vm.load("fn fib(n) = if n < 2 { n } else { fib(n - 1) + fib(n - 2) }\nfn f() = 1000 + fib(15)")
        .unwrap();
    let mut st = vm.start("f", vec![]).unwrap();
    match vm.run(&mut st, StopWhen::Fuel(2000)) {
        RunResult::Paused(Cause::FuelExhausted) => {}
        other => panic!("unexpected: {:?}", other),
    }
    assert!(st.depth() > 1);
    let json = vm.save_state(&st).unwrap();
    let mut vm2 = Funct::new();
    let mut st2 = vm2.restore_state(&json).unwrap();
    match vm2.run(&mut st2, StopWhen::Never) {
        RunResult::Done(v) => assert_eq!(v, int(1610)), // fib(15) = 610
        other => panic!("unexpected: {:?}", other),
    }
}

#[test]
fn step_through_native_call_is_atomic() {
    let mut vm = Funct::new();
    vm.load("fn f() = len([1, 2, 3]) + 1").unwrap();
    let mut st = vm.start("f", vec![]).unwrap();
    let mut steps = 0;
    loop {
        match vm.step(&mut st) {
            funct::StepResult::Running => steps += 1,
            funct::StepResult::Done(v) => {
                assert_eq!(v, int(4));
                break;
            }
            funct::StepResult::Faulted(f) => panic!("{}", f),
        }
        assert!(steps < 100);
    }
}

#[test]
fn fuel_zero_pauses_immediately() {
    let mut vm = Funct::new();
    let mut st = vm.eval_resumable("1 + 1").unwrap();
    match vm.run(&mut st, StopWhen::Fuel(0)) {
        RunResult::Paused(Cause::FuelExhausted) => {}
        other => panic!("unexpected: {:?}", other),
    }
    match vm.run(&mut st, StopWhen::Never) {
        RunResult::Done(v) => assert_eq!(v, int(2)),
        other => panic!("unexpected: {:?}", other),
    }
}

#[test]
fn chained_comparison_is_rejected() {
    let mut vm = Funct::new();
    assert!(vm.eval("1 < 2 < 3").is_err(), "comparison is non-associative");
}

#[test]
fn empty_braces_are_an_empty_record() {
    assert_eq!(eval("len({})"), int(0));
}

#[test]
fn record_with_lambda_values() {
    assert_eq!(
        eval("let obj = { add: (a, b) => a + b, base: 10 }\nobj.add(obj.base, 5)"),
        int(15)
    );
}

#[test]
fn variant_payload_arity_mismatch_does_not_match() {
    assert_eq!(
        eval("match Pair(1, 2) {\n Pair(a) => a,\n Pair(a, b) => a + b\n}"),
        int(3)
    );
}

#[test]
fn deeply_nested_data_round_trips_through_snapshot() {
    let mut vm = Funct::new();
    vm.eval("let data = { xs: [1, (2, [3, { y: Some(4) }])], z: atom([5]) }").unwrap();
    let st = funct::VmState { frames: vec![], stack: vec![], status: funct::Status::Done(Value::Unit) };
    let json = vm.save_state(&st).unwrap();
    let mut vm2 = Funct::new();
    vm2.restore_state(&json).unwrap();
    assert_eq!(vm2.eval("data.xs[1]").unwrap(), vm2.eval("(2, [3, { y: Some(4) }])").unwrap());
    assert_eq!(vm2.eval("@(data.z)").unwrap(), vm2.eval("[5]").unwrap());
}

#[test]
fn atom_containing_itself_serializes() {
    // cycle through the atom table must not hang or stack-overflow
    let mut vm = Funct::new();
    vm.eval("let a = atom(0)\nreset!(a, [a])").unwrap();
    let st = funct::VmState { frames: vec![], stack: vec![], status: funct::Status::Done(Value::Unit) };
    let json = vm.save_state(&st).unwrap();
    let mut vm2 = Funct::new();
    vm2.restore_state(&json).unwrap();
    // the cycle survives: @a is a list whose element is `a` itself
    assert_eq!(vm2.eval("match @a { [inner] => inner == a, _ => false }").unwrap(), Value::Bool(true));
}

#[test]
fn while_loop_pausable_and_resumable_many_times() {
    let mut vm = Funct::new();
    vm.load("fn count(n) {\n let mut i = 0\n while i < n { i = i + 1 }\n i\n}").unwrap();
    let mut st = vm.start("count", vec![int(10_000)]).unwrap();
    let mut pauses = 0u32;
    loop {
        match vm.run(&mut st, StopWhen::Fuel(997)) {
            RunResult::Paused(Cause::FuelExhausted) => pauses += 1,
            RunResult::Done(v) => {
                assert_eq!(v, int(10_000));
                break;
            }
            other => panic!("unexpected: {:?}", other),
        }
        assert!(pauses < 1_000_000);
    }
    assert!(pauses > 5);
}

#[test]
fn string_interpolation_with_nested_call_and_string() {
    assert_eq!(
        eval(r#"fn shout(s) = s + "!"
"hey ${shout("you")} there""#),
        Value::str("hey you! there")
    );
}

#[test]
fn unicode_strings() {
    assert_eq!(eval(r#""héllo → wörld""#), Value::str("héllo → wörld"));
    assert_eq!(eval(r#"len("日本語")"#), int(3));
    assert_eq!(eval(r#""日本語"[1]"#), Value::str("本"));
}

#[test]
fn ufcs_chain() {
    assert_eq!(
        eval("fn double(x) = x * 2\nfn inc(x) = x + 1\n5.double().inc().double()"),
        int(22)
    );
}

#[test]
fn pipe_mixed_with_ufcs() {
    assert_eq!(
        eval("fn double(x) = x * 2\n[1, 2, 3] |> map(x => x.double()) |> sum"),
        int(12)
    );
}

#[test]
fn hot_reload_changing_arity() {
    let mut vm = Funct::new();
    vm.eval("fn f(x) = x + 1").unwrap();
    assert_eq!(vm.eval("f(1)").unwrap(), int(2));
    vm.eval("fn f(x, y) = x + y").unwrap();
    assert_eq!(vm.eval("f(1, 2)").unwrap(), int(3));
    // old-arity call now faults at runtime with a clear message
    let err = vm.eval("f(1)").unwrap_err().to_string();
    assert!(err.contains("expects 2 argument(s)"), "{}", err);
}

#[test]
fn arity_mismatch_faults() {
    let mut vm = Funct::new();
    vm.eval("fn f(x) = x").unwrap();
    let err = vm.eval("f(1, 2)").unwrap_err().to_string();
    assert!(err.contains("expects 1 argument(s), got 2"), "{}", err);
}

#[test]
fn calling_a_non_function_faults() {
    let mut vm = Funct::new();
    let err = vm.eval("let x = 5\nx(1)").unwrap_err().to_string();
    assert!(err.contains("not callable"), "{}", err);
}

#[test]
fn float_display() {
    assert_eq!(eval("str(1.5)"), Value::str("1.5"));
    assert_eq!(eval("str(2.0)"), Value::str("2.0"));
}

#[test]
fn integer_overflow_faults_instead_of_wrapping() {
    let mut vm = Funct::new();
    let err = vm.eval("9223372036854775807 + 1").unwrap_err().to_string();
    assert!(err.contains("overflow"), "{}", err);
}
