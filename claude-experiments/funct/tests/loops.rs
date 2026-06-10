//! break / continue: semantics, nesting, scoping (lambdas don't leak loops),
//! and the operand-stack-cleanup cases (break out of a match arm, break in
//! expression position) that exercise the compiler's depth tracking.

use funct::{Funct, Value};

fn eval(src: &str) -> Value {
    let mut vm = Funct::new();
    vm.eval(src).unwrap_or_else(|e| panic!("eval failed: {}\nsource:\n{}", e, src))
}

fn eval_err(src: &str) -> String {
    let mut vm = Funct::new();
    match vm.eval(src) {
        Ok(v) => panic!("expected error, got {:?}\nsource:\n{}", v, src),
        Err(e) => e.to_string(),
    }
}

fn int(i: i64) -> Value {
    Value::Int(i)
}

#[test]
fn break_in_while() {
    assert_eq!(
        eval("fn f() {\n let mut i = 0\n while true {\n  i = i + 1\n  if i == 5 { break }\n }\n i\n}\nf()"),
        int(5)
    );
}

#[test]
fn break_in_for() {
    assert_eq!(
        eval("fn f() {\n let mut t = 0\n for x in 1..=100 {\n  if x > 4 { break }\n  t = t + x\n }\n t\n}\nf()"),
        int(10)
    );
}

#[test]
fn continue_in_for() {
    // sum of odds 1..10
    assert_eq!(
        eval("fn f() {\n let mut t = 0\n for x in 1..=10 {\n  if x % 2 == 0 { continue }\n  t = t + x\n }\n t\n}\nf()"),
        int(25)
    );
}

#[test]
fn continue_in_while() {
    assert_eq!(
        eval("fn f() {\n let mut i = 0\n let mut t = 0\n while i < 10 {\n  i = i + 1\n  if i % 2 == 0 { continue }\n  t = t + i\n }\n t\n}\nf()"),
        int(25)
    );
}

#[test]
fn nested_loops_break_inner_only() {
    let src = r#"
fn f() {
    let mut hits = 0
    for i in 0..3 {
        for j in 0..10 {
            if j == 2 { break }
            hits = hits + 1
        }
    }
    hits
}
f()
"#;
    assert_eq!(eval(src), int(6)); // 3 outer × 2 inner
}

#[test]
fn break_inside_match_arm_cleans_the_subject() {
    // the match subject sits on the operand stack when break fires; the
    // compiler must pop it or the stack corrupts
    let src = r#"
fn f() {
    let mut out = []
    for x in [1, 2, 3, 4, 5] {
        match x {
            3 => break,
            n => { out = push(out, n * 10) }
        }
    }
    out
}
f()
"#;
    assert_eq!(eval(src), eval("[10, 20]"));
}

#[test]
fn continue_inside_match_arm() {
    let src = r#"
fn f() {
    let mut out = []
    for x in 1..=6 {
        match x % 2 {
            0 => continue,
            _ => { out = push(out, x) }
        }
    }
    out
}
f()
"#;
    assert_eq!(eval(src), eval("[1, 3, 5]"));
}

#[test]
fn break_in_expression_position_cleans_partial_operands() {
    // `1000 + (if ...break...)` leaves the 1000 on the stack at break time
    let src = r#"
fn f() {
    let mut t = 0
    let mut i = 0
    while true {
        i = i + 1
        t = 1000 + (if i == 3 { break } else { i })
    }
    (i, t)
}
f()
"#;
    assert_eq!(eval(src), eval("(3, 1002)"));
}

#[test]
fn drain_loop_idiom() {
    // the chess-widget pattern: `loop { if proc_read(eng) == "" { break } }`
    let src = r#"
let queue = atom([1, 2, 3])
fn next() = match @queue {
    [] => "",
    [x, ..rest] => { reset!(queue, rest); str(x) }
}
fn drain() {
    let mut seen = []
    while true {
        let line = next()
        if line == "" { break }
        seen = push(seen, line)
    }
    seen
}
drain()
"#;
    assert_eq!(eval(src), eval("[\"1\", \"2\", \"3\"]"));
}

#[test]
fn break_outside_loop_is_a_compile_error() {
    let e = eval_err("fn f() {\n break\n}\nf()");
    assert!(e.contains("`break` outside of a loop"), "{}", e);
    let e2 = eval_err("fn f() {\n continue\n}\nf()");
    assert!(e2.contains("`continue` outside of a loop"), "{}", e2);
}

#[test]
fn lambda_inside_loop_cannot_break_the_loop() {
    let e = eval_err("fn f() {\n for x in [1] {\n  let g = () => { break }\n  g()\n }\n}\nf()");
    assert!(e.contains("`break` outside of a loop"), "{}", e);
}

#[test]
fn loop_containing_lambda_still_breaks_fine() {
    let src = r#"
fn f() {
    let mut t = 0
    for x in 1..=10 {
        let double = n => n * 2
        if x == 4 { break }
        t = t + double(x)
    }
    t
}
f()
"#;
    assert_eq!(eval(src), int(12)); // 2+4+6
}

#[test]
fn break_then_code_after_loop_runs_with_clean_stack() {
    let src = r#"
fn f() {
    let mut i = 0
    while true {
        i = i + 1
        if i == 2 { break }
    }
    let after = [i, i * 10]
    sum(after) + 100
}
f()
"#;
    assert_eq!(eval(src), int(122));
}

#[test]
fn pause_and_serialize_inside_loop_with_breaks() {
    use funct::{Cause, RunResult, StopWhen};
    let mut vm = Funct::new();
    vm.load(
        "fn f(n) {\n let mut t = 0\n for x in 1..=n {\n  if x > 1000 { break }\n  t = t + x\n }\n t\n}",
    )
    .unwrap();
    let mut st = vm.start("f", vec![int(1_000_000)]).unwrap();
    match vm.run(&mut st, StopWhen::Fuel(500)) {
        RunResult::Paused(Cause::FuelExhausted) => {}
        other => panic!("unexpected: {:?}", other),
    }
    let json = vm.save_state(&st).unwrap();
    let mut vm2 = Funct::new();
    let mut st2 = vm2.restore_state(&json).unwrap();
    match vm2.run(&mut st2, StopWhen::Never) {
        RunResult::Done(v) => assert_eq!(v, int(500500)),
        other => panic!("unexpected: {:?}", other),
    }
}
