//! End-to-end: parse Rust-flavored source, compile to IR, run it.

use livetype_core::*;

/// Compile `source`, run `entry` with no args on the interpreter, and return
/// (final status, emitted effects).
fn run(source: &str, entry: &str) -> (ActorStatus, Vec<Value>) {
    let mut compiled = compile(source).expect("compile");
    let id = compiled.functions[entry];
    let actor = compiled.runtime.spawn(id, vec![]).expect("spawn");
    compiled.runtime.run();
    (
        compiled.runtime.actors[&actor].status.clone(),
        compiled.runtime.output.clone(),
    )
}

#[test]
fn structs_fields_and_arithmetic() {
    let src = r#"
        struct Account {
            balance: i64,
            fee: i64 = 0,
        }

        fn charge(a: Account, amt: i64) -> i64 {
            let b = a.balance;
            return b - amt;
        }

        fn main() -> i64 {
            let acct = Account { balance: 100 };
            emit(acct.balance);
            let charged = charge(acct, 5);
            return charged + 1;
        }
    "#;
    let (status, out) = run(src, "main");
    assert_eq!(status, ActorStatus::Complete(Value::I64(96)));
    assert_eq!(out, vec![Value::I64(100)]);
}

#[test]
fn while_loop_and_conditionals() {
    // Sum 3 + 2 + 1 into `total`, emitting each counter.
    let src = r#"
        fn main() -> i64 {
            let n = 3;
            let total = 0;
            while 0 < n {
                emit(n);
                total = total + n;
                n = n - 1;
            }
            return total;
        }
    "#;
    let (status, out) = run(src, "main");
    assert_eq!(status, ActorStatus::Complete(Value::I64(6)));
    assert_eq!(out, vec![Value::I64(3), Value::I64(2), Value::I64(1)]);
}

#[test]
fn if_else_picks_a_branch() {
    let src = r#"
        fn pick(x: i64) -> i64 {
            if x < 10 {
                return 111;
            } else {
                return 222;
            }
        }
        fn main() -> i64 {
            return pick(5);
        }
    "#;
    let (status, _) = run(src, "main");
    assert_eq!(status, ActorStatus::Complete(Value::I64(111)));
}

#[test]
fn nested_structs_by_reference() {
    let src = r#"
        struct Money { cents: i64 }
        struct Account { balance: Money }

        fn main() -> i64 {
            let m = Money { cents: 250 };
            let a = Account { balance: m };
            return a.balance.cents;
        }
    "#;
    let (status, _) = run(src, "main");
    assert_eq!(status, ActorStatus::Complete(Value::I64(250)));
}

#[test]
fn a_type_error_is_a_compile_error() {
    // `charge` subtracts a struct reference from an int — rejected at compile.
    let src = r#"
        struct Account { balance: i64 }
        fn bad(a: Account) -> i64 {
            return a - 1;
        }
        fn main() -> i64 { return 0; }
    "#;
    let err = compile(src).unwrap_err();
    assert!(err.contains("bad"), "expected an error mentioning `bad`, got: {err}");
}

#[test]
fn recursion_works() {
    // Countdown to zero via self-recursion: fact-shaped, but returns 0.
    let src = r#"
        fn countdown(n: i64) -> i64 {
            if n < 1 {
                return 0;
            }
            emit(n);
            return countdown(n - 1);
        }
        fn main() -> i64 {
            return countdown(3);
        }
    "#;
    let (status, out) = run(src, "main");
    assert_eq!(status, ActorStatus::Complete(Value::I64(0)));
    assert_eq!(out, vec![Value::I64(3), Value::I64(2), Value::I64(1)]);
}

#[test]
fn mutual_recursion_works() {
    let src = r#"
        fn is_even(n: i64) -> bool {
            if n == 0 { return true; }
            return is_odd(n - 1);
        }
        fn is_odd(n: i64) -> bool {
            if n == 0 { return false; }
            return is_even(n - 1);
        }
        fn main() -> bool {
            return is_even(4);
        }
    "#;
    let (status, _) = run(src, "main");
    assert_eq!(status, ActorStatus::Complete(Value::Bool(true)));
}

#[test]
fn tail_expression_returns() {
    // No explicit `return` — the trailing expression is the result.
    let src = r#"
        fn add(a: i64, b: i64) -> i64 { a + b }
        fn main() -> i64 {
            let x = add(2, 3) * 4;
            x - 2
        }
    "#;
    let (status, _) = run(src, "main");
    assert_eq!(status, ActorStatus::Complete(Value::I64(18)));
}

#[test]
fn all_the_operators() {
    let src = r#"
        fn main() -> i64 {
            let a = 2 + 3 * 4;      // 14
            let b = a - 1;          // 13
            if a == 14 { emit(1); }
            if a != 99 { emit(2); }
            if b <= 13 { emit(3); }
            if a >= 14 { emit(4); }
            if !(a < 10) { emit(5); }
            a + b
        }
    "#;
    let (status, out) = run(src, "main");
    assert_eq!(status, ActorStatus::Complete(Value::I64(27)));
    assert_eq!(out, vec![Value::I64(1), Value::I64(2), Value::I64(3), Value::I64(4), Value::I64(5)]);
}
