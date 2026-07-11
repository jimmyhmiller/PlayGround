//! A frontend-compiled program must run identically on the interpreter and the
//! LLVM JIT — the surface syntax lowers to plain IR, which both executors run.

use livetype::*;

const SRC: &str = r#"
    struct Account {
        balance: i64,
        fee: i64 = 0,
    }

    fn charge(a: &Account, amt: i64) -> i64 {
        return a.balance - amt;
    }

    fn main() -> i64 {
        let acct = Account { balance: 100 };
        emit(acct.balance);
        // Charge 5 a few times over, accumulating with +, -, < and a loop.
        let n = 3;
        let total = 0;
        while 0 < n {
            let c = charge(acct, 5);
            total = total + c;
            n = n - 1;
        }
        return total;
    }
"#;

#[test]
fn frontend_program_matches_on_both_executors() {
    // Interpreter.
    let mut ci = livetype_core::compile(SRC).expect("compile");
    let main_i = ci.functions["main"];
    let ai = ci.runtime.spawn(main_i, vec![]).unwrap();
    ci.runtime.run();

    // JIT (fresh compile so heaps/effects don't collide).
    let mut cj = livetype_core::compile(SRC).expect("compile");
    let main_j = cj.functions["main"];
    let mut aj = JitActor::spawn(&cj.runtime, 1, main_j, vec![]).unwrap();
    drive(&mut cj.runtime, &mut aj, false).unwrap();

    // 3 * (100 - 5) = 285, and each run emits the balance once.
    assert_eq!(
        ci.runtime.actors[&ai].status,
        ActorStatus::Complete(Value::I64(285))
    );
    assert_eq!(ci.runtime.actors[&ai].status, aj.status, "executors diverged");
    assert_eq!(ci.runtime.output, cj.runtime.output);
    assert_eq!(cj.runtime.output, vec![Value::I64(100)]);
}
