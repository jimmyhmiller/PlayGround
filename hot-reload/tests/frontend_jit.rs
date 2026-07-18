//! A frontend-compiled program must run identically on the interpreter-only
//! engine and the always-JIT engine — the surface syntax lowers to plain IR,
//! and both configurations are the same loop.

use livetype::*;

const SRC: &str = r#"
    struct Account {
        balance: i64,
        fee: i64 = 0,
    }

    fn charge(a: Account, amt: i64) -> i64 {
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
fn frontend_program_matches_on_both_configurations() {
    // Interpreter-only engine.
    let ci = livetype_core::compile(SRC).expect("compile");
    let interp = ci.engine.run_call(ci.functions["main"], vec![]);

    // Always-JIT engine (fresh compile so heaps/effects don't collide).
    let cj = livetype_core::compile_on(SRC, jit_engine(0)).expect("compile");
    let jit = cj.engine.run_call(cj.functions["main"], vec![]);

    // 3 * (100 - 5) = 285, and each run emits the balance once.
    assert_eq!(interp, Outcome::Complete(Value::I64(285)));
    assert_eq!(interp, jit, "configurations diverged");
    assert_eq!(ci.engine.output(), cj.engine.output());
    assert_eq!(cj.engine.output(), vec![Value::I64(100)]);
}
