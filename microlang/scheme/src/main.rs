//! A tiny Scheme, running on the core's execution tiers. The frontend is this
//! crate; everything it executes on lives in `microlang` (the core).

use microlang::{BytecodeVm, LowBitModel, Runtime, TreeWalk};
use scheme::run;

const FULL: &str = r#"
    (define (sq n) (* n n))
    (define (sum-to n) (if (< n 1) 0 (+ n (sum-to (- n 1)))))
    (let ((a (sq 6)) (b (sum-to 10)))
      (cond ((< a b) (display a))
            (else    (display b))))
    (+ (sq 6) (sum-to 10))
"#;

const ARITH: &str = "(define (fact n) (if (< n 2) 1 (* n (fact (- n 1))))) (fact 6)";

fn main() {
    // define / let / cond / recursion on the interpreter.
    let mut rt = Runtime::<LowBitModel>::new();
    print!("define/let/cond/recursion  (display prints then) result = ");
    let r = run(&mut rt, &TreeWalk, FULL);
    println!("{}", rt.print(r));

    // arithmetic-only Scheme also runs on the bytecode emit tier, unchanged.
    let mut rt2 = Runtime::<LowBitModel>::new();
    let interp_r = run(&mut rt2, &TreeWalk, ARITH);
    let mut rt3 = Runtime::<LowBitModel>::new();
    let vm = BytecodeVm::<LowBitModel>::new();
    let jit_r = run(&mut rt3, &vm, ARITH);
    println!(
        "(fact 6):  interpreter => {}   bytecode VM => {}",
        rt2.print(interp_r),
        rt3.print(jit_r)
    );

    println!(
        "\nThe Scheme frontend (reader + desugar) lives entirely in the `scheme`\n\
         crate and touches only the core's public API. The core has no idea Scheme\n\
         exists — the same tiers run it that ran the core's own Lisp."
    );
}
