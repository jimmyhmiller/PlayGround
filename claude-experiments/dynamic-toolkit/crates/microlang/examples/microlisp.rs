//! A micro-Lisp with real macros, on the fixnum value model.
//!
//! This exercises the execution/eval axis: reader (code is data), `defmacro`,
//! incremental form-by-form evaluation, and the load-bearing property the batch
//! "declare-all-then-define-all" model could not provide — MACROEXPANSION
//! RE-ENTERS COMPILED CODE. The `add2` macro below calls the user function
//! `inc` *during its own expansion*: compiled code running while we are still
//! compiling the caller. Same `CodeSpace::invoke` used at runtime and at
//! macro-expansion time.

use microlang::{LowBitModel, Runtime};

fn main() {
    let mut rt = Runtime::<LowBitModel>::new();

    // ── define macros and helpers, incrementally ────────────
    // Each form is analyzed against the environment the PREVIOUS forms built.
    // `add2`'s expander references `inc`, a fn defined a line earlier.
    rt.eval_str(
        r#"
        (defmacro unless (c a b) (list 'if c b a))
        (def inc (fn (n) (+ n 1)))
        (defmacro add2 (x) (list '+ x (inc 1)))
        "#,
    );

    // ── show the compile-time data transform ────────────────
    let f = rt.read_all("(unless false 1 2)");
    let ex = rt.macroexpand(f[0]);
    println!("expand (unless false 1 2)  ->  {}", rt.print(ex));

    let f2 = rt.read_all("(add2 40)");
    let ex2 = rt.macroexpand(f2[0]);
    println!(
        "expand (add2 40)           ->  {}      ; 'inc' RAN during expansion",
        rt.print(ex2)
    );

    // ── run a full program on the same runtime ──────────────
    println!("---- program output ----");
    rt.eval_str(
        r#"
        (println (unless false 1 2))
        (def fact (fn (n) (if (< n 2) 1 (* n (fact (- n 1))))))
        (println (fact 5))
        (println (add2 40))
        (def map1 (fn (f xs) (if (nil? xs) xs (cons (f (first xs)) (map1 f (rest xs))))))
        (println (map1 inc (list 10 20 30)))
        "#,
    );
}
