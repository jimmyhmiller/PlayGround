//! A micro-Lisp with real macros, on the fixnum value model.
//!
//! Exercises the execution/eval axis: reader (code is data), `defmacro`,
//! incremental form-by-form evaluation, and the property the batch model could
//! not provide — MACROEXPANSION RE-ENTERS COMPILED CODE (`add2` calls the
//! compiled `inc` during its own expansion).
//!
//! It also demonstrates the fix to design-tension #1: the backend is a value,
//! and backends COMPOSE. The whole program is run a second time through
//! `Traced`, a wrapper backend that observes every call — at every depth,
//! including nested runtime calls — because of open recursion through `top`.

use microlang::{LowBitModel, Runtime, Traced, TreeWalk};

fn main() {
    let mut rt = Runtime::<LowBitModel>::new();
    let cs = TreeWalk;

    // ── define macros and helpers, incrementally ────────────
    rt.eval_str(
        &cs,
        r#"
        (defmacro unless (c a b) (list 'if c b a))
        (def inc (fn (n) (+ n 1)))
        (defmacro add2 (x) (list '+ x (inc 1)))
        "#,
    );

    // ── show the compile-time data transform ────────────────
    let f = rt.read_all("(unless false 1 2)");
    let ex = rt.macroexpand(&cs, f[0]);
    println!("expand (unless false 1 2)  ->  {}", rt.print(ex));

    let f2 = rt.read_all("(add2 40)");
    let ex2 = rt.macroexpand(&cs, f2[0]);
    println!(
        "expand (add2 40)           ->  {}      ; 'inc' RAN during expansion",
        rt.print(ex2)
    );

    // ── run a program on the same runtime ───────────────────
    println!("---- program output ----");
    let prog = r#"
        (println (unless false 1 2))
        (def fact (fn (n) (if (< n 2) 1 (* n (fact (- n 1))))))
        (println (fact 5))
        (println (add2 40))
        (def map1 (fn (f xs) (if (nil? xs) xs (cons (f (first xs)) (map1 f (rest xs))))))
        (println (map1 inc (list 10 20 30)))
    "#;
    rt.eval_str(&cs, prog);

    // ── compose backends: same runtime, wrapped code space ──
    // `Traced` holds a `Box<dyn CodeSpace>` and forwards `top` down, so it
    // counts EVERY call, not just the ones the runtime initiates. This did not
    // typecheck under the old `Runtime<M, C>` design.
    println!("---- traced re-run ----");
    let traced = Traced::new(TreeWalk);
    let mut rt2 = Runtime::<LowBitModel>::new();
    rt2.eval_str(
        &traced,
        r#"
        (def fact (fn (n) (if (< n 2) 1 (* n (fact (- n 1))))))
        (fact 5)
        "#,
    );
    println!(
        "Traced observed {} calls while computing (fact 5) — including the \
         nested recursive ones.",
        traced.invoke_count()
    );
}
