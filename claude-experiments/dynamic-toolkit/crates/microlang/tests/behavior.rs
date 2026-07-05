//! Locks the two axes the sketch is about.

use microlang::{LowBitModel, NanBoxModel, Runtime, Val, ValueModel};

fn eval1<M: ValueModel>(src: &str) -> String {
    let mut rt = Runtime::<M>::new();
    let r = rt.eval_str(src);
    rt.print(r)
}

/// Same source, same engine, correct answer under both value models.
#[test]
fn arithmetic_is_model_independent() {
    assert_eq!(eval1::<LowBitModel>("(+ (* 2 3) (* 4 5))"), "26");
    assert_eq!(eval1::<NanBoxModel>("(+ (* 2 3) (* 4 5))"), "26");
    assert_eq!(eval1::<LowBitModel>("(+ 1.5 2.5)"), "4.0");
    assert_eq!(eval1::<NanBoxModel>("(+ 1.5 2.5)"), "4.0");
}

/// The value axis, measured: the immediate category decides who boxes.
#[test]
fn immediacy_decides_allocation() {
    fn allocs_for<M: ValueModel>(src: &str) -> u64 {
        let mut rt = Runtime::<M>::new();
        let forms = rt.read_all(src);
        let before = rt.allocs;
        for f in forms {
            rt.eval_top(f);
        }
        rt.allocs - before
    }
    // integers: free on LowBit, boxed on NanBox
    assert_eq!(allocs_for::<LowBitModel>("(+ (* 2 3) (* 4 5))"), 0);
    assert!(allocs_for::<NanBoxModel>("(+ (* 2 3) (* 4 5))") > 0);
    // floats: the exact opposite
    assert!(allocs_for::<LowBitModel>("(+ (* 2.0 3.0) (* 4.0 5.0))") > 0);
    assert_eq!(allocs_for::<NanBoxModel>("(+ (* 2.0 3.0) (* 4.0 5.0))"), 0);
}

/// The eval axis: macros expand, and a macro may run compiled code.
#[test]
fn macros_and_reentrancy() {
    let mut rt = Runtime::<LowBitModel>::new();
    rt.eval_str(
        r#"
        (defmacro unless (c a b) (list 'if c b a))
        (def inc (fn (n) (+ n 1)))
        (defmacro add2 (x) (list '+ x (inc 1)))
        "#,
    );
    // expansion is a pure data transform
    let f = rt.read_all("(unless false 1 2)");
    let ex = rt.macroexpand(f[0]);
    assert_eq!(rt.print(ex), "(if false 2 1)");
    // 'inc' (compiled) runs during 'add2' expansion
    let g = rt.read_all("(add2 40)");
    let ex2 = rt.macroexpand(g[0]);
    assert_eq!(rt.print(ex2), "(+ 40 2)");
    // and the whole thing evaluates
    let r = rt.eval_str("(add2 40)");
    assert_eq!(rt.print(r), "42");
}

#[test]
fn recursion_and_higher_order() {
    assert_eq!(
        eval1::<LowBitModel>("(def f (fn (n) (if (< n 2) 1 (* n (f (- n 1)))))) (f 5)"),
        "120"
    );
    assert_eq!(
        eval1::<LowBitModel>(
            "(def m (fn (g xs) (if (nil? xs) xs (cons (g (first xs)) (m g (rest xs))))))
             (def inc (fn (n) (+ n 1)))
             (m inc (list 10 20 30))"
        ),
        "(11 21 31)"
    );
}

#[test]
fn equality_is_structural() {
    assert_eq!(eval1::<LowBitModel>("(= (list 1 2 3) (list 1 2 3))"), "true");
    assert_eq!(eval1::<NanBoxModel>("(= (list 1 2 3) (list 1 2 3))"), "true");
    assert_eq!(eval1::<LowBitModel>("(= (list 1 2) (list 1 2 3))"), "false");
}

/// Nil bootstrapping helper used by examples.
#[test]
fn nil_encodes() {
    let mut rt = Runtime::<LowBitModel>::new();
    let n = rt.encode(Val::Nil);
    assert_eq!(rt.print(n), "nil");
}
