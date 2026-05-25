//! Checkpoint 5: same-image macros.

use clojure::Engine;

fn eval_str(src: &str) -> String {
    let mut e = Engine::new();
    let v = e.eval(src);
    e.print(v)
}

#[test]
fn trivial_macro_returns_literal() {
    // The macro body is a constant; macroexpansion replaces the
    // call form with that constant.
    let src = "(defmacro answer [] 42) (answer)";
    assert_eq!(eval_str(src), "42");
}

#[test]
fn identity_macro_passes_argument_form() {
    // The macro receives its argument as the form (here an int) and
    // returns it unchanged. compile then sees the original form.
    let src = "(defmacro identity-macro [x] x) (identity-macro 42)";
    assert_eq!(eval_str(src), "42");
}

#[test]
fn identity_macro_with_subexpr_arg() {
    // Now the argument is a list `(+ 1 2)`. The macro receives the
    // list (cons-tree pointer), returns it. macroexpansion's recursion
    // into subforms then leaves it as `(+ 1 2)`, which compiles to 3.
    let src = "(defmacro identity-macro [x] x) (identity-macro (+ 1 2))";
    assert_eq!(eval_str(src), "3");
}

#[test]
fn unless_macro_using_cons() {
    // A real macro: it builds a new form via cons.
    //   (unless c body) => (if c nil body)
    let src = "(defmacro unless [c body] \
                 (cons (quote if) (cons c (cons nil (cons body nil))))) \
               (unless false 42)";
    assert_eq!(eval_str(src), "42");
}

#[test]
fn unless_macro_truthy_branch_returns_nil() {
    let src = "(defmacro unless [c body] \
                 (cons (quote if) (cons c (cons nil (cons body nil))))) \
               (unless true 42)";
    assert_eq!(eval_str(src), "nil");
}

#[test]
fn macros_compose() {
    // unless built on if; my-when built on unless. Ensures macro
    // invocations inside macroexpansion compose correctly.
    let src = "\
        (defmacro unless [c body] \
          (cons (quote if) (cons c (cons nil (cons body nil))))) \
        (defmacro my-when [c body] \
          (cons (quote unless) (cons (cons (quote not) (cons c nil)) (cons body nil)))) \
        (my-when (< 1 2) 99)";
    assert_eq!(eval_str(src), "99");
}

#[test]
fn macro_inside_let_body_expands() {
    // Previously the expander returned let-forms unchanged, leaving
    // any nested macro calls unexpanded.
    let src = "\
        (defmacro unless [c body] \
          (cons (quote if) (cons c (cons nil (cons body nil))))) \
        (let [x 5] (unless false x))";
    assert_eq!(eval_str(src), "5");
}

#[test]
fn macro_inside_if_branches_expands() {
    let src = "\
        (defmacro double [x] (cons (quote *) (cons x (cons 2 nil)))) \
        (if true (double 21) (double 0))";
    assert_eq!(eval_str(src), "42");
}

#[test]
fn macro_inside_do_expands() {
    let src = "\
        (defmacro answer [] 42) \
        (do (answer))";
    assert_eq!(eval_str(src), "42");
}

#[test]
fn macro_inside_fn_body_expands() {
    let src = "\
        (defmacro identity-mac [x] x) \
        (def f (fn [n] (identity-mac (* n n)))) \
        (f 7)";
    assert_eq!(eval_str(src), "49");
}

#[test]
fn macro_inside_let_value_expands() {
    let src = "\
        (defmacro answer [] 42) \
        (let [x (answer)] x)";
    assert_eq!(eval_str(src), "42");
}

#[test]
fn fn_calls_still_work_when_macros_exist() {
    // Defining a macro shouldn't break ordinary fn dispatch.
    let src = "\
        (defmacro identity-macro [x] x) \
        (def square (fn [x] (* x x))) \
        (square 7)";
    assert_eq!(eval_str(src), "49");
}
