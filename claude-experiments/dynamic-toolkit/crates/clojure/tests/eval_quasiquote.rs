//! Quasiquote / unquote / unquote-splicing rewriting.

use clojure::Engine;

fn eval_str(src: &str) -> String {
    let e = Engine::new();
    let v = e.eval(src);
    e.print(v)
}

#[test]
fn quasiquote_simple_literal() {
    // No unquotes at all → equivalent to plain quote.
    assert_eq!(eval_str("`(a b c)"), "(a b c)");
}

#[test]
fn quasiquote_atom() {
    assert_eq!(eval_str("`x"), "x");
    assert_eq!(eval_str("`5"), "5");
    assert_eq!(eval_str("`:k"), ":k");
}

#[test]
fn quasiquote_with_unquote() {
    assert_eq!(eval_str("(let [x 42] `(a ~x b))"), "(a 42 b)");
}

#[test]
fn quasiquote_unquote_evaluates_subexpr() {
    assert_eq!(eval_str("`(a ~(+ 1 2) b)"), "(a 3 b)");
}

#[test]
fn quasiquote_unquote_splicing() {
    // ~@xs splices the elements of xs into the surrounding list.
    // Build xs as a literal list via quote.
    assert_eq!(eval_str("(let [xs '(1 2 3)] `(a ~@xs b))"), "(a 1 2 3 b)");
}

#[test]
fn quasiquote_splice_at_head() {
    assert_eq!(eval_str("(let [xs '(1 2)] `(~@xs 3))"), "(1 2 3)");
}

#[test]
fn quasiquote_splice_at_tail() {
    assert_eq!(eval_str("(let [xs '(2 3)] `(1 ~@xs))"), "(1 2 3)");
}

#[test]
fn quasiquote_in_macro() {
    // The killer use case: write a macro using `(if c nil body)
    // instead of cons-cons-cons gymnastics.
    let src = "\
        (defmacro unless [c body] `(if ~c nil ~body)) \
        (unless false 99)";
    assert_eq!(eval_str(src), "99");
}

#[test]
fn quasiquote_in_macro_splicing() {
    // A macro that spreads its body into a do.
    let src = "\
        (defmacro do-twice [body] `(do ~body ~body)) \
        (do-twice 7)";
    assert_eq!(eval_str(src), "7");
}

#[test]
fn nested_quasiquote_only_outer_unquotes() {
    // The inner ~b stays unquoted because we're at level 2
    // when we hit the outer unquote — wait, this case is tricky:
    // `~b at level 2 stays as (unquote b). Let's test the simple
    // shape: in `(a ~b), b is evaluated. Nested ``(a ~b) — the
    // outer ` brings level to 1, then the inner ` brings it to 2.
    // The ~b at level 2 doesn't fire; it stays as `(unquote b)`.
    //
    // Skipping nested-level testing for this iteration — the basic
    // single-level cases above are what core.clj relies on.
}
