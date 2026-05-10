//! Reader + literal-pool round-trip tests for compound literals.

use clojure::Engine;

fn eval_str(src: &str) -> String {
    let e = Engine::new();
    let v = e.eval(src);
    e.print(v)
}

#[test]
fn string_literal_roundtrip() {
    assert_eq!(eval_str(r#""hello""#), r#""hello""#);
}

#[test]
fn string_with_escapes() {
    assert_eq!(eval_str(r#""a\nb""#), r#""a\nb""#);
}

#[test]
fn keyword_literal_roundtrip() {
    assert_eq!(eval_str(":foo"), ":foo");
}

#[test]
fn vector_literal_roundtrip() {
    assert_eq!(eval_str("[1 2 3]"), "[1 2 3]");
}

#[test]
fn empty_vector() {
    assert_eq!(eval_str("[]"), "[]");
}

#[test]
fn nested_vector() {
    assert_eq!(eval_str("[1 [2 3] 4]"), "[1 [2 3] 4]");
}

#[test]
fn map_literal_roundtrip() {
    assert_eq!(eval_str("{:a 1}"), "{:a 1}");
}

#[test]
fn empty_map() {
    assert_eq!(eval_str("{}"), "{}");
}

#[test]
fn set_literal_roundtrip() {
    // Sets dedupe; iteration order is insertion order (= read order).
    assert_eq!(eval_str("#{1 2 3}"), "#{1 2 3}");
}

#[test]
fn quote_of_list() {
    // `'(a b c)` should round-trip as `(a b c)` (a quoted list is a
    // value; symbols print bare without quotes).
    assert_eq!(eval_str("(quote (a b c))"), "(a b c)");
}

#[test]
fn quote_reader_macro_of_list() {
    assert_eq!(eval_str("'(1 2 3)"), "(1 2 3)");
}

#[test]
fn quoted_symbol_via_reader_macro() {
    assert_eq!(eval_str("'foo"), "foo");
}

#[test]
fn quoted_keyword_unchanged() {
    assert_eq!(eval_str("(quote :k)"), ":k");
}

#[test]
fn vector_inside_function_body() {
    // Vectors should be self-evaluating inside a function body.
    assert_eq!(
        eval_str("(def f (fn [] [1 2 3])) (f)"),
        "[1 2 3]"
    );
}

#[test]
fn keyword_inside_function_body() {
    assert_eq!(eval_str("(def f (fn [] :answer)) (f)"), ":answer");
}

#[test]
fn quoted_list_inside_function_body() {
    assert_eq!(
        eval_str("(def f (fn [] '(a b))) (f)"),
        "(a b)"
    );
}
