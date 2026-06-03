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
    assert_eq!(eval_str("(def f (fn [] [1 2 3])) (f)"), "[1 2 3]");
}

#[test]
fn keyword_inside_function_body() {
    assert_eq!(eval_str("(def f (fn [] :answer)) (f)"), ":answer");
}

#[test]
fn string_equality_structural() {
    // Two identical string literals allocate distinct heap objects,
    // but `=` compares byte content.
    assert_eq!(eval_str(r#"(= "foo" "foo")"#), "true");
    assert_eq!(eval_str(r#"(= "foo" "bar")"#), "false");
    assert_eq!(eval_str(r#"(= "" "")"#), "true");
}

#[test]
fn string_equality_with_escapes() {
    assert_eq!(eval_str(r#"(= "a\nb" "a\nb")"#), "true");
    assert_eq!(eval_str(r#"(= "a\nb" "a\tb")"#), "false");
}

#[test]
fn string_vs_other_types_inequal() {
    assert_eq!(eval_str(r#"(= "foo" :foo)"#), "false");
    assert_eq!(eval_str(r#"(= "1" 1)"#), "false");
}

#[test]
fn keyword_equality() {
    // After interning, two `:foo` literals share identity, so the
    // bitwise compare in clj_eq returns true.
    assert_eq!(eval_str("(= :foo :foo)"), "true");
    assert_eq!(eval_str("(= :foo :bar)"), "false");
}

#[test]
fn keyword_equality_via_let_binding() {
    // First binding holds a keyword; comparing against a fresh
    // literal of the same name must still hit the interned object.
    assert_eq!(eval_str("(let [k :marker] (= k :marker))"), "true");
}

#[test]
fn keyword_in_set_then_in_set() {
    // Sets dedupe by bitwise equality, so an identical-name keyword
    // appearing twice should collapse to one entry.
    assert_eq!(eval_str("#{:a :b :a :a}"), "#{:a :b}");
}

#[test]
fn many_strings_in_one_vector() {
    // Stress-test reader rooting: a vector containing enough heap
    // allocations that GC may fire mid-read. With proper rooting,
    // every element must still be intact when the vector is built.
    // Without rooting, earlier elements become stale pointers and
    // the printed output is garbled (or the process crashes).
    let mut src = String::from("[");
    for i in 0..200 {
        if i > 0 {
            src.push(' ');
        }
        src.push_str(&format!("\"s{}\"", i));
    }
    src.push(']');
    let out = eval_str(&src);
    // Sanity: result should round-trip identically.
    assert!(out.starts_with("[\"s0\""), "got: {}", out);
    assert!(out.ends_with("\"s199\"]"), "got: {}", out);
}

#[test]
fn many_keywords_in_one_set() {
    let mut src = String::from("#{");
    for i in 0..200 {
        if i > 0 {
            src.push(' ');
        }
        src.push_str(&format!(":k{}", i));
    }
    src.push('}');
    let out = eval_str(&src);
    assert!(out.starts_with("#{:k0"), "got: {}", out);
    assert!(out.contains(":k199"), "got: {}", out);
}

#[test]
fn quoted_list_inside_function_body() {
    assert_eq!(eval_str("(def f (fn [] '(a b))) (f)"), "(a b)");
}
