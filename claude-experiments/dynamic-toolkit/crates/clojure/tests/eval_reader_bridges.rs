//! Reader-collection accessor externs — the bridges core.clj's
//! `extend-type __ReaderXxx` blocks call to read the underlying
//! storage of the reader's literal collections.

use clojure::Engine;

fn eval_str(src: &str) -> String {
    let e = Engine::new();
    let v = e.eval(src);
    e.print(v)
}

// ── List ──

#[test]
fn list_first() {
    assert_eq!(eval_str("(__reader_list_first '(10 20 30))"), "10");
}

#[test]
fn list_rest() {
    assert_eq!(eval_str("(__reader_list_rest '(10 20 30))"), "(20 30)");
}

#[test]
fn list_count() {
    assert_eq!(eval_str("(__reader_list_count '(10 20 30))"), "3");
    assert_eq!(eval_str("(__reader_list_count '())"), "0");
}

#[test]
fn list_nth() {
    assert_eq!(eval_str("(__reader_list_nth '(10 20 30) 0)"), "10");
    assert_eq!(eval_str("(__reader_list_nth '(10 20 30) 2)"), "30");
}

// ── Vector ──

#[test]
fn vector_count() {
    assert_eq!(eval_str("(__reader_vector_count [10 20 30])"), "3");
    assert_eq!(eval_str("(__reader_vector_count [])"), "0");
}

#[test]
fn vector_nth() {
    assert_eq!(eval_str("(__reader_vector_nth [10 20 30] 0)"), "10");
    assert_eq!(eval_str("(__reader_vector_nth [10 20 30] 2)"), "30");
}

#[test]
fn vector_first_and_rest() {
    assert_eq!(eval_str("(__reader_vector_first [10 20 30])"), "10");
    assert_eq!(eval_str("(__reader_vector_rest [10 20 30])"), "[20 30]");
    assert_eq!(eval_str("(__reader_vector_rest [10])"), "[]");
}

// ── Set ──

#[test]
fn set_count_and_contains() {
    assert_eq!(eval_str("(__reader_set_count #{1 2 3})"), "3");
    assert_eq!(eval_str("(__reader_set_contains #{1 2 3} 2)"), "true");
    assert_eq!(eval_str("(__reader_set_contains #{1 2 3} 99)"), "false");
}

#[test]
fn set_get_indexed() {
    // Insertion order is preserved by our flat backing.
    assert_eq!(eval_str("(__reader_set_get #{1 2 3} 0)"), "1");
    assert_eq!(eval_str("(__reader_set_get #{1 2 3} 2)"), "3");
}

#[test]
fn set_conj_adds_when_absent() {
    assert_eq!(
        eval_str("(__reader_set_count (__reader_set_conj #{1 2} 3))"),
        "3"
    );
}

#[test]
fn set_conj_dedups_when_present() {
    assert_eq!(
        eval_str("(__reader_set_count (__reader_set_conj #{1 2} 1))"),
        "2"
    );
}

#[test]
fn set_contains_string_structurally() {
    // Strings aren't interned; structural compare must hit.
    assert_eq!(
        eval_str(r#"(__reader_set_contains #{"a" "b"} "a")"#),
        "true"
    );
}

// ── Map ──

#[test]
fn map_count() {
    assert_eq!(eval_str("(__reader_map_count {:a 1 :b 2})"), "2");
    assert_eq!(eval_str("(__reader_map_count {})"), "0");
}

#[test]
fn map_lookup_present() {
    assert_eq!(
        eval_str("(__reader_map_lookup {:a 1 :b 2} :a :missing)"),
        "1"
    );
}

#[test]
fn map_lookup_absent_returns_default() {
    assert_eq!(
        eval_str("(__reader_map_lookup {:a 1} :nope :default)"),
        ":default"
    );
}

#[test]
fn map_keys() {
    assert_eq!(eval_str("(__reader_map_keys {:a 1 :b 2})"), "(:a :b)");
}

#[test]
fn map_assoc_overwrites() {
    assert_eq!(
        eval_str("(__reader_map_lookup (__reader_map_assoc {:a 1} :a 99) :a :nope)"),
        "99"
    );
}

#[test]
fn map_assoc_extends() {
    assert_eq!(
        eval_str(
            "(__reader_map_count (__reader_map_assoc {:a 1} :b 2))"
        ),
        "2"
    );
}
