//! Atom: (atom v), @a / (deref a), (reset! a v)

use clojure::Engine;

fn eval_str(src: &str) -> String {
    let e = Engine::new();
    let v = e.eval(src);
    e.print(v)
}

#[test]
fn atom_construct_and_deref() {
    assert_eq!(eval_str("(deref (atom 42))"), "42");
}

#[test]
fn atom_deref_via_reader_macro() {
    assert_eq!(eval_str("@(atom 100)"), "100");
}

#[test]
fn atom_reset_changes_value() {
    let src = "\
        (let [a (atom 1)] \
            (reset! a 99) \
            @a)";
    assert_eq!(eval_str(src), "99");
}

#[test]
fn atom_reset_returns_new_value() {
    assert_eq!(eval_str("(let [a (atom 0)] (reset! a 7))"), "7");
}

#[test]
fn atom_holds_heap_value() {
    // Atoms can hold any value, including keywords.
    let src = "\
        (let [a (atom :first)] \
            (reset! a :second) \
            @a)";
    assert_eq!(eval_str(src), ":second");
}
