//! Stage 3, step 1: code as a first-class comptime value. `(quote FORM)` yields a
//! `Code` value; the `code-*` ops inspect it at compile time (count/nth/sym/int +
//! list?/sym?/int? predicates). `Code` is comptime-only — it has no runtime
//! representation, so it can only be used inside `comptime`/`const`/reflection.

mod common;
use common::build_and_run;

#[test]
fn quote_and_count() {
    assert_eq!(build_and_run("(module a)\n(defn main [] (-> i64) (comptime (code-count (quote (a b c d)))))"), 4);
}

#[test]
fn nth_int_and_count() {
    let code = build_and_run(
        "(module a)\n(defn main [] (-> i64)\n\
         (comptime (iadd (iadd (code-int (code-nth (quote (add 10 20)) 1))\n\
                               (code-int (code-nth (quote (add 10 20)) 2)))\n\
                         (code-count (quote (add 10 20))))))", // 10+20+3
    );
    assert_eq!(code, 33);
}

#[test]
fn predicates() {
    let code = build_and_run(
        "(module a)\n(defn main [] (-> i64)\n\
         (comptime (iadd (if (code-list? (quote (a b))) 1 0)\n\
                         (iadd (if (code-sym? (code-nth (quote (a b)) 0)) 10 0)\n\
                               (if (code-int? (quote 7)) 100 0)))))", // 1+10+100
    );
    assert_eq!(code, 111);
}

#[test]
fn code_value_cant_be_a_runtime_value() {
    // A `Code` value escaping comptime is a clear error, not a miscompile.
    let err = coil::emit_ir("(module a)\n(defn main [] (-> i64) (comptime (quote (a b))))").unwrap_err();
    assert!(err.contains("type code"), "got:\n{err}");
}

// ---- Stage 3, step 2: quasiquote (the build side) -----------------------------

#[test]
fn quasiquote_builds_and_splices() {
    // `(iadd 10 ~x) builds (iadd 10 <x>); unquote splices a comptime value.
    let code = build_and_run(
        "(module a)\n(defn main [] (-> i64)\n\
         (comptime (let [built `(iadd 10 ~(code-int (code-nth (quote (a 20)) 1)))]\n\
           (iadd (code-count built) (code-int (code-nth built 2))))))", // count 3 + 20 = 23
    );
    assert_eq!(code, 23);
}

#[test]
fn quasiquote_nested_lists() {
    // a nested template with an unquote deep inside
    let code = build_and_run(
        "(module a)\n(defn main [] (-> i64)\n\
         (comptime (let [c `(a (b ~(quote 7)) d)]\n\
           (code-int (code-nth (code-nth c 1) 1)))))", // (b 7) → 7
    );
    assert_eq!(code, 7);
}

#[test]
fn unquote_outside_quasiquote_errors() {
    let err = coil::emit_ir("(module a)\n(defn main [] (-> i64) (comptime (code-count ~x)))").unwrap_err();
    assert!(err.contains("unquote"), "got:\n{err}");
}
