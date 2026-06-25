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
