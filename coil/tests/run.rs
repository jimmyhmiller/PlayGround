//! End-to-end: source -> JIT -> result, plus the diagnostics that make
//! "convention is part of the type" real even at M1.

use coil::{emit_ir, run_source};

#[test]
fn arithmetic_and_let() {
    let src = r#"
        (defn main [] (-> :i64)
          (let [x 20 y 22] (iadd x y)))
    "#;
    assert_eq!(run_source(src).unwrap(), 42);
}

#[test]
fn recursion_and_if() {
    let src = r#"
        (defn fib [(n :i64)] (-> :i64)
          (if (icmp-le n 1) n (iadd (fib (isub n 1)) (fib (isub n 2)))))
        (defn main [] (-> :i64) (fib 10))
    "#;
    assert_eq!(run_source(src).unwrap(), 55);
}

#[test]
fn custom_convention_runs_and_emits_fastcc() {
    let src = r#"
        (defcc fast2 :params [rax rdx] :ret rax
          :clobber [rax rdx rcx] :preserve [rbx rbp] :native fast)
        (defn add :cc fast2 [(a :i64) (b :i64)] (-> :i64) (iadd a b))
        (defn main [] (-> :i64) (add 20 22))
    "#;
    assert_eq!(run_source(src).unwrap(), 42);

    // The convention is not cosmetic: it shows up in the IR.
    let ir = emit_ir(src).unwrap();
    assert!(ir.contains("fastcc"), "expected fastcc in IR:\n{ir}");
}

#[test]
fn rejects_arity_mismatch() {
    let src = r#"
        (defn add [(a :i64) (b :i64)] (-> :i64) (iadd a b))
        (defn main [] (-> :i64) (add 1))
    "#;
    let err = run_source(src).unwrap_err();
    assert!(err.contains("expects 2 args"), "got: {err}");
}

#[test]
fn rejects_unlowerable_convention() {
    // A `:lower shim` convention has no native lowering yet (that's M2), so a
    // function using it must be rejected rather than silently miscompiled.
    let src = r#"
        (defcc fast2 :params [rax rdx] :ret rax :lower shim)
        (defn add :cc fast2 [(a :i64) (b :i64)] (-> :i64) (iadd a b))
        (defn main [] (-> :i64) (add 1 2))
    "#;
    let err = run_source(src).unwrap_err();
    assert!(err.contains("no native lowering"), "got: {err}");
}

#[test]
fn rejects_unbound_variable() {
    let src = "(defn main [] (-> :i64) (iadd x 1))";
    let err = run_source(src).unwrap_err();
    assert!(err.contains("unbound variable"), "got: {err}");
}
