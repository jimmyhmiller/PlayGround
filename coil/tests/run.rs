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
fn shim_convention_with_exotic_registers() {
    // A convention LLVM's CC enum cannot express: integer args pinned to rax/rdx
    // (not the SysV rdi/rsi). `isub` is order-sensitive, so a register mix-up in
    // the trampoline or call site would yield -42, not 42. Computing one arg
    // forces real register placement rather than a folded constant.
    let src = r#"
        (defcc reg2 :params [rax rdx] :ret rax
          :clobber [rax rdx rcx rsi rdi r8 r9 r10 r11] :lower shim)
        (defn sub2 :cc reg2 [(a :i64) (b :i64)] (-> :i64) (isub a b))
        (defn main [] (-> :i64)
          (let [x (iadd 40 10)] (sub2 x 8)))
    "#;
    assert_eq!(run_source(src).unwrap(), 42);

    // The shim genuinely lowers to a naked trampoline (not a normal call).
    let ir = emit_ir(src).unwrap();
    assert!(ir.contains("naked"), "expected a naked trampoline in IR:\n{ir}");
    assert!(
        ir.contains("__impl"),
        "expected a ccc impl function in IR:\n{ir}"
    );
}

#[test]
fn shim_convention_recurses() {
    // The shim path must survive recursion: factorial calls itself through the
    // exotic-register convention each time.
    let src = r#"
        (defcc reg2 :params [rax rdx] :ret rax
          :clobber [rax rdx rcx rsi rdi r8 r9 r10 r11] :lower shim)
        (defn fact :cc reg2 [(n :i64) (acc :i64)] (-> :i64)
          (if (icmp-le n 1) acc (fact (isub n 1) (imul n acc))))
        (defn main [] (-> :i64) (fact 5 1))
    "#;
    assert_eq!(run_source(src).unwrap(), 120);
}

#[test]
fn rejects_shim_without_ret() {
    let src = r#"
        (defcc bad :params [rax rdx] :lower shim)
        (defn add :cc bad [(a :i64) (b :i64)] (-> :i64) (iadd a b))
        (defn main [] (-> :i64) (add 1 2))
    "#;
    let err = run_source(src).unwrap_err();
    assert!(err.contains(":ret"), "got: {err}");
}

#[test]
fn rejects_unbound_variable() {
    let src = "(defn main [] (-> :i64) (iadd x 1))";
    let err = run_source(src).unwrap_err();
    assert!(err.contains("unbound variable"), "got: {err}");
}
