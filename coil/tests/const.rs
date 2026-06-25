//! Named scalar constants — `(const NAME VALUE)` / `(const NAME TYPE VALUE)`. A
//! reference elaborates to the literal inline (zero runtime cost): an untyped
//! const re-enters width inference exactly like writing the literal, a typed
//! const pins the width and fit-checks at definition. Consts live in a flat
//! global namespace and are shadowed by locals. (C enum constants / `#define`s
//! lower to these — see tests/cimport.rs.)

mod common;
use common::build_and_run;

#[test]
fn untyped_const_used_as_bare_name() {
    // RED is usable as a value (not just a compile-time macro binding).
    assert_eq!(build_and_run("(module a)\n(const RED 5)\n(defn main [] (-> i64) (iadd RED 37))"), 42);
}

#[test]
fn typed_const_reports_its_declared_width() {
    let src = "(module a)\n(const ANSWER i64 42)\n(defn main [] (-> i64) ANSWER)";
    assert_eq!(build_and_run(src), 42);
}

#[test]
fn untyped_const_adopts_context_width() {
    // An untyped const is the literal: it narrows into a u8 store like `200` would.
    let src = "(module a)\n(const FLAG 200)\n\
               (defn main [] (-> i64)\n\
                 (let [(mut b) (zeroed u8)]\n\
                   (store! (mut b) FLAG)\n\
                   (cast i64 (load (mut b)))))";
    assert_eq!(build_and_run(src), 200);
}

#[test]
fn float_const() {
    let src = "(module a)\n(const HALF f64 0.5)\n\
               (defn main [] (-> i64) (cast i64 (fmul HALF 84.0)))";
    assert_eq!(build_and_run(src), 42);
}

#[test]
fn local_shadows_const() {
    // A parameter named the same as a const wins (locals are resolved first).
    let src = "(module a)\n(const X 99)\n\
               (defn id [(X i64)] (-> i64) X)\n\
               (defn main [] (-> i64) (id 7))";
    assert_eq!(build_and_run(src), 7);
}

#[test]
fn duplicate_const_is_rejected() {
    let err = coil::emit_ir("(module a)\n(const X 5)\n(const X 6)\n(defn main [] (-> i64) X)")
        .unwrap_err();
    assert!(err.contains("more than once"), "got:\n{err}");
}

#[test]
fn const_value_that_does_not_fit_is_rejected() {
    let err = coil::emit_ir("(module a)\n(const BIG u8 999)\n(defn main [] (-> i64) (cast i64 BIG))")
        .unwrap_err();
    assert!(err.contains("does not fit"), "got:\n{err}");
}

#[test]
fn const_value_kind_must_match_declared_type() {
    let err = coil::emit_ir("(module a)\n(const P i32 3.14)\n(defn main [] (-> i64) (cast i64 P))")
        .unwrap_err();
    assert!(err.contains("cannot have type"), "got:\n{err}");
}

// ---- computed consts: a const value can be any compile-time expression --------

#[test]
fn const_computed_by_a_function_at_compile_time() {
    // The value calls a real defn; it's evaluated by the comptime interpreter.
    let src = "(module a)\n\
        (defn fact [(n i64)] (-> i64) (if (icmp-le n 0) 1 (imul n (fact (isub n 1)))))\n\
        (const FACT5 (fact 5))\n\
        (defn main [] (-> i64) (isub FACT5 78))"; // 120 - 78 = 42
    assert_eq!(build_and_run(src), 42);
}

#[test]
fn const_can_reference_an_earlier_const() {
    let src = "(module a)\n\
        (const BASE (iadd 20 1))\n\
        (const DOUBLE (imul BASE 2))\n\
        (defn main [] (-> i64) DOUBLE)"; // 42
    assert_eq!(build_and_run(src), 42);
}

#[test]
fn computed_const_folds_to_a_constant() {
    let ir = coil::emit_ir(
        "(module a)\n\
         (defn fact [(n i64)] (-> i64) (if (icmp-le n 0) 1 (imul n (fact (isub n 1)))))\n\
         (const FACT5 (fact 5))\n\
         (defn main [] (-> i64) FACT5)",
    )
    .unwrap();
    let main = ir.split("@main").nth(1).unwrap_or("");
    let body = &main[..main.find('}').unwrap_or(main.len())];
    assert!(body.contains("ret i64 120"), "const not folded:\n{body}");
}

// ---- aggregate consts become static globals (compile-time data tables) --------

#[test]
fn array_const_is_a_static_table() {
    // A lookup table built at compile time, read at runtime from the global.
    let src = "(module a)\n\
        (defn squares [] (-> (array i64 8))\n\
          (let [(mut t) (zeroed (array i64 8)) (mut i) 0]\n\
            (loop (if (>= (load i) 8) (break)\n\
                    (do (store! (index (mut t) (load i)) (* (load i) (load i)))\n\
                        (store! i (+ (load i) 1)))))\n\
            (load t)))\n\
        (const SQUARES (squares))\n\
        (defn main [] (-> i64) (+ (load (index SQUARES 5)) (load (index SQUARES 6))))"; // 25 + 36 = 61
    assert_eq!(build_and_run(src), 61);
}

#[test]
fn struct_const_is_a_static() {
    let src = "(module a)\n\
        (defstruct Cfg [(w i64) (h i64)])\n\
        (defn mk [] (-> Cfg) (let [(mut c) (zeroed Cfg)] (store! (field (mut c) w) 80) (store! (field (mut c) h) 25) (load c)))\n\
        (const CFG (mk))\n\
        (defn main [] (-> i64) (+ (load (field CFG w)) (load (field CFG h))))"; // 105; %256=105
    assert_eq!(build_and_run(src), 105);
}

#[test]
fn array_const_emits_a_constant_global() {
    let ir = coil::emit_ir(
        "(module a)\n\
         (defn t [] (-> (array i64 3)) (let [(mut a) (zeroed (array i64 3))] (store! (index (mut a) 0) 9) (load a)))\n\
         (const T (t))\n\
         (defn main [] (-> i64) (load (index T 0)))",
    )
    .unwrap();
    assert!(ir.contains("const.a.T") || ir.contains("constant"), "no constant global:\n{ir}");
}
