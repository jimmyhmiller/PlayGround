//! Structured, definition-checked traits: `deftrait` / `impl` / bounded generics
//! (`(T Trait)`), with static (monomorphized) dispatch — concrete calls resolve
//! to the impl, generic bodies are checked against their bounds at the definition,
//! and the type-arg's impl is verified at the call site. `Self` is passed by value
//! (`(a Self)`); codegen reconciles the aggregate-by-value→by-ref ABI at the
//! generic call site. See docs/TRAITS_DESIGN.md.

mod common;
use common::build_and_run;

const PRELUDE: &str = "(module app)\n\
    (deftrait Eq [Self] (eq [(a Self) (b Self)] (-> bool)))\n\
    (defstruct Point [(x i64) (y i64)])\n\
    (impl Eq Point\n\
      (eq [(a Point) (b Point)] (-> bool)\n\
        (and (icmp-eq (load (field a x)) (load (field b x)))\n\
             (icmp-eq (load (field a y)) (load (field b y))))))\n\
    (defn mkpt [(x i64) (y i64)] (-> Point)\n\
      (let [(mut p) (zeroed Point)]\n\
        (store! (field (mut p) x) x) (store! (field (mut p) y) y) (load p)))\n";

#[test]
fn concrete_trait_dispatch() {
    // `eq` on a concrete Point resolves directly to the impl method.
    let code = build_and_run(&format!(
        "{PRELUDE}(defn main [] (-> i64)\n\
           (let [p1 (mkpt 3 4) p2 (mkpt 3 4) p3 (mkpt 3 9)]\n\
             (iadd (if (eq p1 p2) 10 0) (if (eq p1 p3) 100 1))))" // 10 + 1
    ));
    assert_eq!(code, 11);
}

#[test]
fn bounded_generic_deferred_dispatch() {
    // A generic bounded `(T Eq)` calls `eq` on T; mono resolves it per
    // instantiation. The aggregate flows by value through the generic — codegen
    // reconciles it to the impl's by-ref parameter.
    let code = build_and_run(&format!(
        "{PRELUDE}(defn same [(T Eq)] [(a T) (b T)] (-> bool) (eq a b))\n\
         (defn main [] (-> i64)\n\
           (let [p1 (mkpt 3 4) p2 (mkpt 3 4) p3 (mkpt 3 9)]\n\
             (iadd (if (same p1 p2) 10 0) (if (same p1 p3) 100 1))))"
    ));
    assert_eq!(code, 11);
}

#[test]
fn definition_time_bound_check_rejects_unbounded_call() {
    // Calling a trait method on an UNBOUNDED type parameter errors at the
    // definition of the generic — not at some later instantiation.
    let err = coil::emit_ir(&format!(
        "{PRELUDE}(defn bad [T] [(a T) (b T)] (-> bool) (eq a b))\n\
         (defn main [] (-> i64) 0)"
    ))
    .unwrap_err();
    assert!(err.contains("not bounded by 'Eq'"), "got:\n{err}");
}

#[test]
fn instantiation_site_requires_an_impl() {
    // Calling a bounded generic with a type that doesn't implement the trait
    // errors at the call site.
    let err = coil::emit_ir(&format!(
        "{PRELUDE}(defstruct Q [(v i64)])\n\
         (defn same [(T Eq)] [(a T) (b T)] (-> bool) (eq a b))\n\
         (defn main [] (-> i64) (let [(mut q) (zeroed Q)] (if (same (load q) (load q)) 1 0)))"
    ))
    .unwrap_err();
    assert!(err.contains("does not implement 'Eq'"), "got:\n{err}");
}

#[test]
fn impl_must_match_trait_signature() {
    let err = coil::emit_ir(
        "(module app)\n\
         (deftrait Eq [Self] (eq [(a Self) (b Self)] (-> bool)))\n\
         (defstruct P [(x i64)])\n\
         (impl Eq P (eq [(a P)] (-> bool) true))\n\
         (defn main [] (-> i64) 0)",
    )
    .unwrap_err();
    assert!(err.contains("signature doesn't match"), "got:\n{err}");
}

#[test]
fn calling_an_unimplemented_method_is_rejected() {
    // The trait declares `eq` but no impl provides it.
    let err = coil::emit_ir(
        "(module app)\n\
         (deftrait Eq [Self] (eq [(a Self) (b Self)] (-> bool)))\n\
         (defstruct P [(x i64)])\n\
         (impl Eq P)\n\
         (defn main [] (-> i64) 0)",
    )
    .unwrap_err();
    assert!(err.contains("missing method 'eq'"), "got:\n{err}");
}

#[test]
fn two_impls_dispatch_independently() {
    // Eq for two types; a bounded generic resolves each to the right impl.
    let code = build_and_run(
        "(module app)\n\
         (deftrait Eq [Self] (eq [(a Self) (b Self)] (-> bool)))\n\
         (defstruct A [(v i64)])\n\
         (defstruct B [(v i64)])\n\
         (impl Eq A (eq [(a A) (b A)] (-> bool) (icmp-eq (load (field a v)) (load (field b v)))))\n\
         (impl Eq B (eq [(a B) (b B)] (-> bool) false))\n\
         (defn same [(T Eq)] [(a T) (b T)] (-> bool) (eq a b))\n\
         (defn mk [(v i64)] (-> A) (let [(mut a) (zeroed A)] (store! (field (mut a) v) v) (load a)))\n\
         (defn mkb [(v i64)] (-> B) (let [(mut b) (zeroed B)] (store! (field (mut b) v) v) (load b)))\n\
         (defn main [] (-> i64)\n\
           (iadd (if (same (mk 5) (mk 5)) 10 0)       ; A: equal -> 10\n\
                 (if (same (mkb 1) (mkb 1)) 100 1)))  ; B: always false -> 1   => 11",
    );
    assert_eq!(code, 11);
}
