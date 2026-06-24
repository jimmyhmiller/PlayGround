//! Structured, definition-checked traits: `deftrait` / `impl` / bounded generics
//! (`(T Trait)`), with static (monomorphized) dispatch — concrete calls resolve
//! to the impl, generic bodies are checked against their bounds at the definition,
//! and the type-arg's impl is verified at the call site. v1 passes `Self` by
//! pointer (uniform ABI); see docs/TRAITS_DESIGN.md.

mod common;
use common::build_and_run;

const PRELUDE: &str = "(module app)\n\
    (import \"lib/alloc.coil\" :use *)\n\
    (import \"lib/result.coil\" :use *)\n\
    (deftrait Eq [Self] (eq [(a (ptr Self)) (b (ptr Self))] (-> bool)))\n\
    (defstruct Point [(x i64) (y i64)])\n\
    (impl Eq Point\n\
      (eq [(a (ptr Point)) (b (ptr Point))] (-> bool)\n\
        (and (icmp-eq (load (field a x)) (load (field b x)))\n\
             (icmp-eq (load (field a y)) (load (field b y))))))\n\
    (defn mkpt [(al (ptr Allocator)) (x i64) (y i64)] (-> (ptr Point))\n\
      (let [p (unwrap-ptr [Point] (create [Point] al))]\n\
        (store! (field p x) x) (store! (field p y) y) p))\n";

#[test]
fn concrete_trait_dispatch() {
    // `eq` on a concrete Point resolves directly to the impl method.
    let code = build_and_run(&format!(
        "{PRELUDE}(defn main [] (-> i64)\n\
           (let [al (malloc-allocator) p1 (mkpt al 3 4) p2 (mkpt al 3 4) p3 (mkpt al 3 9)]\n\
             (iadd (if (eq p1 p2) 10 0) (if (eq p1 p3) 100 1))))" // 10 + 1
    ));
    assert_eq!(code, 11);
}

#[test]
fn bounded_generic_deferred_dispatch() {
    // A generic bounded `(T Eq)` calls `eq` on T; mono resolves it per instantiation.
    let code = build_and_run(&format!(
        "{PRELUDE}(defn same [(T Eq)] [(a (ptr T)) (b (ptr T))] (-> bool) (eq a b))\n\
         (defn main [] (-> i64)\n\
           (let [al (malloc-allocator) p1 (mkpt al 3 4) p2 (mkpt al 3 4) p3 (mkpt al 3 9)]\n\
             (iadd (if (same p1 p2) 10 0) (if (same p1 p3) 100 1))))"
    ));
    assert_eq!(code, 11);
}

#[test]
fn definition_time_bound_check_rejects_unbounded_call() {
    // Calling a trait method on an UNBOUNDED type parameter errors at the
    // definition of the generic — not at some later instantiation.
    let err = coil::emit_ir(&format!(
        "{PRELUDE}(defn bad [T] [(a (ptr T)) (b (ptr T))] (-> bool) (eq a b))\n\
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
         (defn same [(T Eq)] [(a (ptr T)) (b (ptr T))] (-> bool) (eq a b))\n\
         (defn main [] (-> i64) (let [q (alloc-stack Q)] (if (same q q) 1 0)))"
    ))
    .unwrap_err();
    assert!(err.contains("does not implement 'Eq'"), "got:\n{err}");
}

#[test]
fn impl_must_match_trait_signature() {
    let err = coil::emit_ir(
        "(module app)\n\
         (deftrait Eq [Self] (eq [(a (ptr Self)) (b (ptr Self))] (-> bool)))\n\
         (defstruct P [(x i64)])\n\
         (impl Eq P (eq [(a (ptr P))] (-> bool) true))\n\
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
         (deftrait Eq [Self] (eq [(a (ptr Self)) (b (ptr Self))] (-> bool)))\n\
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
         (import \"lib/alloc.coil\" :use *)\n\
         (import \"lib/result.coil\" :use *)\n\
         (deftrait Eq [Self] (eq [(a (ptr Self)) (b (ptr Self))] (-> bool)))\n\
         (defstruct A [(v i64)])\n\
         (defstruct B [(v i64)])\n\
         (impl Eq A (eq [(a (ptr A)) (b (ptr A))] (-> bool) (icmp-eq (load (field a v)) (load (field b v)))))\n\
         (impl Eq B (eq [(a (ptr B)) (b (ptr B))] (-> bool) false))\n\
         (defn same [(T Eq)] [(a (ptr T)) (b (ptr T))] (-> bool) (eq a b))\n\
         (defn main [] (-> i64)\n\
           (let [al (malloc-allocator)\n\
                 a1 (unwrap-ptr [A] (create [A] al)) a2 (unwrap-ptr [A] (create [A] al))\n\
                 b1 (unwrap-ptr [B] (create [B] al)) b2 (unwrap-ptr [B] (create [B] al))]\n\
             (store! (field a1 v) 5) (store! (field a2 v) 5)\n\
             (iadd (if (same a1 a2) 10 0)       ; A: equal -> 10\n\
                   (if (same b1 b2) 100 1))))   ; B: always false -> 1   => 11",
    );
    assert_eq!(code, 11);
}
