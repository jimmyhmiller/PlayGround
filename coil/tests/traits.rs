//! Structured, definition-checked traits: `deftrait` / `impl` / bounded generics
//! (`(T Trait)`), with static (monomorphized) dispatch — concrete calls resolve
//! to the impl, generic bodies are checked against their bounds at the definition,
//! and the type-arg's impl is verified at the call site. `Eq`/`Hash` live in the
//! auto-loaded prelude (no import). `Self` is passed by value (codegen reconciles
//! the aggregate-by-value→by-ref ABI at the generic call site). See
//! docs/TRAITS_DESIGN.md.

mod common;
use common::build_and_run;

// `Eq` is in the prelude — programs just `impl`/`derive` it. A point + an impl.
const PRELUDE: &str = "(module app)\n\
    (defstruct Point [(x i64) (y i64)])\n\
    (impl Eq Point\n\
      (= [(a Point) (b Point)] (-> bool)\n\
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
             (iadd (if (= p1 p2) 10 0) (if (= p1 p3) 100 1))))" // 10 + 1
    ));
    assert_eq!(code, 11);
}

#[test]
fn bounded_generic_deferred_dispatch() {
    // A generic bounded `(T Eq)` calls `eq` on T; mono resolves it per
    // instantiation. The aggregate flows by value through the generic — codegen
    // reconciles it to the impl's by-ref parameter.
    let code = build_and_run(&format!(
        "{PRELUDE}(defn same [(T Eq)] [(a T) (b T)] (-> bool) (= a b))\n\
         (defn main [] (-> i64)\n\
           (let [p1 (mkpt 3 4) p2 (mkpt 3 4) p3 (mkpt 3 9)]\n\
             (iadd (if (same p1 p2) 10 0) (if (same p1 p3) 100 1))))"
    ));
    assert_eq!(code, 11);
}

#[test]
fn prelude_eq_works_on_i64_with_no_imports() {
    // The prelude provides Eq + (impl Eq i64), so `eq` works bare.
    let code = build_and_run("(module app)\n(defn main [] (-> i64) (if (= 3 3) (if (= 3 4) 0 7) 0))");
    assert_eq!(code, 7);
}

#[test]
fn definition_time_bound_check_rejects_unbounded_call() {
    // Calling a trait method on an UNBOUNDED type parameter errors at the
    // definition of the generic — not at some later instantiation.
    let err = coil::emit_ir(&format!(
        "{PRELUDE}(defn bad [T] [(a T) (b T)] (-> bool) (= a b))\n\
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
         (defn same [(T Eq)] [(a T) (b T)] (-> bool) (= a b))\n\
         (defn main [] (-> i64) (let [(mut q) (zeroed Q)] (if (same (load q) (load q)) 1 0)))"
    ))
    .unwrap_err();
    assert!(err.contains("does not implement 'Eq'"), "got:\n{err}");
}

#[test]
fn impl_must_match_trait_signature() {
    // Conformance against the prelude's Eq (wrong arity).
    let err = coil::emit_ir(
        "(module app)\n\
         (defstruct P [(x i64)])\n\
         (impl Eq P (= [(a P)] (-> bool) true))\n\
         (defn main [] (-> i64) 0)",
    )
    .unwrap_err();
    assert!(err.contains("signature doesn't match"), "got:\n{err}");
}

#[test]
fn calling_an_unimplemented_method_is_rejected() {
    // The (prelude) Eq trait declares `eq` but this impl provides nothing.
    let err = coil::emit_ir(
        "(module app)\n\
         (defstruct P [(x i64)])\n\
         (impl Eq P)\n\
         (defn main [] (-> i64) 0)",
    )
    .unwrap_err();
    assert!(err.contains("missing method '='"), "got:\n{err}");
}

#[test]
fn user_defined_trait_with_deftrait() {
    // A user trait (not in the prelude): deftrait + impl + bounded-generic use.
    let code = build_and_run(
        "(module app)\n\
         (deftrait Show [Self] (mag [(x Self)] (-> i64)))\n\
         (defstruct V [(x i64) (y i64)])\n\
         (impl Show V (mag [(x V)] (-> i64) (iadd (load (field x x)) (load (field x y)))))\n\
         (defn total [(T Show)] [(a T)] (-> i64) (mag a))\n\
         (defn mkv [(x i64) (y i64)] (-> V)\n\
           (let [(mut v) (zeroed V)] (store! (field (mut v) x) x) (store! (field (mut v) y) y) (load v)))\n\
         (defn main [] (-> i64) (total (mkv 30 12)))", // 42
    );
    assert_eq!(code, 42);
}

#[test]
fn derive_generates_a_working_eq_impl() {
    // (derive Eq T) reflects the fields and expands to the impl — no hand-written
    // method, and Eq comes from the prelude (only derive.coil is imported).
    let code = build_and_run(
        "(module app)\n\
         (import \"lib/derive.coil\" :use *)\n\
         (defstruct Point [(x i64) (y i64)])\n\
         (derive Eq Point)\n\
         (defn same [(T Eq)] [(a T) (b T)] (-> bool) (= a b))\n\
         (defn mkpt [(x i64) (y i64)] (-> Point)\n\
           (let [(mut p) (zeroed Point)] (store! (field (mut p) x) x) (store! (field (mut p) y) y) (load p)))\n\
         (defn main [] (-> i64)\n\
           (iadd (if (same (mkpt 3 4) (mkpt 3 4)) 10 0) (if (same (mkpt 3 4) (mkpt 3 9)) 100 1)))", // 10 + 1
    );
    assert_eq!(code, 11);
}

#[test]
fn derive_eq_recurses_through_nested_structs() {
    // A nested struct field uses the nested type's Eq impl (also derived).
    let code = build_and_run(
        "(module app)\n\
         (import \"lib/derive.coil\" :use *)\n\
         (defstruct Point [(x i64) (y i64)])\n\
         (derive Eq Point)\n\
         (defstruct Line [(a Point) (b Point)])\n\
         (derive Eq Line)\n\
         (defn mkpt [(x i64) (y i64)] (-> Point)\n\
           (let [(mut p) (zeroed Point)] (store! (field (mut p) x) x) (store! (field (mut p) y) y) (load p)))\n\
         (defn mkl [(x i64)] (-> Line)\n\
           (let [(mut l) (zeroed Line)] (store! (field (mut l) a) (mkpt x 2)) (store! (field (mut l) b) (mkpt 3 4)) (load l)))\n\
         (defn main [] (-> i64)\n\
           (iadd (if (= (mkl 1) (mkl 1)) 10 0) (if (= (mkl 1) (mkl 9)) 100 1)))", // 10 + 1
    );
    assert_eq!(code, 11);
}

#[test]
fn derive_hash_agrees_for_equal_values() {
    let code = build_and_run(
        "(module app)\n\
         (import \"lib/derive.coil\" :use *)\n\
         (defstruct Point [(x i64) (y i64)])\n\
         (derive Hash Point)\n\
         (defn h [(T Hash)] [(v T)] (-> i64) (hash v))\n\
         (defn mkpt [(x i64) (y i64)] (-> Point)\n\
           (let [(mut p) (zeroed Point)] (store! (field (mut p) x) x) (store! (field (mut p) y) y) (load p)))\n\
         (defn main [] (-> i64)\n\
           (iadd (if (icmp-eq (h (mkpt 3 4)) (h (mkpt 3 4))) 10 0)\n\
                 (if (icmp-eq (h (mkpt 3 4)) (h (mkpt 3 9))) 100 1)))", // 10 + 1
    );
    assert_eq!(code, 11);
}

#[test]
fn derive_unknown_trait_is_rejected() {
    let err = coil::emit_ir(
        "(module app)\n\
         (import \"lib/derive.coil\" :use *)\n\
         (defstruct P [(x i64)])\n\
         (derive Ord P)\n\
         (defn main [] (-> i64) 0)",
    )
    .unwrap_err();
    assert!(err.contains("only Eq and Hash"), "got:\n{err}");
}

#[test]
fn case_defaults_to_eq_trait() {
    // `case` (lib/control.coil) compares with the prelude Eq trait: works on i64
    // keys and on a derived-Eq struct, with no trait import.
    let code = build_and_run(
        "(module app)\n\
         (import \"lib/control.coil\" :use *)\n\
         (import \"lib/derive.coil\" :use *)\n\
         (defstruct P [(v i64)])\n\
         (derive Eq P)\n\
         (defn mkp [(v i64)] (-> P) (let [(mut p) (zeroed P)] (store! (field (mut p) v) v) (load p)))\n\
         (defn classify [(n i64)] (-> i64) (case n 1 100 2 200 999))\n\
         (defn pick [(p P)] (-> i64) (case p (mkp 7) 70 (mkp 8) 80 9))\n\
         (defn main [] (-> i64) (iadd (classify 2) (pick (mkp 8))))", // 200 + 80 = 280; %256 = 24
    );
    assert_eq!(code, 24);
}

#[test]
fn case_on_non_eq_type_is_rejected() {
    // The whole point: a non-Eq key type gives a clear error.
    let err = coil::emit_ir(
        "(module app)\n\
         (import \"lib/control.coil\" :use *)\n\
         (defstruct NoEq [(v i64)])\n\
         (defn mk [(v i64)] (-> NoEq) (let [(mut p) (zeroed NoEq)] (store! (field (mut p) v) v) (load p)))\n\
         (defn f [(x NoEq)] (-> i64) (case x (mk 1) 10 (mk 2) 20 99))\n\
         (defn main [] (-> i64) (f (mk 1)))",
    )
    .unwrap_err();
    assert!(err.contains("does not implement 'Eq'"), "got:\n{err}");
}

#[test]
fn two_impls_dispatch_independently() {
    // Eq for two types; a bounded generic resolves each to the right impl.
    let code = build_and_run(
        "(module app)\n\
         (defstruct A [(v i64)])\n\
         (defstruct B [(v i64)])\n\
         (impl Eq A (= [(a A) (b A)] (-> bool) (icmp-eq (load (field a v)) (load (field b v)))))\n\
         (impl Eq B (= [(a B) (b B)] (-> bool) false))\n\
         (defn same [(T Eq)] [(a T) (b T)] (-> bool) (= a b))\n\
         (defn mk [(v i64)] (-> A) (let [(mut a) (zeroed A)] (store! (field (mut a) v) v) (load a)))\n\
         (defn mkb [(v i64)] (-> B) (let [(mut b) (zeroed B)] (store! (field (mut b) v) v) (load b)))\n\
         (defn main [] (-> i64)\n\
           (iadd (if (same (mk 5) (mk 5)) 10 0)       ; A: equal -> 10\n\
                 (if (same (mkb 1) (mkb 1)) 100 1)))  ; B: always false -> 1   => 11",
    );
    assert_eq!(code, 11);
}
