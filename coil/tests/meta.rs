//! Stage 3, steps 3-4: staged macros. A generator is an ordinary Coil function
//! `(-> Code)`; `(meta (gen …))` runs it at compile time and splices the generated
//! top-level forms — and the rest of the program can DEPEND on what's generated
//! (the elaboration loop checks the generators, runs them, then checks everything).

mod common;
use common::build_and_run;

#[test]
fn generate_and_call_a_function() {
    let code = build_and_run(
        "(module a)\n\
         (defn gen [] (-> Code) `(defn answer [] (-> i64) 42))\n\
         (meta (gen))\n\
         (defn main [] (-> i64) (answer))",
    );
    assert_eq!(code, 42);
}

#[test]
fn generate_an_impl_and_use_it() {
    // The headline: generate (impl Eq P …), then use it through a bounded generic.
    let code = build_and_run(
        "(module a)\n\
         (defstruct P [(v i64)])\n\
         (defn gen-eq [] (-> Code)\n\
           `(impl Eq P (= [(a P) (b P)] (-> bool) (icmp-eq (load (field a v)) (load (field b v))))))\n\
         (meta (gen-eq))\n\
         (defn mkp [(v i64)] (-> P) (let [(mut p) (zeroed P)] (store! (field (mut p) v) v) (load p)))\n\
         (defn same [(T Eq)] [(a T) (b T)] (-> bool) (= a b))\n\
         (defn main [] (-> i64)\n\
           (iadd (if (same (mkp 5) (mkp 5)) 10 0) (if (same (mkp 5) (mkp 9)) 100 1)))", // 11
    );
    assert_eq!(code, 11);
}

#[test]
fn reflection_inside_a_generator() {
    // Typed reflection used in code generation: splice a field count.
    let code = build_and_run(
        "(module a)\n\
         (defstruct P [(a i64) (b i64) (c i64)])\n\
         (defn gen [] (-> Code) `(defn nf [] (-> i64) ~(field-count P)))\n\
         (meta (gen))\n\
         (defn main [] (-> i64) (nf))", // P has 3 fields
    );
    assert_eq!(code, 3);
}

#[test]
fn generator_can_emit_multiple_defs() {
    // A (do …) result splices several top-level forms.
    let code = build_and_run(
        "(module a)\n\
         (defn gen [] (-> Code) `(do (defn one [] (-> i64) 1) (defn two [] (-> i64) 2)))\n\
         (meta (gen))\n\
         (defn main [] (-> i64) (iadd (one) (two)))", // 3
    );
    assert_eq!(code, 3);
}
