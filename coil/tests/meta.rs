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

// ---- expression-position macros (a [Code…] -> Code function, expanded inline) ---

#[test]
fn expression_macro_when() {
    let code = build_and_run(
        "(module a)\n\
         (defn when2 [(c Code) (body Code)] (-> Code) `(if ~c ~body 0))\n\
         (defn main [] (-> i64) (when2 (icmp-lt 1 2) 7))", // (if (icmp-lt 1 2) 7 0)
    );
    assert_eq!(code, 7);
}

#[test]
fn macros_compose_and_reflect() {
    let code = build_and_run(
        "(module a)\n\
         (defstruct P [(a i64) (b i64) (c i64)])\n\
         (defn nf [(t Code)] (-> Code) `~(field-count P))\n\
         (defn dbl [(x Code)] (-> Code) `(iadd ~x ~x))\n\
         (defn main [] (-> i64) (dbl (nf P)))", // dbl(3) = (iadd 3 3) = 6
    );
    assert_eq!(code, 6);
}

#[test]
fn expression_macro_with_arithmetic_in_the_macro() {
    // The macro runs real Coil at expand time (here, picks a branch by a code int).
    let code = build_and_run(
        "(module a)\n\
         (defn twice-if-even [(n Code)] (-> Code)\n\
           (if (= (urem (code-int n) 2) 0) `(imul ~n 2) `(iadd ~n 1)))\n\
         (defn main [] (-> i64) (iadd (twice-if-even 10) (twice-if-even 7)))", // 20 + 8 = 28
    );
    assert_eq!(code, 28);
}

#[test]
fn gensym_is_hygienic() {
    // Each expansion gets a fresh name (no collision between the two calls)…
    let code = build_and_run(
        "(module a)\n\
         (defn dbl-safe [(x Code)] (-> Code) (let [t (gensym)] `(let [~t ~x] (iadd ~t ~t))))\n\
         (defn main [] (-> i64) (iadd (dbl-safe 5) (dbl-safe 3)))", // 10 + 6
    );
    assert_eq!(code, 16);
}

#[test]
fn gensym_avoids_capturing_user_bindings() {
    // …and the macro's temp does not capture a user binding of the same idea.
    let code = build_and_run(
        "(module a)\n\
         (defn add-tmp [(x Code)] (-> Code) (let [t (gensym)] `(let [~t ~x] (iadd ~t 1))))\n\
         (defn main [] (-> i64) (let [t 100] (iadd t (add-tmp 5))))", // 100 + 6
    );
    assert_eq!(code, 106);
}
