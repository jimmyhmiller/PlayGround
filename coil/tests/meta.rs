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
           (if (icmp-eq (urem (code-int n) 2) 0) `(imul ~n 2) `(iadd ~n 1)))\n\
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

#[test]
fn variadic_macro_folds_over_args() {
    // `&` marks a variadic macro: the last param is the Code list of all args.
    // A comptime helper recurses by index (the cond/and/|> pattern).
    let code = build_and_run(
        "(module a)\n\
         (defn vand-from [(cs Code) (i i64)] (-> Code)\n\
           (if (icmp-eq i (code-count cs)) `1\n\
               `(if ~(code-nth cs i) ~(vand-from cs (iadd i 1)) 0)))\n\
         (defn vand [& (cs Code)] (-> Code) (vand-from cs 0))\n\
         (defn main [] (-> i64)\n\
           (iadd (vand (icmp-lt 1 2) (icmp-lt 2 3) (icmp-lt 3 4))\n\
                 (imul 10 (vand (icmp-lt 1 2) (icmp-lt 9 3)))))", // 1 + 0
    );
    assert_eq!(code, 1);
}

#[test]
fn unquote_splicing_multi_body_when() {
    // ~@ splices a Code list's elements into the surrounding list — multi-body when.
    let code = build_and_run(
        "(module a)\n\
         (defn when2 [(c Code) & (body Code)] (-> Code) `(if ~c (do ~@body) 0))\n\
         (defn main [] (-> i64)\n\
           (let [(mut x) 0]\n\
             (when2 (icmp-lt 1 2) (store! (mut x) 5) (store! (mut x) (iadd (load x) 3)))\n\
             (load x)))", // 5 then +3 = 8
    );
    assert_eq!(code, 8);
}

#[test]
fn cond_macro_with_else() {
    // The real cond: variadic (test value) pairs + optional trailing else, via a
    // comptime helper recursing by index in steps of two.
    let code = build_and_run(
        "(module a)\n\
         (defn cond-from [(cs Code) (i i64)] (-> Code)\n\
           (if (icmp-eq (isub (code-count cs) i) 0) `0\n\
             (if (icmp-eq (isub (code-count cs) i) 1) (code-nth cs i)\n\
               `(if ~(code-nth cs i) ~(code-nth cs (iadd i 1)) ~(cond-from cs (iadd i 2))))))\n\
         (defn cond2 [& (cs Code)] (-> Code) (cond-from cs 0))\n\
         (defn classify [(n i64)] (-> i64) (cond2 (icmp-lt n 0) 1 (icmp-eq n 0) 2 3))\n\
         (defn main [] (-> i64) (iadd (classify -5) (iadd (imul 10 (classify 0)) (imul 20 (classify 9)))))", // 1+20+60
    );
    assert_eq!(code, 81);
}

#[test]
fn runaway_macro_recursion_errors_cleanly() {
    // A self-recursive comptime function must give a clean "too deep" error, not
    // overflow the compiler's stack (SIGABRT).
    let err = coil::emit_ir(
        "(module a)\n(defn spin [(c Code)] (-> Code) (spin c))\n(defn main [] (-> i64) (spin 5))",
    )
    .unwrap_err();
    assert!(err.contains("too deep") || err.contains("runaway"), "got:\n{err}");
}
