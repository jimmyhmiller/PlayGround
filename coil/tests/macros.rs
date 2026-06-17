//! User-defined macros: a compile-time Lisp (tree-walking interpreter, no JIT).
//! Programs are run via AOT so we also prove macros fully expand before codegen.

mod common;
use common::build_and_run;
use coil::expand_to_string;

#[test]
fn quasiquote_and_splicing() {
    let src = r#"
        (defmacro when [c & body] `(if ~c (do ~@body) 0))
        (defn main [] (-> :i64)
          (when (icmp-eq 1 1) 10 20 42))
    "#;
    assert_eq!(build_and_run(src), 42);
}

#[test]
fn recursive_macro_cond() {
    let src = r#"
        (defmacro cond [& cs]
          (if (empty? cs) 0
            (if (= (count cs) 1) (first cs)
              `(if ~(first cs) ~(first (rest cs)) (cond ~@(rest (rest cs)))))))
        (defn pick [(n :i64)] (-> :i64)
          (cond (icmp-lt n 0) 1
                (icmp-eq n 0) 2
                3))
        (defn main [] (-> :i64)
          (iadd (iadd (pick 0) (imul (pick 5) 10)) (imul (pick 0) 100)))
    "#;
    // pick(0)=2, pick(5)=3, pick(0)=2  ->  2 + 30 + 200 = 232
    assert_eq!(build_and_run(src), 232);
}

#[test]
fn compile_time_computation_unrolls() {
    // A helper recurses at compile time to build nested imuls.
    let src = r#"
        (def pow-form (lambda [x n]
          (if (= n 0) 1 `(imul ~x ~(pow-form x (- n 1))))))
        (defmacro pow [x n] (pow-form x n))
        (defn main [] (-> :i64) (pow 2 7))
    "#;
    assert_eq!(build_and_run(src), 128); // 2^7

    let expanded = expand_to_string(src).unwrap();
    assert!(
        expanded.contains("(imul 2 (imul 2 (imul 2"),
        "expected unrolled imuls:\n{expanded}"
    );
}

#[test]
fn gensym_avoids_capture_and_double_eval() {
    let src = r#"
        (defmacro double [x]
          (let [t (gensym "t")] `(let [~t ~x] (iadd ~t ~t))))
        (defn main [] (-> :i64) (double (iadd 20 1)))
    "#;
    assert_eq!(build_and_run(src), 42);

    let expanded = expand_to_string(src).unwrap();
    assert!(expanded.contains("t__"), "expected a gensym name:\n{expanded}");
    assert!(
        !expanded.contains("(iadd (iadd 20 1) (iadd 20 1))"),
        "argument should not be duplicated:\n{expanded}"
    );
}

#[test]
fn macro_emits_toplevel_definition() {
    let src = r#"
        (defmacro defconst [name v] `(defn ~name [] (-> :i64) ~v))
        (defconst answer 42)
        (defn main [] (-> :i64) (answer))
    "#;
    assert_eq!(build_and_run(src), 42);
}

#[test]
fn macro_can_target_conventions_and_regions() {
    let src = r#"
        (defcc fast2 :params [rax rdx] :ret rax
          :clobber [rax rdx rcx] :preserve [rbx rbp] :native fast)
        (defmacro defn-fast [name params ret & body]
          `(defn ~name :cc fast2 ~params ~ret ~@body))
        (defn-fast add [(a :i64) (b :i64)] (-> :i64) (iadd a b))
        (defn main [] (-> :i64)
          (let [p (alloc heap)]
            (store! p (add 20 22))
            (let [v (load p)] (free p) v)))
    "#;
    assert_eq!(build_and_run(src), 42);
    assert!(expand_to_string(src).unwrap().contains(":cc fast2"));
}

#[test]
fn macroless_programs_still_work() {
    let src = "(defn main [] (-> :i64) (iadd 40 2))";
    assert_eq!(build_and_run(src), 42);
}
