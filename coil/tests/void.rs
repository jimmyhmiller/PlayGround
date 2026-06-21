//! The `void` return type — a `(-> void)` function/extern yields NO value (an LLVM
//! void return); using its result is a hard error (no-silent-wrong). For void C
//! functions (qsort) and Coil procedures run for effect.

mod common;
use common::build_and_run;

#[test]
fn void_coil_procedure_runs_for_effect() {
    let src = r#"
        (defn bump [(p (ptr i64))] (-> void) (store! p (iadd (load p) 1)))
        (defn main [] (-> i64)
          (let [a (alloc-stack i64)]
            (store! a 41)
            (bump a)          ; void call in statement position
            (bump a)
            (isub (load a) 1)))"#; // 41 + 2 - 1 = 42
    assert_eq!(build_and_run(src), 42);
}

#[test]
fn void_extern_qsort_with_callback() {
    // qsort returns void; declared (-> void) and called for effect.
    let src = r#"
        (extern qsort :cc c [(ptr i8) i64 i64 (fnptr c [(ptr i8) (ptr i8)] i32)] (-> void))
        (defn cmp [(a (ptr i8)) (b (ptr i8))] (-> i32)
          (let [x (load (cast (ptr i64) a)) y (load (cast (ptr i64) b))]
            (if (icmp-lt x y) (cast i32 -1) (if (icmp-gt x y) (cast i32 1) (cast i32 0)))))
        (defn main [] (-> i64)
          (let [arr (alloc-stack (array i64 3))]
            (store! (index arr 0) 50) (store! (index arr 1) 42) (store! (index arr 2) 99)
            (qsort (cast (ptr i8) (index arr 0)) 3 8 (fnptr-of cmp))
            (load (index arr 1))))"#; // sorted [42 50 99] -> arr[1] = 50? no: [42,50,99], arr[1]=50
    assert_eq!(build_and_run(src), 50);
}

#[test]
fn using_a_void_result_is_a_hard_error() {
    let cases = [
        // arithmetic operand
        "(defn p [(x (ptr i64))] (-> void) (store! x 1))\n\
         (defn main [] (-> i64) (let [a (alloc-stack i64)] (iadd (p a) 1)))",
        // let binding
        "(defn p [(x (ptr i64))] (-> void) (store! x 1))\n\
         (defn main [] (-> i64) (let [a (alloc-stack i64) x (p a)] (load a)))",
        // returned where a value is expected
        "(defn p [(x (ptr i64))] (-> void) (store! x 1))\n\
         (defn main [] (-> i64) (let [a (alloc-stack i64)] (p a)))",
    ];
    for src in cases {
        assert!(coil::check_source(src).is_err(), "void misuse must error: {src}");
    }
}

#[test]
fn void_is_rejected_outside_return_position() {
    // parameter
    assert!(coil::check_source(
        "(defn f [(x void)] (-> i64) 0)\n(defn main [] (-> i64) 0)"
    )
    .unwrap_err()
    .contains("only valid as a return type"));
    // struct field
    assert!(coil::check_source(
        "(defstruct S [(v void)])\n(defn main [] (-> i64) 0)"
    )
    .unwrap_err()
    .contains("only valid as a return type"));
}
