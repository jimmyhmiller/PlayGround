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
fn void_rejected_in_type_discarding_synth_paths() {
    // Red-team regressions: the forbid-use check must cover EVERY use position,
    // including the paths that synthesize an operand and discard its type (so they
    // bypass the coerce gate). These both silently leaked a void placeholder before.
    // (1) variadic argument slot — was: compiles + printf(..., i64 0) garbage.
    assert!(coil::check_source(
        "(extern printf :cc c [(ptr i8) ...] (-> i32))\n\
         (defn p [(x (ptr i64))] (-> void) (store! x 1))\n\
         (defn main [] (-> i64) (let [a (alloc-stack i64)] (printf c\"%d\" (p a)) 0))"
    )
    .unwrap_err()
    .contains("variadic argument uses a void value"));
    // (2) llvm-ir operand — was: the placeholder zero substitutes for $0.
    assert!(coil::check_source(
        "(defn p [(x (ptr i64))] (-> void) (store! x 1))\n\
         (defn main [] (-> i64) (let [a (alloc-stack i64)] (llvm-ir i64 [(p a)] \"ret i64 0\")))"
    )
    .unwrap_err()
    .contains("llvm-ir operand uses a void value"));
}

#[test]
fn void_rejected_as_a_type_argument() {
    // The THIRD hole (found by the same synth-and-discard hunt): a void value passed
    // to a generic function/constructor unified its type parameter to void → coerce
    // (void,void) accepted → mono instantiated a void value/param → codegen panic.
    // Fixed at the unification boundary (void can never be a type argument).
    // generic function:
    assert!(coil::check_source(
        "(defn id [T] [(x T)] (-> T) x)\n\
         (defn p [(x (ptr i64))] (-> void) (store! x 1))\n\
         (defn main [] (-> i64) (let [a (alloc-stack i64)] (id (p a)) 0))"
    )
    .unwrap_err()
    .contains("cannot be used as a type argument"));
    // generic constructor (a local generic sum variant — hermetic, no import):
    assert!(coil::check_source(
        "(defsum Opt [T] (Som [(v T)]) (Non []))\n\
         (defn p [(x (ptr i64))] (-> void) (store! x 1))\n\
         (defn main [] (-> i64) (let [a (alloc-stack i64)] (Som (p a)) 0))"
    )
    .unwrap_err()
    .contains("cannot be used as a type argument"));
}

#[test]
fn void_rejected_in_every_value_position() {
    // A full red-team sweep: a void value must be rejected EVERYWHERE a value is
    // read — none of these may compile (and none may panic the compiler).
    let p = "(defn p [(x (ptr i64))] (-> void) (store! x 1))\n";
    let positions = [
        // (description, main body)
        "(defn main [] (-> i64) (let [a (alloc-stack i64)] (store! a (p a)) 0))", // store! value
        "(defn main [] (-> i64) (let [a (alloc-stack i64)] (if (p a) 1 2)))",      // if condition
        "(defn main [] (-> i64) (let [a (alloc-stack i64)] (iadd 1 (if true (p a) 0))))", // if branch
        "(defn main [] (-> i64) (let [a (alloc-stack i64)] (cast i32 (p a))))",    // cast operand
        "(defstruct S [(x i64)])\n\
         (defn main [] (-> i64) (let [a (alloc-stack i64)] (load (field (p a) x))))", // field target
        "(defn main [] (-> i64) (let [a (alloc-stack i64) arr (alloc-stack (array i64 4))] (load (index arr (p a)))))", // index
        "(defsum E (A []) (B []))\n\
         (defn main [] (-> i64) (let [a (alloc-stack i64)] (match (p a) (A [] 0) (B [] 1))))", // match scrutinee
    ];
    for body in positions {
        let src = format!("{p}{body}");
        assert!(coil::check_source(&src).is_err(), "void must be rejected: {body}");
    }
}

#[test]
fn void_return_with_by_value_struct_param_does_not_panic() {
    // A (-> void) fn that also takes a by-value struct → needs_c_abi → c_signature,
    // which used to hit basic_ty(Void) → unreachable!(). Must compile cleanly.
    let src = "(defstruct Big [(a i64) (b i64) (c i64)])\n\
               (extern sink :cc c [Big] (-> void))\n\
               (defn main [] (-> i64) 0)";
    assert!(coil::check_source(src).is_ok());
    // and it must reach codegen without panicking (the unreachable! was in codegen).
    assert!(coil::emit_ir(src).is_ok());
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
