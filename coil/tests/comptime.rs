//! `(comptime E)` — run real Coil code at compile time and splice the literal.
//! Stage 1: the pure scalar subset (arithmetic, comparison, if/let/do/match, sum
//! construction, calls to defns incl. recursion, and the `=` trait). The bridge
//! toward "the whole language at compile time".

mod common;
use common::build_and_run;

#[test]
fn comptime_arithmetic_folds() {
    let code = build_and_run("(module app)\n(defn main [] (-> i64) (comptime (iadd (imul 3 4) 2)))");
    assert_eq!(code, 14);
}

#[test]
fn comptime_calls_a_defn_recursively() {
    // fact is an ordinary runtime function — here it runs at COMPILE time.
    let code = build_and_run(
        "(module app)\n\
         (defn fact [(n i64)] (-> i64) (if (icmp-le n 0) 1 (imul n (fact (isub n 1)))))\n\
         (defn main [] (-> i64) (comptime (fact 5)))", // 120
    );
    assert_eq!(code, 120);
}

#[test]
fn comptime_uses_the_eq_trait() {
    // The SAME `=` (the Eq trait) runs at compile time.
    let code = build_and_run(
        "(module app)\n(defn main [] (-> i64) (comptime (if (= 7 7) (if (= 7 8) 0 42) 0)))",
    );
    assert_eq!(code, 42);
}

#[test]
fn comptime_let_and_match() {
    // let + inline match over a sum value (Stage 1: sums are values; passing one
    // across a function call goes by-reference, which is Stage 1b).
    let code = build_and_run(
        "(module app)\n\
         (defsum Opt (Some [(v i64)]) (Non []))\n\
         (defn main [] (-> i64)\n\
           (comptime (let [a 10 b 32] (match (Some (iadd a b)) (Some [v] v) (Non [] 0)))))", // 42
    );
    assert_eq!(code, 42);
}

#[test]
fn comptime_folds_to_a_constant() {
    // The result is baked in: main is a constant return, no call to fact.
    let ir = coil::emit_ir(
        "(module app)\n\
         (defn fact [(n i64)] (-> i64) (if (icmp-le n 0) 1 (imul n (fact (isub n 1)))))\n\
         (defn main [] (-> i64) (comptime (fact 5)))",
    )
    .unwrap();
    // find main's body
    let main = ir.split("@main").nth(1).unwrap_or("");
    let body = &main[..main.find('}').unwrap_or(main.len())];
    assert!(body.contains("ret i64 120"), "main not folded to a constant:\n{body}");
    assert!(!body.contains("call"), "main still calls at runtime:\n{body}");
}

#[test]
fn comptime_scalar_mutation_works() {
    // Stage 1b: mutable scalar locals + load/store run at compile time.
    let code = build_and_run(
        "(module app)\n(defn main [] (-> i64) (comptime (let [(mut x) 0] (store! x 5) (iadd (load x) 37))))",
    );
    assert_eq!(code, 42);
}

#[test]
fn comptime_rejects_unsupported_clearly() {
    // Something still unsupported (a generic call) errors clearly, not silently.
    let err = coil::emit_ir(
        "(module app)\n\
         (defn id [T] [(x T)] (-> T) x)\n\
         (defn main [] (-> i64) (comptime (id [i64] 5)))",
    )
    .unwrap_err();
    assert!(err.contains("comptime:"), "expected a comptime error, got:\n{err}");
}


// ---- Stage 1b: memory model (mutable locals, loops, aggregates, by-ref args) ---

#[test]
fn comptime_mutable_local_and_loop() {
    // A function using a mutable accumulator + loop, run at compile time.
    let code = build_and_run(
        "(module app)\n\
         (defn sum-to [(n i64)] (-> i64)\n\
           (let [(mut acc) 0 (mut i) 1]\n\
             (loop (if (icmp-gt (load i) n) (break)\n\
                     (do (store! acc (iadd (load acc) (load i)))\n\
                         (store! i (iadd (load i) 1)))))\n\
             (load acc)))\n\
         (defn main [] (-> i64) (comptime (sum-to 10)))", // 55
    );
    assert_eq!(code, 55);
}

#[test]
fn comptime_break_with_value() {
    let code = build_and_run(
        "(module app)\n\
         (defn first-sq-over [(lim i64)] (-> i64)\n\
           (let [(mut i) 0]\n\
             (loop (if (icmp-gt (imul (load i) (load i)) lim) (break (load i)) 0)\n\
                   (store! i (iadd (load i) 1)))))\n\
         (defn main [] (-> i64) (comptime (first-sq-over 50)))", // 8 (8*8=64>50)
    );
    assert_eq!(code, 8);
}

#[test]
fn comptime_by_reference_sum_argument() {
    // Passing a sum across a function call (it goes by reference) now works.
    let code = build_and_run(
        "(module app)\n\
         (defsum Opt (Some [(v i64)]) (Non []))\n\
         (defn unwrap-or [(o Opt) (d i64)] (-> i64) (match o (Some [v] v) (Non [] d)))\n\
         (defn main [] (-> i64) (comptime (iadd (unwrap-or (Some 30) 0) (unwrap-or (Non) 12))))", // 42
    );
    assert_eq!(code, 42);
}

#[test]
fn comptime_struct_field_mutation() {
    let code = build_and_run(
        "(module app)\n\
         (defstruct P [(x i64) (y i64)])\n\
         (defn main [] (-> i64)\n\
           (comptime (let [(mut p) (zeroed P)]\n\
                       (store! (field (mut p) x) 5)\n\
                       (store! (field (mut p) y) 37)\n\
                       (iadd (load (field p x)) (load (field p y))))))", // 42
    );
    assert_eq!(code, 42);
}

// ---- Stage 1c: aggregate results (a comptime form returning a struct/sum) ------

#[test]
fn comptime_returns_a_struct() {
    let code = build_and_run(
        "(module app)\n\
         (defstruct P [(x i64) (y i64)])\n\
         (defn mkp [(a i64) (b i64)] (-> P)\n\
           (let [(mut p) (zeroed P)] (store! (field (mut p) x) a) (store! (field (mut p) y) b) (load p)))\n\
         (defn main [] (-> i64)\n\
           (let [p (comptime (mkp 30 12))] (iadd (load (field p x)) (load (field p y)))))", // 42
    );
    assert_eq!(code, 42);
}

#[test]
fn comptime_returns_a_nested_struct() {
    let code = build_and_run(
        "(module app)\n\
         (defstruct P [(x i64) (y i64)])\n\
         (defstruct Line [(a P) (b P)])\n\
         (defn mkp [(a i64) (b i64)] (-> P) (let [(mut p) (zeroed P)] (store! (field (mut p) x) a) (store! (field (mut p) y) b) (load p)))\n\
         (defn mkline [(x1 i64) (y1 i64) (x2 i64) (y2 i64)] (-> Line)\n\
           (let [(mut l) (zeroed Line)] (store! (field (mut l) a) (mkp x1 y1)) (store! (field (mut l) b) (mkp x2 y2)) (load l)))\n\
         (defn main [] (-> i64)\n\
           (let [l (comptime (mkline 1 2 30 9))]\n\
             (iadd (load (field (field l a) y)) (load (field (field l b) x)))))", // 2 + 30 = 32
    );
    assert_eq!(code, 32);
}

#[test]
fn comptime_returns_a_sum() {
    let code = build_and_run(
        "(module app)\n\
         (defsum Opt (Some [(v i64)]) (Non []))\n\
         (defn main [] (-> i64)\n\
           (let [o (comptime (Some 42))] (match o (Some [v] v) (Non [] 0))))", // 42
    );
    assert_eq!(code, 42);
}

#[test]
fn comptime_array_result_errors_clearly() {
    // Arrays-as-results aren't supported yet (no element type recorded). Bind it
    // (a valid use) so we reach the value-builder rather than a type error.
    let err = coil::emit_ir(
        "(module app)\n\
         (defn mk [] (-> (array i64 2)) (let [(mut a) (zeroed (array i64 2))] (store! (index (mut a) 0) 7) (load a)))\n\
         (defn main [] (-> i64) (let [a (comptime (mk))] 0))",
    )
    .unwrap_err();
    assert!(err.contains("returning an array"), "got:\n{err}");
}

#[test]
fn comptime_division_by_zero_is_caught() {
    let err = coil::emit_ir("(module app)\n(defn main [] (-> i64) (comptime (idiv 1 0)))").unwrap_err();
    assert!(err.contains("division by zero"), "got:\n{err}");
}
