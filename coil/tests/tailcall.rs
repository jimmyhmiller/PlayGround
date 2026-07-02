//! Self-tail-recursion is emitted as an explicit LLVM `musttail` call, so it
//! becomes a frame-reusing jump (a loop) at ANY optimization level — Coil's only
//! loop no longer depends on `-O3`'s tail-call-elimination pass for stack safety.

mod common;
use common::build_and_run;

/// A direct self-recursive call in tail position (through an `if`) is `musttail`.
#[test]
fn self_recursion_through_if_is_musttail() {
    let src = "(defn count [(n i64) (acc i64)] (-> i64)\n\
               (if (icmp-eq n 0) acc (count (isub n 1) (iadd acc n))))\n\
               (defn main [] (-> i64) (count 10 0))";
    let ir = coil::emit_ir(src).unwrap();
    assert!(ir.contains("musttail call"), "self-recursive call should be musttail:\n{ir}");
    // The non-self call from `main` is NOT a tail-eliminable self-call.
    assert!(
        ir.lines().any(|l| l.contains("musttail call") && l.contains("@count")),
        "the recursive @count call should be the musttail one:\n{ir}"
    );
}

/// Self-recursion in tail position through a `let` body is also `musttail`.
#[test]
fn self_recursion_through_let_is_musttail() {
    let src = "(defn go [(n i64)] (-> i64)\n\
               (if (icmp-eq n 0) 0 (let [m (isub n 1)] (go m))))\n\
               (defn main [] (-> i64) (go 5))";
    assert!(coil::emit_ir(src).unwrap().contains("musttail call"), "let-tail self-call should be musttail");
}

/// Self-recursion inside a `match` arm is `musttail` too.
#[test]
fn self_recursion_through_match_is_musttail() {
    let src = "(defsum Nat (Zero) (Succ [(pred i64)]))\n\
               (defn down [(n Nat) (acc i64)] (-> i64)\n\
               (match n (Zero [] acc) (Succ [p] (down (Succ (isub p 1)) (iadd acc 1)))))\n\
               (defn main [] (-> i64) 0)";
    let ir = coil::emit_ir(src).unwrap();
    assert!(ir.contains("musttail call"), "match-arm self-call should be musttail:\n{ir}");
}

/// A NON-self tail call (calling a different function) is left a plain call —
/// `musttail` is only safe for the matching-signature self-recursion case.
#[test]
fn non_self_tail_call_is_not_musttail() {
    let src = "(defn helper [(x i64)] (-> i64) (iadd x 1))\n\
               (defn main [] (-> i64) (helper 41))";
    let ir = coil::emit_ir(src).unwrap();
    // Scope the assertion to `main`'s body: the stdlib that rides along with
    // the prelude has legitimately-musttail self-recursive printers (fmt's
    // udec-digits and friends), so a whole-module scan would false-positive.
    let main_body = ir
        .split("define i64 @main")
        .nth(1)
        .and_then(|rest| rest.split("\n}").next())
        .expect("main in IR");
    assert!(!main_body.contains("musttail"), "a non-self call must not be musttail:\n{main_body}");
}

/// Behavior is unchanged: a deep accumulator recursion still computes correctly
/// (and, via musttail, runs in constant stack).
#[test]
fn deep_self_recursion_is_correct() {
    // sum 1..=1000 = 500500; exit code is the low 8 bits = 500500 % 256 = 20.
    let src = "(defn sum-to [(n i64) (acc i64)] (-> i64)\n\
               (if (icmp-eq n 0) acc (sum-to (isub n 1) (iadd acc n))))\n\
               (defn main [] (-> i64) (sum-to 1000 0))";
    assert_eq!(build_and_run(src), 20);
}
