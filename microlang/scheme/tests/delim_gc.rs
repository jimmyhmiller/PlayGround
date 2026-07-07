//! PROOF OF CONCEPT: you could build a *production* language with delimited
//! continuations on this toolkit.
//!
//! "Production" is the bar the earlier Filinski library derivation (in
//! `conformance.rs`) does not clear: it is welded to undelimited `call/cc` and,
//! more importantly, delimited continuations in a real language must survive
//! garbage collection — a program allocates and collects *while continuations
//! are captured and live on the heap*.
//!
//! This test exercises the two things that make it production-viable:
//!   1. NATIVE `shift`/`reset` primitives (not a CPS library) — the CEK machine
//!      captures only the delimited slice up to the nearest prompt.
//!   2. Those captured continuations SURVIVE A MOVING COLLECTION that relocates
//!      the very heap data they close over, and still resume correctly — any
//!      number of times (multi-shot).
//!
//! The moving GC and the full-continuation execution tier compose here: exactly
//! the combination the 45-way orthogonality matrix deliberately left out.

use microlang::{CekMachine, LowBitModel, Runtime};

fn eval(rt: &mut Runtime<LowBitModel>, prog: &str) -> String {
    let v = scheme::run(rt, &CekMachine, prog);
    scheme::write_value(rt, v)
}

/// Native `shift`/`reset` compute the classic delimited-control results (the same
/// values the Filinski library derivation and Chicken produce in `conformance.rs`).
#[test]
fn native_shift_reset_values() {
    let cases = [
        // k is the delimited continuation `(+ 10 [])`; k(k(5)) = 25; +1 => 26.
        ("(+ 1 (reset (+ 10 (shift k (k (k 5))))))", "26"),
        // k = `(+ 1 [])`; k(5) = 6; *2 => 12.
        ("(* 2 (reset (+ 1 (shift k (k 5)))))", "12"),
        // shift discards k and aborts to the prompt: reset yields 5.
        ("(reset (+ 1 (shift k 5)))", "5"),
        // MULTI-SHOT: k invoked twice inside one shift body. 3 + 6 + 6 = 15.
        ("(reset (+ 1 (shift k (+ 3 (k 5) (k 5)))))", "15"),
    ];
    for (expr, want) in cases {
        let mut rt = Runtime::<LowBitModel>::new();
        let got = eval(&mut rt, expr);
        assert_eq!(got, want, "native delimited control wrong for {expr}");
    }
}

/// THE PROOF: a delimited continuation captured over heap-allocated data survives
/// a moving collection that relocates that data, then resumes — twice.
///
/// The captured continuation is `k = λx. (* 2 (+ x (first data)))`, delimited by
/// `reset`. `(first data)` lives INSIDE the captured slice, so resuming `k` after
/// the collection dereferences `data` at its NEW address. If the collector did not
/// trace the continuation's captured frame, this would be a use-after-move.
#[test]
fn delimited_continuation_survives_moving_gc() {
    let mut rt = Runtime::<LowBitModel>::new();

    let prog = "
        (define k #f)
        (define data (list 7 8 9))
        ;; Capture a delimited continuation that closes over `data`, then abort
        ;; out of the reset (shift body returns 0, so the reset yields 0).
        (define ignored
          (reset (* 2 (+ (shift c (do (set! k c) 0)) (first data)))))
        ;; Allocate garbage and force a MOVING collection: `data` (and the frames
        ;; the captured continuation holds) are relocated to fresh addresses.
        (define junk (list 1 2 3 4 5 6 7 8 9 10))
        (gc)
        ;; Resume the captured continuation TWICE, after the move. Each resumption
        ;; re-reads `data` at its relocated address.
        ;;   (k 5) = (* 2 (+ 5 7)) = 24
        ;;   (k 100) = (* 2 (+ 100 7)) = 214
        (+ (k 5) (k 100))
    ";

    let before = rt.relocated;
    let got = eval(&mut rt, prog);
    let moved = rt.relocated - before;

    // The collection actually relocated objects (not a no-op) ...
    assert!(moved > 0, "expected the collector to relocate objects; moved {moved}");
    // ... and the delimited continuation, resumed multi-shot AFTER the move,
    // computed correctly against the relocated `data`. 24 + 214 = 238.
    assert_eq!(got, "238", "captured continuation did not survive the moving GC");
}
