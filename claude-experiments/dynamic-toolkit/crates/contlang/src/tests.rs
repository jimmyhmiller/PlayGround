use dynir::verify;
use dynvalue::NanBox;

use crate::{lex, lower_program, parse};

fn run_interp(src: &str, entry_name: &str, args: &[u64]) -> u64 {
    let tokens = lex(src);
    let program = parse(tokens);
    let lowered = lower_program(&program);

    for func in &lowered.module.functions {
        verify(func).unwrap_or_else(|errs| {
            eprintln!("IR for {}:\n{}", func.name, func);
            panic!("verification failed for {}: {:?}", func.name, errs);
        });
    }

    let entry = lowered.func_refs[entry_name];
    let roots = dynir::interp::NoGcRoots;
    let interp = dynir::interp::ModuleInterpreter::<NanBox, _>::new(
        &lowered.module,
        &roots,
    );
    match interp.run(entry, args) {
        Ok(dynir::interp::InterpResult::Value(v)) => v,
        Ok(dynir::interp::InterpResult::Void) => 0,
        Ok(other) => panic!("unexpected result: {:?}", other),
        Err(e) => panic!("interpreter error: {:?}", e),
    }
}

// ── Basic Language Tests ────────────────────────────────────

#[test]
fn return_constant() {
    assert_eq!(run_interp("fn main() { 42 }", "main", &[]), 42);
}

#[test]
fn arithmetic() {
    assert_eq!(run_interp("fn main() { (3 + 4) * 2 - 1 }", "main", &[]), 13);
}

#[test]
fn function_calls() {
    let src = "fn double(x) { x + x }  fn main() { double(21) }";
    assert_eq!(run_interp(src, "main", &[]), 42);
}

#[test]
fn if_expression() {
    let src = r#"
        fn abs(x) { if x >= 0 { x } else { 0 - x } }
        fn main() { abs(-5) + abs(3) }
    "#;
    assert_eq!(run_interp(src, "main", &[]), 8);
}

#[test]
fn recursion() {
    let src = r#"
        fn factorial(n) { if n <= 1 { 1 } else { n * factorial(n - 1) } }
        fn main() { factorial(6) }
    "#;
    assert_eq!(run_interp(src, "main", &[]), 720);
}

// ── Continuation Tests ──────────────────────────────────────
//
// The IR's abort_to_prompt pops frames up to and including the prompt
// owner, treating it as if the prompt owner "returned" the abort value
// to its caller. This means:
//
// - For `reset { body }` to work as a delimited control operator, the
//   body (or the part that may abort) must be in a SEPARATE function.
//   The prompt owner is that separate function, and aborting causes it
//   to "return" the abort value to the caller (which sees it as the
//   result of the reset expression).
//
// - Tests below use this pattern: a helper function owns the prompt,
//   and the caller uses the result.

#[test]
fn reset_without_abort() {
    // When no abort happens, the body function returns normally.
    let src = r#"
        fn body() { reset { 42 } }
        fn main() { body() }
    "#;
    assert_eq!(run_interp(src, "main", &[]), 42);
}

#[test]
fn abort_returns_value_to_caller() {
    // abort(v) terminates the prompt owner and returns v to the caller.
    let src = r#"
        fn body() {
            reset {
                abort(99);
                999
            }
        }
        fn main() { body() }
    "#;
    assert_eq!(run_interp(src, "main", &[]), 99);
}

#[test]
fn abort_conditional_early_exit() {
    // Conditional abort — the abort path returns early, else path completes.
    let src = r#"
        fn checked_double(x) {
            reset {
                if x < 0 { abort(0 - 1) } else { x * 2 }
            }
        }
        fn main() { checked_double(5) }
    "#;
    assert_eq!(run_interp(src, "main", &[]), 10);
}

#[test]
fn abort_conditional_early_exit_negative() {
    let src = r#"
        fn checked_double(x) {
            reset {
                if x < 0 { abort(0 - 1) } else { x * 2 }
            }
        }
        fn main() { checked_double(0 - 5) }
    "#;
    assert_eq!(run_interp(src, "main", &[]), u64::MAX); // -1 as u64
}

#[test]
fn capture_and_abort_returns_handle() {
    // Capture a continuation and abort it back to the caller.
    let src = r#"
        fn capture_it() -> cont {
            reset {
                let k: cont = shift();
                abort(k)
            }
        }
        fn main() {
            let k: cont = capture_it();
            // k is a FrameSlice handle — just verify it's non-negative
            1
        }
    "#;
    assert_eq!(run_interp(src, "main", &[]), 1);
}

#[test]
fn capture_abort_resume() {
    // Full cycle: capture → abort handle to caller → resume with value.
    let src = r#"
        fn capture_it() -> cont {
            reset {
                let k: cont = shift();
                abort(k)
            }
        }
        fn do_resume(k: cont, v) {
            resume(k, v)
        }
        fn main() {
            let k: cont = capture_it();
            do_resume(k, 42)
        }
    "#;
    // capture_it: captures, aborts k → caller gets k
    // do_resume(k, 42): resumes at shift() with 42
    //   → k = 42, abort(42) → capture_it "returns" 42
    //   → but this is the resumed continuation, so the result bubbles up
    assert_eq!(run_interp(src, "main", &[]), 42);
}


#[test]
fn multi_shot_with_clone() {
    let src = r#"
        fn capture_it() -> cont {
            reset {
                let k: cont = shift();
                abort(k)
            }
        }
        fn do_clone(k: cont) -> cont { clone(k) }
        fn do_resume(k: cont, v) { resume(k, v) }
        fn main() {
            let k: cont = capture_it();
            let k1: cont = do_clone(k);
            let k2: cont = do_clone(k);
            do_resume(k1, 11)
        }
    "#;
    assert_eq!(run_interp(src, "main", &[]), 11);
}

#[test]
fn nested_resets_via_separate_functions() {
    let src = r#"
        fn inner() { reset { abort(5) } }
        fn main() {
            reset {
                let v = inner();
                v + 10
            }
        }
    "#;
    // inner: abort(5) → inner's reset returns 5 to caller
    // main's reset body: v = 5, v + 10 = 15
    assert_eq!(run_interp(src, "main", &[]), 15);
}

#[test]
fn cross_function_abort() {
    // Abort from a callee goes to the callee's own foreign prompt,
    // which matches the caller's pushed prompt.
    let src = r#"
        fn do_abort(v) { abort(v) }
        fn main() {
            reset {
                do_abort(77);
                999
            }
        }
    "#;
    // do_abort(77) aborts — but the prompt is in main (push_prompt inline).
    // abort pops do_abort, then pops main (has prompt), returns 77.
    // Since main is popped, 999 is never reached.
    assert_eq!(run_interp(src, "main", &[]), 77);
}

#[test]
fn cross_function_capture_and_resume() {
    // capture_it owns the prompt (via reset). It captures the continuation,
    // aborts the handle back to the caller. The caller resumes with a value.
    // The resumed continuation re-enters capture_it at shift(), gets the
    // resume arg, aborts it back → capture_it "returns" the resume arg.
    let src = r#"
        fn capture_it() -> cont {
            reset {
                let k: cont = shift();
                abort(k)
            }
        }
        fn do_resume(k: cont, v) { resume(k, v) }
        fn main() {
            let k: cont = capture_it();
            do_resume(k, 42)
        }
    "#;
    assert_eq!(run_interp(src, "main", &[]), 42);
}

// ═══════════════════════════════════════════════════════════════════
// "Is it really delimited continuations?" — adversarial tests
// ═══════════════════════════════════════════════════════════════════
//
// Real delimited continuations (shift/reset, prompt/control) must:
//   1. Capture the delimited *context* between shift and reset, so that
//      resuming re-runs the intervening computation with the fresh value.
//   2. Make resume(k, v) *return* the prompt's final value to its caller,
//      so subsequent computation can use the result.
//   3. Be re-entrant: the same captured k can be resumed multiple times
//      with observably independent effects.
//
// The existing tests above all share the shape
//     reset { let k = shift(); abort(k) }
// where the "delimited context" is literally just `abort(k)`. Those tests
// cannot distinguish a real implementation from an exception/longjmp.
// The tests below probe each of the three properties directly.

/// Property 1: the computation between shift and reset must be captured.
/// The "delimited context" is `v + 1` — on resume, that addition must
/// re-execute with the resume value bound to v.
#[test]
fn shift_captures_intervening_computation() {
    let src = r#"
        fn main() {
            reset {
                let v = shift |k| { resume(k, 5) };
                v + 1
            }
        }
    "#;
    // Classic (reset (+ 1 (shift k (k 5)))) = 6.
    assert_eq!(run_interp(src, "main", &[]), 6);
}

/// Multi-shot through the new `shift |k|` form: the handler invokes the
/// continuation twice with different values, and each invocation must
/// independently run the delimited context `v + 1`.
#[test]
fn shift_multi_shot_same_handler() {
    let src = r#"
        fn invoke(k: cont, v) { resume(k, v) }
        fn main() {
            reset {
                let v = shift |k| {
                    let a = invoke(k, 10);
                    let b = invoke(k, 32);
                    a + b
                };
                v + 1
            }
        }
    "#;
    // Each k-invocation runs `v + 1`. Results: (10+1) + (32+1) = 44.
    assert_eq!(run_interp(src, "main", &[]), 44);
}

/// If the shift handler never invokes k and just returns a constant, the
/// delimited context `v + 1` is discarded and the reset yields the
/// constant. Matches (reset (+ 1 (shift k 99))) = 99.
#[test]
fn shift_handler_discards_continuation() {
    let src = r#"
        fn main() {
            reset {
                let v = shift |k| { 99 };
                v + 1
            }
        }
    "#;
    assert_eq!(run_interp(src, "main", &[]), 99);
}

/// Prove that post-resume code in the *handler* actually runs and sees
/// the resumed value. If resume were tail-only (the old bug), `r * 2`
/// after it would be silently dropped.
#[test]
fn shift_resume_returns_to_post_resume_code() {
    let src = r#"
        fn main() {
            reset {
                let v = shift |k| {
                    let r = resume(k, 20);
                    r * 2
                };
                v + 1
            }
        }
    "#;
    // resume(k, 20) runs delimited context → 21 → r=21 → r*2=42.
    assert_eq!(run_interp(src, "main", &[]), 42);
}

/// Two captures in the same reset body: the first shift captures a
/// delimited context that itself contains another shift. Exercises
/// shift-after-shift within the same reset.
#[test]
fn shift_two_captures_sequentially() {
    let src = r#"
        fn main() {
            reset {
                let v1 = shift |k1| { resume(k1, 5) };
                let v2 = shift |k2| { resume(k2, 7) };
                v1 + v2
            }
        }
    "#;
    // k1(5) re-enters at "let v2 = ...", runs the second shift, k2(7)
    // re-enters at "v1 + v2", yields 12.
    assert_eq!(run_interp(src, "main", &[]), 12);
}

/// Property 2: resume must return the prompt's value to its caller so
/// the caller can keep computing with it.
#[test]
fn resume_returns_value_to_caller() {
    let src = r#"
        fn capture_it() -> cont {
            reset { let k: cont = shift(); abort(k) }
        }
        fn do_resume(k: cont, v) {
            let r = resume(k, v);
            r + 100
        }
        fn main() {
            let k: cont = capture_it();
            do_resume(k, 42)
        }
    "#;
    // Real delimited: resume(k, 42) returns 42 to do_resume, which then
    // computes 42 + 100 = 142.
    assert_eq!(run_interp(src, "main", &[]), 142);
}

/// Property 3: multi-shot — the same continuation invoked twice yields
/// two independent results that can be combined.
#[test]
fn multi_shot_continuation_is_reentrant() {
    let src = r#"
        fn capture_it() -> cont {
            reset { let k: cont = shift(); abort(k) }
        }
        fn do_clone(k: cont) -> cont { clone(k) }
        fn invoke(k: cont, v) {
            let r = resume(k, v);
            r
        }
        fn main() {
            let k:  cont = capture_it();
            let k1: cont = do_clone(k);
            let k2: cont = do_clone(k);
            let a = invoke(k1, 10);
            let b = invoke(k2, 32);
            a + b
        }
    "#;
    // Real multi-shot: 10 + 32 = 42. A one-shot / stack-replacing impl
    // cannot even execute the second invoke — the first resume erases
    // main's frame.
    assert_eq!(run_interp(src, "main", &[]), 42);
}
