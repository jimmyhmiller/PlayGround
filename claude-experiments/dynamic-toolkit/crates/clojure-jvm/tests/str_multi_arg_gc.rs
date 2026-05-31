//! Reproducer originally for a JIT GC-root bug exposed by clojure.core
//! `str` under GC pressure.
//!
//! Two GC-root bugs that used to crash here are now FIXED:
//!   1. Var-root dangling — a Var's value now lives in the GC-rooted
//!      `VAR_ROOTS` slot table (the var-deref result no longer dangles).
//!   2. JIT closure-spill dangling — clojure.core `str`'s variadic path
//!      recurses `JIT → runtime helper → JIT → …`, and the parked-thread
//!      root walker (`walk_parked_thread_jit_roots`, used by minor GC and
//!      alloc-path major GC) used to STOP at the first intervening runtime
//!      frame, scanning only the deepest JIT frame and dropping every
//!      outer recursion frame's roots. An alloc-path collection then left
//!      those spill slots un-forwarded → stale closure pointer → crash at
//!      a later collection. Fixed by having that walker traverse through
//!      intervening non-JIT frames up to the JIT-entry fence, matching
//!      `walk_jit_ancestor_roots`. Regression test:
//!      `dynlower::tests::parked_walker_traverses_interleaved_host_frames`.
//!
//! This test is STILL ignored: with the GC bugs fixed it now blocks on
//! two SEPARATE, pre-existing issues unrelated to GC root scanning —
//!   * the 64KB stress semi-space OOMs because every `eval_form`
//!     permanently compiles a fresh JIT function and grows the literal
//!     pool (a compile-per-form leak), and
//!   * with a larger heap, the deep `str` native recursion overflows the
//!     thread stack.
//! Run with `--ignored` to observe the current (non-GC) failure.

use clojure_jvm::lang::compiler::Session;
use clojure_jvm::lang::lisp_reader::read_str;

#[test]
#[ignore = "hits a separate pre-existing JIT closure-spill GC-root bug; see module docs"]
fn str_multi_arg_under_gc() {
    let mut sess = Session::new_with_clojure_core();
    // Many string args force repeated toString allocations (each a GC
    // safepoint under pressure). The variadic StringBuilder path is what
    // the loader exercised when it crashed.
    for i in 0..200 {
        let src = format!(
            "(str \"alpha-{i}\" \"beta-{i}\" \"gamma-{i}\" \"delta-{i}\" \"epsilon-{i}\")"
        );
        let form = read_str(&src).expect("read");
        sess.eval_form(form);
    }
}
