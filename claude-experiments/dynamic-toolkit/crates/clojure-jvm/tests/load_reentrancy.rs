//! Validate that `clojure.lang.RT/load` can reentrantly compile + execute
//! a resource's forms while the outer JIT frame (the `load` call itself)
//! is live on the native stack. This is the load-the-Clojure-way mechanism
//! in isolation, before the real protocols port depends on it.

use clojure_jvm::lang::compiler::Session;

#[test]
fn rt_load_reentrant_trivial_resource() {
    let mut sess = Session::new();
    // Directly invoke the runtime loader (bypassing core's `load` path
    // munging) on the inline smoke-test resource, which does
    // `(def reentrancy-marker 4242)`.
    sess.eval_str("(clojure.lang.RT/load \"trivial-reentrancy-test\")");
    // The def from inside the reentrant load must be visible afterward.
    let bits = sess.eval_str("reentrancy-marker");
    let got = clojure_jvm::runtime::arg_to_i64(bits);
    assert_eq!(got, 4242, "reentrant load did not define the marker var");
}
