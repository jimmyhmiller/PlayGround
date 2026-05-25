//! `(throw v)` and `(try ... (catch _ e ...))`.
//!
//! In-function try/catch is wired through dynlower's prompt
//! machinery (`abort_to_prompt` for throw, `push_prompt`/`pop_prompt`
//! for the try region). Cross-function throws still take the
//! `clj_throw` extern path (Rust panic) — see TODO.md for the
//! follow-up that wires `JitOutcome::AbortToPrompt` propagation
//! across compiled functions.

use clojure::Engine;

#[test]
fn throw_evaluates_lazily() {
    // The throw is in the unreachable branch; never fires.
    let e = Engine::new();
    let v = e.eval("(if true 1 (throw \"shouldn't fire\"))");
    assert_eq!(e.print(v), "1");
}

#[test]
fn throw_in_dead_branch_doesnt_fire() {
    let e = Engine::new();
    let v = e.eval(
        "(def f (fn [n] (if (= n 0) :ok (throw \"never\")))) \
         (f 0)",
    );
    assert_eq!(e.print(v), ":ok");
}

#[test]
fn try_returns_body_value_when_no_throw() {
    let e = Engine::new();
    let v = e.eval("(try 42 (catch _ e :unreachable))");
    assert_eq!(e.print(v), "42");
}

#[test]
fn try_catches_throw_and_binds_value() {
    let e = Engine::new();
    let v = e.eval("(try (throw :boom) (catch _ e e))");
    assert_eq!(e.print(v), ":boom");
}

#[test]
fn catch_handler_runs_with_thrown_value() {
    // Handler receives the thrown value and computes from it.
    let e = Engine::new();
    let v = e.eval("(try (throw 7) (catch _ e (+ e 1)))");
    assert_eq!(e.print(v), "8");
}

#[test]
fn try_body_with_multiple_forms() {
    let e = Engine::new();
    let v = e.eval(
        "(try (+ 1 2) (+ 3 4) (throw :late) (catch _ e e))",
    );
    assert_eq!(e.print(v), ":late");
}

#[test]
fn nested_try_innermost_wins() {
    let e = Engine::new();
    let v = e.eval(
        "(try \
           (try (throw :inner) (catch _ e :caught-inner)) \
         (catch _ e :caught-outer))",
    );
    assert_eq!(e.print(v), ":caught-inner");
}

#[test]
fn throw_through_inner_uncaught_reaches_outer() {
    // Inner try doesn't catch (no throw inside its body); outer
    // catches the explicit throw that follows.
    let e = Engine::new();
    let v = e.eval(
        "(try \
           (try 1 (catch _ e :inner-caught)) \
           (throw :outer-throw) \
         (catch _ e e))",
    );
    assert_eq!(e.print(v), ":outer-throw");
}

#[test]
fn cross_function_uncaught_throw_propagates() {
    // The throw is in a callee; it should bubble out through the
    // call site and surface at eval as a panic with the printed
    // value. Verifies the JitOutcome::Exception propagation chain
    // (asm stub → indirect call site → caller → eval).
    let result = std::panic::catch_unwind(|| {
        let e = Engine::new();
        e.eval("(def f (fn [] (throw :boom))) (f)");
    });
    let err = result.expect_err("expected uncaught throw to panic");
    let msg = err
        .downcast_ref::<String>()
        .map(|s| s.clone())
        .or_else(|| err.downcast_ref::<&str>().map(|s| s.to_string()))
        .unwrap_or_default();
    assert!(msg.contains("Exception"), "panic message: {:?}", msg);
    assert!(msg.contains(":boom"), "panic message: {:?}", msg);
}

// === Cross-function CATCH =================================
//
// Implemented via a synthesized try-body fn wrapped in `Invoke`.
// A throw anywhere inside the body (including in nested function
// calls) produces `JitOutcome::Exception(v)`; the outer Invoke's
// exception path routes the value to handler_bb as block_param 0.

#[test]
fn cross_function_caught_one_hop() {
    // Throw inside a callee, caught by try in the caller.
    let e = Engine::new();
    let v = e.eval(
        "(def f (fn [] (throw :boom))) \
         (try (f) (catch _ e e))",
    );
    assert_eq!(e.print(v), ":boom");
}

#[test]
fn cross_function_caught_two_hops() {
    // Throw three call frames deep; caught at top.
    let e = Engine::new();
    let v = e.eval(
        "(def inner (fn [] (throw :deep))) \
         (def middle (fn [] (inner))) \
         (def outer (fn [] (middle))) \
         (try (outer) (catch _ e e))",
    );
    assert_eq!(e.print(v), ":deep");
}

#[test]
fn nested_try_inner_catches_cross_function_throw() {
    let e = Engine::new();
    let v = e.eval(
        "(def f (fn [] (throw :inner-throw))) \
         (try \
           (try (f) (catch _ e [:inner e])) \
         (catch _ e [:outer e]))",
    );
    assert_eq!(e.print(v), "[:inner :inner-throw]");
}

#[test]
fn nested_try_outer_catches_when_inner_rethrows() {
    // Inner catch handler itself throws; outer catches.
    let e = Engine::new();
    let v = e.eval(
        "(try \
           (try (throw :first) (catch _ e (throw :second))) \
         (catch _ e e))",
    );
    assert_eq!(e.print(v), ":second");
}

#[test]
fn cross_function_two_hops_uncaught_throw_propagates() {
    // Three-deep call chain. Verifies Exception bubbles through
    // multiple plain Call sites.
    let result = std::panic::catch_unwind(|| {
        let e = Engine::new();
        e.eval(
            "(def inner (fn [] (throw :deep))) \
             (def middle (fn [] (inner))) \
             (def outer (fn [] (middle))) \
             (outer)",
        );
    });
    let err = result.expect_err("expected uncaught throw to panic");
    let msg = err
        .downcast_ref::<String>()
        .map(|s| s.clone())
        .or_else(|| err.downcast_ref::<&str>().map(|s| s.to_string()))
        .unwrap_or_default();
    assert!(msg.contains(":deep"), "panic message: {:?}", msg);
}

// === ex-info / ex-message / ex-data / ex-cause ===========================

#[test]
fn ex_info_creates_an_exception_with_message_and_data() {
    let e = Engine::new();
    let v = e.eval(
        "(try (throw (ex-info \"bad\" {:k 1})) \
            (catch _ ex (ex-message ex)))",
    );
    assert_eq!(e.print(v), "\"bad\"");
}

#[test]
fn ex_data_returns_the_data_map() {
    let e = Engine::new();
    let v = e.eval(
        "(try (throw (ex-info \"oops\" {:tag :boom})) \
            (catch _ ex (get (ex-data ex) :tag)))",
    );
    assert_eq!(e.print(v), ":boom");
}

// === Type-filtered catches ================================================

#[test]
fn catch_matches_exact_type() {
    let e = Engine::new();
    let v = e.eval(
        "(try (throw (ex-info \"x\" {})) \
            (catch ExceptionInfo ex :got-it))",
    );
    assert_eq!(e.print(v), ":got-it");
}

#[test]
fn catch_skips_non_matching_type_falls_through_to_next() {
    let e = Engine::new();
    let v = e.eval(
        "(try (throw (ex-info \"x\" {})) \
            (catch NotARealType ex :wrong) \
            (catch ExceptionInfo ex :right))",
    );
    assert_eq!(e.print(v), ":right");
}

#[test]
fn catch_default_keyword_matches_anything() {
    let e = Engine::new();
    let v = e.eval(
        "(try (throw (ex-info \"x\" {})) \
            (catch :default ex :caught-default))",
    );
    assert_eq!(e.print(v), ":caught-default");
}

#[test]
fn catch_underscore_matches_anything() {
    let e = Engine::new();
    let v = e.eval(
        "(try (throw :raw-keyword) \
            (catch _ ex ex))",
    );
    assert_eq!(e.print(v), ":raw-keyword");
}

#[test]
fn unmatched_typed_catches_re_raise_and_outer_catches() {
    let e = Engine::new();
    let v = e.eval(
        "(try \
           (try (throw (ex-info \"deep\" {})) \
             (catch NoMatchType ex :wrong)) \
         (catch _ ex (ex-message ex)))",
    );
    assert_eq!(e.print(v), "\"deep\"");
}

// === Finally ==============================================================

#[test]
fn finally_value_ignored_normal_exit() {
    let e = Engine::new();
    let v = e.eval("(try 42 (catch _ e :u) (finally :ignored))");
    assert_eq!(e.print(v), "42");
}

#[test]
fn finally_value_ignored_caught_exit() {
    let e = Engine::new();
    let v = e.eval(
        "(try (throw :boom) (catch _ e :caught) (finally :ignored))",
    );
    assert_eq!(e.print(v), ":caught");
}

#[test]
fn finally_side_effect_normal_exit() {
    let e = Engine::new();
    e.eval("(def log (atom :before))");
    let v = e.eval(
        "(try 99 (catch _ e :u) (finally (reset! log :after)))",
    );
    assert_eq!(e.print(v), "99");
    let log = e.eval("@log");
    assert_eq!(e.print(log), ":after");
}

#[test]
fn finally_side_effect_caught_exit() {
    let e = Engine::new();
    e.eval("(def log2 (atom :before))");
    let v = e.eval(
        "(try (throw :boom) \
            (catch _ e :handled) \
            (finally (reset! log2 :after)))",
    );
    assert_eq!(e.print(v), ":handled");
    let log = e.eval("@log2");
    assert_eq!(e.print(log), ":after");
}

// === Built-in errors as catchable exceptions ==============================

#[test]
fn arity_mismatch_throws_arity_exception() {
    // Calling a 1-arg fn with 2 args throws ArityException.
    let e = Engine::new();
    let v = e.eval(
        "(def f (fn [x] x)) \
         (try (f 1 2) (catch ArityException ex :caught))",
    );
    assert_eq!(e.print(v), ":caught");
}

#[test]
fn arity_exception_has_a_message() {
    let e = Engine::new();
    let v = e.eval(
        "(def g (fn [a b] (+ a b))) \
         (try (g 1) (catch ArityException ex (ex-message ex)))",
    );
    // We just verify the message is a non-nil string.
    let s = e.print(v);
    assert!(s.contains("arg"), "expected arity message, got {s:?}");
}

#[test]
fn multi_arity_no_matching_throws_arity_exception() {
    // No clause matches 3 args.
    let e = Engine::new();
    let v = e.eval(
        "(def h (fn ([a] :one) ([a b] :two))) \
         (try (h 1 2 3) (catch ArityException ex :caught))",
    );
    assert_eq!(e.print(v), ":caught");
}

#[test]
fn arity_catchall_works_too() {
    let e = Engine::new();
    let v = e.eval(
        "(def k (fn [x] x)) \
         (try (k 1 2) (catch _ ex :caught))",
    );
    assert_eq!(e.print(v), ":caught");
}

#[test]
fn finally_runs_when_handler_itself_throws() {
    // The catch handler re-throws. Finally MUST still run — the
    // exception then propagates to the outer try.
    let e = Engine::new();
    e.eval("(def finally_log (atom :before))");
    let v = e.eval(
        "(try \
           (try (throw :first) \
             (catch _ e (throw :second)) \
             (finally (reset! finally_log :ran))) \
         (catch _ e e))",
    );
    // Outer catches the re-thrown :second.
    assert_eq!(e.print(v), ":second");
    // And the finally ran before the re-throw.
    let log = e.eval("@finally_log");
    assert_eq!(e.print(log), ":ran");
}

#[test]
fn finally_runs_when_finally_itself_throws_overriding_original() {
    // Real Clojure semantics: if finally throws, its exception
    // wins over any in-flight exception.
    let e = Engine::new();
    let v = e.eval(
        "(try \
           (try (throw :original) \
             (catch _ e (throw :handler-throw)) \
             (finally (throw :finally-throw))) \
         (catch _ e e))",
    );
    assert_eq!(e.print(v), ":finally-throw");
}

#[test]
fn finally_runs_when_body_throws_unmatched_handler_arms() {
    // No matching catch arm + finally: finally runs, then re-raise.
    let e = Engine::new();
    e.eval("(def fin2_log (atom :before))");
    let v = e.eval(
        "(try \
           (try (throw (ex-info \"x\" {})) \
             (catch NoMatch e :wrong) \
             (finally (reset! fin2_log :ran))) \
         (catch _ e :outer-caught))",
    );
    assert_eq!(e.print(v), ":outer-caught");
    let log = e.eval("@fin2_log");
    assert_eq!(e.print(log), ":ran");
}

#[test]
fn finally_only_no_catch_normal_exit() {
    let e = Engine::new();
    e.eval("(def log4 (atom :before))");
    let v = e.eval("(try 7 (finally (reset! log4 :ran)))");
    assert_eq!(e.print(v), "7");
    let log = e.eval("@log4");
    assert_eq!(e.print(log), ":ran");
}
