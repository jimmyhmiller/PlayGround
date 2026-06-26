//! `defer`/`scope` guards — a pure macro over `block`/`break` (lib/control.coil).
//! Cleanups run LIFO on BOTH normal exit and `return-from` early exit; any
//! break/continue that would escape the scope past its cleanups is a hard error
//! (the no-silent-skip guarantee), while control targeting an in-body loop or the
//! scope itself is allowed.

mod common;
use common::build_and_run;

const IMPORT: &str = "(module app)\n(import \"lib/control.coil\" :use *)\n";

fn run_with(body: &str) -> i32 {
    build_and_run(&format!("{IMPORT}{body}"))
}

fn err_of(body: &str) -> String {
    coil::check_source(&format!("{IMPORT}{body}")).unwrap_err()
}

#[test]
fn cleanup_runs_on_normal_exit() {
    // body: log += 10; defer: log += 1 ⇒ 11.
    let code = run_with(
        r#"(defn main [] (-> :i64)
             (let [(mut log) 0]
               (scope :s
                 (defer (store! log (iadd (load log) 1)))
                 (store! log (iadd (load log) 10)))
               (load log)))"#,
    );
    assert_eq!(code, 11);
}

#[test]
fn cleanups_run_lifo() {
    // Two defers: A doubles, B increments. LIFO ⇒ B then A: 5→6→12.
    let code = run_with(
        r#"(defn main [] (-> :i64)
             (let [(mut log) 0]
               (scope :s
                 (defer (store! log (imul (load log) 2)))
                 (defer (store! log (iadd (load log) 1)))
                 (store! log 5))
               (load log)))"#,
    );
    assert_eq!(code, 12);
}

#[test]
fn cleanup_runs_on_return_from_early_exit() {
    // return-from as the GENUINE LAST form of the scope (no dead code after it) —
    // the scope wrapper's outer break value is then a divergent `do` (type Never),
    // which must not break break-type unification. The defer must still fire.
    let code = run_with(
        r#"(defn main [] (-> :i64)
             (let [(mut log) 0]
               (scope :s
                 (defer (store! log (iadd (load log) 100)))
                 (store! log 1)
                 (return-from :s 0))
               (load log)))"#,
    );
    assert_eq!(code, 101); // body sets 1, return-from exits, defer +100
}

#[test]
fn break_continue_targeting_in_body_loop_is_allowed() {
    // A break to an in-body loop stays inside the scope: must compile + run.
    assert_eq!(
        run_with("(defn main [] (-> :i64) (scope :s (loop :inner (break :inner 9))))"),
        9,
    );
    // An unlabeled break inside an in-body `for` is fine; defer still runs.
    let code = run_with(
        r#"(defn main [] (-> :i64)
             (let [(mut acc) 0]
               (scope :s
                 (defer (store! acc (iadd (load acc) 100)))
                 (for [i 0 999] (if (icmp-ge i 3) (break) (store! acc (iadd (load acc) i)))))
               (load acc)))"#,
    );
    assert_eq!(code, 103); // 0+1+2 = 3, plus defer 100
}

// ---- the no-silent-skip guarantee: every escaping transfer hard-errors ----

#[test]
fn labeled_break_escaping_scope_errors() {
    let e = err_of("(defn main [] (-> :i64) (loop :outer (scope :s (defer 0) (break :outer 7))))");
    assert!(e.contains("escapes the (scope"), "got: {e}");
}

#[test]
fn break_escaping_through_in_body_loop_errors() {
    // The break is inside an in-body loop but targets a label OUTSIDE the scope.
    let e = err_of(
        "(defn main [] (-> :i64) (loop :outer (scope :s (defer 0) (loop :inner (break :outer 1)))))",
    );
    assert!(e.contains("escapes the (scope"), "got: {e}");
}

#[test]
fn break_escaping_through_nested_scope_errors() {
    let e = err_of(
        "(defn main [] (-> :i64) (scope :s1 (defer 0) (scope :s2 (defer 0) (break :s1 1))))",
    );
    assert!(e.contains("escapes the (scope"), "got: {e}");
}

#[test]
fn unlabeled_break_at_scope_level_errors() {
    let e = err_of("(defn main [] (-> :i64) (loop :outer (scope :s (defer 0) (break))))");
    assert!(e.contains("unlabeled break/continue cannot exit"), "got: {e}");
}

#[test]
fn continue_targeting_scope_label_errors() {
    let e = err_of("(defn main [] (-> :i64) (scope :s (defer 0) (continue :s)))");
    assert!(e.contains("cannot target the scope label"), "got: {e}");
}

#[test]
fn defer_outside_scope_errors() {
    let e = err_of("(defn main [] (-> :i64) (defer (load 0)) 1)");
    assert!(e.contains("only valid as a DIRECT form inside (scope"), "got: {e}");
}

#[test]
fn defer_nested_in_if_errors() {
    let e = err_of("(defn main [] (-> :i64) (scope :s (if 1 (defer 0) 0) 5))");
    assert!(e.contains("only valid as a DIRECT form inside (scope"), "got: {e}");
}
