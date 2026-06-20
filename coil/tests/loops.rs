//! The `loop`/`break`/`continue` structured-control-flow primitive (the only 3
//! core forms added), and the `while`/`for` MACROS built over it in
//! lib/control.coil — proof the loop *features* live in userland, not the core.

mod common;
use common::build_and_run;

// ---- core primitive -------------------------------------------------------

#[test]
fn loop_break_with_value() {
    // Sum 0..10 = 45, returned by `break`.
    let code = build_and_run(
        r#"(defn main [] (-> :i64)
             (let [(mut i) 0 (mut acc) 0]
               (loop
                 (if (icmp-ge (load i) 10)
                     (break (load acc))
                     (do (store! acc (iadd (load acc) (load i)))
                         (store! i (iadd (load i) 1)))))))"#,
    );
    assert_eq!(code, 45);
}

#[test]
fn loop_continue_skips_rest_of_body() {
    // Sum only even i in 0..10 = 20; `continue` jumps back to the header.
    let code = build_and_run(
        r#"(defn main [] (-> :i64)
             (let [(mut i) 0 (mut acc) 0]
               (loop
                 (if (icmp-ge (load i) 10) (break (load acc))
                   (do (let [cur (load i)]
                         (store! i (iadd cur 1))
                         (if (icmp-eq (irem cur 2) 1)
                             (continue)
                             (store! acc (iadd (load acc) cur)))))))))"#,
    );
    assert_eq!(code, 20);
}

#[test]
fn labeled_break_exits_outer_loop() {
    // Inner loop breaks the labeled outer loop with a value.
    let code = build_and_run(
        r#"(defn main [] (-> :i64)
             (let [(mut n) 0]
               (loop :outer
                 (loop
                   (store! n (iadd (load n) 1))
                   (if (icmp-ge (load n) 5) (break :outer (load n)) (continue))))))"#,
    );
    assert_eq!(code, 5);
}

#[test]
fn break_outside_loop_is_an_error() {
    let err = coil::check_source("(defn main [] (-> :i64) (break 3))").unwrap_err();
    assert!(err.contains("break outside of a loop"), "got: {err}");
}

#[test]
fn continue_outside_loop_is_an_error() {
    let err = coil::check_source("(defn main [] (-> :i64) (continue))").unwrap_err();
    assert!(err.contains("continue outside of a loop"), "got: {err}");
}

#[test]
fn break_to_unknown_label_is_an_error() {
    let err = coil::check_source(
        "(defn main [] (-> :i64) (loop (break :nope 1)))",
    )
    .unwrap_err();
    assert!(err.contains("unknown loop label ':nope'"), "got: {err}");
}

#[test]
fn break_value_type_mismatch_is_an_error() {
    let err = coil::check_source(
        "(defn main [] (-> :i64) (loop (if 1 (break 5) (break (zeroed (ptr i64))))))",
    )
    .unwrap_err();
    assert!(err.contains("different value types"), "got: {err}");
}

#[test]
fn break_typed_as_never_needs_no_dummy_branch() {
    // break/continue type as Never (bottom), so `(if c (do …non-i64…) (break))`
    // reconciles to the non-diverging branch's type — no dummy value required.
    let code = build_and_run(
        r#"(defn main [] (-> :i64)
             (let [(mut acc) 0 (mut i) 0]
               (loop
                 (if (icmp-lt (load i) 5)
                     (do (store! acc (iadd (load acc) (load i)))
                         (store! i (iadd (load i) 1)))
                     (break)))
               (load acc)))"#,
    );
    assert_eq!(code, 10);
}

#[test]
fn never_branch_in_if_yields_other_type() {
    // (if c (break v) 5) — the break arm is Never, so the `if` is i64.
    let code = build_and_run(
        "(defn pick [(c i64)] (-> i64) (loop (break (if (icmp-eq c 1) (break 99) 5))))
         (defn main [] (-> :i64) (pick 0))",
    );
    assert_eq!(code, 5);
}

// ---- derived macros (lib/control.coil) ------------------------------------

const IMPORT: &str = "(module app)\n(import \"lib/control.coil\")\n";

fn run_with(body: &str) -> i32 {
    build_and_run(&format!("{IMPORT}{body}"))
}

#[test]
fn while_macro_sums() {
    let code = run_with(
        r#"(defn main [] (-> :i64)
             (let [(mut i) 0 (mut acc) 0]
               (while (icmp-lt (load i) 10)
                 (store! acc (iadd (load acc) (load i)))
                 (store! i (iadd (load i) 1)))
               (load acc)))"#,
    );
    assert_eq!(code, 45);
}

#[test]
fn for_macro_sums() {
    let code = run_with(
        r#"(defn main [] (-> :i64)
             (let [(mut acc) 0]
               (for [i 0 10] (store! acc (iadd (load acc) i)))
               (load acc)))"#,
    );
    assert_eq!(code, 45);
}

#[test]
fn for_continue_does_not_skip_increment() {
    // Skip odd i via `continue`; the counter must still advance (C semantics).
    let code = run_with(
        r#"(defn main [] (-> :i64)
             (let [(mut acc) 0]
               (for [i 0 10]
                 (if (icmp-eq (irem i 2) 1) (continue) 0)
                 (store! acc (iadd (load acc) i)))
               (load acc)))"#,
    );
    assert_eq!(code, 20);
}

#[test]
fn for_break_exits_early() {
    let code = run_with(
        r#"(defn main [] (-> :i64)
             (let [(mut acc) 0]
               (for [i 0 100]
                 (if (icmp-ge i 5) (break) 0)
                 (store! acc (iadd (load acc) i)))
               (load acc)))"#,
    );
    assert_eq!(code, 10);
}

#[test]
fn for_bad_binding_hard_errors() {
    let src = format!("{IMPORT}(defn main [] (-> :i64) (for [i 0] 1))");
    let err = coil::check_source(&src).unwrap_err();
    assert!(err.contains("binding must be [i LO HI]"), "got: {err}");
}
