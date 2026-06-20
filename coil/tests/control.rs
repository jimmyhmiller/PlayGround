//! lib/control.coil — control-flow sugar as pure macros (when/unless/cond and
//! the comptime-unrolled inline-for). Nothing in the compiler knows about them;
//! these tests compile+run real programs that `import "lib/control.coil"`.

mod common;
use common::build_and_run;

const IMPORT: &str = "(module app)\n(import \"lib/control.coil\")\n";

fn run_with(body: &str) -> i32 {
    build_and_run(&format!("{IMPORT}{body}"))
}

#[test]
fn inline_for_unrolls_and_sums() {
    // i = 0,1,2,3 unrolled; sum = 6.
    let code = run_with(
        r#"(defn main [] (-> :i64)
             (let [(mut acc) 0]
               (inline-for [i 0 4] (store! acc (iadd (load acc) i)))
               (load acc)))"#,
    );
    assert_eq!(code, 6);
}

#[test]
fn inline_for_empty_range_is_zero() {
    // LO >= HI ⇒ no iterations; the form is 0, leaving acc untouched.
    let code = run_with(
        r#"(defn main [] (-> :i64)
             (let [(mut acc) 7]
               (inline-for [i 5 5] (store! acc 999))
               (load acc)))"#,
    );
    assert_eq!(code, 7);
}

#[test]
fn nested_inline_for() {
    // 3x3 grid, count = 9 increments of 1.
    let code = run_with(
        r#"(defn main [] (-> :i64)
             (let [(mut n) 0]
               (inline-for [i 0 3]
                 (inline-for [j 0 3]
                   (store! n (iadd (load n) 1))))
               (load n)))"#,
    );
    assert_eq!(code, 9);
}

#[test]
fn when_and_unless() {
    // when fires (true), unless fires (condition false): 1 + 10 + 100 = 111.
    let code = run_with(
        r#"(defn main [] (-> :i64)
             (let [(mut a) 1]
               (when   (icmp-eq (load a) 1) (store! a (iadd (load a) 10)))
               (unless (icmp-eq (load a) 0) (store! a (iadd (load a) 100)))
               (load a)))"#,
    );
    assert_eq!(code, 111);
}

#[test]
fn cond_picks_first_true_then_else() {
    // 21 -> middle clause -> 2; a value below all -> else 3.
    let pick = r#"(defn classify [(x :i64)] (-> :i64)
                    (cond (icmp-lt x 10) 1
                          (icmp-lt x 30) 2
                          3))"#;
    assert_eq!(run_with(&format!("{pick}\n(defn main [] (-> :i64) (classify 21))")), 2);
    assert_eq!(run_with(&format!("{pick}\n(defn main [] (-> :i64) (classify 5))")), 1);
    assert_eq!(run_with(&format!("{pick}\n(defn main [] (-> :i64) (classify 99))")), 3);
}

#[test]
fn inline_for_bad_binding_hard_errors() {
    // Misuse must hard-error at expansion (no silent wrong code), via the
    // compile-time `error` builtin.
    let src = format!("{IMPORT}(defn main [] (-> :i64) (inline-for [i 0] 1))");
    let err = coil::check_source(&src).unwrap_err();
    assert!(err.contains("binding must be [i LO HI]"), "got: {err}");
}
