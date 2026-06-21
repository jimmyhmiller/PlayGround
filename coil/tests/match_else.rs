//! lib/match.coil — `match-else`, a `match` with a catch-all `_` arm, implemented
//! as a PURE LIBRARY MACRO over the comptime reflection bridge (variant-sum +
//! sum-variants): it reflects the sum from an arm's variant name and expands `_`
//! into an explicit arm for each uncovered variant. No compiler `_` support.

mod common;
use common::build_and_run;

const SUM: &str = "(module app)\n\
    (import \"lib/match.coil\" :use *)\n\
    (defsum E (A []) (B [(n i64)]) (C [(n i64)]) (D []) (F []))\n";

#[test]
fn catch_all_collapses_uncovered_variants() {
    // B -> n, C -> n*10, everything else (A/D/F) -> 0, via one `_`.
    let code = build_and_run(&format!(
        "{SUM}(defn v [(e E)] (-> i64) (match-else e (B [n] n) (C [n] (imul n 10)) (_ 0)))\n\
         (defn main [] (-> i64) (iadd (v (B 2)) (iadd (v (C 4)) (v (D)))))" // 2 + 40 + 0
    ));
    assert_eq!(code, 42);
}

#[test]
fn catch_all_with_all_but_one_covered() {
    let code = build_and_run(&format!(
        "{SUM}(defn v [(e E)] (-> i64)\n\
           (match-else e (A [] 1) (B [n] 2) (C [n] 3) (D [] 4) (_ 40)))\n\
         (defn main [] (-> i64) (iadd (v (F)) (v (B 9))))" // 40 (F via _) + 2
    ));
    assert_eq!(code, 42);
}

#[test]
fn catch_all_binds_dont_collide_with_user_names() {
    // The generated arms use gensym binds, so a user `n` in the default is safe.
    let code = build_and_run(&format!(
        "{SUM}(defn v [(e E) (n i64)] (-> i64) (match-else e (B [n] n) (_ n)))\n\
         (defn main [] (-> i64) (iadd (v (B 2) 100) (v (A) 40)))" // 2 (arm n) + 40 (param n)
    ));
    assert_eq!(code, 42);
}

#[test]
fn last_arm_must_be_wildcard() {
    let err = coil::check_source(&format!(
        "{SUM}(defn v [(e E)] (-> i64) (match-else e (B [n] n)))\n(defn main [] (-> i64) 0)"
    ))
    .unwrap_err();
    assert!(err.contains("last arm must be (_") , "got: {err}");
}

#[test]
fn needs_an_explicit_arm_to_infer_the_sum() {
    let err = coil::check_source(&format!(
        "{SUM}(defn v [(e E)] (-> i64) (match-else e (_ 0)))\n(defn main [] (-> i64) 0)"
    ))
    .unwrap_err();
    assert!(err.contains("at least one explicit"), "got: {err}");
}
