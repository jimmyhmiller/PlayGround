//! Seq / rest-arg semantics vs REAL Clojure.
//!
//! `seq_oracle/probe.clj` runs byte-identically on microclj and on real
//! Clojure; `seq_oracle/expected.txt` is what `clojure -M probe.clj` actually
//! printed (regenerate with `clojure-stub/tests/seq_oracle/refresh.sh`). Real
//! Clojure is the specification, so nothing here is hand-authored — a
//! hand-written expectation would only encode what we ASSUME Clojure does, and
//! this whole file exists because that assumption was wrong.
//!
//! microclj does not match on every line yet. The gap is named, line by line,
//! in `KNOWN_DIVERGENCES` — with the reason. The test asserts BOTH directions:
//!
//!   * every line not listed must match Clojure exactly, and
//!   * every listed line must still actually diverge.
//!
//! So a fix cannot land without deleting its entry (the list can only shrink),
//! and an unnoticed regression on a line we already match fails the build.

use std::process::Command;

/// Lines where microclj knowingly differs from Clojure, and why. DELETE an
/// entry when you fix it — the test fails if a listed line starts matching.
const KNOWN_DIVERGENCES: &[(&str, &str)] = &[
    // ── `apply` copies its seq instead of passing it through ──────────────
    // Clojure hands the applied seq to the callee UNTOUCHED: `identical?`
    // holds, laziness survives, and the rest arg keeps the source's shape
    // (chunked stays chunked, a list stays a list). microclj flattens it into
    // a fresh cons list, so every shape predicate reads "cons list" and a
    // lazy source is forced. Fixing this needs `apply` to resolve the callee's
    // arity and install the remaining seq as the rest slot — a change to the
    // call convention across all five backends.
    ("apply/range-3", "apply copies the seq; no passthrough"),
    ("apply/range-100", "apply copies the seq; no passthrough"),
    ("apply/vec-100", "apply copies the seq; no passthrough"),
    ("apply/lazy-map", "apply copies the seq; no passthrough"),
    ("apply/req1-range", "apply copies the seq; no passthrough"),
    ("apply/leading+seq", "apply copies the seq; no passthrough"),
    ("identical/range", "apply copies the seq; no passthrough"),
    ("lazy/unforced-after-apply", "apply FORCES the seq it should pass through"),
    // ── a direct variadic call builds a cons list, not an ArraySeq ────────
    // Clojure's rest arg for a direct call is an ArraySeq: `list?` is FALSE.
    // microclj conses, so `list?` is true. Only `list?` differs — every other
    // predicate and value already agrees.
    ("direct/1-extra", "rest arg is a cons list, not an ArraySeq: list? true"),
    ("direct/3-extra", "rest arg is a cons list, not an ArraySeq: list? true"),
    ("direct/40-extra", "rest arg is a cons list, not an ArraySeq: list? true"),
    ("req1/direct", "rest arg is a cons list, not an ArraySeq: list? true"),
    ("req2/direct", "rest arg is a cons list, not an ArraySeq: list? true"),
    // ── seq TYPES this dialect does not model ────────────────────────────
    // `counted?`/`chunked-seq?` are `instance?` checks on Clojure's concrete
    // seq classes. `range` is a LongRange there (chunked AND counted); here it
    // is a plain LazySeq. And Clojure splits vector chunks (ChunkedSeq, which
    // IS Counted) from lazy chunks (ChunkedCons, which is not) — this dialect
    // has ONE ChunkedCons, so it cannot answer both ways.
    ("chunked/range", "range is a LazySeq here, not a chunked LongRange"),
    ("counted/range", "range is a LazySeq here, not a counted LongRange"),
    ("counted/vec-seq", "one ChunkedCons type; Clojure's vector ChunkedSeq is Counted"),
];

fn run_probe(jit: bool) -> Vec<(String, String)> {
    let root = concat!(env!("CARGO_MANIFEST_DIR"), "/..");
    let probe = format!("{}/clojure-stub/tests/seq_oracle/probe.clj", root);
    let mut cmd = Command::new(env!("CARGO_BIN_EXE_microclj"));
    if jit {
        cmd.arg("--jit");
    }
    let out = cmd.arg(&probe).current_dir(root).output().expect("run microclj");
    assert!(
        out.status.success(),
        "probe failed:\n{}",
        String::from_utf8_lossy(&out.stderr)
    );
    parse(&String::from_utf8_lossy(&out.stdout))
}

fn parse(s: &str) -> Vec<(String, String)> {
    s.lines()
        .filter_map(|l| l.split_once('\t'))
        .map(|(a, b)| (a.to_string(), b.to_string()))
        .collect()
}

fn check(jit: bool) {
    let expected = parse(include_str!("seq_oracle/expected.txt"));
    let got = run_probe(jit);
    assert_eq!(expected.len(), got.len(), "probe emitted a different number of lines");

    let mut regressions = Vec::new();
    let mut fixed = Vec::new();
    for ((elabel, eval), (glabel, gval)) in expected.iter().zip(got.iter()) {
        assert_eq!(elabel, glabel, "probe lines out of order");
        let known = KNOWN_DIVERGENCES.iter().find(|(l, _)| l == elabel);
        match (known, eval == gval) {
            // matches Clojure and wasn't expected to: someone fixed it.
            (Some((l, why)), true) => fixed.push(format!("  {l}  (was: {why})")),
            // differs from Clojure and isn't a known gap: a real regression.
            (None, false) => {
                regressions.push(format!("  {elabel}\n    clojure: {eval}\n    microclj: {gval}"))
            }
            _ => {}
        }
    }
    assert!(
        regressions.is_empty(),
        "microclj diverges from real Clojure on {} line(s) that previously matched:\n{}",
        regressions.len(),
        regressions.join("\n")
    );
    assert!(
        fixed.is_empty(),
        "{} line(s) now MATCH Clojure but are still listed in KNOWN_DIVERGENCES.\n\
         Delete their entries — the list must only ever shrink:\n{}",
        fixed.len(),
        fixed.join("\n")
    );
}

#[test]
fn seq_semantics_match_clojure_treewalk() {
    check(false);
}

#[test]
#[cfg(feature = "jit")]
fn seq_semantics_match_clojure_jit() {
    check(true);
}
