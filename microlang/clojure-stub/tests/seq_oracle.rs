//! Seq / rest-arg / identity semantics vs the REAL implementations.
//!
//! `seq_oracle/probe.clj` runs byte-identically on THREE runtimes: microclj,
//! JVM Clojure, and ClojureScript (on node). The two spec files are what those
//! implementations actually printed — never hand-authored, regenerate with
//! `seq_oracle/refresh.sh`:
//!
//!   expected.txt       JVM Clojure
//!   expected_cljs.txt  ClojureScript
//!
//! WHY TWO. The rule (Jimmy) for what microclj must reproduce: where Clojure
//! and ClojureScript AGREE, that is the language and we match it. Where they
//! DISAGREE, the answer is an artifact of the host, and copying the JVM's
//! accident would be copying an accident. Keeping both answers on disk makes
//! that rule EXECUTABLE rather than a judgement call — and it earned its keep
//! immediately, overturning three lines this file previously called microclj
//! bugs (`counted/vec-seq`, `apply/vec-100`, `apply/leading+seq`): Clojure and
//! ClojureScript disagree on all three, and microclj already answers as
//! ClojureScript does. There was nothing to fix.
//!
//! So the test asserts:
//!
//!   * AGREED lines (clj == cljs): microclj must match, unless listed in
//!     KNOWN_DIVERGENCES with a reason. That list can only ever SHRINK — a
//!     listed line that starts matching fails until its entry is deleted.
//!   * HOST-ARTIFACT lines (clj != cljs): microclj must still answer like ONE
//!     of the two real implementations. It may pick a side, but it may not
//!     invent a third answer.

use std::collections::HashMap;
use std::process::Command;

/// AGREED lines where microclj still differs from both real implementations,
/// and why. DELETE an entry when you fix it.
const KNOWN_DIVERGENCES: &[(&str, &str)] = &[
    // ── a range is chunked and is its own seq, but is not COUNTED ─────────
    // Clojure's LongRange and ClojureScript's Range agree that a range is
    // CHUNKED, COUNTED, and its OWN seq. The first and third now hold (the
    // first chunk is built eagerly, so `(range n)` IS a ChunkedCons rather
    // than a LazySeq wrapping one). `counted?` still does not: a ChunkedCons
    // chain has no O(1) length, which is the one thing a real Range type
    // would add.
    ("counted/range", "a ChunkedCons chain has no O(1) count; a real Range would"),
    ("apply/range-3/counted?", "a ChunkedCons chain has no O(1) count"),
    ("apply/range-100/counted?", "a ChunkedCons chain has no O(1) count"),
    ("apply/req1-range/counted?", "a ChunkedCons chain has no O(1) count"),
    // ── a direct variadic call conses instead of building an ArraySeq ─────
    // Clojure (ArraySeq) and ClojureScript (IndexedSeq) agree: a rest arg is
    // seq? true, counted? true, and list? FALSE. microclj conses, so only
    // `list?` differs — every other predicate and value already agrees.
    // (`apply` no longer comes through here; it shares the applied seq.)
    ("direct/1-extra/list?", "rest arg is a cons list, not an ArraySeq"),
    ("direct/3-extra/list?", "rest arg is a cons list, not an ArraySeq"),
    ("req1/direct/list?", "rest arg is a cons list, not an ArraySeq"),
    ("req2/direct/list?", "rest arg is a cons list, not an ArraySeq"),
    // ── counted? is derived from list?, so a Cons chain reads as counted ──
    // `(apply f 1 2 coll)` conses the leftover leading args onto coll, as
    // Clojure does. Clojure and ClojureScript disagree about whether the
    // result is `list?` (a host artifact — see the disagreement report from
    // refresh.sh) but AGREE it is not `counted?`. Here `counted?` is defined
    // as `(or (vector? x) (map? x) (set? x) (list? x) (string? x))` and this
    // dialect has ONE cons type, so list? true drags counted? true with it.
    ("apply/leading+seq/counted?", "counted? is derived from list?; one cons type"),
];

fn parse(s: &str) -> Vec<(String, String)> {
    s.lines()
        .filter_map(|l| l.split_once('\t'))
        .map(|(a, b)| (a.to_string(), b.to_string()))
        .collect()
}

fn run_probe(jit: bool) -> Vec<(String, String)> {
    let root = concat!(env!("CARGO_MANIFEST_DIR"), "/..");
    let probe = format!("{}/clojure-stub/tests/seq_oracle/probe.clj", root);
    let mut cmd = Command::new(env!("CARGO_BIN_EXE_microclj"));
    if jit {
        cmd.arg("--jit");
    }
    let out = cmd.arg(&probe).current_dir(root).output().expect("run microclj");
    assert!(out.status.success(), "probe failed:\n{}", String::from_utf8_lossy(&out.stderr));
    parse(&String::from_utf8_lossy(&out.stdout))
}

fn check(jit: bool) {
    let clj = parse(include_str!("seq_oracle/expected.txt"));
    let cljs: HashMap<String, String> =
        parse(include_str!("seq_oracle/expected_cljs.txt")).into_iter().collect();
    let got = run_probe(jit);
    assert_eq!(clj.len(), got.len(), "probe emitted a different number of lines than expected.txt");

    let mut regressions = Vec::new();
    let mut fixed = Vec::new();
    let mut third_answers = Vec::new();

    for ((label, want), (glabel, gval)) in clj.iter().zip(got.iter()) {
        assert_eq!(label, glabel, "probe lines out of order");
        let cljs_val = cljs.get(label);
        let agreed = cljs_val.is_none_or(|c| c == want);
        let known = KNOWN_DIVERGENCES.iter().find(|(l, _)| l == label);

        if !agreed {
            // Host artifact: Clojure and ClojureScript disagree, so neither is
            // "the language". microclj may pick a side — but not a third one.
            let c = cljs_val.expect("checked above");
            if gval != want && gval != c {
                third_answers.push(format!(
                    "  {label}\n    clojure:  {want}\n    cljs:     {c}\n    microclj: {gval}"
                ));
            }
            continue;
        }
        match (known, want == gval) {
            (Some((l, why)), true) => fixed.push(format!("  {l}  (was: {why})")),
            (None, false) => regressions
                .push(format!("  {label}\n    clojure+cljs: {want}\n    microclj:     {gval}")),
            _ => {}
        }
    }

    assert!(
        regressions.is_empty(),
        "microclj diverges on {} line(s) where Clojure and ClojureScript AGREE \
         (i.e. real semantics, not a host artifact):\n{}",
        regressions.len(),
        regressions.join("\n")
    );
    assert!(
        third_answers.is_empty(),
        "on {} host-artifact line(s) microclj invented a THIRD answer. Clojure and \
         ClojureScript disagree there, so either is defensible — but inventing a new \
         answer is not:\n{}",
        third_answers.len(),
        third_answers.join("\n")
    );
    assert!(
        fixed.is_empty(),
        "{} line(s) now MATCH but are still listed in KNOWN_DIVERGENCES.\n\
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
