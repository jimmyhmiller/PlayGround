//! **Instrument 1: the static fidelity audit.**
//!
//! The CFG analysis view is built by [`crate::lower`] on top of the reversible
//! base IR. It is allowed to be *incomplete* (bail loudly) but it must never be
//! *silently lossy* (represent a construct in a way that drops semantically
//! relevant data, with no error). See `CFG_FIDELITY.md`.
//!
//! This module turns "is the CFG complete?" into a checkable classification.
//! Every base-IR op-kind that can reach the lowering falls into exactly one
//! bucket:
//!
//! - [`Fidelity::Faithful`] — recognized and represented without semantic loss.
//! - [`Fidelity::HardError`] — not recognized; `lower_op`'s catch-all turns it
//!   into a hard `Err` and the whole function bails. *Acceptable* (loud).
//! - [`Fidelity::KnownLossy`] — recognized but drops data. **The bug registry.**
//!   Each entry carries a one-line justification. Closing a loss = lowering it
//!   faithfully + deleting its registry row.
//! - [`Fidelity::StructuralWrapper`] — program scaffolding (`jsir.file` /
//!   `jsir.program`) consumed by `lower_function`, not `lower_op`.
//!
//! What this instrument catches: a NEW op-kind the front-end starts emitting
//! (coverage ratchet, via the snapshot test), and any drift in the declared
//! loss set. What it does NOT catch: a construct that is *recognized* but
//! silently mis-lowered in a way we did not declare. That is the job of the
//! behavioral differential (Instrument 2, [`crate::interp`] + the oracle/fuzzer
//! tests), which runs the CFG and diffs it against Node.

use std::collections::BTreeSet;

use crate::lower::HANDLED_OP_KINDS;

/// Where a base-IR op-kind sits on the fidelity contract.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Fidelity {
    /// Recognized and represented without semantic loss.
    Faithful,
    /// Not recognized; lowering bails with a hard `Err`. Loud, acceptable.
    HardError,
    /// Recognized but drops semantically relevant data. Carries a justification.
    KnownLossy(&'static str),
    /// Present in the IR tree but intentionally **not lowered into the analyzed
    /// function**, and faithful precisely because it carries no data flow into
    /// it (e.g. a top-level `import` whose binding the function reads as a
    /// `Global`, or a directive whose only effect, the memo opt-out, is decided
    /// upstream in `detect.rs`). Verified to lower without bailing. Carries a
    /// justification.
    Ignored(&'static str),
    /// Program scaffolding handled by `lower_function`, not `lower_op`.
    StructuralWrapper,
}

impl Fidelity {
    /// A stable one-token tag for snapshotting.
    pub fn tag(&self) -> &'static str {
        match self {
            Fidelity::Faithful => "faithful",
            Fidelity::HardError => "hard-error",
            Fidelity::KnownLossy(_) => "KNOWN-LOSSY",
            Fidelity::Ignored(_) => "ignored",
            Fidelity::StructuralWrapper => "wrapper",
        }
    }
}

/// **The known-lossy registry.** Op-kinds that `lower.rs` recognizes but whose
/// representation drops semantically relevant data. Every entry is a reviewed,
/// justified loss; nothing silent slips past the audit. Removing a loss means
/// lowering the construct faithfully and deleting its row here.
// EMPTY. Every base-IR op-kind is now either faithfully lowered (and behaviorally
// validated by the differential / fuzzer, `tests/differential.rs`) or a loud
// hard-error. The losses that used to live here were all closed:
//   - closure bodies (arrow / function expression / object method) are lowered
//     into nested CFGs and execute correctly, including captured-object mutation
//     and every parameter shape (positional, destructured, arrow params);
//   - spread (`[...a]`, `{...o}`, `f(...a)`) is splatted faithfully by the
//     interpreter via `Cfg::spread_positions`.
//
// NOTE on scope: this registry tracks **lowering fidelity** (does the CFG drop
// semantically relevant data — proven by behavioral equivalence to Node). It is
// NOT the React-parity gate. The React analyses still consume the coarse
// `MakeArray` + property-granular-capture view of a closure for dependency/scope
// inference rather than the nested body's per-capture effects. That is *sound*
// (never wrong, only imprecise vs React) and is a precision item tracked by the
// corpus gate (`examples/corpus.rs` / PARITY.md), not a lowering loss.
pub const KNOWN_LOSSY: &[(&str, &str)] = &[];

/// Op-kinds consumed by `lower_function` directly (the program/file wrappers),
/// not routed through `lower_op`'s per-op match.
pub const STRUCTURAL_WRAPPERS: &[&str] = &["jsir.file", "jsir.program"];

/// Op-kinds that appear in the IR but are intentionally not lowered into the
/// analyzed function and lose nothing by it (see [`Fidelity::Ignored`]). Each is
/// verified by the empirical bail-check (`tests/fidelity.rs`) to lower without
/// error, so this is a *reviewed* set, not an assumption.
pub const IGNORED_FAITHFUL: &[(&str, &str)] = &[
    (
        "jsir.import_declaration",
        "top-level import; not part of the function body. The imported binding is \
         read inside the function as a `Global`, so no data flow is dropped.",
    ),
    (
        "jsir.directive",
        "a directive prologue (`'use strict'` / `'use no memo'`) carries no data \
         flow; the only semantic one, the memo opt-out, is honored in detect.rs.",
    ),
];

fn lossy_reason(kind: &str) -> Option<&'static str> {
    KNOWN_LOSSY.iter().find(|(k, _)| *k == kind).map(|(_, r)| *r)
}

fn ignored_reason(kind: &str) -> Option<&'static str> {
    IGNORED_FAITHFUL.iter().find(|(k, _)| *k == kind).map(|(_, r)| *r)
}

/// Classify a single op-kind given the set of op-kinds the lowering recognizes.
pub fn classify(kind: &str) -> Fidelity {
    if STRUCTURAL_WRAPPERS.contains(&kind) {
        return Fidelity::StructuralWrapper;
    }
    if let Some(reason) = lossy_reason(kind) {
        return Fidelity::KnownLossy(reason);
    }
    if let Some(reason) = ignored_reason(kind) {
        return Fidelity::Ignored(reason);
    }
    if HANDLED_OP_KINDS.contains(&kind) {
        Fidelity::Faithful
    } else {
        Fidelity::HardError
    }
}

/// The result of auditing a set of reachable op-kinds.
pub struct Audit {
    /// `(op-kind, classification)`, sorted by op-kind.
    pub rows: Vec<(String, Fidelity)>,
    /// Internal-consistency violations that should fail the audit test.
    pub violations: Vec<String>,
}

impl Audit {
    /// Count of op-kinds in each bucket.
    pub fn counts(&self) -> (usize, usize, usize, usize) {
        let mut f = 0;
        let mut h = 0;
        let mut l = 0;
        let mut w = 0;
        for (_, c) in &self.rows {
            match c {
                Fidelity::Faithful => f += 1,
                Fidelity::HardError => h += 1,
                Fidelity::KnownLossy(_) => l += 1,
                // Ignored counts as faithful for the headline (it loses nothing).
                Fidelity::Ignored(_) => f += 1,
                Fidelity::StructuralWrapper => w += 1,
            }
        }
        (f, h, l, w)
    }

    /// A stable, line-per-op snapshot of the classification, for the ratchet test.
    pub fn snapshot(&self) -> String {
        let mut s = String::new();
        for (kind, c) in &self.rows {
            s.push_str(kind);
            s.push(' ');
            s.push_str(c.tag());
            s.push('\n');
        }
        s
    }
}

/// Audit a set of op-kinds reachable in the base IR (e.g. collected across the
/// fixture corpus). Produces a per-op classification plus any internal
/// consistency violations:
///
/// - a `KNOWN_LOSSY` entry that the lowering no longer recognizes (so it would
///   really be a `HardError` now) — the registry row is stale and must be
///   removed (the loss was closed or the arm deleted);
/// - a `STRUCTURAL_WRAPPER` that *is* in `HANDLED_OP_KINDS` (double-handled);
/// - a registered loss / wrapper that never actually appears in the corpus
///   (dead registry entry worth pruning) — reported as a violation so the
///   registry stays honest.
pub fn audit(reachable: &BTreeSet<String>) -> Audit {
    let mut rows: Vec<(String, Fidelity)> = Vec::new();
    let mut violations: Vec<String> = Vec::new();

    for kind in reachable {
        rows.push((kind.clone(), classify(kind)));
    }
    rows.sort_by(|a, b| a.0.cmp(&b.0));

    // KNOWN_LOSSY rows must be recognized by the lowering, else they are stale.
    for (kind, _) in KNOWN_LOSSY {
        if !HANDLED_OP_KINDS.contains(kind) {
            violations.push(format!(
                "KNOWN_LOSSY entry `{kind}` is not in HANDLED_OP_KINDS: it would now \
                 hard-error, so the lowering arm was removed. Delete the registry row."
            ));
        }
        if !reachable.contains(*kind) {
            violations.push(format!(
                "KNOWN_LOSSY entry `{kind}` never appears in the corpus: prune the dead \
                 registry row (or point the audit at a corpus that exercises it)."
            ));
        }
    }
    for kind in STRUCTURAL_WRAPPERS {
        if HANDLED_OP_KINDS.contains(kind) {
            violations.push(format!(
                "STRUCTURAL_WRAPPER `{kind}` is also in HANDLED_OP_KINDS (double-handled)."
            ));
        }
    }
    // IGNORED_FAITHFUL must NOT be in HANDLED (if it were, it is really lowered,
    // not ignored) and must actually appear, else the row is dead.
    for (kind, _) in IGNORED_FAITHFUL {
        if HANDLED_OP_KINDS.contains(kind) {
            violations.push(format!(
                "IGNORED_FAITHFUL entry `{kind}` is in HANDLED_OP_KINDS: it is lowered, \
                 not ignored. Reclassify it (faithful) and remove the registry row."
            ));
        }
        if !reachable.contains(*kind) {
            violations.push(format!(
                "IGNORED_FAITHFUL entry `{kind}` never appears in the corpus: prune the \
                 dead registry row."
            ));
        }
    }

    Audit { rows, violations }
}
