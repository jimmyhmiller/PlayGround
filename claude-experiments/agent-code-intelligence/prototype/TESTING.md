# Agent evaluation of the quarry prototype

Three agents that had no part in writing the spec or the prototype were given
the tool cold — just `quarry.py`, `USAGE.md`, and a copy of the fixture
project — and asked to use it, judge it, and break it. Two rounds of fixes
came out of their reports; the protocol-level lessons were folded back into
`../SPEC.md` §11.

## Round 1 — task agent (rename refactor)

Completed the full `Config.timeout → request_timeout` rename including YAML,
README, and comment mentions, verified four independent ways (check clean,
zero residual textual matches, ref-count conservation old→new, tests pass).

What worked, in their words: `impact --change rename` producing "the exact
worry-list an agent needs" (the partition of will-be-rewritten vs.
yours-to-handle); envelope honesty notes "genuinely calibrating" — they ran a
verification sweep *because* a note told them name-matching over-includes;
ref-count conservation as a cheap correctness argument grep can't offer.

Bugs found (both fixed): definition occurrences indexed at the `def` keyword's
column so renaming a function's definition line silently no-opped; and that
failed rename returned `applied: true, files_changed: 1` with an empty diff —
"a tool whose entire pitch is 'you may act on the envelope' must never
misreport a write."

Verdict: prefers quarry over grep for rename-shaped work; vs. an LSP bridge,
LSP renames are better scoped semantically but say nothing about YAML/README
breakage, "which quarry's impact did and which was the genuinely novel value."

## Round 1 — trust agent (dead code, impact, comprehension)

Correctly identified `legacy_retry_delay` and `clamp` as transitively dead and
deleted them — explicitly *because* the envelope let them: `complete` +
`built_pct: 100` + empty `files_errored` + a confidence level whose error
direction is over-inclusion means zero really is zero, then closed the
declared blind spots (dynamic access, attribute errors) with a textual sweep
and a compiler-grade overlay check inside the same tool. Their summary of the
core claim: "an empty grep tells you nothing about why it's empty... quarry's
envelope pinned down all three failure modes explicitly."

Bugs found (all fixed): `graph --edge callees` silently ignored `depth` while
stamping `complete` — "proves the envelope's honesty is per-verb
implementation discipline, not a structural guarantee"; `search:symbol`
silently dropped the path filter; L0 coverage ambiguous about whether
non-indexed files were searched; `tests_affected` over-promising (direct-only).

## Round 2 — adversarial agent

23 numbered findings: 6 CRITICAL (wrong-but-confident answers or corruption),
8 MAJOR, 9 MINOR, plus an honesty scorecard and 11 protocol-level gaps. The
criticals, all fixed: occurrence clamping answering with a different symbol;
anchor filters falling back to "any symbol on the line"; garbage diffs
becoming silent no-op overlays (whose `check` then "verifies" the unpatched
world); overlay content leaking to disk via `edit.rename --mode apply`;
rename-to-keyword writing five syntax-broken files while reporting
`complete`; `check` on a YAML file promising `clean: true` having checked
nothing.

What survived attack, verbatim credit: anchor drift → clean `ANCHOR_MISMATCH`
with actual line text; content-hash snapshot cache honesty "genuinely solid";
unicode rename correct (byte offsets, right-to-left splicing); overlay path
escapes rejected; deterministic cursor walks with stable totals; per-sub-result
error isolation in batch; honest `UNSUPPORTED` with fallbacks.

Their protocol feedback (the "complete at stated confidence" loophole,
locator occurrence semantics, patch strictness, overlay lifecycle, transform
apply-gating, scope-narrowing disclosure, resolution disclosure, error
epistemics, budget/cursor edges, batch envelope shape, the stdin contract)
became SPEC.md §11 A1–A11 wholesale.

## Where things stand

All CRITICAL and MAJOR findings and the actionable MINORs are fixed and
regression-tested (`quarry.py`, two fix commits). Deliberately left as known
prototype limitations: no cache integrity checksum (hand-editing
`.quarry/cache` is out of threat model), `fnmatch` glob semantics (`*`
crosses `/`), and everything `capabilities` already reports as missing
(`type-of`, `search:ast`, semantic receiver resolution — the L3 tier by
design).

The meta-result: the envelope discipline worked as designed when implemented,
and every failure the agents caught was a place the implementation (or the
spec) left room to be confidently wrong. The evaluators independently
converged on the same conclusion the spec bet on — trust comes from the
protocol making dishonesty a conformance failure, not from the tool being
smart.
