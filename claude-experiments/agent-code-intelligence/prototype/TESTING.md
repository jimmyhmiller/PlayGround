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

## Round 3 — the "proven" bar

Three gaps stood between "validated argument" and "proven": honesty as a
checkable property, a real compiler at L3, and A3 made enumerable. All three
were built:

- **`conformance.py`** — a black-box protocol conformance suite: 46 named
  checks, one per envelope law and per §11 amendment, runnable against any
  implementation via `--cmd`. It provisions its own fixture (including a
  syntax-broken file, same-named symbols, and a `getattr` string reference),
  asserts exact error codes and payload fields, verifies disk-state truth for
  transforms, walks cursors to exhaustion, and checks determinism. The
  prototype passes 46/46.
- **L3 via mypy** — `check` at genuine compiler grade (syntax + name +
  attribute + type, `--check-untyped-defs`), with mypy-absent → degraded,
  timeout → partial, crash → structured error. `type-of` is implemented
  through the protocol's own overlay mechanism (a `reveal_type` probe in a
  hypothetical world) and refuses to answer when mypy errors at the probe
  line — closing its own would-be wrong-but-confident bug (mypy "reveals"
  `Any` for undefined names). `impact:patch` diffs mypy diagnostics between
  base and patched worlds.
- **SPEC §7.1** — the per-level obligations and disclosed-blind-spots table,
  making amendment A3 mechanical.
- **Scale** — against the `requests` library source (35 files, 909 symbols,
  14,357 occurrences): 214 ms cold index, ~100 ms warm queries, full-project
  mypy check in 4 s finding 130 genuine diagnostics; role+budget filtering
  cut 167 name-matches for `request` to the 20 real call sites.

## Round 4 — final adversarial validation

A fourth independent agent was given three claims to verify or refute, plus
license to attack.

- **Claim 1 (L3 catches what grep cannot): HOLDS.** Found the planted
  `config.timeout = "5"` type bug on its first `check` call; fixed it;
  verified clean; ground-truth tests pass.
- **Claim 2 (type-of answers real questions): HOLDS.** Three probes at
  compiler confidence — including flow-sensitive narrowing (`lhs` declared
  `Expr | int`, correctly reported as `Expr` post-assignment) — and the
  honesty probe (out-of-scope name) returned a structured error naming the
  mypy reason, not a fabricated type.
- **Claim 3 (honesty is checkable): HOLDS, with a real crack found.** The
  suite passed 41/41 and was judged "genuinely hard, not soft" — but the
  agent found one new wrong-but-confident answer: `refs` and `impact:delete`
  returned `complete` + zero for a symbol referenced via a
  `getattr("name")` string, a mention quarry's own L0 search could see. Its
  verdict: *"proven for read-only navigation, unproven for destructive
  authorization."*

That crack was closed the way the protocol demands: `refs` now reports a
`textual_mentions` count + sample + note on every response (the §7.1 L1
dynamic-access disclosure), `impact:delete` includes textual/string mentions
as possible breaks, and — decisively — the exploit scenario became
conformance checks (A3-2 now fails any implementation whose zero omits the
disclosure; A3-3 fails an impact:delete that reports a clean zero). The
agent's secondary finding (import-line rename skipped due to a column bug,
honestly surfaced as a conflict) was fixed via ast alias positions and is
now check A7-4. The impact family, ambiguity candidates, and brief — the
verbs the agent noted had no conformance coverage — are covered. Suite:
46/46; the validator's exact exploit replayed against the fixed build now
returns the mention attached to the zero and the delete verdict.

## Where things stand

Every finding from all four rounds is either fixed and regression-tested or
explicitly out of scope: no cache integrity checksum (hand-editing
`.quarry/cache` is out of threat model), `fnmatch` glob semantics (`*`
crosses `/`), `search:ast` and full semantic receiver resolution (reported
as missing by `capabilities`, per the honest-levels design), and the
remaining L3 blind spots §7.1 requires disclosing rather than closing
(reflection beyond string mentions, `exec`/`eval`, untyped `Any` regions).

The claim this exercise supports: **proven, in the specific sense that the
honesty contract is now a mechanically checked property** — 46 black-box
conformance checks that any implementation must pass, four independent
agents who acted on the envelopes and were never misled by a check-covered
verb, and every discovered way to be confidently wrong converted into a
conformance check that would catch its recurrence. What remains unproven is
breadth, not the contract: one language, one compiler backend, one repo of
real-world scale. The meta-result stands as the design lesson: trust comes
from the protocol making dishonesty a conformance failure, not from the
tool being smart — and that is now enforced by code, not discipline.
