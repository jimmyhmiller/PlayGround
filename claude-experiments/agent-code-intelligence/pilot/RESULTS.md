# Pilot results: quarry vs. steelman LSP

40 agent runs — 10 tasks × 2 arms × k=2 — same model, same harness, same
rules, equalized one-page docs, fresh generated corpus (`mill/`) with
ground-truth-by-construction traps. Scored mechanically (`tasks.py score`);
raw per-run rows in `results.csv`.

## Headline numbers

| | LSP (basedpyright bridge) | quarry |
|---|---|---|
| Task success | **20/20** | **20/20** |
| Catastrophic actions (wrong delete, broken tests left) | 0 | 0 |
| Mean assigned-tool calls per run | 4.2 | 4.9 |
| Mean tokens per run | 40.8k | 41.7k |
| Runs that had to fall back to reading files | **6/20 (30%)** | **1/20 (5%)** |

Every task went 2/2 for both arms: dead-code judgment including both traps
(`getattr`-dispatched probe, settings-key-dispatched hook), the same-name
collision rename, the docs-inclusive rename, both latent type bugs, the
signature change, receiver-exact navigation, transitive impact analysis,
and the call chain.

## What the pilot actually found

**1. On success rate at this scale, the arms are indistinguishable — and
that is a real result.** EVAL.md pre-registered this as a falsifier
condition: a steelman LSP bridge matching quarry on success at similar
cost means the protocol's gains don't show up in outcomes *under these
conditions*. The conditions matter: a 300-line corpus any agent can hold
in its head, trap-heavy tasks, agents handed a polished usage doc naming
each tool's blind spots, and a strong model. The pilot cannot detect
outcome differences here; it was sized to shake out the harness and find
where the interfaces diverge behaviorally.

**2. Where they diverge is tool coverage — who has to leave the tool.**
All six LSP fallbacks are the same shape: information that exists outside
the reference graph. Enclosing functions of reference hits (t10), the
`getattr` registry trap (t1 ×2), the settings-string dispatch (t2, t3),
the README mention (t5). The steelman bridge is compiler-exact inside the
graph and structurally silent outside it, so the agent must notice the
silence and go read files — which every run did, correctly, *because the
task design and doc warned of exactly these hazards*. Quarry's one
fallback was the mirror image: receiver disambiguation (its syntactic
refs can't tell `eng.process` from `writer.process` by themselves) after
a `type-of` crash — a genuine bug the pilot surfaced (probe insertion
after a block-opener line emptied the block; fixed, conformance 46/46).

**3. Each arm's home turf showed as call-efficiency, not correctness.**
LSP's receiver-exact `references` answered t8 in 2 calls both times, and
its semantic rename did t4 in 4; quarry took 4–7 calls on those, composing
refs + judgment. Conversely quarry answered the trap tasks fully in-band
(`textual_mentions` and `impact:delete` carried the getattr/config
evidence), while LSP runs did the same work by hand outside the tool.

**4. The safety margins differ in kind.** Both arms went 0-for-0 on
catastrophes, but by different routes: quarry runs cited the tool's own
disclosures as their evidence ("0 static refs BUT a settings.yml textual
mention"); LSP runs cited their own out-of-band file reading. The first
is machine-checkable evidence an unattended agent gets by default; the
second depended on the agent knowing to look. This pilot's agents always
knew to look — the full eval's no-doc and alpha-renamed conditions exist
to test whether that holds when they aren't told.

## The Haiku replication (same 40 runs, smaller model)

The obvious ceiling-effect suspect was model strength, so the identical
trial was re-run with Claude Haiku 4.5 as the agent (`results-haiku.csv`).
One scorer regex was loosened after the fact (it rejected functionally
correct positional `True`/`False` arguments at t7 call sites — a scorer
bug, not an agent failure).

| | LSP (Fable) | quarry (Fable) | LSP (Haiku) | quarry (Haiku) |
|---|---|---|---|---|
| Success | 20/20 | 20/20 | 20/20 | 20/20 |
| Catastrophes | 0 | 0 | 0 | 0 |
| Mean tool calls | 4.2 | 4.9 | 2.5 | 3.5 |
| Mean tokens | 40.8k | 41.7k | 34.6k | 36.3k |
| Self-reported fallbacks | 6/20 | 1/20 | 3/20 | 4/20 |

What the replication actually showed:

- **The ceiling holds even for a small model.** On a context-sized corpus
  with blind-spot-naming docs and a cheap, trustworthy test suite, Haiku
  goes 40/40 too. Model strength was not the binding constraint; the
  *conditions* are. This strengthens the case that separation requires
  the full eval's harder conditions, not a weaker model.
- **Method changed even though outcomes didn't.** Haiku runs used ~40%
  fewer tool calls and visibly less analysis: several Haiku LSP runs
  resolved the dead-code traps *empirically* — delete the function, watch
  the tests fail, restore it — where every Fable run reasoned it out
  first (and Fable LSP runs explicitly checked the registry for dynamic
  dispatch; Haiku LSP t1 runs skipped that check and were saved by the
  function actually being dead). Delete-and-see lands safely only because
  this corpus has a fast suite that covers the dynamic paths; on a repo
  with weaker tests, that method converts directly into the catastrophic
  false-delete the trap was designed to catch. The safety margin under
  Haiku came from the test suite, not from the code-intelligence tool —
  for either arm.
- **Self-reported fallback is unreliable at the smaller tier.** The
  Fable pattern (LSP 6 vs quarry 1, all six being the same
  outside-the-graph shape) inverted into noise under Haiku (3 vs 4, with
  inconsistent labeling of what counts as a fallback). The metric is
  real but must be measured from transcripts, not self-report — as the
  full-eval design already recommends.

## Limits of this pilot (read before quoting the table)

- One tiny corpus a model can hold in context; no alpha-renamed repo, so
  reading files was always cheap and reliable.
- The docs explicitly warned each arm of its blind spots — a best-case
  deployment. The interesting condition (no doc, or misleadingly generic
  doc) was not run.
- The staleness gauntlet (family 6 in EVAL.md) was not run — it is the
  family where LSP's silent-empty behavior is predicted to produce
  wrong-confident answers, and nothing here tests that prediction.
- No grep baseline arm, k=2, one model: directional only, no statistics.

## What this recommends for the full eval

Fallback rate is a live, cheap metric — keep it, but instrument it from
transcripts rather than self-report. Add the no-doc condition and the
staleness family before trusting any "arms are equal" conclusion. Scale
the corpus past context-holding size (alpha-renamed real repos), where
"go read the files" stops being free and the 30%-vs-5% coverage gap is
predicted to turn into an outcome gap — that prediction is now the
specific hypothesis the full eval should test.
