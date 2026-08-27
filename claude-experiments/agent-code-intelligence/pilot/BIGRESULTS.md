# Scaled eval results: granite corpus (quarry vs. steelman LSP)

32 runs — 8 tasks × 2 arms × k=2 — over the generated `granite` codebase
(150 files, ~15k LOC, 105 same-named `process` methods, config+getattr
dispatch), under hard conditions: minimal command-list docs with no
blind-spot warnings, a deliberately weak visible test suite, and a hidden
scoring suite as the catastrophe detector. Raw rows:
`big-runs/results.csv` (scratch); harness: `bigcorpus.py` + `bigtasks.py`.

## Headline

| | LSP (pyright bridge) | quarry |
|---|---|---|
| Success | 16/16 | 16/16 |
| Catastrophes (hidden suite) | 0 | 0 |
| Mean tokens per run | 51.1k | 52.4k |
| Median assigned-tool calls | ~11 | ~14 |
| **Worst-case calls** | **36** | **859** |
| Wall-clock, worst family | 166 s (g8) | 396 s (g8) |

Success parity survives even at 15k LOC with no hand-holding — but this
time the eval produced real, interpretable separation, and it is
**bidirectional by task family**, not a blanket win for either side.

## Where each arm paid

**Receiver-bound navigation (g5: call sites of one `process` among 105
same-named methods).** LSP: exact answer, 10 calls, ~37k tokens, 45 s —
pyright's references are receiver-resolved natively. Quarry: exact answer
both runs, but via brute force — `refs` returned 67 name-matched
candidates and the agent ran compiler `type-of` probes per candidate: 72
and 140 calls, ~56–66k tokens, 156–242 s. Same correctness, 1.5–1.8×
tokens, 3–5× latency. This is the quantified price of quarry's declared
L1-syntactic receiver gap — the agent bought L3 receiver resolution
retail, one probe at a time.

**Deep reachability (g8: 6-hop chain + unreachable decoy).** Both arms
correct on both verdicts including the decoy. LSP: 19–29 calls, 76–81k
tokens. Quarry: one run went 23 calls / 68k, the other **859 calls /
95k** — it scripted a hop-by-hop BFS shelling out per edge rather than
using `graph --depth`, then verified hops by reading source. The
capability exists in the protocol; the agent didn't compose it. Worst-case
behavior under a missing-or-untrusted abstraction is itself a finding:
tails, not medians, are where interface gaps show.

**Dead-code judgment (g1–g4, incl. 2 config-dispatch traps).** All 16
correct, zero catastrophes — and here the cost gradient reverses: quarry
averaged ~44k tokens/run vs LSP's ~53k, and on the traps quarry runs were
both cheaper (9–10 calls) and evidence-complete in-band
(`textual_mentions` pointing at the exact maintenance.cfg line). LSP runs
got there too, but by leaving the tool to read the config once pyright
returned定义-only references.

**Type-bug hunt (g6) and hooks audit (g7).** Parity on success; quarry
slightly cheaper on g6 (one `check` sweep = all five bugs), slightly
noisier on g7.

## Two integrity findings (the eval auditing itself)

- **The g7 key was wrong and the agents caught it.** The generator
  sampled "also statically called" hooks from *eligible* functions
  without ever inserting the calls; all four had zero static call sites.
  Three independent runs across both arms converged on the empty set
  against the key; verification against the edge list confirmed the
  agents were right. The key was corrected (true answer: `[]`) and the
  flaw noted: converging disagreement between independent runs and a key
  is a key bug until proven otherwise.
- **The traps leaked.** Trap-dead functions carried the docstring
  "Maintenance hook (see profile configuration)" — a label that hands
  any agent the answer once it reads the function. The 8/8 trap
  correctness is therefore an upper bound, not a measurement. A rerun
  needs innocuous docstrings on traps (and on safe-dead functions,
  identical ones).

## Rematch after the L2 upgrade

The two tail-makers were then built into quarry (receiver inference with
a proven/unknown/excluded partition; `graph --edge path` reachability
with receiver-exact edges), the torn-snapshot risk was fixed with atomic
verify-reread loads (id-scored hammer: 325/325 responses consistent with
their claimed snapshot, zero torn views — the earlier "1 stale" was the
scorer's own race), and conformance grew to 50 checks including the new
obligations (SPEC A12/A13). Fresh agents then re-ran the two tasks
quarry had lost on cost:

| | quarry before | quarry after | LSP (reference) |
|---|---|---|---|
| g5 nav — tool calls | 72 / 140 | **3 / 4** | 10 |
| g5 nav — tokens / wall | 56–66k / 156–242 s | 40k / 34 s | 37k / 45 s |
| g8 reach — tool calls | 23 / 859 | **4 / 4** | 19 / 29 |
| g8 reach — tokens / wall | 68–96k / 298–396 s | 40k / 40 s | 76–81k / 155–166 s |

All four runs exact against the key (4/4 receiver-proven sites with 167
collisions excluded; the 6-hop chain plus a correct unreachable verdict
on the decoy). The measured gaps didn't narrow — they inverted: quarry
now answers both families in fewer calls and less wall-clock than the
steelman bridge, while keeping the disclosure channel that won the
judgment family. Which is the levels thesis (§7) demonstrated end to
end: the tails were never protocol properties, they were the price of
the missing L2 capabilities, and building those capabilities — with the
partition disclosed, per A13 — removes the tails without giving up
honesty.

## What this eval settles, and what it doesn't

Settled: at 15k LOC with hazard-free docs and weak tests, both interfaces
still get strong models to correct outcomes — but the *cost topology*
finally separated, and it matches each design's declared gaps exactly:
LSP pays where evidence lives outside the reference graph; quarry pays
where receiver identity or transitive closure must be assembled from
syntactic parts. Neither arm produced a single confidently-wrong action
across 112 total runs in the three evals — every remaining separation is
efficiency, latency, and tail behavior, not safety, for models of this
strength on corpora of this shape.

Not settled: whether parity holds when the traps don't self-describe;
weaker models under these hard conditions (the small-pilot Haiku result
suggests method degradation is where risk concentrates); and truly
adversarial repos (metaprogramming, re-exports, generated symbols) where
both tools' L-level obligations (SPEC §7.1) start binding. The protocol
conclusion stands sharpened: quarry's next capability investment should
be exactly the two tail-makers — index-backed receiver resolution and a
trusted transitive `graph` (its L2/L3 tier) — because that is where
agents currently spend 5–60× the calls buying them by hand.
