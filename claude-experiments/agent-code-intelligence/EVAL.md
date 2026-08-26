# Evaluating Quarry vs. LSP (vs. grep)

A design for an agent-outcome evaluation: same model, same harness, same
tasks — only the code-intelligence tooling changes. The question is not
"which protocol is prettier" but *does an agent using it produce better,
cheaper, safer outcomes?*

## 1. Arms

| Arm | Tooling | Purpose |
|-----|---------|---------|
| **grep** | rg + Read + Edit | The actual status quo baseline |
| **LSP (steelman)** | pyright-langserver behind a well-built bridge: warm server, document sync handled by the bridge, definition / references / hover / pull-diagnostics / rename exposed as tools addressed by file:line:col | Beating a badly-configured bridge proves nothing; the LSP arm must be the best bridge a motivated engineer would build |
| **quarry** | the prototype CLI | The candidate |

Each arm runs in two modes: **tool-only** (file reads disallowed except to
apply edits — isolates the interface) and **tool+reads** (realistic; also
yields the abandonment metric below). Every arm gets an equally polished
one-page usage doc — tool-doc quality is a confound, so it is equalized,
not left to chance.

## 2. Controlling the priors confound

Models carry heavy training priors: deep fluency with grep, moderate
familiarity with LSP concepts, zero exposure to quarry — and often
memorized knowledge of well-known open-source repos. Both leak into any
naive eval.

**Corpus: alpha-renamed real codebases.** Take 3–4 real Python repos
(small/medium/large; e.g. a requests-sized library, a Flask-sized
framework slice, one CLI tool), mechanically rename every identifier,
string-key, and filename stem to neutral generated names while preserving
structure, behavior, and a passing test suite. This kills "I already know
this repo" and forces genuine tool use in every arm. Keep one un-renamed
repo as a familiarity-condition contrast.

## 3. Task families (mechanical ground truth)

Ground truth is generated offline — a semantic index (pyright/SCIP) for
the symbol graph, plus actually performing each change and running the
checker and tests. Tasks are then sampled programmatically from the answer
key, so the suite scales to hundreds of items with no hand labeling.

1. **Navigation.** "List all true call sites of X", "give the call chain
   from A to B", "what implements Y". Scored as precision/recall against
   the ground-truth graph.
2. **Dead-code judgment.** Planted truly-dead functions *and* traps:
   `getattr`-string references, registry/config-key coupling, re-exports.
   Scored on the keep/delete decision; false deletes weighted as
   catastrophic. (This is the trust claim, made measurable.)
3. **Refactors.** Renames with collateral (same-named symbols on other
   types, YAML keys, doc mentions), signature changes with call-site
   updates. Scored: tests pass + zero residual old-name + collateral
   handled; broken intermediate states counted.
4. **Planted type bugs.** Code that parses, where every name exists, but
   that breaks at runtime (str-into-int assignments, attribute typos on
   typed objects). Scored found/fixed, plus calls-to-detection.
5. **Impact questions.** "What breaks if this signature changes?" — ground
   truth from actually making the change and running checker+tests.
6. **Staleness gauntlet.** Tasks that start during index warm-up, or where
   the harness mutates files mid-task. Measures wrong-confident
   conclusions per arm: LSP's silent-empty-during-warm-up failure mode vs.
   quarry's `INDEX_BUILDING`/`partial` contract.

## 4. Metrics

- **Task success** (per family's scorer) and **catastrophic actions**
  (wrong delete, corrupted file, broken tests left behind).
- **Cost**: total tokens (in+out), tool calls, wall-clock per solved task.
- **Calibration**: every agent assertion of the form "there are no
  references / this is safe / the project is clean" is checked against
  ground truth. A tool that makes agents confidently wrong is worse than
  one that makes them slow, and this metric is where that shows.
- **Tool-abandonment rate**: in tool+reads mode, the fraction of
  information needs the agent satisfied by reading/searching files rather
  than using the assigned tool. This turns the folklore claim "agents
  route around LSP" into a number — and applies the same skepticism to
  quarry.
- **Time/calls to first productive use** — a proxy for interface
  learnability given zero training prior.

## 5. Protocol and statistics

- k = 3–5 seeded runs per task per arm; identical model, temperature,
  system scaffold, and token ceiling.
- Paired per-task comparisons (each task seen by every arm), bootstrap
  CIs; report per-family, not just aggregate.
- Pre-registered hypotheses, e.g.:
  - H1: quarry ≥ steelman-LSP on refactor success at ≤ 0.75× tokens.
  - H2: quarry reduces false-delete rate on trap tasks vs. both arms.
  - H3: abandonment(quarry) < abandonment(LSP).
  - H4: on the staleness gauntlet, wrong-confident conclusions:
    quarry ≈ 0 < LSP.
- **Falsifiers, stated up front.** If steelman-LSP matches quarry on
  success at similar cost, the protocol's gains don't justify a new
  standard. If agents abandon quarry at LSP-like rates, the thesis
  "agents skip LSP because of its interface" is wrong — the story is
  tool-use habits, not protocol design. Either outcome is a real result.

## 6. Build plan

| Item | Effort | Notes |
|------|--------|-------|
| LSP bridge (steelman arm) | ~1 day | pyright-langserver + thin client; multilspy shortcuts it |
| Alpha-renamer | ~½ day | libcst/ast rename preserving tests |
| Ground-truth generator + task sampler | ~1 day | offline index + run-the-change |
| Harness (arm config, scorers, run ledger) | ~1 day | subagent-per-run; scorers are scripts, not judges |

Full run (3 repos × ~40 tasks × 3 arms × 2 modes × 4 seeds ≈ 3k sessions)
is CI-scale. A **pilot** — one renamed repo, 10 tasks across families 2–4,
quarry vs. LSP, k=2 — is a few hundred dollars of tokens and enough to
shake out the harness and get a directional read before paying for the
full matrix.
