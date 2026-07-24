# bat — browser auto tests

Browser e2e tests that are **never flaky because of timing** and **fully
explainable when they fail**. A small DSL + runner on top of Playwright.

```
flow "add item to cart"

given seed "catalog-basic"
given user "shopper" signed-in

go /products
  expect heading "Products"

click button "Add to cart" in listitem "Blue Widget"
  expect request POST /api/cart ok
  expect appear status "Added to cart"
  expect text "1" in testid "cart-count"
```

## The two axioms

1. **The DSL can only express state transitions.** Every step is an action plus
   the observable effects it must cause. There is no `wait`, no `sleep`, no
   timeout parameter — the tokens don't exist. The runtime waits on *events*
   (a named response, a navigation, an element appearing), never on clocks.
2. **The DSL is data, not code.** A flow parses to a flat JSON plan. That's what
   buys atomic replay, complete traces, and agent-writability.

Key mechanics:

- **Act-and-observe is one transaction.** Observers (request matchers,
  appear/gone watchers) are armed *before* the action fires, so a toast that
  lives 150ms is caught deterministically. Transient watchers use bat's own
  injected MutationObserver + one-shot checks — Playwright's poller can miss
  sub-200ms transients (see `scripts/toast-debug.ts` for the receipts).
- **Settlement** converges on app quiescence: declared request expectations
  matched, tracked traffic drained, navigation landed, task queue quiet. The
  per-step budget is runner config, never test text; blowing it produces a
  stuck-state report, not `TimeoutError`.
- **Targets are semantic** (`button "Save"`, `row "Blue Widget"`); raw CSS and
  XPath don't exist; `testid` is the sole escape hatch. Ambiguity is a hard
  error listing every match — bat never picks "the first one".
- **Worlds are data.** Seeds merge commutatively with conflict detection and
  referential closure; the world is rebuilt from empty per flow; adapters climb
  a capability ladder (L0 trust-me → L4 time-travel) where every operator you
  provide buys a stronger *checked* guarantee. See `docs/WORLD.md`.
- **Failures are stories**: what was expected, what was observed, the network
  during the step (with response completion order), the page's semantic tree,
  and the exact replay command.
- **Every failure gets a causal explanation, never a bare timeout — and never
  a verdict.** Whether an observed state is "broken" depends on user
  expectations, which only the flow's author knows. So bat reports evidence:
  what was expected vs what the settled page showed; what the page did during
  the step (requests, completion order); whether identical reruns reproduce it
  ("fully reproducible: stable behavior, not a timing variation") or not ("NOT
  deterministic," with the outcome cross-tabbed against response completion
  order) — and where interpretation is required, both readings, so deciding
  whether the app or the expectation should change takes a minute, not an
  afternoon. Near-miss target names get "closest present: …"; chaos-induced
  failures are separated by a clean rerun.
- **Simulated bad conditions** (seeded latency / failure injection via
  `--latency`, `--fail-rate`, `--seed`) are recorded in every trace and
  attributed per request — a chaos-induced failure can never masquerade as a
  real one. Latency alone never fails a flow; that invariant is executable.

## The claim, property-tested

Flakiness is an outcome that depends on timing the test did not mean to
encode. So the claim is a property, and it's in the suite
(`src/timing.property.test.ts`):

> ∀ timing profiles P (server latency 0ms–3s, toast lifetimes 60–400ms):
> bat(buy.flow, P) = pass — while the same journey written as raw Playwright
> with explicit 1.5s tolerances has an outcome that is a function of P.

The world algebra's laws (commutativity, associativity, idempotence, canonical
fingerprints, symmetric conflicts) and the DSL round-trip
(`parse(format(step)) ≡ step`) are verified with fast-check across ~1,500
generated cases per run.

## CI

`bat run --junit` writes `.bat/junit.xml`; on GitHub Actions, failures emit
`::error` annotations on the flow file/line and a job-summary report
automatically. Failing steps capture screenshots into the run directory
(config `screenshots`: `on-failure` default, `steps`, `off`).

## Inspecting runs: `bat ui`

`bat ui --config <app>` serves a local viewer over `.bat/runs`: every run's
steps with per-effect verdicts, the network table (completion order, injected
conditions, streaming), page-error attribution, failure screenshots, the
semantic tree, the "why this failed" explanation — and a button to replay the
failing step (headed) right from the report.

## Try it

```sh
npm install && npm run build

# unit + e2e self-tests (spins up the fixture shop with random API latency)
npm test

# the anti-flake stress harness: all flows, N times, random latency every run
npx tsx scripts/gauntlet.ts 15

# invariant: injected latency alone must never fail a flow
npx tsx scripts/chaos-gauntlet.ts 10 42

# run a genuinely racy page (fire-and-forget refetch); on failure the report
# ends with 'why this failed:' — rerun evidence + completion-order cross-tab
npx tsx scripts/debug-once.ts fixtures/shop/e2e/flaky/flaky-cart.flow

# CLI against the fixture shop
npx tsx fixtures/shop/serve.ts &        # starts the app on :4173 with BAT_TEST=1
npx tsx src/cli.ts run --config fixtures/shop
npx tsx src/cli.ts doctor --config fixtures/shop
npx tsx src/cli.ts inspect http://localhost:4173
npx tsx src/cli.ts replay fixtures/shop/e2e/flows/buy.flow:4 --config fixtures/shop
```

## Layout

- `docs/SPEC.md` — grammar + execution semantics (the contract)
- `docs/WORLD.md` — the world algebra + adapter capability ladder
- `src/dsl/` — parser + IR
- `src/world/` — seed algebra, adapter, verification
- `src/runner/` — targets, settlement engine, transient watchers, traces, replay
- `src/server/` — `createWorldHandler` (mount the world adapter over HTTP; 404s unless `BAT_TEST=1`)
- `fixtures/shop/` — deliberately flaky demo app + L4 world adapter
- `scripts/gauntlet.ts` — the non-flakiness claim, executable
