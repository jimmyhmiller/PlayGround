# bat — browser auto tests

A DSL and runner for browser e2e tests that are **never flaky because of timing**
and **fully explainable when they fail**. Built on Playwright.

## The two axioms

1. **The DSL can only express state transitions.** The atomic unit is a *step*:
   an action plus the observable effects it must cause. There is no way to
   express duration — no `wait`, no `sleep`, no timeout parameter, no retry
   count. The tokens do not exist. Waiting is the runtime's job, and the
   runtime waits on *events* (a named network response, a router transition,
   an element appearing), never on clocks.

2. **The DSL is data, not code.** No conditionals, no loops, no closures. A
   flow file parses to a flat JSON plan (the IR). Steps are serializable
   records, which is what buys atomic replay, perfect traces, and
   agent-writability.

## Flow grammar

Line-oriented. Indentation groups effect lines under their action line.
One action plus its effect block is one atomic step. Comments start with `#`.

```
flow "add item to cart"

given seed "catalog-basic"
given seed "discounts"
given patch products "blue-widget" stock 0
given user "shopper" signed-in
given clock 2026-07-22T10:00:00Z

go /products
  expect heading "Products"
  expect count listitem 12 in list "product-list"

click button "Add to cart" in listitem "Blue Widget"
  expect request POST /api/cart ok
  expect text "1" in testid "cart-count"

click link "Cart"
  expect url /cart
  expect row "Blue Widget" in table "cart-items"
  let price = text in testid "line-total" of row "Blue Widget"

click button "Checkout"
  expect url /checkout
  expect text $price in region "order-summary"
```

### Givens

| form | meaning |
|---|---|
| `given seed "<name>"` | merge the named seed into the flow's world description |
| `given patch <type> "<key>" <field> <value>` | override one field of one fact, applied after merge (ordered, explicit) |
| `given user "<key>" signed-in` | mint a real session for the user fact `<key>` via the world adapter |
| `given clock <ISO-8601 instant>` | freeze/virtualize the browser clock at a fixed instant (Playwright clock API) |
| `given stub <METHOD> <path> <status> [json <literal>]` | stub a network route (declared, replayable) |

Seeds merge commutatively (see WORLD.md). Patches are the only ordered given.

### Actions

Closed set. Every mutating action MUST declare at least one `expect` —
an unobserved action is a compile error, because that is where races hide.

| action | form |
|---|---|
| `go` | `go <path>` |
| `click` | `click <target>` |
| `dblclick` | `dblclick <target>` |
| `fill` | `fill <target> "<text>"` (clears then types) |
| `select` | `select <target> "<option label>"` |
| `check` / `uncheck` | `check <target>` |
| `press` | `press "<key>" [in <target>]` |
| `hover` | `hover <target>` |
| `upload` | `upload <target> "<relative file path>"` |

### Targets

Semantic only. `<kind> "<name>"` compiles to Playwright `getByRole(kind, { name })`
for ARIA role kinds. Non-role kinds:

- `text "<literal>"` — getByText
- `field "<label>"` — getByLabel (form controls)
- `placeholder "<text>"` — getByPlaceholder
- `testid "<id>"` — getByTestId. The sole escape hatch.

Role kinds (v1): `button link heading textbox checkbox radio combobox option
row cell table list listitem region dialog alert status tab tabpanel menu
menuitem img banner navigation main article form group`.

Scoping: `<target> in <target>` (chainable with `of`:
`cell "x" of row "y" in table "z"` scopes right-to-left).

Ambiguity (two matches at act time) is a **hard error** listing every match.
The runner never picks "the first one."

### Effects

| form | meaning |
|---|---|
| `expect <target>` | target is visible after settlement |
| `expect no <target>` | target absent/hidden after settlement |
| `expect text "<t>" in <target>` | element's text contains `<t>` |
| `expect exact text "<t>" in <target>` | element's text equals `<t>` |
| `expect value "<v>" in <target>` | input value equals |
| `expect count <kind> <n> [in <target>]` | number of matches |
| `expect url <path>` | page url path (+query) matches |
| `expect request <METHOD> <path-pattern> [ok\|<status>]` | this step causes a matching request that resolves with the status (`ok` = 2xx). Armed **before** the action. |
| `expect appear <target>` | explicit transient watcher: armed before the action, so it catches elements that appear and vanish (toasts) |
| `expect gone <target>` | target present at act time must be gone after settlement |
| `let <name> = text in <target>` | capture observed text into `$name` (recorded in trace) |

`$name` interpolates into any quoted string or path.

Path patterns: literal paths, `*` matches one segment, `**` matches any suffix.

## Execution semantics

### Act-and-observe is one transaction

For each step the runner, in order:

1. Resolves and validates the action target (ambiguity/absence = actionable error).
2. **Arms all observers first**: request matchers for declared `expect request`,
   appearance watchers for `expect appear`/`expect toast`-like effects,
   console/pageerror collectors, navigation listeners.
3. Dispatches the action.
4. **Settles** (below).
5. Evaluates remaining state effects against the settled page.
6. Emits a trace record and a checkpoint.

### Settlement

After the action, the runner converges on quiescence:

- all declared `expect request` matchers have resolved;
- the in-flight tracked request count (fetch/XHR/document) has drained to zero;
- no navigation is mid-flight; and
- two consecutive animation frames have run with a drained microtask queue.

There is a per-step **budget** at the bottom (physics demands one). It lives in
runner config, never in test text. Exceeding it does not say "TimeoutError" —
it produces a **stuck-state report**: which requests are still pending, which
navigation never fired, the settled ARIA snapshot, and a per-effect verdict diff.

Any `pageerror` or console error during a step fails the step by default,
attributed to the action that triggered it (opt-out per flow:
`allow console-errors`).

### Determinism

- `given clock` virtualizes time in the page. Timers and `Date.now()` are
  deterministic; the runner auto-advances the virtual clock only while idle in
  settlement, so app-side debounce/polling still runs, but on the virtual
  timeline.
- `given stub` declares network stubs in the flow file — replayable data.
- The world is rebuilt from a pure description per flow (WORLD.md).

## Traces, checkpoints, replay

Every run writes `.bat/runs/<flow>/<runid>/`:

- `trace.json` — per step: the plan (intent), pre-state URL + ARIA snapshot,
  every observed event during settlement (requests with status, console,
  navigations), per-effect verdicts with expected/observed diffs, captures.
- `checkpoint-<n>.json` — URL + storageState + world fingerprint after step n.
- `report.txt` — the human/agent-readable story of a failure.

`bat replay <flow>:<step>` re-runs one step:

1. **Hermetic tier** — flow declares stubs for all traffic: restore checkpoint, run.
2. **Snapshot tier** — world adapter implements `snapshot`/`restore`: restore
   world snapshot + browser checkpoint, run the one step.
3. **Fallback tier** — reseed, fast-forward steps 1..n-1 (still fully settled,
   reduced trace verbosity), then run step n with full observation.

The replay report states which tier ran and why.

## CLI

```
bat check <flow...>     parse + static checks only (no browser)
bat run <flow...>       run flows
bat replay <flow>:<n>   atomic replay of one step
bat inspect <url>       dump the semantic tree (roles/names/testids) of a page
bat doctor              report world adapter capability level + next rung
```

Static checks (`bat check`, always run before `run`): grammar, unknown
action/kind, mutating action without effects, unknown seed name, seed merge
conflicts, dangling refs, patch targets, `$var` used before `let`.

## Config

`bat.config.json` at project root:

```jsonc
{
  "baseUrl": "http://localhost:3000",
  "world": { "module": "./e2e/world/world.ts" },   // or { "http": "http://localhost:3000/api/__bat" }
  "seeds": "./e2e/world/*.seed.ts",
  "flows": "./e2e/flows/**/*.flow",
  "stepBudgetMs": 15000,                             // runtime physics, not test semantics
  "headless": true
}
```
