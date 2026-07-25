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

2. **The DSL is data, not code.** No conditionals, no closures, no arbitrary
   control flow. A flow parses to steps that are serializable records — what
   buys perfect traces and agent-writability. Iteration comes in two forms,
   both of which keep determinism (the real requirement) intact:
   - `for` over a **literal table** is unrolled at *parse* time into flat steps.
   - `for each` over a **live collection** expands at *runtime*: it reads the
     matching elements from the settled page and runs its body per element.
     Determinism is preserved because the collection is read from a settled
     point, which is reproducible given the seeded world + prior steps — the
     same property the fallback replay tier already relies on. Each element is
     pinned with an injected marker so an iteration survives the DOM mutating
     underneath it (removing rows, re-rendering). Every executed iteration is
     an ordinary settled, explained, contiguously-numbered step.

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

### Loops (`for` … `do`) — parse-time unrolling

A `for` iterates a body over a **literal table** of rows and is expanded into
flat steps before anything runs. Each declared `$var` is substituted (by name)
into the body for each row:

```
for $cat $all in            # `in` is optional readability sugar
  "Electronics" "All Electronics"
  "Clothing"    "All Clothing"
  "Books"       "All Books"
do
  click link "$cat"
    expect text "$all"
```

- Each row supplies exactly one value per declared variable (mismatch is a
  parse error). Values are used verbatim — there are no transforms (lowercasing
  etc.); put the exact string you need in the table.
- The body is any sequence of action+effect steps; it may span multiple steps.
- Unrolled steps carry an **iteration label** — a failure reports
  `step 3 … [iteration 2/3: $cat="Clothing"]` and `bat replay flow:3` replays
  exactly that iteration. This is the payoff of unrolling over a runtime loop.
- literal `for` loops may nest (cartesian unroll).

### `for each` — runtime iteration over a live collection

When the number of items is only known at runtime (rows in a cart, search
results, a list that grows or shrinks), `for each` iterates the actual page:

```
for each row in table "cart-items" as $row
  click button "Remove" in $row
    expect gone $row
```

- The collection target (`row in table "cart-items"`) is resolved against the
  **settled** page; its match count is whatever is on the page.
- `as $var` binds each element; use `$var` as a **scope** in the body
  (`in $row`, `of $row`). It is not a string — it names an element.
- Each element is **pinned** (an injected marker attribute) at loop entry, so
  removing/​reordering rows during the body does not misalign later iterations.
- Every iteration's body steps are ordinary steps — settled, explained, and
  labelled `[iteration k/N: $row="…"]`. An empty collection runs the body zero
  times (not a failure); the container step reports `(N matches)`.
- Replay addresses iterations directly: the report's display numbers are stable,
  so `bat replay flow:7` reproduces state through the loop and focuses that one
  iteration. (`--fast`'s per-step checkpoint tier can't span a runtime loop, so
  replaying across one uses the reseed tier automatically.)
- `for each` cannot iterate `let`-captured strings (use `for` for literal data);
  its collection must be a live page target.

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
| `drag` | `drag <target> to <target>` |
| `switch tab` | `switch tab <path-pattern>` (activate an open tab) |
| `close tab` | `close tab` (return to the most-recent remaining tab) |

### Targets

Semantic only. `<kind> "<name>"` compiles to Playwright `getByRole(kind, { name })`
for ARIA role kinds. Non-role kinds:

- `text "<literal>"` — getByText
- `field "<label>"` — getByLabel (form controls)
- `placeholder "<text>"` — getByPlaceholder
- `testid "<id>"` — getByTestId. The sole escape hatch.
- `frame "<name>"` — SCOPE ONLY (via `in`/`of`): an iframe matched by name,
  title, or src substring; nestable.

Role kinds (v1): `button link heading textbox checkbox radio combobox option
row cell table list listitem region dialog alert status tab tabpanel menu
menuitem img banner navigation main article form group`.

Scoping: `<target> in <target>` (chainable with `of`:
`cell "x" of row "y" in table "z"` scopes right-to-left).

Names match case-insensitively as substrings (Playwright semantics). If a name
is ambiguous, a **unique exact-name match wins** (`field "Search"` resolves to
the element labelled exactly "Search" even when "search-results" also
substring-matches). Any remaining ambiguity is a **hard error** listing every
match. The runner never picks "the first one."

### Effects

| form | meaning |
|---|---|
| `expect <target>` | target is visible after settlement |
| `expect no <target>` | target absent/hidden after settlement |
| `expect text "<t>" in <target>` | element's text contains `<t>` |
| `expect exact text "<t>" in <target>` | element's text equals `<t>` |
| `expect matches text "<re>" in <target>` | element's text matches the regex (`/…/flags` or bare source) |
| `expect title "<t>"` / `expect exact title …` / `expect matches title …` | browser tab title |
| `expect value "<v>" in <target>` / `expect value matches "<re>" in <target>` | input value |
| `expect attribute "<attr>" "<v>" of <target>` / `… matches "<re>" of …` | element attribute (href, aria-*, data-*) |
| `expect checked <target>` / `expect unchecked <target>` | checkbox/radio state |
| `expect enabled <target>` / `expect disabled <target>` | control enablement |
| `expect selected "<label>" in <target>` | a `<select>`'s chosen option label |
| `expect count <kind> [>=\|<=\|>\|<] <n> [in <target>]` | number of matches (exact, or a comparison) |
| `expect url <path>` | page url path (+query) matches |
| `expect request <METHOD> <path-pattern> [ok\|<status>] [containing "<t>"]` | this step causes a matching request that resolves with the status (`ok` = 2xx). `containing` matches the request BODY — how you pin a GraphQL operation. Armed **before** the action. |
| `expect ws sent "<t>" [on <path-pattern>]` / `expect ws received "<t>" [on <path-pattern>]` | a websocket frame containing `<t>` (optionally on a socket path). Armed **before** the action; settlement waits for it. |
| `expect appear <target>` | explicit transient watcher: armed before the action, so it catches elements that appear and vanish (toasts) |
| `expect gone <target>` | target present at act time must be gone after settlement |
| `let <name> = text in <target>` | capture element text into `$name` |
| `let <name> = value in <target>` | capture an input's value |
| `let <name> = attribute "<attr>" of <target>` | capture an attribute (e.g. an `href` to navigate) |
| `let <name> = count <kind> ["name"] [in <target>]` | capture a match count |
| `let <name> = query "<param>"` | capture a current-url query parameter (create → redirect → `:id`) |

`$name` interpolates into any quoted string or path.

Path patterns: literal paths, `*` matches one segment, `**` matches any suffix.
A pattern with `?` requires the named query params to match; extra params in
the URL are ignored (frameworks append their own, e.g. Next.js `_rsc=`).

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

**Realtime**: SSE responses (`text/event-stream`) are deliberately long-lived
and EXEMPT from drain — an SSE app settles normally, and the stream appears in
traces marked as live. Websocket frames are recorded per step and matched by
armed `expect ws` effects; frame-driven DOM churn is covered by mutation-quiet.

**Tabs / popups** are first-class. `expect tab <path>` is armed before the
action and gates settlement (an unexpected tab is recorded, not punished);
`switch tab <path>` makes the matching open tab the active page — all later
actions, effects, and settlement run there; `close tab` returns to the most
recently used remaining page.

**Native dialogs** have real vocabulary: `expect dialog "<msg>" accept` /
`dismiss` / `accept "<prompt text>"` declares the response BEFORE the action,
so the reply is deterministic data. An undeclared dialog is dismissed and
fails the step, naming the exact line to add (opt-out: `allow dialogs`).

**iframes** are a scope kind: `click button "Pay" in frame "checkout"` reaches
into an iframe matched by name / title / src substring; composable and
nestable. `frame` is scope-only — `click frame "x"` is a parse error.

**Downloads**: `expect download "<name>"` matches by suggested filename and
saves the file into the run directory as evidence (`download-<name>` in the
trace).

**Drag-and-drop**: `drag <target> to <target>`.

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
bat run <flow...>       run flows; every failure gets a causal explanation
bat replay <flow>:<n>   atomic replay of one step
bat inspect <url>       dump the semantic tree (roles/names/testids) of a page
bat doctor              report world adapter capability level + next rung
```

### Automatic failure explanation

bat never issues a verdict — it can't. An app's timing may legitimately vary,
and whether an observed state is "broken" depends on user expectations, which
only the flow's author knows (the flow *is* the attempt to encode them). What
bat does instead: any time a run fails, it replaces "couldn't find X after n
seconds" with a causal, evidence-backed **`why this failed:`** section:

1. **What was expected vs what the settled page showed** — always concrete
   values, never bare timeouts.
2. **What the page did during the step** — requests with status and completion
   order, injected-condition annotations, errors, navigations.
3. **Reproducibility** — the flow is rerun N times (config `rerunsOnFailure`,
   default 4) with the identical seeded world and steps.
   - Reruns disagree → "NOT deterministic," and the outcome is cross-tabulated
     against response completion order. When order explains it exactly, the
     report states the fact and offers **both readings**: if a user must always
     end in the expected state, the app doesn't guarantee it under every
     ordering; if every ordering's end state is acceptable, the expectation is
     stricter than the app's actual contract. Deciding which is a question of
     intent — the report equips the human to answer it in a minute.
   - All reruns fail identically → "fully reproducible: stable behavior, not a
     timing variation" — rerunning will never change it.
   - Rerun traces are saved next to the run's trace for side-by-side diffing.
4. **Near-miss facts** — a target that matches nothing, when something close
   exists ("closest present: heading \"Products\""), is pointed out as a naming
   mismatch in the flow.
5. **Conditions check** — with simulated bad conditions active, one clean rerun
   runs without them; if it passes, the report states the failure only occurs
   under the injected conditions, with both readings (missing app resilience
   vs an over-harsh profile).

### Simulated bad conditions

Conditions are runtime physics, never test semantics — they live in config or
CLI flags (`--latency 200-1500 --fail-rate 0.05 --seed 7`), never in flow
files, and require a seed so every chaos run is reproducible. Every injection
is recorded: the report header announces the active profile, and each affected
request is annotated (`[+842ms injected latency]`, `[injected failure
(conditions)]`) so a chaos-induced failure can never masquerade as a real one.
`given stub` routes are registered after the condition route and therefore win:
stubbed traffic is hermetic and immune to chaos. Invariant (enforced by
`scripts/chaos-gauntlet.ts`): injected latency alone must never fail a flow.

Static checks (`bat check`, always run before `run`): grammar, unknown
action/kind, mutating action without effects, unknown seed name, seed merge
conflicts, dangling refs, patch targets, `$var` used before `let`.

## Parallelism

World isolation is structural, never cooperative: parallel workers each get an
ISOLATED app instance (own port) and an isolated world — two workers can run
the same flow simultaneously because they are never in the same world.

```jsonc
{
  "workers": 4,                                  // or bat run --workers 4
  "baseUrl": "http://localhost:{port}",          // {port}/{index} substitute per worker
  "app": {                                        // bat launches one instance per worker
    "command": "npx next start -p {port}",
    "readyUrl": "/login",                        // polled until the app responds
    "env": { "POSTGRES_URL": "postgres://localhost/myapp_w{index}" }
  },
  "world": { "module": "./e2e/world/world.ts" }  // may export createWorld(env)
}
```

- The `app` spec makes bat own the app lifecycle (also with 1 worker — no more
  "start the app first"). PORT/BAT_PORT/BAT_WORKER are set in the child env;
  all app output is retained under `.bat/app-logs/worker-N.log`; a startup
  failure reports the log tail, never a bare timeout.
- A world module may export `createWorld(env: { index, port, baseUrl })` to
  bind each worker slot to its own isolated state — e.g. create + migrate
  `myapp_w<index>` on first use (see apps/nextjs-dashboard/e2e/world/world.ts).
  HTTP worlds get `{port}`/`{index}` substituted in `world.http` instead.
- Requesting workers > 1 without an `app` spec is a hard error explaining
  exactly what to add — bat never runs parallel flows against a shared world.

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
