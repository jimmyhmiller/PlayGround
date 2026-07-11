# Scry — Status: What Works, What Doesn't

_Last updated: 2026-07-11. This is the honest source-of-truth for the current build.
For design rulings see `DECISIONS.md`; for the as-built log + Coil friction see `06-implementation.md`._

## What Scry is

A statically-typed, class-based OO language whose **product is runtime observability**.
Every object lives in a per-type slab arena, so "show me every live `Agent`" is a slab
walk, not instrumentation you bolt on. A browser viewer ships day one and is a **live
REPL into the running program** — you don't watch a stream of events, you evaluate
expressions against the real heap, invoke methods, run declared actions, and hot-swap
code, all while the program runs.

Implemented in **Coil** (a Lisp-syntax low-level language). The compiler is a
lexer → typechecker → bytecode VM; there is no JIT yet. ~301 golden + end-to-end tests
gate every change (`python3 tests/run-tests.py`).

- **Thesis:** observability-first. The viewer is not a debugger bolted on — it's the point.
- **Rulings that won't change:** Java-style nominal interfaces (no inheritance, ever);
  real OS threads day one (async/await later); the only viewer wire op is `eval`
  (`{id, source} → {id, value|error}`) — no subscription/push/delta protocol.

---

## Quick start

```bash
coil build                       # build the `scry` binary (Coil toolchain required)
./scry run examples/assistant.scry   # run a program (opens a live viewer on :7400+)
./scry check foo.scry            # typecheck only, no run
./scry inspect foo.scry          # serve static class structure WITHOUT running
./scry portal                    # reverse-proxy hub on :7357 — discovers & lists all programs
python3 tests/run-tests.py       # the full gate (~301 tests)
```

CLI subcommands (all real): `run`, `check`, `inspect`, `portal`, `eval`, `schema-json`,
`parse-dump`, `dump-types`, `dump-bytecode`.

---

## The language — what works

Everything in this section is implemented, typechecked, and covered by golden tests.

### Types & declarations
- **Classes** with typed fields, `init`-style construction with definite-assignment
  checking, memberwise named construction `Account(owner: c, kind: Kind.Checking)`.
- **Methods** with `self`, `Void` returns, method chaining.
- **Java-style interfaces** — nominal, `implements`, multiple interfaces per class,
  itable-based dynamic dispatch, interface-typed fields and lists. No inheritance.
- **Enums** — nullary variants and variants **with payloads**; equality; exhaustive
  `match` with payload binding and wildcards.
- **Objects** (singletons).
- **Generics** — monomorphized: generic classes, generic functions, nested generics
  (`Map<String, List<Message>>`).
- **Top-level functions** (`fn fib(n: Int) -> Int`), recursion, self-tail-calls.

### Expressions & control flow
- `let` / `var` / typed `let`, assignment, `if/else if/else` (statement and expression),
  ternary, `while`, `for … in` over lists, `return`.
- Arithmetic (Int + Float, correct precedence), comparisons, boolean short-circuit, unary.
- **Strings** — plain, escapes, `${}` interpolation (including nested), `+` concat, slicing,
  `charCodeAt`/`fromCharCode`, `toInt`/`toFloat`, `length`.
- **`Result` + `try`** (`?`-style) — `try` only legal in a function returning `Result`
  (enforced); propagation works.
- **`match`** — on enums (with payload binding), on primitives/strings (wildcard required).

### Standard library (built-in)
- `List` (new/push/get/len, `for … in`), `Map` (new/set/get/has/keys/len, Int or String keys).
- `Mutex` (new/lock/unlock/get/set) — real mutual exclusion across threads.
- `Console` (log/print/**readLine**), `Clock` (now/**sleep**), `Env.get`.
- `Thread.spawn(Runnable)` / `Thread.join` — **real pthreads**, N-thread stop-the-world
  safepoints, atomic-bump arena allocation. `readLine` and `sleep` are
  safepoint-cooperative, so the viewer stays live while a program blocks at a prompt.

### Written in Scry (not built-in)
- `std/json.scry` — a JSON parser/serializer written in the language itself.
- `std/agent.scry` — the `Model` interface + agent-loop scaffolding.

---

## The runtime — what works

- **Bytecode VM** — untagged slots, typed opcodes, itable interface dispatch.
  `fib(35)` ≈ 0.91s; on our benchmarks Scry is faster than CPython on tight loops and
  allocation, roughly par on recursion, slower on integer-heavy inner loops (no JIT).
- **Per-type slab arenas** — each type gets its own arena; instance enumeration is a
  slab walk. This is what makes `Agent.instances()` cheap and the whole observability
  story work.
- **Real threads** — OS threads, STW safepoints, atomic-bump allocation, `Mutex`.
- **Uncrashable eval** — an eval from the viewer can **never** kill the process. Syntax
  errors, stale references, arena-OOM, bad opcodes, and even internal compiler-phase
  invariants all come back as a clean `{id, error:{kind,…}}` JSON while the server stays
  live. (This was hard-won: parser, VM, **and** compiler/type-table hard-`exit` sites all
  route through an eval landing pad — `evalrt.coil`.)
- **Concurrent, non-blocking viewer** — the eval server is thread-per-connection. A long
  eval (`fib(33)`, ~380ms) no longer freezes read-only polls: the running eval lends its
  stopped heap to a waiting reflection poll at a safepoint, so `types()` during `fib(33)`
  returns in ~1ms instead of blocking. No two evals ever execute concurrently; mutating
  evals stay exclusive; reflection reads happen with language threads parked.

---

## Live code change — what works, what's rejected

You can redefine code in a running program by evaluating a definition. The rule is:
**changes that can't invalidate existing instances are applied live; everything else is a
loud, explicit rejection** (never a silent corruption).

**Allowed live:**
- Swap a method/function **body** (generation counter bumps, method-table swap at STW,
  object identity preserved, threads see it instantly).
- **Add** a field to a class (32B/slot headroom; old instances get a default; additive only).
- Add a method to an **interface** with a default-free signature (conformance re-checked).

**Rejected loudly (by design — no migration engine yet):**
- Field **remove**, **rename**, or **retype**.
- Method **signature change**; **adding** a method to a class.
- `migrate { … }` blocks.
- Redefining an **enum** or an **object**.
- A new body that fails typecheck (rejected, old body kept).

---

## Observability & the viewer — what works

Served from disk (vendored no-build React 18 + htm), dark-first, `Cache-Control: no-store`
so you never get a stale `app.js`. The wire surface is `POST /eval` plus read-only
reflection ops: `types`, `schema`, `fields`, `methods`, `graph`, `views`, `actions`,
`functions`, `trace`, `generation`.

- **Map view (default)** — bespoke nested-containment visualization (ownership = nesting,
  size = live-instance mass, shared instances share an identity color with
  hover-highlight-all-appearances). A generic force-graph was explicitly rejected; the
  program's own structure drives the picture. Functions live **in** the map (not a
  separate mode) — click one to get its recursion tree.
- **List view** — census ribbon of types with live counts; click a type → its instances →
  an instance detail inspector.
- **Instance inspector** — fields, clickable ref-navigation (with breadcrumb stack),
  methods, and **invoke**. Invoke opens the card to show output; params get typed pickers:
  entity → dropdown of live instances **or `+ create new`** (a recursive constructor form,
  so you can build a whole object graph inline), enum → variant dropdown, bool → toggle,
  numbers → validated, strings → auto-quoted. No bare-word crashes.
- **`view Name for T { … }`** — a first-class language construct: the program declares how
  a type should be rendered (`title:`, `size: byCount`, `badge:`, `section "…" { field as
  timeline | chips | rows | card | heat }`). The app speaks to its own visualization.
- **`action "Label" for T (params) { body }`** — the mirror of `view`: curated interaction
  affordances (state changes / side effects) rendered as green ACTION buttons above the
  method list. Desugars to hidden synthetic methods; full reuse of the invoke path.
- **`trace(expr)`** — records a call tree (bounded 20k nodes, zero-cost when off), rendered
  as a collapsible recursion tree with stats. This is how a function like `fib` "visualizes."
- **Static inspection with no running process** — `scry inspect` serves class structure
  without running; `scry portal` discovers projects in the working tree, shells a transient
  `schema-json` on demand (cached by path+mtime), and serves schema/views/actions
  statically with **zero lingering process**. Running programs register and appear as live
  cards you can jump into. Static views (List + inspector schema) work for 0-instance types.

---

## Real-world capability — what works

- **Real HTTP(S) via libcurl** — `Http.request` with real TLS/cert validation. Uses a
  cooperative multi-interface so in-flight requests stay live-inspectable; `HttpResponse`
  is an arena entity you can browse.
- **A real LLM agent loop** — verified live against DeepSeek's Anthropic-compatible
  endpoint (default `deepseek-v4-pro` @ `https://api.deepseek.com/anthropic`; key from
  `DEEPSEEK_API_KEY | DEEPSEEK_KEY | ANTHROPIC_API_KEY`). Real tool-use: the model picks a
  tool + args, Scry runs it, the model uses the result. Every `Message` / `ToolCall` /
  `HttpResponse` is browsable, pausable, and hot-swappable live. `deepseek-v4-pro` is a
  reasoning model, so the first content block is `thinking` — the parser skips it.

### Flagship examples
- `examples/assistant.scry` — interactive Claude-Code-like REPL. `readLine` is
  safepoint-cooperative (evals run while it's blocked at the prompt); `research <topic>`
  spawns subagent threads; `loop <task>` is a pausable background loop; ships
  Pause/Resume/Ask/Spawn actions and `view AgentBoard` / `view Pulse`. Messages sent from
  the viewer echo in the terminal and redraw the prompt.
- `examples/kanban.scry` — structure, shared instances, views, actions.
- `examples/bank.scry` — actions that mutate live state.
- `examples/todo.scry` — interactive stdin.
- `examples/functions.scry` — `fib`/`gcd`/`ack` for the trace/recursion-tree view.

---

## What does NOT work yet (limitations)

These are real, current gaps. Most are deliberate PoC cut-lines, not bugs.

- **No garbage collector.** Each type's arena has a fixed cap; exhausting it is a hard
  `OutOfArenaSpace` error ("GC not implemented in this build"). Long-running or
  high-allocation programs will hit the wall (e.g. a tight `Point`-allocating loop caps
  around 100k). This is the biggest limitation.
- **No lambdas / closures.** Explicit parse error: "lambdas/closures are not supported in
  this PoC." Functions are top-level only; no first-class function values.
- **No generic bounds.** Generics are monomorphized but a bound on a type parameter
  (`T: Comparable`) is rejected: "generic bound is not implemented in this build."
- **No async/await.** Concurrency is real OS threads only (by design — async is post-PoC).
- **No JIT.** Bytecode interpreter only; integer-heavy inner loops are slower than CPython.
- **No migration engine.** Field remove/rename/retype, method signature change, method
  add, `migrate` blocks, and enum/object redefinition are all rejected (see Live code
  change above). Redefinition is additive-only.
- **No interface default-method bodies.** Interfaces declare signatures only (leaning
  permanently no for the PoC).
- **Recursive "create new" in the invoke form is depth-bounded** (max 3) to keep the UI
  finite.

---

## Test & quality status

- **~301 golden + end-to-end tests, all green.** Categories: `parse/` (AST snapshots +
  parse errors), `check/` (50 typecheck-error cases `e01`–`e50` + 25 success cases
  `s01`–`s25`), `eval/` (the reflection/live-change/trace wire surface, ~90 cases),
  `run/` (end-to-end program output, ~90 cases), `run-arenas/`, `run-err/`, `http/`,
  `app/` (full CLI/agent-loop e2e).
- The typechecker is strict and well-covered: 50 distinct error classes have golden
  expected messages (arity, type mismatch, non-exhaustive match, unknown field/method/
  variant, `try` misuse, view/action validation, definite-assignment, etc.).
- **Uncrashable-by-eval is a tested invariant** — bad input always returns a typed error,
  never a crash or white-screen.
- Known flaky: `ui_smoke` is a browser test that also fails on a pristine binary; it is not
  a regression signal.

## Recently fixed (this cycle)

- **Process crash on method invoke** — unguarded `exit()` in compiler/type-name paths could
  quit the whole process during an eval; all such sites now route through the eval landing
  pad. An eval can no longer kill the program.
- **Viewer freeze during long evals** — thread-per-connection + consistent-read handoff;
  `types()` during `fib(33)` went 380ms → 1ms. (This resolves the long-standing OPEN
  question in `DECISIONS.md` about how the eval channel interleaves with running threads.)
- **Two latent memory bugs** surfaced by the concurrency stress and fixed: source buffers
  weren't NUL-terminated (a use-after-free once buffers are freed), and arena objects were
  published `LIVE` with zeroed fields before their constructor ran (a concurrent reader
  could deref a null `String` field — now they point at a shared empty-string sentinel).

---

## Where to read more

- `DECISIONS.md` — 17 pinned rulings (don't re-litigate) + remaining open questions.
- `00-vision.md` … `05-milestones.md` — thesis, language, runtime, live semantics, viewer, milestones.
- `06-implementation.md` — as-built handoffs + the Coil friction log (read before writing Coil here).
- `DEMO.md` — demo scripts.
