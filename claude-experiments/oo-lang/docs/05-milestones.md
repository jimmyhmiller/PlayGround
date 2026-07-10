# 05 — Milestones

This is the cut-line document. Five milestones, M0 → M4, each one a runnable thing you can
point at and say "this now works," ending at the demo in `00-vision.md`. Nothing in here is
aspirational — if a capability isn't in some milestone's IN list, it does not exist for the
PoC, and the thing that would normally use it either doesn't exist yet or hard-errors. See
"What we fake" at the bottom for the exhaustive list of PoC-era stubs and exactly how each
one fails loudly.

The ordering is load-bearing: M1 (viewer) cannot exist before M0 gives it per-type arenas to
walk; M2 (invoke) cannot exist before M1 gives the viewer a channel to send requests on; M3
(live change) cannot exist before M2 gives us a safe way to mutate a running interpreter at
all. Nothing here is parallelizable across milestones — it's a strict chain.

## M0 — Minimal interpreter

**Goal:** `oo run foo.oo` compiles and runs a CLI program with classes, methods, and
generics — and every object it allocates already lives in a per-type arena, before any GC
exists to sweep them.

**IN**
- Lexer/parser for the pinned surface syntax: `class`, fields with type annotations, `fn`
  methods, `self`, top-level `fn main()`, `if`/`else`, `while`, `for`-over-`List<T>`,
  `enum` (with payloads) and `match` (per `01-language.md` §1.3 — pulled into M0, not left
  for later, because status-shaped types like `AgentStatus` are needed starting M2's
  demo, and M4's `Tool` polymorphism below builds directly on `enum`/`match`),
  arithmetic/comparison/boolean operators, string interpolation, single-file programs.
- Primitive types: `Int`, `Float`, `Bool`, `String`. One built-in generic: `List<T>`
  (push/get/len/iterate). No `Map`/`Set` yet unless a demo class needs one — add narrowly,
  don't build a stdlib.
- Nominal static type checker: field types, method signatures, call-site arg/return
  checking, `self` typed to the enclosing class. No inheritance means method resolution
  is a flat lookup on one class — no vtables, no MRO. Generics are checked structurally
  per call site. Per `01-language.md` §2.4, `02-runtime.md` §5, and `DECISIONS.md` #6,
  **generic classes are monomorphized from day one**: `Inventory<Tool>` and
  `Inventory<Task>` are distinct compiled types with distinct arenas, exactly like any
  other pair of classes — this is not an M0 shortcut, it's required for the per-type-arena
  enumeration model to apply to generic classes at all (M0's own acceptance demo below
  proves this with `--dump-arenas` on `Inventory<T>`). Boxed-uniform/erased representation
  is real but scoped narrowly to **generic functions** (e.g. `fn first<T>(...)`), which
  touch no arena and can be compiled once with an erased calling convention without
  breaking anything the viewer or GC depends on. Type errors are compile-time hard
  failures with a source span; there is no `any` escape hatch.
- Bytecode compiler + stack-based VM, direct descendant of the `clox` port
  (`coil/apps/clox/`): same dispatch-loop shape, extended with typed local slots (the
  typechecker has already proven types, so the VM doesn't need runtime tags for anything
  the checker covers — only for values whose static type is a class, so the GC/arena code
  can identify what it's holding).
- **Per-type arenas from the first allocation.** Every `class Foo` gets its own slab the
  first time a `Foo` is constructed. Layout is fixed-size records (the class's field
  layout is static — no dynamic shape) packed contiguously, magazine/slab style, so
  "every live `Foo`" is a linear walk with no pointer chasing through a general heap.
  Each slot gets a stable identity: `(type_id, slab_index, generation)` — this triple
  *is* the `Agent#7`-style id the viewer will use later, and it must be designed now,
  because retrofitting identity onto objects after the fact means walking every existing
  reference to add it.
- CLI: `oo run foo.oo` — parses, typechecks, compiles, runs to completion, ordinary
  stdout/stderr/exit code. `oo run --dump-arenas foo.oo` — internal debug command that
  prints, at exit, each type's slab occupancy (a sanity check for M0 that doesn't need a
  UI to verify the arena design works).

**OUT**
- No garbage collection. Slabs only grow. See "What we fake" — this has a hard, loud
  failure mode, not a silent one.
- No `interface`/`implements` mechanism, no operator overloading, no closures/lambdas, no
  exceptions (use a `Result`-shaped return or crash — exceptions are not a PoC
  requirement). This does **not** mean the PoC has zero polymorphism: the one place the
  demo genuinely needs a heterogeneous slot (`Agent.tools: List<Tool>`, per
  `01-language.md` §2.3 and `DECISIONS.md` #10's "swap a tool" beat) is met by M4's
  `enum`-dispatch fallback, not by interfaces — see M4's IN list below and the OPEN-items
  section at the end of this doc.
- No modules or multi-file programs. No package manager, no imports beyond the one file.
- No concurrency of any kind — straight-line single-threaded execution, top to bottom.
- No viewer, no server, no redefinition. Those are M1–M3.

**Acceptance demo:** write a ~60-line `.oo` file with 3 classes (one generic, e.g.
`Inventory<T>`), fields, methods that call each other and mutate `self`, a loop building a
`List`. `oo run foo.oo` prints the expected output. `oo run --dump-arenas foo.oo` shows one
slab per class with the expected occupancy count, proving instances of different classes
never share a slab.

**Risk notes**
- Deciding the object-identity scheme (`type_id/slab_index/generation`) *is* deciding a big
  chunk of the eventual wire protocol's `Agent#7` id format — get this wrong here and every
  later milestone inherits the mistake. Nail it in M0, not "later, in the viewer."
- Monomorphizing generic *classes* (`Inventory<T>` and every instantiation of it) is not
  optional here, despite multiplying codegen paths right when the compiler is least
  mature — `DECISIONS.md` #6's per-type-arena model has no other way to make "enumerate
  every live `Inventory<Tool>`" cheap and well-defined, and M0's own acceptance demo below
  depends on it. Boxed-uniform is real, but scoped to generic *functions* only, which never
  touch an arena. An earlier draft of this doc scoped boxed-uniform to generic classes too
  ("the simpler PoC path") — that was wrong: it silently contradicts this same milestone's
  `--dump-arenas` acceptance check on `Inventory<T>`, which only proves anything if
  `Inventory<T>` is actually monomorphized per instantiation.
- The bytecode format has to carry enough static type information for the arena/GC code to
  walk a stack frame or a record and know which slots are class-typed pointers vs. raw
  ints/floats — this needs to be right from the first instruction set revision, because M1
  onward assumes the runtime can always answer "what type is this value" without a runtime
  tag check.

## M1 — Viewer, read-only

**Goal:** point a browser at a running M0 program and see its live objects — type list with
counts, searchable instance tables, instance detail — without touching or mutating anything.

**IN**
- Runtime embeds a server (transport TBD by `02-runtime.md` §7's risk note — WebSocket or
  a simpler long-poll/SSE, that doc's call, not this one's), on by default, printing
  `viewer: http://localhost:7357` (or `--viewer=off` to disable). This is the *only* new
  runtime capability M1 adds beyond M0 — everything else is read access to structures M0
  already built.
- **The wire protocol is the eval channel, in full, from day one:** request
  `{"id": <int>, "source": "<oo-lang source text>"}`, response `{"id": <int>, "value":
  ...}` or `{"id": <int>, "error": "..."}`. There is no `types`/`instances`/`instance` op
  set distinct from later milestones — M1 does not get its own wire shape that M2/M3 later
  replace or extend. Every milestone's new capability is a new *kind of source text* the
  server is willing to run, never a new request/response shape.
  ```json
  → {"id": 1, "source": "Type.summary()"}
  ← {"id": 1, "value": [{"type": "Agent", "count": 3}, {"type": "Message", "count": 47}]}

  → {"id": 2, "source": "Agent.instances()"}
  ← {"id": 2, "value": [{"ref": "Agent#7", "type": "Agent", "name": "researcher",
                          "status": "Paused", "conversation": {"ref": "Conversation#2"}}]}

  → {"id": 3, "source": "Agent#7"}
  ← {"id": 3, "value": {"ref": "Agent#7", "type": "Agent", "name": "researcher",
                         "status": "Paused", "conversation": {"ref": "Conversation#2"},
                         "methods": [{"name": "pause", "params": [], "returns": "Unit"},
                                     {"name": "resume", "params": [], "returns": "Unit"}]}}
  ```
  Class-typed fields serialize as `{"ref": "Type#n"}` (using the M0 identity triple),
  never inlined — the viewer fetches on click via a further `eval`, so instance payloads
  stay flat and cheap. Exact stdlib method names (`Type.summary()`, `.instances()`) are
  `01-language.md`'s call; the shape of the request/response envelope is what's pinned here.
- **M1's viewer UI only ever emits canned, UI-generated `source` text — there is no
  free-form input box yet (that's M2's REPL dock).** The type index sends a fixed
  summary expression, drilling into a type sends that type's `.instances()` expression,
  clicking a row sends that instance's own ref expression. The instance-search box compiles
  down to one of a tiny, fixed family of filter expressions: `field == literal`,
  `field != literal`, numeric `<`/`>`/`<=`/`>=`, string `contains`, no boolean combinators,
  no nested field paths — a search box, not a query planner, expressed as generated source
  text rather than a bespoke query op.
- Browser UI: type index (name, live count, sparkline/arrow if the count is climbing) →
  click into a searchable instance table (columns = fields) → click a row into an instance
  detail view (every field, typed, refs are links; methods listed with signatures but
  **not clickable yet** — that's M2).
- Freshness: the client re-sends the same canned `eval` request on a fixed short interval
  (e.g. every 250ms) while a view is open, and gets a fresh `value` back each time. This is
  the client re-asking, not the server noticing a mutation and pushing anything — there is
  no dirty-bit or write-barrier machinery in the interpreter, and none is needed, because
  nothing is ever sent that the client didn't just ask for. It looks identical to the
  audience.
- **M1's server-side eval gate — hard-enforced by the runtime, not a UI convention.**
  Because the wire channel already speaks full `eval`, and `eval` can run anything the
  language can express — including a mutating method call or a redefinition — M1 cannot
  rely on "the UI only ever sends read expressions" as its safety boundary; nothing stops
  a different client on the same socket from sending `Agent#7.resume()` or a new method
  body, and M1 has none of M2's invoke-marshaling or M3's typecheck-and-swap machinery
  built yet to run one safely. So the server classifies parsed `source` before running
  anything: a **definition** (`class`, `fn`, `enum` at top level, or a redefinition of an
  existing method) or an expression whose evaluation would **call a mutating method**
  (anything of the shape `expr.method(args)` that isn't a plain field read the typechecker
  already knows is side-effect-free) is refused outright, with a hard error, and nothing
  runs:
  ```json
  → {"id": 4, "source": "Agent#7.resume()"}
  ← {"id": 4, "error": "mutating eval (method calls, definitions) is disabled until M2 — M1 is read-only"}
  ```
  Field reads, `.instances()`-style enumeration, and comparison/filter expressions are
  permitted; nothing that could mutate state or install new code is, regardless of what
  the UI happens to offer. This gate is exactly what M2 lifts (for method calls) and M3
  lifts (for definitions) — see those milestones' IN lists.

**OUT**
- No invoke UI, no REPL dock, no free-form eval exposed to a person — M1's viewer only ever
  emits the canned expressions above, and the server-side gate refuses anything else even
  if it arrives some other way (see IN).
- Mutating eval (method calls, definitions) hard-refused at the server, per the gate above
  — not merely absent from the UI.
- No true server-initiated push of any kind — revisit only if client-driven re-eval proves
  visibly laggy in rehearsal.
- No auth/access control on the embedded server — PoC assumes localhost, trusted operator.

**Acceptance demo:** run the M0 demo program with the viewer on, open the printed URL,
see the type index with counts, click a type, see a real instance table, type a query into
the search box and see it filter, click an instance and see every field typed with refs as
links. Watch a count climb while the program's own loop keeps allocating, with no code in
the program aware the viewer exists. Separately, send a hand-crafted `{"id": 99, "source":
"Agent#7.resume()"}` at the raw socket (bypassing the UI) and confirm the server still
refuses it — proving the read-only boundary is enforced by the runtime, not by what buttons
the page happens to show.

**Risk notes**
- JSON-serializing a `List<T>` field usefully (not just "List<Message>#3, opaque") needs
  a decision: show length + first N elements? A dedicated array-view? Keep it to
  "length + link to a filtered instance table pre-scoped to this list's contents" —
  avoids inventing a second serialization shape for collections.
- Re-eval interval is a real tuning knob: too slow and the "climbing counts" beat looks
  static; too fast and re-running/re-serializing a large arena's `.instances()` on every
  tick burns CPU the demo program needs for its own work. An interpreter-side global
  monotonic "anything mutated" counter, bumped on every field write, lets the server
  short-circuit re-serialization when nothing has changed since the last identical
  request — a caching optimization inside the eval handler, not a protocol feature; the
  wire shape and the client's "re-ask, get an answer" behavior are unaffected either way.
- The mutating-eval gate above needs a real classifier, not a string-matching heuristic —
  it has to walk the same parsed/typechecked AST the interpreter would run, so "is this a
  definition" and "does this call resolve to a method with side effects" are answered by
  the typechecker, not by pattern-matching the source text (which a client could trivially
  get around with whitespace or aliasing).

## M2 — Interaction

**Goal:** invoke methods and evaluate expressions against the live process from the
viewer, and see the results reflected both in the viewer and in the program's own output.

**IN**
- **M1's server-side eval gate is lifted for method calls.** The wire shape doesn't
  change — still `{"id", "source"}` → `{"id", "value"}`/`{"id", "error"}` — what changes is
  that the classifier M1 added now *permits* an expression whose evaluation calls a
  mutating method: `{"id": 5, "source": "Agent#7.resume()"}` → runtime resolves the method
  on the live instance, runs it, and returns `{"id": 5, "value": null}`. There is no
  separate `invoke` op distinct from `eval` — a click on `pause()` in the instance detail
  view is the viewer generating exactly this shape of `source` and sending it down the same
  channel M1 already speaks; the previously-canned UI now also emits method-call source
  text, not just reads.
- **The viewer grows a REPL dock**: a free-form text input that sends whatever the user
  types as `source`, verbatim, over the same channel — e.g. typing `Task#3.setPriority(1)`
  in the dock sends `{"id": 6, "source": "Task#3.setPriority(1)"}`. This is not a new
  wire feature; it's the UI finally exposing the channel with no canned-expression
  restriction, now that the server permits method calls. It is *not* a general expression
  language extension — no new bindings, no `let` — one expression per request, evaluated
  in the context of the running program's classes and live instances. The gate still
  refuses definitions (redefining a class or method) at this point — see OUT — so the REPL
  dock's freedom is real but bounded, matching what the server will actually run.
- Argument marshaling: only primitive-typed literal arguments (`Int`, `Float`, `Bool`,
  `String`) written directly in the call's source text are accepted. A method whose
  signature includes a class-typed or generic-typed parameter cannot be invoked from the
  viewer in the PoC — there's no handle-literal syntax for "pass me that other live
  instance" yet, and this milestone doesn't add one.
- **The invoke-safety mechanism** (this milestone's actual hard problem, and the answer
  to the OPEN "how is the viewer's eval/invoke channel safe" question in
  `DECISIONS.md`): the interpreter runs a single OS thread. The bytecode compiler emits a
  **safepoint check** at every loop back-edge and every call site. A pending `eval`
  request is queued; the main interpreter thread services the queue when it hits the next
  safepoint, runs the evaluated expression to completion (itself just more bytecode,
  subject to the same safepoints), then resumes whatever it was doing. No second thread
  ever touches interpreter state. This is a *proposal*, not a final ruling —
  `03-live-semantics.md` and `02-runtime.md` own the durable answer; M2 needs *a* working
  mechanism and this is the one we build first because it needs no lock-free data
  structures and no GC-safepoint infrastructure that doesn't already have to exist for
  M0's future GC anyway.
- **OPEN, proposed stopgap for concurrency**: the demo needs "agents take turns" to look
  concurrent without real OS threads. M2 ships a cooperative single-threaded turn
  scheduler (a builtin `Task` queue the interpreter drains one bytecode-safepoint-bounded
  turn at a time) rather than deciding the language's real concurrency model. This
  sidesteps the OPEN concurrency-model question for the PoC; it is explicitly not a
  proposed answer for the language, only a scheduling trick inside the demo app.

**OUT**
- No real OS-thread concurrency, no `async`/`await` surface syntax — see the stopgap above.
- **No live code redefinition.** The eval gate still hard-refuses a `source` that parses as
  a definition (`class`, `fn`/method redefinition, `enum`), exactly as M1 did — M2 only
  lifts the gate for method calls, not for definitions. Lifting it for definitions, plus
  the typecheck-against-live-class and method-table-swap machinery that makes doing so
  safe, is M3's entire IN list.
- No object-typed or generic-typed invoke arguments from the wire.

**Acceptance demo:** in the viewer's instance detail for `researcher`, click `pause()` →
in the terminal, the status line flips to `⏸ paused` within a frame. Open the REPL dock,
type `Task#3.setPriority(1)` → the TUI's task list visibly re-sorts. Re-opening (or the
periodic re-eval of) `researcher`'s detail pane shows the updated `status` field — there is
no separate push event, the pane's next re-eval of `Agent#7` just returns the new value.

**Risk notes**
- This is the riskiest milestone in the whole plan. Get the safepoint placement wrong
  (e.g., missing one inside a hot loop) and an invoke request can stall indefinitely with
  no feedback — the UX must surface "waiting for the interpreter to reach a safepoint"
  rather than hang silently, and after a timeout, error explicitly rather than spin
  forever unexplained.
- If any demo code does blocking I/O (a real LLM API call) without yielding at a
  safepoint first, invoke requests queue behind it for the full duration of the call.
  Any blocking operation the demo app performs must be wrapped in a runtime-provided
  primitive that checks the safepoint queue before and after the blocking call — this is
  the concrete shape the OPEN concurrency model needs to eventually generalize.
- The REPL dock is the first place a person, not just the UI, controls `source` text
  directly. The M1 gate (still active for definitions in M2, per OUT) has to be re-verified
  against genuinely free-form input, not just the fixed shapes M1's canned UI ever
  produced — a classifier that only worked because the UI never generated an edge case is
  not the same as one that's actually sound against arbitrary text.

## M3 — Live code change

**Goal:** redefine a method body on a class with live instances, while the program keeps
running, with the type checker refusing bad edits before they ever reach the interpreter.

**IN**
- **The eval gate is lifted for definitions, the one restriction M1/M2 kept in place.**
  Live code change is not a new wire feature — it's `eval` of a `source` that happens to be
  a definition instead of an expression, over the exact same `{"id", "source"}` →
  `{"id", "value"}`/`{"id", "error"}` channel M1 already speaks: `{"id": 7, "source": "fn
  summarize() -> String { ... }"}` submitted against `Agent`. Method-body swap: given
  source for a new body of `Agent.summarize`, recompile just that method, typecheck its
  signature against the **existing, already-loaded** class's field table and method table
  (param types, return type must match exactly — no signature change in M3). If it doesn't
  typecheck, the swap is refused — the response is `{"id": 7, "error": "<exact type
  error>"}` — and the old bytecode keeps running; nothing about the running program
  changes. If it typechecks, the method table's bytecode pointer for `Agent.summarize` is
  swapped atomically at the next safepoint (the same safepoint-queue mechanism M2 built for
  method-call eval), and the response is `{"id": 7, "value": "method Agent.summarize
  replaced — next call uses new code"}`.
- **In-flight-call semantics, pinned for the PoC**: a call to `summarize()` already
  executing when the swap happens finishes on the *old* bytecode — the swap affects only
  calls made after the swap point. This is the only sane cut without a much deeper
  per-frame versioning design; `03-live-semantics.md` owns whether this is the permanent
  answer or a first cut.
- **Shape change, restricted to additive-with-default**: a class may gain a new field
  only if the new field declaration carries a mandatory default-value expression. On
  accepting the redefinition, the runtime walks the class's arena once and writes the
  default into the new field's slot for every existing live instance (an O(live instances)
  one-time pass), then all future constructions include it normally.
- Editing surface: either an editor-saved file the runtime watches, or a code pane in the
  viewer that sends the new method source over the wire — DECISIONS.md doesn't pin which,
  M3 builds file-watch first (simpler, no new UI) and the viewer code pane as a thin
  wrapper around the same "submit new method source" runtime entry point.

**OUT**
- **M3 ships a strict subset of `03-live-semantics.md`'s field-change design, explicitly,
  not the full mechanism.** `03-live-semantics.md` treats field removal as always
  representable (never rejected) via a user-supplied migration function plus
  per-instance quarantine on migration failure, and gets field rename "for free" as a
  remove+add bridged by that same migration function. None of that ships in M3 — only
  the additive-field-with-a-static-default case above is built. This is a deliberate PoC
  scope cut against `03-live-semantics.md`'s fuller design, not an oversight, and no
  milestone after M3 revisits it; migration functions and per-instance quarantine are
  deferred past M4.
- **Field removal or retype while live instances exist is not supported in the PoC and
  hard-errors** (the migration-function + quarantine mechanism `03-live-semantics.md`
  describes for this exact case is the deferred fuller design, not something M3 builds):
  `redefine Agent: field 'status' removed/retyped while 3 live instances exist — not
  supported in this build, restart the process to apply this change.` This is a real,
  loud error, never a silent drop of the field or a null-fill of a retyped value.
- Renaming a field (semantically remove+add) hard-errors the same way — no migration
  function to bridge old/new values exists in M3.
- Changing a class's generic parameters, adding/removing a class entirely while live,
  changing a method's parameter or return type — all hard-error, all out of scope.

**Acceptance demo:** the `00-vision.md` 3:30 beat exactly. Edit `Agent.summarize`'s prompt-
building line to something that doesn't typecheck (e.g., concatenate a `String` with a
`Task`), save — the viewer shows the type error, the running program is unaffected, old
`summarize` still runs. Fix it, save again — `method Agent.summarize replaced — next call
uses new code`. The reviewer's next summary in the terminal is visibly different, with
zero restart and all prior `Message` instances intact.

**Risk notes**
- The in-flight-call semantics decision (old code finishes, new code only for future
  calls) has a sharp edge: if a call is recursive or long-running, a caller can observe
  "the method changed" only slowly, which is correct but can look like a bug in a live
  demo if not narrated. Rehearse the timing so the swap confirmation message lands before
  the next call, not mid-call.
- The additive-field-with-default pass touches every live instance of that class
  synchronously — fine at demo scale (dozens of instances), a real scaling problem the
  moment arenas hold thousands; note it and move on, don't solve it in the PoC.
- Swapping a method-table pointer while another thread might be *reading* the old
  pointer to make a call is exactly the kind of race the M2 safepoint mechanism exists to
  prevent — M3 must do the swap only inside the same safepoint-serviced section M2 built,
  not bolt on a second, different synchronization scheme.

## M4 — Demo app + polish

**Goal:** the actual agent-TUI PoC application, and enough visual/UX polish that the full
`00-vision.md` five-minute script runs live, unscripted, without apology.

**IN**
- `agents.oo`: the demo program. Classes `Agent`, `Conversation`, `Message`, `Tool`,
  `ToolCall`, `Task` (per `DECISIONS.md` #10). A cooperative turn scheduler (M2's stopgap)
  drives 3 named agents (`researcher`, `coder`, `reviewer`) against a small shared task
  list, each turn appending `Message`s and occasionally a `ToolCall`.
- **`Agent.tools: List<Tool>` via `01-language.md` §2.3's enum-dispatch fallback**, not
  real interfaces (no milestone in this doc scopes `interface`/`implements` — see M0's OUT
  section above). Concretely: `enum Tool { Shell(ShellTool), Search(SearchTool) }`, and a
  tool call dispatches with `match tool { Shell(t) -> t.call(args), Search(t) ->
  t.call(args) }`. `Agent.tools` holds a heterogeneous mix of concrete tool classes by
  wrapping each in the `Tool` enum; `DECISIONS.md` #10's "swap a tool" demo beat is
  "replace the `Tool` value in the list with a different enum case." The cost this accepts,
  named explicitly: adding a genuinely new kind of tool means editing the `Tool` enum
  itself, not writing an independent class that opts in — acceptable for a PoC with two
  tool kinds. Real `interface`/`implements` support (`01-language.md` §2.3's default
  proposal) remains the answer if the language grows past the PoC; nothing here blocks it.
- Terminal rendering: a minimal ANSI redraw loop (cursor positioning + periodic repaint of
  status lines and a scrolling log pane) — not a general TUI toolkit, not a curses binding.
  Scope it to exactly what the demo script needs: 3 status lines, one scrolling pane.
  Control bytes (cursor-positioning escapes, ESC/`0x1B`) are constructed via the `\xHH`
  raw-byte string escape now specified in `01-language.md` §3.1.
- A `FakeModel`/`ScriptedModel` class standing in for real LLM calls, so the demo is
  deterministic and doesn't depend on network access or API cost during rehearsal or the
  live take. It must be an honestly-named, ordinary entity — visible in the viewer's type
  list as `ScriptedModel`, not disguised as a real model integration and not a hidden
  shim inside `Agent`. If a real model call is feasible without jeopardizing determinism,
  prefer it; the fake is a deliberate, visible fixture either way, never a silent stand-in.
- Actual visual design pass on the viewer UI: typography, layout, color, motion for the
  "count climbing" and "field changed" moments — per `DECISIONS.md` #11, this has to look
  designed, not like unstyled HTML. Budget real time for this; it is not a free side
  effect of M1–M3's plumbing.
- Full run-through rehearsal of the exact `00-vision.md` script, twice back to back,
  without restarting the process or fudging state between runs.

**OUT**
- Multi-file oo-lang programs, a package manager, anything not needed to make one demo
  file work.
- Any demo scenario other than the agent TUI.
- Cross-platform terminal support beyond the primary dev machine (macOS Terminal).

**Acceptance demo:** the whole `00-vision.md` script, live, twice in a row, cold start to
close, no slides, nothing faked except the explicitly-visible `ScriptedModel`.

**Risk notes**
- "Looks genuinely good" is the easiest thing in this whole plan to underscope — it is a
  real design task, not a CSS pass at the end. Treat it as its own work item with its own
  time budget, and pull in outside design help (per the project brief) rather than
  ship default browser chrome.
- A demo that depends on hitting exact timestamps is fragile under real presenting
  conditions (questions, latency, fat fingers). The acceptance bar is "correct and
  reproducible if narrated slower," not "exactly 5:00 on the nose."
- `ScriptedModel` determinism has to survive re-running the whole script twice without a
  process restart — if its script position isn't reset or is stateful in a way that leaks
  across "runs" within the same process, the second take will visibly diverge from the
  first. Design it to be replay-safe from the start.

## Demo-beat → milestone dependency

| `00-vision.md` beat | Needs |
|---|---|
| 0:00 — start program, TUI renders | M0 (interpreter, classes/methods run) + M4 (the actual demo app and its terminal rendering) |
| 0:30 — open viewer, type index with live counts | M1 (embedded server, eval channel, canned summary eval, client re-eval interval) |
| 1:30 — drill into instances, search by field, watch messages append mid-run | M1 (instance table, query, instance detail, all as canned eval expressions re-sent on an interval) |
| 2:30 — invoke `pause()`, TUI reflects it; eval `setPriority(1)` | M2 (mutating eval, REPL dock, safepoint mechanism) |
| 3:30 — live-edit `summarize()`, typecheck-refuse then accept | M3 (method-body swap, in-flight semantics, redefinition typechecking) |
| 4:30 — close browser tab, reopen, `Ctrl-C` the TUI | M1 (server survives client disconnect) + M0 (ordinary process semantics, no image) |

## What we fake — and how each one fails loudly

Per the hard rule: nothing in the PoC silently returns a placeholder or a default. Every
boundary below is a real, named, loud failure, not a stub that quietly does the wrong
thing.

1. **No GC (M0–M2).** Arenas only grow. Hitting a configured max slab size before real GC
   exists is `OutOfArenaSpace: Agent arena full at 100000 slots — GC not implemented in
   this build, restart the process.` Never wraps around, never overwrites a live slot.
2. **No interfaces/traits.** `interface`/`trait`/`impl` keywords, if written, are a hard
   parse/compile error: `interfaces are not implemented in this build`. Never silently
   parsed and ignored. This is a real scope cut, not a silent gap: the one place the PoC
   would otherwise reach for interface-style polymorphism (`Agent.tools`) is met instead
   by `enum Tool { ... }` dispatch (M4), per `01-language.md` §2.3's documented fallback —
   no code anywhere needs `interface` to compile for the PoC to run.
3. **Generics are unconstrained (no bounds).** Bound syntax, if written (`<T: Comparable>`),
   hard compile-errors rather than being silently accepted and unenforced.
4. **Field removal/retype under live redefinition (M3).** Hard error naming the class,
   the field, and the live-instance count, refusing the redefinition outright — see M3's
   OUT section for the exact message shape. Note this is narrower than
   `03-live-semantics.md`'s full design (which makes field removal always representable
   via a migration function and per-instance quarantine, never rejected) — the PoC
   deliberately ships only the additive-default subset of that design; the fuller
   mechanism is deferred past M4, not abandoned.
5. **Non-primitive invoke/eval arguments from the viewer (M2).** The wire responds
   `{"id": <n>, "error": "unsupported arg type Conversation for invoke; only Int/
   Float/Bool/String supported in this build"}`. Never coerced, never null-filled.
6. **Real OS-thread concurrency / `async`/`await` (M2 stopgap).** If demo or future source
   uses a spawn/async primitive beyond the cooperative turn-queue, it hard-errors `spawn
   not implemented in this build` — never silently degrades to a no-op.
7. **`ScriptedModel` in the demo app (M4).** Not a language-level stub at all — an
   honestly-named, viewer-visible entity type. The rule this satisfies isn't "hard-error,"
   it's "never disguised": anyone browsing the running demo in the viewer sees
   `ScriptedModel`, not something dressed up to look like a real API integration.
8. **Invoke stuck behind a non-yielding blocking call (M2 risk, not a normal path).** If a
   safepoint isn't reached within a timeout, the pending `eval` reports
   `{"id": <n>, "error": "invoke timed out waiting for interpreter safepoint"}`
   rather than hanging the viewer with no feedback forever.
9. **Mutating eval before its milestone (M1/M2).** M1 hard-refuses any `source` that would
   call a mutating method or install a definition; M2 lifts that only for method calls,
   still refusing definitions. Both refusals are the same shape: `{"id": <n>, "error":
   "mutating eval (method calls, definitions) is disabled until M2 — M1 is read-only"}` in
   M1, `{"id": <n>, "error": "live code redefinition is disabled until M3"}` in M2. Never a
   silent no-op, never a partially-applied definition.

## OPEN items this document depends on but does not resolve

Per `DECISIONS.md`, these remain open; this document proposes concrete, load-bearing
stopgaps for the PoC without claiming to answer them for the language:

- **Interfaces/traits** — the `interface`/`implements` mechanism itself
  (`01-language.md` §2.3's default proposal) is left fully deferred past the PoC. This
  does *not* mean the PoC needs zero polymorphism: `Agent.tools: List<Tool>` must hold a
  heterogeneous mix of concrete tool classes (`01-language.md` §2.3), so M4 adopts
  `01-language.md` §2.3's documented enum-dispatch fallback (`enum Tool { Shell(ShellTool),
  Search(SearchTool) }`) as the concrete, load-bearing answer for that field — see M4's IN
  list. Real interfaces remain the target design if the language grows past the PoC.
- **Concurrency model** — PoC ships a cooperative single-threaded turn scheduler (M2) as
  a demo-scoped scheduling trick, explicitly not proposed as the language's answer.
- **Viewer invoke/eval safety** — PoC proposes a safepoint-checked single-interpreter-
  thread mechanism (M2); `02-runtime.md`/`03-live-semantics.md` own whether this is final.
- **Name** — this document uses "oo-lang" throughout as the working directory name only,
  per `00-vision.md`'s open naming discussion.
