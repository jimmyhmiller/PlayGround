# 05 — Milestones

This is the cut-line document. Six milestones, M0 → M5, each one a runnable thing you can
point at and say "this now works," ending at the demo in `00-vision.md`. Nothing in here is
aspirational — if a capability isn't in some milestone's IN list, it does not exist for the
PoC, and the thing that would normally use it either doesn't exist yet or hard-errors. See
"What we fake" at the bottom for the exhaustive list of PoC-era stubs and exactly how each
one fails loudly.

The ordering is load-bearing, and it changed once real threads stopped being a stopgap and
became a day-one language feature (`DECISIONS.md` #4b). **Threads are their own milestone,
M1, landing right after M0's single-threaded interpreter is solid** — not bolted onto the
end near the demo app. Reasoning: M3 (interaction/invoke) and M4 (live code change) both need
a mechanism that can pause *every* mutator thread at once to run an eval or swap a method
table safely. Building that mechanism once, in M1, against a small, well-understood test
program is far cheaper than inventing a single-thread-only stopgap in M3 and then reworking
it into a real multi-thread stop-the-world (STW) pauser later — and the old stopgap (a
cooperative single-threaded turn scheduler) is exactly the throwaway this ordering avoids.
So: M0 gives every later milestone per-type arenas and (now) interfaces to build on; M1 gives
everything after it real OS threads plus a generalized STW safepoint that can park all of
them; M2 (viewer) needs M0's arenas and, in practice, runs against a runtime that already has
M1's thread-safe magazines; M3 (invoke) cannot exist before M2 gives it a channel to send
requests on, *and* cannot be built honestly before M1 gives it real cross-thread STW to reuse;
M4 (live change) cannot exist before M3 gives a safe way to mutate a running interpreter, and
leans on M1's STW generalization directly. Nothing here is parallelizable across
milestones — it's a strict chain.

## M0 — Minimal interpreter

**Goal:** `scry run foo.scry` compiles and runs a CLI program with classes, methods,
interfaces, and generics — and every object it allocates already lives in a per-type arena,
before any GC exists to sweep them.

**IN**
- Lexer/parser for the pinned surface syntax: `class`, fields with type annotations, `fn`
  methods, `self`, top-level `fn main()`, `if`/`else`, `while`, `for`-over-`List<T>`,
  `enum` (with payloads) and `match` (per `01-language.md` §1.3 — pulled into M0, not left
  for later, because status-shaped types like `AgentStatus` are needed starting M3's
  demo), **`interface` declarations and explicit `class ... implements InterfaceName`
  clauses** (Jimmy ruling, `DECISIONS.md` #4 — Java-style interfaces are IN the language
  and IN the PoC, not a post-PoC nice-to-have), arithmetic/comparison/boolean operators,
  string interpolation, single-file programs.
  ```
  interface Greeter {
    fn greet() -> String
  }
  class Friendly implements Greeter {
    name: String
    fn greet() -> String { "Hi, I'm " + self.name }
  }
  ```
- Primitive types: `Int`, `Float`, `Bool`, `String`. One built-in generic: `List<T>`
  (push/get/len/iterate). No `Map`/`Set` yet unless a demo class needs one — add narrowly,
  don't build a stdlib.
- Nominal static type checker: field types, method signatures, call-site arg/return
  checking, `self` typed to the enclosing class. No inheritance among classes means
  concrete method resolution is still a flat lookup on one class — no vtables, no MRO, for
  a plain `class`. **Interfaces get real conformance checking**: a class declaring
  `implements Foo` must provide every one of `Foo`'s method signatures (name, params,
  return type) exactly, or it's a compile error naming the missing method; an interface
  type is usable anywhere a type annotation is (field, parameter, return type); assigning
  or passing a concrete instance into an interface-typed slot is checked at compile time
  against that class's declared `implements` list — there is no structural/duck-typed
  fallback, conformance is nominal only. A class may implement more than one interface. An
  interface may **not** extend another interface in this PoC (no interface inheritance) —
  a deliberate scope cut in the same spirit as "no inheritance of any kind" for classes,
  not an `OPEN` question. **Default method bodies on interfaces are out** (`DECISIONS.md`
  lists this `OPEN`, leaning no for the PoC) — an interface method declares a signature
  only; if source gives one a body, it hard-errors (see "What we fake"). Generics are
  checked structurally per call site. Per `01-language.md` §2.4, `02-runtime.md` §5, and
  `DECISIONS.md` #6, **generic classes are monomorphized from day one**: `Inventory<Tool>`
  and `Inventory<Task>` are distinct compiled types with distinct arenas, exactly like any
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
  can identify what it's holding), **and with a distinct interface-typed dispatch path**
  (`02-runtime.md` §5): an interface-typed slot is **the same single-word raw pointer** a
  class-typed slot is — no second word, no fat value — because every instance's own header
  already carries a `type-id` (the same identity used by `type_id/slab_index/generation`
  below), so the concrete type behind an interface reference is recovered with one load off
  the pointee itself rather than carried alongside the pointer. Calling a method through an
  interface-typed value (`OP_CALL_VIRTUAL`) loads that `type-id`, indexes into the concrete
  class's compile-time-built itable for the interface in question, and calls through it;
  calling a method on a concretely-typed value stays the flat static lookup from before.
  Deciding this dispatch path now matters because retrofitting it onto every call site after
  the fact is much more expensive than building it in from M0's first instruction set.
- **Per-type arenas from the first allocation.** Every `class Foo` gets its own slab the
  first time a `Foo` is constructed. Layout is fixed-size records (the class's field
  layout is static — no dynamic shape) packed contiguously, magazine/slab style, so
  "every live `Foo`" is a linear walk with no pointer chasing through a general heap. An
  interface value is a *view* over some concrete class's arena slot — interfaces do not
  get their own arena, there is nothing to enumerate at the interface level that isn't
  already enumerable per concrete type. Each slot gets a stable identity:
  `(type_id, slab_index, generation)` — this triple *is* the `Agent#7`-style id the
  viewer will use later, and it must be designed now, because retrofitting identity onto
  objects after the fact means walking every existing reference to add it.
- CLI: `scry run foo.scry` — parses, typechecks, compiles, runs to completion, ordinary
  stdout/stderr/exit code. `scry run --dump-arenas foo.scry` — internal debug command that
  prints, at exit, each type's slab occupancy (a sanity check for M0 that doesn't need a
  UI to verify the arena design works).

**OUT**
- No garbage collection. Slabs only grow. See "What we fake" — this has a hard, loud
  failure mode, not a silent one. (No milestone in this document adds GC — it stays out
  for the whole PoC.)
- No operator overloading, no closures/lambdas, no exceptions (use a `Result`-shaped
  return or crash — exceptions are not a PoC requirement).
- No default methods on interfaces, no interface-extends-interface (see IN above).
- No modules or multi-file programs. No package manager, no imports beyond the one file.
- **No concurrency of any kind yet — straight-line single-threaded execution, top to
  bottom.** Real threads are M1, deliberately not bundled into M0: M0's job is proving the
  single-threaded interpreter, typechecker, and arena model are solid before multiplying
  everything by N mutator threads.
- No viewer, no server, no redefinition. Those are M2–M4.

**Acceptance demo:** write a ~70-line `.scry` file with 3 classes (one generic, e.g.
`Inventory<T>`) plus one small interface implemented by two of them (e.g. `interface
Greeter` implemented by two classes, stored in a `List<Greeter>` and called through that
list to prove itable dispatch — not switch-like — picks the right method per concrete
instance), fields, methods that call each other and mutate `self`, a loop building a
`List`. `scry run foo.scry` prints the expected output, including each `Greeter` in the
list greeting differently despite being called through one uniform interface-typed loop
variable. `scry run --dump-arenas foo.scry` shows one slab per *concrete* class with the
expected occupancy count — proving instances of different classes never share a slab, and
that the interface itself never got a slab (there's nothing to allocate).

**Risk notes**
- Deciding the object-identity scheme (`type_id/slab_index/generation`) *is* deciding a big
  chunk of the eventual wire protocol's `Agent#7` id format — get this wrong here and every
  later milestone inherits the mistake. Nail it in M0, not "later, in the viewer."
- Monomorphizing generic *classes* (`Inventory<T>` and every instantiation of it) is not
  optional here, despite multiplying codegen paths right when the compiler is least
  mature — `DECISIONS.md` #6's per-type-arena model has no other way to make "enumerate
  every live `Inventory<Tool>`" cheap and well-defined, and M0's own acceptance demo below
  depends on it. Boxed-uniform is real, but scoped to generic *functions* only, which never
  touch an arena.
- The bytecode format has to carry enough static type information for the arena/GC code to
  walk a stack frame or a record and know which slots are class-typed pointers, raw
  ints/floats, or interface-typed pointers (a single-word pointer, same as a class-typed
  slot, per `02-runtime.md` §5) — this needs to be right from the first instruction set
  revision, because M2 onward assumes the runtime can always answer "what type is this
  value, and if it's an interface value, what concrete type backs it" with one header load
  off the pointee, without a runtime tag on the reference itself.
- The itable layout decided here is what M5's `Agent.tools: List<Tool>` calling convention
  rests on, and what M2's viewer must be able to serialize (a `Tool`-typed field has to
  show both its declared interface type and the concrete backing type, e.g. `{"ref":
  "ShellTool#1", "type": "ShellTool", "as": "Tool"}`) — get the representation wrong here
  and both of those later milestones inherit the mistake, same as the identity scheme risk
  above.

## M1 — Real threads + stop-the-world safepoints

**Goal:** the runtime can spawn real OS threads that share the heap, allocate safely into
per-type arenas concurrently, and can be stopped, all of them, at once, for coordinated
work — the generalized mechanism M3's invoke and M4's live redefinition both depend on,
built once here instead of twice (a single-thread stopgap, then a real one).

**IN**
- **Real OS thread spawn/join as a language primitive** (Coil's `lib/thread.coil` backs
  this — real pthreads, not green threads). Exact surface is `OPEN` per `DECISIONS.md`
  ("thread API surface"); this milestone proposes the minimal shape `01-language.md` §5.1
  already pins, not a new one:
  ```
  interface Runnable {
    fn run() -> Void
  }
  object Thread {
    fn spawn(task: Runnable) -> ThreadHandle
  }
  class ThreadHandle {
    fn join() -> Void
  }
  ```
  `Thread.spawn` takes a `Runnable` rather than a bare function value — the PoC has no
  closures/lambdas anywhere (M0's OUT list), so "the work this thread runs, plus whatever
  context it needs" is an ordinary class that `implements Runnable` and captures that
  context as fields, e.g. `Thread.spawn(AgentWorker(agent: researcher))`. This is
  sufficient for the demo (each agent's turn loop is a `Runnable` closed over its own
  `Agent` field, not a parameterized spawn).
- **Thread-safe per-type arena allocation.** Each OS thread gets its own magazine — a
  thread-local chunk of a type's slab it can bump-allocate from without taking a lock on
  the common path. Only refilling an exhausted magazine from the shared slab takes a
  lock/atomic (Coil's `lib/atomic.coil`). This is `DECISIONS.md` #4b's "thread-safe
  per-type arena allocation (per-thread magazines)" made concrete.
- **A generalized stop-the-world safepoint.** Every thread executes a safepoint check at
  loop back-edges and call sites — the same instruction-level hook M0's bytecode already
  needed for a future GC, now genuinely load-bearing. When anything requests a stop (this
  milestone's own tests; later, M3's eval and M4's redefinition), a global flag is raised
  and every thread parks at its next safepoint; the requester proceeds only once *every*
  thread has reported parked; releasing the flag resumes all of them together. This
  replaces the old single-thread "safepoint queue" idea outright — with real concurrent
  mutators there is no single interpreter thread to queue requests against, so the
  mechanism has to coordinate N threads for real.
- **Minimal synchronization primitives exposed to the language.** At least a mutex,
  backed by `lib/atomic.coil`, enough for the demo app's shared task list
  (`DECISIONS.md`'s "thread API surface" `OPEN` item covers whether anything richer —
  channels, condition variables — is needed; this milestone proposes mutex-only for the
  PoC and flags the rest `OPEN`).
- Even though full GC stays out of the PoC (see M0's OUT), the STW infrastructure built
  here is exactly what an eventual GC would need — same mechanism, just not wired to a
  collector. Building it honestly now means a real GC is a smaller, additive step later,
  not a rewrite.

**OUT**
- Still no garbage collection — arenas only grow, unaffected by this milestone.
- No viewer, no eval channel yet (M2/M3).
- **No `async`/`await` surface syntax** — `DECISIONS.md` #4b is explicit that async/await
  comes later, post-PoC, and is not to be confused with real thread spawn/join, which is
  in the language now. A hard-error stub for `async`/`await`, if written, is in "What we
  fake" below.
- No thread cancellation, no thread-local exception handling, no thread pools/executors —
  raw spawn/join and a mutex is the whole surface for the PoC.
- No partial/read-only safepoint optimization — this milestone builds only the full,
  all-thread STW. Whether a read-only eval (M3) can someday get away with a cheaper
  partial safepoint is `DECISIONS.md`'s open "how the eval channel interleaves with
  running threads" question; M1 doesn't answer it, it only builds the one mechanism every
  later milestone can rely on unconditionally.

**Acceptance demo:** a small `.scry` program spawns N real OS threads, each independently
constructing instances of the same class in a loop (proving thread-safe magazine
allocation doesn't corrupt or lose slots — `--dump-arenas` shows the correct total
occupancy with no duplicated or missing slots) and each incrementing a shared counter
behind a mutex (proving the mutex primitive is real), then joins all N and prints a
correct final count. Separately, an internal test hook requests an STW pause while all N
threads are mid-loop and confirms, via an instrumentation counter, that every thread was
actually parked before the pause is considered acquired, and that all resume together —
proving the safepoint mechanism stops *everyone*, not just whichever thread asked.

**Risk notes**
- This is the highest-risk milestone in the plan — real data races are notoriously easy to
  pass in rehearsal and reappear live. Gate this milestone's acceptance on stress/TSan-style
  testing, not a single clean demo run (precedent: tallyc's spawn/join+Slice parallelism
  was TSan-gated for the same reason).
- GC-root correctness across multiple live stacks isn't exercised yet (no GC exists), but
  the identity scheme from M0 plus the per-thread magazines built here must already be
  shaped so a future GC could walk N thread stacks for roots without a redesign — decide
  the stack-walking contract's shape now even though nothing calls it yet.
- Interfaces from M0 interact with thread safety directly, but not as a torn-read problem:
  an interface-typed value is a single pointer, same as a class-typed one (`02-runtime.md`
  §5), so there's no second word to tear. The real hazard is publication ordering: another
  thread dispatching `OP_CALL_VIRTUAL` on that pointer loads `type-id` off the pointee's own
  header, so the header's `type-id` must be fully written *before* the instance's pointer is
  ever made visible to another thread (e.g. stored into a shared field or passed to
  `Thread.spawn`) — a construction-visibility ordering bug, not a torn-pointer one, and this
  milestone must rule that out, not assume it away.
- STW-for-everything is a blunt instrument. If M3's invoke ends up pausing every real
  thread for something as cheap as a single field read, that may be visibly laggy — flag
  this now as the concrete cost of the `OPEN` "read vs. definition safepoint" question, to
  be revisited in M3, not solved here.

## M2 — Viewer, read-only

**Goal:** point a browser at a running M0+M1 program and see its live objects — type list
with counts, searchable instance tables, instance detail — without touching or mutating
anything.

**IN**
- Runtime embeds a server (transport TBD by `02-runtime.md` §7's risk note — WebSocket or
  a simpler long-poll/SSE, that doc's call, not this one's), on by default, printing
  `viewer: http://localhost:7357` (or `--viewer=off` to disable). This is the *only* new
  runtime capability M2 adds beyond M0/M1 — everything else is read access to structures
  already built.
- **The wire protocol is the eval channel, in full, from day one:** request
  `{"id": <int>, "source": "<Scry source text>"}`, response `{"id": <int>, "value":
  ...}` or `{"id": <int>, "error": "..."}`. There is no `types`/`instances`/`instance` op
  set distinct from later milestones — M2 does not get its own wire shape that M3/M4 later
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
  stay flat and cheap. **An interface-typed field serializes as its concrete backing
  instance's ref, tagged with the declared interface name**, e.g. `{"ref": "ShellTool#1",
  "type": "ShellTool", "as": "Tool"}` — the viewer always has a real, concrete, clickable
  instance to link to; there is nothing "interface-shaped" living in the arena to
  serialize instead. Exact stdlib method names (`Type.summary()`, `.instances()`) are
  `01-language.md`'s call; the shape of the request/response envelope is what's pinned here.
- **M2's viewer UI only ever emits canned, UI-generated `source` text — there is no
  free-form input box yet (that's M3's REPL dock).** The type index sends a fixed
  summary expression, drilling into a type sends that type's `.instances()` expression,
  clicking a row sends that instance's own ref expression. The instance-search box compiles
  down to one of a tiny, fixed family of filter expressions: `field == literal`,
  `field != literal`, numeric `<`/`>`/`<=`/`>=`, string `contains`, no boolean combinators,
  no nested field paths — a search box, not a query planner, expressed as generated source
  text rather than a bespoke query op.
- Browser UI: type index (name, live count, sparkline/arrow if the count is climbing) →
  click into a searchable instance table (columns = fields) → click a row into an instance
  detail view (every field, typed, refs are links; methods listed with signatures but
  **not clickable yet** — that's M3).
- Freshness: the client re-sends the same canned `eval` request on a fixed short interval
  (e.g. every 250ms) while a view is open, and gets a fresh `value` back each time. This is
  the client re-asking, not the server noticing a mutation and pushing anything — there is
  no dirty-bit or write-barrier machinery in the interpreter, and none is needed, because
  nothing is ever sent that the client didn't just ask for. It looks identical to the
  audience.
- **M2's server-side eval gate — hard-enforced by the runtime, not a UI convention.**
  Because the wire channel already speaks full `eval`, and `eval` can run anything the
  language can express — including a mutating method call or a redefinition — M2 cannot
  rely on "the UI only ever sends read expressions" as its safety boundary; nothing stops
  a different client on the same socket from sending `Agent#7.resume()` or a new method
  body, and M2 has none of M3's invoke-marshaling or M4's typecheck-and-swap machinery
  built yet to run one safely. So the server classifies parsed `source` before running
  anything: a **definition** (`class`, `fn`, `enum`, `interface` at top level, or a
  redefinition of an existing method) or an expression whose evaluation would **call a
  mutating method** (anything of the shape `expr.method(args)` that isn't a plain field
  read the typechecker already knows is side-effect-free) is refused outright, with a hard
  error, and nothing runs:
  ```json
  → {"id": 4, "source": "Agent#7.resume()"}
  ← {"id": 4, "error": "mutating eval (method calls, definitions) is disabled until M3 — M2 is read-only"}
  ```
  Field reads, `.instances()`-style enumeration, and comparison/filter expressions are
  permitted; nothing that could mutate state or install new code is, regardless of what
  the UI happens to offer. This gate is exactly what M3 lifts (for method calls) and M4
  lifts (for definitions) — see those milestones' IN lists.
- **Instance enumeration must reckon with per-thread magazines (M1).** A magazine
  currently checked out to a mutator thread holds slots that are live but not yet folded
  back into the shared slab's bookkeeping. Reading a consistent count/list therefore needs
  either a brief, cheap use of M1's STW mechanism around the enumeration eval, or a defined
  tolerance for reading slightly-stale magazine state. This milestone proposes the former —
  reuse M1's STW for a momentary pause around `.instances()`/`Type.summary()` evals, rather
  than inventing a second, lock-free snapshot scheme — since the mechanism already exists
  by M2 and a read-only viewer pausing the world for a few microseconds is an acceptable
  cost; flagged here as a proposal `02-runtime.md` should confirm, not a settled answer.

**OUT**
- No invoke UI, no REPL dock, no free-form eval exposed to a person — M2's viewer only ever
  emits the canned expressions above, and the server-side gate refuses anything else even
  if it arrives some other way (see IN).
- Mutating eval (method calls, definitions) hard-refused at the server, per the gate above
  — not merely absent from the UI.
- No true server-initiated push of any kind — revisit only if client-driven re-eval proves
  visibly laggy in rehearsal.
- No auth/access control on the embedded server — PoC assumes localhost, trusted operator.

**Acceptance demo:** run the M0/M1 demo program (extended to spawn a couple of worker
threads, so the viewer is proven against a genuinely multi-threaded runtime, not just
main-thread allocations) with the viewer on, open the printed URL, see the type index with
counts, click a type, see a real instance table, type a query into the search box and see
it filter, click an instance and see every field typed with refs as links (including an
interface-typed field showing its concrete backing instance). Watch a count climb while the
program's own threads keep allocating concurrently, with no code in the program aware the
viewer exists. Separately, send a hand-crafted `{"id": 99, "source":
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
- The enumeration-under-concurrent-mutation issue above (magazines mid-flight on other
  threads) is a real correctness question, not a cosmetic one — a viewer that occasionally
  undercounts or shows a torn record because it read across a magazine boundary without
  pausing anything would be a genuine bug, not just an ugly one; confirm the STW-around-
  enumeration proposal actually holds before M2 is called done.

## M3 — Interaction

**Goal:** invoke methods and evaluate expressions against the live, multi-threaded process
from the viewer, and see the results reflected both in the viewer and in the program's own
output.

**IN**
- **M2's server-side eval gate is lifted for method calls.** The wire shape doesn't
  change — still `{"id", "source"}` → `{"id", "value"}`/`{"id", "error"}` — what changes is
  that the classifier M2 added now *permits* an expression whose evaluation calls a
  mutating method: `{"id": 5, "source": "Agent#7.resume()"}` → runtime resolves the method
  on the live instance, runs it, and returns `{"id": 5, "value": null}`. There is no
  separate `invoke` op distinct from `eval` — a click on `pause()` in the instance detail
  view is the viewer generating exactly this shape of `source` and sending it down the same
  channel M2 already speaks; the previously-canned UI now also emits method-call source
  text, not just reads.
- **The viewer grows a REPL dock**: a free-form text input that sends whatever the user
  types as `source`, verbatim, over the same channel — e.g. typing `Task#3.setPriority(1)`
  in the dock sends `{"id": 6, "source": "Task#3.setPriority(1)"}`. This is not a new
  wire feature; it's the UI finally exposing the channel with no canned-expression
  restriction, now that the server permits method calls. It is *not* a general expression
  language extension — no new bindings, no `let` — one expression per request, evaluated
  in the context of the running program's classes and live instances. The gate still
  refuses definitions (redefining a class, method, `enum`, or `interface`) at this point —
  see OUT — so the REPL dock's freedom is real but bounded, matching what the server will
  actually run.
- Argument marshaling: only primitive-typed literal arguments (`Int`, `Float`, `Bool`,
  `String`) written directly in the call's source text are accepted. A method whose
  signature includes a class-typed, interface-typed, or generic-typed parameter cannot be
  invoked from the viewer in the PoC — there's no handle-literal syntax for "pass me that
  other live instance" yet, and this milestone doesn't add one.
- **The invoke-safety mechanism reuses M1's stop-the-world, it does not reinvent one.**
  This is this milestone's actual hard problem, and the concrete (not final) answer to the
  `DECISIONS.md` `OPEN` question of how the viewer's eval channel interleaves with running
  threads: a pending `eval` request that needs to run anything beyond a lock-free read
  triggers M1's all-thread STW; once every mutator thread reports parked, the embedded
  server's own thread runs the evaluated expression directly (itself just more bytecode,
  subject to the same safepoints as everything else) — there is no longer "the"
  interpreter thread to hand work to, because real agent threads (M1, exercised fully once
  M5's demo app exists) are genuinely concurrent, so the server thread does the work itself
  while the world is stopped, then releases the pause. Whether cheap reads can eventually
  skip full STW for a narrower partial-safepoint is the piece of the `OPEN` question this
  milestone does not resolve — flagged, not decided, per `DECISIONS.md`.
- The old cooperative single-threaded turn-scheduler stopgap is gone — nothing in this
  milestone replaces it, because it's no longer needed. Real threads (M1) already make
  "agents take turns" genuinely true instead of simulated: each agent is a real OS thread
  doing its own thing, and STW is what makes stopping to inspect or mutate any one of them
  safe.

**OUT**
- **No live code redefinition.** The eval gate still hard-refuses a `source` that parses as
  a definition (`class`, `fn`/method redefinition, `enum`, `interface`), exactly as M2 did —
  M3 only lifts the gate for method calls, not for definitions. Lifting it for definitions,
  plus the typecheck-against-live-class and method-table-swap machinery that makes doing so
  safe, is M4's entire IN list.
- No object-typed, interface-typed, or generic-typed invoke arguments from the wire.
- No `async`/`await` surface syntax — still post-PoC, per `DECISIONS.md` #4b.

**Acceptance demo:** in the viewer's instance detail for `researcher`, click `pause()` →
in the terminal, the status line flips to `⏸ paused` within a frame, with `researcher`,
`coder`, and `reviewer` genuinely running on three separate OS threads the whole time (a
lightweight preview program is enough here; the full three-agent demo app is M5). Open the
REPL dock, type `Task#3.setPriority(1)` → the TUI's task list visibly re-sorts. Re-opening
(or the periodic re-eval of) `researcher`'s detail pane shows the updated `status` field —
there is no separate push event, the pane's next re-eval of `Agent#7` just returns the new
value.

**Risk notes**
- STW-per-invoke is a real latency cost now that "the world" means N genuinely concurrent
  OS threads, not one interpreter loop with a queue. Get the timing wrong and even a single
  field-read invoke visibly stalls every agent thread for its duration — the UX must
  surface "waiting for the interpreter to reach a safepoint" rather than hang silently, and
  after a timeout, error explicitly rather than spin forever unexplained.
- Blocking I/O without a safepoint check inside any agent thread's own turn now blocks the
  *entire* STW pause, not just one queued request — every other thread, plus the eval
  channel, waits behind it. Any blocking operation the demo app performs must be wrapped in
  a runtime-provided primitive that checks the safepoint queue before and after the
  blocking call; this is sharper than the old single-thread version of this risk, because
  the blast radius is now every thread in the process, not a private queue.
- The REPL dock is the first place a person, not just the UI, controls `source` text
  directly. The M2 gate (still active for definitions in M3, per OUT) has to be re-verified
  against genuinely free-form input, not just the fixed shapes M2's canned UI ever
  produced — a classifier that only worked because the UI never generated an edge case is
  not the same as one that's actually sound against arbitrary text.

## M4 — Live code change

**Goal:** redefine a method body on a class with live instances, while the program keeps
running on real threads, with the type checker refusing bad edits before they ever reach
the interpreter.

**IN**
- **The eval gate is lifted for definitions, the one restriction M2/M3 kept in place.**
  Live code change is not a new wire feature — it's `eval` of a `source` that happens to be
  a definition instead of an expression, over the exact same `{"id", "source"}` →
  `{"id", "value"}`/`{"id", "error"}` channel M2 already speaks: `{"id": 7, "source": "fn
  summarize() -> String { ... }"}` submitted against `Agent`. Method-body swap: given
  source for a new body of `Agent.summarize`, recompile just that method, typecheck its
  signature against the **existing, already-loaded** class's field table and method table
  (param types, return type must match exactly — no signature change in M4). If it doesn't
  typecheck, the swap is refused — the response is `{"id": 7, "error": "<exact type
  error>"}` — and the old bytecode keeps running; nothing about the running program
  changes. If it typechecks, the swap happens inside the same all-thread STW pause M1 built
  and M3 reused for invoke — every real agent thread is parked, the method table's
  bytecode pointer for `Agent.summarize` is swapped, then all threads resume — and the
  response is `{"id": 7, "value": "method Agent.summarize replaced — next call uses new
  code"}`. This is exactly the dependency named up top: M4 leans on M1's STW generalization
  directly, not on a bespoke redefinition-only synchronization scheme.
- **In-flight-call semantics, pinned for the PoC**: a call to `summarize()` already
  executing on some agent's thread when the swap happens finishes on the *old* bytecode —
  the swap affects only calls made after the swap point, on whichever thread makes them.
  This is the only sane cut without a much deeper per-frame versioning design;
  `03-live-semantics.md` owns whether this is the permanent answer or a first cut.
- **Shape change, restricted to additive-with-default**: a class may gain a new field
  only if the new field declaration carries a mandatory default-value expression. On
  accepting the redefinition, the runtime walks the class's arena once (during the same
  STW pause) and writes the default into the new field's slot for every existing live
  instance (an O(live instances) one-time pass), then all future constructions include it
  normally.
- Editing surface: either an editor-saved file the runtime watches, or a code pane in the
  viewer that sends the new method source over the wire — `DECISIONS.md` doesn't pin which,
  M4 builds file-watch first (simpler, no new UI) and the viewer code pane as a thin
  wrapper around the same "submit new method source" runtime entry point.

**OUT**
- **M4 ships a strict subset of `03-live-semantics.md`'s field-change design, explicitly,
  not the full mechanism.** `03-live-semantics.md` treats field removal as always
  representable (never rejected) via a user-supplied migration function plus
  per-instance quarantine on migration failure, and gets field rename "for free" as a
  remove+add bridged by that same migration function. None of that ships in M4 — only
  the additive-field-with-a-static-default case above is built. This is a deliberate PoC
  scope cut against `03-live-semantics.md`'s fuller design, not an oversight, and no
  milestone after M4 revisits it; migration functions and per-instance quarantine are
  deferred past M5.
- **Field removal or retype while live instances exist is not supported in the PoC and
  hard-errors** (the migration-function + quarantine mechanism `03-live-semantics.md`
  describes for this exact case is the deferred fuller design, not something M4 builds):
  `redefine Agent: field 'status' removed/retyped while 3 live instances exist — not
  supported in this build, restart the process to apply this change.` This is a real,
  loud error, never a silent drop of the field or a null-fill of a retyped value.
- Renaming a field (semantically remove+add) hard-errors the same way — no migration
  function to bridge old/new values exists in M4.
- Changing a class's generic parameters, adding/removing a class or interface entirely
  while live, changing a method's parameter or return type, changing a class's
  `implements` list — all hard-error, all out of scope.

**Acceptance demo:** the `00-vision.md` 3:30 beat exactly. Edit `Agent.summarize`'s prompt-
building line to something that doesn't typecheck (e.g., concatenate a `String` with a
`Task`), save — the viewer shows the type error, the running program (still with its agents
on real threads) is unaffected, old `summarize` still runs. Fix it, save again — `method
Agent.summarize replaced — next call uses new code`. The reviewer's next summary in the
terminal is visibly different, with zero restart and all prior `Message` instances intact.

**Risk notes**
- The in-flight-call semantics decision (old code finishes, new code only for future
  calls) has a sharp edge: if a call is recursive, long-running, or simply in progress on
  another agent's thread at the moment of the swap, a caller can observe "the method
  changed" only slowly, which is correct but can look like a bug in a live demo if not
  narrated. Rehearse the timing so the swap confirmation message lands before the next
  call, not mid-call.
- The additive-field-with-default pass touches every live instance of that class
  synchronously, during an STW pause — fine at demo scale (dozens of instances), a real
  scaling problem the moment arenas hold thousands and every other agent thread is frozen
  for the duration; note it and move on, don't solve it in the PoC.
- Swapping a method-table pointer while another *real* thread might be reading the old
  pointer to make a call is exactly the kind of race M1's STW mechanism exists to prevent —
  this is no longer a hypothetical, it's a genuine multi-thread hazard with real agent
  threads running. M4 must do the swap only inside the same STW-serviced section M1 built
  and M3 already reused, not bolt on a second, different synchronization scheme.

## M5 — Demo app + polish

**Goal:** the actual agent-TUI PoC application, and enough visual/UX polish that the full
`00-vision.md` five-minute script runs live, unscripted, without apology.

**IN**
- `agents.scry`: the demo program. Classes `Agent`, `Conversation`, `Message`, `Task`, plus
  an `interface Tool` implemented by `ShellTool` and `SearchTool` (per `DECISIONS.md` #10,
  #2). `AgentStatus` stays an `enum` (`DECISIONS.md` #4 — enums remain a language feature,
  they're just not the polymorphism story anymore).
- **Three named agents (`researcher`, `coder`, `reviewer`) each run on their own real OS
  thread** (M1's `Thread.spawn`), against a small shared task list guarded by M1's mutex
  primitive, each thread appending `Message`s and occasionally a `ToolCall` independently
  and concurrently. This is `DECISIONS.md` #4b and #10 made literal — the cooperative
  turn-scheduler stopgap that used to drive this demo is gone; three real threads produce
  the "agents working at once" beat honestly instead of simulating it.
- **`Agent.tools: List<Tool>` via real interfaces**, superseding the old enum-dispatch
  fallback entirely (`DECISIONS.md` #2 is explicit that this supersedes it):
  ```
  interface Tool {
    fn call(args: String) -> String
  }
  class ShellTool implements Tool { fn call(args: String) -> String { ... } }
  class SearchTool implements Tool { fn call(args: String) -> String { ... } }
  ```
  `Agent.tools` holds a genuinely heterogeneous `List<Tool>`; a tool call is just
  `tool.call(args)`, dispatched through the itable M0 built — no `match` on a wrapper enum.
  `DECISIONS.md` #10's "swap a tool" demo beat is "replace the `Tool` value in the list
  with a different concrete implementer," and — unlike the old enum fallback — adding a
  genuinely new kind of tool is now an independent class that `implements Tool`, with no
  edit to any existing type required. The cost the old fallback accepted (new tool kind =
  edit the `enum`) is gone.
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
  effect of M2–M4's plumbing.
- Full run-through rehearsal of the exact `00-vision.md` script, twice back to back,
  without restarting the process or fudging state between runs — with three genuinely
  concurrent agent threads running the whole time, this is also the first real end-to-end
  exercise of the threading design under live, unscripted conditions.

**OUT**
- Multi-file Scry programs, a package manager, anything not needed to make one demo
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
  first. Design it to be replay-safe from the start — and now that replay-safety has to
  hold across three genuinely concurrent OS threads instead of a single cooperative
  scheduler's deterministic turn order, thread-interleaving nondeterminism is a second,
  independent way the second take could diverge from the first. Rehearse with repeated
  runs (and ideally thread-sanitizer instrumentation) specifically hunting for
  order-dependent flakiness before the live take, not just script-position bugs.

## Demo-beat → milestone dependency

| `00-vision.md` beat | Needs |
|---|---|
| 0:00 — start program, TUI renders | M0 (interpreter, classes/methods/interfaces run) + M1 (real threads — each agent on its own OS thread from the first frame) + M5 (the actual demo app and its terminal rendering) |
| 0:30 — open viewer, type index with live counts | M2 (embedded server, eval channel, canned summary eval, client re-eval interval) |
| 1:30 — drill into instances, search by field, watch messages append mid-run | M2 (instance table, query, instance detail, all as canned eval expressions re-sent on an interval) |
| 2:30 — invoke `pause()`, TUI reflects it; eval `setPriority(1)` | M3 (mutating eval, REPL dock) + M1 (the all-thread STW pause invoke actually runs inside) |
| 3:30 — live-edit `summarize()`, typecheck-refuse then accept | M4 (method-body swap, in-flight semantics, redefinition typechecking) + M1 (STW pause the swap happens inside) |
| 4:30 — close browser tab, reopen, `Ctrl-C` the TUI | M2 (server survives client disconnect) + M0 (ordinary process semantics, no image) |

## What we fake — and how each one fails loudly

Per the hard rule: nothing in the PoC silently returns a placeholder or a default. Every
boundary below is a real, named, loud failure, not a stub that quietly does the wrong
thing.

1. **No GC (whole PoC).** Arenas only grow, on every thread, for the life of the process.
   Hitting a configured max slab size before real GC exists is `OutOfArenaSpace: Agent
   arena full at 100000 slots — GC not implemented in this build, restart the process.`
   Never wraps around, never overwrites a live slot.
2. **Interface default method bodies.** Interfaces themselves are real (`DECISIONS.md` #2,
   #4 — nominal, `implements`, itable dispatch, all in M0). What's faked is only the
   `OPEN`, lean-no PoC restriction on *default* method bodies: if an `interface` declares a
   method with a body instead of a bare signature, it's a hard compile error,
   `default interface methods are not implemented in this build`, never silently parsed
   and ignored or silently treated as a required-override signature.
3. **Generics are unconstrained (no bounds).** Bound syntax, if written (`<T: Comparable>`),
   hard compile-errors rather than being silently accepted and unenforced.
4. **Field removal/retype under live redefinition (M4).** Hard error naming the class,
   the field, and the live-instance count, refusing the redefinition outright — see M4's
   OUT section for the exact message shape. Note this is narrower than
   `03-live-semantics.md`'s full design (which makes field removal always representable
   via a migration function and per-instance quarantine, never rejected) — the PoC
   deliberately ships only the additive-default subset of that design; the fuller
   mechanism is deferred past M5, not abandoned.
5. **Non-primitive invoke/eval arguments from the viewer (M3).** The wire responds
   `{"id": <n>, "error": "unsupported arg type Conversation for invoke; only Int/
   Float/Bool/String supported in this build"}`. Never coerced, never null-filled.
6. **`async`/`await` surface syntax.** Real thread spawn/join is genuinely implemented
   starting M1 — that is no longer a fake. What's still faked, per `DECISIONS.md` #4b's
   "async stuff can come later," is any `async`/`await` keyword or primitive: if written,
   it hard-errors `async/await is not implemented in this build — use Thread.spawn/join`,
   never silently degrading to a no-op or silently aliasing to a thread spawn.
7. **`ScriptedModel` in the demo app (M5).** Not a language-level stub at all — an
   honestly-named, viewer-visible entity type. The rule this satisfies isn't "hard-error,"
   it's "never disguised": anyone browsing the running demo in the viewer sees
   `ScriptedModel`, not something dressed up to look like a real API integration.
8. **Invoke stuck behind a non-yielding blocking call (M3 risk, not a normal path).** If a
   safepoint isn't reached within a timeout on *any* thread, the pending `eval` reports
   `{"id": <n>, "error": "invoke timed out waiting for interpreter safepoint"}`
   rather than hanging the viewer with no feedback forever — this is now a whole-process
   stall (every real thread waits on STW), not just one queued request, so the timeout and
   error message matter even more than they did under the old single-thread design.
9. **Mutating eval before its milestone (M2/M3).** M2 hard-refuses any `source` that would
   call a mutating method or install a definition; M3 lifts that only for method calls,
   still refusing definitions. Both refusals are the same shape: `{"id": <n>, "error":
   "mutating eval (method calls, definitions) is disabled until M3 — M2 is read-only"}` in
   M2, `{"id": <n>, "error": "live code redefinition is disabled until M4"}` in M3. Never a
   silent no-op, never a partially-applied definition.

## OPEN items this document depends on but does not resolve

Per `DECISIONS.md`, these remain open; this document proposes concrete, load-bearing
stopgaps for the PoC without claiming to answer them for the language:

- **Thread API surface** — M1 proposes `Thread.spawn(task: Runnable) -> ThreadHandle` /
  `.join()` plus a mutex as the whole PoC surface (per `01-language.md` §5.1). Whether the
  language needs channels, condition variables, atomics as
  a first-class type, or a richer spawn signature (arguments, return values) is left open;
  nothing past the PoC is blocked by this minimal surface, but nothing here claims it's the
  final answer either.
- **Viewer invoke/eval safety, read vs. definition safepoints** — M1 builds one mechanism
  (full, all-thread STW) and M3/M4 both reuse it unconditionally. Whether read-only evals
  could eventually use a cheaper partial safepoint instead of pausing every thread (per
  `DECISIONS.md`'s framing: "evals that read can likely run at a partial safepoint;
  definition evals need full stop-the-world") is not resolved here — `02-runtime.md`/
  `03-live-semantics.md` own whether this is worth building for the PoC or stays a future
  optimization.
- **Interface default methods** — `DECISIONS.md` #4 leans no for the PoC; M0 ships nominal
  `implements` and itable dispatch without them, and a default method body hard-errors (see
  "What we fake" #2) rather than being silently accepted.
