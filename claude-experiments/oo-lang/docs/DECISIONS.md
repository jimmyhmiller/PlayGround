# Scry — Pinned Decisions

Source of truth for design docs. If a doc contradicts this file, the doc is wrong.
The raw project brief is in `../claude.md` — read it first; this file pins what it left open.

## Decided

1. **Thesis.** A statically-typed, OO-style language whose *runtime observability* is the
   product: humans understand what AI systems are doing at a high level, with the ability
   to drill into every detail. The live viewer is not tooling bolted on later — it is a
   day-one deliverable, co-equal with the language.
2. **Implementation language: Coil** (`~/Documents/Code/PlayGround/coil`, run `coil guide`).
   Not Rust. We need friction-free manual memory management for arenas/GC.
3. **Execution: bytecode interpreter** (clox port at `coil/apps/clox/` is precedent).
   Optimizing JIT is explicitly deferred; design bytecode so a JIT remains possible.
4. **Object model:** concrete entity types (classes with data members + methods),
   **no inheritance** of any kind. Generics yes. **Java-style interfaces are IN** (Jimmy
   ruling): nominal, explicit `implements`, interface types usable as field/param/return
   types, dynamic dispatch through them. Interfaces are the polymorphism story — the
   enum-dispatch fallback for `Agent.tools` is superseded. Subclassing remains absent.
   (Default methods on interfaces: OPEN, lean no for PoC.)
4b. **Concurrency: real OS threads, day one** (Jimmy ruling: "proper threads"). The
   language exposes thread spawn/join; the demo app's agents run on real threads.
   async/await surface syntax comes LATER (post-PoC) — not a cooperative turn-scheduler
   stopgap, actual threads. Runtime consequences are accepted: multi-threaded mutators
   over the shared heap, thread-safe per-type arena allocation (per-thread magazines),
   stop-the-world safepoints that park ALL threads (for GC, migrations, and definition
   evals). Coil has real pthreads + atomics (lib/thread.coil, lib/atomic.coil).
5. **Surface syntax: TS/Kotlin-ish braces.**
   ```
   class User {
     name: String
     age: Int
     fn greet() -> String { "Hello, " + self.name }
   }
   class Inventory<T> {
     items: List<T>
     fn add(item: T) { self.items.push(item) }
   }
   ```
6. **Memory: one arena (slab/magazine-style) per entity type.** Enumerating all live
   instances of a type must be cheap — this powers the viewer. GC is designed around
   per-type arenas (sweep = walk the slab; free lists per slab).
7. **Not Smalltalk:** no image, no baked-in state. You run an ordinary program (ordinary
   process, ordinary stdin/stdout); the viewer attaches as a lens over the live process.
8. **Viewer: browser UI over a live REPL — the ONLY wire operation is eval.** This is the
   brief's "maybe actually in REPL", taken literally (Jimmy ruling, emphatic). The running
   program embeds a server whose sole job is: receive an expression or definition in the
   language, evaluate it against the live heap (at a safepoint), return the serialized
   result. Every viewer pane is sugar over evals — type list is eval of instance counts,
   instance table is eval of `Agent.instances()`, detail is field reads, invoking a method
   is eval, live code change is eval of a definition (new generation), the REPL dock is
   just raw access to the same channel. Refresh = re-eval (on interval/focus/after-action).
   There is NO subscription/delta/dirty-tracking/push protocol, no seq numbers, no
   coalesced ticks. We are IN the running program inspecting the heap directly, not
   consuming a message feed. Result serialization must carry instance references as stable
   ids so the viewer can render them as clickable links.
9. **Live change is in scope:** redefine a method/class while the program runs; the design
   must say precisely how that interacts with static types and existing instances.
10. **PoC demo app: an agent TUI.** A terminal app that runs AI agents — entity types like
    `Agent`, `Conversation`, `Message`, `Tool`, `ToolCall`, `Task`, with
    `interface Tool` implemented by `ShellTool`/`SearchTool` and each agent on its own
    real thread. Demo moment: agents running in the terminal; in the viewer you browse
    live `Agent` instances, inspect a `Conversation` mid-run, call a method (pause an
    agent, edit a task, swap a tool), and the TUI reflects it instantly. Dogfoods the
    "language for AI" thesis.
11. **Demo-worthiness matters.** The viewer must look genuinely good, not programmer-art.

12. **Name: Scry** (Jimmy ruling). CLI `scry`, so `scry run agents.scry`. The directory
    stays `oo-lang`; every doc and example uses Scry. (File extension `.scry`: default,
    flag if a doc has a reason to prefer something shorter.)
13. **Multi-program portal — reverse-proxy hub** (Jimmy ruling). You launch ONE persistent
    portal at a fixed URL and sit at it. Each `scry run` binds an ephemeral port and
    registers {name, pid, port, schema, start-time} with the portal; programs pop up as
    cards when they start and disappear (or grey out) when they exit. The portal serves the
    viewer shell and REVERSE-PROXIES the eval channel to the right program (`/p/<id>/eval`)
    — one origin, no port juggling. Supersedes the one-server-per-program model from
    decision #8's implementation (the eval channel itself is unchanged — still the only wire
    op; the portal just routes it). This demos far better than a bare per-program URL.
14. **Static views — the class graph before it runs** (Jimmy ruling). The typechecker
    already resolves the full static structure (field-type refs, `implements`, generic
    instantiations, method sigs — it's in the mono table + itables). Expose it as a schema
    dump and render a **node-link class-relationship graph** (classes/interfaces/enums as
    nodes; field/implements/generic edges), clickable to fields/methods/source. Two entry
    points: `scry inspect <file>` registers the static schema with the portal WITHOUT
    running (pure "see the code first"); `scry run` registers the schema at startup so the
    graph is visible immediately, then the SAME nodes gain live instance counts and become
    drillable once main() populates the arenas. Static graph and live inspector are one
    unified view, not two.

15. **Visualization: bespoke, meaningful, program-declared — NOT a generic graph** (Jimmy ruling,
    emphatic; the off-the-shelf force-graph is rejected). Two pillars:
    (a) **Default view = nested containment.** Ownership becomes nesting (a Conversation holds its
    Messages, an Agent holds its Conversation); node size ∝ live instance count (mass is visible);
    utility/noise types recede to a faded periphery. Position and size CARRY MEANING; the layout is
    deterministic (never physics/jitter). **Shared entities** (referenced by multiple owners — a Tool
    held by every Agent) are the hard case: do NOT nest arbitrarily and do NOT fall back to a graph.
    Give each shared entity a stable **identity color** and render it as a colored chip everywhere it
    is referenced; hovering one highlights all its appearances. Light edges optional. Keep it flexible.
    (b) **Program-declared views = a first-class `view` construct.** The program declares how its
    entities should be seen: `view Name for T { title: <field>; size: byCount; section "…" { <field>
    as timeline(order: …) | chips | rows | card } }`. Reflection surfaces the view specs + the data
    they need; a hand-built bespoke renderer honors them. This is the "the program speaks to how it
    ought to be visualized" requirement. Build order: bespoke default view first, then the `view`
    construct. A visual MOCKUP artifact is being produced for sign-off BEFORE any runtime work.

## Open (docs should propose, flag as OPEN, not silently decide)

- **How the viewer's eval channel interleaves with running threads** — evals that read
  can likely run at a partial safepoint; definition evals need full stop-the-world.
- **Thread API surface** — spawn/join shape, what synchronization primitives (mutex?
  channels? atomics?) the language exposes for the PoC.
- **Interface default methods** — lean no for PoC.

## Document set

- `00-vision.md` — thesis, non-goals, demo narrative
- `01-language.md` — syntax, type system, entity model, generics, stdlib sketch
- `02-runtime.md` — bytecode VM, object layout, per-type arenas, GC, instance enumeration
- `03-live-semantics.md` — redefinition semantics in a running statically-typed system
- `04-viewer.md` — viewer UX + runtime↔viewer wire protocol
- `05-milestones.md` — PoC scope, demo script, cut lines
