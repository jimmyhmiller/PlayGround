# oo-lang — Pinned Decisions

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
   **no inheritance** of any kind. Generics yes. Interfaces/traits are the open question
   (see below), subclassing is not.
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
    `Agent`, `Conversation`, `Message`, `Tool`, `ToolCall`, `Task`. Demo moment: agents
    running in the terminal; in the viewer you browse live `Agent` instances, inspect a
    `Conversation` mid-run, call a method (pause an agent, edit a task, swap a tool),
    and the TUI reflects it instantly. Dogfoods the "language for AI" thesis.
11. **Demo-worthiness matters.** The viewer must look genuinely good, not programmer-art.

## Open (docs should propose, flag as OPEN, not silently decide)

- **Name.** "oo-lang" is a working directory name, not the name.
- **Interfaces/traits** for polymorphism without inheritance — needed for PoC or deferred?
- **Concurrency model** of the interpreted language (the demo app needs at least
  async-ish agent turns + a responsive TUI).
- **How the viewer's eval/invoke channel is safe** while the program runs (stop-the-world
  on request? safepoints?).

## Document set

- `00-vision.md` — thesis, non-goals, demo narrative
- `01-language.md` — syntax, type system, entity model, generics, stdlib sketch
- `02-runtime.md` — bytecode VM, object layout, per-type arenas, GC, instance enumeration
- `03-live-semantics.md` — redefinition semantics in a running statically-typed system
- `04-viewer.md` — viewer UX + runtime↔viewer wire protocol
- `05-milestones.md` — PoC scope, demo script, cut lines
