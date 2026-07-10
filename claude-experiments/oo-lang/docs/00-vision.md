# 00 — Vision

> Working name: **oo-lang** (directory name, not the name — see [OPEN: Name](#open-name) at the bottom).

## The thesis

AI systems are the most opaque software we have ever run in production. An agent framework
is a pile of objects — agents, conversations, tools, tool calls, tasks — mutating each other
thousands of times a minute, and the state of the art for understanding it is `print()` and
a JSON log viewer. You cannot ask a running agent system the most basic questions: *what
agents exist right now? what is this one doing? what's in that conversation? what happens
if I pause it?*

oo-lang is a statically-typed, OO-style language where **runtime observability is the
product**. Not a debugger you attach when things go wrong. Not tracing you bolt on. The
runtime is built, from the allocator up, so that every live object is enumerable, browsable,
searchable, and pokeable — and a good-looking browser viewer that does all of that ships as
a co-equal, day-one deliverable with the compiler.

The bet: if the language's memory layout makes "list every live `Agent`" an O(live agents)
slab walk instead of a heap crawl, and the viewer is always one URL away, then understanding
a running AI system stops being archaeology and starts being *looking at it*.

## What it looks like

The language is deliberately unexciting. TS/Kotlin-ish braces, classes with data members and
methods, generics, static types, **no inheritance**. If you can read TypeScript you can read
this on day one:

```
class Agent {
  name: String
  model: String
  status: AgentStatus
  conversation: Conversation
  tools: List<Tool>

  fn pause() { self.status = AgentStatus.Paused }
  fn resume() { self.status = AgentStatus.Running }
  fn currentTask() -> Task { self.conversation.task }
}

class Conversation {
  task: Task
  messages: List<Message>

  fn append(m: Message) { self.messages.push(m) }
  fn tokenEstimate() -> Int { ... }
}

class Inventory<T> {
  items: List<T>
  fn add(item: T) { self.items.push(item) }
}
```

What's exciting is what the runtime does with it. Every entity type gets its own arena
(slab/magazine-style allocation, one arena per class). That single decision powers
everything the viewer needs:

- **Enumerate**: all live `Agent`s = walk the `Agent` slab. No heap scan, no registry
  objects the programmer maintains, no forgetting to register.
- **Search**: the viewer filters instances by field values because the runtime knows every
  field's name, type, and offset — it's a statically-typed language; the metadata is free.
- **Watch**: mutate `agent.status` and the viewer's detail pane updates, pushed over a
  WebSocket the runtime embeds.
- **Invoke**: click a method in the viewer, fill in typed arguments, run it against the
  live process.

Concretely, the viewer speaks a small JSON protocol to the embedded server:

```json
→ {"op": "instances", "type": "Agent", "query": "status == Paused"}
← {"op": "instances", "type": "Agent",
   "rows": [{"id": "Agent#7", "name": "researcher", "model": "fable-5",
             "status": "Paused", "conversation": "Conversation#7", "tools": "List<Tool>#3"}]}

→ {"op": "invoke", "target": "Agent#7", "method": "resume", "args": []}
← {"op": "result", "target": "Agent#7", "value": null}
← {"op": "changed", "target": "Agent#7", "fields": {"status": "Running"}}
```

An ordinary program plus a lens. That's the whole shape of it.

## Why this is not Smalltalk

Everyone's first reaction will be "so, Smalltalk." The lineage is real — live objects you
browse and message — but three deliberate choices make it a different thing:

1. **Static types.** People will not adopt a dynamically-typed language in 2026 for
   serious systems, and they're right not to: types are how you understand code you didn't
   write, which is the whole thesis. Yes, this makes live redefinition genuinely hard —
   changing a method while instances exist has to answer "what do existing callers and
   existing instances do now?" precisely, not vibes. That's a design problem we take on
   head-first (`03-live-semantics.md`), not a reason to give up types.
2. **No image.** State is never baked into the program. You edit source files, you run a
   program, it starts from `main`, it exits. Version control works. Deployment works.
   The liveness lives in the *runtime of a normal process*, not in a persistent world
   you sculpt.
3. **Ordinary processes.** Your program is a normal executable with normal stdin/stdout —
   a CLI tool, a TUI, a server. The viewer *attaches* to it from a browser. There is no
   special environment the program must live inside. Kill the viewer tab; the program
   doesn't care.

Smalltalk made the environment the world. We make the world an ordinary process and hand
you X-ray glasses.

## Non-goals

- **No inheritance.** Not "discouraged" — absent. No subclassing, no virtual dispatch
  hierarchies, no diamond problems. Composition and generics. (Interfaces/traits for
  polymorphism without inheritance are **OPEN** — see `01-language.md`; the PoC may not
  need them at all.)
- **No image.** Covered above; worth repeating because every "live" feature will tempt
  us toward one. Any design that requires serializing the heap to disk to be useful is
  wrong.
- **No JIT (yet).** Execution is a bytecode interpreter, implemented in Coil (the clox
  port at `coil/apps/clox/` proved the approach runs at C speed for the interpreter loop).
  A Java-style optimizing JIT is explicitly deferred; the bytecode format must not paint
  us out of one, but no milestone depends on it.
- **Not a general-purpose ecosystem play.** No package manager, no LSP, no self-hosting
  ambitions in scope. One language, one runtime, one viewer, one killer demo.
- **Not an observability SDK for other languages.** The observability comes from owning
  the allocator and the object model. Retrofitting this onto Python is a different (worse)
  project.

## The demo

The proof-of-concept application is an **agent TUI**: a terminal app, written in oo-lang,
that runs AI agents against tasks. Entity types: `Agent`, `Conversation`, `Message`,
`Tool`, `ToolCall`, `Task`. It dogfoods the thesis — the language for understanding AI
systems, demonstrated on an AI system.

The demo is one terminal window and one browser window, side by side. Minute by minute:

**0:00 — Start the program.** In the terminal: `oo run agents.oo`. A TUI comes up: three
agents (`researcher`, `coder`, `reviewer`) working a shared task list. Status lines tick,
a log pane scrolls tool calls. It looks like any agent runner. Nothing about it suggests
there's anything to see beyond what it prints.

**0:30 — Open the viewer.** The runtime printed `viewer: http://localhost:7357` at startup.
Open it. The audience sees a clean entity-type index — not programmer-art, an actually
designed UI:

```
Agent          3 live
Conversation   3 live
Message       47 live     ▲ climbing
Tool           5 live
ToolCall      12 live     ▲ climbing
Task           8 live
```

The counts on `Message` and `ToolCall` are visibly climbing as the agents work. First
audience beat: *the program didn't do anything to make this happen. No annotations, no
registry, no instrumentation. These are just its live objects.*

**1:30 — Drill in.** Click `Agent` → a searchable table of the three instances with their
fields as columns. Click `researcher` → instance detail: every field, typed; references
like `conversation: Conversation#2` are links. Click through to the conversation and the
audience is reading the agent's messages *mid-run*, new ones appearing as the agent
produces them. Type `status == Waiting` into the instance search and filter agents by a
field predicate.

**2:30 — Poke it.** In `researcher`'s detail pane, the methods are listed with their
signatures. Click `pause()` → invoke. In the *terminal*, the researcher's status line
flips to `⏸ paused` within a frame. Second audience beat: *the viewer isn't a read-only
dashboard; it's a two-way lens on the same objects the TUI is rendering.* Open a `Task`,
call `setPriority(1)`, watch the TUI's task list re-sort itself.

**3:30 — The "I changed it live" moment.** The `reviewer` agent's summaries are too long.
Open its `summarize` method in the viewer's code pane (or in your editor — the runtime
accepts redefinitions either way), change the prompt-building line, save. The runtime
**typechecks the new method body against the live class before swapping it in** — make a
type error and the viewer shows the error and refuses; the program never sees a
half-applied change. Fix it, save again: `method Agent.summarize replaced — next call uses
new code`. The very next review in the terminal is short. No restart, no lost
conversations, all 47 `Message` instances still there. Third audience beat, the one people
remember: *I changed the behavior of a statically-typed program while it ran, and the type
checker had my back.*

**4:30 — Close the loop.** Kill the browser tab. The TUI keeps running, indifferent —
it's an ordinary process; the viewer was just a lens. Re-open the URL; everything's there
again. `Ctrl-C` the TUI; the program exits like any program. No image was harmed.

Total: under five minutes, no slides, nothing faked. Every capability shown falls directly
out of a pinned design decision: per-type arenas (the counts and tables), the embedded
HTTP/WS server (the attach/detach), static metadata (typed fields, typed invoke forms),
and live-redefinition semantics (the swap).

## Who it's for

- **People building agent systems** who currently understand their own software through
  log spelunking. This is the primary audience and the demo speaks directly to them.
- **People operating AI systems** — the "what is it doing *right now*" question, asked by
  someone who didn't write the code. The viewer is designed to be legible to a reader,
  not just the author: concrete entity types, real field names, no memory addresses.
- **Language-runtime people**, as an existence proof: observability-first is a viable
  design axis for a statically-typed language, and it costs an allocator design, not a
  research program.

It is explicitly **not** aimed (yet) at people who need a mature general-purpose language.
The PoC wins by being the best possible way to *watch and steer* one class of program.

## OPEN: Name

Five candidates, each pitched at the "lens over a live system" identity. None checked for
trademark/ecosystem collisions yet; **Lucid** in particular collides with a historical
dataflow language.

1. **Scry** — divination by gazing into a live surface. Short, verb-able ("scry the
   process"), `.scry` files, `scry run`. Current favorite for punch.
2. **Vivarium** — an enclosure built for observing live things. Nails the semantics
   (the runtime *is* a vivarium for objects); slightly long; risks image connotations
   we explicitly reject.
3. **Aperture** — the thing you open to see in. Clean, professional; heavily used name
   (Portal, Apple's old photo app).
4. **Plainview** — everything in plain view. Reads well in prose ("run it under
   Plainview"), zero mysticism, a bit sedate.
5. **Lucid** — transparent, and "lucid dreaming" = aware inside a running world.
   Best meaning, worst collision.

## OPEN: flagged elsewhere, not decided here

- **Interfaces/traits** — needed for the PoC or deferred? Proposed in `01-language.md`.
- **Concurrency model** — the demo needs concurrent-ish agent turns plus a responsive
  TUI; proposed in `01-language.md`/`02-runtime.md`.
- **Safety of the viewer's invoke channel** — invoking `pause()` from the viewer while
  the interpreter is mid-turn needs a precise mechanism (safepoints vs. stop-the-world on
  request); proposed in `02-runtime.md`. The demo script above assumes it works; it does
  not assume how.
