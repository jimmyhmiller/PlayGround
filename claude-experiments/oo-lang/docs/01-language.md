# 01 — Language

Scope: surface syntax, type system, built-in types, the entity/value line, and enough of a
concurrency story to write the demo app. This doc is precise on purpose — `02-runtime.md`
(object layout, arenas, GC) and `03-live-semantics.md` (redefinition) are built directly on
top of the decisions made here. Where DECISIONS.md lists something as open, this doc takes
a position and marks it **OPEN** rather than silently deciding it.

Working name for the language throughout code samples: `oo` (`.oo` files, `oo run foo.oo`).
Actual name is unresolved — see `00-vision.md`.

## 1. Surface syntax by example

### 1.1 Classes, fields, constructors

A class is a named, fixed set of typed fields plus methods. No inheritance — not "rare,"
absent from the grammar.

```
class Message {
  role: String
  content: String
  createdAt: Int

  fn init(role: String, content: String) {
    self.role = role
    self.content = content
    self.createdAt = Clock.now()
  }

  fn preview(n: Int) -> String {
    if self.content.len() <= n { self.content } else { self.content.slice(0, n) + "..." }
  }
}
```

If a class has no `init`, the compiler synthesizes a memberwise constructor from the field
list in declaration order — this is the common case for plain data holders:

```
class Task {
  title: String
  priority: Int
  done: Bool
}
```

Construction always uses named arguments and looks like a call on the class name, whether
the constructor is synthesized or hand-written:

```
let t = Task(title: "scrape docs", priority: 2, done: false)
let m = Message(role: "user", content: "summarize the PR")
```

`init` must assign every declared field exactly once on every path (definite-assignment,
checked statically, Swift-style) — a class can never observably exist half-initialized.
There is no field-level `null`; a field that isn't always known at construction time is
typed `Option<T>` (§3.3), not left unset.

`self` refers to the receiving instance; there is no implicit `this`/`self`-elision. Fields
are always accessed as `self.field`, never bare.

### 1.2 Generics

Both classes and functions take type parameters:

```
class Inventory<T> {
  items: List<T>

  fn init() { self.items = List<T>() }
  fn add(item: T) { self.items.push(item) }
  fn count() -> Int { self.items.len() }
}

fn first<T>(items: List<T>) -> Option<T> {
  if items.len() == 0 { None } else { Some(items.get(0)) }
}

fn largest<T: Comparable>(items: List<T>) -> T {
  var best = items.get(0)
  for x in items {
    if x.compareTo(best) > 0 { best = x }
  }
  best
}
```

`T: Comparable` is a generic bound against an interface (§2.3). Generics are monomorphized,
not erased — see §2.4 for why that matters for this language specifically (it isn't just a
performance choice here).

### 1.3 Enums and pattern matching

Enums are tagged unions with optional payloads — this is how `Option`, `Result`, and plain
status types like `AgentStatus` are all expressed with one mechanism:

```
enum AgentStatus {
  Waiting
  Running
  Paused
  Done
}

enum Shape {
  Circle(Float)
  Rectangle(Float, Float)
}
```

`AgentStatus` is the canonical status enum used everywhere else in this doc set —
`04-viewer.md`'s wire-protocol examples and `00-vision.md`'s demo script (which queries
`status == Waiting`) both refer to exactly these four variants, no more, no fewer. This is
also the reason `AgentStatus` itself carries no payload here: `Shape` below is the payload
example instead.

`match` is an expression that destructures an enum value and must be exhaustive over its
cases (the compiler checks this statically — new cases added to an enum are a compile
error at every non-exhaustive `match` on it, which is the point):

```
fn area(s: Shape) -> Float {
  match s {
    Circle(r) -> 3.14159 * r * r
    Rectangle(w, h) -> w * h
  }
}

fn describe(status: AgentStatus) -> String {
  match status {
    Waiting -> "waiting"
    Running -> "running"
    Paused -> "paused"
    Done -> "done"
    _ -> "unknown"
  }
}
```

`match` also works on primitives with literal patterns and a required `_` fallthrough:

```
match code {
  200 -> "ok"
  404 -> "not found"
  _ -> "other"
}
```

There is no structural/positional destructuring beyond one level of enum payload for the
PoC (no nested pattern matching, no guards). That's a deliberate scope cut, not a
type-system limitation — nothing here blocks adding it later.

### 1.4 Functions and control flow

Top-level functions exist outside classes; `main` is the entry point:

```
fn clamp(x: Int, lo: Int, hi: Int) -> Int {
  if x < lo { lo } else if x > hi { hi } else { x }
}

fn main() {
  let n = clamp(15, 0, 10)
  print(n)
}
```

`if`/`else` is an expression (both arms must agree in type when used as a value); `while`
and `for`-`in` are statements:

```
var i = 0
while i < 10 {
  print(i)
  i = i + 1
}

for msg in conversation.messages {
  print(msg.role + ": " + msg.preview(80))
}
```

`let` bindings are immutable, `var` bindings are mutable — this applies to locals; fields
are always mutable through `self` (there's no separate `val`/`readonly` field modifier in
the PoC).

String interpolation is built in:

```
let greeting = "Hello, ${self.name}, you have ${tasks.len()} tasks"
```

### 1.5 Error handling

Errors are values, not exceptions — there is no `throw`/`try`/`catch` in the language.
Fallible functions return `Result<T, E>` (§3.4), and `?` is postfix sugar that returns the
`Err` early from the enclosing function (which must itself return a compatible `Result`):

```
fn parsePriority(s: String) -> Result<Int, ParseError> {
  let n = Int.parse(s)?          // returns Err(ParseError(...)) from parsePriority if parse fails
  if n < 0 || n > 5 {
    Err(ParseError("priority out of range: " + s))
  } else {
    Ok(n)
  }
}
```

There is no `panic`/abort path a normal program takes on recoverable failure. Truly
unrecoverable states (a broken runtime invariant, not a user error) are a distinct
mechanism owned by the runtime, not surfaced in this doc.

### 1.6 Modules and imports

One file is one module; module path mirrors file path from the project root. No implicit
global namespace beyond the current module.

```
// file: agents/tools/shell.oo
module agents.tools

class ShellTool {
  command: String
  fn call(args: Map<String, String>) -> Result<String, ToolError> { ... }
}
```

```
// file: main.oo
import std.collections.{List, Map}
import agents.tools.{ShellTool, SearchTool}
import agents.tools.ShellTool as Shell

fn main() { ... }
```

### 1.7 Singleton objects

Namespaced utility functions (no meaningful instance data, but still entities — see §4) use
`object`, a class with exactly one implicit instance created at first use:

```
object Clock {
  fn now() -> Int { __builtin_epoch_millis() }
}

object Console {
  fn log(s: String) { __builtin_write_stdout(s + "\n") }
}
```

`Clock.now()` is a call on the singleton instance, not a `static` keyword bolted onto
`class` — there's one construct for "a thing with fields and methods" (§4 leans on this).

### 1.8 `migrate` blocks (live-edit surface)

When a live edit adds or renames fields on a class with live instances, the edit may carry
a `migrate` block giving each new field its value — a constant default, or a function of
the entire old instance:

```
migrate Agent v1 -> v2 {
  notes: ""                                        // default form
  summaryTone: (old) => old.status == AgentStatus.Paused
    ? Tone.Terse
    : Tone.Normal                                  // derived from an old field
}
```

This is surface syntax only; when it runs, the quiescence gate, and per-instance
failure/quarantine are all owned by `03-live-semantics.md`. The PoC ships only additive
fields with a declaration-site default (`05-milestones.md` M3); a `migrate` block, if
written, is a hard compile error in the PoC build (`migrate blocks are not implemented in
this build`) — the syntax is pinned here so 03's fuller design has a defined surface.

## 2. Type system

- **Nominal, not structural.** Two classes with identical fields are different types.
  Nominal typing is what makes the viewer's type index meaningful (`Agent` means the
  `Agent` you declared, not "anything shaped like it") and what makes per-type arenas
  well-defined (§4).
- **Static, fully checked before run.** Every field, parameter, and return type is
  annotated or inferred from an unambiguous local context (`let x = Task(...)` infers
  `Task`; there is no whole-program type inference).
- **No inheritance, no subtyping between classes.** No `extends`, no virtual dispatch
  hierarchy, no diamond problem, no "is-a" relationship between two `class` declarations,
  ever. This is pinned in DECISIONS.md and non-negotiable for the PoC.
- **No `null`.** Absence is `Option<T>` (§3.3).

### 2.1 Nominal typing and the viewer

Nominal typing isn't just a type-theory preference here — it's load-bearing for the
product. The viewer's entity index is literally the set of nominal class declarations in
the program; "browse all `Agent`s" only means something because `Agent` is a fixed,
named, checked type, not a shape two different classes happen to satisfy.

### 2.2 No inheritance — how you actually reuse code

With no inheritance, code reuse across classes is composition (a field of another class's
type) plus generics, full stop:

```
class Conversation {
  history: List<Message>
  fn append(m: Message) { self.history.push(m) }
}

class Agent {
  name: String
  conversation: Conversation     // composition, not inheritance
}
```

### 2.3 Interfaces — OPEN, not needed for the PoC

DECISIONS.md leaves "interfaces/traits for polymorphism without inheritance" open. Per
`05-milestones.md`'s actual scope, **the PoC demo does not need this**: the one place the
demo wants interface-style polymorphism — `Agent.tools: List<Tool>` holding a
heterogeneous mix of tool kinds — is met by the **enum-dispatch fallback** `05-milestones.md`
M4 adopts (spelled out at the end of this section), built entirely from §1.3's enums and
`match`, which ship in M0. "Swap a tool" (DECISIONS.md #10) means replacing a `Tool` enum
value in that list with a different case. Generic bounds (`T: Comparable`, §1.2) are also
out of scope for the PoC (05-milestones.md, "What we fake" #3) and hard-error if written.

What follows is therefore a **post-PoC extension point**, not a PoC requirement: a
concrete proposal for the day interfaces/traits do get built, so the design isn't
speculative when that day comes, and so `interface`/`implements` (which hard-error if
written in the PoC build) have a real target to hard-error *toward*.

**Proposal (post-PoC):** interfaces are a separate declaration (not a class), classes opt
in explicitly, dispatch is dynamic but resolved without a per-object vtable pointer:

```
interface Tool {
  fn name() -> String
  fn call(args: Map<String, String>) -> Result<String, ToolError>
}

class ShellTool implements Tool {
  command: String
  fn name() -> String { "shell" }
  fn call(args: Map<String, String>) -> Result<String, ToolError> { ... }
}

class SearchTool implements Tool {
  index: String
  fn name() -> String { "search" }
  fn call(args: Map<String, String>) -> Result<String, ToolError> { ... }
}
```

Deliberately `implements`, not `class ShellTool : Tool` — Kotlin overloads the colon for
both superclass and interface, and this language has no superclass, ever; reusing that
syntax would visually suggest inheritance we don't have. An interface-typed reference
(`Tool` used as a field or parameter type) is a two-word value: `(class-id, instance
handle)`. The class-id is the same tag every object already carries for the arena/GC
(§4), so calling `tool.call(args)` is a jump-table lookup keyed on that existing tag —
no extra pointer stored per object, and the viewer always shows the concrete class
(`ShellTool#3`), never a bare interface name, even when reached through a `Tool`-typed
field.

Generic bounds (`T: Comparable` in §1.2) use the same `interface` declarations. This is
the one form of polymorphism-without-inheritance the language has; there is no other.

**Resolved for the PoC (per 05-milestones.md): interfaces do not ship in M0–M4.** The
`interface`/`implements` keywords, if written, are a hard compile error in this build
("What we fake" #2). The demo's tool heterogeneity is carried instead by the
**enum-dispatch fallback** `05-milestones.md` M4 adopts — concrete classes wrapped in enum
cases, dispatched with an ordinary `match`:

```
enum Tool {
  Shell(ShellTool)
  Search(SearchTool)
}

fn callTool(tool: Tool, args: Map<String, String>) -> Result<String, ToolError> {
  match tool {
    Shell(t) -> t.call(args)
    Search(t) -> t.call(args)
  }
}
```

The cost, named in `05-milestones.md`: adding a genuinely new kind of tool means editing
the `Tool` enum itself, not writing an independent class that opts in — acceptable for a
PoC with two tool kinds. What remains **OPEN** is only the question this section actually
answers: *when* interfaces are eventually built post-PoC, is the
jump-table-on-existing-class-tag mechanism above the right design, or does `02-runtime.md`
want something else once per-type arenas are further along? That question has no bearing
on M0–M4 and doesn't need resolving before the demo ships.

### 2.4 Generics: monomorphized, not erased

`Inventory<Tool>` and `Inventory<Task>` are different concrete types with different
arenas, the same way `Agent` and `Task` are — a generic class instantiated at a type
argument produces a distinct entity type, shown in the viewer as `Inventory<Tool>`. This
falls directly out of the per-type-arena decision: there's no way to "list every live
`Inventory<Tool>`" cheaply if `Inventory<Tool>` and `Inventory<Task>` instances share a
slab. Compiling generics by monomorphization (stamp out a concrete class per instantiation
site, like Rust/C++, not type-erase like Java/TS) is therefore not a performance
nice-to-have here — it's required for the entity model to apply to generic classes at
all. Generic *functions* (not classes) may still be compiled once with erased/boxed
representations where no arena is involved; only generic classes need monomorphized
identity.

## 3. Built-in types and stdlib sketch

### 3.1 Primitives

| Type | Notes |
|---|---|
| `Bool` | `true` / `false` |
| `Int` | 64-bit signed |
| `Float` | 64-bit IEEE double |
| `String` | immutable, UTF-8, `+` concatenates, `"${expr}"` interpolates |

**String escape sequences.** Inside a string literal, `\` introduces an escape:
`\n` (newline), `\t` (tab), `\\` (backslash), `\"` (double quote), and `\xHH` (a raw byte,
two hex digits — the general escape hatch for emitting a literal control byte, e.g.
`\x1b` for ESC). This is not a nice-to-have: `01-language.md` §1.7's `Console.log`
example (`__builtin_write_stdout(s + "\n")`) and `05-milestones.md`'s M4 ANSI terminal
renderer (cursor-positioning escape codes) both mechanically require constructing control
bytes inside a literal, and without a documented escape grammar there is no specified way
to do that. Any other character after `\` is a compile error, not a silently-passed-through
backslash. A literal must still be valid UTF-8 once fully assembled — `\x1b` followed by
ordinary ASCII is fine; a `\xHH` byte that leaves the literal as invalid UTF-8 (e.g. a lone
continuation byte) is a compile-time error on the literal, not a runtime one.

### 3.2 Collections

```
class Inventory<T> { items: List<T> ... }   // as shown above

let xs: List<Int> = List<Int>()
xs.push(1)
xs.push(2)
xs.get(0)          // Int
xs.len()           // Int
for x in xs { ... }

let m: Map<String, Int> = Map<String, Int>()
m.set("a", 1)
m.get("a")         // Option<Int>, not Int — see §3.3
m.containsKey("a") // Bool
m.len()            // Int
```

`List<T>` and `Map<K,V>` are library types with native VM support for their storage
(contiguous growable buffer / hash table), but they are **values**, not entities — see §4.
`m.get(k)` returning `Option<V>` rather than an unchecked `V` is the same policy as every
other partial lookup in the language (§3.3).

### 3.3 `Option<T>`, not nullability

**Decision: no `null`, no nullable types (`T?`). Absence is `Option<T>`.**

```
enum Option<T> {
  None
  Some(T)
}
```

Justification:

- **Uniformity with `Result<T,E>` (§3.4).** Both "a value might not be there" and "an
  operation might fail" are the same shape of problem (a checked, exhaustively-matched
  enum) in this language. Adding nullable types on top would mean two different absence
  mechanisms for the type checker, the viewer, and the object layout to special-case.
  One mechanism, reused, is simpler to implement correctly and simpler to explain.
- **The viewer needs a real answer for "what does this field show when it's absent."**
  A `Some(v)` / `None` value is a normal, statically-typed value the viewer can render
  identically to any other enum (`Option<Task> = None`) — no separate "null" rendering
  path, no ambiguity between "field not loaded yet" and "field genuinely null."
- **No null-pointer-style bugs to chase in a language whose whole pitch is
  understandability.** A `Conversation` field typed `Task` (not `Option<Task>`) is a
  static guarantee the object always has a task; the compiler enforces it at every
  construction site via definite assignment (§1.1). There is nothing to null-check.
- Kotlin's `T?` is *sugar over the same idea* (an option type with compiler-blessed
  syntax); we're choosing the explicit enum over the sugar for the PoC because it's less
  surface area to design and implement correctly, not because `T?` is a bad idea. Adding
  `T?` as pure sugar over `Option<T>` later is compatible with everything in this doc.

### 3.4 `Result<T, E>`

```
enum Result<T, E> {
  Ok(T)
  Err(E)
}
```

Every fallible stdlib function returns `Result`, never aborts. `?` (§1.5) is the only
control-flow sugar over it; there is no `try`/`catch`.

## 4. Entities vs. values

**Decision (this doc's position, not yet in DECISIONS.md): every `class` (and `object`,
which is a class with one instance) is an entity. There is no separate value-class kind
in the PoC.** Concretely:

| Kind | Examples | Identity | Arena | Viewer-browsable |
|---|---|---|---|---|
| **Entity** | any `class`/`object` you declare — `Agent`, `Conversation`, `Message`, `Task`, `Clock` | yes (handle) | yes, one arena per concrete type | yes, top-level in the type index |
| **Value** | `Bool`, `Int`, `Float`, `String`, `List<T>`, `Map<K,V>`, and any `enum` (including `Option`/`Result`) | no (compared by content) | no | no top-level index; visible only nested inside an entity's field in the detail view |

Justification for drawing the line at "every `class` is an entity, no exceptions":

- It matches the brief exactly — "we have a user class ... entity types" — and it's the
  simplest possible rule: one keyword (`class`), one semantics (arena + identity +
  browsable). No second class-like keyword to explain in a PoC that's supposed to be
  legible on day one.
- It means the entity/value line falls exactly on the enum/class boundary, which is
  already a real boundary in the type system (§1.3 vs §1.1) — no new concept, just a
  consequence of one already needed for pattern matching.
- The cost is real and worth naming: a class used purely as a small, high-churn value
  (e.g., if someone modeled a 2D `Point` as a `class`) still gets a full arena slot,
  identity, and a viewer row — overkill for something you'd want compared and copied by
  value. The PoC's entity types (`Agent`, `Message`, `Task`, ...) are exactly the kind of
  thing that *should* have identity and show up in the viewer, so this cost doesn't bite
  the demo, but it will bite the first person who reaches for `class` to model a value.

**OPEN:** whether a later revision needs a second, lighter kind — a `struct` or `value
class` that has fields and methods like a class but is compared/copied by value, has no
identity, no arena, and doesn't appear in the viewer's top-level index (analogous to the
value/reference-type split in Swift or C#). Nothing in this doc blocks adding it: it would
just be a second keyword with the value-column semantics from the table above. Deferred
because the PoC's entity types don't need it and inventing it now is speculative; flagging
it here since the question ("do ALL class instances live in arenas and show in the
viewer") was asked explicitly and deserves a visible answer rather than a buried one.

## 5. Concurrency — OPEN

DECISIONS.md leaves the concurrency model open. Two different things are easy to
conflate here, so this section keeps them apart: **what M0–M4 actually ship** (no
concurrency surface syntax at all — see `05-milestones.md`), and **a longer-term
proposal for the language** that this doc flags as explicitly post-PoC, not built for the
demo.

### 5.1 What the PoC actually does (no new syntax)

Per `05-milestones.md`, M0 ships "no concurrency of any kind — straight-line
single-threaded execution, top to bottom," and M2's turn-taking illusion is "a
cooperative single-threaded turn scheduler... explicitly not a proposed answer for the
language, only a scheduling trick inside the demo app." Concretely, that trick needs zero
language features beyond what M0 already has: `agents.oo`'s `main` runs an ordinary,
single-threaded round-robin loop that calls a plain, synchronous method on each agent in
turn. There is no `async`, no `await`, no `spawn` anywhere in the source the PoC compiles.

```
fn runTurn(agent: Agent) -> Result<(), AgentError> {
  if agent.status != AgentStatus.Waiting { return Ok(()) }
  agent.status = AgentStatus.Running
  let reply = Llm.complete(agent.conversation)?
  agent.conversation.append(Message(role: "assistant", content: reply))
  agent.status = AgentStatus.Waiting
  Ok(())
}

fn main() {
  let agents = List<Agent>()
  agents.push(Agent(name: "researcher", ...))
  agents.push(Agent(name: "coder", ...))
  agents.push(Agent(name: "reviewer", ...))

  while true {
    for a in agents { runTurn(a) }
    Tui.render(agents)   // ordinary function call, same thread, same turn
  }
}
```

This reads as "concurrent-ish" to an audience watching three status lines tick, but it is
literally an ordinary loop — `Llm.complete` is `M4`'s `ScriptedModel`, which returns
synchronously and deterministically (no real network I/O to wait on), so there is nothing
in the PoC that actually needs to suspend mid-call. The viewer's invoke/eval channel
(`04-viewer.md`, `02-runtime.md`) stays safe the same way M2 designs it regardless: the
bytecode compiler emits a safepoint check at every loop back-edge and call site, and a
pending invoke is serviced at the next one — the program author writes nothing to enable
that; it's a property of the compiler, not of this section's syntax.

### 5.2 Future direction (explicitly post-PoC, not built for M0–M4)

The rest of this section is a proposal for what a real concurrency model for the language
could look like once the PoC no longer bounds the scope — not a description of anything
M0–M4 build. Per `05-milestones.md` ("What we fake" #6), if any `spawn`/`async` primitive
beyond the cooperative turn-queue above is written, it hard-errors `spawn not implemented
in this build` rather than silently doing something.

**Proposal: single OS thread, cooperative scheduling, `async`/`await`/`spawn`.** The
interpreter owns one event loop multiplexing three kinds of work on one thread: agent
turns awaiting LLM API calls, the TUI's input/render loop, and the embedded viewer's
HTTP/WebSocket connections (`04-viewer.md`). `spawn` schedules a new cooperative task;
`await` suspends only the calling task.

```
async fn runTurn(agent: Agent) -> Result<(), AgentError> {
  agent.status = AgentStatus.Running
  let reply = await Llm.complete(agent.conversation)
  agent.conversation.append(Message(role: "assistant", content: reply))
  agent.status = AgentStatus.Waiting
}

fn main() {
  let agents = List<Agent>()
  agents.push(Agent(name: "researcher", ...))
  agents.push(Agent(name: "coder", ...))
  agents.push(Agent(name: "reviewer", ...))

  for a in agents {
    spawn {
      while true {
        if a.status == AgentStatus.Waiting { runTurn(a) }
      }
    }
  }

  Tui.run(agents)   // drives the terminal render loop on the same thread
}
```

Why single-threaded cooperative rather than OS threads, if/when this gets built:

- **No concurrent mutation of arenas.** The per-type-arena GC (`02-runtime.md`) can
  assume "no other thread is allocating or sweeping right now" — a hugely simpler first
  GC than anything with cross-thread synchronization, and the arena design itself
  (contiguous slabs, cheap enumeration) is where the actual novelty of this project
  lives; concurrent GC is a distraction from that in the PoC.
- **The demo's concurrency is I/O-bound, not CPU-bound.** Three agents waiting on LLM
  API responses is exactly what a cooperative event loop is for; nothing in the demo
  script needs true parallelism.
- **The viewer's invoke channel gets simpler too.** If everything runs on one thread with
  explicit `await` suspension points, "pause the world to safely invoke a method from the
  viewer" (the other OPEN item, safety of the invoke channel — `02-runtime.md`) reduces
  to "run the invoke between two scheduler turns," not a real stop-the-world across
  threads.

What this defers, on purpose: true OS-thread parallelism for CPU-bound work. If that's
ever needed, the brief's own "magazine allocator" instinct (thread-local magazines
refilling from a shared central slab, jemalloc/tcmalloc-style) is exactly the mechanism
that would let per-type arenas go multi-threaded without abandoning cheap enumeration —
but building that now, before anything in the PoC needs it, would be speculative. Flagging
as **OPEN**: whether the PoC ships strictly single-threaded (this proposal) or needs OS
threads sooner because `02-runtime.md`'s GC design turns out to want them anyway.

## 6. Complete example: a slice of the agent TUI

This example is **PoC-legal**: it uses only what `05-milestones.md`'s M0–M4 actually
build, and it would compile and run under that plan as written. That means no
`interface`/`implements` (§2.3 — deferred past the PoC) and no `async`/`await`/`spawn`
(§5.1 — the PoC's turn-taking is an ordinary round-robin loop, not new syntax). For the
future-direction version of the concurrency part of this same program, see §5.2.

Exercises: classes, fields, `init`, generics, enums, `match`, `Option`/`Result`/`?`,
control flow, modules, `object` singletons, and the PoC's plain synchronous turn loop.

```
// file: agents/core.oo
module agents.core

enum AgentStatus {
  Waiting
  Running
  Paused
  Done
}

class Message {
  role: String
  content: String
  createdAt: Int

  fn init(role: String, content: String) {
    self.role = role
    self.content = content
    self.createdAt = Clock.now()
  }
}

class Conversation {
  messages: List<Message>

  fn init() { self.messages = List<Message>() }

  fn append(m: Message) { self.messages.push(m) }

  fn tokenEstimate() -> Int {
    var total = 0
    for m in self.messages { total = total + m.content.len() / 4 }
    total
  }
}

enum ToolError {
  NotFound(String)
  Failed(String)
}

// Tool heterogeneity via the enum-dispatch fallback (§2.3, adopted by 05-milestones.md
// M4): concrete classes wrapped in enum cases, dispatched with `match`. No interface,
// no subclasses.
class ShellTool {
  command: String

  fn call(args: Map<String, String>) -> Result<String, ToolError> {
    match args.get("cmd") {
      Some(cmd) -> Ok(__builtin_run_shell(cmd))
      None -> Err(ToolError.Failed("missing cmd arg"))
    }
  }
}

class SearchTool {
  index: String

  fn call(args: Map<String, String>) -> Result<String, ToolError> {
    match args.get("q") {
      Some(q) -> Ok("results for " + q + " in " + self.index)
      None -> Err(ToolError.Failed("missing q arg"))
    }
  }
}

enum Tool {
  Shell(ShellTool)
  Search(SearchTool)
}

fn toolName(t: Tool) -> String {
  match t {
    Shell(_) -> "shell"
    Search(_) -> "search"
  }
}

fn callTool(t: Tool, args: Map<String, String>) -> Result<String, ToolError> {
  match t {
    Shell(s) -> s.call(args)
    Search(s) -> s.call(args)
  }
}

class Agent {
  name: String
  status: AgentStatus
  conversation: Conversation
  tools: List<Tool>

  fn init(name: String, tools: List<Tool>) {
    self.name = name
    self.status = AgentStatus.Waiting
    self.conversation = Conversation()
    self.tools = tools
  }

  fn pause() { self.status = AgentStatus.Paused }
  fn resume() { self.status = AgentStatus.Waiting }

  fn findTool(name: String) -> Option<Tool> {
    for t in self.tools {
      if toolName(t) == name { return Some(t) }
    }
    None
  }

  // Plain, synchronous method — called once per turn by main's round-robin loop
  // (§5.1). No async/await; ScriptedModel (M4) returns synchronously.
  fn runTurn() -> Result<(), AgentError> {
    if self.status != AgentStatus.Waiting { return Ok(()) }
    self.status = AgentStatus.Running
    let reply = Llm.complete(self.conversation)?
    self.conversation.append(Message(role: "assistant", content: reply))
    self.status = AgentStatus.Waiting
    Ok(())
  }
}

object Clock {
  fn now() -> Int { __builtin_epoch_millis() }
}
```

```
// file: main.oo
import std.collections.{List}
import agents.core.{Agent, Tool, ShellTool, SearchTool}
import ui.tui.{Tui}

fn main() {
  let tools = List<Tool>()
  tools.push(Tool.Shell(ShellTool(command: "sh")))
  tools.push(Tool.Search(SearchTool(index: "docs")))

  let agents = List<Agent>()
  agents.push(Agent(name: "researcher", tools: tools))
  agents.push(Agent(name: "coder", tools: tools))
  agents.push(Agent(name: "reviewer", tools: tools))

  // Ordinary, single-threaded round-robin loop — this is the PoC's whole
  // "agents take turns" mechanism (§5.1), not a scheduler primitive.
  while true {
    for a in agents {
      match a.runTurn() {
        Ok(_) -> {}
        Err(e) -> Console.log("agent " + a.name + " turn failed")
      }
    }
    Tui.render(agents)   // renders status lines, reflects self.status changes live
  }
}
```

Every entity here — every `Agent`, `Conversation`, `Message`, `ShellTool`, `SearchTool` —
lives in its own arena the moment it's constructed, with no code in this file doing
anything to make that true (`Tool` itself is an enum, a value wrapping a reference to the
concrete tool entity — §1.3, §4). That's the whole thesis, expressed as an
ordinary-looking program.
