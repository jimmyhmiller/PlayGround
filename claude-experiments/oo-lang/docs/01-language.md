# 01 — Language

Scope: surface syntax, type system, built-in types, the entity/value line, and enough of a
concurrency story to write the demo app. This doc is precise on purpose — `02-runtime.md`
(object layout, arenas, GC) and `03-live-semantics.md` (redefinition) are built directly on
top of the decisions made here. Where DECISIONS.md lists something as open, this doc takes
a position and marks it **OPEN** rather than silently deciding it.

The language is named **Scry** (`DECISIONS.md` #12). CLI is `scry`; source files are
`.scry` (e.g. `scry run agents.scry`).

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
// file: agents/tools/shell.scry
module agents.tools

class ShellTool {
  command: String
  fn call(args: Map<String, String>) -> Result<String, ToolError> { ... }
}
```

```
// file: main.scry
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

### 2.3 Interfaces

**Java-style interfaces are IN the language and IN the PoC (DECISIONS.md #4, Jimmy
ruling).** This supersedes an earlier draft of this section, which proposed shipping only
an enum-dispatch fallback for `Agent.tools` and deferring real interfaces past the PoC —
that fallback is dead. Interfaces are the polymorphism story: nominal, explicit
`implements`, interface types usable as field/parameter/return types, dispatch resolved
dynamically. Enums remain exactly what §1.3 says they are (tagged unions, and the
mechanism behind `Option`/`Result`/`AgentStatus`) — they are just no longer standing in
for interfaces.

An interface is a separate declaration (not a class): a named set of method signatures,
no fields, no bodies.

```
interface Tool {
  fn name() -> String
  fn call(args: Map<String, String>) -> Result<String, ToolError>
}
```

A class opts in explicitly with `implements`, and must provide a body for every method the
interface declares:

```
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
syntax would visually suggest inheritance we don't have.

**Static conformance checking.** `implements Tool` is checked at the class declaration,
not at each call site: every method `Tool` declares must be present on `ShellTool` with an
exactly matching signature (same parameter types, same return type), or the class fails to
compile with the missing/mismatched method named explicitly. A class may `implements`
more than one interface; conformance is checked against each independently.

**Interface types in field/parameter/return position.** `Tool` is a real type once
declared, usable anywhere a class type is:

```
class Agent {
  name: String
  tools: List<Tool>          // heterogeneous: holds ShellTool and SearchTool values alike
}

fn describe(t: Tool) -> String { t.name() }
```

`Agent.tools: List<Tool>` is exactly the demo's "swap a tool" beat (DECISIONS.md #10):
replacing a list element with a different concrete class that also implements `Tool`, no
enum to edit, no `match` to extend — the actual value of interfaces over the old
enum-dispatch fallback (the fallback's named cost was that a new tool kind meant editing
the `Tool` enum itself; a new tool kind is now just a new class that opts in).

**Dispatch.** An interface-typed reference (`Tool` used as a field or parameter type) is a
two-word value: `(class-id, instance handle)`. The class-id is the same tag every object
already carries for the arena/GC (§4), so calling `tool.call(args)` is a jump-table lookup
keyed on that existing tag — no extra vtable pointer stored per object. The viewer always
shows the concrete class (`ShellTool#3`), never a bare interface name, even when reached
through a `Tool`-typed field.

**No default methods (DECISIONS.md, OPEN, lean no for the PoC).** Every method an
interface declares must be implemented by every conforming class; there is no method body
on the interface declaration itself and no inherited-default mechanism. This is a real
scope cut, not just a syntax omission: it sidesteps the diamond-shaped questions default
methods raise (what does a class inherit when it implements two interfaces that default
the same method?) in a language whose entire pitch is "no inheritance, no diamond
problem." **Flagged OPEN**: if the PoC's actual interfaces (`Tool`, `ToolError`-adjacent
helpers) turn out to want shared default behavior, the answer is more likely a plain
top-level function taking the interface type (`fn describeTool(t: Tool) -> String`) than
inherited defaults — but that's a proposal, not a ruling.

**Interfaces as generic bounds — lean minimal for the PoC, flag OPEN.** §1.2 already uses
`fn largest<T: Comparable>(items: List<T>) -> T` with `Comparable` an interface. The
minimal rule this doc takes a position on: `T: SomeInterface` requires that whatever
concrete type is substituted for `T` at a given instantiation site has an `implements
SomeInterface` declaration — checked at monomorphization time (§2.4), the same moment
every other generic-class instantiation is type-checked, so no new checking phase is
needed. A type parameter may carry at most one bound in the PoC; there is no `T: A + B`
bound-combination syntax. **Flagged OPEN**: whether call-site bound checking needs to
happen anywhere other than at monomorphization, and whether multi-bound type parameters
are needed before the demo's actual generic code (`Inventory<T>`, `first<T>`,
`largest<T: Comparable>`) is written — nothing in the demo as scoped so far needs more
than a single bound.

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

## 5. Concurrency — real OS threads, day one

**DECISIONS.md #4b (Jimmy ruling): real OS threads, day one.** This supersedes an earlier
draft of this section, which proposed a single-threaded cooperative scheduler for the PoC
and pushed real threads to a post-PoC future. That's inverted now: the demo app's agents
each run on their own actual OS thread, starting with the PoC, not a scheduling illusion
on top of one interpreter thread. `async`/`await` surface syntax is the thing pushed
later — see §5.5 — not threads themselves.

### 5.1 Thread API surface

Minimal proposal for the PoC:

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

`Thread.spawn` takes a `Runnable` rather than a bare function value: the language has no
closures/lambdas in the PoC surface, so "the work this thread runs, plus whatever context
it needs" is an ordinary class that `implements Runnable` and captures that context as
fields — the same composition-over-inheritance idiom §2.2 already uses everywhere else,
not a special case invented for threads. `spawn` returns a `ThreadHandle`, not a bare
`Thread` — the language avoids naming both the spawning namespace and the handle it
returns identically.

```
class AgentWorker implements Runnable {
  agent: Agent

  fn run() -> Void {
    while true {
      agent.runTurn()
    }
  }
}

fn main() {
  let researcher = Agent(name: "researcher", tools: tools)
  let handle = Thread.spawn(AgentWorker(agent: researcher))
  // ...
  handle.join()
}
```

**Flagged OPEN (DECISIONS.md's "thread API surface" item):** this is a minimal proposal,
not a ruling — whether `join()` returns a value, whether spawn failures are represented as
`Result`, and whether the PoC needs anything beyond spawn/join (e.g. a way to ask "is this
thread still running" without joining) are all open.

### 5.2 Synchronization: `Mutex<T>`

Real threads mean the demo has genuine shared mutable state: each `Agent` is mutated by
its own worker thread (§5.1) and *read* by the main thread's TUI render loop and by the
viewer's eval channel (`04-viewer.md`). That's a real cross-thread read/write pattern, not
a hypothetical one, so the PoC needs at least one synchronization primitive, not zero.

**Proposal: `Mutex<T>`**, a guarded value with explicit lock/unlock (no closures needed,
consistent with §5.1):

```
class Mutex<T> {
  fn init(value: T)
  fn lock() -> Void     // blocks until this thread holds the mutex
  fn unlock() -> Void   // releases it
  fn get() -> T          // read the guarded value — valid only while locked
  fn set(value: T) -> Void  // replace the guarded value — valid only while locked
}
```

`set` exists alongside `get` because the guarded value is often a plain value/enum
(`AgentStatus`, not a class with in-place-mutating methods) — replacing it wholesale is
the only way to change it. A guarded *class* value (`Conversation` below) can instead be
mutated in place by calling its own methods after `get()`, with no `set()` needed.

The `Agent` fields a second thread actually needs to touch are the ones wrapped in
`Mutex<T>`; everything else on `Agent` is conventionally owned by the agent's own worker
thread and touched by nobody else:

```
class Agent {
  name: String
  status: Mutex<AgentStatus>
  conversation: Mutex<Conversation>
  tools: List<Tool>          // only ever touched by this agent's own thread
}
```

`Channel<T>` is not proposed as a PoC primitive on its own merits — nothing in the demo's
agent-TUI has agents messaging each other directly. It shows up instead as the natural
generalization of the old single-thread "safepoint queue" idea (`02-runtime.md`, M2) once
there's more than one real thread: the viewer's eval channel needs *some* thread-safe
queue to hand a pending eval/invoke request to the thread that owns the target instance
and get its result back, and a `Channel<T>` is the obvious shape for that. This doc takes
that as a working assumption, not a ruling — see §5.4.

### 5.3 Data races and the ownership rule — OPEN

**Stated plainly:** a field that is *not* wrapped in `Mutex<T>` and is touched by more
than one thread is a real data race in the PoC. The language has no borrow checker, no
`Send`/`Sync`-style trait, and no compiler enforcement that a non-`Mutex` field is only
ever touched by one thread — that rule above (worker thread owns everything except the
`Mutex`-wrapped fields) is a **convention the demo's own code follows**, not something the
type checker verifies. This is a real, named gap, not a silent one: a program that ignores
the convention and reaches into another agent's non-`Mutex` field from a second thread
compiles and races.

**Flagged OPEN:** whether the PoC needs the type checker to enforce this (e.g., a field
attribute or a rule that non-`Mutex<T>` class-typed fields simply cannot be read from a
`Runnable` that didn't construct the instance) or whether "the demo's own code is written
correctly by convention" is an acceptable PoC-scoped answer given the alternative is
building an ownership/borrow system this project doesn't otherwise need. Leaning toward
the latter for the PoC, but this is a proposal, not a ruling.

### 5.4 Safepoints and stop-the-world, across real threads

DECISIONS.md #4b accepts the runtime consequences of real threads up front: multi-threaded
mutators over the shared heap, thread-safe per-type arena allocation (per-thread
magazines refilling from a shared central slab, jemalloc/tcmalloc-style — the same
mechanism an earlier draft of this section proposed only for a hypothetical
multi-threaded future), and **stop-the-world safepoints that park ALL threads** for GC, a
live-edit migration (`03-live-semantics.md`), or a viewer definition-eval
(`04-viewer.md`). The bytecode compiler still emits a safepoint check at every loop
back-edge and call site — same mechanism the old single-thread design used — except now
*every* running thread must reach one before a stop-the-world operation can proceed, not
just one interpreter loop. Object layout and arena mechanics for this belong to
`02-runtime.md`, not here; what this doc pins is that the language's threads are real
enough that a blocking call on one agent's thread (a real network call, say) blocks only
that thread, not the whole program — one of the actual benefits of real threads over the
old cooperative design, which needed every blocking operation to explicitly yield at a
safepoint to avoid stalling every other agent.

**Flagged OPEN (DECISIONS.md's "eval-channel interleaving" item):** whether a read-only
viewer eval (e.g. `Agent#7`'s field values) can run at a partial safepoint touching only
the target agent's thread, or whether every eval, read or write, needs the full
all-threads stop-the-world. Definition evals (redefining a method/class) need the full
stop-the-world regardless — that much isn't open.

### 5.5 `async`/`await` — explicitly post-PoC

Per DECISIONS.md #4b ("async stuff can come later"), `async`/`await` surface syntax is
not part of the PoC and this doc does not propose a design for it here. If it's ever
built, it would most likely sit on top of the real threads in this section (e.g. as sugar
for a task scheduled onto a thread pool) rather than replace them — but that's speculation
for a later doc revision, not something this section commits to.

## 6. Complete example: a slice of the agent TUI

This is the canonical shape of `agents.scry`, updated for the interfaces and threads
rulings (DECISIONS.md #4, #4b): real `interface`/`implements` (§2.3) carries the tool
heterogeneity that an earlier draft gave to an enum-dispatch fallback, and each agent runs
its own turn loop on its own real OS thread (§5.1), synchronized through `Mutex<T>` (§5.2)
where the main thread's TUI render loop needs to read it. What it does *not* use is
`async`/`await` — that stays explicitly post-PoC (§5.5).

Exercises: classes, fields, `init`, `interface`/`implements`, generics, enums, `match`,
`Option`/`Result`/`?`, control flow, modules, `object` singletons, `Thread.spawn`/`.join()`,
and `Mutex<T>`.

```
// file: agents/core.scry
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

// Tool heterogeneity via real interfaces (§2.3 — supersedes an earlier enum-dispatch
// fallback): any class that implements Tool can live in Agent.tools below, with no
// enum to edit when a new tool kind is added.
interface Tool {
  fn name() -> String
  fn call(args: Map<String, String>) -> Result<String, ToolError>
}

class ShellTool implements Tool {
  command: String

  fn name() -> String { "shell" }

  fn call(args: Map<String, String>) -> Result<String, ToolError> {
    match args.get("cmd") {
      Some(cmd) -> Ok(__builtin_run_shell(cmd))
      None -> Err(ToolError.Failed("missing cmd arg"))
    }
  }
}

class SearchTool implements Tool {
  index: String

  fn name() -> String { "search" }

  fn call(args: Map<String, String>) -> Result<String, ToolError> {
    match args.get("q") {
      Some(q) -> Ok("results for " + q + " in " + self.index)
      None -> Err(ToolError.Failed("missing q arg"))
    }
  }
}

class Agent {
  name: String
  // Wrapped in Mutex<T> because the main thread's TUI render loop and the viewer's
  // eval channel both read these fields while this agent's own worker thread (below)
  // mutates them (§5.2). `tools` is not wrapped: only this agent's own thread ever
  // touches it, by convention (§5.3, flagged OPEN — not statically enforced).
  status: Mutex<AgentStatus>
  conversation: Mutex<Conversation>
  tools: List<Tool>

  fn init(name: String, tools: List<Tool>) {
    self.name = name
    self.status = Mutex<AgentStatus>(AgentStatus.Waiting)
    self.conversation = Mutex<Conversation>(Conversation())
    self.tools = tools
  }

  // A later live edit adding a field to Agent would ship as a `migrate` block (§1.8)
  // giving the new field's value for every live instance; this class, as written,
  // needs none yet.

  fn pause() {
    self.status.lock()
    self.status.set(AgentStatus.Paused)
    self.status.unlock()
  }

  fn resume() {
    self.status.lock()
    self.status.set(AgentStatus.Waiting)
    self.status.unlock()
  }

  fn findTool(name: String) -> Option<Tool> {
    for t in self.tools {
      if t.name() == name { return Some(t) }
    }
    None
  }

  // Runs on this agent's own worker thread (§5.1), driven by AgentWorker below.
  // ScriptedModel (M4) returns synchronously and deterministically in the PoC, so this
  // never actually blocks yet — but note for when it does: a blocking Llm.complete call
  // parks this thread somewhere it can't reach a safepoint, so it does NOT cost only this
  // agent. Per 02-runtime.md §7, request-global-stop then blocks waiting for this thread
  // specifically, delaying every GC, shape migration, and viewer invoke/definition-eval
  // process-wide for as long as the call is outstanding — other agents' threads keep
  // running, but nothing that needs a full stop-the-world can land until this call returns.
  fn runTurn() -> Result<(), AgentError> {
    self.status.lock()
    if self.status.get() != AgentStatus.Waiting {
      self.status.unlock()
      return Ok(())
    }
    self.status.set(AgentStatus.Running)
    self.status.unlock()

    self.conversation.lock()
    let reply = Llm.complete(self.conversation.get())?
    self.conversation.get().append(Message(role: "assistant", content: reply))
    self.conversation.unlock()

    self.status.lock()
    self.status.set(AgentStatus.Waiting)
    self.status.unlock()
    Ok(())
  }
}

// The thread this agent runs on (§5.1): implements Runnable rather than taking a bare
// function value, since the PoC has no closures — the agent to drive is just a field.
class AgentWorker implements Runnable {
  agent: Agent

  fn run() -> Void {
    while true {
      match self.agent.runTurn() {
        Ok(_) -> {}
        Err(e) -> Console.log("agent " + self.agent.name + " turn failed")
      }
    }
  }
}

object Clock {
  fn now() -> Int { __builtin_epoch_millis() }
}
```

```
// file: main.scry
import std.collections.{List}
import agents.core.{Agent, Tool, ShellTool, SearchTool, AgentWorker}
import ui.tui.{Tui}

fn main() {
  let tools = List<Tool>()
  tools.push(ShellTool(command: "sh"))
  tools.push(SearchTool(index: "docs"))
  // No enum wrapper needed — ShellTool and SearchTool sit side by side in one
  // List<Tool> purely because both implement Tool (§2.3).

  let researcher = Agent(name: "researcher", tools: tools)
  let coder = Agent(name: "coder", tools: tools)
  let reviewer = Agent(name: "reviewer", tools: tools)

  let agents = List<Agent>()
  agents.push(researcher)
  agents.push(coder)
  agents.push(reviewer)

  // Each agent runs its own turn loop on its own real OS thread (§5.1, DECISIONS.md
  // #4b) — no cooperative scheduler, no round-robin loop driving all three from one
  // thread.
  let handles = List<ThreadHandle>()
  handles.push(Thread.spawn(AgentWorker(agent: researcher)))
  handles.push(Thread.spawn(AgentWorker(agent: coder)))
  handles.push(Thread.spawn(AgentWorker(agent: reviewer)))

  // Tui.render reads each agent's Mutex-guarded status/conversation fields (§5.2) —
  // the one place this file's code crosses threads.
  while true {
    Tui.render(agents)
  }
}
```

Every entity here — every `Agent`, `Conversation`, `Message`, `ShellTool`, `SearchTool`,
`AgentWorker` — lives in its own arena the moment it's constructed, with no code in this
file doing anything to make that true. `Tool` is an interface, not an enum: a
`Tool`-typed slot in `Agent.tools` holds a real `ShellTool` or `SearchTool` instance
directly, dispatched through the class-tag jump table §2.3 describes, not through a
`match`. That's the whole thesis, expressed as an ordinary-looking program that now also
happens to be genuinely multi-threaded.
