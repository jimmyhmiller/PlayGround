# funct

A small, functional, embeddable language implementing
[`funct-spec.md`](../../../open-source/rhai/funct-spec.md): `|>` pipes + UFCS,
real pattern matching, atoms as the only escaping mutable state, and — the
core thesis — a **stack-based bytecode VM whose state is fully reified**: you
can stop between any two instructions, inspect everything, snapshot it, write
it to disk, and resume it later (even in a new process).

```bash
cargo run -- examples/demo.ft
cargo test
```

## The language, in one example

```text
type Shape = Circle { radius: Float } | Square { side: Float }

fn area(s) = match s {
    Circle { radius } => 3.14159 * radius * radius,
    Square { side } => side * side,
}

let total = shapes |> map(area) |> sum     // pipes; UFCS: shapes.map(area) works too

let counter = atom(0)                       // the only mutable thing there is
swap!(counter, n => n + 1)

fn parse_pair(a, b) {
    let x = parse_int(a)?                   // Result + ? propagation
    let y = parse_int(b)?
    Ok((x, y))
}
```

Patterns support destructuring, guards, ranges (`0..10`), or-patterns
(`1 | 2`), `as`-binding, list rests (`[x, ..rest]`), record/variant fields
with `..`. A subjectless `match { ... }` is an anonymous one-argument
function, so `x |> match { ... }` just works. Loops support `break` and
`continue` (also as match-arm bodies); multi-line conditions may continue
with a leading `and`/`or`.

## A real program

`examples/widgets/chess.ft` (~1,700 lines) is a full port of a rhai chess
widget: complete move generation (castling, en passant, promotion picker,
check/mate), SAN + PGN, drag & drop, a review mode with eval bar/graph —
and a UCI chess engine driven entirely from script over real subprocess
pipes. `tests/chess_widget.rs` plays it end to end, including games against
real Stockfish and a full-strength analysis sweep, via the reusable widget
host harness in `tests/common/mod.rs` (real `proc_spawn/write/read` etc.).

## Tests, in funct

Mark any zero-argument fn with `#[test]` — right next to the code it tests
or in a separate file — and run `funct test <file-or-dir>`:

```text
export fn area(s) = ...

#[test]
fn squares_have_side_squared_area() {
    assert_eq(area(square(3.0)), 9.0)   // also: assert, assert_ne, fail
}
```

Assertions fault with the failing function, line, and both values. Each test
runs in a fresh engine (full isolation — top-level atoms reset), the file's
directory is the module root (tests can import the module under test), and
annotated fns are ordinary fns otherwise — importing a module never runs its
tests. Discovery is by annotation only, never by name.

Widget-style files declare their host interface so they stay testable
outside the host: `extern fn request_render()` / `extern let canvas_w` bind
to host-registered natives/globals when embedded, and fault loudly if used
without one — so `funct test examples/widgets/chess.ft` runs the pure
move-generation/FEN/SAN tests inline in the widget itself.

## Modules

File-based: `"math/vec"` resolves to `<module root>/math/vec.ft` (the script's
directory when run via the CLI; `vm.set_module_root(..)` when embedding).
Imports are **explicit only — there is no wildcard import**, deliberately:

```text
import { lerp, clamp } from "math/vec"     // named
import { lerp as mix } from "math/vec"     // renamed
import "math/vec" as vec                   // qualified: vec.lerp(...)
import "math/vec"                          // alias defaults to last segment: vec
```

Only `export`ed items are importable:

```text
// math/vec.ft
export fn lerp(a, b, t) = a + (b - a) * t
export let origin = (0.0, 0.0)
fn helper(x) = ...                         // module-private
```

Modules see the prelude, registered natives, their own definitions, and their
imports — nothing else (no access to the main program's globals or other
modules' internals; the compile error tells you to import). Modules load once
and are cached (shared atom state across importers), cycles are detected and
reported with the chain, and `vm.reload_module(path)` hot-swaps a module's
functions for all importers. Host modules from `register_module("math", ...)`
are importable with the same syntax. Snapshots are self-contained: a restored
state keeps its imports working without the module files.

## Embedding (Rust interop)

```rust
let mut vm = Funct::new();

vm.register1("double", |x: i64| x * 2);              // auto conversions
vm.register1("read", |p: String| -> Result<String, String> { ... });  // ⇒ script Ok/Err

vm.register_type::<Player>("Player")
    .ctor2("new_player", |name: String, hp: i64| Player { name, hp })
    .field("hp", |p| p.hp)
    .method1("damage", |p, n: i64| { p.hp -= n; p.hp });

vm.eval(r#"
    let p = new_player("hero", 10)
    p.damage(3)          // every registered fn works as method, pipe, or call
"#)?;

let dmg: i64 = vm.call_typed("compute_damage", vals![20, 2])?;  // Rust → script
```

`register_raw` gives a native full engine access (e.g. `vm.call_value` to
invoke script closures). `register_module("math", ...)` exposes a record of
functions as `math.lerp(...)`. `vm.set_global("canvas_w", v)` injects host
values between calls.

**JSON bridge**: `Value::to_json()` / `Value::from_json()` convert to/from
`serde_json::Value` (atoms/closures/natives fail loudly — never silently
dropped), and `serde_json::Value` implements `FromValue`/`ToValue`, so
natives can take/return JSON directly. In-script: `json_parse`/`json_stringify`.

**Thread-safety**: values are Arc/Mutex-backed and `Value`/`Funct`/`VmState`
are `Send + Sync` — the engine (and a paused, mid-execution VmState) can move
across threads or live in thread-safe containers (Bevy resources, etc.).
The contract: registered natives must be `Send + Sync` and host types `Send`.
(Benchmarked vs an Rc build: the atomics cost is noise next to interpretation
overhead, so there is no non-sync mode; `value::shared` is the single seam if
one were ever wanted.)

## The reified VM

Script frames never touch the host call stack — `CALL` pushes a frame onto a
heap `VmState` and the same `step()` loop continues, so *every* instruction
boundary is a safe point:

```rust
let mut st = vm.start("work", vec![Value::Int(100)])?;
match vm.run(&mut st, StopWhen::Fuel(500)) {           // deterministic budget
    RunResult::Paused(Cause::FuelExhausted) => {
        st.frames.last();                              // inspect locals/ip
        let snapshot = st.clone();                     // time travel
        let json = vm.save_state(&st)?;                // ... or persist
        std::fs::write("state.json", json)?;
    }
    ...
}

// later — possibly a different process:
let mut vm2 = Funct::new();          // re-register the same natives first
let mut st2 = vm2.restore_state(&std::fs::read_to_string("state.json")?)?;
vm2.run(&mut st2, StopWhen::Never);  // picks up exactly where it left off
```

`run` also supports `StopWhen::Breakpoints(lines)` and `StopWhen::NextLine`;
`vm.step(&mut st)` executes exactly one instruction.

### Serialization limits (by design, loudly enforced)

* A snapshot contains the full code table, globals, atoms (with identity and
  cycles preserved), frames and operand stack. Pure-script state always
  round-trips.
* `Native` **values** (host objects) cannot be serialized — `save_state`
  fails with a clear error rather than silently dropping them.
* Native **functions** serialize as names; the restoring process must
  register the same natives first or `restore_state` fails loudly.
* A native call (including script closures it invokes reentrantly, e.g.
  `swap!`'s function) is atomic: you cannot pause *inside* it, only between
  instructions around it.

## Atoms & hot reload

* `atom(v)`, `@a`, `swap!(a, f)`, `reset!(a, v)`, `watch(a, key, f)` —
  watchers fire after each successful mutation.
* `vm.capture_atoms()` / `vm.restore_atoms(json)` snapshot just the mutable
  root set (spec §7): persist on close, re-eval the program on reopen
  (atoms get the same deterministic ids), restore values into them.
* Re-`eval`ing a `fn` hot-swaps it by name: all callers and stored closures
  pick up the new code (calls resolve through the table by FnId). In-flight
  frames finish on the code they started with; the swap happens between two
  `step()`s, which is always an instruction boundary.

## Implementation map

| file | what |
|------|------|
| `src/lexer.rs` | tokens, significant newlines, string interpolation |
| `src/parser.rs` | recursive descent; desugars pipes/UFCS/subjectless match |
| `src/compiler.rs` | AST → bytecode; closures/upvalues; `let mut` → shared Cells; tail calls |
| `src/bytecode.rs` | instructions, compiled patterns, fn protos (all serde) |
| `src/vm.rs` | the engine: step loop, values ops, pattern matcher, atoms |
| `src/interop.rs` | `ToValue`/`FromValue`, `register*`, `register_type`, modules |
| `src/snapshot.rs` | state/atom serialization with identity & cycle preservation |
| `src/prelude.rs` | natives + stdlib written in funct |
| `src/json.rs` | `Value` ⇄ `serde_json::Value` |

## Stdlib (prelude)

* **fns**: `map filter fold sum reverse to_list unwrap_or` (written in funct)
* **math**: `sqrt sin cos tan atan2 exp ln log10 floor ceil round abs min max
  clamp to_int to_float parse_int parse_float` — UFCS makes `(2.25).sqrt()` work
* **strings**: `contains starts_with ends_with split chars join slice replace
  trim to_lower to_upper index_of` (→ `Some(i)`/`None`, never -1), `str`,
  interpolation `"{expr}"`
* **lists**: `len push first last rest pop insert_at remove_at contains
  index_of is_empty slice sort sort_by`
* **records**: `has get assoc dissoc merge keys values entries`
* **nested paths**: `get_in assoc_in update update_in` and atom conveniences
  `swap_in!(state, ["ui", "clicks"], n => n + 1)` / `reset_in!` (fire watchers)
* **atoms**: `atom deref swap! reset! watch unwatch`
* **json**: `json_parse json_stringify` (both return `Result`)
* **misc**: `typeof print println`

## Deviations / decisions of note (v0.1)

* `match {` always starts a *subjectless* match; parenthesize a record
  literal subject: `match ({ x: 1 }) { ... }` (same flavor of restriction as
  Rust struct literals in conditions).
* In a match-arm guard, a bare `ident =>` is the arm's arrow; parenthesize
  lambdas inside guards.
* `{ x }` is the record shorthand (per spec), so a block whose only content
  is a lone identifier needs nothing — just don't write braces.
* Imports are explicit only: named `import { a, b } from "m"` or qualified
  `import "m" as m` — wildcard imports do not exist and `import *` is a
  dedicated error. `export type` parses but creates no binding (variant tags
  are dynamic, not namespaced).
* Gradual types (M7) are deferred: annotations in `type` definitions parse
  and are discarded; match exhaustiveness is a runtime `no-match` fault.
* Integer ops fault on overflow rather than wrapping; `/ % **` on two Ints
  stay Ints; conditions and `and`/`or` require real Bools.
