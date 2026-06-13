# funct

<!--
  INTRO — to be filled in.
  A few sentences on what funct is and why it exists. The one idea worth landing: a small functional scripting language whose entire VM state is reified, so execution can be paused, snapshotted, and resumed.
-->
_A small, functional, embeddable scripting language with a fully reified bytecode VM you can pause, snapshot, and resume._

```bash
cargo run -- examples/demo.ft     # run a script
cargo test                        # the test suite
```

**Docs:** [Language Guide](docs/guide.md) · [Standard Library](docs/stdlib.md) · [Spec](docs/funct-spec.md)

---

## A taste of the language

```text
// types are tagged unions of records
type Shape = Circle { radius: Float } | Square { side: Float }

fn area(s) = match s {
    Circle { radius } => 3.14159 * radius * radius,
    Square { side }   => side * side,
}

let shapes = [Circle { radius: 1.0 }, Square { side: 2.0 }]
let total  = shapes |> map(area) |> sum        // pipes — or UFCS: shapes.map(area).sum()

// the only escaping mutable state is an atom
let hits = atom(0)
swap!(hits, n => n + 1)

// Result + `?` propagation, then pattern matching
fn parse_pair(a, b) {
    let x = parse_int(a)?
    let y = parse_int(b)?
    Ok((x, y))
}

match parse_pair("3", "4") {
    Ok((x, y)) => println("sum = ${x + y}"),   // string interpolation
    Err(msg)   => println("oops: ${msg}"),
}
```

Patterns cover destructuring, guards, ranges (`0..10`), or-patterns (`1 | 2`), `as`-binding, and list rests (`[x, ..rest]`). A subjectless `match { ... }` is just an anonymous one-argument function, so `x |> match { ... }` works.

> funct is **dynamically typed**. The annotations in `type` definitions (`radius: Float`) are documentation — they are parsed but not checked.

---

## Embedding in Rust

funct is a library first. The engine, every value, and even a paused mid-execution state are `Send + Sync`, so they move across threads and live in host containers (e.g. Bevy resources).

```rust
use funct::{Funct, vals};

let mut vm = Funct::new();

// expose Rust functions — arguments and return values convert automatically
vm.register1("double", |x: i64| x * 2);
vm.register2("add", |a: i64, b: i64| a + b);

// or expose a whole Rust type, with constructors, fields, and methods
vm.register_type::<Player>("Player")
    .ctor2("new_player", |name: String, hp: i64| Player { name, hp })
    .field("hp", |p| p.hp)
    .method1("damage", |p, n: i64| { p.hp -= n; p.hp });

// or bundle functions into a module the script can import
vm.register3("lerp_impl", |a: f64, b: f64, t: f64| a + (b - a) * t);
let lerp = vm.native_fn("lerp_impl").unwrap();
vm.register_module("math", vec![("lerp", lerp)]);

// every registered fn also works as a method or in a pipe inside the script
vm.eval(r#"
    import { lerp } from "math"
    let p = new_player("hero", 10)
    p.damage(3)
    lerp(0.0, 10.0, 0.5)
"#)?;

// and call back into the script from Rust, with a typed return
let result: i64 = vm.call_typed("compute", vals![20, 2])?;
```

`register_raw` gives a native full engine access; `set_global` injects host values between calls. A JSON bridge converts `Value` ⇄ `serde_json::Value` in both directions.

---

## funct vs Rhai

funct began as a replacement for a [Rhai](https://rhai.rs) widget host, so the comparison is direct — and Rhai is the more sensible default for most projects. Rhai is mature and widely used in production, has far more documentation and a larger feature set, runs `no_std`, and has years of hardening behind it. funct is new (0.1) and small.

Where funct differs:

- **Reified, resumable VM.** funct compiles to bytecode and keeps all execution state as plain data, so a running program can be paused mid-execution, serialized to disk, and resumed later — even in another process. Rhai walks the AST directly and has no equivalent of suspending and restoring a half-finished call.
- **Language ergonomics.** `|>` pipes with UFCS, ML-style tagged unions with real pattern matching, `Result`/`Option` with `?`, and atoms as the single source of mutable state.
- **Performance.** On a handful of compute micro-benchmarks (`bench/`), funct's bytecode VM currently runs ahead of Rhai 1.25 on recursion and tight loops and behind it on list-building:

  | benchmark | funct vs Rhai 1.25 |
  |---|---|
  | `fib(30)` (recursion) | ~2.2× faster |
  | tight `for` loops | ~1.1× faster |
  | `collatz` (`while` + arithmetic) | ~1.2× faster |
  | `map`/`filter`/`fold` list building | ~1.7× slower |
  | startup | ~on par |

  These are small synthetic programs on one machine, not a general claim — see `bench/README.md` for the methodology.

Pick Rhai for maturity and ecosystem. Pick funct if pausing/snapshotting execution or the functional ergonomics are what you need.

---

## Built for live editing: pause, snapshot, resume

funct was written to host live, hot-reloadable widgets inside [jim](https://github.com/jimmyhmiller/jim), a code editor. A widget is a script the editor drives through lifecycle functions (`on_init`, `render`, `on_click`, `on_frame`); the editor registers a host surface of natives the script imports (`examples/widgets/host.ft`). `examples/widgets/chess.ft` is a real ~1,700-line widget — full move generation, SAN/PGN, and a UCI engine driven over subprocess pipes — exercised end to end in `tests/chess_widget.rs`.

The reason for a reified VM is exactly this use case. Because every instruction boundary is a safe point and all state is plain data, a widget can be paused between frames, snapshotted to disk, restored in a fresh process, and hot-reloaded function-by-function without losing its state.

```rust
use funct::{Funct, Value, StopWhen, RunResult, Cause};

let mut vm = Funct::new();
vm.eval(/* a long-running function `work` */)?;

// run with a deterministic instruction budget instead of to completion
let mut st = vm.start("work", vec![Value::Int(100)])?;
match vm.run(&mut st, StopWhen::Fuel(500)) {
    RunResult::Paused(Cause::FuelExhausted) => {
        // `st` is plain data: inspect it, `.clone()` it (time travel),
        // or serialize the whole VM — globals, atoms, frames, stack.
        let json = vm.save_state(&st)?;
        std::fs::write("state.json", json)?;
    }
    RunResult::Done(value)  => { /* finished */ }
    RunResult::Faulted(f)   => { /* runtime fault, as a value */ }
    _ => {}
}

// ...later, possibly in a different process — re-register the same natives,
// then pick up exactly where it left off:
let mut vm = Funct::new();
let mut st = vm.restore_state(&std::fs::read_to_string("state.json")?)?;
vm.run(&mut st, StopWhen::Never);
```

Budgets compose: `StopWhen::Fuel(n)` is a deterministic instruction quota, `StopWhen::Deadline(d)` is a wall-clock limit (funct advances an epoch counter from a lazily-started ticker thread), and `StopWhen::Budget { fuel, deadline }` enforces both at once — whichever trips first, with the `Cause` (`FuelExhausted` / `DeadlineReached`) telling you which. For full control the host can drive the epoch itself — `vm.epoch()` hands out the `Arc<AtomicU64>` to bump from a timer or frame loop, paired with `set_deadline` and `StopWhen::Epoch`.

`run` also stops on `StopWhen::Breakpoints(lines)` or `StopWhen::NextLine`, and `vm.step(&mut st)` executes a single instruction. Snapshots round-trip all pure-script state (with atom identity and cycles preserved); host objects can't be serialized and fail loudly rather than dropping silently.

---

## License

MIT — see [LICENSE](LICENSE).
