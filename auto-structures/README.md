# autostruct

> A little language where you never declare your data types. You write programs
> against abstract *collections*, the compiler watches **how you use** each one,
> and it picks the best concrete data structure for you — then runs the
> specialized program or compiles it to specialized JavaScript.

This is a modern, working take on the old RAND idea (RM-5290 / AD-656449): a
programmer should describe *what* data they need and *what operations* they do
with it, and let the system choose the representation. You say "I need a thing I
can test membership on and ask for the minimum of," and the compiler decides
that's a balanced search tree — you never type `BTreeSet` anywhere.

```
                 ┌──────────┐   usage    ┌──────────────┐   pick      ┌───────────────┐
  program.shape ─►  parse   ├─ analysis ─►  per-variable ├─ structure ─►  specialized   │
                 └──────────┘            │   op profile  │  selection  │  run  /  JS     │
                                         └──────────────┘             └───────────────┘
```

## The idea in one screen

You write this (no types, anywhere):

```
let counts = collection()
for w in text {
  if has(counts, w) {
    put(counts, w, get(counts, w) + 1)
  } else {
    put(counts, w, 1)
  }
}
for w in sorted(counts) {
  print(w, get(counts, w))
}
```

`autostruct` notices that `counts` is used with `put` / `get` (keyed lookups)
**and** is traversed with `sorted` (ordering matters), and reports:

```
 collection `counts`
   used as : get, has, put, sorted
   reason  : keyed lookups (put/get) AND ordered traversal (sorted/min/max) → an ordered map
   chosen  : BTreeMap — O(log n), ordered
   (naive  : linear assoc list + sort)
```

…and then runs the program with an actual `BTreeMap` behind `counts`.

## How selection works

For every variable bound by `let v = collection()`, the analyzer collects the
set of operations applied to `v` anywhere in the program, then maps that
**usage profile** to a concrete representation:

| What you do with it                          | Inferred kind   | Specialized structure | Naive baseline        |
|----------------------------------------------|-----------------|-----------------------|-----------------------|
| `put` / `get` / `keys`                       | Map             | `HashMap`             | linear assoc list     |
| …plus `min` / `max` / `sorted`               | Ordered map     | `BTreeMap`            | assoc list + sort     |
| `add` / `has` / `del`                        | Set             | `HashSet`             | linear scan           |
| …plus `min` / `max` / `sorted`               | Ordered set     | `BTreeSet`            | linear scan + sort    |
| `append` / `at` / `set_at` / `push` / `pop`  | Sequence/Stack  | `Vec`                 | `Vec`                 |
| `enqueue` / `dequeue` / `front`              | Queue           | `VecDeque` (O(1) pop) | `Vec` (O(n) shift)    |

The rule of thumb: *associative access wins over positional wins over set
membership*, and any demand for ordering upgrades a hash structure to its
ordered (tree) counterpart.

## Two backends, identical meaning

Every kind has two implementations that obey the exact same semantics:

* **specialized** — the best structure for the observed usage, and
* **naive** — a linear structure of the same family.

Because they agree on behavior, the program prints the same thing either way;
only the speed differs. That difference is the whole point, and `bench` measures
it on a membership-heavy workload:

```
$ autostruct bench examples/bench.shape
  naive (linear structures) : 107.5ms
  specialized (inferred)    : 6.5ms
  speedup                   : 16.6x faster
```

## Compile to specialized JavaScript

`autostruct js` lowers each abstract collection to the matching native JS
structure — `Set`, `Map`, `Array`, or a small `Queue` class — emitting native
calls (`.has`, `.set`, `.push`, `.dequeue()`), so the generated code is already
specialized with no generic dispatch:

```
$ autostruct js examples/bfs.shape
let graph = new Map();
let visited = new Set();
let frontier = new Queue();
let order = [];
...
```

The generated JS produces byte-for-byte the same output as the Rust
interpreter (the test suite checks this for every example).

## Usage

```
autostruct run     <file.shape>            analyze, then run with inferred structures
autostruct run     <file.shape> --naive    run with naive linear structures instead
autostruct analyze <file.shape>            print the inference report only
autostruct js      <file.shape>            compile to specialized JavaScript
autostruct bench   <file.shape>            specialized vs. naive, with a speedup number
```

`autostruct <file.shape>` is shorthand for `run`.

## The language

Tiny, dynamically-typed, imperative:

* values: integers, booleans, strings, `nil`, and collections
* `let x = …`, assignment, `if/else`, `while`, `for x in coll`, `break`,
  `continue`
* arithmetic `+ - * / %`, comparisons, `and` / `or` / `not`, string `+`
* list literals `[1, 2, 3]` (always a dynamic array)
* `collection()` — the abstract collection whose representation is inferred
* builtins: `add has del put get keys append at set_at push pop peek enqueue
  dequeue front min max sorted size len range str print`

Iteration order of a hash set/map is unspecified by design; use `sorted(c)` when
order matters (the examples do).

## Layout

```
src/lexer.rs       tokenizer
src/parser.rs      recursive-descent parser
src/ast.rs         syntax tree
src/analysis.rs    usage analysis + representation selection  ← the interesting part
src/value.rs       runtime values + every concrete structure (specialized & naive)
src/interp.rs      tree-walking interpreter
src/codegen_js.rs  compile-to-specialized-JavaScript backend
examples/*.shape   word count, dedup, BFS, ordered set, stack, benchmark
tests/cli.rs       end-to-end tests (inference + output + backend agreement)
```

## Build & test

```
cargo build --release
cargo test
./target/release/autostruct run examples/word_count.shape
```
