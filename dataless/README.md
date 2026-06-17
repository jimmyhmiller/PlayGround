# dataless

A modern, working realization of **R. M. Balzer, "Dataless Programming"**
(RAND Memorandum **RM-5290-ARPA**, July 1967).

> "The programmer should be able to construct his program in terms of the
> logical processing required without regard to either the representation of
> data or the method of accessing and updating. This concept we call *Dataless
> Programming*."  — RM-5290, p. 2

You write the program **once**, as pure logic. How the data is actually
stored — array, linked list, doubly-linked list, or even *computed on the fly
by a function* — lives in a **separate declarations file** and can be swapped
without changing a single character of the program. Same program, same output;
only the performance and storage change.

```
dataless compare examples/shapes.dl \
    --decl examples/shapes_array.decl \
    --decl examples/shapes_list.decl \
    --decl examples/shapes_double.decl
```
```
RESULT: every representation produced identical output. ✓
  shape count: 30
  total area: 9455
  ...
```
…while the trace shows the array doing 60 positional steps and the lists 495.
The program file was never touched.

## What the paper actually requires (and where this meets it)

The first thing I built for this prompt was the *wrong idea* — a language that
*infers* one fixed data structure from how operations are named. That is almost
the opposite of the paper. Balzer's thesis is **separation**: the program is
representation-free, and the representation is supplied *separately* and is
*swappable*. This implementation is built to that thesis.

| RM-5290 criterion | Where it lives here |
|---|---|
| **Single canonical form** `name(expr)` for *both* data and function references (p. 7) | `Expr::Ref`; `name(handle)` resolves against the handle's collection — stored field or `computed` function, indistinguishably |
| Representation **declared separately**, swappable without editing the program: "changing a collection from an array to a list … merely changes a data declaration" (p. 7) | `.dl` program file vs. `.decl` declarations file; `dataless compare` proves identical output across `ARRAY` / `LIST` / `DOUBLE_LIST` |
| A name can be a stored datum **or a computed function**, switched in the declaration (p. 8) | `computed area(c) = …`; writes to a computed reference are ignored, reads run the function (`examples/circle*.decl`) |
| Built-in representations: ARRAY, LIST, DOUBLE LIST, RING, … (pp. 8–9) | `repr.rs`: real `ARRAY`, `LIST` (forward links), `DOUBLE_LIST` (forward+back), each with honest costs |
| **INSERT / DELETE** defined for all representations (p. 11) | `insert <coll>`, `insert after <h> in <coll>`, `delete <h>` |
| **FOR-clause**: `… FOR EACH coll SUCH THAT (cond)` (p. 13) | `for each <coll> such that (cond) { … }` |
| **Search expressions**: "there exists … such that" (p. 37) | `there exists x in <coll> such that (cond)` |
| **Generators**: lazy cursors yielding the next matching member on request (p. 12) | `generate g over <coll> such that (cond)` + `next g` |
| **Implied qualification**: a bare field means the current member (p. 38) | inside `for each` / generators / `computed` bodies, bare `area` ≡ `area(current member)` |
| **STATE statements**: `{ON|WHENEVER}(bool) stmt` fired by instrumented updates (p. 9) | `whenever (cond) { … }`, fired on false→true after every field update |

## The canonical reference, concretely

In the program you only ever write `name(handle)`:

```
area(c)            # read
area(c) = 50       # write (ignored if `area` is computed)
```

Whether `area` is a stored field or a function is decided in the `.decl`:

```
# stored:
collection circles as ARRAY { radius : int, area : int }

# computed — the program is byte-for-byte identical:
collection circles as ARRAY { radius : int }
computed area(c) = 314 * radius(c) * radius(c) / 100
```

A member handle (the paper's "bug"/pointer) dereferences in O(1) for *every*
representation. What differs is **positional lookup** `member(coll, k)` and
**insert/delete in the middle**:

| | ARRAY | LIST | DOUBLE_LIST |
|---|---|---|---|
| `member(coll, k)` | O(1) | O(k) | O(k) |
| `insert after handle` | O(n) | O(1) | O(1) |
| `delete handle` | O(n) | O(n) | O(1) |
| `field(handle)` | O(1) | O(1) | O(1) |

These are real (the structures are really linked); `--trace` reports the work.

## Usage

```
dataless run     <program.dl> --decl <decls.decl> [--trace]
dataless compare <program.dl> --decl <a.decl> --decl <b.decl> [--decl ...]
dataless decls   <decls.decl>
```

`compare` is the demonstration: it runs the one program under every declaration
file and asserts the outputs are identical, printing the differing costs.

## The language

A program (`.dl`) is a sequence of statements — no type or representation ever
appears:

* `let x = …`, assignment, `if/else`, `while`, `repeat <n> with <v> { … }`
* arithmetic `+ - * / %`, comparisons, `and` / `or` / `not`, text `+`
* `print(...)`
* canonical references `name(handle)`; `it`; bare fields (implied qualification)
* `size(coll)`, `member(coll, pos)`, `insert coll`, `insert after h in coll`,
  `delete h`
* `for each coll [such that (cond)] { … }`
* `there exists x in coll such that (cond)`
* `generate g over coll such that (cond)` / `next g`
* `whenever (cond) { … }`

A declarations file (`.decl`):

```
collection <name> as ARRAY|LIST|DOUBLE_LIST {
    <field> : int|text|bool [= <const>]
    ...
}
computed <name>(<param>) = <expression>
```

## Layout

```
src/value.rs    runtime values + member handles ("bugs")
src/repr.rs     ARRAY / LIST / DOUBLE_LIST backends with real costs   ← the point
src/lexer.rs    shared tokenizer
src/ast.rs      program + declaration syntax trees
src/parser.rs   recursive-descent parser for both files
src/interp.rs   reference resolution, implied qualification, generators, STATE
src/main.rs     run / compare / decls
examples/       shapes (representation swap), circle (data vs function),
                accounts (generators, search, STATE, insert/delete)
tests/cli.rs    asserts identical output across representations + each facility
```

## Build & test

```
cargo build --release
cargo test
dataless compare examples/circle.dl \
    --decl examples/circle_stored.decl --decl examples/circle_computed.decl
```

## Honest limitations

This captures the *core* of the language, not all of 1967 PL/1. RING /
DOUBLE_RING and nested STRUCTURE representations from the paper are not
implemented (the three list/array reps already demonstrate representation
independence); `ON`-units are limited to the `WHENEVER` form evaluated in the
declaring scope; and there is one global member-handle space. The load-bearing
claim — *one unchanged program, separately-declared swappable representations,
identical results* — is implemented and tested.
