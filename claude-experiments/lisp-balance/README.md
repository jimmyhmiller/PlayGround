# lisp-balance

A small, dependency-free Rust library that takes (possibly broken) Lisp / Clojure
source and returns a structurally balanced version. Missing closing delimiters
are inserted using indentation as the cue; stray or mismatched closers are
dropped. Strings and `;` line comments are respected.

The intended use case is post-processing source produced by tools (LLMs,
template generators, partial edits) that frequently get bracket counts wrong
even when the indentation is right.

This was extracted from [`paredit-like`](../paredit-like) — specifically the
algorithm that lives in `src/parinfer_simple.rs` there. See `PARINFER_FIX.md`
in that project for the original motivation (the upstream `parinfer-rust`
mis-handled some LLM-style brace indentation).

## Usage

```toml
[dependencies]
lisp-balance = { path = "../lisp-balance" }
```

```rust
use lisp_balance::balance;

let fixed = balance("(defn foo\n  (+ 1 2");
assert_eq!(fixed, "(defn foo\n  (+ 1 2))");
```

The single entry point is the free function:

```rust
pub fn balance(source: &str) -> String;
```

It always returns a string. Already-balanced input is returned unchanged.

## Examples

Auto-close at end of file:

```text
in:  (defn foo
       (+ 1 2
out: (defn foo
       (+ 1 2))
```

Drop extra/mismatched closers:

```text
in:  (foo bar}})
out: (foo bar)
```

Indentation drives nesting (LLM-style 2-space indent inside a brace map):

```text
in:  (attributes {
       :sym_name @test
       :function_type (!function (inputs) (results i32))
     })
out: (attributes {
       :sym_name @test
       :function_type (!function (inputs) (results i32))
     })
```

Strings and comments are not touched:

```text
in:  (foo "{bar}" baz)        ; unchanged
in:  (foo bar) ; {unclosed    ; unchanged — `{` is inside a comment
```

## Algorithm

Single pass over the source, maintaining a stack of open delimiters. For each
delimiter:

- **Open `(`, `[`, `{`** — push, record the line indent and column of the open.
- **Close `)`, `]`, `}`** — if it matches the top of the stack, pop and emit;
  if mismatched (or the stack is empty), drop it. When the line ends with a
  run of two or more closers, indentation is consulted to decide whether each
  closer in that tail should be emitted now or held over for a later line.
- **End of line** — auto-close any opener whose `line_indent` is greater than
  the next non-blank, non-comment line's indent. This is what inserts missing
  closers on dedent.
- **End of file** — close anything still open, appended to the last line, in
  Lisp convention.

A few subtle points worth knowing if you read the source:

- Within a closing-tail (multiple closers at line end), the rule uses
  `max(line_indent, col_pos)` of the opener so that closers don't get
  mis-attributed across forms.
- For end-of-line auto-close, the rule uses `line_indent` only (not
  `col_pos`). Using `col_pos` here would wrongly close forms whose siblings
  are indented under the form head rather than under its first argument —
  e.g. an `scf.if` with two `(region ...)` siblings.
- An open `"` at end of input suppresses EOF auto-close: an unclosed string
  literal is left alone rather than having `)`/`]`/`}` injected into it.

## Limitations

- `;` line comments and `"..."` strings only. Block comments (`#| ... |#`,
  `#_(...)`), reader macros (`#"..."`, `#'`, etc.), Janet long strings, and
  Hy bracket strings are not specially handled.
- Indentation is taken at face value. Tabs are treated as a single
  whitespace character; if you mix tabs and spaces inconsistently the result
  may not be what you want.
- Line endings are normalised to `\n`. CRLF input is split on `\n`/`\r\n`
  and rejoined with `\n` only — round-tripping CRLF source through
  `balance` will lose the `\r`s.
- This is a heuristic, not a parser. It will produce *some* balanced output
  for almost any input, but the structure it picks reflects the indentation
  it sees. Garbage indentation in, garbage structure out.

## Tests

```sh
cargo test
```

The test suite covers the cases from the original extraction (Clojure
defn/let/cond/threading, nested maps, mixed delimiters, the `scf.if` sibling
regions case, type annotations, etc.).

## License

MIT
