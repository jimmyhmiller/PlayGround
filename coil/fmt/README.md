# coil fmt — a pretty printer for Coil, written in Coil

A source formatter in the style of [Prettier](https://prettier.io) — lay a form out
flat if it fits the target width, otherwise break it — but respecting the
**Clojure indentation conventions** for how Lisp code lines up. Written entirely in
Coil, over the standard library.

```
coil run fmt/fmt.coil -- <file.coil>            # print the formatted source
coil run fmt/fmt.coil -- --check <file.coil>    # exit 1 if not already formatted
coil run fmt/fmt.coil -- --write <file.coil>    # reformat the file in place
```

Target width is 120 columns.

## What it preserves

Formatting only ever changes whitespace and line breaks. Comments, blank lines
(collapsed to at most one), and every atom/string/number are kept **verbatim** —
verified against the whole codebase by `fmt/check.sh` (token-equivalence + idempotence).

## Reflow policy (hybrid)

Width-driven like Prettier, but it respects two author signals:

- **blank lines** between forms are kept (at most one);
- a form the **author already split across lines stays split** (its group is forced
  to break). A form the author wrote on one line is re-flowed by width.

Indentation is always normalized.

## Layout conventions

Per the Clojure style guide, the head symbol decides how a broken form lines up:

| kind                                   | example |
|----------------------------------------|---------|
| function call (default) — align under the first arg | `(some-fn a`<br>`         b)` |
| macro / body form — 2-space body indent | `(when test`<br>`  body)` |
| `defn` — signature on the head line, body +2 | `(defn f [x] (-> i64)`<br>`  body)` |
| `let`/`loop`/… — binding vector laid out in **pairs** | `(let [a 1`<br>`      b 2]`<br>`  body)` |
| `cond`/`case` — clauses in **pairs**, aligned | `(cond t1 r1`<br>`      t2 r2)` |

A comment is always the last thing on its line: nothing is ever joined onto a
line after a comment.

## Structure

| file        | what |
|-------------|------|
| `cst.coil`  | reader → a Concrete Syntax Tree that preserves comments, blank lines (`nl-before`), and verbatim token text |
| `doc.coil`  | a Wadler/Prettier **Doc** algebra + width-driven renderer (`DNest` relative indent, `DAlign` anchor-to-column, `DGroup` flat-if-fits) |
| `rules.coil`| lowers the CST to a Doc, applying the layout conventions above |
| `fmt.coil`  | the CLI |
| `dump.coil` | debug: print the CST node tree (used by the safety gate) |
| `check.sh`  | safety gate: token-equivalence + idempotence over the codebase |
| `sample.coil` | a small fixture exercising comments, prefixes, strings, bindings |

## Known limitations (v1)

- Trailing comments are placed two spaces after the code, not column-aligned to
  match neighboring lines.
- `cond`/`case` results that don't fit drop to the next line at the clause
  indentation, rather than a `+2` hang under their test.
- `if` uses a 2-space body indent rather than aligning both branches under the test.
