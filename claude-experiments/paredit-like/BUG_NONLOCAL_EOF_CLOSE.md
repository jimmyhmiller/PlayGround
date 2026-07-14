# Bug: `balance` closes a missing top-level paren at EOF, silently nesting the rest of the file

**Severity: high** — `balance --in-place` produces paren-balanced but structurally
corrupted output, and the visible diff is a single character far from the edit, so it
survives review. Hit in real use (scry module-system work, 2026-07-13): an agent edited
one function in a ~1400-line Coil file, left it one `)` short, ran
`paredit-like balance --in-place`, and got a "stray" extra `)` at the END of the file
inside an untouched function (`scry-dump-bytecode`) — every top-level form after the
edit had been silently nested inside the unclosed one. Cost a bisect to find.

## Minimal repro (9 lines)

Input — `alpha` is missing its final `)`; `beta` and `gamma` are complete, column-0
top-level forms (also checked in as `test_nonlocal_eof_close.lisp`):

```lisp
(defn alpha [x] (-> i64)
  (+ x 1)

(defn beta [y] (-> i64)
  (* y 2))

(defn gamma [z] (-> i64)
  (- z 3))
```

Run: `paredit-like balance test_nonlocal_eof_close.lisp --in-place`

**Expected** (Parinfer indent-mode: a new column-0 opener means everything open must
close first — close `alpha` at the end of its own body):

```lisp
(defn alpha [x] (-> i64)
  (+ x 1))

(defn beta [y] (-> i64)
  (* y 2))

(defn gamma [z] (-> i64)
  (- z 3))
```

**Actual** (`alpha` left open; the missing `)` appended after `gamma`, so `beta` and
`gamma` are now *children of `alpha`*):

```lisp
(defn alpha [x] (-> i64)
  (+ x 1)

(defn beta [y] (-> i64)
  (* y 2))

(defn gamma [z] (-> i64)
  (- z 3)))
```

The output is paren-balanced (so nothing downstream complains) but semantically wrong,
and the change lands at the last line of the file rather than at the broken function.
On the real 1400-line file, dropping one trailing `)` from ANY of ~40 sampled lines in
the first 1300 lines produced exactly this: the only changes were the perturbed line
and the file's last line.

## Root cause

`src/parinfer_simple.rs`, `balance()`, the auto-close-on-dedent loop (~line 286):

```rust
let opener_indent = open_info.line_indent;
if next_indent < line_indent && opener_indent > next_indent {
    ... close ...
}
```

`opener_indent > next_indent` is a **strict** comparison and `next_indent` has a floor
of 0 (that's also `find_next_indent`'s EOF value). A top-level opener has
`line_indent == 0`, so `0 > 0` is false and **an opener at column 0 can never be closed
by the dedent rule** — no matter how many column-0 top-level forms follow. The open
delimiter rides the stack to the end-of-file fallback (~line 305), which appends all
remaining closers to the last line "(Lisp convention)". That fallback is what turns a
local mistake into whole-file nesting.

Note `balance()` early-returns the source verbatim when the file is already well-formed
(`is_well_formed`, ~line 129), so this bug only fires on inputs that genuinely need
fixing — i.e. precisely the mid-edit files the tool exists for. That also means a fix
here can't regress balanced files.

## Suggested fix direction

The missing rule is Parinfer indent-mode's core one: **a line whose indentation is ≤ an
open delimiter's opening indentation, and which begins a new form (an opener token, not
a closer/continuation), closes that delimiter first.** Concretely, in the dedent loop,
when the *next* non-empty line starts with an opening delimiter at `next_indent ≤
opener_indent` (the `next_line_has_closers` guard already excludes closer-led lines),
close the opener at the end of the current line — i.e. for that case the comparison
must be `opener_indent >= next_indent`, not `>`.

The conservative variant, if `>=` in general feels risky for the odd
continuation-dedented-to-column-0 style (the llvm-ir-ish case the comments guard): keep
`>` as-is, and add one special case — when the next line is at column 0 AND starts with
an opener, drain the entire stack onto the current line. Column-0 openers starting new
top-level forms is as unambiguous as indentation signals get, and it exactly covers the
observed corruption.

Either way, please add regression tests for:
1. the repro above (missing close in form N followed by complete top-level forms),
2. the same with the broken form in the middle of 3+ forms,
3. locality: balancing a file with one broken function changes only that function.

## How it was reproduced / fuzzed

Perturbation fuzz over a real 1400-line valid file: for a sample of lines ending in
`)`, drop that one char, run `balance`, diff against the original. Every "drop" sample
produced the non-local EOF change described above ("add an extra `)`" perturbations did
not — the closer-acceptance path handles those locally). Happy to share the harness;
it's ~30 lines of Python.
