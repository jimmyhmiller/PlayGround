# Autofix — Design

**Status: SHIPPED.** `(suggest NODE MSG REPLACEMENT)` (code op 44) and `coil lint
[--fix|--diff]` are implemented; the first rule on top of them is the nested-`if`
lint in `metaprog-poc/condlint.coil`. Builds on `METAPROGRAMS.md` (the checker hook,
`warn`/`report`) and `SEMANTIC_METAPROGRAMS.md` (checkers run post-typecheck against
the authoritative model). See **§12 — What shipped** at the end for where each piece
lives and the three places reality differed from this design.

## Goal

Today `metaprog-poc/lint.coil` says *"prefer a clean operator over `icmp-*`"* and
stops there. A human then edits the file. We want the rule that found the problem to
also **repair the source**:

```
coil lint app.coil --use lint.coil --fix
```

and have `app.coil` on disk come back with `(icmp-lt x y)` rewritten to `(< x y)` —
every other byte of the file, including comments and the author's formatting,
untouched.

## Non-goal: this is not `(transform …)`

There is already a whole-program rewriter. It is the wrong tool, and the distinction
is the reason autofix needs a new mechanism rather than a new demo:

| | `(transform FN)` | autofix |
|---|---|---|
| rewrites | the in-memory program, every build | the file on disk, on request |
| the source file | is never what compiles | is the only artifact |
| means | "in this dialect, `inc` *is* `iadd`" | "you should have written `<`" |
| audience | the compiler | the author |

A transform that "fixed" `icmp-lt` would compile something other than what you
wrote, forever, silently. An autofix hands you a diff and gets out of the way. The
two must not be conflated: **a normal `coil build` must never mutate your source.**

---

## 1. Everything needed already exists

As with the semantic model, the work is not "compute new information" — it is
"expose what the compiler already has."

| What a fixer needs | Where it already is |
|---|---|
| The byte range to replace | `Sexp {lo, hi, source}` (`reader.coil:30`) — on **every** node |
| Whether a node is real source | `Sexp.ctxt` (0 = written by hand, ≠0 = macro-expanded) + `code-from-user?` |
| Node identity across rounds | `Sexp.nid` (S0, shipped) |
| The source text to splice into | `LS.sources` → `sm-src-text` (`diag.coil`) |
| A diagnostic carrying a span | `Diag {lo, hi, msg, source, ctxt}` (`ast.coil:199`) |
| A collector that runs to completion | `warn-list` / `warn-push` (`comptime.coil:653`) |
| A pretty-printer with a verbatim leaf | `fmt/doc.coil` — Wadler `Doc`, `DText` |
| Types, to know a fix is valid | `type-of`, `code-decl` (S1/S2, shipped) |

So the additions are: one rule-author op, one side table, one printer, and one CLI
mode.

---

## 2. Why fixes need the semantic layer (the motivating bug)

The obvious first rule is already in the tree, and **it is not safely fixable as
written**. `lint.coil` flags every `(icmp-lt a b)` and suggests `<`. But `<` is not a
primitive — the prelude defines it as an `Ord` **trait method**:

```coil
(deftrait Ord [Self] (<  [(a Self) (b Self)] (-> bool)) …)
(impl Ord i64 (<  [(a i64) (b i64)] (-> bool) (icmp-lt a b)) …)
```

`Ord` is implemented for `i64`, `f64`, and `(ptr T)` — but *not* for `u8`, `i32`, or
any other narrow width. So `(icmp-lt a b)` on two `u8` values is perfectly good code
that the "obvious" fix would rewrite into a program that **does not compile**
(`'u8' does not implement 'Ord'`).

A purely syntactic fixer cannot see this. A checker can, because checkers run after
typechecking and `(type-of NODE)` returns the inferred type:

```coil
(defn ord-operand? [(n Code)] (-> bool)
  (let [t (type-of n)] (or (code-eq t `i64) (code-eq t `f64))))
(defn fixable-icmp? [(f Code)] (-> bool)
  (if (ord-operand? (code-nth f 1)) (ord-operand? (code-nth f 2)) false))
```

(Pointers used to be the example here — but `(icmp-lt p q)` on pointers never
compiled either, so no such code existed to break. Narrow integer widths are the real
case. Verify the example, not just the argument.)

**This is the load-bearing argument for the whole design: report is syntactic, but
fix is semantic.** A fix is a claim about meaning, so it belongs in the layer that
has meaning. That, in turn, is why fixes are authored inside checkers and not in
some separate `.lintrc` of regexes.

---

## 3. The rule-author API

One new op, mirroring `warn` (op 32) exactly:

```coil
(suggest NODE MSG REPLACEMENT)   ; warn at NODE; propose replacing NODE with REPLACEMENT
```

`REPLACEMENT` is a **`Code` value, not a string.** The rule builds it with the same
quote/unquote vocabulary it already uses:

```coil
(defn lint-node [(f Code)] (-> Code)
  (if (is-icmp? (code-nth f 0))
      (if (fixable-icmp? f)
          (suggest f "prefer a clean operator over icmp-* on i64"
                   `(~(clean-op (code-nth f 0)) ~(code-nth f 1) ~(code-nth f 2)))
          (warn f "icmp-* on a non-Ord type"))   ; flagged, but no fix offered
      …))
```

Code, not strings, for three reasons: string-building in a checker is exactly where
the closure limitation bites (`METAPROGRAMS.md`: "checkers can't call imported string
functions"); a string fix is unhygienic and unparseable-until-too-late; and `Code`
carries the spans that make §4 work.

Two more forms cover what replacement alone cannot express:

```coil
(suggest-delete NODE MSG)         ; remove NODE entirely — unused import, dead form
(suggest-maybe NODE MSG REPL)     ; a fix that may change behavior (see §7)
```

`suggest-delete` is not sugar. The canonical autofix — "remove the unused import" —
is a deletion, and no `Code` value denotes "nothing".

### What a rule may NOT do

A checker records suggestions. It does **not** open files, and it has no way to. The
op pushes onto a side table and returns; only the `--fix` driver (§5) ever writes to
disk. This keeps `coil build` pure by construction rather than by convention.

---

## 4. The heart: verbatim-if-spanned rendering

The replacement is `Code`, but the file is text. Rendering `Code` back to text is
where a naive fixer destroys the file.

Consider `` `(< ~(code-nth f 1) ~(code-nth f 2)) ``. Two of those three nodes are
**the author's original nodes**, unquoted straight out of the program. If we
pretty-print the whole tree we reprint those operands from the AST — losing any
comment inside them, any string escape the author chose, any line break they wanted,
and any formatting in a subtree that might be hundreds of lines long.

The rule that avoids this falls out of `Sexp` for free:

> **When rendering a replacement node: if the node has a real source span
> (`source ≥ 0`, `ctxt = 0`) pointing into a file we have text for, emit its original
> bytes `[lo,hi)` verbatim. Otherwise it is freshly constructed — pretty-print it.**

So `` `(< ~a ~b) `` renders as: `(< ` + *the exact original bytes of `a`* + ` ` +
*the exact original bytes of `b`* + `)`. The operands are preserved byte-for-byte no
matter how large or how comment-laden. Only the four bytes that actually changed are
new. This is what `clippy`'s `snippet()` does by hand at every call site; here it is
a property of the printer, because every node already knows where it came from.

Implementation: a `Sexp → Doc` printer over `fmt/doc.coil`, where the spanned case is
a single `DText` of the original slice and the fresh case reuses `fmt/rules.coil`'s
existing layout. The whole replacement is rendered inside a `DAlign` anchored at the
target span's starting column, so a fix that has to break across lines indents to
where the form actually sits. Then only that span is spliced; **the rest of the file
is not reformatted.** A fix must never hand someone who does not run `fmt` a
whole-file diff.

---

## 5. The fix table and the apply loop

### The side table

Mirrors `warn-list`, and is dropped on the floor by every build that isn't `--fix`:

```coil
(defstruct Fix [(source i64) (lo i64) (hi i64) (repl (ptr Sexp))
                (kind i64)        ; 0 = replace, 1 = delete
                (applic i64)])    ; 0 = safe, 1 = maybe
```

### Filtering (before anything is applied)

A suggestion is **dropped**, silently and without ceremony, if:

1. **`ctxt ≠ 0`** — the node came from a macro expansion. Its span points at the
   macro's definition or its call site; "fixing" it would corrupt the macro or write
   nonsense into an unrelated form. This is the trap `clippy` guards with
   `in_external_macro`, and here it is one field compare. **This filter is not
   optional and not configurable.**
2. **`code-from-user?` is false** — the node is bundled stdlib (`<…>`); there is no
   file to edit.
3. The node's file is not among the files `--fix` was pointed at.

Note that (1) also disposes of the `nid`-vs-span sharp edge from
`SEMANTIC_METAPROGRAMS.md` §7.3: a macro-duplicated subtree shares a span, but it
also has `ctxt ≠ 0`, so it is never an edit target. Fixes only ever touch spans that
are unique by construction — the ones a human typed.

### Applying

Within one file, sort surviving fixes by **descending `lo`** and apply back-to-front,
so each splice leaves earlier offsets valid. Overlapping fixes (an outer form and an
inner one both rewritten) cannot both be applied against the same text: take the
first, **drop the conflicting ones for this round**, and let the next round find them
again on re-analysis.

Which makes the top level a **fixpoint with a fuel guard** — the same shape as macro
expansion and the transform fixpoint (`run-transform-fixpoint`), and the same shape
as `eslint --fix`'s ten passes:

```
loop (max 8 rounds):
  compile → run checkers → collect fixes → filter (§5) → resolve overlaps
  if no fixes → done
  write files; if a round changes nothing → done
if fuel exhausted → report the remaining diagnostics and exit non-zero
```

Fuel exhaustion is a real outcome and must be reported, not swallowed: it means two
rules disagree (A rewrites X→Y, B rewrites Y→X) and the pair is broken. Ping-ponging
silently until a timeout is the failure mode to avoid; naming the two rules is the
feature.

---

## 6. The gate

The tree must be no worse after `--fix` than before. Two levels:

**Level 1 — it still compiles (always on).** After each round, the rewritten file
must read + resolve + typecheck. If it doesn't, **revert the round** and report which
rule produced code that doesn't compile. Fail closed: the working tree is always at
least as valid as it started. (This is the gate-and-revert discipline that
`hivemind` uses, and for the same reason: an agent that can only land what a gate
accepts cannot make things worse.)

**Level 2 — `--verify`, opt-in.** Compile the artifact before and after and compare
the emitted code. Byte-equality proves the fix changed nothing but the text.

An honest caveat on level 2, to be settled in F2 rather than assumed: `<` on `i64` is
a **trait-method call** that lowers to `icmp-lt` only after the impl is inlined. So
`(icmp-lt x y)` and `(< x y)` are near-certainly identical in the final optimized
artifact, but may well differ in an unoptimized IR dump (a call vs. a raw op). The
comparison must therefore run at the final artifact level, and `--verify` stays
opt-in until measured across the corpus. Do not ship a gate whose green depends on an
unverified claim about the optimizer.

---

## 7. Applicability

Not every fix is a safe mechanical rewrite; `rustc` learned this and grades every
suggestion. Two levels are enough:

- **safe** (`suggest`) — semantics-preserving. `--fix` applies it.
- **maybe** (`suggest-maybe`) — plausible but may change behavior. **Never** applied
  by `--fix`; shown in the diagnostic as `help: try: …` and applied only under
  `--fix --risky`, which additionally implies `--verify`.

The default diagnostic gains a suggestion line whether or not you ever run `--fix`,
which is most of the value for free:

```
app.coil:5:7: warning: prefer a clean operator over icmp-* on i64
  |
5 |   (if (icmp-lt x y) …
  |       ^^^^^^^^^^^^^
help: try: (< x y)
```

---

## 8. CLI

```
coil lint app.coil --use lint.coil            # report + `help: try:` lines (no writes)
coil lint app.coil --use lint.coil --fix      # apply safe fixes in place
coil lint app.coil --use lint.coil --fix --diff     # print a unified diff, write nothing
coil lint app.coil --use lint.coil --fix --verify   # gate each round on artifact equality
coil lint app.coil --use lint.coil --fix --risky    # also apply `maybe` fixes (implies --verify)
```

`lint` is a thin sibling of `fmt-cmd` (`driver.coil:1024`); `--use` already exists
(`driver.coil:570`) and already means "import this metaprogram module without editing
the source". `--fix` refuses to write to a dirty git tree unless `--allow-dirty`
(clippy's rule, and a good one: the undo button for an autofix is `git checkout`, so
insist the button exists).

---

## 9. Delivery — additive, oracle-gated

Each phase ships alone and holds the rebootstrap fixpoint. Nothing here changes a
dump, so the oracle stays byte-exact throughout.

- **F0 — `(suggest …)` records, nothing applies.** The op (next free code after 36),
  the `Fix` table, the `ctxt`/user filters, and the `help: try:` diagnostic line —
  rendered by a first-cut printer that pretty-prints everything. No `--fix`, no
  writes. Already useful: every existing checker can offer suggestions, and the
  filters get exercised before anything can damage a file.
- **F1 — the verbatim printer.** `Sexp → Doc` over `fmt/doc.coil` with the
  spanned-node `DText` rule (§4). Test: for every node in the corpus, rendering a
  replacement built as `` `~NODE `` must reproduce the original bytes exactly. That
  test is the whole design in one assertion, and it can run over the compiler's own
  source (~30 files, the biggest Coil corpus we have).
- **F2 — `coil lint --fix`.** The apply loop, the fixpoint, level-1 gate, `--diff`,
  the dirty-tree guard. Settle the §6 caveat and decide whether `--verify` can be
  default.
- **F3 — a real rule set.** Port `lint.coil` to `suggest` with the `type-of` guard
  from §2, then the rules that pay for the machinery: unused import
  (`suggest-delete`), `(if c x 0)` → `(when c x)`, redundant `do`. The dogfood target
  is `coil lint --fix` over `selfhost/src/*.coil` producing a diff a human would
  approve.

**Tests, in the repo's style:** (a) the F1 round-trip identity above; (b)
**idempotence** — `--fix` twice produces a byte-identical file, which is `fmt`'s own
discipline (`coil fmt --check`); (c) **token-equivalence outside the fixed span** — the
untouched bytes of the file are untouched, verbatim; (d) artifact equality on a
`--verify` corpus.

---

## 10. Open questions

1. **Warn here, fix there.** `(suggest NODE MSG REPL)` warns and fixes at the same
   node. Highlighting the head symbol while replacing the whole call needs a 4-arg
   form. Cheap to add; unclear it's wanted. Defer until a rule asks.
2. **Rule identity.** For `--fix --only=RULE`, `# allow(RULE)` comments, and naming
   the culprits on fuel exhaustion, a fix needs to know which rule produced it. There
   is no rule-name concept today — a checker is just a registered function. Adding
   the registered function's name to `Fix` is nearly free and probably right.
3. **Suppression.** `# allow(…)` comments are the obvious mechanism, but comments do
   not survive into `Sexp` — only `fmt`'s CST sees them. Suppression may have to be
   span-based (look up the line's preceding comment in the source text) rather than
   node-based.
4. **Multi-file atomicity.** If file A's fix applies and file B's round fails the
   gate, do we revert A? Proposal: yes — revert the whole round, so the tree is
   always some clean fixpoint state, never a partial one.
5. **`--verify` default.** §6. Measure first.

---

## 11. Summary

Every `Sexp` already carries `(source, lo, hi, ctxt, nid)`, which is exactly a fix's
edit target plus the safety check that makes it honest. The design is: **(1)** rules
emit `(suggest NODE MSG REPLACEMENT)` where the replacement is `Code`, so fixes are
written in the language's existing quote/unquote vocabulary and can consult
`type-of`/`code-decl` — because a fix is a semantic claim, not a syntactic one;
**(2)** the printer renders any node with a real source span as its **original
bytes**, so a fix touches only what it changes and preserves comments and formatting
everywhere else; **(3)** `ctxt ≠ 0` is dropped unconditionally, so macro-generated
code is never edited; **(4)** application is a gated fixpoint that reverts any round
that fails to compile, so the tree never gets worse. Checkers only ever record; only
`coil lint --fix` writes.

---

## 12. What shipped

`(suggest NODE MSG REPLACEMENT)` is code op **44**, parsed alongside `warn` and
handled by the same function (`comptime.coil::cop-warn-or-suggest`) — the diagnostic
half of the two is identical, so it is written once. It records into a `Suggestion`
table that mirrors `warn-list` and that only `coil lint --fix` ever reads; the ctxt/
user filters run at record time (`cop-record-fix`). The verbatim-if-spanned renderer
is `sug-emit!` in the same file, and `sug-print-help` puts a `help: try:` line under
every warning a suggestion produced — so an ordinary build shows the fix whether or
not anyone runs `--fix`.

`coil lint` (`driver.coil::lint-cmd`) is a two-phase command: a quiet fuel-guarded
apply loop (`lint-fix-loop`), then one analysis pass that reports whatever is left.
Each round's analysis is the previous round's gate; a round that stops compiling is
reverted whole (`lint-restore`).

Three things the design got wrong, found by running it:

1. **A fix can silently delete a comment, and §5's filters do not catch it.** A
   replacement is assembled from the author's nodes, so anything *inside* a node
   survives — but in a Lisp the text *between* two nodes is a comment, and no `Code`
   value records it. Collapsing an `if` chain whose branches have comments between
   test and body would drop them.

   The first answer was to refuse: count the comment marks in the span being replaced
   and in the text replacing it (`sug-drops-comment?`, string-literal aware) and
   downgrade the fix to advice if any went missing. That is the wrong trade. On the
   self-hosted compiler it declined 36 of 178 real fixes, and "there is a comment near
   it" is not a reason to leave a staircase standing — it just moves the work back to
   a human for the cases most likely to *need* the comment kept.

   So the renderer carries them instead. It walks the ORIGINAL span alongside the
   replacement it is building: a cursor tracks how far into the source the emitted
   nodes have reached, and when the next node starts past the cursor, the comments in
   the skipped-over text are written out first (`sug-gap-before` / `sug-carry-at!`).
   Nodes are emitted in source order by any fix that reuses them, so the cursor only
   moves forward; a node that would move it backwards carries nothing rather than
   duplicating text. Three details the layout needs:

   * A comment ends its line, so a replacement carrying one can never be laid out
     flat. The width pass runs *before* the carry and cannot find out on its own, so
     a span containing a comment forces broken layout up front.
   * fmt distinguishes a comment BETWEEN a clause test and its body — which forces the
     hang: two spaces, the comment, then the body alone on the next line
     (`fmt/rules.coil` layout-clauses) — from one between clauses, which is its own
     row. The renderer reproduces both.
   * A comment after the last node the replacement reused still sits inside the span
     being replaced, and is written after the rendered form rather than going out with
     the closing parens.

   `sug-drops-comment?` is kept as a backstop for a rule whose fix drops a node
   outright, but it no longer fires on the `if`-chain rule.

2. **Verbatim text has to be re-indented, and the column must come from the buffer.**
   A form that moves left takes its multi-line body with it; copying the bytes
   unchanged strands the body at the old indent. `sug-put-reindent!` shifts every line
   after the first by the column delta (a leftward shift only removes spaces, so it
   cannot eat code). The column itself is derived from the output buffer
   (`sug-cur-col`), not threaded down the recursion — a node's true column depends on
   everything already written before it, and passing an estimate down is exactly how
   re-indentation goes wrong.

3. **The source table did not outlive a pipeline run.** `set-cli-sources` held a
   pointer to an `ArrayList` header living in an `LS` on the pipeline's stack — and
   the pipeline runs on its own pthread. Every existing consumer (`code-file`,
   `code-src`, `code-doc`) reads it *during* the run, so nothing had noticed; `--fix`
   reads it *after*, to know which files to write, and segfaulted. The box now holds
   the header by value.

A fourth thing the design did not anticipate: **checkers see the program after macro
expansion**, so every `cond`, `when` and `case` in the file has already become nested
ifs by the time a rule looks at it. A rule about `if` therefore needs to tell the
author's ifs from the expander's. That is code op **45**, `(code-macro? NODE)` — the
same `ctxt ≠ 0` test the fix filter applies internally, exposed to rules.

`--verify` (§6 level 2) and `suggest-maybe` (§7) are still unimplemented.

## Where the renderer still differs from `coil fmt`

The renderer aims to emit what `coil fmt` would, so a fixed file needs no reformatting,
and it does for the shapes the `if`-chain rule produces on ordinary code. Three shapes
still differ, all of them cases where fmt breaks a line the renderer keeps:

* a clause whose TEST spans several lines (fmt always drops the body to its own line),
* a standalone carried comment's column in a deeply nested clause,
* `let` binding vectors, which fmt pairs up and the renderer does not.

None change meaning. A large `--fix` run over a tree that is kept fmt-clean should be
followed by `coil fmt --write` on the files it touched; that is what was done for the
self-hosted compiler.
