# Metaprogram PoC — the four kinds, on today's machinery

A metaprogram is a Coil function that runs at compile time and operates on the
program. This shows the **Checker** kind (reject power) working now; the
**Transformer** kind is in `../gc-dialect-poc` (a GC as a metaprogram).

## checker.coil — a use-after-free checker that VETOES compilation

`check-uaf` is a macro `[Code…] -> Code` that scans a sequence for a `(free X)`
followed by a later use of `X`, and calls `error` (aborting the build) if it
finds one — otherwise it returns the body unchanged. Uses only Code builtins.

```sh
../coil run ok.coil    # clean → compiles & runs
../coil run bad.coil   # use-after-free → REJECTED: "check-uaf: use after free …"
```

Limitation: scans the straight-line top-level sequence handed to it (wrap a region
in `(check-uaf …)`). Making it *automatic* over the whole program — no wrapping —
is the compiler-level `(checker f)` hook in `docs/METAPROGRAMS.md` (Phase 1.1).

## condlint.coil — a lint that FIXES: nested ifs → cond

`lint-nested-if` finds an `if` chain of three or more tests and proposes the `cond`
it should have been, with `:else` as the final clause. It reports with
`(suggest NODE MSG REPLACEMENT)`, so the proposal is a real `Code` value built from
the author's own test and body nodes — which is what lets `coil lint --fix` splice it
in while reprinting those branches as their **original bytes**.

```sh
cd metaprog-poc
../coil lint condlint_test.coil --use condlint-on.coil            # report + `help: try:`
../coil lint condlint_test.coil --use condlint-on.coil --diff     # the patch, no writes
../coil lint condlint_test.coil --use condlint-on.coil --fix      # apply it
```

`condlint_test.coil` covers the three cases that matter: a three-test staircase (fixed),
a two-armed `if` (left alone — that is what `if` is for), and a `cond` the author already
wrote. The last one is the subtle one: checkers run on the **expanded** program, so that
`cond` is already nested ifs by the time the rule sees it. `(code-macro? NODE)` is what
tells the expander's ifs from the author's.

Three properties the fix keeps, and how to see them:

- **Behaviour** — run the program before and after; the exit code is the same.
- **Idempotence** — a second `--fix` produces a byte-identical file.
- **Your comments** — a chain with a comment between a test and its body cannot be
  collapsed without deleting it, so it is reported with a `note:` and left alone.

`--fix` loops to a fixpoint (a chain nested inside another chain's body takes a second
round) and reverts any round that stops compiling. Design: `docs/AUTOFIX.md`.
