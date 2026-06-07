# Faster-than-oxc JS parsing with a SIMD front end

Goal: turn JS text into something carrying *as much information as a standard
AST*, as fast as possible — faster than oxc — single-threaded (no multicore).

## TL;DR results (Apple Silicon, `aot-bench/src/bin/tape.rs`)

| grammar | our best | vs oxc bare AST | vs oxc parse+semantic |
|---|---|---|---|
| flat expressions + var decls | ~420 MiB/s | **~2.3×** | ~4.2× |
| nested (calls/member/index/array/grouping) | ~300 MiB/s | **~1.67×** | ~3.4× |

All paths are validated: the flat **tape** is checked for RPN balance, and the
op_ext / two-pass / fused variants are asserted byte-identical to each other.

Run: `cargo run --release --manifest-path aot-bench/Cargo.toml --bin tape -- 20000`

## The journey (each step measured)

1. **Diagnosis.** SIMD stage-1 (token boundaries) is ~5 GB/s — ~1% of total and
   irreducibly cheap. The parse has *no hotspot*: cost is spread across the
   precedence recursion and per-token lexing. So SIMD can't speed the parse loop;
   it has to change the *algorithm* and the *output representation*.

2. **Flat tape, not columnar IR.** The original `jsir-parse` built a columnar
   `jsir_ir::Module` (interning, ValueIds, regions) and ran 0.46× oxc — the IR
   build *doubled* parse time. Replacing it with a self-contained **postorder
   (RPN) tape** — each node a fixed record `(kind, flags, nargs, start, end)`,
   children implicit as the preceding `nargs` completed subtrees — jumped to
   **1.57× oxc**. No interning, no ValueIds, no regions; `nargs` + spans
   reconstruct the whole tree (and every name/operator/literal).

3. **Lexing is the bottleneck, not parsing.** Splitting the fused pass:
   tree-building from a token array runs ~900–1100 MiB/s (≈6× oxc); turning
   boundaries into classified tokens runs ~290 MiB/s. A *skeleton* lexer
   (bit-scan + `word_end` + build, no classify) hits ~1200 MiB/s — so per-token
   *classification + end-finding* is a ~4× tax.

4. **Use the `word_masks` structure for ends.** A token's end for idents/numbers
   is the first 0 in the word bitmap (`word_end`) — and since `e`/`x`/`n` are
   word chars, that already captures hex / bigint / `1e5` (only decimals and
   signed exponents need a scalar fallback). This lean tokenizer is ~1.7× the old
   one. Two-pass (lean-lex → array → parse) → **~2.2× oxc**.

5. **`.simd` `op_ext` bitmap (structural operator adjacency).** The remaining
   lexer cost is `match_operator`'s look-ahead. The stage-1 kernel
   (`examples/js_stage1_ops.simd`) now emits a third bitmap in the same NEON
   byte-pass: `op_ext = op_bits & (op_bits >> 1)` — set where two operator chars
   are adjacent (a *possible* multi-char operator). The tokenizer does the full
   maximal-munch only where the bit is set (or at a 64-byte boundary, which the
   `>>1` can't see across); standalone `=` `+` `<` … skip it entirely. Sound by
   over-approximation. → **~2.3× oxc** on flat code.

6. **Fusion vs two-pass depends on grammar depth.** On flat code the two tight
   loops (lex→array, then parse) pipeline better than interleaving, so two-pass
   wins despite the array round-trip. On *nested* code the parser is heavier and
   fusion wins. The combined best — **fused-lean + op_ext** (no token array +
   the SIMD op_ext shortcut) — is **1.67× oxc** on nested, 2.3× on flat.

## Architecture

```
bytes ──SIMD──▶ start_masks, word_masks, op_ext_masks   (examples/js_stage1_ops.simd, AOT, native NEON)
            │
            ▼
   lean tokenizer: walk start bitmap; ends from word_masks (idents/numbers) or
   op_ext-gated match_operator (punctuators); strings/comments scalar
            │
            ▼
   Pratt / postfix parser → flat RPN tape (Vec<Node>, 12 B/node, no interning)
```

The tape is the deliverable: a flat, pointer-free array you can walk once to
materialize any AST shape, with full source spans.

## Grammar covered (subset, with real structure)

literals (number/string/bool/null/bigint), identifiers, unary/binary/assignment/
update with correct precedence, member `a.b`, index `a[b]`, calls `f(x,y)`, array
literals `[…]`, grouping `(…)`, expression statements, `let`/`var`/`const`.
Excludes (for now): objects, functions, control flow, regex, templates.

## Why no multicore

Excluded by request. Everything above is single-threaded; statement boundaries
found via the structural bitmaps would make data-parallel parsing easy later, but
the wins here are all from representation + SIMD structure, not threads.
