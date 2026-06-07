# Connecting a parser to the SIMD JS lexer

This lexer is built in two stages, and a parser can hook in at **three levels** of
"cooked-ness". Pick the level that matches how much control your parser wants over
lexing.

```
 bytes
   │
   ▼  Layer A — stage-1 (SIMD, examples/js_stage1.simd, ~8 GB/s)
 start_masks[]  +  word_masks[]          (two bitmaps, 1 bit / input byte)
   │
   ▼  Layer B — pull Lexer (src/js.rs)   ← recommended; mirrors oxc's parser↔lexer protocol
 next_token() / peek() + re_lex_regex() / re_lex_template_tail()
   │
   ▼  Layer C — driver
 Vec<Token>  or  drive(emit)             (our default regex/template policy applied)
```

Everything below is in **byte offsets into the original source**; tokens are
`(kind, start, end)` with `end` exclusive. No copies are made — tokens are spans
into your input buffer.

---

## 0. The data contract (applies to every layer)

- **Input** is a `&[u8]` of the source. Pad it to a multiple of **64 bytes** with
  `0x00` for stage-1 (the SIMD stream processes 64-byte chunks). The padding NULs
  are never reported as tokens.
- **Bitmaps**: allocate `padded_len / 64 + 1` `u64`s for each of `start_masks` and
  `word_masks`.
- **Bit layout**: bit `i` of `masks[c]` corresponds to input byte `c*64 + i`
  (little-endian within the word; use `trailing_zeros` to find the next set bit).
- Keep the original (unpadded) `len`; tokenization is bounded by it.

---

## Layer A — use stage-1 directly (the SIMD bitmaps)

Stage-1 is context-blind and does the embarrassingly-parallel part: it marks every
**token-start byte** and every **word byte**.

```
fn js_stage1(input: ptr[u8], start_masks: ptr[u64], word_masks: ptr[u64]) -> i32[1]
```

- `start_masks[c]` bit `i` set  ⟺  byte `c*64+i` **begins a token**: a word char
  whose predecessor is not a word char (ident/number start), **or** any
  non-word/non-whitespace/non-NUL byte (punctuator, quote, operator, `/`).
- `word_masks[c]` bit `i` set  ⟺  byte is a word char `[A-Za-z0-9_$]`.
- Return value = number of token-start bits (a popcount, handy for sizing).

It is **context-blind**: there are spurious starts inside strings/comments/regex/
templates. A parser using Layer A directly must resolve that context itself.

**What you get cheaply from the bitmaps:**
- *Next token start at/after byte p*: `ctz` over `start_masks` from `p` (skips all
  whitespace in bulk).
- *Identifier/number-run end from start p*: first 0-bit at/after `p` in
  `word_masks` (a bit-scan; no byte re-read).

**Two ways to call it:**

1. **Linked (AOT).** `cargo run --release -- compile examples/js_stage1.simd -o out/`
   produces `out/libjs_stage1.a` + `out/js_stage1.rs` (generated `extern "C"`
   bindings). Link the `.a`, call the generated wrapper. This is the path for a
   parser that ships a binary.
2. **JIT.** Build the module with `codegen::{create_context, compile_module,
   lower_to_llvm}` and `melior::ExecutionEngine::lookup("js_stage1")`. The raw ABI
   is the MLIR memref unpacking — each `ptr` becomes `(alloc_ptr, aligned_ptr,
   offset, size, stride)`, and the `i32[1]` return comes back in `s0` (read as
   `f32` bits, then `to_bits`). See `bench.rs::litmask`/`prof` for the exact
   `extern "C" fn(...)` signature.

Use Layer A if your parser wants to build its own lexer on top of SIMD
boundary-finding (e.g. its own token kinds, its own context machine).

---

## Layer B — the pull Lexer + re-lex hooks (recommended)

`src/js.rs` turns the bitmaps into an on-demand token stream that a parser drives
exactly like oxc's parser drives its lexer. The lexer is **context-free**; the
parser supplies the two context-sensitive decisions through callbacks.

```rust
pub struct Lexer<'a> { /* … */ }
impl<'a> Lexer<'a> {
    pub fn new(src: &'a [u8], start_masks: &'a [u64], word_masks: &'a [u64]) -> Self;
    pub fn next_token(&mut self) -> Option<Token>;   // consume next token
    pub fn peek(&self)            -> Option<Token>;   // look ahead, no consume
    pub fn re_lex_regex(&mut self, at: Token) -> Token;             // hook 1
    pub fn re_lex_template_tail(&mut self, at: Token) -> Token;     // hook 2
}

pub struct Token { pub kind: TokKind, pub start: usize, pub end: usize }
impl Token { pub fn text<'a>(&self, src: &'a [u8]) -> &'a str; }

pub enum TokKind {
    Ident, Keyword, Number, String,
    TemplateNoSub, TemplateHead, TemplateMiddle, TemplateTail,
    Regex, Punct, LineComment, BlockComment,
}
```

### The two decisions only a parser can make

These are exactly the non-regular parts of JS — they need grammar context, so the
lexer hands them back to you:

**1. Regex vs division.** `next_token` always returns a `/` or `/=` as a `Punct`.
If your parser is in **expression position** (where a `RegularExpressionLiteral`
is allowed), call `re_lex_regex(tok)`; it re-scans from the `/` and returns a
`Regex` token, advancing the cursor past the closing `/flags`.

```rust
let mut tok = lex.next_token().unwrap();
if tok.kind == TokKind::Punct && src[tok.start] == b'/' && parser.expression_position() {
    tok = lex.re_lex_regex(tok);   // now TokKind::Regex
}
```

**2. Template `${ … }` nesting.** A `` ` `` yields `TemplateHead` (`` `…${ ``) or
`TemplateNoSub` (`` `…` ``). After `TemplateHead` you parse the substitution
expression normally (pulling tokens); when you reach the `}` that closes it,
`next_token` returns it as a `Punct` — call `re_lex_template_tail(tok)` to get the
`TemplateMiddle` (`` }…${ ``) or `TemplateTail` (`` }…` ``) instead.

```rust
// inside parse_template():
loop {
    let head = /* TemplateHead just consumed */;
    parser.parse_expression();              // consumes the ${ expr } body
    let close = lex.next_token().unwrap();   // the `}` as Punct
    let part = lex.re_lex_template_tail(close);
    match part.kind {
        TokKind::TemplateMiddle => continue, // another ${ … } follows
        TokKind::TemplateTail   => break,    // template literal done
        _ => unreachable!(),
    }
}
```

This is the identical protocol to oxc's `next_regex` / `next_template_substitution_tail`
(which is exactly what the bundled oxc fork exposes), so a parser written against
oxc's lexer maps over almost mechanically.

### Comments
`LineComment` / `BlockComment` are returned as tokens. Most parsers treat them as
trivia — either skip them in your driver, or wrap `next_token` to skip comment
kinds and stash them for a comment table.

---

## Layer C — a pre-driven token stream

If your parser is happy with our **default** regex/template policy (the standard
prev-significant-token heuristic + a brace-depth stack — the same one
`bench::oxc_tokens_standalone` uses), skip the hooks entirely:

```rust
pub fn tokenize(src, start_masks, word_masks) -> Vec<Token>;        // materialized
pub fn drive<F: FnMut(Token)>(src, start_masks, word_masks, emit);  // streaming callback
```

`drive` is the generic engine; pass your own `emit` closure to consume tokens on
the fly without allocating a `Vec`. Use this when the parser doesn't need to make
the regex/template calls itself (e.g. it re-derives context anyway, or you're
prototyping). You give up per-call control of the two decisions above.

---

## Keyword handling — and a free speedup for your parser

The lexer can label keywords for you, or leave them to you:

- **Matched-work** (`tokenize` / `count_tokens`): identifiers that are keywords
  come back as `TokKind::Keyword`. This reads the identifier bytes to classify.
- **Parser-mode** (currently the `count_tokens_parser_mode` driver; internally the
  classifier is generic — `classify::<false>` skips the keyword check): keywords
  come back as plain `Ident`; the lexer only consults the keyword set at a `/` (for
  the regex decision). **Token boundaries are identical** — only the keyword
  *label* is deferred.

If your parser already interns identifiers (most do), it can resolve keyword-ness
from the interned id and use parser-mode — which measured **~1.7× (minified) /
~2.0× (readable)** vs oxc, because the lexer never re-reads identifier bodies. This
is the V8-style division of labor and is recommended if your parser interns.

> Note: parser-mode is wired today through the *driver* (`count_tokens_parser_mode`),
> not yet as a flag on the pull `Lexer` (it hardcodes `classify::<true>`). Exposing
> it on `Lexer::new` is a one-line const-generic addition — say the word and I'll
> add `Lexer::new_parser_mode(...)` so Layer B can defer keywords too.

---

## Wiring mechanics

- **Stage-1** is reusable anywhere via the compiled `.a` + generated `.rs` (Layer A,
  option 1) — link it from your parser's crate/binary regardless of language-host.
- **Stage-2** (`src/js.rs`) currently lives in this **binary** crate. To call
  `js::Lexer`/`tokenize` from a separate parser crate, add a `src/lib.rs` with
  `pub mod js;` (all the needed items are already `pub`) and depend on it as a lib.
  (Ask and I'll add it.)
- **Cross-language parser** (parser not in Rust): two options —
  1. take Layer A bitmaps over a C ABI and implement the pull loop in your host, or
  2. run Layer C in a thin Rust shim and hand the parser a flat token array
     (`kind: u8, start: u32, end: u32` triples) over FFI or a pipe.

A convenience entry point worth adding (say the word):
```rust
// lex(src) → runs stage-1, returns a ready-to-drive Lexer over owned bitmaps
pub fn lex(src: &[u8]) -> (Vec<u64>, Vec<u64>);   // (start_masks, word_masks)
```

---

## Scope / caveats (so the parser knows what it's getting)

- **ASCII-focused.** Identifier classification covers `[A-Za-z0-9_$]`; Unicode
  identifier characters and `\u{…}` escapes in identifiers are not handled. Fine
  for most real-world JS; not spec-complete.
- **No TypeScript.** Operators are JS maximal-munch; we emit `>>`/`>=`/`>>>` as one
  token (oxc splits `>` for generics — a parser-side concern if you do TS).
- **regex & templates are yours.** By design (Layer B). With Layer C you get our
  heuristic, which matches oxc on every file in the test corpus but isn't a
  substitute for real grammar context on adversarial input.
- **Correctness oracle.** `reconstruct(src, &tokens)` rebuilds the source from the
  token spans + whitespace gaps and checks it byte-for-byte — useful as an
  integration smoke test once you wire it up.
```
