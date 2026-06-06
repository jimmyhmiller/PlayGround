# simd-lang

A little experiment: what if there was a language where every value is a SIMD vector?

The idea was to see how far you can get with a SIMD-first language that compiles to native code via MLIR. The test case is a JSON structural parser — it hits about 3.5 GB/s on Apple M1 for stage 1 (~73% of simdjson) and ~1.15 GB/s for a full DOM parse (~76% of simdjson).

The whole thing is pretty rough around the edges. It was built iteratively as an exploration, not designed upfront.

## Building & running

```bash
# Compile a .simd file to a static library + Rust bindings
cargo run --release -- compile examples/json_stage1.simd -o output/

# Run tests
cargo test --release

# Emit MLIR (useful for debugging)
cargo run --release -- compile examples/json_stage1.simd --emit-mlir
```

## The language

Every value is a fixed-width vector. There are no scalar variables — a "scalar" is just a width-1 vector.

```
-- types
u8[64]       -- 64 bytes
i32[4]       -- 4 ints
bool[64]     -- 64 booleans
u64[1]       -- "scalar" u64
ptr[u8]      -- pointer to a buffer
```

### Functions

```
fn add(a: f32[8], b: f32[8]) -> f32[8] {
    return a + b
}
```

### Operators

All element-wise: `+` `-` `*` `/` `&` `|` `^` `~` `>>` `<<` `>` `<` `>=` `<=` `==` `!=`

`>>` and `<<` are bit shifts. For sliding elements within a vector, use `lane_shr(v, n)` / `lane_shl(v, n)`.

### Literals & broadcasting

Literals broadcast to match the other operand:

```
result = chunk + 1       -- 1 broadcasts to chunk's width
mask = chunk == '"'      -- char literal broadcasts
inverted = data ^ ~0     -- ~0 broadcasts to all-ones
```

For standalone literals (no binary context), add a type annotation:

```
all_ones: u64[1] = ~0
zero: u8[64] = 0
```

Vector literals:

```
table = [u8: 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]
zeros = [u8: 0; 64]     -- repeat syntax
```

### Masked expressions

Per-lane select:

```
result = [mask] value_if_true : value_if_false
```

### Reductions & scans

```
total = +/ values                  -- sum reduction
prefix_xor = scan.xor(bits, seed) -- inclusive prefix XOR with carry
```

Reductions: `+/` `*/` `|/` `&/` `max/` `min/`

Scans: `scan.add` `scan.xor` `scan.max` `scan.preceding_any`

### Streams

Process a buffer in chunks with carried state:

```
stream chunk: u8[64] over input carry (count: i32[1] = 0) {
    matches = chunk == 'x'
    carry count = count + popcount(matches)
}
return count
```

`chunk_offset` is automatically available as the byte offset of the current chunk. Carry variables are available after the stream with their final values.

### If/else

```
if cond {
    x = a
} else {
    x = b
}
```

Both branches execute; variables assigned in both are merged with `select`.

### Structs

```
struct Particle[N] {
    pos_x: f32[N]
    vel_x: f32[N]
}
```

### Gather

```
chars = gather(input, positions)
chars = input.[positions]          -- equivalent
```

## Builtins

**Vector:** `iota(N)`, `extract(vec, lane)`, `compress(data, mask)`, `popcount(mask)`, `to_i32(vec)`, `sqrt(vec)`, `split_lo(vec)`, `split_hi(vec)`

**Lane shift:** `lane_shr(v, n)`, `lane_shl(v, n)`

**Bitmask:** `to_bitmask(bool_vec)`, `from_bitmask(u64, width)`, `clmul(a, b)`, `ctz(v)`, `clear_lowest_bit(v)`

**Table lookup:** `tbl(table_u8x16, indices)`

**Memory:** `gather(ptr, indices)`, `store(ptr, data)`, `store_at(ptr, offset, data)`, `compressstore(ptr, offset, data, mask)`, `load[Type](ptr)`, `loadu[Type](ptr)`

## JSON parser

The point of this whole thing. `examples/json_stage1.simd` is a ~50 line SIMD stage 1 that finds JSON structural characters using CLMUL prefix-XOR for string tracking. `src/json.rs` has a Rust tape builder (stage 2) that produces a simdjson-style DOM.

The DOM is verified by round-trip tests: parse → walk tape → reconstruct JSON → compare character-for-character with original.

On Apple M1, 3.35 MB JSON:

| | simdjson | simd-lang |
|---|---------|-----------|
| Stage 1 | 4.78 GB/s | 3.5 GB/s |
| Full DOM parse | 1.51 GB/s | 1.15 GB/s |

## JavaScript lexer

A second test case, same two-stage shape as JSON. `examples/js_stage1.simd` is a
SIMD pass that classifies every byte and emits a stream of **token-start
positions** — context-blind: word-starts (`[A-Za-z0-9_$]` whose predecessor
isn't a word char) plus every punctuator/quote/operator byte. The
"previous byte was a word char" relation is carried *exactly* across 64-byte
chunk boundaries via the bitmask (`(word_bits << 1) | carry`), not the
shift-in-zero approximation.

`src/js.rs` is the scalar Stage 2 that consumes that position stream and owns
the parts that are irreducibly sequential — the ~15% SIMD genuinely can't do:

- **strings / comments / template / regex interiors** — scanned, with the
  spurious stage-1 starts inside them skipped;
- **regex-vs-division** (`/ab+c/g` vs `a / b`) — disambiguated by the previous
  significant token, the classic lexer/parser feedback that makes JS impossible
  to lex context-free;
- **template `${ ... }` nesting** — a brace/template stack (a context-free,
  not regular, property);
- **longest-match multi-char operators** (`===`, `>>>=`, `=>`, …).

Verified the same way as the JSON DOM — a round-trip: the tokens plus the
whitespace gaps between them reconstruct the source byte-for-byte, proving every
byte is covered exactly once. See the tests in `src/js.rs`
(`cargo test --release js::`): cross-chunk boundary correctness, strings vs.
comments that contain each other's delimiters, nested templates, regex-vs-divide,
and number/keyword/identifier classification.

### Comparing against oxc

`cargo run --release --features bench -- tokens <file.js>` dumps both token
streams (ours → `/tmp/ours.tokens`, [oxc](https://oxc.rs) → `/tmp/oxc.tokens`)
and aligns them; `-- bench <file.js> N` measures throughput.

**Getting oxc to lex standalone.** oxc's lexer is a co-routine of the parser:
the parser resolves regex-vs-division and template `${}` nesting and feeds the
answers back. Run the lexer *alone* and it desyncs at the first regex literal —
on lodash it reads `/&(?:amp|lt|gt|quot|#39);/g` as division and dies at byte
4650 (0.9% of the file). So a naïve "oxc tokenizer" benchmark is meaningless.

The fix is to replicate that feedback. A small fork of oxc widens two methods to
`pub` — `next_regex` and `next_template_substitution_tail` — and `bench.rs` drives
them with the *same* prev-significant-token heuristic our stage-2 uses (plus a
template brace-depth stack). With that, **oxc lexes the entire file with 0 errors**,
and its stream matches ours almost exactly:

| file | our tokens | oxc tokens | exact span matches | differences |
|---|---|---|---|---|
| `lodash.js` (544 KB) | 42,191 (incl. 842 comments) | 41,377 | 41,327 / 41,349 | 22 |
| `three.min.js` (603 KB) | 193,056 | 193,266 | 192,955 | 98 |

Every remaining difference is oxc's *intentional* `>`-splitting (it emits `>`
singly so the parser can reassemble `>>`/`>=`/`>>>` in generics) — a third
feedback hook, not a disagreement about what the tokens are.

**Throughput** (Apple M1, both lexing 100% correctly, non-materializing on both
sides — tokens produced and discarded, matching how a parser consumes them):

| file | stage-1 SIMD | full (s1+s2) | oxc | stage-1 vs oxc | **full vs oxc** |
|---|---|---|---|---|---|
| `three.min.js` (minified, 603 KB) | ~8000 MB/s | 299–325 | 247–255 | ~31× | **1.21×** |
| `vue.global.prod.js` (minified, 148 KB) | ~8000 MB/s | 199–216 | 165–173 | ~48× | **1.20×** |
| `lodash.js` (readable, 544 KB) | ~8000 MB/s | 1063–1160 | 805–874 | ~10× | **1.32×** |

We **beat oxc on every file** while producing the *same* token stream (192,955 /
193,179 spans identical on three.min; the rest are oxc's intentional `>`-splitting),
including keyword classification — i.e. doing the same work oxc does.

How it got there (each step measured, output byte-identical throughout):

1. **Bitmaps, not a position array.** Stage-1 stopped `compressstore`-ing ~1 MB of
   i32 token positions and instead stores two bitmaps (`start_masks`, `is_word`,
   one u64 per 64-byte chunk). Stage-1 throughput **700 → 8000 MB/s** — the
   position array was a 10× memory-traffic bottleneck. Stage-1 is now irrelevant
   to the total; the whole cost is stage-2.
2. **Pull lexer + non-materializing driver.** `src/js.rs` exposes a `Lexer` a
   parser drives token-by-token (`next_token`/`peek` + `re_lex_regex` /
   `re_lex_template_tail` hooks); templates split into head/middle/tail. Counting
   tokens without storing them (as a parser does, and as oxc's loop does) removed
   the `Vec` materialization from both sides.
3. **memchr for comment/string interiors** (borrowed from oxc) — line/block
   comments scan to their end with SIMD `memchr`/`memmem`.
4. **Word-at-a-time bitmap iteration** (simdjson-style) — the current 64-bit word
   lives in a register and bits are cleared as tokens are consumed, instead of
   re-loading + re-masking per token. The single biggest stage-2 win.
5. **Cheap per-token bookkeeping** — a 256-entry lead-byte class table, a
   single-char-punctuator fast class (no look-ahead), a one-`bool` regex-context
   flag (replacing per-token `Option`+`from_utf8`), and a (length, first-letter)
   keyword pre-filter that rejects most identifiers without reading their bytes.

**Why not 2×?** The pure bitmap-iteration floor is ~100 GB/s — iteration is free;
100% of the remaining cost is reading the token's bytes to classify it, which oxc
pays too. Our edge over oxc is that we *skip* whitespace/comment bytes via the
bitmap (so the win is larger on readable code, 1.32×, than on minified, ~1.20×,
where there's little whitespace to skip). Closing the rest to 2× would require
**SIMD batch classification** — gathering and classifying many token lead-bytes
per vector op — which the variable-token-length, sequential regex/template context
makes a substantial separate effort. A simpler lever: **deferring keyword
recognition to the parser** (emit `Ident` for keywords, a standard design — the
parser resolves them during interning) measures **1.44× (minified) / 1.66×
(readable)**, but does less than oxc, so it isn't reported above as the
apples-to-apples number.

Caveat in oxc's favor throughout: it produces full Unicode/escape/TS handling
where ours is ASCII-focused.

## How it works

```
.simd source → lexer → parser → type checker → MLIR codegen → LLVM → .a + .rs
```

Errors include source locations (`error at 3:5: undefined variable: foo`). The type checker catches width mismatches, bad builtin args, etc. before codegen.

The codegen emits MLIR's vector dialect operations which LLVM lowers to NEON instructions. The `stream` construct becomes an `scf.for` loop with carried values.
