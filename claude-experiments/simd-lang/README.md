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

## How it works

```
.simd source → lexer → parser → type checker → MLIR codegen → LLVM → .a + .rs
```

Errors include source locations (`error at 3:5: undefined variable: foo`). The type checker catches width mismatches, bad builtin args, etc. before codegen.

The codegen emits MLIR's vector dialect operations which LLVM lowers to NEON instructions. The `stream` construct becomes an `scf.for` loop with carried values.
