# SIMD Language Design Doc

A low-level kernel language where SIMD is the primitive, not an optimization target.
Every construct maps directly to a SIMD operation. No scalar variables, no loops,
no implicit broadcasting.

---

## Core Philosophy

Scalars do not exist as a distinct concept. Everything is a vector. A scalar is just
a width-1 vector or a broadcast constant. The language cannot express serial iteration
or scalar branching — only lane-wise operations, masked operations, reductions, and scans.

---

## Type System

Width is part of the type, written as a bracket suffix:

```
f32[8]    -- 8-wide f32
i16[16]
bool[32]  -- mask register
f32[_]    -- native machine width, resolved at compile time
```

`f32[_]` is the idiomatic type for values that should match the machine's natural register
width. It is resolved once at compile time per target, not polymorphic at runtime.

Width changes are explicit. There is no implicit broadcasting or truncation. If widths
do not match, it is a type error.

```
broadcast(v, to=16)   -- explicit, required
narrow(v)             -- explicit truncation
widen(v)              -- explicit extension
```

---

## Structs

Structs are always SoA. The width annotation goes on the struct, not the fields.
Field access returns a wide vector at the struct's width.

```
struct Particle {
    pos_x: f32,
    pos_y: f32,
    vel_x: f32,
    vel_y: f32,
    mass:  f32,
}[1024]
```

`p.pos_x` returns `f32[1024]`, not `f32`. The layout in memory is field-first:
all `pos_x` values contiguous, then all `pos_y`, etc. There is no AoS layout.
The language does not model 3D math vectors or spatial dimensions — those are
decomposed into flat scalar fields by the caller.

---

## Operations

Lane-wise by default. Any binary op on two vectors of matching width is lane-wise:

```
a + b
a * b + c    -- fused multiply-add where available
a > b        -- produces bool[N], not a scalar bool
```

---

## Masked Operations

Masks are first-class. They are produced by comparisons and logical ops on wide vectors.
The mask syntax gates an operation over active lanes:

```
[mask] a + b          -- masked op, inactive lanes undefined
[mask] a + b : c      -- masked blend, inactive lanes take c
```

The brackets contain any expression that produces a `bool[N]`. This replaces if/else
entirely. There is no branching construct.

```
alive    = p.mass > 0.0
new_vel  = [alive] p.vel + gravity * dt : p.vel

in_bounds = pos_x > -500.0 & pos_x < 500.0
          & pos_y > -500.0 & pos_y < 500.0
new_mass  = [in_bounds] p.mass : 0.0
```

---

## Reductions

Reductions collapse across lanes using APL-style `/` notation:

```
+/ v        -- horizontal sum
*/ v        -- horizontal product
|/ mask     -- any lane set
&/ mask     -- all lanes set
max/ v      -- horizontal max
min/ v      -- horizontal min
```

These always reduce to width 1 (a scalar or broadcast value). Chaining is natural:

```
dot = +/ (a * b)
```

---

## Scans

Scans are prefix reductions that preserve width:

```
scan.add(v)    -- prefix sum
scan.xor(v)    -- prefix XOR, used for string interior detection
scan.max(v)
```

Output is the same width as input. Each lane holds the reduction of all preceding lanes.

---

## Shuffles

Named structural operations for common patterns:

```
reverse(v)
broadcast(v, lane=0)
zip(a, b)       -- interleave lanes
(evens, odds) = unzip(v)
rotate(v, n)
```

For patterns not covered by named ops, an explicit lane mapping:

```
v = [3,2,1,0 <- src]              -- reverse, explicit form
v = [a.0, b.0, a.1, b.1 <- a, b] -- interleave two sources
```

The arrow notation reads as "wiring" — each output lane is routed from the specified
source lane.

---

## Memory

No pointer arithmetic. Loads and stores are typed and explicit about alignment:

```
v = load[f32[8]](ptr)       -- aligned load
v = loadu[f32[8]](ptr)      -- unaligned, explicit
store(ptr, v)
store(ptr, v, mask)         -- masked store
```

Gather and scatter use `.[]` notation to distinguish from normal indexing:

```
result   = src.[indices]          -- gather
result   = src.[indices, mask]    -- masked gather
dst.[indices] = src               -- scatter
dst.[indices, mask] = src         -- masked scatter
```

---

## Example: Particle Update Kernel

```
struct Particle {
    pos_x: f32,
    pos_y: f32,
    vel_x: f32,
    vel_y: f32,
    mass:  f32,
}[1024]

fn update(p: Particle[1024], dt: f32[_], gravity: f32[_]) -> Particle[1024] {

    alive = p.mass > 0.0
    drag  = 1.0 - (0.01 * dt)

    new_vel_x = [alive] p.vel_x * drag : p.vel_x
    new_vel_y = [alive] (p.vel_y + gravity * dt) * drag : p.vel_y

    speed_sq  = new_vel_x * new_vel_x + new_vel_y * new_vel_y
    too_fast  = speed_sq > 10000.0
    speed     = sqrt(speed_sq)
    new_vel_x = [too_fast] (new_vel_x / speed) * 100.0 : new_vel_x
    new_vel_y = [too_fast] (new_vel_y / speed) * 100.0 : new_vel_y

    new_pos_x = p.pos_x + new_vel_x * dt
    new_pos_y = p.pos_y + new_vel_y * dt

    in_bounds = new_pos_x > -500.0 & new_pos_x < 500.0
              & new_pos_y > -500.0 & new_pos_y < 500.0
    new_mass  = [in_bounds] p.mass : 0.0

    return Particle[1024] {
        pos_x: new_pos_x,
        pos_y: new_pos_y,
        vel_x: new_vel_x,
        vel_y: new_vel_y,
        mass:  new_mass,
    }
}
```

---

## Example: SIMD JSON Structural Detection

```
struct Masks {
    is_quote:      bool[32],
    is_escape:     bool[32],
    is_structural: bool[32],
    is_whitespace: bool[32],
}

fn classify(chunk: u8[32]) -> Masks {
    is_quote      = chunk == '"'
    is_escape     = chunk == '\\'
    is_whitespace = chunk == ' ' | chunk == '\t'
                  | chunk == '\n' | chunk == '\r'
    is_structural = chunk == '{' | chunk == '}'
                  | chunk == '[' | chunk == ']'
                  | chunk == ':' | chunk == ','

    return Masks {
        is_quote:      is_quote,
        is_escape:     is_escape,
        is_structural: is_structural,
        is_whitespace: is_whitespace,
    }
}

fn string_mask(is_quote: bool[32], is_escape: bool[32]) -> bool[32] {
    -- Mask out escaped quotes
    real_quote = is_quote & ~scan.preceding_any(is_escape)
    -- Prefix XOR: flips at each real quote boundary
    -- bytes inside strings come out true
    return scan.xor(real_quote)
}

fn structural_indices(m: Masks, inside_string: bool[32]) -> u32[*] {
    structural = m.is_structural & ~inside_string
    return compress_indices(structural)
}
```

---

## Compilation via MLIR

The compiler targets MLIR's `vector` dialect directly, with no custom dialect.
The lowering pipeline is:

```
Source → AST → MLIR vector dialect → LLVM dialect → LLVM IR → target
```

### Type Mapping

| Language      | MLIR vector dialect         |
|---------------|-----------------------------|
| `f32[8]`      | `vector<8xf32>`             |
| `bool[32]`    | `vector<32xi1>`             |
| `f32[_]`      | `vector<?xf32>` or fixed at compile time per target |

### Operation Mapping

| Language                  | MLIR op                                  |
|---------------------------|------------------------------------------|
| `a + b`                   | `arith.addf` on vectors                  |
| `a > b`                   | `arith.cmpf` → `vector<Nxi1>`            |
| `[mask] a + b : c`        | `arith.select` with vector mask          |
| `+/ v`                    | `vector.reduction <add>`                 |
| `scan.add(v)`             | `vector.scan <add>`                      |
| `scan.xor(v)`             | `vector.scan <xor>`                      |
| `load[f32[8]](ptr)`       | `vector.load`                            |
| `loadu[f32[8]](ptr)`      | `vector.load` (alignment attr = 1)       |
| `store(ptr, v, mask)`     | `vector.maskedstore`                     |
| `src.[indices]`           | `vector.gather`                          |
| `dst.[indices] = src`     | `vector.scatter`                         |
| `compress_indices(mask)`  | `vector.compress_store` + index extract  |
| `zip(a, b)`               | `vector.shuffle` with interleave indices |
| `reverse(v)`              | `vector.shuffle` with reversed indices   |
| `broadcast(v, lane=0)`    | `vector.shuffle` with constant indices   |

### SoA Struct Lowering

Structs are lowered before hitting the vector dialect. Each field becomes an
independent `vector.load` / `vector.store`. The struct type is erased; field
accesses become direct loads from the appropriate base pointer + field offset.

### Masked Operations

`[mask] a + b : c` lowers to `arith.select` in the simple case. Where the
backend supports native masked execution (AVX-512, SVE), LLVM will lower
`arith.select` + the op to a merged masked instruction. No special handling
needed in the frontend — LLVM's backend handles this.

### Width Polymorphism (`f32[_]`)

`f32[_]` is resolved at compile time to the target's natural vector width.
This is a compile-time constant, not a runtime parameter. On AVX2 it becomes
`f32[8]`, on AVX-512 `f32[16]`, on NEON `f32[4]`. Width mismatches after
resolution are type errors caught before codegen.

### Target Coverage via LLVM Backend

By targeting MLIR's vector dialect and lowering through LLVM, the following
targets are supported without additional work:

- x86 SSE4 / AVX2 / AVX-512
- ARM NEON / SVE
- RISC-V V extension
- WebAssembly SIMD

---

## Open Questions

- **`compress_indices` return type:** `u32[*]` uses `*` to mean "variable length
  output." The actual count depends on the popcount of the mask at runtime. This
  needs a dynamic vector type or a fixed upper bound. MLIR's `vector` dialect
  prefers static shapes — worth deciding whether to use a fixed-size output with
  a separate length value, or accept a dynamic vector here.

- **Masked FMA:** `[mask] a * b + c` should fuse to a masked FMA. Whether this
  falls out of LLVM's backend or needs explicit pattern matching in the lowering
  pass is unverified.

- **`scan.preceding_any`:** Used in the JSON example. This is not a standard
  scan — it's a "was the immediately preceding lane set" query. May need to be
  expressed as a `vector.shuffle` + `&` rather than a scan primitive. Needs
  investigation.
