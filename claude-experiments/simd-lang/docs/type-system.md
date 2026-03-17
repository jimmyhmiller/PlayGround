# SIMD Language — Type System & Core Abstractions

## 1. The Central Idea

Every value is a vector. There are no scalars. A "scalar" is a width-1 vector or a
broadcast constant. The type system tracks two things for every value: **what kind of
element** and **how many lanes**.

Width is a **compile-time parameter**. It flows through structs and functions via
comptime generics, inspired by Mojo's parameterization model. The compiler
monomorphizes — generating specialized code for each concrete width.

---

## 2. Types

### 2.1 Scalar Element Types

| Type   | Size | Description              |
|--------|------|--------------------------|
| `f32`  | 32b  | IEEE 754 single float    |
| `f64`  | 64b  | IEEE 754 double float    |
| `i8`   | 8b   | Signed 8-bit integer     |
| `i16`  | 16b  | Signed 16-bit integer    |
| `i32`  | 32b  | Signed 32-bit integer    |
| `i64`  | 64b  | Signed 64-bit integer    |
| `u8`   | 8b   | Unsigned 8-bit integer   |
| `u16`  | 16b  | Unsigned 16-bit integer  |
| `u32`  | 32b  | Unsigned 32-bit integer  |
| `u64`  | 64b  | Unsigned 64-bit integer  |
| `bool` | 1b   | Boolean / mask bit       |

Element types never appear alone — they must have a width to form a vector type.

### 2.2 Vector Types

```
f32[8]      -- 8 lanes of f32
i16[16]     -- 16 lanes of i16
bool[32]    -- 32-bit mask register
f32[N]      -- N lanes, where N is a comptime parameter
f32[_]      -- native machine width (sugar for a target-resolved comptime value)
```

Width is always a compile-time constant. `f32[8]` and `f32[16]` are distinct,
incompatible types. No implicit broadcasting or truncation.

### 2.3 Width Kinds

| Kind       | Syntax   | Meaning                                              |
|------------|----------|------------------------------------------------------|
| Fixed      | `[8]`    | Exactly 8 lanes. Literal integer.                    |
| Param      | `[N]`    | Comptime parameter — resolved at monomorphization.   |
| Native     | `[_]`    | Target's natural SIMD width. Sugar for a fixed value resolved early in compilation. |

**Native width resolution:** `f32[_]` becomes `f32[8]` on AVX2, `f32[16]` on AVX-512,
`f32[4]` on NEON. This happens once at the start of compilation, before type checking.
After resolution, only `Fixed` and `Param` widths exist.

---

## 3. Compile-Time Parameters

### 3.1 The Core Mechanism

Width is a compile-time parameter, not a runtime value. Functions and structs declare
comptime parameters in square brackets:

```
fn dot[N](a: f32[N], b: f32[N]) -> f32[1] {
    return +/ (a * b)
}

struct Particle[N] {
    pos_x: f32[N],
    pos_y: f32[N],
    vel_x: f32[N],
    vel_y: f32[N],
    mass:  f32[N],
}
```

`N` is not a runtime variable — it is resolved at each call site to a concrete integer.
The compiler generates a separate `dot_8`, `dot_16`, etc. for each width used.

### 3.2 Syntax

Comptime parameters appear in `[...]` after the name in definitions:

```
-- Struct definition: N is a comptime width parameter
struct Particle[N] {
    pos_x: f32[N],
    pos_y: f32[N],
}

-- Function definition: N is a comptime width parameter
fn update[N](p: Particle[N], dt: f32[N]) -> Particle[N] { ... }

-- Optional annotation for clarity
fn dot[N: comptime](a: f32[N], b: f32[N]) -> f32[1] { ... }
```

At usage sites, concrete values fill the parameter:

```
p: Particle[1024]       -- Particle with 1024-wide fields
update[1024](p, dt)     -- explicit instantiation (if inference doesn't apply)
```

### 3.3 Parameter Inference

When a comptime parameter appears in a function's argument types, it can be inferred
from the caller's argument types:

```
fn dot[N](a: f32[N], b: f32[N]) -> f32[1] { ... }

x: f32[8]
y: f32[8]
result = dot(x, y)   -- N inferred as 8 from x and y
```

Inference works by unification: the compiler matches the argument's concrete type
against the parameter's pattern and extracts the comptime value.

### 3.4 Multiple Comptime Parameters

Functions and structs can have multiple comptime parameters:

```
struct Grid[W, H] {
    data: f32[W],
    meta: i32[H],
}

fn reshape[M, N](src: f32[M]) -> f32[N] { ... }
```

### 3.5 Relationship to Native Width

`f32[_]` is sugar. It resolves to a fixed integer before comptime parameter binding.
A function using `[_]` is not generic — it's specialized to one target:

```
fn process(v: f32[_]) -> f32[_] { ... }
-- On AVX2, this is exactly: fn process(v: f32[8]) -> f32[8]
```

To write truly width-generic code, use a comptime parameter:

```
fn process[N](v: f32[N]) -> f32[N] { ... }
-- Works for any width
```

---

## 4. Structs

### 4.1 SoA Layout

Structs are always Structure of Arrays. Width is a comptime parameter on the struct:

```
struct Particle[N] {
    pos_x: f32[N],
    pos_y: f32[N],
    mass:  f32[N],
}
```

In memory, `Particle[1024]` is laid out as three contiguous arrays:
```
[pos_x_0, pos_x_1, ..., pos_x_1023]
[pos_y_0, pos_y_1, ..., pos_y_1023]
[mass_0,  mass_1,  ..., mass_1023 ]
```

Field access returns a vector at the struct's width:
- `p.pos_x` where `p: Particle[1024]` → `f32[1024]`

### 4.2 Struct Construction

Struct literals provide the comptime width and field values:

```
return Particle[N] {
    pos_x: new_pos_x,
    pos_y: new_pos_y,
    mass:  new_mass,
}
```

### 4.3 Struct Erasure

Structs are a compile-time abstraction. They are erased before codegen — each field
becomes an independent vector load/store. No struct value exists at runtime.

---

## 5. Type Rules

### 5.1 Width Matching

Binary operations require matching widths:

```
a: f32[8], b: f32[8]  →  a + b : f32[8]     ✓
a: f32[8], b: f32[16] →  a + b               ✗ width mismatch
```

No implicit broadcasting. Use explicit width-change operations:

```
broadcast(v, to=16)
narrow(v)
widen(v)
```

### 5.2 Element Type Matching

Binary arithmetic requires matching element types:

```
a: f32[8], b: f32[8]  →  a + b : f32[8]     ✓
a: f32[8], b: i32[8]  →  a + b               ✗ element type mismatch
```

No implicit numeric coercion.

### 5.3 Operation Type Rules

| Operation              | Input types            | Output type      |
|------------------------|------------------------|------------------|
| `a + b`, `a * b`, etc. | `T[N]`, `T[N]`         | `T[N]`           |
| `a > b`, `a == b`      | `T[N]`, `T[N]`         | `bool[N]`        |
| `m1 & m2`, `m1 \| m2`  | `bool[N]`, `bool[N]`   | `bool[N]`        |
| `~m`                   | `bool[N]`              | `bool[N]`        |
| `[mask] body : fallback` | `bool[N]`, `T[N]`, `T[N]` | `T[N]`     |
| `+/ v`                 | `T[N]`                 | `T[1]`           |
| `scan.add(v)`          | `T[N]`                 | `T[N]`           |
| `load[T[N]](ptr)`      | ptr                    | `T[N]`           |
| `src.[indices]`        | ptr, `u32[N]`          | `T[N]`           |

### 5.4 Mask Width Agreement

Mask width must match the operation's vector width:

```
mask: bool[8], a: f32[8]  →  [mask] a + 1.0 : a     ✓
mask: bool[16], a: f32[8] →  [mask] a + 1.0 : a      ✗ mask width ≠ operand width
```

### 5.5 Literal Inference

Numeric literals are the one exception to "no implicit broadcasting." They are
inherently width-agnostic and are broadcast to match the other operand:

```
p.mass > 0.0       -- 0.0 broadcast to f32[N] to match p.mass
1.0 - (0.01 * dt)  -- 1.0 and 0.01 broadcast to match dt's width
```

The compiler inserts explicit broadcast nodes during type checking.

### 5.6 Variable Binding

Variables are declared by first assignment. Type is inferred:

```
alive = p.mass > 0.0          -- alive: bool[N]
drag  = 1.0 - (0.01 * dt)     -- drag: f32[N]
```

Reassignment must maintain the same type.

---

## 6. Masked Execution Model

Masked operations replace control flow entirely. There is no `if/else`:

```
-- Instead of: if (alive) { vel = vel + gravity * dt; }
vel = [alive] vel + gravity * dt : vel
```

Both body and fallback are always evaluated. The mask selects per-lane which result
to keep. This maps directly to hardware blend/select instructions.

Without a fallback, inactive lanes are undefined:

```
[mask] a + b          -- inactive lanes: undefined
[mask] a + b : c      -- inactive lanes: take c
```

---

## 7. Type Checking Strategy

### 7.1 Pipeline

```
Source → Parse → Resolve Native Widths → Type Check → Monomorphize → Codegen
```

### 7.2 Phase 1: Width Resolution

Replace all `[_]` with the target's native width. After this, only `Fixed(N)` and
`Param(name)` widths remain.

### 7.3 Phase 2: Type Inference & Checking

Walk each function body top-to-bottom:
1. Infer variable types from first assignment.
2. Infer literal types from binary operation context (insert broadcasts).
3. Verify width matching on all operations.
4. Verify element type matching.
5. Verify mask width agreement.
6. Verify comptime param consistency (same `N` resolves to same value everywhere).

### 7.4 Phase 3: Monomorphization

For each call to a comptime-parameterized function, generate a specialized version
with concrete widths. This is standard monomorphization:

```
dot[8](x, y)    → generates dot_8
dot[16](a, b)   → generates dot_16
```

Struct types are similarly specialized — `Particle[1024]` generates field accessors
and constructors for width 1024.

---

## 8. Implementation: Type Representation

```rust
enum ScalarType {
    F32, F64,
    I8, I16, I32, I64,
    U8, U16, U32, U64,
    Bool,
}

struct VecType {
    scalar: ScalarType,
    width: u64,  // always concrete after monomorphization
}

enum ResolvedType {
    Vec(VecType),
    Struct { name: String, width: u64 },
    Void,
}
```

### LLVM Mapping (inkwell)

| SIMD-lang type | LLVM type via inkwell                  |
|----------------|----------------------------------------|
| `f32[8]`       | `f32_type.vec_type(8)`                 |
| `i32[4]`       | `i32_type.vec_type(4)`                 |
| `bool[8]`      | `i1_type.vec_type(8)` (custom i1)      |
| struct fields   | Separate vector values (SoA erased)   |

### Operations → LLVM

| Operation           | LLVM via inkwell                              |
|---------------------|-----------------------------------------------|
| `a + b` (float)     | `build_float_add` on vector types             |
| `a + b` (int)       | `build_int_add` on vector types               |
| `a > b`             | `build_float_compare` → vector `<N x i1>`     |
| `[mask] a : b`      | `build_select` with vector mask               |
| `+/ v`              | LLVM vector reduction intrinsic               |
| `scan.add(v)`       | Sequence of shuffles + adds                   |
| `load[f32[8]](ptr)` | `build_load` with vector type + alignment     |
| gather/scatter       | LLVM masked gather/scatter intrinsics        |

---

## 9. Open Questions

### 9.1 Cast Operations

The language needs explicit type casts. Candidates:
- `as_f32(v)` / `as_i32(v)` — element type conversion
- `bitcast[f32[N]](v)` — reinterpret bits

### 9.2 Comptime Arithmetic

Should comptime parameters support arithmetic?
```
fn interleave[N](a: f32[N], b: f32[N]) -> f32[N * 2] { ... }
```
This requires the type system to evaluate `N * 2` at compile time.

### 9.3 Comptime Conditionals

Mojo and Zig support `comptime if` for specialization:
```
fn reduce[N](v: f32[N]) -> f32[1] {
    comptime if N == 1 {
        return v
    }
    -- recursive halving...
}
```
Useful for width-dependent optimization. Not essential for v1.

### 9.4 Dynamic Width `[*]`

`compress_indices(mask)` returns variable-length output. Options:
1. Return `T[N]` (same as mask width) + length value. Unused lanes are garbage.
2. Write to memory buffer + return count.

Recommendation: Option 1 — return `(T[N], u32[1])` tuple.
