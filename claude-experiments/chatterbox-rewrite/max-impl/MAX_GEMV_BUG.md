# MAX bug: `linalg.matmul` with `N=1, transpose_b=True` on AMD/fp32 writes only one output row

**Status**: Not filed upstream (per user request). Not reported in existing modular/modular issues (#5700 is a different naive-matmul-kernel bug).

**Severity**: Silent wrong-output. No error, no NaN — first row of C is correct, remaining rows stay at their pre-call value (zero for a fresh buffer).

**Affected configurations confirmed**:
- AMD GPU (Strix Halo gfx1151)
- Input dtype `float32` (not `bfloat16`)
- C shape `(M, 1)`, A shape `(M, K)`, B shape `(1, K)` with `transpose_b=True`
- `M >= 1`, any `K` we've tried (e.g. `M=60, K=512`)

By inspection of source, the bug should reproduce on any AMD GPU when the dispatch chain lands in the gemv-with-transpose branch.

---

## Minimal repro (Mojo)

```mojo
from gpu.host import DeviceContext
from layout import Idx, TileTensor, row_major
from linalg.matmul import matmul

def main() raises:
    var ctx = DeviceContext()
    comptime M = 60
    comptime K = 512
    comptime N = 1

    # A = float(i*K + j); B = ones(N, K); bias=0
    # Expected: C[i, 0] = sum_j A[i, j] * B[0, j] = sum_j (i*K + j) = i*K^2 + K*(K-1)/2
    # i=0: 0 + 130816 = 130816
    # i=1: 262144 + 130816 = 392960
    var a = ctx.enqueue_create_buffer[DType.float32](M * K)
    var b = ctx.enqueue_create_buffer[DType.float32](N * K)
    var c = ctx.enqueue_create_buffer[DType.float32](M * N)
    b.enqueue_fill(1.0)
    c.enqueue_fill(0.0)
    with a.map_to_host() as h:
        for i in range(M * K):
            h[i] = Float32(i)

    var a_t = TileTensor(a, row_major(Idx(M), Idx(K)))
    var b_t = TileTensor(b, row_major(Idx(N), Idx(K)))
    var c_t = TileTensor(c, row_major(Idx(M), Idx(N)))

    matmul[target="gpu", transpose_b=True](c_t, a_t, b_t, DeviceContextPtr(ctx))
    ctx.synchronize()

    with c.map_to_host() as h:
        for i in range(8):
            print("C[", i, "] =", h[i])
```

**Observed output:**

```
C[ 0 ] = 130816.0      ✓ correct
C[ 1 ] = 0.0           ✗ expected 392960.0
C[ 2 ] = 0.0           ✗ expected 655104.0
C[ 3 ] = 0.0           ✗ expected 917248.0
... (all subsequent rows: 0.0)
```

If we set `c.enqueue_fill(42.0)` first, rows 1..M-1 stay at 42.0 — they are **never written**, not even with garbage.

### Cross-runtime comparison (the dispositive evidence)

Same shapes, same data, same `transpose_b=True`. Only the runtime differs:

| Runtime                                          | C[0]   | C[1]    | C[2..7]     |
|--------------------------------------------------|--------|---------|-------------|
| PyTorch (CPU) — `A @ B.T`                        | 130816 | 392960  | all correct |
| MAX `linalg.matmul[target="cpu"]`                | 130816 | 392960  | all correct |
| **MAX `linalg.matmul[target="gpu"]`** (AMD/fp32) | 130816 | **0.0** | **all 0.0** |

**MAX itself produces the correct answer on CPU** for the exact same call.
The bug only appears in the GPU dispatch — which is the dispositive proof
this isn't a usage error.

Reproduction scripts in this repo:
- Torch reference: `scripts/torch_linear_60x512x1.py`
- MAX CPU: `tests/test_linear_60x512x1_max_cpu.mojo`
- MAX GPU (fails): `tests/test_linear_60x512x1.mojo`

---

## Why it happens — code-level proof

All file references are against modular/modular at the version shipped in
`max>=26.4.0.dev2026050106` (the mojopkg path is
`.pixi/envs/default/lib/mojo/linalg.mojopkg`). I traced via the source tree at
`max/kernels/src/linalg/`.

### Step 1. `linalg.matmul.matmul` enters the dispatcher

`linalg/matmul/__init__.mojo:79`:

```mojo
fn matmul[ ..., target: StaticString = "cpu" ](
    c: NDBuffer[mut=True, _, 2, _, _],
    a: NDBuffer[_, 2, _, _],
    b: NDBuffer[_, 2, _, _],
    ctx: Optional[DeviceContext],
) raises:
```

With `target="gpu"`, this routes to `_matmul_gpu` in `linalg/matmul/gpu/__init__.mojo`.

### Step 2. AMD/float32 path skips the fast kernels

`linalg/matmul/gpu/__init__.mojo:519`:

```mojo
alias matmul_supported_format_amd = (
    (a_type is DType.bfloat16 or a_type in amd_float8_dtypes)
    and b_type == a_type
    and c_type in (DType.float32, DType.bfloat16)
)
alias matmul_supported_format =
    matmul_supported_format_amd if has_amd_gpu_accelerator()
    else matmul_supported_format_nvidia
```

For `a_type = float32, b_type = float32` on AMD, `matmul_supported_format_amd
== False`. So the entire `if (matmul_supported_format and …)` block at
line 620 is skipped.

### Step 3. The `n == 1` shortcut

The next dispatch (line ~803) is:

```mojo
@parameter
if not a_type.is_float8():
    if n == 1 or m == 1:
        gemv_gpu[
            transpose_b=transpose_b,
            elementwise_lambda_fn=elementwise_lambda_wrapper,
        ](c, a, b, ctx)
        return
```

With `N=1`, control transfers to `gemv_gpu` in `linalg/gemv.mojo:111`.

### Step 4. `gemv_gpu` selects `GEMV_KERNEL`

`linalg/gemv.mojo:140`:

```mojo
if n == 1:
    @parameter
    if a.type is DType.bfloat16:
        if k % simd_width == 0:
            kernel_func = GEMVAlgorithm.GEMV_KERNEL_VECTOR
        else:
            kernel_func = GEMVAlgorithm.GEMV_KERNEL
    else:
        kernel_func = GEMVAlgorithm.GEMV_KERNEL
elif m == 1 and transpose_b == True:
    ...
```

For float32, `kernel_func = GEMV_KERNEL`. Note this top-level selector
**branches on `n==1` first and ignores `transpose_b`**. The next `elif`
(`m == 1 and transpose_b == True`) is the *intended* path for a (1, K) input
vector multiplied by a transposed (N, K) matrix — but we never reach it
because our case already matched `n == 1`.

### Step 5. The buggy launch in `gemv_gpu_dispatch`

`linalg/gemv.mojo:716` (the `transpose_b == True` arm of
`GEMV_KERNEL`):

```mojo
elif kernel_func is GEMVAlgorithm.GEMV_KERNEL and transpose_b == True:
    logger.info("Executing: GEMV_KERNEL (with transpose)")

    alias kernel = gemv_kernel[
        c.type,
        b.type,          # ← (1) types swapped: original-b plays "a" role
        a.type,
        reduction_method = warp.ReductionMethod.WARP,
        transpose_b=transpose_b,
        elementwise_lambda_fn=elementwise_lambda_fn,
    ]
    ctx.enqueue_function_checked[kernel, kernel](
        c_buffer,
        b_buffer,        # ← (2) buffers swapped: B passed as A
        a_buffer,
        n,               # ← (3) shape arg "m" receives our N (=1)
        m,               # ← (4) shape arg "n" receives our M
        k,
        grid_dim=ceildiv(n, WARPS_PER_BLOCK),   # ← (5) grid sized by N
        block_dim=WARP_SIZE * WARPS_PER_BLOCK,
    )
```

This swap is **correct** for the intended case (M=1, transpose_b=True): "C = a
@ B.T where a is 1×K, B is N×K → C is 1×N. View it as the gemv `C.T = B @
a.T`, i.e. (N×K) @ (K×1) → (N×1)". The kernel gets `(b_buffer as 'A',
a_buffer as 'B', shape m=N, n=1, k=K)`, launches `ceildiv(N, WARPS_PER_BLOCK)`
warps, and produces N output rows that the kernel then writes back at
`Index(0, warp_id)`.

**The dispatch is wrong for our case** (N=1, transpose_b=True, M=60). The
same swap is applied: shape arg "m" gets our N=1, shape arg "n" gets our M=60.
With `grid_dim = ceildiv(N=1, 16) = 1`, only 1 thread block runs (64 threads
on AMD, 1 warp). The kernel's row guard:

`linalg/gemv.mojo:147` in `gemv_kernel`:

```mojo
if warp_id >= m: return
```

…with the swapped `m=1`, allows only `warp_id == 0` to compute. That warp
computes the dot product correctly for *what it thinks is* row 0 — it reads
the original B (1 row) and the original A's row 0 — and writes the result.

The output write:

`linalg/gemv.mojo:165`:

```mojo
if lane_id() == 0:
    if elementwise_lambda_fn:
        elementwise_lambda[c_type, 1](
            reverse_idx[transpose_b](Int(warp_id), 0),
            accum.cast[c_type](),
        )
    else:
        c[warp_id] = accum.cast[c_type]()
```

With `transpose_b=True`, `reverse_idx` swaps the output coordinates:

```mojo
fn reverse_idx[transpose: Bool](x: Int, y: Int) -> IndexList[2]:
    return Index(y, x) if transpose else Index(x, y)
```

So we write to C at `Index(0, 0)`. The kernel only had one warp, so only one
element of the (M=60, N=1) output gets written.

**Summary of the bug**: the top-level selector at `gemv.mojo:140` matches
**any case with `n == 1`** and picks `GEMV_KERNEL` regardless of `transpose_b`.
The downstream `transpose_b == True` dispatch arm at `gemv.mojo:716` is
only correct under the assumption that `m == 1` (the *other* case that's
supposed to route there from the `elif m == 1 and transpose_b == True`
branch). When a `(M>>1, N=1, transpose_b=True)` call slips through, the
launch grid is sized by `N=1`, only one warp runs, and only the first row of
the output is written.

---

## Workarounds (in order of preference)

1. **Don't use `transpose_b=True` for `N=1` linear projections.** Store the
   weight as `(in_features, out_features)` and call `matmul[target="gpu",
   transpose_b=False]`. The `transpose_b=False` branch at `gemv.mojo:692`
   uses `grid_dim=ceildiv(m, WARPS_PER_BLOCK)`, which is correct.
2. **Pad `N` to a multiple of 4.** This makes
   `amdgpu_matmul_cond = has_amd_gpu_accelerator() and n % 4 == 0` true,
   `matmul_supported_format` would also need to be true (bf16/fp8 only), so on
   fp32 this still falls through. So padding alone doesn't help on fp32
   without also changing dtype.
3. **Bypass `matmul` for this shape.** Write a custom dot-product kernel:
   ```mojo
   @always_inline
   @parameter
   @__copy_capture(x_btc_p, w_ptr, bias_ptr, f0o_ptr)
   def fn[width: Int, rank: Int, alignment: Int = 1](idx: IndexList[rank]):
       var i = idx[0]
       var acc: Float32 = bias_ptr[0]
       for k in range(K):
           acc += x_btc_p[i * K + k] * w_ptr[k]
       f0o_ptr[i] = acc
   elementwise[fn, simd_width=1, target="gpu"](IndexList[1](M), dctx)
   ```
   This is what max-impl currently uses for the f0_predictor classifier.

---

## Suggested fix (for upstream's reference, not filed)

The simplest correct dispatch in `gemv.mojo:140` would split the `n == 1`
branch on `transpose_b`:

```mojo
if n == 1 and transpose_b == False:
    # Original `n == 1` path: matrix-column-vector, no transpose.
    kernel_func = GEMVAlgorithm.GEMV_KERNEL

elif n == 1 and transpose_b == True:
    # C = A @ B.T with B as (1, K) — equivalent to the m == 1 case
    # *without* the swap. Route to the non-transpose kernel by treating
    # B as a (K,) column vector and computing C[m] = sum_k A[m,k] * B[k]
    # directly.
    kernel_func = GEMVAlgorithm.GEMV_KERNEL   # but dispatch to no-transpose arm

elif m == 1 and transpose_b == True:
    # existing intended path
    ...
```

…and adjust the launch arms accordingly. Or, equivalently, drop the swap in
the `transpose_b == True` arm when `n == 1` (since with N=1 the swap is
redundant — `C.T` and `C` have the same shape).

---

## Verification artifacts in this repo

- `tests/test_linear_60x512x1.mojo` — minimal Mojo repro using our `Linear`
  wrapper (which calls `linalg.matmul[transpose_b=True]`). Output:
  row 0 correct, rows 1..59 = 0.
- `tests/test_source_path_parity.mojo` — original failing parity test for
  `f0_predictor.classifier`. The classifier is exactly this shape
  (`(M=60, K=512, N=1, transpose_b=True)`).
- `src/hift_generator.mojo:614-636` — the workaround (custom dot-product
  kernel) that restored bit-exact parity with upstream torch (`max-abs
  9.8e-7, rel L2 1.4e-6` against the dumped oracle).
