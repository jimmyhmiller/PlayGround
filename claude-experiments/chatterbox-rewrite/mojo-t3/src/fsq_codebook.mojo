"""FSQ codebook (Finite Scalar Quantization) — speech-token index encoder.

Matches s3tokenizer.model_v2.FSQCodebook.encode:
    h = project_down(x)            # Linear: (B*T, D) → (B*T, 8)
    h = tanh(h) * 0.9990000128746033
    h = round(h) + 1               # ∈ {0, 1, 2}
    powers = [3^0, 3^1, ..., 3^7]  # length 8 (= 2^level for level=3)
    mu = sum_i h[..., i] * powers[i]   # → integer index ∈ [0, 6561)

The project_down Linear is folded into a single fused kernel.
"""
from std.math import tanh
from std.gpu import block_idx, thread_idx
from layout import TileTensor, TensorLayout, row_major


def fsq_encode_kernel[
    dtype: DType,
    XLayout: TensorLayout,
    WLayout: TensorLayout,
    BLayout: TensorLayout,
    OutLayout: TensorLayout,
    DIM: Int,          # input embedding dim
    BLOCK: Int,
](
    indices: TileTensor[DType.int32, OutLayout, MutAnyOrigin],  # (B, T)
    x: TileTensor[dtype, XLayout, MutAnyOrigin],                 # (B, T, DIM)
    w: TileTensor[dtype, WLayout, MutAnyOrigin],                 # (8, DIM) — project_down
    bias: TileTensor[dtype, BLayout, MutAnyOrigin],              # (8,)
    batch: Int, time: Int,
):
    """One thread = one (b, t) → produces a single token index. Launch: grid = B*T."""
    comptime assert x.flat_rank == 3
    comptime assert w.flat_rank == 2
    comptime assert bias.flat_rank == 1
    comptime assert indices.flat_rank == 2
    var bid = block_idx.x
    var tid = thread_idx.x
    if tid != 0:
        return
    var t = bid % time
    var b = bid // time
    # Compute 8-dim projection.
    var h0: Float32 = 0.0
    var h1: Float32 = 0.0
    var h2: Float32 = 0.0
    var h3: Float32 = 0.0
    var h4: Float32 = 0.0
    var h5: Float32 = 0.0
    var h6: Float32 = 0.0
    var h7: Float32 = 0.0
    for i in range(DIM):
        var xv = rebind[Scalar[dtype]](x[b, t, i]).cast[DType.float32]()
        h0 += xv * rebind[Scalar[dtype]](w[0, i]).cast[DType.float32]()
        h1 += xv * rebind[Scalar[dtype]](w[1, i]).cast[DType.float32]()
        h2 += xv * rebind[Scalar[dtype]](w[2, i]).cast[DType.float32]()
        h3 += xv * rebind[Scalar[dtype]](w[3, i]).cast[DType.float32]()
        h4 += xv * rebind[Scalar[dtype]](w[4, i]).cast[DType.float32]()
        h5 += xv * rebind[Scalar[dtype]](w[5, i]).cast[DType.float32]()
        h6 += xv * rebind[Scalar[dtype]](w[6, i]).cast[DType.float32]()
        h7 += xv * rebind[Scalar[dtype]](w[7, i]).cast[DType.float32]()
    # Add bias.
    h0 += rebind[Scalar[dtype]](bias[0]).cast[DType.float32]()
    h1 += rebind[Scalar[dtype]](bias[1]).cast[DType.float32]()
    h2 += rebind[Scalar[dtype]](bias[2]).cast[DType.float32]()
    h3 += rebind[Scalar[dtype]](bias[3]).cast[DType.float32]()
    h4 += rebind[Scalar[dtype]](bias[4]).cast[DType.float32]()
    h5 += rebind[Scalar[dtype]](bias[5]).cast[DType.float32]()
    h6 += rebind[Scalar[dtype]](bias[6]).cast[DType.float32]()
    h7 += rebind[Scalar[dtype]](bias[7]).cast[DType.float32]()
    # tanh * 0.999 + round + 1 → {0, 1, 2}.
    comptime SCALE: Float32 = 0.9990000128746033
    var s0 = tanh(h0) * SCALE
    var s1 = tanh(h1) * SCALE
    var s2 = tanh(h2) * SCALE
    var s3 = tanh(h3) * SCALE
    var s4 = tanh(h4) * SCALE
    var s5 = tanh(h5) * SCALE
    var s6 = tanh(h6) * SCALE
    var s7 = tanh(h7) * SCALE
    # Python's round() does banker's rounding but ranges (-0.999, 0.999) make
    # this equivalent to: floor(s+0.5) for s>=0, ceil(s-0.5) for s<0.
    # torch.round uses round-half-to-even. For (-0.999, 0.999) the only half
    # value is 0.5, which rounds to 0 with banker's; but s = tanh(z)*0.999
    # never equals exactly 0.5 in practice (z ≈ 0.549).
    var r0 = Int(s0 + 0.5) if s0 >= 0.0 else -Int(-s0 + 0.5)
    var r1 = Int(s1 + 0.5) if s1 >= 0.0 else -Int(-s1 + 0.5)
    var r2 = Int(s2 + 0.5) if s2 >= 0.0 else -Int(-s2 + 0.5)
    var r3 = Int(s3 + 0.5) if s3 >= 0.0 else -Int(-s3 + 0.5)
    var r4 = Int(s4 + 0.5) if s4 >= 0.0 else -Int(-s4 + 0.5)
    var r5 = Int(s5 + 0.5) if s5 >= 0.0 else -Int(-s5 + 0.5)
    var r6 = Int(s6 + 0.5) if s6 >= 0.0 else -Int(-s6 + 0.5)
    var r7 = Int(s7 + 0.5) if s7 >= 0.0 else -Int(-s7 + 0.5)
    # Add +1 → {0, 1, 2}.
    r0 += 1; r1 += 1; r2 += 1; r3 += 1
    r4 += 1; r5 += 1; r6 += 1; r7 += 1
    # mu = sum r_i * 3^i.
    var mu: Int = r0 + r1 * 3 + r2 * 9 + r3 * 27 + r4 * 81 + r5 * 243 + r6 * 729 + r7 * 2187
    indices[b, t] = Int32(mu)
