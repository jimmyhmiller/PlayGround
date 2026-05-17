"""
RoPE GPU kernel for the Llama backbone of Chatterbox T3.

Spec (matches transformers.apply_rotary_pos_emb exactly):
  q, k:    (B, H, S, D)             input dtype
  cos, sin (B, S, D)                 input dtype, broadcasts across H
  out[b,h,s,i] = q[b,h,s,i] * cos[b,s,i] + rotate_half(q)[b,h,s,i] * sin[b,s,i]

  rotate_half(x)[i] =
    -x[i + D/2]   if i <  D/2
     x[i - D/2]   if i >= D/2

Computation is in fp32; cast back to input dtype on store. Cos/sin themselves
are stored in input dtype (HF casts cos/sin to x.dtype before the multiply).

Launch: one block per (B*H*S) row, D threads per block, one element each.
"""

from std.gpu import block_idx, thread_idx
from layout import TileTensor, TensorLayout


def rope_kernel[
    dtype: DType,
    QLayout: TensorLayout,
    CSLayout: TensorLayout,
    OutLayout: TensorLayout,
    D: Int,
    HALF: Int,
](
    output: TileTensor[dtype, OutLayout, MutAnyOrigin],
    q: TileTensor[dtype, QLayout, MutAnyOrigin],
    cos: TileTensor[dtype, CSLayout, MutAnyOrigin],
    sin: TileTensor[dtype, CSLayout, MutAnyOrigin],
    n_heads: Int,
    seq_len: Int,
):
    comptime assert q.flat_rank == 4, "expected (B,H,S,D)"
    comptime assert output.flat_rank == 4, "expected (B,H,S,D)"
    comptime assert cos.flat_rank == 3, "expected (B,S,D)"
    comptime assert sin.flat_rank == 3, "expected (B,S,D)"
    comptime assert HALF * 2 == D, "D must be even"

    # Unflatten linear block index into (b, h, s).
    var bid = block_idx.x
    var i = thread_idx.x  # 0..D-1

    var s = bid % seq_len
    var h = (bid // seq_len) % n_heads
    var b = bid // (seq_len * n_heads)

    # Load q[b,h,s,i] and the paired element used by rotate_half.
    var q_i = rebind[Scalar[dtype]](q[b, h, s, i]).cast[DType.float32]()

    # rotate_half(x)[i] = (-x[i + HALF]) if i < HALF else (x[i - HALF])
    var rh: Float32 = 0.0
    if i < HALF:
        var paired = rebind[Scalar[dtype]](q[b, h, s, i + HALF]).cast[DType.float32]()
        rh = -paired
    else:
        var paired = rebind[Scalar[dtype]](q[b, h, s, i - HALF]).cast[DType.float32]()
        rh = paired

    var c = rebind[Scalar[dtype]](cos[b, s, i]).cast[DType.float32]()
    var sn = rebind[Scalar[dtype]](sin[b, s, i]).cast[DType.float32]()

    var result_fp32 = q_i * c + rh * sn
    var result = result_fp32.cast[dtype]()
    output[b, h, s, i] = rebind[output.ElementType](result)
