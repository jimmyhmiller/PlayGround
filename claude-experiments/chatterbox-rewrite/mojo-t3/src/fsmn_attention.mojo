"""FSMN multi-head attention kernels.

Matches s3tokenizer.model_v2.FSMNMultiHeadAttention:

  query = Linear(D, D)   (with bias)
  key   = Linear(D, D, bias=False)
  value = Linear(D, D)   (with bias)
  out   = Linear(D, D)   (with bias)
  fsmn_block = Conv1d(D, D, kernel_size=31, padding=0, groups=D, bias=False)
               (depthwise conv with center-aligned pad of 15 left, 15 right)

Per-step:
  q = query(x); k = key(x); v = value(x)
  split into (B, S, H, Dh)
  apply RoPE to q, k (using cat-concat-twice RoPE convention from s3tokenizer)
  scale = (D/H)^-0.25     # both Q and K multiplied → effective scale is 1/sqrt(D/H)
  fsm_mem = fsmn_block(pad(v.reshape(B, S, D))).reshape(B, S, D)   # depthwise on v
  fsm_mem += v.reshape(B, S, D)
  fsm_mem *= mask_pad    # zero out padded timesteps
  q = q.permute(0, 2, 1, 3) * scale
  k = k.permute(0, 2, 3, 1) * scale   # (B, H, Dh, S)
  qk = q @ k                          # (B, H, S, S)
  qk += mask
  attn = softmax(qk)
  out_attn = (attn @ v.permute(0, 2, 1, 3)).permute(0, 2, 1, 3).flatten(start_dim=2)
  return out(out_attn) + fsm_mem

This is largely standard MHA + a parallel FSMN memory branch via grouped conv.
"""
from std.gpu import block_idx, thread_idx
from layout import TileTensor, TensorLayout, row_major


def fsmn_depthwise_conv_kernel[
    dtype: DType,
    XLayout: TensorLayout,
    WLayout: TensorLayout,
    OutLayout: TensorLayout,
    KSIZE: Int, LEFT_PAD: Int, RIGHT_PAD: Int,
    BLOCK: Int,
](
    output: TileTensor[dtype, OutLayout, MutAnyOrigin],   # (B, S, D)
    x: TileTensor[dtype, XLayout, MutAnyOrigin],          # (B, S, D)
    w: TileTensor[dtype, WLayout, MutAnyOrigin],          # (D, 1, KSIZE)  depth=1 group
    batch: Int, seq: Int, d: Int,
):
    """Depthwise (groups=D) Conv1d over S, with constant-pad LEFT_PAD/RIGHT_PAD zeros.

    out[b, s, c] = sum_k x[b, s + k - LEFT_PAD, c] * w[c, 0, k]
        where out-of-range positions read 0.

    Launch: grid = B*S, block_dim = BLOCK over D.
    """
    comptime assert x.flat_rank == 3
    comptime assert w.flat_rank == 3
    comptime assert output.flat_rank == 3
    var bid = block_idx.x
    var tid = thread_idx.x
    var s = bid % seq
    var b = bid // seq
    var c = tid
    while c < d:
        var acc: Float32 = 0.0
        for k in range(KSIZE):
            var src = s + k - LEFT_PAD
            if src >= 0 and src < seq:
                var xv = rebind[Scalar[dtype]](x[b, src, c]).cast[DType.float32]()
                var wv = rebind[Scalar[dtype]](w[c, 0, k]).cast[DType.float32]()
                acc += xv * wv
        output[b, s, c] = rebind[output.ElementType](acc.cast[dtype]())
        c += BLOCK


def fsmn_memory_kernel[
    dtype: DType,
    FLayout: TensorLayout, VLayout: TensorLayout,
    MaskLayout: TensorLayout,
    OutLayout: TensorLayout,
    BLOCK: Int,
](
    output: TileTensor[dtype, OutLayout, MutAnyOrigin],   # (B, S, D)
    fsmn_conv: TileTensor[dtype, FLayout, MutAnyOrigin],   # (B, S, D) — conv output
    v: TileTensor[dtype, VLayout, MutAnyOrigin],           # (B, S, D) — value
    mask_pad: TileTensor[dtype, MaskLayout, MutAnyOrigin], # (B, S, 1) — 1 for valid, 0 for pad
    batch: Int, seq: Int, d: Int,
):
    """output = (fsmn_conv + v) * mask_pad."""
    comptime assert fsmn_conv.flat_rank == 3
    comptime assert v.flat_rank == 3
    comptime assert mask_pad.flat_rank == 3
    comptime assert output.flat_rank == 3
    var bid = block_idx.x
    var tid = thread_idx.x
    var s = bid % seq
    var b = bid // seq
    var m = rebind[Scalar[dtype]](mask_pad[b, s, 0]).cast[DType.float32]()
    var c = tid
    while c < d:
        var fv = rebind[Scalar[dtype]](fsmn_conv[b, s, c]).cast[DType.float32]()
        var vv = rebind[Scalar[dtype]](v[b, s, c]).cast[DType.float32]()
        output[b, s, c] = rebind[output.ElementType](((fv + vv) * m).cast[dtype]())
        c += BLOCK


def rope_s3tokenizer_kernel[
    dtype: DType,
    XLayout: TensorLayout,
    CSLayout: TensorLayout,
    OutLayout: TensorLayout,
    D: Int,
    HALF: Int,
](
    output: TileTensor[dtype, OutLayout, MutAnyOrigin],   # (B, S, H, D)
    x: TileTensor[dtype, XLayout, MutAnyOrigin],          # (B, S, H, D)
    cos: TileTensor[dtype, CSLayout, MutAnyOrigin],       # (S, HALF) — real part of freqs_cis
    sin: TileTensor[dtype, CSLayout, MutAnyOrigin],       # (S, HALF) — imag part of freqs_cis
    seq: Int, n_heads: Int,
):
    """Apply s3tokenizer's RoPE to x in layout (B, S, H, D).

    For each (b, s, h, i):
      let cs_idx = i if i < HALF else i - HALF   (cos/sin tiled twice)
      let half_l = x[b, s, h, :HALF]; half_r = x[b, s, h, HALF:]
      xr = cat(-half_r, half_l, dim=-1)
      out[b, s, h, i] = x[b, s, h, i] * cos[s, cs_idx] + xr[..., i] * sin[s, cs_idx]

    Launch: grid = B*S*H, block_dim = D.
    """
    comptime assert x.flat_rank == 4
    comptime assert cos.flat_rank == 2
    comptime assert sin.flat_rank == 2
    comptime assert output.flat_rank == 4
    comptime assert HALF * 2 == D
    var bid = block_idx.x
    var i = thread_idx.x
    if i >= D: return
    var h = bid % n_heads
    var s = (bid // n_heads) % seq
    var b = bid // (n_heads * seq)
    var x_i = rebind[Scalar[dtype]](x[b, s, h, i]).cast[DType.float32]()
    var xr: Float32 = 0.0
    if i < HALF:
        var paired = rebind[Scalar[dtype]](x[b, s, h, i + HALF]).cast[DType.float32]()
        xr = -paired
    else:
        var paired = rebind[Scalar[dtype]](x[b, s, h, i - HALF]).cast[DType.float32]()
        xr = paired
    var cs_idx = i if i < HALF else (i - HALF)
    var c = rebind[Scalar[dtype]](cos[s, cs_idx]).cast[DType.float32]()
    var sn = rebind[Scalar[dtype]](sin[s, cs_idx]).cast[DType.float32]()
    var result: Float32 = x_i * c + xr * sn
    output[b, s, h, i] = rebind[output.ElementType](result.cast[dtype]())


def multiply_mask_3d_kernel[
    dtype: DType, XL: TensorLayout, ML: TensorLayout, OL: TensorLayout, BLOCK: Int,
](
    dst: TileTensor[dtype, OL, MutAnyOrigin],
    x: TileTensor[dtype, XL, MutAnyOrigin],
    m: TileTensor[dtype, ML, MutAnyOrigin],
    n0: Int, n1: Int, n2: Int,
):
    """dst[i0, i1, i2] = x[i0, i1, i2] * m[i0, i1, 0]. Launch: grid=n0*n1, block_dim=BLOCK over n2."""
    comptime assert x.flat_rank == 3
    comptime assert m.flat_rank == 3
    comptime assert dst.flat_rank == 3
    var bid = block_idx.x
    var tid = thread_idx.x
    var i1 = bid % n1
    var i0 = bid // n1
    var mv = rebind[Scalar[dtype]](m[i0, i1, 0]).cast[DType.float32]()
    var i2 = tid
    while i2 < n2:
        var xv = rebind[Scalar[dtype]](x[i0, i1, i2]).cast[DType.float32]()
        dst[i0, i1, i2] = rebind[dst.ElementType]((xv * mv).cast[dtype]())
        i2 += BLOCK


def permute_bshd_to_bhsd_kernel[
    dtype: DType, XL: TensorLayout, OL: TensorLayout,
    HH: Int, DDH: Int,
](
    dst: TileTensor[dtype, OL, MutAnyOrigin],
    x: TileTensor[dtype, XL, MutAnyOrigin],
    batch: Int, seq: Int,
):
    """(B, S, H, Dh) → (B, H, S, Dh). Launch: grid=B*S*H, block_dim=DDH."""
    comptime assert x.flat_rank == 4
    comptime assert dst.flat_rank == 4
    var bid = block_idx.x
    var d = thread_idx.x
    if d >= DDH: return
    var h = bid % HH
    var s = (bid // HH) % seq
    var b = bid // (HH * seq)
    var v = rebind[Scalar[dtype]](x[b, s, h, d]).cast[DType.float32]()
    dst[b, h, s, d] = rebind[dst.ElementType](v.cast[dtype]())


def permute_bhsd_to_bsd_kernel[
    dtype: DType, XL: TensorLayout, OL: TensorLayout,
    HH: Int, DDH: Int,
](
    dst: TileTensor[dtype, OL, MutAnyOrigin],
    x: TileTensor[dtype, XL, MutAnyOrigin],
    batch: Int, seq: Int,
):
    """(B, H, S, Dh) → (B, S, H*Dh). Launch: grid=B*S*H, block_dim=DDH."""
    comptime assert x.flat_rank == 4
    comptime assert dst.flat_rank == 3
    var bid = block_idx.x
    var d = thread_idx.x
    if d >= DDH: return
    var h = bid % HH
    var s = (bid // HH) % seq
    var b = bid // (HH * seq)
    var v = rebind[Scalar[dtype]](x[b, h, s, d]).cast[DType.float32]()
    dst[b, s, h * DDH + d] = rebind[dst.ElementType](v.cast[dtype]())


def scale_4d_kernel[
    dtype: DType, L: TensorLayout, BLOCK: Int,
](
    dst: TileTensor[dtype, L, MutAnyOrigin],
    n0: Int, n1: Int, n2: Int, n3: Int,
    scale: Float32,
):
    """In-place scale: dst[i, j, k, l] *= scale. Launch: grid=n0*n1*n2, block_dim=BLOCK."""
    comptime assert dst.flat_rank == 4
    var bid = block_idx.x
    var tid = thread_idx.x
    var i2 = bid % n2
    var rem = bid // n2
    var i1 = rem % n1
    var i0 = rem // n1
    var i3 = tid
    while i3 < n3:
        var v = rebind[Scalar[dtype]](dst[i0, i1, i2, i3]).cast[DType.float32]()
        dst[i0, i1, i2, i3] = rebind[dst.ElementType]((v * scale).cast[dtype]())
        i3 += BLOCK
