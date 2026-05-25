"""RoPE table builder — extracted from src/text_embed.mojo to avoid pulling in T3 deps."""
from std.gpu.host import DeviceContext, DeviceBuffer
from std.runtime.asyncrt import DeviceContextPtr
from std.algorithm.functional import elementwise, IndexList
from std.math import sin as msin, cos as mcos, exp as mexp


def build_rope_tables(
    mut ctx: DeviceContext,
    max_ctx: Int, head_dim: Int,
    mut cos_buf: DeviceBuffer[DType.float32],
    mut sin_buf: DeviceBuffer[DType.float32],
) raises:
    """HF Llama RoPE: cos/sin tables for positions [0, max_ctx)."""
    var cp = cos_buf.unsafe_ptr()
    var sp = sin_buf.unsafe_ptr()
    var d_half = head_dim // 2

    @always_inline
    @parameter
    @__copy_capture(cp, sp, max_ctx, head_dim, d_half)
    def rope_fn[width: Int, rank: Int, alignment: Int = 1](idx: IndexList[rank]):
        var i = idx[0]
        var t = i // head_dim
        var k = i - t * head_dim
        var k_half: Int
        if k < d_half:
            k_half = k
        else:
            k_half = k - d_half
        var exponent: Float32 = (2.0 * Float32(k_half)) / Float32(head_dim)
        var inv_freq = mexp(-exponent * 9.210340371976184)
        var pos: Float32 = Float32(t) * inv_freq
        cp[i] = mcos(pos)
        sp[i] = msin(pos)
    elementwise[rope_fn, simd_width=1, target="gpu"](
        IndexList[1](max_ctx * head_dim), DeviceContextPtr(ctx),
    )
