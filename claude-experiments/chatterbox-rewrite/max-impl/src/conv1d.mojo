"""Conv1d expressed via `std.algorithm.functional.elementwise[..., target="gpu"]`.

MAX exposes a Conv2D primitive (`_interpreter_ops/conv_ops.mojo`) but no
direct Conv1D. The Conv2D implementation is itself just a capturing closure
dispatched through `elementwise` — we follow the exact same pattern here,
specialised for the 1D case used everywhere in Chatterbox.

PyTorch nn.Conv1d semantics: input (B, C_in, L_in), weight (C_out, C_in/groups, K),
bias (C_out,) optional, with stride/padding/dilation. Output (B, C_out, L_out).
"""
from std.gpu.host import DeviceContext, DeviceBuffer
from std.runtime.asyncrt import DeviceContextPtr
from std.algorithm.functional import elementwise, IndexList


@fieldwise_init
struct Conv1d(Copyable, Movable):
    var weight: DeviceBuffer[DType.float32]   # (C_out, C_in/groups, K)
    var bias:   DeviceBuffer[DType.float32]   # (C_out,) — may be a zero-len dummy
    var c_in:   Int
    var c_out:  Int
    var k:      Int
    var stride: Int
    var padding: Int
    var dilation: Int
    var groups: Int
    var has_bias: Bool


def conv1d_forward(
    mut ctx: DeviceContext,
    mut module: Conv1d,
    mut x_buf: DeviceBuffer[DType.float32],     # (B, C_in, L_in)
    mut out_buf: DeviceBuffer[DType.float32],   # (B, C_out, L_out)
    batch: Int, l_in: Int, l_out: Int,
) raises:
    """Apply 1D convolution. Caller supplies pre-computed l_out:
       l_out = (l_in + 2*pad - dilation*(k-1) - 1) // stride + 1.
    """
    var x_ptr = x_buf.unsafe_ptr()
    var w_ptr = module.weight.unsafe_ptr()
    var b_ptr = module.bias.unsafe_ptr()
    var out_ptr = out_buf.unsafe_ptr()

    var c_in = module.c_in
    var c_out = module.c_out
    var k = module.k
    var stride = module.stride
    var pad = module.padding
    var dil = module.dilation
    var groups = module.groups
    var has_bias = module.has_bias
    var ic_per_g = c_in // groups
    var oc_per_g = c_out // groups

    var total = batch * c_out * l_out
    var dctx = DeviceContextPtr(ctx)

    @always_inline
    @parameter
    @__copy_capture(
        x_ptr, w_ptr, b_ptr, out_ptr,
        c_in, c_out, k, stride, pad, dil,
        l_in, l_out, ic_per_g, oc_per_g, has_bias,
    )
    def func[width: Int, rank: Int, alignment: Int = 1](idx: IndexList[rank]):
        var i = idx[0]
        # Decompose i into (b, oc, lo).
        var b = i // (c_out * l_out)
        var rem = i - b * c_out * l_out
        var oc = rem // l_out
        var lo = rem - oc * l_out

        var g = oc // oc_per_g
        var ic_start = g * ic_per_g

        var acc: Float32 = 0.0
        if has_bias:
            acc = b_ptr[oc]

        for kk in range(k):
            var li = lo * stride - pad + kk * dil
            if li >= 0 and li < l_in:
                for ic_off in range(ic_per_g):
                    var ic = ic_start + ic_off
                    var x_idx = b * c_in * l_in + ic * l_in + li
                    var w_idx = oc * ic_per_g * k + ic_off * k + kk
                    acc += x_ptr[x_idx] * w_ptr[w_idx]
        out_ptr[i] = acc

    elementwise[func, simd_width=1, target="gpu"](
        IndexList[1](total), dctx,
    )
