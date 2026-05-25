"""LSTM built from MAX ops (`linalg.matmul` + `elementwise`).

PyTorch nn.LSTM semantics:
    pre_gates = x_t @ W_ih.T + b_ih + h_prev @ W_hh.T + b_hh     # (B, 4H)
    i = sigmoid(pre_gates[:, 0:H])
    f = sigmoid(pre_gates[:, H:2H])
    g = tanh(pre_gates[:, 2H:3H])
    o = sigmoid(pre_gates[:, 3H:4H])
    c = f * c_prev + i * g
    h = o * tanh(c)

We orchestrate this per timestep on the host, issuing two `linalg.matmul`
calls and one fused `elementwise` GPU op per step. Weights are (4*H, IN)
and (4*H, H) — the same packing PyTorch uses for weight_ih and weight_hh.
"""
from std.math import exp, tanh
from std.sys.info import has_accelerator
from std.gpu.host import DeviceContext, DeviceBuffer
from std.runtime.asyncrt import DeviceContextPtr
from std.algorithm.functional import elementwise, IndexList
from layout import Idx, TileTensor, row_major

from linalg.matmul import matmul as nn_matmul


@fieldwise_init
struct LSTMLayer(Copyable, Movable):
    """One LSTM layer. Weights packed PyTorch-style: 4*H rows = [i, f, g, o]."""
    var weight_ih: DeviceBuffer[DType.float32]    # (4*H, IN)
    var weight_hh: DeviceBuffer[DType.float32]    # (4*H, H)
    var bias_ih:   DeviceBuffer[DType.float32]    # (4*H,)
    var bias_hh:   DeviceBuffer[DType.float32]    # (4*H,)
    var input_size:  Int
    var hidden_size: Int


def lstm_layer_forward(
    mut ctx: DeviceContext,
    mut layer: LSTMLayer,
    mut x_buf: DeviceBuffer[DType.float32],         # (B, T, IN)
    mut h_seq_out_buf: DeviceBuffer[DType.float32], # (B, T, H) — hidden output sequence
    mut h_state_buf: DeviceBuffer[DType.float32],   # (B, H) — persistent h_prev (zero-init by caller)
    mut c_state_buf: DeviceBuffer[DType.float32],   # (B, H) — persistent c_prev (zero-init by caller)
    mut pre_xw_buf:  DeviceBuffer[DType.float32],   # (B, 4*H) — scratch for x@W_ih.T at one timestep
    mut pre_hw_buf:  DeviceBuffer[DType.float32],   # (B, 4*H) — scratch for h@W_hh.T at one timestep
    batch: Int, time: Int,
) raises:
    """Run one LSTM layer over the full sequence using MAX ops only.

    The caller pre-allocates scratch buffers for the per-timestep gate projections
    so we don't allocate inside the per-step loop.

    Each timestep is two `linalg.matmul` calls plus one `elementwise` GPU op
    that fuses (bias-add + 4 activations + cell update + hidden write).
    """
    var IN = layer.input_size
    var H = layer.hidden_size
    var H4 = 4 * H
    var dctx = DeviceContextPtr(ctx)

    var x_t_full = TileTensor(x_buf, row_major(Idx(batch), Idx(time), Idx(IN)))
    var h_seq_t = TileTensor(h_seq_out_buf, row_major(Idx(batch), Idx(time), Idx(H)))
    var w_ih_t = TileTensor(layer.weight_ih, row_major(Idx(H4), Idx(IN)))
    var w_hh_t = TileTensor(layer.weight_hh, row_major(Idx(H4), Idx(H)))

    var x_ptr = x_buf.unsafe_ptr()
    var h_seq_ptr = h_seq_out_buf.unsafe_ptr()
    var h_state_ptr = h_state_buf.unsafe_ptr()
    var c_state_ptr = c_state_buf.unsafe_ptr()
    var pre_xw_ptr = pre_xw_buf.unsafe_ptr()
    var pre_hw_ptr = pre_hw_buf.unsafe_ptr()
    var b_ih_ptr = layer.bias_ih.unsafe_ptr()
    var b_hh_ptr = layer.bias_hh.unsafe_ptr()

    # Wrap state and scratch with 2-D layouts for matmul.
    var h_state_2d = TileTensor(h_state_buf, row_major(Idx(batch), Idx(H)))
    var pre_xw_2d  = TileTensor(pre_xw_buf, row_major(Idx(batch), Idx(H4)))
    var pre_hw_2d  = TileTensor(pre_hw_buf, row_major(Idx(batch), Idx(H4)))

    for t in range(time):
        # 1) pre_xw = x[:, t, :] @ W_ih.T  → (B, 4H)
        #    Build a (B, IN) view of x at timestep t. We use a 2D TileTensor
        #    over the slice; since x is row-major (B, T, IN), the slice at
        #    timestep t is non-contiguous along B but contiguous along IN.
        #    Easiest: build per-timestep slice via runtime layout offsets.
        var t_const = t
        var x_step_ptr = x_ptr + t * IN   # start of x[0, t, 0]; stride between B rows = T*IN

        # The simplest approach: per-step gather of x[:, t, :] into a contiguous
        # (B, IN) buffer with a small elementwise kernel, then matmul. For B=1
        # this is just a noncontig load — we skip the copy and use a custom
        # elementwise that bakes the (B,T,IN) → (B, 4H) matmul. But cleaner is
        # to do an explicit gather via elementwise so the matmul sees standard
        # contiguous inputs.
        #
        # For now we materialise the timestep slice into a scratch buffer.
        # (This is still done via `elementwise`, no custom GPU kernel.)
        # NOTE: For B=1 this is a no-op pointer offset; we exploit that.
        comptime BATCH_IS_1 = True   # specialized fast path for B==1
        if batch == 1:
            # x_step is the contiguous range x_ptr[t*IN .. (t+1)*IN). Create a
            # (1, IN) TileTensor view over the existing buffer.
            var x_step_t = TileTensor(
                x_step_ptr, row_major(Idx(1), Idx(IN))
            )
            nn_matmul[target="gpu", transpose_b=True](pre_xw_2d, x_step_t, w_ih_t, dctx)
        else:
            # General path: gather x[:, t, :] into a contig (B, IN) scratch via
            # elementwise. (Implementation deferred — VoiceEncoder runs B=1.)
            raise Error("LSTM B>1 path needs an explicit gather; B=1 only for now")

        # 2) pre_hw = h_prev @ W_hh.T  → (B, 4H)
        nn_matmul[target="gpu", transpose_b=True](pre_hw_2d, h_state_2d, w_hh_t, dctx)

        # 3) Fuse: pre_gates = pre_xw + b_ih + pre_hw + b_hh; apply gates;
        #    update c_state, h_state, h_seq via a single elementwise pass
        #    over H output elements per batch row.
        @always_inline
        @parameter
        @__copy_capture(
            pre_xw_ptr, pre_hw_ptr, b_ih_ptr, b_hh_ptr,
            c_state_ptr, h_state_ptr, h_seq_ptr,
            H, time, t_const,
        )
        def gate_func[width: Int, rank: Int, alignment: Int = 1](idx: IndexList[rank]):
            var pos = idx[0]
            var bi = pos // H
            var hi = pos - bi * H
            # Compute the four pre-gates at position (bi, hi). Width=1 (we don't
            # vectorise here because gates are split into 4 strided regions).
            var pre_i = pre_xw_ptr[bi * 4 * H + 0 * H + hi] + b_ih_ptr[0 * H + hi] \
                      + pre_hw_ptr[bi * 4 * H + 0 * H + hi] + b_hh_ptr[0 * H + hi]
            var pre_f = pre_xw_ptr[bi * 4 * H + 1 * H + hi] + b_ih_ptr[1 * H + hi] \
                      + pre_hw_ptr[bi * 4 * H + 1 * H + hi] + b_hh_ptr[1 * H + hi]
            var pre_g = pre_xw_ptr[bi * 4 * H + 2 * H + hi] + b_ih_ptr[2 * H + hi] \
                      + pre_hw_ptr[bi * 4 * H + 2 * H + hi] + b_hh_ptr[2 * H + hi]
            var pre_o = pre_xw_ptr[bi * 4 * H + 3 * H + hi] + b_ih_ptr[3 * H + hi] \
                      + pre_hw_ptr[bi * 4 * H + 3 * H + hi] + b_hh_ptr[3 * H + hi]
            var i_g = Float32(1.0) / (Float32(1.0) + exp(-pre_i))
            var f_g = Float32(1.0) / (Float32(1.0) + exp(-pre_f))
            var g_g = tanh(pre_g)
            var o_g = Float32(1.0) / (Float32(1.0) + exp(-pre_o))
            var c_p = c_state_ptr[bi * H + hi]
            var c_n = f_g * c_p + i_g * g_g
            var h_n = o_g * tanh(c_n)
            c_state_ptr[bi * H + hi] = c_n
            h_state_ptr[bi * H + hi] = h_n
            h_seq_ptr[bi * time * H + t_const * H + hi] = h_n
        elementwise[gate_func, simd_width=1, target="gpu"](
            IndexList[1](batch * H), dctx,
        )
