"""High-level Module wrappers built strictly on MAX `nn.*`/`linalg.*` ops
plus `std.algorithm.functional.elementwise` for pointwise math.

No hand-rolled GPU kernels. Where MAX lacks a primitive (SiLU, GELU,
bias-add, residual-add, etc.), we express it as a single capturing closure
invoked through `elementwise[..., target="gpu"]` — exactly the same
dispatch pattern MAX's own `_interpreter_ops/*.mojo` uses.

Convention: all tensors are float32 unless noted. All DeviceBuffers must
already be GPU-allocated.
"""
from std.sys.info import simd_width_of, has_accelerator
from std.gpu.host import DeviceContext, DeviceBuffer
from std.runtime.asyncrt import DeviceContextPtr
from std.algorithm.functional import elementwise, IndexList
from std.math import exp, erf, sqrt
from layout import Idx, TileTensor, row_major

from nn.normalization import layer_norm as nn_layer_norm, rms_norm as nn_rms_norm
from nn.softmax import softmax as nn_softmax
from nn.activations import relu as nn_relu, leaky_relu as nn_leaky_relu
# Note: gelu and elu exist in MAX source but are not exposed in our nightly's
# `nn.activations` module. Until the build catches up we inline the math
# (still ASIC-friendly elementwise; just not the MAX SIMD helper).
from nn.gather_scatter import gather as nn_gather
from linalg.matmul import matmul as nn_matmul


# ============================================================================
# Linear: y = x @ W.T + b
# ============================================================================

@fieldwise_init
struct Linear(Copyable, Movable):
    var weight: DeviceBuffer[DType.float32]    # (OUT, IN)
    var bias:   DeviceBuffer[DType.float32]    # (OUT,) — may be a zero-len dummy
    var in_features: Int
    var out_features: Int
    var has_bias: Bool


def linear_forward(
    mut ctx: DeviceContext,
    mut module: Linear,
    mut x_buf: DeviceBuffer[DType.float32],
    mut out_buf: DeviceBuffer[DType.float32],
    m: Int,
) raises:
    """Computes out = x @ W.T (+ bias) via `linalg.matmul`.

    Bias-add is dispatched as a fused `elementwise` op rather than a custom kernel.
    """
    var dctx = DeviceContextPtr(ctx)
    var x_t = TileTensor(x_buf, row_major(Idx(m), Idx(module.in_features)))
    var w_t = TileTensor(module.weight,
        row_major(Idx(module.out_features), Idx(module.in_features)))
    var out_t = TileTensor(out_buf, row_major(Idx(m), Idx(module.out_features)))
    nn_matmul[target="gpu", transpose_b=True](out_t, x_t, w_t, dctx)

    if module.has_bias:
        var out_ptr = out_buf.unsafe_ptr()
        var bias_ptr = module.bias.unsafe_ptr()
        var n = module.out_features
        var total = m * n

        @always_inline
        @parameter
        @__copy_capture(out_ptr, bias_ptr, n)
        def bias_func[width: Int, rank: Int, alignment: Int = 1](idx: IndexList[rank]):
            var i = idx[0]
            var col = i % n
            var cur = out_ptr[i]
            out_ptr[i] = cur + bias_ptr[col]
        elementwise[bias_func, simd_width=1, target="gpu"](
            IndexList[1](total), DeviceContextPtr(ctx),
        )


# ============================================================================
# LayerNorm — wraps nn.normalization.layer_norm
# ============================================================================

@fieldwise_init
struct LayerNorm(Copyable, Movable):
    var gamma: DeviceBuffer[DType.float32]
    var beta:  DeviceBuffer[DType.float32]
    var feature_dim: Int
    var eps: Float32


def layer_norm_forward(
    mut ctx: DeviceContext,
    mut module: LayerNorm,
    mut x_buf: DeviceBuffer[DType.float32],
    mut out_buf: DeviceBuffer[DType.float32],
    batch_dim: Int,
) raises:
    """LayerNorm over the last (feature) dim. Treats input as (batch_dim, feature_dim)."""
    var in_ptr = x_buf.unsafe_ptr()
    var out_ptr = out_buf.unsafe_ptr()
    var gamma_ptr = module.gamma.unsafe_ptr()
    var beta_ptr = module.beta.unsafe_ptr()
    var feature_dim = module.feature_dim
    var eps = module.eps

    var shape = IndexList[2](batch_dim, feature_dim)
    var gamma_shape = IndexList[1](feature_dim)

    @always_inline
    @parameter
    @__copy_capture(in_ptr, feature_dim)
    def input_fn[width: Int, rank: Int, alignment: Int](coords: IndexList[rank]) -> SIMD[DType.float32, width]:
        var c = rebind[IndexList[2]](coords)
        var idx = c[0] * feature_dim + c[1]
        return in_ptr.load[width=width, alignment=alignment](idx)

    @always_inline
    @parameter
    @__copy_capture(gamma_ptr)
    def gamma_fn[width: Int, rank: Int, alignment: Int](coords: IndexList[rank]) -> SIMD[DType.float32, width]:
        var c = rebind[IndexList[1]](coords)
        return gamma_ptr.load[width=width, alignment=alignment](c[0])

    @always_inline
    @parameter
    @__copy_capture(out_ptr, feature_dim)
    def output_fn[width: Int, rank: Int, alignment: Int](coords: IndexList[rank], val: SIMD[DType.float32, width]):
        var c = rebind[IndexList[2]](coords)
        var idx = c[0] * feature_dim + c[1]
        out_ptr.store[width=width, alignment=alignment](idx, val)

    var beta_tt = TileTensor(beta_ptr, row_major(Idx(feature_dim)))
    var dctx = DeviceContextPtr(ctx)

    nn_layer_norm[
        DType.float32, 2,
        input_fn, gamma_fn, output_fn,
        target="gpu",
    ](shape, gamma_shape, beta_tt, eps, dctx)


# ============================================================================
# RMSNorm — wraps nn.normalization.rms_norm
# ============================================================================

@fieldwise_init
struct RMSNorm(Copyable, Movable):
    var gamma: DeviceBuffer[DType.float32]
    var feature_dim: Int
    var eps: Float32


def rms_norm_forward(
    mut ctx: DeviceContext,
    mut module: RMSNorm,
    mut x_buf: DeviceBuffer[DType.float32],
    mut out_buf: DeviceBuffer[DType.float32],
    batch_dim: Int,
) raises:
    """RMSNorm over the last (feature) dim."""
    var in_ptr = x_buf.unsafe_ptr()
    var out_ptr = out_buf.unsafe_ptr()
    var gamma_ptr = module.gamma.unsafe_ptr()
    var feature_dim = module.feature_dim
    var eps = module.eps
    var shape = IndexList[2](batch_dim, feature_dim)

    @always_inline
    @parameter
    @__copy_capture(in_ptr, feature_dim)
    def input_fn[width: Int, rank: Int](coords: IndexList[rank]) -> SIMD[DType.float32, width]:
        var c = rebind[IndexList[2]](coords)
        var idx = c[0] * feature_dim + c[1]
        return in_ptr.load[width=width](idx)

    @always_inline
    @parameter
    @__copy_capture(out_ptr, feature_dim)
    def output_fn[width: Int, rank: Int, alignment: Int](coords: IndexList[rank], val: SIMD[DType.float32, width]):
        var c = rebind[IndexList[2]](coords)
        var idx = c[0] * feature_dim + c[1]
        out_ptr.store[width=width, alignment=alignment](idx, val)

    var gamma_tt = TileTensor(gamma_ptr, row_major(Idx(feature_dim)))
    var dctx = DeviceContextPtr(ctx)

    nn_rms_norm[
        DType.float32, 2,
        input_fn, output_fn,
        target="gpu",
    ](shape, gamma_tt, eps, Float32(0.0), dctx)


# ============================================================================
# Softmax — wraps nn.softmax.softmax
# ============================================================================

def softmax_lastdim(
    mut ctx: DeviceContext,
    mut x_buf: DeviceBuffer[DType.float32],
    mut out_buf: DeviceBuffer[DType.float32],
    batch_dim: Int,
    axis_dim: Int,
) raises:
    """Softmax over last axis. Treats input as (batch_dim, axis_dim)."""
    var in_ptr = x_buf.unsafe_ptr()
    var out_ptr = out_buf.unsafe_ptr()
    var shape = IndexList[2](batch_dim, axis_dim)

    @always_inline
    @parameter
    @__copy_capture(in_ptr, axis_dim)
    def input_fn[width: Int, rank: Int](coords: IndexList[rank]) -> SIMD[DType.float32, width]:
        var c = rebind[IndexList[2]](coords)
        var idx = c[0] * axis_dim + c[1]
        return in_ptr.load[width=width](idx)

    var out_t = TileTensor(out_ptr, row_major(Idx(batch_dim), Idx(axis_dim)))
    var dctx = DeviceContextPtr(ctx)

    nn_softmax[
        DType.float32,
        simd_width_of[DType.float32](),
        2,
        input_fn,
        target="gpu",
    ](shape, out_t, 1, dctx)


# ============================================================================
# Activations dispatched via `elementwise[..., target="gpu"]`
# ============================================================================

def relu(
    mut ctx: DeviceContext,
    mut in_buf: DeviceBuffer[DType.float32],
    mut out_buf: DeviceBuffer[DType.float32],
    n: Int,
) raises:
    """y = max(0, x). Uses `nn.activations.relu` per-element."""
    var in_ptr = in_buf.unsafe_ptr()
    var out_ptr = out_buf.unsafe_ptr()

    @always_inline
    @parameter
    @__copy_capture(in_ptr, out_ptr)
    def func[width: Int, rank: Int, alignment: Int = 1](idx: IndexList[rank]):
        var i = idx[0]
        var x = in_ptr.load[width=width, alignment=alignment](i)
        out_ptr.store[width=width, alignment=alignment](i, nn_relu(x))
    elementwise[func, simd_width=4, target="gpu"](
        IndexList[1](n), DeviceContextPtr(ctx),
    )


def silu(
    mut ctx: DeviceContext,
    mut in_buf: DeviceBuffer[DType.float32],
    mut out_buf: DeviceBuffer[DType.float32],
    n: Int,
) raises:
    """y = x * sigmoid(x). Pure-elementwise — implemented via `elementwise`
    since MAX doesn't expose SiLU as a `nn` primitive."""
    var in_ptr = in_buf.unsafe_ptr()
    var out_ptr = out_buf.unsafe_ptr()

    @always_inline
    @parameter
    @__copy_capture(in_ptr, out_ptr)
    def func[width: Int, rank: Int, alignment: Int = 1](idx: IndexList[rank]):
        var i = idx[0]
        var x = in_ptr.load[width=width, alignment=alignment](i)
        var sig = SIMD[DType.float32, width](1.0) / (
            SIMD[DType.float32, width](1.0) + exp(-x)
        )
        out_ptr.store[width=width, alignment=alignment](i, x * sig)
    elementwise[func, simd_width=4, target="gpu"](
        IndexList[1](n), DeviceContextPtr(ctx),
    )


def gelu(
    mut ctx: DeviceContext,
    mut in_buf: DeviceBuffer[DType.float32],
    mut out_buf: DeviceBuffer[DType.float32],
    n: Int,
) raises:
    """Exact GELU: y = 0.5 * x * (1 + erf(x / sqrt(2))).

    MAX's `nn.activations.gelu` is the same math but not exposed in our
    bundled nightly mojopkg yet.
    """
    var in_ptr = in_buf.unsafe_ptr()
    var out_ptr = out_buf.unsafe_ptr()

    @always_inline
    @parameter
    @__copy_capture(in_ptr, out_ptr)
    def func[width: Int, rank: Int, alignment: Int = 1](idx: IndexList[rank]):
        var i = idx[0]
        var x = in_ptr.load[width=width, alignment=alignment](i)
        alias INV_SQRT2 = SIMD[DType.float32, width](0.7071067811865476)
        var half = SIMD[DType.float32, width](0.5)
        var one  = SIMD[DType.float32, width](1.0)
        out_ptr.store[width=width, alignment=alignment](
            i, half * x * (one + erf(x * INV_SQRT2))
        )
    elementwise[func, simd_width=4, target="gpu"](
        IndexList[1](n), DeviceContextPtr(ctx),
    )


# ============================================================================
# Embedding — wraps nn.gather_scatter.gather with axis=0
# ============================================================================

@fieldwise_init
struct Embedding(Copyable, Movable):
    var table: DeviceBuffer[DType.float32]    # (V, D)
    var vocab_size: Int
    var dim: Int


def embedding_forward(
    mut ctx: DeviceContext,
    mut module: Embedding,
    mut ids_buf: DeviceBuffer[DType.int64],
    mut out_buf: DeviceBuffer[DType.float32],
    b: Int, s: Int,
) raises:
    """Out[bi, si, d] = table[ids[bi, si], d] via `nn.gather_scatter.gather`
    along axis=0 of the table."""
    var ids_t = TileTensor(ids_buf, row_major(Idx(b), Idx(s)))
    var table_t = TileTensor(module.table, row_major(Idx(module.vocab_size), Idx(module.dim)))
    var out_t = TileTensor(out_buf, row_major(Idx(b), Idx(s), Idx(module.dim)))
    var dctx = DeviceContextPtr(ctx)
    nn_gather[axis=0, target="gpu"](out_t, table_t, ids_t, context=dctx)


# ============================================================================
# Residual add (out += other) — dispatched via `elementwise`
# ============================================================================

def residual_add(
    mut ctx: DeviceContext,
    mut out_buf: DeviceBuffer[DType.float32],
    mut other_buf: DeviceBuffer[DType.float32],
    n: Int,
) raises:
    """Elementwise out[i] += other[i]."""
    var out_ptr = out_buf.unsafe_ptr()
    var other_ptr = other_buf.unsafe_ptr()

    @always_inline
    @parameter
    @__copy_capture(out_ptr, other_ptr)
    def func[width: Int, rank: Int, alignment: Int = 1](idx: IndexList[rank]):
        var i = idx[0]
        var a = out_ptr.load[width=width, alignment=alignment](i)
        var b = other_ptr.load[width=width, alignment=alignment](i)
        out_ptr.store[width=width, alignment=alignment](i, a + b)
    elementwise[func, simd_width=4, target="gpu"](
        IndexList[1](n), DeviceContextPtr(ctx),
    )
