"""
CausalConditionalCFM Euler ODE solver parity test.

Uses upstream's pre-computed per-step estimator outputs (`cfm_step_NN_output.bin`)
as inputs. This isolates the Euler solver math from the estimator network so
we can verify the loop is correct before porting the ConditionalDecoder.

Math per step (from flow_matching.py:138-141):
    dxdt, cfg_dxdt = split(out, B, dim=0)
    dxdt = (1 + cfg_rate) * dxdt - cfg_rate * cfg_dxdt
    dt   = t_span[i+1] - t_span[i]
    x    = x + dt * dxdt

Compares to cfm_mel_out.
"""

from std.math import ceildiv
from std.sys import has_accelerator
from std.testing import TestSuite, assert_almost_equal, assert_equal
from std.gpu import block_idx, thread_idx
from std.gpu.host import DeviceContext, DeviceBuffer
from layout import TileTensor, TensorLayout, row_major

from fixture import Tensor, load_fp32


comptime BATCH = 1
comptime MEL_C = 80
comptime MEL_T = 752
comptime N_STEPS = 10
comptime CFG_RATE: Float32 = 0.7
comptime POINTWISE_BLOCK = 256


def upload(buf: DeviceBuffer[DType.float32], data: List[Float32], n: Int) raises:
    with buf.map_to_host() as h:
        for i in range(n):
            h[i] = data[i]


# CFG combine + scaled-add into x: x_new = x + dt * ((1+r)*dxdt - r*cfg_dxdt).
# estimator_out is (2*B, C, T) — first half = conditional, second = unconditional.
def cfm_euler_step_kernel[
    dtype: DType,
    XLayout: TensorLayout,
    EstLayout: TensorLayout,
    OutLayout: TensorLayout,
    B: Int, C: Int, T: Int,
    BLOCK: Int,
](
    output: TileTensor[dtype, OutLayout, MutAnyOrigin],     # (B, C, T)
    x: TileTensor[dtype, XLayout, MutAnyOrigin],             # (B, C, T)
    estimator_out: TileTensor[dtype, EstLayout, MutAnyOrigin],  # (2*B, C, T)
    dt: Float32,
    cfg_rate: Float32,
):
    """Combined CFG + Euler step."""
    comptime assert x.flat_rank == 3
    comptime assert estimator_out.flat_rank == 3
    comptime assert output.flat_rank == 3

    var idx = block_idx.x * BLOCK + thread_idx.x
    var n = B * C * T
    if idx >= n:
        return

    var t = idx % T
    var c = (idx // T) % C
    var b = idx // (T * C)

    var xv = rebind[Scalar[dtype]](x[b, c, t]).cast[DType.float32]()
    var d_cond = rebind[Scalar[dtype]](estimator_out[b, c, t]).cast[DType.float32]()
    var d_uncond = rebind[Scalar[dtype]](estimator_out[b + B, c, t]).cast[DType.float32]()
    var dxdt = (1.0 + cfg_rate) * d_cond - cfg_rate * d_uncond
    var y = xv + dt * dxdt
    output[b, c, t] = rebind[output.ElementType](y.cast[dtype]())


def test_cfm_euler_fp32() raises:
    comptime assert has_accelerator(), "Requires GPU"

    var fix = "tests/fixtures/s3gen/"
    var z_init = load_fp32(fix + "cfm_z_init.bin")           # (1, 80, 752)
    var t_span = load_fp32(fix + "cfm_t_span.bin")           # (11,)
    var exp = load_fp32(fix + "cfm_mel_out.bin")             # (1, 80, 752)

    assert_equal(z_init.shape[0], BATCH)
    assert_equal(z_init.shape[1], MEL_C)
    assert_equal(z_init.shape[2], MEL_T)
    assert_equal(t_span.shape[0], N_STEPS + 1)

    var n_x = BATCH * MEL_C * MEL_T
    var n_est = 2 * BATCH * MEL_C * MEL_T

    var ctx = DeviceContext()
    var x_buf = ctx.enqueue_create_buffer[DType.float32](n_x)
    var x_next_buf = ctx.enqueue_create_buffer[DType.float32](n_x)
    var est_buf = ctx.enqueue_create_buffer[DType.float32](n_est)

    upload(x_buf, z_init.data, n_x)

    comptime x_layout = row_major[BATCH, MEL_C, MEL_T]()
    comptime est_layout = row_major[2 * BATCH, MEL_C, MEL_T]()

    var x_t = TileTensor(x_buf, x_layout)
    var x_next_t = TileTensor(x_next_buf, x_layout)
    var est_t = TileTensor(est_buf, est_layout)

    comptime step_k = cfm_euler_step_kernel[
        DType.float32, type_of(x_layout), type_of(est_layout), type_of(x_layout),
        BATCH, MEL_C, MEL_T, POINTWISE_BLOCK,
    ]

    for i in range(N_STEPS):
        # Load this step's estimator output.
        var fname = "cfm_step_" + ("0" if i < 10 else "") + String(i) + "_output.bin"
        var est = load_fp32(fix + fname)
        upload(est_buf, est.data, n_est)

        var t_cur = t_span.data[i]
        var t_next = t_span.data[i + 1]
        var dt = Float32(t_next - t_cur)

        ctx.enqueue_function[step_k, step_k](
            x_next_t, x_t, est_t, dt, CFG_RATE,
            grid_dim=ceildiv(n_x, POINTWISE_BLOCK), block_dim=POINTWISE_BLOCK,
        )
        ctx.synchronize()
        # Copy x_next → x for next iter.
        with x_next_buf.map_to_host() as src:
            with x_buf.map_to_host() as dst:
                for k in range(n_x):
                    dst[k] = src[k]

    var max_abs: Float32 = 0.0
    var sum_abs: Float64 = 0.0
    with x_buf.map_to_host() as h:
        for i in range(n_x):
            var d = h[i] - exp.data[i]
            if d < 0.0: d = -d
            if d > max_abs: max_abs = d
            sum_abs += Float64(d)
            assert_almost_equal(h[i], exp.data[i], atol=1.0e-5)
    print("CFM Euler solver fp32 (10 steps, real S3Gen) — max abs:", max_abs,
          " mean abs:", sum_abs / Float64(n_x))


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
