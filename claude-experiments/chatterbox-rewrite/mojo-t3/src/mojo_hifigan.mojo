"""
Python extension module: mojo_hifigan.

Drop-in (eventually) for s3gen.mel2wav.inference in paper-audiobooks, sidestepping
the MIOpen Winograd corruption that bites upstream HiFiGAN on gfx1151.

Current state: real conv_pre + leaky_relu + ups[0] of HiFiGAN exposed as a
Python-callable function for end-to-end I/O validation. Caller passes a numpy
mel of fixed shape (1, 80, 32); we return the (1, 256, 256) output after the
first upsample stage. This proves:
  - numpy → Mojo TileTensor conversion works
  - real Chatterbox weights loaded from disk reach the GPU correctly
  - the result returned to Python matches upstream bit-tolerantly

The rest of the HiFiGAN pipeline (stages 1/2 + conv_post + iSTFT) is mechanical
to add once this skeleton is plumbed in.
"""

from std.math import ceildiv
from std.os import abort
from std.python import Python, PythonObject
from std.python.bindings import PythonModuleBuilder
from std.sys import has_accelerator
from std.gpu.host import DeviceContext, DeviceBuffer
from layout import TileTensor, row_major

from fixture import load_fp32
from conv import conv1d_kernel, transposed_conv1d_kernel, leaky_relu_kernel


# Hard-coded shapes for the current first-stage demonstration.
# T_MEL=32 matches our synthetic test fixture; real production uses ~252.
comptime BATCH = 1
comptime MEL_C = 80
comptime MEL_T = 32
comptime PRE_C = 512
comptime CP_K = 7
comptime S0_C = 256
comptime S0_T = 256
comptime UP_K = 16
comptime UP_STRIDE = 8
comptime UP_PAD = 4
comptime POINTWISE_BLOCK = 256


@export
def PyInit_mojo_hifigan() -> PythonObject:
    try:
        var m = PythonModuleBuilder("mojo_hifigan")
        m.def_function[conv_pre_step]("conv_pre_step")
        m.def_function[ups0_step]("ups0_step")
        return m.finalize()
    except e:
        abort(String("failed to create mojo_hifigan module: ", e))


def _upload_np_to_buf(buf: DeviceBuffer[DType.float32], arr: PythonObject, n: Int) raises:
    """Copy a numpy (or python sequence) of floats into a DeviceBuffer.

    arr is expected to be a contiguous numpy float32 array; we use .flatten()
    to get a 1-D view, then iterate the elements as PythonObjects.
    """
    var flat = arr.flatten()
    with buf.map_to_host() as h:
        for i in range(n):
            h[i] = Float32(py=flat[i])


def _buf_to_np(buf: DeviceBuffer[DType.float32], n: Int, shape: PythonObject) raises -> PythonObject:
    """Return a freshly-allocated numpy float32 array of the given shape with
    the buffer contents copied in."""
    var numpy = Python.import_module("numpy")
    var py_arr = numpy.empty(shape, dtype=numpy.float32)
    var flat = py_arr.flatten()  # NOTE: flatten makes a copy; we write back below
    var data = numpy.empty(Python.tuple(n), dtype=numpy.float32)
    with buf.map_to_host() as h:
        for i in range(n):
            data[i] = h[i]
    return data.reshape(shape)


def conv_pre_step(
    mel: PythonObject,
    weights_dir: PythonObject,
) raises -> PythonObject:
    """Run conv_pre(mel) where mel is a numpy (1, 80, 32) float32 array.

    weights_dir is the directory holding weights/conv_pre__weight.bin etc.
    Returns numpy (1, 512, 32) float32.
    """
    comptime assert has_accelerator(), "Requires GPU"

    var w_dir = String(py=weights_dir)
    var w = load_fp32(w_dir + "/conv_pre__weight.bin")
    var b = load_fp32(w_dir + "/conv_pre__bias.bin")

    var n_mel = BATCH * MEL_C * MEL_T
    var n_w = PRE_C * MEL_C * CP_K
    var n_out = BATCH * PRE_C * MEL_T

    var ctx = DeviceContext()
    var mel_buf = ctx.enqueue_create_buffer[DType.float32](n_mel)
    var w_buf = ctx.enqueue_create_buffer[DType.float32](n_w)
    var b_buf = ctx.enqueue_create_buffer[DType.float32](PRE_C)
    var out_buf = ctx.enqueue_create_buffer[DType.float32](n_out)

    _upload_np_to_buf(mel_buf, mel, n_mel)
    with w_buf.map_to_host() as h:
        for i in range(n_w): h[i] = w.data[i]
    with b_buf.map_to_host() as h:
        for i in range(PRE_C): h[i] = b.data[i]

    comptime mel_layout = row_major[BATCH, MEL_C, MEL_T]()
    comptime w_layout = row_major[PRE_C, MEL_C, CP_K]()
    comptime bias_layout = row_major[PRE_C]()
    comptime out_layout = row_major[BATCH, PRE_C, MEL_T]()

    var mel_t = TileTensor(mel_buf, mel_layout)
    var w_t = TileTensor(w_buf, w_layout)
    var b_t = TileTensor(b_buf, bias_layout)
    var out_t = TileTensor(out_buf, out_layout)

    comptime kernel = conv1d_kernel[
        DType.float32, type_of(mel_layout), type_of(w_layout),
        type_of(bias_layout), type_of(out_layout), CP_K, True,
    ]
    ctx.enqueue_function[kernel, kernel](
        out_t, mel_t, w_t, b_t,
        BATCH, MEL_C, PRE_C, MEL_T, MEL_T, 1, 3, 1,
        grid_dim=BATCH * PRE_C * MEL_T, block_dim=1,
    )
    ctx.synchronize()

    return _buf_to_np(out_buf, n_out, Python.tuple(BATCH, PRE_C, MEL_T))


def ups0_step(
    x_lrelu: PythonObject,
    weights_dir: PythonObject,
) raises -> PythonObject:
    """Run the first upsample's transposed conv (ups[0]) on (1, 512, 32) ->
    (1, 256, 256). Caller has already applied leaky_relu.
    """
    comptime assert has_accelerator(), "Requires GPU"

    var w_dir = String(py=weights_dir)
    var w = load_fp32(w_dir + "/ups__0__weight.bin")
    var b = load_fp32(w_dir + "/ups__0__bias.bin")

    var n_in = BATCH * PRE_C * MEL_T
    var n_w = PRE_C * S0_C * UP_K
    var n_out = BATCH * S0_C * S0_T

    var ctx = DeviceContext()
    var in_buf = ctx.enqueue_create_buffer[DType.float32](n_in)
    var w_buf = ctx.enqueue_create_buffer[DType.float32](n_w)
    var b_buf = ctx.enqueue_create_buffer[DType.float32](S0_C)
    var out_buf = ctx.enqueue_create_buffer[DType.float32](n_out)

    _upload_np_to_buf(in_buf, x_lrelu, n_in)
    with w_buf.map_to_host() as h:
        for i in range(n_w): h[i] = w.data[i]
    with b_buf.map_to_host() as h:
        for i in range(S0_C): h[i] = b.data[i]

    comptime in_layout = row_major[BATCH, PRE_C, MEL_T]()
    comptime w_layout = row_major[PRE_C, S0_C, UP_K]()
    comptime bias_layout = row_major[S0_C]()
    comptime out_layout = row_major[BATCH, S0_C, S0_T]()

    var in_t = TileTensor(in_buf, in_layout)
    var w_t = TileTensor(w_buf, w_layout)
    var b_t = TileTensor(b_buf, bias_layout)
    var out_t = TileTensor(out_buf, out_layout)

    comptime kernel = transposed_conv1d_kernel[
        DType.float32, type_of(in_layout), type_of(w_layout),
        type_of(bias_layout), type_of(out_layout), UP_K, True,
    ]
    ctx.enqueue_function[kernel, kernel](
        out_t, in_t, w_t, b_t,
        BATCH, PRE_C, S0_C, MEL_T, S0_T, UP_STRIDE, UP_PAD, 1,
        grid_dim=BATCH * S0_C * S0_T, block_dim=1,
    )
    ctx.synchronize()

    return _buf_to_np(out_buf, n_out, Python.tuple(BATCH, S0_C, S0_T))
