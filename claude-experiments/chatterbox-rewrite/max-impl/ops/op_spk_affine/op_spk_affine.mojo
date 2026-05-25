"""op_spk_affine: L2-normalize 192-d speaker embedding, then project to 80-d via Linear.

Pipeline: emb_192 -> l2_normalize -> spk_embed_affine_layer(192, 80) -> spks.
"""
from std.os import abort
from std.python import PythonObject
from std.python.bindings import PythonModuleBuilder
from std.memory import OpaquePointer
from std.gpu.host import DeviceContext, DeviceBuffer
from std.runtime.asyncrt import DeviceContextPtr
from std.math import sqrt

from modules import Linear, linear_forward
from fixture import load_fp32


@fieldwise_init
struct OpState(Movable):
    var ctx: DeviceContext
    var spk_affine: Linear


def _upload(buf: DeviceBuffer[DType.float32], data: List[Float32], n: Int) raises:
    with buf.map_to_host() as h:
        for i in range(n):
            h[i] = data[i]


def _upload_fp32(mut ctx: DeviceContext, path: String) raises -> DeviceBuffer[DType.float32]:
    var t = load_fp32(path)
    var n = len(t.data)
    var buf = ctx.enqueue_create_buffer[DType.float32](n)
    _upload(buf, t.data, n)
    return buf^


def _ctx_from_python(device_context_ptr: PythonObject) raises -> DeviceContext:
    var addr = Int(py=device_context_ptr)
    if addr == 0:
        raise Error("op_spk_affine requires a GPU device context")
    var opaque = OpaquePointer[MutExternalOrigin](unsafe_from_address=addr)
    return DeviceContextPtr(opaque).get_device_context()


def init_op(
    weights_base_path: PythonObject, device_context_ptr: PythonObject
) raises -> PythonObject:
    var base = String(py=weights_base_path)
    var ctx = _ctx_from_python(device_context_ptr)
    # weights/s3gen/flow/spk_embed_affine_layer/{weight,bias}.bin
    var w = _upload_fp32(ctx, base + "/spk_embed_affine_layer/weight.bin")
    var b = _upload_fp32(ctx, base + "/spk_embed_affine_layer/bias.bin")
    var spk_affine = Linear(w^, b^, 192, 80, True)
    ctx.synchronize()

    var ptr = alloc[OpState](1)
    ptr.init_pointee_move(OpState(ctx^, spk_affine^))
    return PythonObject(Int(ptr))


def forward(
    handle: PythonObject,
    emb_buffer: PythonObject,   # (B, 192) GPU
    spks_buffer: PythonObject,  # (B, 80) GPU
    b_obj: PythonObject,
) raises -> PythonObject:
    var addr = Int(py=handle)
    if addr == 0:
        raise Error("op_spk_affine: null handle")
    var state_ptr = UnsafePointer[OpState, MutExternalOrigin](unsafe_from_address=addr)
    var b = Int(py=b_obj)

    var emb_addr = Int(py=emb_buffer._data_ptr())
    var spks_addr = Int(py=spks_buffer._data_ptr())
    var emb_n = Int(py=emb_buffer.num_elements)
    var spks_n = Int(py=spks_buffer.num_elements)

    var emb_ptr = UnsafePointer[Float32, MutExternalOrigin](unsafe_from_address=emb_addr)
    var spks_ptr = UnsafePointer[Float32, MutExternalOrigin](unsafe_from_address=spks_addr)

    # L2-normalize emb (B, 192) in place on host-mapped scratch — we'll do
    # this by pulling to host, normalizing, pushing back. Tiny size (B*192).
    var emb_buf = DeviceBuffer[DType.float32](state_ptr[].ctx, emb_ptr, emb_n, owning=False)
    with emb_buf.map_to_host() as h:
        for bi in range(b):
            var s: Float32 = 0.0
            for i in range(192):
                var v = h[bi * 192 + i]
                s += v * v
            var inv = 1.0 / sqrt(s + Float32(1.0e-12))
            for i in range(192):
                h[bi * 192 + i] = h[bi * 192 + i] * inv

    # Now run the Linear.
    var spks_buf = DeviceBuffer[DType.float32](state_ptr[].ctx, spks_ptr, spks_n, owning=False)
    linear_forward(state_ptr[].ctx, state_ptr[].spk_affine, emb_buf, spks_buf, b)
    state_ptr[].ctx.synchronize()
    return PythonObject(None)


def destroy_op(handle: PythonObject) raises -> PythonObject:
    var addr = Int(py=handle)
    if addr == 0:
        return PythonObject(None)
    var state_ptr = UnsafePointer[OpState, MutExternalOrigin](unsafe_from_address=addr)
    state_ptr.destroy_pointee()
    state_ptr.free()
    return PythonObject(None)


@export
def PyInit_op_spk_affine() -> PythonObject:
    try:
        var b = PythonModuleBuilder("op_spk_affine")
        b.def_function[init_op]("init_op")
        b.def_function[forward]("forward")
        b.def_function[destroy_op]("destroy_op")
        return b.finalize()
    except e:
        abort(String("failed to create op_spk_affine module: ", e))
