"""op_campplus: CAMPPlus xvector backbone as a Mojo .so op.

Phase B of the MAX-style per-op refactor. Validates:
1. Heap-allocated weight state shared across calls via opaque int handle
2. GPU device buffer handoff: max.driver.Buffer._data_ptr() → DeviceBuffer
3. Shared DeviceContext via _device_context_ptr() from Python
4. End-to-end parity with src/ direct-call test on the same fixtures

Python API (3 functions):
- init_op(weights_base_path, device_context_ptr) -> handle (int)
- xvector_forward(handle, in_buf, out_buf, B, T_in) -> None
- destroy_op(handle) -> None

Activations cross the FFI boundary as `max.driver.Buffer` (GPU-resident);
weights live inside the heap-allocated state owned by Mojo.
"""

from std.os import abort
from std.memory import OpaquePointer
from std.python import PythonObject
from std.python.bindings import PythonModuleBuilder
from std.gpu.host import DeviceContext, DeviceBuffer
from std.runtime.asyncrt import DeviceContextPtr

from campplus import XVectorBackbone, xvector_forward as _xvector_forward, campplus_speaker_embedding as _campplus_speaker_embedding
from fcm import FCM
from weights_campplus import load_xvector_backbone, load_fcm


# ---------------------------------------------------------------------------
# Persistent op state.
# ---------------------------------------------------------------------------

@fieldwise_init
struct OpState(Movable):
    """Heap-resident state for one op_campplus instance.

    Weight tensors live on the GPU; we hold their owning `DeviceBuffer`s
    inside `backbone`. The same `DeviceContext` is shared with the rest of
    the orchestrator (constructed from the address Python passes in).
    """
    var ctx: DeviceContext
    var backbone: XVectorBackbone
    var fcm: FCM


# ---------------------------------------------------------------------------
# Helpers shared with MAX op_utils (vendored to avoid an external import path).
# ---------------------------------------------------------------------------

@always_inline
def _make_ptr(addr: Int) -> UnsafePointer[Float32, MutExternalOrigin]:
    return UnsafePointer[Float32, MutExternalOrigin](unsafe_from_address=addr)


def _ctx_from_python(device_context_ptr: PythonObject) raises -> DeviceContext:
    """Reconstruct a DeviceContext from the int address Python passed in.

    Mirrors max._interpreter_ops.op_utils._get_ctx, then unwraps the
    DeviceContextPtr to get back the real DeviceContext.
    """
    var addr = Int(py=device_context_ptr)
    if addr == 0:
        raise Error("op_campplus requires a GPU device context (got null)")
    var opaque = OpaquePointer[MutExternalOrigin](unsafe_from_address=addr)
    var dcp = DeviceContextPtr(opaque)
    return dcp.get_device_context()


# ---------------------------------------------------------------------------
# Public Python-facing functions.
# ---------------------------------------------------------------------------

def init_op(
    weights_base_path: PythonObject,
    device_context_ptr: PythonObject,
) raises -> PythonObject:
    """Load CAMPPlus weights to GPU; return a heap-allocated state address.

    Args:
        weights_base_path: e.g. "weights/s3gen/speaker_encoder".
        device_context_ptr: int from `Accelerator()._device_context_ptr()`.

    Returns:
        An opaque int handle. Pass to xvector_forward / destroy_op.
    """
    var base = String(py=weights_base_path)
    var ctx = _ctx_from_python(device_context_ptr)
    var backbone = load_xvector_backbone(ctx, base)
    var fcm = load_fcm(ctx, base + "/head")
    ctx.synchronize()

    var ptr = alloc[OpState](1)
    ptr.init_pointee_move(OpState(ctx^, backbone^, fcm^))
    return PythonObject(Int(ptr))


def xvector_forward(
    handle: PythonObject,
    in_buffer: PythonObject,     # (B, 320, T_in) f32 on GPU
    out_buffer: PythonObject,    # (B, 192) f32 on GPU
    b_obj: PythonObject,
    t_in_obj: PythonObject,
) raises -> PythonObject:
    """Run the CAMPPlus xvector backbone forward pass.

    Wraps the input/output GPU pointers as non-owning DeviceBuffers, then
    calls src/campplus.mojo:xvector_forward. The state's ctx is the same
    device context the orchestrator owns, so the resulting buffer is
    visible to subsequent ops without a host roundtrip.
    """
    var addr = Int(py=handle)
    if addr == 0:
        raise Error("op_campplus: null state handle")
    var state_ptr = UnsafePointer[OpState, MutExternalOrigin](
        unsafe_from_address=addr
    )

    var b = Int(py=b_obj)
    var t_in = Int(py=t_in_obj)

    var in_addr = Int(py=in_buffer._data_ptr())
    var in_n = Int(py=in_buffer.num_elements)
    var out_addr = Int(py=out_buffer._data_ptr())
    var out_n = Int(py=out_buffer.num_elements)

    var expected_in = b * 320 * t_in
    if in_n < expected_in:
        raise Error(
            "in_buffer too small: " + String(in_n)
            + " < expected " + String(expected_in)
        )
    var expected_out = b * 192
    if out_n < expected_out:
        raise Error(
            "out_buffer too small: " + String(out_n)
            + " < expected " + String(expected_out)
        )

    # Wrap GPU pointers as non-owning DeviceBuffers (skill-documented pattern).
    var in_buf = DeviceBuffer[DType.float32](
        state_ptr[].ctx, _make_ptr(in_addr), in_n, owning=False,
    )
    var out_buf = DeviceBuffer[DType.float32](
        state_ptr[].ctx, _make_ptr(out_addr), out_n, owning=False,
    )

    _xvector_forward(
        state_ptr[].ctx, state_ptr[].backbone, in_buf, out_buf, b, t_in,
    )
    state_ptr[].ctx.synchronize()
    return PythonObject(None)


def speaker_embedding(
    handle: PythonObject,
    in_buffer: PythonObject,     # (B, 80, T_fbank) f32 on GPU — kaldi fbank output
    out_buffer: PythonObject,    # (B, 192) f32 on GPU
    b_obj: PythonObject,
    t_in_obj: PythonObject,
) raises -> PythonObject:
    """Full CAMPPlus pipeline: FCM(80→320 channels) + xvector → 192-d embed."""
    var addr = Int(py=handle)
    if addr == 0:
        raise Error("op_campplus: null state handle")
    var state_ptr = UnsafePointer[OpState, MutExternalOrigin](unsafe_from_address=addr)
    var b = Int(py=b_obj)
    var t_in = Int(py=t_in_obj)

    var in_addr = Int(py=in_buffer._data_ptr())
    var in_n = Int(py=in_buffer.num_elements)
    var out_addr = Int(py=out_buffer._data_ptr())
    var out_n = Int(py=out_buffer.num_elements)

    var expected_in = b * 80 * t_in
    if in_n < expected_in:
        raise Error("in_buffer too small: " + String(in_n) + " < " + String(expected_in))
    var expected_out = b * 192
    if out_n < expected_out:
        raise Error("out_buffer too small: " + String(out_n) + " < " + String(expected_out))

    var in_buf = DeviceBuffer[DType.float32](
        state_ptr[].ctx, _make_ptr(in_addr), in_n, owning=False,
    )
    var out_buf = DeviceBuffer[DType.float32](
        state_ptr[].ctx, _make_ptr(out_addr), out_n, owning=False,
    )

    _campplus_speaker_embedding(
        state_ptr[].ctx, state_ptr[].fcm, state_ptr[].backbone,
        in_buf, out_buf, b, t_in,
    )
    state_ptr[].ctx.synchronize()
    return PythonObject(None)


def destroy_op(handle: PythonObject) raises -> PythonObject:
    """Free the heap-allocated state.

    After this, the handle is invalid — do not call xvector_forward again.
    """
    var addr = Int(py=handle)
    if addr == 0:
        return PythonObject(None)
    var state_ptr = UnsafePointer[OpState, MutExternalOrigin](
        unsafe_from_address=addr
    )
    state_ptr.destroy_pointee()
    state_ptr.free()
    return PythonObject(None)


# ---------------------------------------------------------------------------
# Module init.
# ---------------------------------------------------------------------------

@export
def PyInit_op_campplus() -> PythonObject:
    try:
        var b = PythonModuleBuilder("op_campplus")
        b.def_function[init_op](
            "init_op",
            docstring="init_op(weights_base_path, device_ctx_ptr) -> handle",
        )
        b.def_function[xvector_forward](
            "xvector_forward",
            docstring="xvector_forward(handle, in_buf, out_buf, B, T_in)",
        )
        b.def_function[speaker_embedding](
            "speaker_embedding",
            docstring="speaker_embedding(handle, fbank_buf, emb_out, B, T_in) — FCM + xvector",
        )
        b.def_function[destroy_op](
            "destroy_op",
            docstring="destroy_op(handle) — release the heap-allocated state",
        )
        return b.finalize()
    except e:
        abort(String("failed to create op_campplus module: ", e))
