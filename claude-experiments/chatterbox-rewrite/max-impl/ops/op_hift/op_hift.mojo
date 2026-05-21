"""op_hift: NSF-HiFiGAN mel-to-audio. F0 predictor + sine source + iSTFT.

Public API:
- init_op(hift_base, device_ctx) -> handle
- forward(handle, mel_buf, audio_out_buf, B, T_out_mel) -> None
- destroy_op(handle) -> None
"""
from std.os import abort
from std.memory import OpaquePointer
from std.python import PythonObject
from std.python.bindings import PythonModuleBuilder
from std.gpu.host import DeviceContext, DeviceBuffer
from std.runtime.asyncrt import DeviceContextPtr

from hift_generator import (
    HiFTGenerator, hann_window_periodic_fill,
    f0_predictor_forward, f0_upsample_nearest, source_module_forward,
    build_s_stft_from_signal, hift_decode_trunk, istft_forward,
)
from weights import load_hift_generator


comptime N_FFT = 16
comptime N_OUT = 18    # n_fft + 2
comptime HOP = 4


@fieldwise_init
struct OpState(Movable):
    var ctx: DeviceContext
    var hift: HiFTGenerator


def _ctx_from_python(device_context_ptr: PythonObject) raises -> DeviceContext:
    var addr = Int(py=device_context_ptr)
    if addr == 0:
        raise Error("op_hift requires a GPU device context")
    var opaque = OpaquePointer[MutExternalOrigin](unsafe_from_address=addr)
    return DeviceContextPtr(opaque).get_device_context()


def init_op(
    hift_base: PythonObject, device_context_ptr: PythonObject
) raises -> PythonObject:
    var base = String(py=hift_base)
    var ctx = _ctx_from_python(device_context_ptr)
    var hift = load_hift_generator(ctx, base)
    ctx.synchronize()
    var ptr = alloc[OpState](1)
    ptr.init_pointee_move(OpState(ctx^, hift^))
    return PythonObject(Int(ptr))


def forward(
    handle: PythonObject,
    mel_buf: PythonObject,        # (B, 80, T_out_mel) GPU f32
    audio_out_buf: PythonObject,  # (B, T_audio) GPU f32
    b_obj: PythonObject,
    t_out_mel_obj: PythonObject,
) raises -> PythonObject:
    var state_ptr = UnsafePointer[OpState, MutExternalOrigin](
        unsafe_from_address=Int(py=handle)
    )
    var ctx_ref = state_ptr[].ctx
    var B = Int(py=b_obj)
    var T_OUT_MEL = Int(py=t_out_mel_obj)

    var T_HIFT = T_OUT_MEL * 120 + 1
    var T_AUDIO_FULL = T_OUT_MEL * 480
    var T_S_FRAMES = T_AUDIO_FULL // HOP + 1
    var T_AUDIO = (T_HIFT - 1) * HOP

    var mel_addr = Int(py=mel_buf._data_ptr())
    var mel_n = Int(py=mel_buf.num_elements)
    var mel_ptr = UnsafePointer[Float32, MutExternalOrigin](unsafe_from_address=mel_addr)
    var mel = DeviceBuffer[DType.float32](ctx_ref, mel_ptr, mel_n, owning=False)

    var ao_addr = Int(py=audio_out_buf._data_ptr())
    var ao_n = Int(py=audio_out_buf.num_elements)
    var ao_ptr = UnsafePointer[Float32, MutExternalOrigin](unsafe_from_address=ao_addr)
    var audio_out = DeviceBuffer[DType.float32](ctx_ref, ao_ptr, ao_n, owning=False)

    # F0 predictor + sine source.
    var f0 = ctx_ref.enqueue_create_buffer[DType.float32](B * T_OUT_MEL)
    f0_predictor_forward(ctx_ref, state_ptr[].hift.f0_predictor, mel, f0, B, T_OUT_MEL)
    var f0_up = ctx_ref.enqueue_create_buffer[DType.float32](B * T_AUDIO_FULL)
    f0_upsample_nearest(ctx_ref, f0, f0_up, B, T_OUT_MEL, 480)

    var sine_merge = ctx_ref.enqueue_create_buffer[DType.float32](B * 1 * T_AUDIO_FULL)
    source_module_forward(
        ctx_ref, state_ptr[].hift.m_source, f0_up, sine_merge,
        B, T_AUDIO_FULL,
        sampling_rate=24000, harmonic_num=8,
        sine_amp=Float32(0.1), noise_std=Float32(0.003),
        voiced_threshold=Float32(10.0),
    )

    var window_s = ctx_ref.enqueue_create_buffer[DType.float32](N_FFT)
    hann_window_periodic_fill(ctx_ref, window_s, N_FFT)
    var s_stft = ctx_ref.enqueue_create_buffer[DType.float32](B * N_OUT * T_S_FRAMES)
    build_s_stft_from_signal(
        ctx_ref, sine_merge, window_s, s_stft,
        B, T_AUDIO_FULL, N_FFT, HOP, T_S_FRAMES,
    )

    var spec = ctx_ref.enqueue_create_buffer[DType.float32](B * N_OUT * T_HIFT)
    hift_decode_trunk(
        ctx_ref, state_ptr[].hift, mel, s_stft, spec,
        B, T_OUT_MEL, T_S_FRAMES, T_HIFT, use_source=True,
    )

    var window_i = ctx_ref.enqueue_create_buffer[DType.float32](N_FFT)
    hann_window_periodic_fill(ctx_ref, window_i, N_FFT)
    istft_forward(ctx_ref, spec, window_i, audio_out, B, N_FFT, T_HIFT, T_AUDIO)
    ctx_ref.synchronize()
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
def PyInit_op_hift() -> PythonObject:
    try:
        var b = PythonModuleBuilder("op_hift")
        b.def_function[init_op]("init_op")
        b.def_function[forward]("forward")
        b.def_function[destroy_op]("destroy_op")
        return b.finalize()
    except e:
        abort(String("failed to create op_hift module: ", e))
