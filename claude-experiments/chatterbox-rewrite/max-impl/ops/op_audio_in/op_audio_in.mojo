"""op_audio_in: Voice preprocessing — 16k wav → (prompt_token, cond_prompt_tok, prompt_feat, spk_embed_256).

Encapsulates the full audio frontend in one .so:
- Kaldi fbank → CAMPPlus mel input
- 24k mel for prompt_feat
- s3tokenizer log-mel → prompt_token (full 10s) + cond_prompt_speech_tokens (6s)
- VoiceEncoder mel + multi-partial inference

The orchestrator calls init_op once with all weight paths, then forward() with
the 24k and 16k audio buffers. CAMPPlus runs in a separate op (op_campplus)
because we already validated that one in Phase B.
"""
from std.os import abort
from std.memory import OpaquePointer
from std.python import Python, PythonObject
from std.python.bindings import PythonModuleBuilder
from std.gpu.host import DeviceContext, DeviceBuffer
from std.runtime.asyncrt import DeviceContextPtr

from fixture import load_fp32
from s3tokenizer import S3Tokenizer, s3tokenizer_forward
from voice_encoder import VoiceEncoder, voice_encoder_inference
from kaldi_fbank import (
    kaldi_fbank_forward, kaldi_subtract_column_mean,
    build_povey_window, build_kaldi_mel_filterbank,
)
from mel_24k import (
    mel_24k_forward, build_hann_window as build_hann_24k,
    build_librosa_mel_filterbank as build_mel_fb_24k,
)
from mel_s3tok import (
    log_mel_s3tok_forward, build_hann_window_full as build_hann_s3tok,
    build_librosa_mel_filterbank_s3tok,
)
from mel_ve import mel_ve_forward
from rope_local import build_rope_tables
from weights_audio_in import load_s3tokenizer, load_voice_encoder


# ---------------------------------------------------------------------------
# Persistent state.
# ---------------------------------------------------------------------------

@fieldwise_init
struct OpState(Movable):
    var ctx: DeviceContext
    var s3tok: S3Tokenizer
    var ve: VoiceEncoder
    # Pre-built sample-rate-derived constants (windows + mel filterbanks).
    var win_s3tok: DeviceBuffer[DType.float32]      # (400,)
    var mel_fb_s3tok: DeviceBuffer[DType.float32]   # (128, 201)
    var win_24k: DeviceBuffer[DType.float32]        # (1920,)
    var mel_fb_24k: DeviceBuffer[DType.float32]     # (80, 961)
    var kaldi_win: DeviceBuffer[DType.float32]      # (400,)
    var kaldi_mel_fb: DeviceBuffer[DType.float32]   # (80, 257)
    var win_ve: DeviceBuffer[DType.float32]         # (400,) hann
    var mel_fb_ve: DeviceBuffer[DType.float32]      # (40, 201)
    var cos_st: DeviceBuffer[DType.float32]         # (4096, 64) RoPE cos
    var sin_st: DeviceBuffer[DType.float32]         # (4096, 64) RoPE sin


def _ctx_from_python(device_context_ptr: PythonObject) raises -> DeviceContext:
    var addr = Int(py=device_context_ptr)
    if addr == 0:
        raise Error("op_audio_in requires a GPU device context")
    var opaque = OpaquePointer[MutExternalOrigin](unsafe_from_address=addr)
    return DeviceContextPtr(opaque).get_device_context()


def _wrap_buf(
    mut ctx: DeviceContext, addr: Int, n: Int
) -> DeviceBuffer[DType.float32]:
    var ptr = UnsafePointer[Float32, MutExternalOrigin](unsafe_from_address=addr)
    return DeviceBuffer[DType.float32](ctx, ptr, n, owning=False)


def _wrap_buf_i32(
    mut ctx: DeviceContext, addr: Int, n: Int
) -> DeviceBuffer[DType.int32]:
    var ptr = UnsafePointer[Int32, MutExternalOrigin](unsafe_from_address=addr)
    return DeviceBuffer[DType.int32](ctx, ptr, n, owning=False)


# ---------------------------------------------------------------------------
# init_op: load weights, build sample-rate-derived constants, return handle.
# ---------------------------------------------------------------------------

def init_op(
    s3t_base: PythonObject,
    ve_base: PythonObject,
    device_context_ptr: PythonObject,
) raises -> PythonObject:
    var s3t_path = String(py=s3t_base)
    var ve_path = String(py=ve_base)
    var ctx = _ctx_from_python(device_context_ptr)

    var s3tok = load_s3tokenizer(ctx, s3t_path)
    var ve = load_voice_encoder(ctx, ve_path)

    # s3tok mel filterbank (16k, 128 mels, n_fft=400, hop=160)
    var win_s3tok = ctx.enqueue_create_buffer[DType.float32](400)
    build_hann_s3tok(ctx, win_s3tok, 400)
    var mel_fb_s3tok = ctx.enqueue_create_buffer[DType.float32](128 * 201)
    build_librosa_mel_filterbank_s3tok(ctx, mel_fb_s3tok, 128, 400, Float64(16000.0))

    # 24k mel filterbank (24k, 80 mels, n_fft=1920, hop=480)
    var win_24k = ctx.enqueue_create_buffer[DType.float32](1920)
    build_hann_24k(ctx, win_24k, 1920)
    var mel_fb_24k = ctx.enqueue_create_buffer[DType.float32](80 * 961)
    build_mel_fb_24k(ctx, mel_fb_24k, 80, 1920, Float64(24000.0),
                       Float64(0.0), Float64(8000.0))

    # Kaldi fbank (16k, 80 mels, window=400, padded=512, shift=160)
    var kaldi_win = ctx.enqueue_create_buffer[DType.float32](400)
    build_povey_window(ctx, kaldi_win, 400)
    var kaldi_mel_fb = ctx.enqueue_create_buffer[DType.float32](80 * 257)
    build_kaldi_mel_filterbank(ctx, kaldi_mel_fb, 80, 512, Float64(16000.0),
                                 Float64(20.0), Float64(0.0))

    # VE mel (16k, 40 mels, n_fft=400, hop=160).
    var win_ve = ctx.enqueue_create_buffer[DType.float32](400)
    build_hann_24k(ctx, win_ve, 400)
    var mel_fb_ve = ctx.enqueue_create_buffer[DType.float32](40 * 201)
    build_mel_fb_24k(ctx, mel_fb_ve, 40, 400, Float64(16000.0),
                       Float64(0.0), Float64(8000.0))

    # RoPE tables for s3tokenizer.
    var cos_st = ctx.enqueue_create_buffer[DType.float32](4096 * 64)
    var sin_st = ctx.enqueue_create_buffer[DType.float32](4096 * 64)
    build_rope_tables(ctx, 4096, 64, cos_st, sin_st)

    ctx.synchronize()

    var ptr = alloc[OpState](1)
    ptr.init_pointee_move(OpState(
        ctx^, s3tok^, ve^,
        win_s3tok^, mel_fb_s3tok^, win_24k^, mel_fb_24k^,
        kaldi_win^, kaldi_mel_fb^, win_ve^, mel_fb_ve^, cos_st^, sin_st^,
    ))
    return PythonObject(Int(ptr))


# ---------------------------------------------------------------------------
# Individual forward functions (call as needed from orchestrator).
# ---------------------------------------------------------------------------

def compute_log_mel_s3tok(
    handle: PythonObject,
    wav_16k_buf: PythonObject,    # (n_samples,) GPU f32
    out_log_mel_buf: PythonObject, # (128, T_mel) GPU f32
    n_samples_obj: PythonObject,
    t_mel_obj: PythonObject,
) raises -> PythonObject:
    """16kHz wav → (128, T_mel) log-mel for s3tokenizer."""
    var state_ptr = UnsafePointer[OpState, MutExternalOrigin](
        unsafe_from_address=Int(py=handle)
    )
    var n_samples = Int(py=n_samples_obj)
    var t_mel = Int(py=t_mel_obj)
    var wav = _wrap_buf(state_ptr[].ctx, Int(py=wav_16k_buf._data_ptr()), Int(py=wav_16k_buf.num_elements))
    var out = _wrap_buf(state_ptr[].ctx, Int(py=out_log_mel_buf._data_ptr()), Int(py=out_log_mel_buf.num_elements))
    log_mel_s3tok_forward(
        state_ptr[].ctx, wav, state_ptr[].win_s3tok, state_ptr[].mel_fb_s3tok,
        out, n_samples, t_mel,
    )
    state_ptr[].ctx.synchronize()
    return PythonObject(None)


def s3tokenize(
    handle: PythonObject,
    log_mel_buf: PythonObject,
    tokens_out_buf: PythonObject,
    mask_pad_buf: PythonObject,
    attn_mask_buf: PythonObject,
    b_obj: PythonObject,
    t_mel_obj: PythonObject,
) raises -> PythonObject:
    """Run s3tokenizer: (B, 128, T_mel) log-mel → (B, T_token) int32 tokens.

    Orchestrator pre-fills mask_pad with 1.0 and attn_mask with 0.0.
    """
    var state_ptr = UnsafePointer[OpState, MutExternalOrigin](
        unsafe_from_address=Int(py=handle)
    )
    var b = Int(py=b_obj)
    var t_mel = Int(py=t_mel_obj)
    var mel = _wrap_buf(state_ptr[].ctx, Int(py=log_mel_buf._data_ptr()), Int(py=log_mel_buf.num_elements))
    var tokens = _wrap_buf_i32(state_ptr[].ctx, Int(py=tokens_out_buf._data_ptr()), Int(py=tokens_out_buf.num_elements))
    var mp = _wrap_buf(state_ptr[].ctx, Int(py=mask_pad_buf._data_ptr()), Int(py=mask_pad_buf.num_elements))
    var am = _wrap_buf(state_ptr[].ctx, Int(py=attn_mask_buf._data_ptr()), Int(py=attn_mask_buf.num_elements))
    s3tokenizer_forward(
        state_ptr[].ctx, state_ptr[].s3tok, mel, tokens,
        state_ptr[].cos_st, state_ptr[].sin_st, mp, am,
        b, t_mel,
    )
    state_ptr[].ctx.synchronize()
    return PythonObject(None)


def compute_mel_24k(
    handle: PythonObject,
    wav_24k_buf: PythonObject,
    out_mel_mt_buf: PythonObject,    # (80, T_frames) GPU f32 — orchestrator transposes as needed
    n_samples_obj: PythonObject,
    t_frames_obj: PythonObject,
) raises -> PythonObject:
    """24kHz wav → (80, T_frames) log-mel for prompt_feat."""
    var state_ptr = UnsafePointer[OpState, MutExternalOrigin](
        unsafe_from_address=Int(py=handle)
    )
    var n_samples = Int(py=n_samples_obj)
    var t_frames = Int(py=t_frames_obj)
    var wav = _wrap_buf(state_ptr[].ctx, Int(py=wav_24k_buf._data_ptr()), Int(py=wav_24k_buf.num_elements))
    var out = _wrap_buf(state_ptr[].ctx, Int(py=out_mel_mt_buf._data_ptr()), Int(py=out_mel_mt_buf.num_elements))
    mel_24k_forward(
        state_ptr[].ctx, wav, state_ptr[].win_24k, state_ptr[].mel_fb_24k,
        out, n_samples, t_frames,
    )
    state_ptr[].ctx.synchronize()
    return PythonObject(None)


def compute_kaldi_fbank(
    handle: PythonObject,
    wav_16k_buf: PythonObject,
    out_fbank_bft_buf: PythonObject,  # (80, T_fbank) GPU f32 after column mean + transpose
    n_samples_obj: PythonObject,
    t_fbank_obj: PythonObject,
) raises -> PythonObject:
    """16kHz wav → (80, T_fbank) Kaldi log-mel features, ready for CAMPPlus input."""
    var state_ptr = UnsafePointer[OpState, MutExternalOrigin](
        unsafe_from_address=Int(py=handle)
    )
    var n_samples = Int(py=n_samples_obj)
    var t_fbank = Int(py=t_fbank_obj)
    var wav = _wrap_buf(state_ptr[].ctx, Int(py=wav_16k_buf._data_ptr()), Int(py=wav_16k_buf.num_elements))

    # Intermediate: kaldi outputs (T, 80); transpose to (80, T) for CAMPPlus.
    var tf_tmp = state_ptr[].ctx.enqueue_create_buffer[DType.float32](t_fbank * 80)
    kaldi_fbank_forward(
        state_ptr[].ctx, wav, state_ptr[].kaldi_win, state_ptr[].kaldi_mel_fb,
        tf_tmp, n_samples, t_fbank,
    )
    kaldi_subtract_column_mean(state_ptr[].ctx, tf_tmp, t_fbank, 80)

    # Transpose (T_fbank, 80) → (80, T_fbank).
    var out_addr = Int(py=out_fbank_bft_buf._data_ptr())
    var out_ptr = UnsafePointer[Float32, MutExternalOrigin](unsafe_from_address=out_addr)
    with tf_tmp.map_to_host() as src:
        # Pull source to host once; write the transpose into the dest buffer.
        # Output buffer is GPU-resident; we map it for the write.
        var out_buf = _wrap_buf(state_ptr[].ctx, out_addr, Int(py=out_fbank_bft_buf.num_elements))
        with out_buf.map_to_host() as dst:
            for f in range(80):
                for t in range(t_fbank):
                    dst[f * t_fbank + t] = src[t * 80 + f]
    state_ptr[].ctx.synchronize()
    return PythonObject(None)


def compute_ve_mel(
    handle: PythonObject,
    wav_16k_buf: PythonObject,
    out_ve_mel_buf: PythonObject,   # (T_ve, 40) GPU f32
    n_samples_obj: PythonObject,
    t_ve_obj: PythonObject,
) raises -> PythonObject:
    """16kHz wav → (T_ve, 40) mel for VoiceEncoder input."""
    var state_ptr = UnsafePointer[OpState, MutExternalOrigin](
        unsafe_from_address=Int(py=handle)
    )
    var n_samples = Int(py=n_samples_obj)
    var t_ve = Int(py=t_ve_obj)
    var wav = _wrap_buf(state_ptr[].ctx, Int(py=wav_16k_buf._data_ptr()), Int(py=wav_16k_buf.num_elements))
    var out = _wrap_buf(state_ptr[].ctx, Int(py=out_ve_mel_buf._data_ptr()), Int(py=out_ve_mel_buf.num_elements))
    mel_ve_forward(state_ptr[].ctx, wav, state_ptr[].win_ve, state_ptr[].mel_fb_ve,
                    out, n_samples, t_ve)
    state_ptr[].ctx.synchronize()
    return PythonObject(None)


def voice_encode(
    handle: PythonObject,
    ve_mel_buf: PythonObject,
    embed_out_buf: PythonObject,    # (256,) GPU f32
    t_ve_obj: PythonObject,
) raises -> PythonObject:
    """Multi-partial VE inference → 256-d speaker embedding."""
    var state_ptr = UnsafePointer[OpState, MutExternalOrigin](
        unsafe_from_address=Int(py=handle)
    )
    var t_ve = Int(py=t_ve_obj)
    var mel = _wrap_buf(state_ptr[].ctx, Int(py=ve_mel_buf._data_ptr()), Int(py=ve_mel_buf.num_elements))
    var out = _wrap_buf(state_ptr[].ctx, Int(py=embed_out_buf._data_ptr()), Int(py=embed_out_buf.num_elements))
    voice_encoder_inference(state_ptr[].ctx, state_ptr[].ve, mel, out, t_ve)
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
def PyInit_op_audio_in() -> PythonObject:
    try:
        var b = PythonModuleBuilder("op_audio_in")
        b.def_function[init_op]("init_op")
        b.def_function[compute_log_mel_s3tok]("compute_log_mel_s3tok")
        b.def_function[s3tokenize]("s3tokenize")
        b.def_function[compute_mel_24k]("compute_mel_24k")
        b.def_function[compute_kaldi_fbank]("compute_kaldi_fbank")
        b.def_function[compute_ve_mel]("compute_ve_mel")
        b.def_function[voice_encode]("voice_encode")
        b.def_function[destroy_op]("destroy_op")
        return b.finalize()
    except e:
        abort(String("failed to create op_audio_in module: ", e))
