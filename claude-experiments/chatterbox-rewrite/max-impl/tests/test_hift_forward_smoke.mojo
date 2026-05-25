"""Runtime smoke test for the HiFTGenerator decode trunk forward.

Loads real upstream NSF-HiFiGAN weights and runs conv_pre → 3 ups stages with
MRF resblocks → conv_post on a small T_mel=4 input. Source fusion is skipped
(use_source=False) since source_downs/resblocks need full audio-length STFT
input which we don't synthesize here.

Output is the (B, 18, T_out+1) conv_post tensor — caller would split into
magnitude/phase and apply iSTFT to get audio. iSTFT impl is the next step.
"""
from std.sys import has_accelerator
from std.math import sin
from std.testing import TestSuite, assert_true
from std.gpu.host import DeviceContext

from weights import load_hift_generator
from hift_generator import (
    hift_decode_trunk, istft_forward, hann_window_periodic_fill,
)


comptime B = 1
comptime T_MEL = 4
comptime MEL = 80
# T grows: 4 → 32 → 256 → 1024 → +1 reflection = 1025
comptime T_OUT = 1025
comptime N_OUT = 18
comptime N_FFT = 16
comptime HOP = 4
# After iSTFT (un-padded): T_audio = (T_OUT - 1) * HOP = 1024 * 4 = 4096
comptime T_AUDIO = (T_OUT - 1) * HOP


def test_hift_forward_smoke() raises:
    comptime assert has_accelerator(), "Requires GPU"
    var ctx = DeviceContext()
    print("[hift-fwd] loading HiFTGenerator from weights/s3gen/mel2wav/...")
    var hift = load_hift_generator(ctx, "weights/s3gen/mel2wav")

    var mel = ctx.enqueue_create_buffer[DType.float32](B * MEL * T_MEL)
    with mel.map_to_host() as h:
        for c in range(MEL):
            for ti in range(T_MEL):
                h[c * T_MEL + ti] = sin(Float32(c) * 0.05 + Float32(ti) * 0.1) * 0.1

    # Unused when use_source=False, but allocate something legal.
    var s_stft = ctx.enqueue_create_buffer[DType.float32](B * N_OUT * 1)

    var spec_out = ctx.enqueue_create_buffer[DType.float32](B * N_OUT * T_OUT)
    print("[hift-fwd] running decode trunk (T_mel=", T_MEL, " → T_out=", T_OUT, ")...")
    hift_decode_trunk(
        ctx, hift, mel, s_stft, spec_out, B, T_MEL, 1, T_OUT, use_source=False,
    )
    ctx.synchronize()

    var n_nan = 0
    var sum_abs: Float32 = 0.0
    with spec_out.map_to_host() as h:
        for i in range(B * N_OUT * T_OUT):
            var v = h[i]
            if v != v: n_nan += 1
            if v < 0.0: sum_abs -= v
            else:       sum_abs += v
    print("[hift-fwd] pre-iSTFT output mean-abs=",
          sum_abs / Float32(B * N_OUT * T_OUT), " nan_count=", n_nan)
    assert_true(n_nan == 0, "no NaNs in HiFT pre-iSTFT output")

    # Run iSTFT: (B, 18, T_OUT) → (B, T_AUDIO).
    var window = ctx.enqueue_create_buffer[DType.float32](N_FFT)
    hann_window_periodic_fill(ctx, window, N_FFT)
    var audio = ctx.enqueue_create_buffer[DType.float32](B * T_AUDIO)
    print("[hift-fwd] running iSTFT (T_frames=", T_OUT, " → T_audio=", T_AUDIO, ")...")
    istft_forward(ctx, spec_out, window, audio, B, N_FFT, T_OUT, T_AUDIO)
    ctx.synchronize()

    var n_nan_a = 0
    var sum_abs_a: Float32 = 0.0
    var max_a: Float32 = 0.0
    with audio.map_to_host() as h:
        for i in range(B * T_AUDIO):
            var v = h[i]
            if v != v: n_nan_a += 1
            var av = v
            if av < 0.0: av = -av
            sum_abs_a += av
            if av > max_a: max_a = av
    print("[hift-fwd] audio mean-abs=", sum_abs_a / Float32(B * T_AUDIO),
          " max=", max_a, " nan_count=", n_nan_a)
    assert_true(n_nan_a == 0, "no NaNs in audio output")


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
