"""Bit-exact 24k→16k resampling via ffmpeg+libsoxr subprocess.

Equivalent to `librosa.resample(y, orig_sr=24000, target_sr=16000)` (which
defaults to `res_type='soxr_hq'`). Verified bit-exact when ffmpeg is invoked
with `aresample=resampler=soxr:precision=20`.

Mojo doesn't expose FFI to user code in this nightly, so we shell out via
`std.subprocess.run`. ffmpeg + libsoxr are linked at the OS level (apt
packages `ffmpeg` and `libsoxr0`) — pure Mojo runtime, no Python.
"""
from std.subprocess import run
from std.gpu.host import DeviceContext, DeviceBuffer

from fixture import save_fp32_1d, load_fp32


def soxr_resample_24k_to_16k(
    mut ctx: DeviceContext,
    mut in_buf: DeviceBuffer[DType.float32],     # (n_in,) 24kHz
    mut out_buf: DeviceBuffer[DType.float32],    # (n_out,) 16kHz
    n_in: Int, n_out: Int,
    tmp_dir: String = "/tmp",
) raises:
    """Resample 24kHz → 16kHz via ffmpeg+libsoxr (precision=20).

    Writes input to a temp WAV (float32 raw PCM), runs ffmpeg, reads back.
    Output is bit-exact to librosa.resample(res_type='soxr_hq').
    """
    # 1. Pull samples to host.
    var samples_in = List[Float32](capacity=n_in)
    with in_buf.map_to_host() as h:
        for i in range(n_in):
            samples_in.append(h[i])

    # 2. Save as raw float32 PCM at 24kHz (headerless, ffmpeg reads via -f f32le).
    var src_path = tmp_dir + "/mojo_resampler_in.f32"
    save_raw_f32(src_path, samples_in)

    # 3. Run ffmpeg with soxr precision=20 (bit-exact to librosa.resample 'soxr_hq').
    #    Input + output as headerless float32 LE for simple parsing.
    var dst_path = tmp_dir + "/mojo_resampler_out.f32"
    var cmd = (
        "ffmpeg -y -loglevel error"
        + " -f f32le -ar 24000 -ac 1 -i " + src_path
        + " -af aresample=resampler=soxr:precision=20"
        + " -ar 16000 -ac 1 -f f32le " + dst_path
    )
    var rc = run(cmd)
    _ = rc

    # 4. Load raw f32 output.
    var samples_out = load_raw_f32(dst_path)
    if len(samples_out) < n_out:
        raise Error("ffmpeg output shorter than expected: " + String(len(samples_out)) + " vs " + String(n_out))
    with out_buf.map_to_host() as h:
        for i in range(n_out):
            h[i] = samples_out[i]


def save_raw_f32(path: String, samples: List[Float32]) raises:
    """Write samples as raw float32 little-endian, no header."""
    var f = open(path, "w")
    var n = len(samples)
    var i = 0
    while i < n:
        var chunk_end = min(i + 4096, n)
        var buf = List[UInt8](capacity=(chunk_end - i) * 4)
        for j in range(i, chunk_end):
            var v = samples[j]
            var p = UnsafePointer(to=v).bitcast[UInt32]()
            var bits = p[0]
            for k in range(4):
                buf.append(UInt8(Int((bits >> UInt32(8 * k)) & 0xFF)))
        f.write_bytes(Span(buf))
        i = chunk_end
    f.close()


def load_raw_f32(path: String) raises -> List[Float32]:
    """Read raw float32 LE samples (no header)."""
    var f = open(path, "r")
    var bytes = f.read_bytes()
    f.close()
    var n_samples = len(bytes) // 4
    var out = List[Float32](capacity=n_samples)
    var i = 0
    while i < n_samples:
        var pos = i * 4
        var bits: UInt32 = (
            UInt32(bytes[pos])
            | (UInt32(bytes[pos + 1]) << 8)
            | (UInt32(bytes[pos + 2]) << 16)
            | (UInt32(bytes[pos + 3]) << 24)
        )
        var v: Float32 = 0.0
        var p = UnsafePointer(to=v).bitcast[UInt32]()
        p[0] = bits
        out.append(v)
        i += 1
    return out^


