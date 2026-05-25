"""op_load_wav: Phase A feasibility prototype.

Mirrors MAX's per-op .so pattern (see _interpreter_ops/matmul_ops.mojo). The
goal of this op is to validate the toolchain — Mojo .so + PythonModuleBuilder
+ mojo.importer + max.driver.Buffer pointer handoff — works in our nightly.

Public functions (called from Python):
- `get_wav_size(path) -> (n_samples, sample_rate)`: read header only.
- `load_wav_into(out_buffer, path)`: read a 16-bit PCM mono/stereo WAV,
  decode to float32 in [-1, 1], mix multichannel to mono, write into the
  pre-allocated host Buffer at out_buffer._data_ptr().

The Python side allocates the output Buffer (sized via get_wav_size) and
passes it in. We never copy back; we write directly to the buffer's pointer.
"""

from std.os import abort
from std.python import Python, PythonObject
from std.python.bindings import PythonModuleBuilder
from std.subprocess import run


# ---------------------------------------------------------------------------
# Internal helpers — little-endian byte reads.
# ---------------------------------------------------------------------------

@always_inline
def _read_i16_le(bytes: List[UInt8], off: Int) -> Int:
    var v: Int16 = 0
    v = v | Int16(bytes[off])
    v = v | (Int16(bytes[off + 1]) << Int16(8))
    return Int(v)


@always_inline
def _read_i32_le(bytes: List[UInt8], off: Int) -> Int:
    var v: Int32 = 0
    for k in range(4):
        v = v | (Int32(bytes[off + k]) << Int32(8 * k))
    return Int(v)


@fieldwise_init
struct WavHeader(Movable):
    var n_samples: Int           # per-channel sample count after mono mix
    var sample_rate: Int
    var num_channels: Int
    var bits_per_sample: Int
    var data_off: Int            # byte offset of PCM data
    var data_size: Int


def _parse_wav_header(bytes: List[UInt8]) raises -> WavHeader:
    var n_total = len(bytes)
    if n_total < 44:
        raise Error("WAV too small: " + String(n_total) + " bytes")
    # "RIFF"
    if bytes[0] != 0x52 or bytes[1] != 0x49 or bytes[2] != 0x46 or bytes[3] != 0x46:
        raise Error("not a RIFF file")
    # "WAVE"
    if bytes[8] != 0x57 or bytes[9] != 0x41 or bytes[10] != 0x56 or bytes[11] != 0x45:
        raise Error("not a WAVE file")

    var off = 12
    var sample_rate = 0
    var num_channels = 1
    var bits_per_sample = 16
    var data_off = 0
    var data_size = 0
    while off + 8 <= n_total:
        var c0 = bytes[off]
        var c1 = bytes[off + 1]
        var c2 = bytes[off + 2]
        var c3 = bytes[off + 3]
        var chunk_size = _read_i32_le(bytes, off + 4)
        if c0 == 0x66 and c1 == 0x6D and c2 == 0x74 and c3 == 0x20:  # "fmt "
            num_channels = _read_i16_le(bytes, off + 10)
            sample_rate = _read_i32_le(bytes, off + 12)
            bits_per_sample = _read_i16_le(bytes, off + 22)
        elif c0 == 0x64 and c1 == 0x61 and c2 == 0x74 and c3 == 0x61:  # "data"
            data_off = off + 8
            data_size = chunk_size
            break
        off += 8 + chunk_size

    if data_off == 0:
        raise Error("no data chunk in WAV")
    if bits_per_sample != 16:
        raise Error(
            "only 16-bit PCM WAVs supported; got "
            + String(bits_per_sample) + " bits"
        )
    var bytes_per_sample = (bits_per_sample // 8) * num_channels
    if bytes_per_sample == 0:
        raise Error("bad WAV format: zero bytes per sample")
    var n_samples = data_size // bytes_per_sample

    return WavHeader(
        n_samples=n_samples,
        sample_rate=sample_rate,
        num_channels=num_channels,
        bits_per_sample=bits_per_sample,
        data_off=data_off,
        data_size=data_size,
    )


# ---------------------------------------------------------------------------
# Public Python-facing dispatchers.
# ---------------------------------------------------------------------------

def get_wav_size(path_obj: PythonObject) raises -> PythonObject:
    """Return (n_samples_after_mono_mix, sample_rate) as a Python tuple."""
    var path = String(py=path_obj)
    var f = open(path, "r")
    var bytes = f.read_bytes()
    f.close()
    var hdr = _parse_wav_header(bytes)
    return Python.tuple(hdr.n_samples, hdr.sample_rate)


def load_wav_into(
    out_buffer: PythonObject, path_obj: PythonObject
) raises -> PythonObject:
    """Decode WAV samples into the pre-allocated host buffer.

    `out_buffer` must be a `max.driver.Buffer` of dtype float32 on a host
    device, with shape (n_samples,) matching get_wav_size()[0]. We write to
    `out_buffer._data_ptr()` directly. Returns the sample rate.
    """
    var path = String(py=path_obj)
    var f = open(path, "r")
    var bytes = f.read_bytes()
    f.close()
    var hdr = _parse_wav_header(bytes)

    var buf_capacity = Int(py=out_buffer.num_elements)
    if buf_capacity < hdr.n_samples:
        raise Error(
            "out_buffer too small: "
            + String(buf_capacity)
            + " < "
            + String(hdr.n_samples)
        )

    var addr = Int(py=out_buffer._data_ptr())
    var out_ptr = UnsafePointer[Float32, MutExternalOrigin](
        unsafe_from_address=addr
    )
    var inv = Float32(1.0 / 32768.0)
    var i = 0
    while i < hdr.n_samples:
        var sum: Float32 = 0.0
        var ch = 0
        while ch < hdr.num_channels:
            var pos = hdr.data_off + (i * hdr.num_channels + ch) * 2
            var s16 = _read_i16_le(bytes, pos)
            sum = sum + Float32(s16) * inv
            ch = ch + 1
        out_ptr[i] = sum / Float32(hdr.num_channels)
        i = i + 1

    return PythonObject(hdr.sample_rate)


# ---------------------------------------------------------------------------
# Resampler: 24k → 16k via ffmpeg+libsoxr subprocess.
#
# Bit-exact to librosa.resample(res_type='soxr_hq') when ffmpeg invoked with
# aresample=resampler=soxr:precision=20. Mojo nightly doesn't expose FFI so
# we shell out via std.subprocess.run — ffmpeg and libsoxr are OS-level
# (apt packages ffmpeg + libsoxr0). Pure Mojo runtime, no Python in this path.
# ---------------------------------------------------------------------------

def _save_raw_f32(path: String, ptr: UnsafePointer[Float32, MutAnyOrigin], n: Int) raises:
    """Write n float32 samples to path as headerless little-endian PCM."""
    var f = open(path, "w")
    var i = 0
    while i < n:
        var chunk_end = min(i + 4096, n)
        var buf = List[UInt8](capacity=(chunk_end - i) * 4)
        for j in range(i, chunk_end):
            var v = ptr[j]
            var p = UnsafePointer(to=v).bitcast[UInt32]()
            var bits = p[0]
            for k in range(4):
                buf.append(UInt8(Int((bits >> UInt32(8 * k)) & 0xFF)))
        f.write_bytes(Span(buf))
        i = chunk_end
    f.close()


def _load_raw_f32_into(path: String, out_ptr: UnsafePointer[Float32, MutExternalOrigin], n_out: Int) raises:
    """Read up to n_out float32 LE samples from path; write to out_ptr[0..n_out)."""
    var f = open(path, "r")
    var bytes = f.read_bytes()
    f.close()
    var n_available = len(bytes) // 4
    if n_available < n_out:
        raise Error("resample output shorter than expected: "
                    + String(n_available) + " < " + String(n_out))
    var i = 0
    while i < n_out:
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
        out_ptr[i] = v
        i += 1


def resample_into(
    in_buffer: PythonObject,
    out_buffer: PythonObject,
    src_sr_obj: PythonObject,
    dst_sr_obj: PythonObject,
    n_in_obj: PythonObject,
    n_out_obj: PythonObject,
    tmp_dir_obj: PythonObject,
) raises -> PythonObject:
    """Resample in_buffer (src_sr → dst_sr) into out_buffer via ffmpeg+soxr.

    Both buffers must be host-side float32 (mono). The orchestrator pre-sizes
    out_buffer to ceil(n_in * dst_sr / src_sr) samples.
    """
    var src_sr = Int(py=src_sr_obj)
    var dst_sr = Int(py=dst_sr_obj)
    var n_in = Int(py=n_in_obj)
    var n_out = Int(py=n_out_obj)
    var tmp_dir = String(py=tmp_dir_obj)

    var in_addr = Int(py=in_buffer._data_ptr())
    var out_addr = Int(py=out_buffer._data_ptr())
    var in_ptr = UnsafePointer[Float32, MutAnyOrigin](unsafe_from_address=in_addr)
    var out_ptr = UnsafePointer[Float32, MutExternalOrigin](unsafe_from_address=out_addr)

    var src_path = tmp_dir + "/resampler_in.f32"
    var dst_path = tmp_dir + "/resampler_out.f32"
    _save_raw_f32(src_path, in_ptr, n_in)

    var cmd = (
        "ffmpeg -y -loglevel error"
        + " -f f32le -ar " + String(src_sr) + " -ac 1 -i " + src_path
        + " -af aresample=resampler=soxr:precision=20"
        + " -ar " + String(dst_sr) + " -ac 1 -f f32le " + dst_path
    )
    var rc = run(cmd)
    _ = rc

    _load_raw_f32_into(dst_path, out_ptr, n_out)
    return PythonObject(None)


def predict_resample_size(
    n_in_obj: PythonObject, src_sr_obj: PythonObject, dst_sr_obj: PythonObject
) raises -> PythonObject:
    """Estimate output sample count for given input.

    soxr's actual output length is ceil(n_in * dst/src). Round up to be safe.
    """
    var n_in = Int(py=n_in_obj)
    var src_sr = Int(py=src_sr_obj)
    var dst_sr = Int(py=dst_sr_obj)
    # ceil(n_in * dst / src)
    var n_out = (n_in * dst_sr + src_sr - 1) // src_sr
    return PythonObject(n_out)


# ---------------------------------------------------------------------------
# Module init.
# ---------------------------------------------------------------------------

@export
def PyInit_op_load_wav() -> PythonObject:
    """Create the Python module surface for this op."""
    try:
        var b = PythonModuleBuilder("op_load_wav")
        b.def_function[get_wav_size](
            "get_wav_size",
            docstring="get_wav_size(path) -> (n_samples, sample_rate)",
        )
        b.def_function[load_wav_into](
            "load_wav_into",
            docstring="load_wav_into(buffer, path) -> sample_rate",
        )
        b.def_function[resample_into](
            "resample_into",
            docstring="resample_into(in_buf, out_buf, src_sr, dst_sr, n_in, n_out, tmp_dir)",
        )
        b.def_function[predict_resample_size](
            "predict_resample_size",
            docstring="predict_resample_size(n_in, src_sr, dst_sr) -> n_out_upper_bound",
        )
        return b.finalize()
    except e:
        abort(String("failed to create op_load_wav module: ", e))
