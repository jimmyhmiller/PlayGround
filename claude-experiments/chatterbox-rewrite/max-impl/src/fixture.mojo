"""
Fixture loader for the binary tensor format produced by oracle/dump_*.py.

Format (little-endian):
  i64        rank
  i64[rank]  shape
  i32        dtype_tag    (0 = fp32; 1 = bf16-as-uint16, not yet handled here)
  payload    raw element bytes
"""

from std.io.file import open


@fieldwise_init
struct Tensor(Movable):
    var data: List[Float32]
    var shape: List[Int]

    def rank(self) -> Int:
        return len(self.shape)

    def numel(self) -> Int:
        var n = 1
        for i in range(len(self.shape)):
            n *= self.shape[i]
        return n


@fieldwise_init
struct TensorBF16(Movable):
    var data: List[BFloat16]
    var shape: List[Int]

    def rank(self) -> Int:
        return len(self.shape)

    def numel(self) -> Int:
        var n = 1
        for i in range(len(self.shape)):
            n *= self.shape[i]
        return n


@fieldwise_init
struct TensorI64(Movable):
    var data: List[Int64]
    var shape: List[Int]

    def rank(self) -> Int:
        return len(self.shape)

    def numel(self) -> Int:
        var n = 1
        for i in range(len(self.shape)):
            n *= self.shape[i]
        return n


def _read_i64_le(read bytes: List[UInt8], offset: Int) -> Int:
    var v: Int64 = 0
    for k in range(8):
        v = v | (Int64(bytes[offset + k]) << Int64(8 * k))
    return Int(v)


def _read_i32_le(read bytes: List[UInt8], offset: Int) -> Int:
    var v: Int32 = 0
    for k in range(4):
        v = v | (Int32(bytes[offset + k]) << Int32(8 * k))
    return Int(v)


def load_fp32(path: String) raises -> Tensor:
    var f = open(path, "r")
    var bytes = f.read_bytes()
    f.close()

    var rank = _read_i64_le(bytes, 0)
    var shape = List[Int]()
    var off = 8
    for _ in range(rank):
        shape.append(_read_i64_le(bytes, off))
        off += 8
    var tag = _read_i32_le(bytes, off)
    off += 4
    if tag != 0:
        raise Error("expected fp32 tag 0, got " + String(tag))

    var n = 1
    for i in range(len(shape)):
        n *= shape[i]

    # Reinterpret the raw byte payload as fp32. We go through List's pretyped
    # unsafe_ptr() so we never have to declare an UnsafePointer field.
    var src = bytes.unsafe_ptr().bitcast[Scalar[DType.float32]]()
    var elem_offset = off // 4  # header layout guarantees 4-byte alignment
    var data = List[Float32](capacity=n)
    for i in range(n):
        data.append(src[elem_offset + i])

    return Tensor(data^, shape^)


def load_bf16(path: String) raises -> TensorBF16:
    var f = open(path, "r")
    var bytes = f.read_bytes()
    f.close()

    var rank = _read_i64_le(bytes, 0)
    var shape = List[Int]()
    var off = 8
    for _ in range(rank):
        shape.append(_read_i64_le(bytes, off))
        off += 8
    var tag = _read_i32_le(bytes, off)
    off += 4
    if tag != 1:
        raise Error("expected bf16 tag 1, got " + String(tag))

    var n = 1
    for i in range(len(shape)):
        n *= shape[i]

    # Reinterpret raw uint16 payload as bf16 bits.
    var src = bytes.unsafe_ptr().bitcast[Scalar[DType.bfloat16]]()
    var elem_offset = off // 2  # 2 bytes per bf16; off is multiple of 2
    var data = List[BFloat16](capacity=n)
    for i in range(n):
        data.append(src[elem_offset + i])

    return TensorBF16(data^, shape^)


def save_fp32_1d(path: String, data: List[Float32]) raises:
    """Write a 1-D Float32 buffer in the same fixture format load_fp32 reads.
    Header: i64 rank=1; i64 shape[0]=len(data); i32 tag=0; payload.
    """
    var f = open(path, "w")
    var n = len(data)

    # Build header bytes.
    var hdr = List[UInt8](capacity=20)
    # rank = 1 (i64 LE)
    var r: Int64 = 1
    for k in range(8):
        hdr.append(UInt8(Int((r >> Int64(8 * k)) & 0xFF)))
    # shape[0] = n (i64 LE)
    var sh: Int64 = Int64(n)
    for k in range(8):
        hdr.append(UInt8(Int((sh >> Int64(8 * k)) & 0xFF)))
    # tag = 0 (i32 LE)
    var tg: Int32 = 0
    for k in range(4):
        hdr.append(UInt8(Int((tg >> Int32(8 * k)) & 0xFF)))
    # Write header as a single Span.
    f.write_bytes(Span(hdr))

    # Write payload in 1024-element chunks.
    var i = 0
    while i < n:
        var chunk_end = min(i + 1024, n)
        var buf = List[UInt8](capacity=(chunk_end - i) * 4)
        for j in range(i, chunk_end):
            var v = data[j]
            var p = UnsafePointer(to=v).bitcast[UInt32]()
            var bits = p[0]
            for k in range(4):
                buf.append(UInt8(Int((bits >> UInt32(8 * k)) & 0xFF)))
        f.write_bytes(Span(buf))
        i = chunk_end
    f.close()


def load_wav_with_sr(path: String) raises -> (Tensor, Int):
    """Read a mono PCM-16 WAV file and return (float32 samples in [-1,1], sample_rate)."""
    var t = load_wav(path)
    # Re-read the sample rate from the fmt chunk.
    var f = open(path, "r")
    var bytes = f.read_bytes()
    f.close()
    var n_total = len(bytes)
    var off = 12
    var sr: Int = 0
    while off + 8 <= n_total:
        var c0 = bytes[off]
        var c1 = bytes[off + 1]
        var c2 = bytes[off + 2]
        var c3 = bytes[off + 3]
        var chunk_size = _read_i32_le(bytes, off + 4)
        if c0 == 0x66 and c1 == 0x6D and c2 == 0x74 and c3 == 0x20:  # "fmt "
            sr = Int(_read_i32_le(bytes, off + 12))
            break
        off += 8 + chunk_size
    return (t^, sr)


def load_wav(path: String) raises -> Tensor:
    """Read a mono PCM-16 WAV file and return float32 samples in [-1, 1].

    Returns a 1-D Tensor; shape[0] = num samples. Stores the sample rate in
    shape[1] via convention is hacky — instead callers should use a separate
    helper if they need it. Here we assume standard 24 kHz; for now the only
    metadata we surface is the sample count.
    """
    var f = open(path, "r")
    var bytes = f.read_bytes()
    f.close()

    # Standard 44-byte canonical WAV header.
    # Minimum sanity check: starts with "RIFF" and "WAVE".
    if bytes[0] != 0x52 or bytes[1] != 0x49 or bytes[2] != 0x46 or bytes[3] != 0x46:
        raise Error("not a RIFF file: " + path)
    if bytes[8] != 0x57 or bytes[9] != 0x41 or bytes[10] != 0x56 or bytes[11] != 0x45:
        raise Error("not a WAVE file: " + path)

    # Find the "fmt " chunk (usually starts at offset 12) and the "data" chunk.
    var n_total = len(bytes)
    var off = 12
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
            num_channels = Int(_read_i16_le(bytes, off + 10))
            bits_per_sample = Int(_read_i16_le(bytes, off + 22))
        elif c0 == 0x64 and c1 == 0x61 and c2 == 0x74 and c3 == 0x61:  # "data"
            data_off = off + 8
            data_size = chunk_size
            break
        off += 8 + chunk_size

    if data_off == 0:
        raise Error("no data chunk in WAV: " + path)
    if bits_per_sample != 16:
        raise Error("only 16-bit PCM WAVs supported; got "
                    + String(bits_per_sample) + " bits")

    var n_samples = data_size // (2 * num_channels)
    var data = List[Float32](capacity=n_samples)
    var inv = 1.0 / 32768.0
    var i = 0
    while i < n_samples:
        # Mix down to mono if multi-channel: average channels.
        var sum: Float32 = 0.0
        for ch in range(num_channels):
            var pos = data_off + (i * num_channels + ch) * 2
            var s16: Int16 = Int16(_read_i16_le(bytes, pos))
            sum += Float32(s16) * Float32(inv)
        data.append(sum / Float32(num_channels))
        i += 1

    var shape = List[Int]()
    shape.append(n_samples)
    return Tensor(data^, shape^)


def _read_i16_le(read bytes: List[UInt8], offset: Int) -> Int:
    var v: Int16 = 0
    v = v | (Int16(bytes[offset]))
    v = v | (Int16(bytes[offset + 1]) << Int16(8))
    return Int(v)


def save_wav(path: String, samples: List[Float32], sample_rate: Int = 24000) raises:
    """Write a mono PCM-16 WAV file directly from Mojo.

    Standard 44-byte canonical WAV header followed by little-endian int16 samples.
    Samples are clamped to [-1, 1] then scaled by 32767.
    """
    var n = len(samples)
    var data_bytes = n * 2
    var riff_size = 36 + data_bytes

    var f = open(path, "w")
    var hdr = List[UInt8](capacity=44)

    # "RIFF"
    hdr.append(0x52); hdr.append(0x49); hdr.append(0x46); hdr.append(0x46)
    # chunk size = 36 + data_bytes
    var rs: Int32 = Int32(riff_size)
    for k in range(4):
        hdr.append(UInt8(Int((rs >> Int32(8 * k)) & 0xFF)))
    # "WAVE"
    hdr.append(0x57); hdr.append(0x41); hdr.append(0x56); hdr.append(0x45)
    # "fmt "
    hdr.append(0x66); hdr.append(0x6D); hdr.append(0x74); hdr.append(0x20)
    # subchunk1 size = 16
    hdr.append(16); hdr.append(0); hdr.append(0); hdr.append(0)
    # audio format = 1 (PCM)
    hdr.append(1); hdr.append(0)
    # num channels = 1
    hdr.append(1); hdr.append(0)
    # sample rate
    var sr: Int32 = Int32(sample_rate)
    for k in range(4):
        hdr.append(UInt8(Int((sr >> Int32(8 * k)) & 0xFF)))
    # byte rate = sample_rate * 1 * 2
    var br: Int32 = Int32(sample_rate * 2)
    for k in range(4):
        hdr.append(UInt8(Int((br >> Int32(8 * k)) & 0xFF)))
    # block align = 1 * 2
    hdr.append(2); hdr.append(0)
    # bits per sample = 16
    hdr.append(16); hdr.append(0)
    # "data"
    hdr.append(0x64); hdr.append(0x61); hdr.append(0x74); hdr.append(0x61)
    # data size
    var ds: Int32 = Int32(data_bytes)
    for k in range(4):
        hdr.append(UInt8(Int((ds >> Int32(8 * k)) & 0xFF)))

    f.write_bytes(Span(hdr))

    # Write samples in chunks.
    var i = 0
    while i < n:
        var chunk_end = min(i + 4096, n)
        var buf = List[UInt8](capacity=(chunk_end - i) * 2)
        for j in range(i, chunk_end):
            var v = samples[j]
            if v > 1.0: v = 1.0
            if v < -1.0: v = -1.0
            var pcm: Int = Int(v * 32767.0)
            if pcm > 32767: pcm = 32767
            if pcm < -32768: pcm = -32768
            # 2's-complement little-endian int16.
            var pcm_u = pcm & 0xFFFF
            buf.append(UInt8(pcm_u & 0xFF))
            buf.append(UInt8((pcm_u >> 8) & 0xFF))
        f.write_bytes(Span(buf))
        i = chunk_end
    f.close()


@fieldwise_init
struct TensorI32(Movable):
    var data: List[Int32]
    var shape: List[Int]


def load_i32(path: String) raises -> TensorI32:
    var f = open(path, "r")
    var bytes = f.read_bytes()
    f.close()

    var rank = _read_i64_le(bytes, 0)
    var shape = List[Int]()
    var off = 8
    for _ in range(rank):
        shape.append(_read_i64_le(bytes, off))
        off += 8
    var tag = _read_i32_le(bytes, off)
    off += 4
    if tag != 3:
        raise Error("expected i32 tag 3, got " + String(tag))

    var n = 1
    for i in range(len(shape)):
        n *= shape[i]

    var data = List[Int32](capacity=n)
    for i in range(n):
        data.append(Int32(_read_i32_le(bytes, off + i * 4)))

    return TensorI32(data^, shape^)


def load_i64(path: String) raises -> TensorI64:
    var f = open(path, "r")
    var bytes = f.read_bytes()
    f.close()

    var rank = _read_i64_le(bytes, 0)
    var shape = List[Int]()
    var off = 8
    for _ in range(rank):
        shape.append(_read_i64_le(bytes, off))
        off += 8
    var tag = _read_i32_le(bytes, off)
    off += 4
    if tag != 2:
        raise Error("expected i64 tag 2, got " + String(tag))

    var n = 1
    for i in range(len(shape)):
        n *= shape[i]

    # Header is 8 + 8*rank + 4 bytes; tag bytes break 8-byte alignment so we
    # read each i64 from raw bytes rather than via a bitcast pointer.
    var data = List[Int64](capacity=n)
    for i in range(n):
        data.append(Int64(_read_i64_le(bytes, off + i * 8)))

    return TensorI64(data^, shape^)
