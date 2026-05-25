"""op_write_wav: Write a host-side Buffer of float32 mono samples to a 16-bit PCM WAV.

Symmetric counterpart of op_load_wav. No state — just one stateless function.
"""

from std.os import abort
from std.python import PythonObject
from std.python.bindings import PythonModuleBuilder


def write_wav(
    in_buffer: PythonObject,
    n_obj: PythonObject,
    sample_rate_obj: PythonObject,
    path_obj: PythonObject,
) raises -> PythonObject:
    """Write `n` float32 samples from in_buffer to `path` as mono PCM-16 WAV."""
    var path = String(py=path_obj)
    var n = Int(py=n_obj)
    var sample_rate = Int(py=sample_rate_obj)
    var addr = Int(py=in_buffer._data_ptr())
    var src = UnsafePointer[Float32, MutAnyOrigin](unsafe_from_address=addr)

    var data_bytes = n * 2
    var riff_size = 36 + data_bytes

    var f = open(path, "w")
    var hdr = List[UInt8](capacity=44)
    # "RIFF"
    hdr.append(0x52); hdr.append(0x49); hdr.append(0x46); hdr.append(0x46)
    var rs: Int32 = Int32(riff_size)
    for k in range(4):
        hdr.append(UInt8(Int((rs >> Int32(8 * k)) & 0xFF)))
    # "WAVE"
    hdr.append(0x57); hdr.append(0x41); hdr.append(0x56); hdr.append(0x45)
    # "fmt "
    hdr.append(0x66); hdr.append(0x6D); hdr.append(0x74); hdr.append(0x20)
    # subchunk1 size = 16
    hdr.append(16); hdr.append(0); hdr.append(0); hdr.append(0)
    # PCM, mono
    hdr.append(1); hdr.append(0)
    hdr.append(1); hdr.append(0)
    # sample rate
    var sr: Int32 = Int32(sample_rate)
    for k in range(4):
        hdr.append(UInt8(Int((sr >> Int32(8 * k)) & 0xFF)))
    # byte rate = sample_rate * 2
    var br: Int32 = Int32(sample_rate * 2)
    for k in range(4):
        hdr.append(UInt8(Int((br >> Int32(8 * k)) & 0xFF)))
    # block align = 2, bits = 16
    hdr.append(2); hdr.append(0)
    hdr.append(16); hdr.append(0)
    # "data"
    hdr.append(0x64); hdr.append(0x61); hdr.append(0x74); hdr.append(0x61)
    var ds: Int32 = Int32(data_bytes)
    for k in range(4):
        hdr.append(UInt8(Int((ds >> Int32(8 * k)) & 0xFF)))
    f.write_bytes(Span(hdr))

    var i = 0
    while i < n:
        var chunk_end = min(i + 4096, n)
        var buf = List[UInt8](capacity=(chunk_end - i) * 2)
        for j in range(i, chunk_end):
            var v: Float32 = src[j]
            if v > 1.0: v = 1.0
            if v < -1.0: v = -1.0
            var pcm: Int = Int(v * 32767.0)
            if pcm > 32767: pcm = 32767
            if pcm < -32768: pcm = -32768
            var pcm_u = pcm & 0xFFFF
            buf.append(UInt8(pcm_u & 0xFF))
            buf.append(UInt8((pcm_u >> 8) & 0xFF))
        f.write_bytes(Span(buf))
        i = chunk_end
    f.close()
    return PythonObject(None)


@export
def PyInit_op_write_wav() -> PythonObject:
    try:
        var b = PythonModuleBuilder("op_write_wav")
        b.def_function[write_wav](
            "write_wav",
            docstring="write_wav(buffer, n_samples, sample_rate, path) -> None",
        )
        return b.finalize()
    except e:
        abort(String("failed to create op_write_wav module: ", e))
