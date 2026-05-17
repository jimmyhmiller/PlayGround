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
