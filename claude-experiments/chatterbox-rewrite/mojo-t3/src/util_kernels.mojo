"""
Small composition utilities used to wire layers together:
  add_kernel               — pointwise out = a + b
  bshd_to_bhsd_kernel      — reshape (B, S, H*D) → (B, H, S, D) via copy
  bhsd_to_bshd_kernel      — reshape (B, H, S, D) → (B, S, H*D) via copy
  print_rss(label)         — host-side: print current /proc/self/status VmRSS
"""

from std.gpu import block_idx, thread_idx
from std.io.file import open
from layout import TileTensor, TensorLayout


def print_rss(label: String) raises:
    """Print VmRSS and VmSize (kB) from /proc/self/status."""
    var f = open("/proc/self/status", "r")
    var content = f.read()
    f.close()
    var rss_kb = String("?")
    var vsz_kb = String("?")
    # Mojo's splitlines() also splits on tabs, which breaks "VmSize:\t123 kB"
    # apart into two records. Split explicitly on '\n' (byte 10) instead.
    var n_all = content.byte_length()
    var line_start = 0
    var i = 0
    while i <= n_all:
        var at_end = i == n_all
        var is_nl = (not at_end) and Int(content.as_bytes()[i]) == 10
        if at_end or is_nl:
            if i > line_start:
                var line_len = i - line_start
                # Read bytes for this line.
                var bs = content.as_bytes()
                # Compare prefix bytes against "VmRSS:" / "VmSize:".
                var is_rss = (
                    line_len >= 6
                    and Int(bs[line_start + 0]) == Int(ord("V"))
                    and Int(bs[line_start + 1]) == Int(ord("m"))
                    and Int(bs[line_start + 2]) == Int(ord("R"))
                    and Int(bs[line_start + 3]) == Int(ord("S"))
                    and Int(bs[line_start + 4]) == Int(ord("S"))
                    and Int(bs[line_start + 5]) == Int(ord(":"))
                )
                var is_vsz = (
                    line_len >= 7
                    and Int(bs[line_start + 0]) == Int(ord("V"))
                    and Int(bs[line_start + 1]) == Int(ord("m"))
                    and Int(bs[line_start + 2]) == Int(ord("S"))
                    and Int(bs[line_start + 3]) == Int(ord("i"))
                    and Int(bs[line_start + 4]) == Int(ord("z"))
                    and Int(bs[line_start + 5]) == Int(ord("e"))
                    and Int(bs[line_start + 6]) == Int(ord(":"))
                )
                if is_rss or is_vsz:
                    var j = line_start + (7 if is_vsz else 6)
                    # Skip whitespace (space, tab).
                    while j < i:
                        var b = Int(bs[j])
                        if b != 32 and b != 9:
                            break
                        j += 1
                    # Collect digits.
                    var num = String("")
                    while j < i:
                        var b = Int(bs[j])
                        if b < 48 or b > 57:
                            break
                        num += chr(b)
                        j += 1
                    if is_rss:
                        rss_kb = num
                    else:
                        vsz_kb = num
            line_start = i + 1
        i += 1
    print("[mem]", label, "| VmSize=", vsz_kb, "kB | VmRSS=", rss_kb, "kB")


def add_kernel[
    dtype: DType,
    ALayout: TensorLayout,
    BLayout: TensorLayout,
    OutLayout: TensorLayout,
    BLOCK: Int,
](
    output: TileTensor[dtype, OutLayout, MutAnyOrigin],
    a: TileTensor[dtype, ALayout, MutAnyOrigin],
    b: TileTensor[dtype, BLayout, MutAnyOrigin],
    n_elems: Int,
):
    """Pointwise out[i] = a[i] + b[i] over a 2D (rows, cols) buffer.

    Inputs and output must share the same 2D layout. Computation in fp32;
    cast back at write time.
    """
    comptime assert a.flat_rank == 2
    comptime assert b.flat_rank == 2
    comptime assert output.flat_rank == 2

    var idx = block_idx.x * BLOCK + thread_idx.x
    if idx >= n_elems:
        return

    var cols = Int(a.dim[1]())
    var r = idx // cols
    var c = idx % cols

    var av = rebind[Scalar[dtype]](a[r, c]).cast[DType.float32]()
    var bv = rebind[Scalar[dtype]](b[r, c]).cast[DType.float32]()
    var s = av + bv
    output[r, c] = rebind[output.ElementType](s.cast[dtype]())


def bshd_to_bhsd_kernel[
    dtype: DType,
    InLayout: TensorLayout,
    OutLayout: TensorLayout,
    BATCH: Int, SEQ: Int, N_HEADS: Int, HEAD_DIM: Int,
](
    output: TileTensor[dtype, OutLayout, MutAnyOrigin],   # (B, H, S, D)
    inp: TileTensor[dtype, InLayout, MutAnyOrigin],        # (B, S, H, D)
):
    """Permute axes (B, S, H, D) → (B, H, S, D)."""
    comptime assert inp.flat_rank == 4
    comptime assert output.flat_rank == 4

    # Launch: grid = (B*H*S), block = D. Each thread copies one element.
    var bid = block_idx.x
    var d = thread_idx.x

    var s = bid % SEQ
    var h = (bid // SEQ) % N_HEADS
    var b = bid // (SEQ * N_HEADS)

    var v = rebind[Scalar[dtype]](inp[b, s, h, d])
    output[b, h, s, d] = rebind[output.ElementType](v)


def bhsd_to_bshd_kernel[
    dtype: DType,
    InLayout: TensorLayout,
    OutLayout: TensorLayout,
    BATCH: Int, SEQ: Int, N_HEADS: Int, HEAD_DIM: Int,
](
    output: TileTensor[dtype, OutLayout, MutAnyOrigin],   # (B, S, H, D)
    inp: TileTensor[dtype, InLayout, MutAnyOrigin],        # (B, H, S, D)
):
    """Permute axes (B, H, S, D) → (B, S, H, D)."""
    comptime assert inp.flat_rank == 4
    comptime assert output.flat_rank == 4

    var bid = block_idx.x
    var d = thread_idx.x

    # Same flat indexing as bshd_to_bhsd but read/write swapped.
    var s = bid % SEQ
    var h = (bid // SEQ) % N_HEADS
    var b = bid // (SEQ * N_HEADS)

    var v = rebind[Scalar[dtype]](inp[b, h, s, d])
    output[b, s, h, d] = rebind[output.ElementType](v)
