"""Sequence-dim concat kernel for T3CondEnc and friends.

Stacks A (B, T_a, D) | B (B, T_b, D) | C (B, T_c, D) along T axis.
Output is (B, T_a + T_b + T_c, D).
"""
from std.gpu import block_idx, thread_idx
from layout import TileTensor, TensorLayout


def concat3_t_kernel[
    dtype: DType,
    ALayout: TensorLayout,
    BLayout: TensorLayout,
    CLayout: TensorLayout,
    OutLayout: TensorLayout,
    T_A: Int, T_B: Int, T_C: Int,
    BLOCK: Int,
](
    output: TileTensor[dtype, OutLayout, MutAnyOrigin],   # (B, T_a+T_b+T_c, D)
    a: TileTensor[dtype, ALayout, MutAnyOrigin],          # (B, T_a, D)
    b: TileTensor[dtype, BLayout, MutAnyOrigin],          # (B, T_b, D)
    c: TileTensor[dtype, CLayout, MutAnyOrigin],          # (B, T_c, D)
    batch: Int, d: Int,
):
    """output[bi, t, di] = a[bi, t, di]                if t < T_A
                          b[bi, t-T_A, di]            elif t < T_A+T_B
                          c[bi, t-T_A-T_B, di]        else.

    Launch: grid = B * (T_A+T_B+T_C), block_dim = BLOCK over D.
    """
    comptime assert a.flat_rank == 3
    comptime assert b.flat_rank == 3
    comptime assert c.flat_rank == 3
    comptime assert output.flat_rank == 3
    comptime T_TOTAL = T_A + T_B + T_C
    var bid = block_idx.x
    var tid = thread_idx.x
    var t = bid % T_TOTAL
    var bi = bid // T_TOTAL
    var di = tid
    while di < d:
        var v: Float32 = 0.0
        if t < T_A:
            v = rebind[Scalar[dtype]](a[bi, t, di]).cast[DType.float32]()
        elif t < T_A + T_B:
            v = rebind[Scalar[dtype]](b[bi, t - T_A, di]).cast[DType.float32]()
        else:
            v = rebind[Scalar[dtype]](c[bi, t - T_A - T_B, di]).cast[DType.float32]()
        output[bi, t, di] = rebind[output.ElementType](v.cast[dtype]())
        di += BLOCK
