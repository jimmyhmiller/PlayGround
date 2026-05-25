"""
Embedding and LM-head support kernels for T3.

  embed_lookup_kernel:   ids (B,S) + table (V,D) → x (B,S,D)
  add_pos_emb_kernel:    x (B,S,D) += pos_table[base_pos + s] (B,S,D)
  argmax_kernel:         logits (B,S,V) → ids (B,S) int64 (lane-0 write)

These are intentionally simple: T3 uses standard nn.Embedding / nn.Linear
patterns and a temperature=0 (argmax) sampling path is enough for parity
testing. Real sampling (top-p/min-p/rep-penalty) comes later.
"""

from std.gpu import barrier, block_idx, thread_idx, lane_id, WARP_SIZE
from std.gpu.primitives import warp
from std.gpu.memory import AddressSpace
from layout import TileTensor, TensorLayout, row_major, stack_allocation


def embed_lookup_kernel[
    dtype: DType,
    IdLayout: TensorLayout,
    TableLayout: TensorLayout,
    OutLayout: TensorLayout,
    HIDDEN: Int,
](
    output: TileTensor[dtype, OutLayout, MutAnyOrigin],          # (B, S, D)
    ids: TileTensor[DType.int64, IdLayout, MutAnyOrigin],         # (B, S)
    table: TileTensor[dtype, TableLayout, MutAnyOrigin],          # (V, D)
    batch: Int,
    seq: Int,
):
    """out[b,s,d] = table[ids[b,s], d].

    Launch: grid = B*S, block_dim = HIDDEN.
    """
    comptime assert ids.flat_rank == 2
    comptime assert table.flat_rank == 2
    comptime assert output.flat_rank == 3

    var bid = block_idx.x
    var d = thread_idx.x

    var s = bid % seq
    var b = bid // seq

    var token_id_i64 = rebind[Scalar[DType.int64]](ids[b, s])
    var token_id = Int(token_id_i64)

    var v = rebind[Scalar[dtype]](table[token_id, d])
    output[b, s, d] = rebind[output.ElementType](v)


def add_pos_emb_kernel[
    dtype: DType,
    InLayout: TensorLayout,
    PosLayout: TensorLayout,
    OutLayout: TensorLayout,
    HIDDEN: Int,
](
    output: TileTensor[dtype, OutLayout, MutAnyOrigin],          # (B, S, D)
    inp: TileTensor[dtype, InLayout, MutAnyOrigin],               # (B, S, D)
    pos_table: TileTensor[dtype, PosLayout, MutAnyOrigin],        # (P, D)
    batch: Int,
    seq: Int,
    base_pos: Int,
):
    """out[b,s,d] = inp[b,s,d] + pos_table[base_pos + s, d].

    Launch: grid = B*S, block_dim = HIDDEN.
    """
    comptime assert inp.flat_rank == 3
    comptime assert pos_table.flat_rank == 2
    comptime assert output.flat_rank == 3

    var bid = block_idx.x
    var d = thread_idx.x

    var s = bid % seq
    var b = bid // seq

    var xv = rebind[Scalar[dtype]](inp[b, s, d]).cast[DType.float32]()
    var pv = rebind[Scalar[dtype]](pos_table[base_pos + s, d]).cast[DType.float32]()
    var v = xv + pv
    output[b, s, d] = rebind[output.ElementType](v.cast[dtype]())


def argmax_kernel[
    dtype: DType,
    LogitsLayout: TensorLayout,
    OutLayout: TensorLayout,
    VOCAB: Int,
    BLOCK: Int,
](
    output: TileTensor[DType.int64, OutLayout, MutAnyOrigin],     # (B, S)
    logits: TileTensor[dtype, LogitsLayout, MutAnyOrigin],        # (B, S, V)
    batch: Int,
    seq: Int,
):
    """ids[b,s] = argmax_v logits[b,s,v].

    Launch: grid = B*S, block_dim = BLOCK. Threads cooperatively reduce over
    VOCAB elements using a warp-level argmax. We encode each lane's
    (value, index) as a pair and reduce by picking the larger value (ties
    broken by smaller index).
    """
    comptime assert logits.flat_rank == 3
    comptime assert output.flat_rank == 2
    comptime NUM_WARPS = BLOCK // WARP_SIZE

    var bid = block_idx.x
    var tid = thread_idx.x

    var s = bid % seq
    var b = bid // seq

    # Stage 1: per-thread best.
    var local_val: Float32 = -3.4e38
    var local_idx: Int = -1
    var col = tid
    while col < VOCAB:
        var v = rebind[Scalar[dtype]](logits[b, s, col]).cast[DType.float32]()
        if v > local_val:
            local_val = v
            local_idx = col
        col += BLOCK

    # Stage 2: warp-level argmax via shared memory.
    # Reduce within a warp using shuffle_down: pair up values, pick the
    # winner, broadcast. Mojo's warp.max only handles values, not paired
    # (value, index), so we do it explicitly.
    var step: UInt32 = UInt32(WARP_SIZE // 2)
    while step >= 1:
        var other_val = warp.shuffle_down(local_val, step)
        var other_idx_u32 = warp.shuffle_down(UInt32(local_idx), step)
        var other_idx = Int(other_idx_u32)
        if other_val > local_val:
            local_val = other_val
            local_idx = other_idx
        elif other_val == local_val and other_idx >= 0 and (local_idx < 0 or other_idx < local_idx):
            # Tie-break: prefer the lower index (matches torch.argmax).
            local_idx = other_idx
        step = step // 2

    # Stage 3: cross-warp reduce via shared memory.
    var val_sm = stack_allocation[
        DType.float32, address_space=AddressSpace.SHARED
    ](row_major[NUM_WARPS]())
    var idx_sm = stack_allocation[
        DType.int64, address_space=AddressSpace.SHARED
    ](row_major[NUM_WARPS]())
    if lane_id() == 0:
        val_sm[tid // WARP_SIZE] = local_val
        idx_sm[tid // WARP_SIZE] = Int64(local_idx)
    barrier()

    if tid < WARP_SIZE:
        var v: Float32 = -3.4e38
        var idx: Int = -1
        if tid < NUM_WARPS:
            v = rebind[Scalar[DType.float32]](val_sm[tid])
            idx = Int(rebind[Scalar[DType.int64]](idx_sm[tid]))
        step = UInt32(WARP_SIZE // 2)
        while step >= 1:
            var ov = warp.shuffle_down(v, step)
            var oi = Int(warp.shuffle_down(UInt32(idx), step))
            if ov > v:
                v = ov
                idx = oi
            elif ov == v and oi >= 0 and (idx < 0 or oi < idx):
                idx = oi
            step = step // 2
        if tid == 0:
            output[b, s] = Int64(idx)
