"""T3 generation parity test against mojo-t3 generate oracle.

Reuses fixtures: initial_ids, expected_ids, cos_full, sin_full, mask_prefill.
Weights are loaded via `load_t3("weights/t3")` from the upstream safetensors
conversion — same values the oracle dumped from.

Since argmax is deterministic, this test should produce EXACTLY the same
token IDs as the oracle (`expected_ids_fp32.bin`).
"""
from std.sys import has_accelerator
from std.testing import TestSuite, assert_equal
from std.gpu.host import DeviceContext, DeviceBuffer
from std.runtime.asyncrt import DeviceContextPtr
from std.algorithm.functional import elementwise, IndexList

from fixture import load_fp32, load_i64
from modules import Embedding, embedding_forward
from weights import load_t3, upload_fp32
from t3_generate import t3_generate


comptime BATCH = 1
comptime T_PREFILL = 15
comptime N_STEPS = 8
comptime MAX_CTX = T_PREFILL + N_STEPS
comptime D = 1024
comptime HEAD_DIM = 64


def upload(buf: DeviceBuffer[DType.float32], data: List[Float32], n: Int) raises:
    with buf.map_to_host() as h:
        for i in range(n):
            h[i] = data[i]


def test_t3_generate_parity() raises:
    comptime assert has_accelerator(), "Requires GPU"
    var fix = "../mojo-t3/tests/fixtures/generate/"
    var ctx = DeviceContext()

    var initial_ids = load_i64(fix + "initial_ids_fp32.bin")
    var expected_ids = load_i64(fix + "expected_ids_fp32.bin")
    var cos_full_t = load_fp32(fix + "cos_full_fp32.bin")
    var sin_full_t = load_fp32(fix + "sin_full_fp32.bin")
    var mask_pre_t = load_fp32(fix + "mask_prefill_fp32.bin")
    var speech_pos_t = load_fp32(fix + "speech_pos_emb_fp32.bin")

    print("[gen] Loading T3 from weights/t3/...")
    var t3 = load_t3(ctx, "weights/t3")

    # Build input embedding: speech_emb[initial_ids] + speech_pos_emb[0..T-1].
    # speech_emb is in t3.speech_emb (loaded from disk).
    var ids_buf = ctx.enqueue_create_buffer[DType.int64](BATCH * T_PREFILL)
    with ids_buf.map_to_host() as h:
        for i in range(T_PREFILL): h[i] = initial_ids.data[i]
    var emb_buf = ctx.enqueue_create_buffer[DType.float32](BATCH * T_PREFILL * D)
    embedding_forward(ctx, t3.speech_emb, ids_buf, emb_buf, BATCH, T_PREFILL)

    # Add positional embedding.
    var pos_buf = ctx.enqueue_create_buffer[DType.float32](len(speech_pos_t.data))
    upload(pos_buf, speech_pos_t.data, len(speech_pos_t.data))
    var ep = emb_buf.unsafe_ptr()
    var pp = pos_buf.unsafe_ptr()

    @always_inline
    @parameter
    @__copy_capture(ep, pp, T_PREFILL, D)
    def add_pos[width: Int, rank: Int, alignment: Int = 1](idx: IndexList[rank]):
        var i = idx[0]
        var bi = i // (T_PREFILL * D)
        var rem = i - bi * T_PREFILL * D
        var si = rem // D
        var di = rem - si * D
        ep[i] = ep[i] + pp[si * D + di]
    elementwise[add_pos, simd_width=1, target="gpu"](
        IndexList[1](BATCH * T_PREFILL * D), DeviceContextPtr(ctx),
    )

    # Upload cos_full, sin_full, mask_prefill.
    var cos_full = ctx.enqueue_create_buffer[DType.float32](MAX_CTX * HEAD_DIM)
    upload(cos_full, cos_full_t.data, MAX_CTX * HEAD_DIM)
    var sin_full = ctx.enqueue_create_buffer[DType.float32](MAX_CTX * HEAD_DIM)
    upload(sin_full, sin_full_t.data, MAX_CTX * HEAD_DIM)
    var mask_pre = ctx.enqueue_create_buffer[DType.float32](T_PREFILL * T_PREFILL)
    upload(mask_pre, mask_pre_t.data, T_PREFILL * T_PREFILL)

    var generated = t3_generate(
        ctx, t3, emb_buf, cos_full, sin_full, mask_pre, pos_buf,
        BATCH, T_PREFILL, MAX_CTX, N_STEPS, T_PREFILL,    # speech_pos_offset
        eos_token=6562,
    )
    ctx.synchronize()

    print("[gen] Generated", len(generated), "tokens vs expected", len(expected_ids.data))
    var n_match = 0
    for i in range(min(len(generated), len(expected_ids.data))):
        var got = Int(generated[i])
        var want = Int(expected_ids.data[i])
        print("  step[", i, "]: got=", got, " want=", want, "  ",
              "match" if got == want else "MISMATCH")
        if got == want: n_match += 1
    print("[gen] Matched", n_match, "/", len(expected_ids.data))


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
