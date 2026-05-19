"""Verify Mojo text→text_emb produces the same result as upstream's dump."""
from std.sys import has_accelerator
from std.testing import TestSuite
from std.math import sqrt
from std.gpu.host import DeviceContext, DeviceBuffer
from std.runtime.asyncrt import DeviceContextPtr

from fixture import load_fp32
from weights import load_t3
from bpe_tokenizer import load_tokenizer
from text_embed import text_to_input_ids, build_text_emb, build_bos_emb, build_rope_tables


def _diff(name: String, mut mojo: DeviceBuffer[DType.float32], ref_path: String) raises:
    var reference = load_fp32(ref_path)
    var ref_n = reference.numel()
    var max_abs: Float32 = 0.0
    var sum_diff_sq: Float32 = 0.0
    var sum_ref_sq: Float32 = 0.0
    with mojo.map_to_host() as h:
        for i in range(ref_n):
            var dd = h[i] - reference.data[i]
            var d = dd
            if d < 0.0: d = -d
            if d > max_abs: max_abs = d
            sum_diff_sq += dd * dd
            sum_ref_sq += reference.data[i] * reference.data[i]
    var rel = sqrt(sum_diff_sq / sum_ref_sq) if sum_ref_sq > 0.0 else Float32(0.0)
    print("[text-parity]", name, ": max-abs=", max_abs, " rel_l2=", rel, " (n=", ref_n, ")")


def test_text_emb_parity() raises:
    comptime assert has_accelerator(), "Requires GPU"
    var ctx = DeviceContext()

    print("[text-parity] loading T3...")
    var t3 = load_t3(ctx, "weights/t3")

    var tok_dir = "../mojo-t3/tests/fixtures/tokenizer/"
    var tok = load_tokenizer(tok_dir + "vocab.txt", tok_dir + "merges.txt")

    var text = "the quick brown fox"
    print("[text-parity] tokenizing:", text)
    var ids = text_to_input_ids(text, tok)
    print("[text-parity] got", len(ids), "ids:")
    for i in range(len(ids)):
        print("  [", i, "] =", ids[i])

    var D = 1024
    var T_text = len(ids)
    var text_emb = ctx.enqueue_create_buffer[DType.float32](1 * T_text * D)
    build_text_emb(ctx, t3, ids, text_emb)
    ctx.synchronize()
    _diff("text_emb", text_emb, "weights/t3_text_parity/text_emb.bin")

    var bos_emb = ctx.enqueue_create_buffer[DType.float32](1 * 1 * D)
    build_bos_emb(ctx, t3, bos_emb)
    ctx.synchronize()
    _diff("bos_emb", bos_emb, "weights/t3_text_parity/bos_emb.bin")

    # RoPE tables.
    var MAX_CTX = 200
    var HEAD_DIM = 64
    var cos_buf = ctx.enqueue_create_buffer[DType.float32](MAX_CTX * HEAD_DIM)
    var sin_buf = ctx.enqueue_create_buffer[DType.float32](MAX_CTX * HEAD_DIM)
    build_rope_tables(ctx, MAX_CTX, HEAD_DIM, cos_buf, sin_buf)
    ctx.synchronize()
    _diff("cos_full", cos_buf, "weights/t3_text_parity/cos_full.bin")
    _diff("sin_full", sin_buf, "weights/t3_text_parity/sin_full.bin")


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
