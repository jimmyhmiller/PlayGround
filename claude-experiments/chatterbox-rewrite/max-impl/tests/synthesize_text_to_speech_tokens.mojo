"""Run T3 generation in Mojo on a real text prompt with real cond_emb.

Pipeline:
  text "the quick brown fox" → BPE tokenize → ids
  load cond_emb (from upstream, dumped via FCM/CAMPPlus chain)
  load text_emb (already includes pos_emb)
  load bos_emb (speech_emb[start] + speech_pos[0])
  prefix = concat([cond_emb, text_emb, bos_emb])  # (1, 48, 1024)
  Mojo t3_generate(prefix, ...) → speech tokens

This is the first Mojo run with a real-text-derived prefix. cond_emb still
comes from upstream torch (the FCM 2D head isn't ported yet); everything else
runs in pure Mojo on real upstream weights.
"""
from std.sys import has_accelerator
from std.testing import TestSuite
from std.gpu.host import DeviceContext, DeviceBuffer
from std.runtime.asyncrt import DeviceContextPtr
from std.algorithm.functional import elementwise, IndexList

from fixture import load_fp32, load_i64
from weights import load_t3, upload_fp32
from t3_generate import t3_generate


comptime B = 1
comptime T_COND = 34
comptime T_TEXT = 13
comptime T_BOS = 1
comptime T_PREFIX = T_COND + T_TEXT + T_BOS    # 48
comptime D = 1024
comptime HEAD_DIM = 64
comptime MAX_CTX = 200
comptime MAX_NEW = 100
comptime EOS = 6562   # speech eos token (per upstream T3 hp.stop_speech_token)


def test_text_to_speech_tokens() raises:
    comptime assert has_accelerator(), "Requires GPU"
    var ctx = DeviceContext()
    var fix = "weights/t3_text_parity/"

    print("[t2s] loading T3 from weights/t3/...")
    var t3 = load_t3(ctx, "weights/t3")

    # Load cond_emb + text_emb + bos_emb.
    var cond_emb = upload_fp32(ctx, fix + "cond_emb.bin")
    var text_emb = upload_fp32(ctx, fix + "text_emb.bin")
    var bos_emb = upload_fp32(ctx, fix + "bos_emb.bin")

    # Concat into prefix (1, 48, 1024).
    var prefix = ctx.enqueue_create_buffer[DType.float32](B * T_PREFIX * D)
    var ce = cond_emb.unsafe_ptr()
    var te = text_emb.unsafe_ptr()
    var be = bos_emb.unsafe_ptr()
    var px = prefix.unsafe_ptr()

    @always_inline
    @parameter
    @__copy_capture(ce, te, be, px)
    def cat_func[width: Int, rank: Int, alignment: Int = 1](idx: IndexList[rank]):
        var i = idx[0]
        var bi = i // (T_PREFIX * D)
        var rem = i - bi * T_PREFIX * D
        var ti = rem // D
        var di = rem - ti * D
        if ti < T_COND:
            px[i] = ce[bi * T_COND * D + ti * D + di]
        elif ti < T_COND + T_TEXT:
            var src_t = ti - T_COND
            px[i] = te[bi * T_TEXT * D + src_t * D + di]
        else:
            var src_t = ti - T_COND - T_TEXT
            px[i] = be[bi * T_BOS * D + src_t * D + di]
    elementwise[cat_func, simd_width=1, target="gpu"](
        IndexList[1](B * T_PREFIX * D), DeviceContextPtr(ctx),
    )

    # cos/sin full RoPE tables.
    var cos_full = upload_fp32(ctx, fix + "cos_full.bin")
    var sin_full = upload_fp32(ctx, fix + "sin_full.bin")

    # Build causal prefill mask: (T_PREFIX, T_PREFIX) with -inf above diagonal.
    var mask = ctx.enqueue_create_buffer[DType.float32](T_PREFIX * T_PREFIX)
    with mask.map_to_host() as h:
        for r in range(T_PREFIX):
            for c in range(T_PREFIX):
                if c > r:
                    h[r * T_PREFIX + c] = -1.0e30
                else:
                    h[r * T_PREFIX + c] = 0.0

    # speech_pos_emb table (4100, 1024) — used when generating speech tokens.
    var speech_pos = upload_fp32(ctx, fix + "speech_pos_emb_full.bin")

    print("[t2s] running T3 generate (prefix=", T_PREFIX, " max_new=", MAX_NEW, ")...")
    # speech_pos_offset: when generating the FIRST speech token (BOS is already at
    # position T_PREFIX-1 within the prefix and has its pos_emb already included),
    # the NEXT speech token (step 0 of decode) should use speech_pos[1].
    # So speech_pos_offset = 1.
    var generated = t3_generate(
        ctx, t3, prefix, cos_full, sin_full, mask, speech_pos,
        B, T_PREFIX, MAX_CTX, MAX_NEW, speech_pos_offset=1, eos_token=EOS,
    )
    ctx.synchronize()
    print("[t2s] generated", len(generated), "speech tokens:")
    for i in range(len(generated)):
        print("  [", i, "] =", Int(generated[i]))


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
