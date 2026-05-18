"""Smoke test: load real upstream T3CondEnc weights.

Reuses speech_emb from the parent T3 (cond_prompt tokens share its table).
"""
from std.sys import has_accelerator
from std.testing import TestSuite, assert_true
from std.gpu.host import DeviceContext

from weights import load_t3, load_t3_cond_enc


def test_load_cond_enc_real_weights() raises:
    comptime assert has_accelerator(), "Requires GPU"
    var ctx = DeviceContext()
    print("[cond_enc] loading T3 first to get shared speech_emb...")
    var t3 = load_t3(ctx, "weights/t3")
    print("[cond_enc] loading cond_enc from weights/t3/cond_enc/...")
    var cond = load_t3_cond_enc(ctx, "weights/t3", t3.speech_emb)
    ctx.synchronize()
    print("[cond_enc] d_model=", cond.d_model,
          " n_queries=", cond.perceiver.n_queries,
          " speaker_embed=", cond.speaker_embed_size)
    assert_true(cond.d_model == 1024, "d_model should be 1024")
    assert_true(cond.perceiver.n_queries == 32, "n_queries should be 32")
    assert_true(cond.speaker_embed_size == 256, "speaker_embed_size should be 256")


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
