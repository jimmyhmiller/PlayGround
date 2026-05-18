"""Convert upstream Chatterbox safetensors checkpoints into max-impl binaries.

Reads upstream weight names from `chatterbox/ckpt/*.safetensors` files and
writes them into the fixture format that `src/fixture.mojo` reads:
  i64 rank | i64[rank] shape | i32 tag=0 | fp32 payload (little-endian)

Reuses non-NN helpers only (file I/O + numpy reshapes). All neural-net code
remains in Mojo.

Usage:
  python scripts/convert_weights.py \\
    --ve <voice_encoder.safetensors> \\
    --t3 <t3.safetensors> \\
    --s3gen <s3gen.safetensors> \\
    --s3t <s3tokenizer.pt> \\
    --out tests/fixtures/weights/
"""
import os, struct, sys, argparse
import numpy as np

try:
    from safetensors import safe_open
except ImportError:
    print("Need: pip install safetensors", file=sys.stderr)
    sys.exit(1)


# ----------------------------------------------------------------------------
# I/O helpers
# ----------------------------------------------------------------------------

def write_tensor(path, arr):
    arr = np.ascontiguousarray(np.asarray(arr, dtype=np.float32))
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "wb") as f:
        f.write(struct.pack("<q", len(arr.shape)))
        f.write(struct.pack(f"<{len(arr.shape)}q", *arr.shape))
        f.write(struct.pack("<i", 0))
        f.write(arr.tobytes())


# ----------------------------------------------------------------------------
# Weight-name mappings (upstream → max-impl)
# ----------------------------------------------------------------------------

def voice_encoder_map(k):
    """VoiceEncoder weights — PyTorch nn.LSTM stack + nn.Linear proj."""
    if k.startswith("lstm.weight_ih_l") or k.startswith("lstm.weight_hh_l"):
        return f"ve/{k.split('.')[-1]}"
    elif k.startswith("lstm.bias_ih_l") or k.startswith("lstm.bias_hh_l"):
        return f"ve/{k.split('.')[-1]}"
    elif k == "proj.weight":
        return "ve/proj_w"
    elif k == "proj.bias":
        return "ve/proj_b"
    return None


def t3_map(k):
    """T3 Llama-30L backbone weights.

    Upstream uses HF Llama naming: `tfmr.layers.{L}.self_attn.q_proj.weight` etc.
    We map to layer{L}/qw, kw, vw, ow, gate_w, up_w, down_w, in_norm, post_norm.
    """
    # Embedding tables.
    if k == "text_emb.weight":
        return "t3/text_emb_w"
    if k == "text_pos_emb.emb.weight":
        return "t3/text_pos_w"
    if k == "speech_emb.weight":
        return "t3/speech_emb_w"
    if k == "speech_pos_emb.emb.weight":
        return "t3/speech_pos_w"
    if k == "speech_head.weight":
        return "t3/speech_head_w"
    if k == "tfmr.norm.weight":
        return "t3/final_norm_w"
    # Layer weights.
    if k.startswith("tfmr.layers."):
        parts = k.split(".")
        L = parts[2]
        comp = ".".join(parts[3:])
        m = {
            "input_layernorm.weight": "in_norm",
            "post_attention_layernorm.weight": "post_norm",
            "self_attn.q_proj.weight": "qw",
            "self_attn.k_proj.weight": "kw",
            "self_attn.v_proj.weight": "vw",
            "self_attn.o_proj.weight": "ow",
            "mlp.gate_proj.weight": "gate_w",
            "mlp.up_proj.weight": "up_w",
            "mlp.down_proj.weight": "down_w",
        }.get(comp)
        if m:
            return f"t3/layer{L}/{m}"
    # Cond enc.
    if k == "cond_enc.spkr_enc.weight":
        return "t3/cond_enc/spkr_w"
    if k == "cond_enc.spkr_enc.bias":
        return "t3/cond_enc/spkr_b"
    if k == "cond_enc.emotion_adv_fc.weight":
        return "t3/cond_enc/emo_w"
    if k.startswith("cond_enc.perceiver."):
        parts = k.split(".")
        if parts[2] == "pre_attention_query":
            return "t3/cond_enc/perceiver/pre_q"
        elif parts[2] == "attn":
            comp = ".".join(parts[3:])
            m = {
                "norm.weight": "perc_norm_w",
                "norm.bias":   "perc_norm_b",
                "to_q.weight": "perc_q_w", "to_q.bias": "perc_q_b",
                "to_k.weight": "perc_k_w", "to_k.bias": "perc_k_b",
                "to_v.weight": "perc_v_w", "to_v.bias": "perc_v_b",
                "proj_out.weight": "perc_o_w", "proj_out.bias": "perc_o_b",
            }.get(comp)
            if m:
                return f"t3/cond_enc/perceiver/{m}"
    return None


def s3gen_map(k):
    """s3gen weights — pass-through structure with `.` → `/` for filesystem layout.

    Upstream keys are nested module paths; we mirror the tree on disk so the
    Mojo loader can read flow/.../weight.bin directly. Top-level branches:
      flow.*               UpsampleConformerEncoder + CFM (encoder+estimator)
      mel2wav.*            HiFiGAN vocoder
      speaker_encoder.*    CAMPPlus
      tokenizer.*          Bundled S3TokenizerV2 (handled by s3tokenizer_map)
    """
    if k.startswith("tokenizer."):
        return None
    return "s3gen/" + k.replace(".", "/")


def s3tokenizer_map(k):
    """S3Tokenizer (bundled inside s3gen.safetensors under `tokenizer.*`).

    Layout: 2 strided Conv1d (conv1, conv2), 6 ResidualAttentionBlocks with
    FSMN attention + MLP, then FSQ project_down. Mel filter table is a tensor
    `_mel_filters` (128 × 201) — kept as helper, not used in graph.
    """
    if k == "tokenizer._mel_filters":
        return "s3t/mel_filters"
    if k == "tokenizer.encoder.conv1.weight":
        return "s3t/conv1_w"
    if k == "tokenizer.encoder.conv1.bias":
        return "s3t/conv1_b"
    if k == "tokenizer.encoder.conv2.weight":
        return "s3t/conv2_w"
    if k == "tokenizer.encoder.conv2.bias":
        return "s3t/conv2_b"
    if k == "tokenizer.quantizer._codebook.project_down.weight":
        return "s3t/project_down_w"
    if k == "tokenizer.quantizer._codebook.project_down.bias":
        return "s3t/project_down_b"
    if k.startswith("tokenizer.encoder.blocks."):
        parts = k.split(".")
        L = parts[3]
        comp = ".".join(parts[4:])
        m = {
            "attn_ln.weight": "attn_ln_w",
            "attn_ln.bias":   "attn_ln_b",
            "mlp_ln.weight":  "mlp_ln_w",
            "mlp_ln.bias":    "mlp_ln_b",
            "attn.query.weight": "qw",
            "attn.query.bias":   "qb",
            "attn.key.weight":   "kw",
            # key has no bias upstream
            "attn.value.weight": "vw",
            "attn.value.bias":   "vb",
            "attn.out.weight":   "ow",
            "attn.out.bias":     "ob",
            "attn.fsmn_block.weight": "fsmn_w",
            "mlp.0.weight": "mlp_fc1_w",
            "mlp.0.bias":   "mlp_fc1_b",
            "mlp.2.weight": "mlp_fc2_w",
            "mlp.2.bias":   "mlp_fc2_b",
        }.get(comp)
        if m:
            return f"s3t/block{L}/{m}"
    return None


# ----------------------------------------------------------------------------
# Conversion driver
# ----------------------------------------------------------------------------

def convert(in_path, out_dir, mapper, label):
    """Open `in_path` via safetensors.safe_open and write keys via `mapper`."""
    n_kept = 0
    n_skipped = 0
    with safe_open(in_path, framework="numpy") as f:
        for upstream_key in f.keys():
            arr = f.get_tensor(upstream_key)
            mapped = mapper(upstream_key)
            if mapped is None:
                n_skipped += 1
                continue
            out_path = os.path.join(out_dir, mapped + ".bin")
            write_tensor(out_path, arr)
            n_kept += 1
    print(f"[{label}] kept={n_kept} skipped={n_skipped} → {out_dir}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ve", help="VoiceEncoder safetensors")
    ap.add_argument("--t3", help="T3 backbone safetensors")
    ap.add_argument("--s3gen", help="s3gen safetensors")
    ap.add_argument("--s3t", help="s3tokenizer (currently bundled in s3gen.safetensors)")
    ap.add_argument("--out", required=True, help="Output directory")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    if args.ve:    convert(args.ve, args.out, voice_encoder_map, "VE")
    if args.t3:    convert(args.t3, args.out, t3_map, "T3")
    if args.s3gen: convert(args.s3gen, args.out, s3gen_map, "s3gen")
    if args.s3t:   convert(args.s3t, args.out, s3tokenizer_map, "s3t")


if __name__ == "__main__":
    main()
