"""Dump upstream T3-side prefix + generated speech tokens for a real text +
default ref voice.

Captures everything Mojo needs to run T3 generation end-to-end on real text,
short-circuiting the FCM/CAMPPlus/T3CondEnc chain (which we'll wire in later).

Writes:
  t3_text_parity/text_tokens.bin    int64 (1, T_text+2) with BOS/EOS
  t3_text_parity/cond_emb.bin       float32 (1, T_cond=34, 1024) — output of T3CondEnc
  t3_text_parity/bos_emb.bin        float32 (1, 1, 1024) — speech_emb(start) + speech_pos[0]
  t3_text_parity/text_emb.bin       float32 (1, T_text+2, 1024) — text_emb(tokens) + text_pos
  t3_text_parity/expected_tokens.bin int64 (1, T_speech) — upstream-generated speech tokens
  t3_text_parity/speech_pos_emb.bin float32 (max_speech, 1024) — for Mojo decode step
"""
import os, struct
import numpy as np
import torch

CKPT = "/home/jimmyhmiller/.cache/huggingface/hub/models--ResembleAI--chatterbox/snapshots/ef85ce7bef2f3f1a74d0d837d379d2fcb68203cd"
OUT = "weights/t3_text_parity"


def write_tensor(path, arr):
    arr = np.ascontiguousarray(np.asarray(arr, dtype=np.float32))
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "wb") as f:
        f.write(struct.pack("<q", len(arr.shape)))
        f.write(struct.pack(f"<{len(arr.shape)}q", *arr.shape))
        f.write(struct.pack("<i", 0))
        f.write(arr.tobytes())


def write_i64(path, arr):
    arr = np.ascontiguousarray(np.asarray(arr, dtype=np.int64))
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "wb") as f:
        f.write(struct.pack("<q", len(arr.shape)))
        f.write(struct.pack(f"<{len(arr.shape)}q", *arr.shape))
        f.write(struct.pack("<i", 2))
        f.write(arr.tobytes())


def main():
    from chatterbox.tts import ChatterboxTTS
    import torchaudio

    text = "the quick brown fox"
    ref_path = "/home/jimmyhmiller/.config/paper-audiobooks/default-voice.wav"

    print(f"Loading Chatterbox...")
    m = ChatterboxTTS.from_pretrained("cpu")
    t3 = m.t3
    print("T3 hp:", t3.hp)

    # Build the conditionals (calls voice_encoder, s3tokenizer, etc.).
    print("Preparing conditionals from ref voice...")
    m.prepare_conditionals(ref_path)
    t3_cond = m.conds.t3

    # Capture spks: F.normalize(xvector_192d) → spk_embed_affine_layer (192→80).
    s3gen_ref = m.conds.gen
    xvec = s3gen_ref["embedding"]   # (1, 192) — the CAMPPlus output
    import torch.nn.functional as F
    spks = m.s3gen.flow.spk_embed_affine_layer(F.normalize(xvec, dim=1))   # (1, 80)
    print(f"spks shape: {spks.shape}, mean-abs: {spks.abs().mean().item():.4f}")
    print("t3_cond fields:")
    for f in t3_cond.__dict__:
        v = getattr(t3_cond, f)
        if isinstance(v, torch.Tensor):
            print(f"  {f}: {v.shape}")
        else:
            print(f"  {f}: {v}")

    # Tokenize text.
    text_tokens = m.tokenizer.text_to_tokens(text)
    print(f"Text '{text}' → tokens shape {text_tokens.shape}: {text_tokens.tolist()}")

    # _ensure_BOT_EOT adds start_text_token and stop_text_token.
    BOT = t3.hp.start_text_token
    EOT = t3.hp.stop_text_token
    print(f"BOT={BOT}, EOT={EOT}")

    # Build text_tokens with BOT/EOT (matches t3.inference internal handling).
    full_text_tokens = torch.cat([
        torch.tensor([[BOT]], dtype=text_tokens.dtype),
        text_tokens,
        torch.tensor([[EOT]], dtype=text_tokens.dtype),
    ], dim=1)
    print(f"Full text tokens shape: {full_text_tokens.shape}: {full_text_tokens.tolist()}")

    # Build cond_emb via the model's prepare_conditioning path.
    with torch.inference_mode():
        cond_emb = t3.prepare_conditioning(t3_cond)   # (1, T_cond, 1024)
    print(f"cond_emb shape: {cond_emb.shape}")

    # Build text_emb (token emb + pos emb).
    with torch.inference_mode():
        text_emb = t3.text_emb(full_text_tokens) + t3.text_pos_emb(full_text_tokens)
    print(f"text_emb shape: {text_emb.shape}")

    # Build BOS embedding.
    BOS = t3.hp.start_speech_token
    bos_token = torch.tensor([[BOS]], dtype=torch.long)
    with torch.inference_mode():
        bos_emb = t3.speech_emb(bos_token) + t3.speech_pos_emb.get_fixed_embedding(0)
    print(f"bos_emb shape: {bos_emb.shape}")

    # Skip upstream generation — the CFG batching makes this awkward to call
    # from a script. We have the prefix; Mojo can run generation on it via
    # the verified t3_generate.
    print("(skipping upstream generation — Mojo will run t3_generate on the prefix)")
    speech_tokens = torch.zeros(1, 1, dtype=torch.long)   # placeholder

    # Dump full speech_pos_emb table.
    max_speech = t3.hp.max_mel_seq_len if hasattr(t3.hp, 'max_mel_seq_len') else t3.speech_pos_emb.emb.weight.shape[0]
    speech_pos_table = t3.speech_pos_emb.emb.weight.detach().cpu().numpy()
    print(f"speech_pos_emb table: {speech_pos_table.shape}")

    # Build a large enough cos/sin table for RoPE. T3 backbone uses HF Llama
    # RoPE on head_dim=64. Inverse freqs:
    #   inv_freq[k] = 1 / 10000^(2k/D), k in [0, D/2)
    #   pos[t, k] = t * inv_freq[k]
    # Then HF style: cos/sin of shape (max_pos, head_dim) where odd half = even half.
    MAX_CTX = 200   # plenty for cond(34) + text(13) + bos(1) + speech(<= 152)
    HEAD_DIM = 64
    inv_freq = 1.0 / (10000.0 ** (torch.arange(0, HEAD_DIM, 2, dtype=torch.float32) / HEAD_DIM))
    pos = torch.arange(MAX_CTX, dtype=torch.float32).unsqueeze(1) * inv_freq.unsqueeze(0)  # (MAX_CTX, HEAD_DIM/2)
    # HF Llama duplicates the half-d for full head_dim: cat([pos, pos], dim=-1).
    pos_full = torch.cat([pos, pos], dim=-1)
    cos_full = torch.cos(pos_full)
    sin_full = torch.sin(pos_full)

    write_i64(f"{OUT}/text_tokens.bin", full_text_tokens.numpy())
    write_tensor(f"{OUT}/cond_emb.bin", cond_emb.detach().cpu().numpy())
    write_tensor(f"{OUT}/text_emb.bin", text_emb.detach().cpu().numpy())
    write_tensor(f"{OUT}/bos_emb.bin", bos_emb.detach().cpu().numpy())
    write_i64(f"{OUT}/expected_tokens.bin", speech_tokens.detach().cpu().numpy())
    write_tensor(f"{OUT}/speech_pos_emb_full.bin", speech_pos_table)
    write_tensor(f"{OUT}/cos_full.bin", cos_full.numpy())
    write_tensor(f"{OUT}/sin_full.bin", sin_full.numpy())
    write_tensor(f"{OUT}/spks.bin", spks.detach().cpu().numpy())
    print(f"Wrote oracle to {OUT}/")
    print(f"text_tokens (with BOT/EOT): {full_text_tokens.tolist()}")
    print(f"cos/sin tables: {cos_full.shape}")
    print(f"speech_pos_emb_full: {speech_pos_table.shape}")


if __name__ == "__main__":
    main()
