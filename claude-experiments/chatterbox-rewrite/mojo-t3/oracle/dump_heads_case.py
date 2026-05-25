"""
T3 head/embedding kernel parity case using the real T3 speech embedding,
positional embedding, and LM-head weights.

Three independent fixtures:

  embed_lookup/
    table.bin       (V=8194, D=1024)   speech_emb.weight
    ids.bin         (B=1, S=8) int64   sample token ids
    expected.bin    (B, S, D)
  pos_emb_add/
    pos_table.bin   (P=4100, D=1024)   speech_pos_emb.emb.weight
    x.bin           (B=1, S=8, D)
    base_pos        encoded in expected (start position offset)
    expected.bin    (B, S, D)          x + pos_table[base_pos..base_pos+S]
  argmax/
    logits.bin      (B=1, S=4, V=8194) — synthetic with known argmax per row
    expected.bin    (B, S) int64
"""

from __future__ import annotations

import struct
from pathlib import Path

import numpy as np
import torch
from huggingface_hub import hf_hub_download
from safetensors.torch import load_file

ROOT = Path(__file__).resolve().parents[1]
FIX = ROOT / "tests" / "fixtures"
OUT = FIX / "heads"
OUT.mkdir(parents=True, exist_ok=True)

V_SPEECH = 8194
D = 1024
P_SPEECH = 4100


def write_tensor(path: Path, arr: np.ndarray) -> None:
    if arr.dtype == np.float32:
        tag, raw = 0, arr.astype(np.float32, copy=False).tobytes()
    elif arr.dtype == np.uint16:
        tag, raw = 1, arr.astype(np.uint16, copy=False).tobytes()
    elif arr.dtype == np.int64:
        tag, raw = 2, arr.astype(np.int64, copy=False).tobytes()
    else:
        raise TypeError(f"unsupported dtype {arr.dtype}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as f:
        f.write(struct.pack("<q", len(arr.shape)))
        f.write(struct.pack(f"<{len(arr.shape)}q", *arr.shape))
        f.write(struct.pack("<i", tag))
        f.write(raw)


def load_t3_state() -> dict:
    ckpt = hf_hub_download(repo_id="ResembleAI/chatterbox", filename="t3_cfg.safetensors")
    return load_file(ckpt)


def main() -> None:
    state = load_t3_state()

    # ---- embed_lookup ----
    emb = state["speech_emb.weight"].to(torch.float32)  # (V, D)
    assert emb.shape == (V_SPEECH, D), emb.shape
    ids = torch.tensor([[0, 1, 100, 1000, 5000, 6561, 8000, 8193]], dtype=torch.int64)
    expected_emb = emb[ids[0]].unsqueeze(0)  # (1, 8, D)

    write_tensor(OUT / "embed_lookup_table_fp32.bin", emb.numpy().astype(np.float32))
    write_tensor(OUT / "embed_lookup_ids.bin", ids.numpy().astype(np.int64))
    write_tensor(OUT / "embed_lookup_expected_fp32.bin", expected_emb.numpy().astype(np.float32))
    print(f"[embed_lookup] table {tuple(emb.shape)} ids {tuple(ids.shape)} expected {tuple(expected_emb.shape)}")

    # ---- pos_emb_add ----
    pos_table = state["speech_pos_emb.emb.weight"].to(torch.float32)  # (P, D)
    assert pos_table.shape == (P_SPEECH, D), pos_table.shape
    seq = 8
    base_pos = 12  # arbitrary nonzero offset to verify the offset path
    g = torch.Generator(device="cpu").manual_seed(0xC0FFEE)
    x_in = torch.randn(1, seq, D, generator=g, dtype=torch.float32)
    expected_pos = x_in + pos_table[base_pos : base_pos + seq].unsqueeze(0)

    write_tensor(OUT / "pos_table_fp32.bin", pos_table.numpy().astype(np.float32))
    write_tensor(OUT / "pos_x_fp32.bin", x_in.numpy().astype(np.float32))
    write_tensor(OUT / "pos_meta.bin",
                 np.array([base_pos, seq], dtype=np.int64))
    write_tensor(OUT / "pos_expected_fp32.bin", expected_pos.numpy().astype(np.float32))
    print(f"[pos_emb_add] pos_table {tuple(pos_table.shape)} base_pos={base_pos} expected {tuple(expected_pos.shape)}")

    # ---- argmax ----
    # Use the real LM-head output for a tiny hidden-states input — but that
    # would require running the backbone here. Simpler: synthesize logits
    # with known max indices via the speech_head matmul on a few random
    # hidden states. This still exercises the realistic V=8194 vocab.
    speech_head = state["speech_head.weight"].to(torch.float32)  # (V, D)
    assert speech_head.shape == (V_SPEECH, D), speech_head.shape
    n_rows = 4
    g2 = torch.Generator(device="cpu").manual_seed(0xBEEF)
    hidden = torch.randn(1, n_rows, D, generator=g2, dtype=torch.float32)
    logits = hidden @ speech_head.T  # (1, n_rows, V)
    expected_ids = logits.argmax(dim=-1).to(torch.int64)  # (1, n_rows)

    write_tensor(OUT / "argmax_logits_fp32.bin", logits.numpy().astype(np.float32))
    write_tensor(OUT / "argmax_expected.bin", expected_ids.numpy().astype(np.int64))
    print(f"[argmax] logits {tuple(logits.shape)} expected_ids {expected_ids[0].numpy().tolist()}")

    # Also dump speech_head.weight in case we want a future end-to-end LM-head test.
    write_tensor(OUT / "speech_head_fp32.bin", speech_head.numpy().astype(np.float32))


if __name__ == "__main__":
    main()
