"""Dump text-embedding + positional-embedding fixture.

Produces text_emb_with_pos = text_emb_table[ids] + text_pos_emb_table[positions].
"""
import os, struct, importlib.util
import numpy as np
import torch
import torch.nn as nn


OUT_DIR = os.path.join(os.path.dirname(__file__), "..", "tests", "fixtures", "text_prefix")
os.makedirs(OUT_DIR, exist_ok=True)


def write_tensor(path, arr):
    arr = np.ascontiguousarray(np.asarray(arr, dtype=np.float32))
    with open(path, "wb") as f:
        f.write(struct.pack("<q", len(arr.shape)))
        f.write(struct.pack(f"<{len(arr.shape)}q", *arr.shape))
        f.write(struct.pack("<i", 0))
        f.write(arr.tobytes())


def save(name, t):
    if isinstance(t, torch.Tensor): t = t.detach().cpu().numpy()
    write_tensor(os.path.join(OUT_DIR, name), t)


# Reuse the upstream LearnedPositionEmbeddings.
spec = importlib.util.spec_from_file_location(
    "lpe", "/home/jimmyhmiller/Documents/Code/Playground/claude-experiments/chatterbox-rewrite/chatterbox/src/chatterbox/models/t3/modules/learned_pos_emb.py",
)
lpe_mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(lpe_mod)


def main():
    torch.manual_seed(0)
    VOCAB = 704
    D = 1024
    MAX_TEXT = 2050  # max_text_tokens + 2

    text_emb = nn.Embedding(VOCAB, D).eval()
    text_pos = lpe_mod.LearnedPositionEmbeddings(MAX_TEXT, D).eval()

    # Simulate 17 tokenized text ids.
    text_tokens = torch.randint(0, VOCAB, (1, 17))
    with torch.inference_mode():
        text_e = text_emb(text_tokens)            # (1, 17, 1024)
        text_p = text_pos(text_tokens)            # (1, 17, 1024)
        out = text_e + text_p
    print("text_tokens:", text_tokens.shape, "out:", out.shape)
    print("out[0, 0, :4]:", out[0, 0, :4].tolist())

    save("text_tokens.bin", text_tokens.float())
    save("out.bin", out)
    save("text_emb_w.bin", text_emb.weight)
    save("text_pos_w.bin", text_pos.emb.weight)
    print("dumped to", OUT_DIR)


if __name__ == "__main__":
    main()
