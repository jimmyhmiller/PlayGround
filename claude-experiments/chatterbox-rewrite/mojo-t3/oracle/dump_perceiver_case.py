"""Dump Perceiver resampler fixture (input, weights, expected output)."""
import os, struct, importlib.util
import numpy as np
import torch


OUT_DIR = os.path.join(os.path.dirname(__file__), "..", "tests", "fixtures", "perceiver")
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


def main():
    spec = importlib.util.spec_from_file_location(
        "p", "/home/jimmyhmiller/Documents/Code/Playground/claude-experiments/chatterbox-rewrite/chatterbox/src/chatterbox/models/t3/modules/perceiver.py",
    )
    p = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(p)

    torch.manual_seed(0)
    P = p.Perceiver(pre_attention_query_token=32, pre_attention_query_size=1024,
                    embedding_dim=1024, num_attn_heads=4).eval()
    Tk = 150
    h = torch.randn(1, Tk, 1024)
    with torch.inference_mode():
        out = P(h)
    print("h:", h.shape, "out:", out.shape)
    print("out[0,0,:4]:", out[0, 0, :4].tolist())

    save("h.bin", h)
    save("out.bin", out)
    save("pre_attention_query.bin", P.pre_attention_query)
    # AttentionBlock2 weights — used for BOTH cross and self attention.
    save("attn_norm_w.bin", P.attn.norm.weight)
    save("attn_norm_b.bin", P.attn.norm.bias)
    save("to_q_w.bin", P.attn.to_q.weight)
    save("to_q_b.bin", P.attn.to_q.bias)
    save("to_k_w.bin", P.attn.to_k.weight)
    save("to_k_b.bin", P.attn.to_k.bias)
    save("to_v_w.bin", P.attn.to_v.weight)
    save("to_v_b.bin", P.attn.to_v.bias)
    save("proj_out_w.bin", P.attn.proj_out.weight)
    save("proj_out_b.bin", P.attn.proj_out.bias)
    with open(os.path.join(OUT_DIR, "meta.txt"), "w") as f:
        f.write(f"B=1\nTk={Tk}\nSq=32\nD=1024\nH=4\nDh=256\n")
    print("dumped to", OUT_DIR)


if __name__ == "__main__":
    main()
