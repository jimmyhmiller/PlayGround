"""Dump a single FSMN ResidualAttentionBlock fixture (LN + FSMN + LN + MLP)."""
import os, struct, sys, importlib.util
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


OUT_DIR = os.path.join(os.path.dirname(__file__), "..", "tests", "fixtures", "fsmn_resblock")
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


# Reuse helpers from FSMN dump.
sys.path.insert(0, os.path.dirname(__file__))
spec = importlib.util.spec_from_file_location("fmod", os.path.join(os.path.dirname(__file__), "dump_fsmn_attn_case.py"))
fmod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(fmod)


class ResBlock(nn.Module):
    def __init__(self, n_state, n_head, kernel_size=31):
        super().__init__()
        self.attn = fmod.FSMNAttn(n_state, n_head, kernel_size)
        self.attn_ln = nn.LayerNorm(n_state, eps=1e-5)
        n_mlp = n_state * 4
        self.mlp = nn.Sequential(
            nn.Linear(n_state, n_mlp),
            nn.GELU(),
            nn.Linear(n_mlp, n_state),
        )
        self.mlp_ln = nn.LayerNorm(n_state)

    def forward(self, x, mask_pad, freqs_cis):
        x = x + self.attn(self.attn_ln(x), mask_pad, freqs_cis)
        x = x + self.mlp(self.mlp_ln(x))
        return x


def main():
    torch.manual_seed(0)
    B, S, D, H = 1, 24, 128, 4
    Dh = D // H
    block = ResBlock(D, H).eval()
    x = torch.randn(B, S, D)
    mask_pad = torch.ones(B, S, 1)
    freqs_cis = fmod.precompute_freqs_cis(Dh, S * 2)[:S]
    with torch.inference_mode():
        out = block(x, mask_pad, freqs_cis)
    print("x:", x.shape, "out:", out.shape)
    print("out[0,0,:4]:", out[0, 0, :4].tolist())

    save("x.bin", x)
    save("out.bin", out)
    save("mask_pad.bin", mask_pad)
    real = torch.view_as_real(freqs_cis)
    half = Dh // 2
    save("cos.bin", real[:, :half, 0])
    save("sin.bin", real[:, :half, 1])
    # attn_ln weights.
    save("attn_ln_w.bin", block.attn_ln.weight)
    save("attn_ln_b.bin", block.attn_ln.bias)
    # mlp_ln weights.
    save("mlp_ln_w.bin", block.mlp_ln.weight)
    save("mlp_ln_b.bin", block.mlp_ln.bias)
    # FSMN attn weights.
    save("q_w.bin", block.attn.query.weight); save("q_b.bin", block.attn.query.bias)
    save("k_w.bin", block.attn.key.weight)
    save("v_w.bin", block.attn.value.weight); save("v_b.bin", block.attn.value.bias)
    save("out_w.bin", block.attn.out.weight); save("out_b.bin", block.attn.out.bias)
    save("fsmn_w.bin", block.attn.fsmn_block.weight)
    # MLP weights.
    save("mlp_fc1_w.bin", block.mlp[0].weight); save("mlp_fc1_b.bin", block.mlp[0].bias)
    save("mlp_fc2_w.bin", block.mlp[2].weight); save("mlp_fc2_b.bin", block.mlp[2].bias)
    print("dumped to", OUT_DIR)


if __name__ == "__main__":
    main()
