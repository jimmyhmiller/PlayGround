"""Dump full AudioEncoderV2 fixture (2 strided conv + N blocks).

Smaller config for fast Mojo unit-test:
  n_mels=80, n_state=128, n_head=4, n_layer=2, stride=2.
"""
import os, struct, sys, importlib.util
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


OUT_DIR = os.path.join(os.path.dirname(__file__), "..", "tests", "fixtures", "audio_encoder_v2")
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


# Reuse FSMN attn from dump_fsmn_attn_case.
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
            nn.Linear(n_state, n_mlp), nn.GELU(), nn.Linear(n_mlp, n_state),
        )
        self.mlp_ln = nn.LayerNorm(n_state)

    def forward(self, x, mask_pad, freqs_cis):
        x = x + self.attn(self.attn_ln(x), mask_pad, freqs_cis)
        x = x + self.mlp(self.mlp_ln(x))
        return x


class AudioEncoderV2(nn.Module):
    def __init__(self, n_mels=80, n_state=128, n_head=4, n_layer=2, stride=2):
        super().__init__()
        self.stride = stride
        self.conv1 = nn.Conv1d(n_mels, n_state, kernel_size=3, stride=stride, padding=1)
        self.conv2 = nn.Conv1d(n_state, n_state, kernel_size=3, stride=2, padding=1)
        Dh = n_state // n_head
        self.freqs_cis = fmod.precompute_freqs_cis(Dh, 1024 * 2)
        self.blocks = nn.ModuleList([ResBlock(n_state, n_head) for _ in range(n_layer)])

    def forward(self, x, mask_pad):
        """Simplified forward — assumes x_len matches T everywhere (no padding).

        mask_pad enters as (B, T_out, 1) for the final blocks. We ignore the
        in-conv masking since x_len matches T (test case has no padding).
        """
        x = F.gelu(self.conv1(x))    # (B, n_state, T/stride)
        x = F.gelu(self.conv2(x))    # (B, n_state, T/(stride*2))
        x = x.permute(0, 2, 1)        # (B, T_out, n_state)
        T_out = x.shape[1]
        freqs_cis = self.freqs_cis[:T_out].to(x.device)
        for block in self.blocks:
            x = block(x, mask_pad, freqs_cis)
        return x


def main():
    torch.manual_seed(0)
    n_mels, n_state, n_head, n_layer, stride = 80, 128, 4, 2, 2
    T_in = 24
    enc = AudioEncoderV2(n_mels, n_state, n_head, n_layer, stride).eval()

    # T_out = T_in / (stride * 2) = 24 / 4 = 6. But our conv preserves dim w/ pad=1.
    # T_after_conv1 = (T_in + 2 - 3) // stride + 1 = 12; T_after_conv2 = (12+2-3)//2 + 1 = 6.
    x = torch.randn(1, n_mels, T_in)
    # mask_pad after both convs.
    T_out = ((T_in + 2 - 3) // stride + 1)   # after conv1
    T_out = ((T_out + 2 - 3) // 2 + 1)        # after conv2
    print(f"T_in={T_in}, T_out={T_out}")
    mask_pad = torch.ones(1, T_out, 1)
    with torch.inference_mode():
        out = enc(x, mask_pad)
    print("x:", x.shape, "out:", out.shape)
    print("out[0,0,:4]:", out[0, 0, :4].tolist())

    save("x.bin", x)
    save("out.bin", out)
    save("mask_pad.bin", mask_pad)
    Dh = n_state // n_head
    half = Dh // 2
    real = torch.view_as_real(enc.freqs_cis[:T_out])
    save("cos.bin", real[:, :half, 0])
    save("sin.bin", real[:, :half, 1])

    save("conv1_w.bin", enc.conv1.weight)
    save("conv1_b.bin", enc.conv1.bias)
    save("conv2_w.bin", enc.conv2.weight)
    save("conv2_b.bin", enc.conv2.bias)

    for L, block in enumerate(enc.blocks):
        pre = f"L{L}_"
        save(f"{pre}attn_ln_w.bin", block.attn_ln.weight)
        save(f"{pre}attn_ln_b.bin", block.attn_ln.bias)
        save(f"{pre}mlp_ln_w.bin", block.mlp_ln.weight)
        save(f"{pre}mlp_ln_b.bin", block.mlp_ln.bias)
        save(f"{pre}q_w.bin", block.attn.query.weight)
        save(f"{pre}q_b.bin", block.attn.query.bias)
        save(f"{pre}k_w.bin", block.attn.key.weight)
        save(f"{pre}v_w.bin", block.attn.value.weight)
        save(f"{pre}v_b.bin", block.attn.value.bias)
        save(f"{pre}out_w.bin", block.attn.out.weight)
        save(f"{pre}out_b.bin", block.attn.out.bias)
        save(f"{pre}fsmn_w.bin", block.attn.fsmn_block.weight)
        save(f"{pre}mlp_fc1_w.bin", block.mlp[0].weight)
        save(f"{pre}mlp_fc1_b.bin", block.mlp[0].bias)
        save(f"{pre}mlp_fc2_w.bin", block.mlp[2].weight)
        save(f"{pre}mlp_fc2_b.bin", block.mlp[2].bias)

    with open(os.path.join(OUT_DIR, "meta.txt"), "w") as f:
        f.write(f"B=1\nT_in={T_in}\nT_out={T_out}\nn_mels={n_mels}\nn_state={n_state}\nn_head={n_head}\nn_layer={n_layer}\nstride={stride}\n")
    print("dumped to", OUT_DIR)


if __name__ == "__main__":
    main()
