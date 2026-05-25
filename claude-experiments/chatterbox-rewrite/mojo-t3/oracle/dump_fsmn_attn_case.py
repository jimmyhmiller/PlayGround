"""Dump FSMN multi-head attention fixture (single block, small dims)."""
import os, struct, math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


OUT_DIR = os.path.join(os.path.dirname(__file__), "..", "tests", "fixtures", "fsmn_attn")
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


def precompute_freqs_cis(dim, end, theta=10000.0):
    freqs = 1.0 / (theta**(torch.arange(0, dim, 2)[:(dim // 2)].float() / dim))
    t = torch.arange(end)
    freqs = torch.outer(t, freqs).float()
    freqs_cis = torch.polar(torch.ones_like(freqs), freqs)
    return torch.cat((freqs_cis, freqs_cis), dim=-1)


def apply_rotary_emb(xq, xk, freqs_cis):
    real = torch.view_as_real(freqs_cis)
    cos, sin = real[:, :, 0], real[:, :, 1]
    cos = cos.unsqueeze(0).unsqueeze(2).to(xq.dtype)
    sin = sin.unsqueeze(0).unsqueeze(2).to(xq.dtype)
    D = xq.shape[-1]
    half_l, half_r = xq[:, :, :, :D//2], xq[:, :, :, D//2:]
    xq_r = torch.cat((-half_r, half_l), dim=-1)
    half_l, half_r = xk[:, :, :, :D//2], xk[:, :, :, D//2:]
    xk_r = torch.cat((-half_r, half_l), dim=-1)
    return xq * cos + xq_r * sin, xk * cos + xk_r * sin


class FSMNAttn(nn.Module):
    def __init__(self, n_state, n_head, kernel_size=31):
        super().__init__()
        self.n_state = n_state
        self.n_head = n_head
        self.query = nn.Linear(n_state, n_state, bias=True)
        self.key   = nn.Linear(n_state, n_state, bias=False)
        self.value = nn.Linear(n_state, n_state, bias=True)
        self.out   = nn.Linear(n_state, n_state, bias=True)
        self.fsmn_block = nn.Conv1d(n_state, n_state, kernel_size,
                                    stride=1, padding=0, groups=n_state, bias=False)
        self.left_padding = (kernel_size - 1) // 2
        self.right_padding = kernel_size - 1 - self.left_padding
        self.pad_fn = nn.ConstantPad1d((self.left_padding, self.right_padding), 0.0)

    def forward(self, x, mask_pad, freqs_cis):
        # x: (B, S, D)   mask_pad: (B, S, 1) — float 1/0
        q = self.query(x)
        k = self.key(x)
        v = self.value(x)
        B, S, D = q.shape
        H = self.n_head
        Dh = D // H
        q4 = q.view(B, S, H, Dh)
        k4 = k.view(B, S, H, Dh)
        v4 = v.view(B, S, H, Dh)
        q4, k4 = apply_rotary_emb(q4, k4, freqs_cis)

        # FSMN memory: depthwise conv over S, with constant pad LEFT/RIGHT.
        # v4 reshape back to (B, S, D)
        v_bsd = v4.reshape(B, S, D)
        # mask_pad is (B, S, 1) → broadcast.
        v_masked = v_bsd * mask_pad
        v_t = v_masked.transpose(1, 2)         # (B, D, S)
        v_padded = self.pad_fn(v_t)             # (B, D, S + L + R)
        fsm_conv = self.fsmn_block(v_padded)   # (B, D, S)
        fsm_conv = fsm_conv.transpose(1, 2)    # (B, S, D)
        fsm_mem = (fsm_conv + v_masked) * mask_pad

        # Attention.
        scale = (Dh ** -0.25)
        q_p = q4.permute(0, 2, 1, 3) * scale   # (B, H, S, Dh)
        k_p = k4.permute(0, 2, 3, 1) * scale   # (B, H, Dh, S)
        v_p = v4.permute(0, 2, 1, 3)           # (B, H, S, Dh)
        qk = q_p @ k_p                          # (B, H, S, S)
        # mask: float bias (zeros for valid, very-negative for pad)
        attn = F.softmax(qk.float(), dim=-1).to(q_p.dtype)
        attn_out = (attn @ v_p).permute(0, 2, 1, 3).flatten(start_dim=2)  # (B, S, D)
        out = self.out(attn_out) + fsm_mem
        return out


def main():
    torch.manual_seed(0)
    B, S, D, H = 1, 24, 128, 4
    Dh = D // H
    KSIZE = 31
    model = FSMNAttn(D, H, KSIZE).eval()
    x = torch.randn(B, S, D)
    mask_pad = torch.ones(B, S, 1)              # all valid
    freqs_cis = precompute_freqs_cis(Dh // 2 * 2, S * 2)[:S]  # (S, Dh)
    with torch.inference_mode():
        out = model(x, mask_pad, freqs_cis)
    print("x:", x.shape, "out:", out.shape)
    print("out[0, 0, :4]:", out[0, 0, :4].tolist())

    # Save fixture.
    save("x.bin", x)
    save("out.bin", out)
    save("mask_pad.bin", mask_pad)
    # freqs_cis → save just the unique half (S, HALF=Dh//2) since cat(x, x, -1)
    # yields the same value in both halves.
    real = torch.view_as_real(freqs_cis)
    half = Dh // 2
    cos = real[:, :half, 0]   # (S, HALF)
    sin = real[:, :half, 1]   # (S, HALF)
    save("cos.bin", cos)
    save("sin.bin", sin)
    # Weights.
    save("q_w.bin", model.query.weight)
    save("q_b.bin", model.query.bias)
    save("k_w.bin", model.key.weight)
    save("v_w.bin", model.value.weight)
    save("v_b.bin", model.value.bias)
    save("out_w.bin", model.out.weight)
    save("out_b.bin", model.out.bias)
    save("fsmn_w.bin", model.fsmn_block.weight)  # (D, 1, KSIZE)
    print("dumped to", OUT_DIR)


if __name__ == "__main__":
    main()
