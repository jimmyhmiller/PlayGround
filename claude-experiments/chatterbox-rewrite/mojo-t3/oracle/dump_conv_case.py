"""
conv1d / transposed_conv1d / leaky_relu parity cases.

Small synthetic tensors with realistic HiFiGAN-ish shapes:
  conv1d:           (B=1, C_in=4, L=16) with K=7, stride=1, padding=3, dilation=1
  conv1d_dilated:   (B=1, C_in=4, L=16) with K=3, stride=1, padding=2, dilation=2
  transposed:       (B=1, C_in=4, L=8)  with K=8, stride=4, padding=2  (upsample ×4)
  leaky_relu:       (B=1, C=4, L=16) slope=0.1

Outputs: x, w, bias, expected (and the conv hyperparameters as a small i64 array).
"""

from __future__ import annotations

import struct
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "tests" / "fixtures" / "conv"
OUT.mkdir(parents=True, exist_ok=True)


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


def main() -> None:
    g = torch.Generator(device="cpu").manual_seed(0xC0FFEE)

    # ---- conv1d (K=7, stride=1, padding=3, dilation=1) ----
    B, C_in, L = 1, 4, 16
    C_out = 6
    K = 7
    stride, padding, dilation = 1, 3, 1
    x = torch.randn(B, C_in, L, generator=g, dtype=torch.float32)
    w = torch.randn(C_out, C_in, K, generator=g, dtype=torch.float32) * 0.3
    bias = torch.randn(C_out, generator=g, dtype=torch.float32) * 0.1
    y = F.conv1d(x, w, bias=bias, stride=stride, padding=padding, dilation=dilation)
    print(f"[conv1d] x{tuple(x.shape)} w{tuple(w.shape)} -> y{tuple(y.shape)}")
    write_tensor(OUT / "conv1d_x.bin", x.numpy())
    write_tensor(OUT / "conv1d_w.bin", w.numpy())
    write_tensor(OUT / "conv1d_bias.bin", bias.numpy())
    write_tensor(OUT / "conv1d_expected.bin", y.numpy())
    write_tensor(OUT / "conv1d_meta.bin",
                 np.array([B, C_in, C_out, L, y.shape[2], K, stride, padding, dilation],
                          dtype=np.int64))

    # ---- conv1d_dilated (K=3, dilation=2) — exercises dilation path ----
    K2 = 3
    pad2 = 2
    dil2 = 2
    w2 = torch.randn(C_out, C_in, K2, generator=g, dtype=torch.float32) * 0.3
    bias2 = torch.randn(C_out, generator=g, dtype=torch.float32) * 0.1
    y2 = F.conv1d(x, w2, bias=bias2, stride=1, padding=pad2, dilation=dil2)
    print(f"[conv1d_dilated] -> y{tuple(y2.shape)}")
    write_tensor(OUT / "conv1d_dil_w.bin", w2.numpy())
    write_tensor(OUT / "conv1d_dil_bias.bin", bias2.numpy())
    write_tensor(OUT / "conv1d_dil_expected.bin", y2.numpy())
    write_tensor(OUT / "conv1d_dil_meta.bin",
                 np.array([B, C_in, C_out, L, y2.shape[2], K2, 1, pad2, dil2],
                          dtype=np.int64))

    # ---- transposed_conv1d (K=8, stride=4, padding=2 — typical upsample ×4) ----
    L_t = 8
    Kt = 8
    stride_t = 4
    pad_t = 2
    xt = torch.randn(B, C_in, L_t, generator=g, dtype=torch.float32)
    wt = torch.randn(C_in, C_out, Kt, generator=g, dtype=torch.float32) * 0.2
    bt = torch.randn(C_out, generator=g, dtype=torch.float32) * 0.1
    yt = F.conv_transpose1d(xt, wt, bias=bt, stride=stride_t, padding=pad_t, dilation=1)
    print(f"[transposed] x{tuple(xt.shape)} w{tuple(wt.shape)} -> y{tuple(yt.shape)}")
    write_tensor(OUT / "tconv1d_x.bin", xt.numpy())
    write_tensor(OUT / "tconv1d_w.bin", wt.numpy())
    write_tensor(OUT / "tconv1d_bias.bin", bt.numpy())
    write_tensor(OUT / "tconv1d_expected.bin", yt.numpy())
    write_tensor(OUT / "tconv1d_meta.bin",
                 np.array([B, C_in, C_out, L_t, yt.shape[2], Kt, stride_t, pad_t, 1],
                          dtype=np.int64))

    # ---- leaky_relu ----
    x_l = torch.randn(B, C_out, L, generator=g, dtype=torch.float32)
    y_l = F.leaky_relu(x_l, negative_slope=0.1)
    write_tensor(OUT / "leaky_x.bin", x_l.numpy())
    write_tensor(OUT / "leaky_expected.bin", y_l.numpy())
    print(f"[leaky_relu] x{tuple(x_l.shape)} slope=0.1")

    # ---- Snake activation ----
    C_s, L_s = 8, 16
    x_s = torch.randn(B, C_s, L_s, generator=g, dtype=torch.float32)
    alpha = torch.randn(C_s, generator=g, dtype=torch.float32) * 0.5 + 1.0
    y_s = x_s + (1.0 / (alpha.unsqueeze(0).unsqueeze(-1) + 1e-9)) * torch.sin(x_s * alpha.unsqueeze(0).unsqueeze(-1)) ** 2
    write_tensor(OUT / "snake_x.bin", x_s.numpy())
    write_tensor(OUT / "snake_alpha.bin", alpha.numpy())
    write_tensor(OUT / "snake_expected.bin", y_s.numpy())
    write_tensor(OUT / "snake_meta.bin", np.array([B, C_s, L_s], dtype=np.int64))
    print(f"[snake] x{tuple(x_s.shape)} alpha{tuple(alpha.shape)}")

    # ---- ResBlock parity (1 sub-block, dilations=[1, 3, 5]) ----
    # Mirrors hifigan.py:ResBlock.forward for one dilation step.
    # We dump weights+alphas+input, and the expected post-ResBlock output.
    # Test will verify our Mojo-side composition (snake → conv → snake → conv → +residual)
    # for the *first* dilation only; full ResBlock chains three of these.
    C_r = 8
    L_r = 32
    K_r = 3
    dilations = [1, 3, 5]
    x_r = torch.randn(B, C_r, L_r, generator=g, dtype=torch.float32)
    w1_list, w2_list, b1_list, b2_list, a1_list, a2_list = [], [], [], [], [], []
    for dil in dilations:
        pad1 = ((K_r - 1) * dil) // 2          # convs1: dilation=dil
        pad2 = ((K_r - 1) * 1) // 2            # convs2: dilation=1 always
        w1 = torch.randn(C_r, C_r, K_r, generator=g) * 0.1
        b1 = torch.randn(C_r, generator=g) * 0.05
        w2 = torch.randn(C_r, C_r, K_r, generator=g) * 0.1
        b2 = torch.randn(C_r, generator=g) * 0.05
        a1 = torch.ones(C_r) + torch.randn(C_r, generator=g) * 0.1
        a2 = torch.ones(C_r) + torch.randn(C_r, generator=g) * 0.1
        w1_list.append((w1, b1, pad1))
        w2_list.append((w2, b2, 1, pad2))
        a1_list.append(a1)
        a2_list.append(a2)

    def snake(x, alpha):
        a = alpha.unsqueeze(0).unsqueeze(-1)
        return x + (1.0 / (a + 1e-9)) * torch.sin(x * a) ** 2

    x_cur = x_r
    for i in range(len(dilations)):
        w1, b1, pad1 = w1_list[i]
        w2, b2, dil2, pad2 = w2_list[i]
        a1 = a1_list[i]
        a2 = a2_list[i]
        xt = snake(x_cur, a1)
        xt = F.conv1d(xt, w1, bias=b1, stride=1, padding=pad1, dilation=dilations[i])
        xt = snake(xt, a2)
        xt = F.conv1d(xt, w2, bias=b2, stride=1, padding=pad2, dilation=1)
        x_cur = x_cur + xt
    y_r = x_cur

    write_tensor(OUT / "resblock_x.bin", x_r.numpy())
    write_tensor(OUT / "resblock_expected.bin", y_r.numpy())
    for i, dil in enumerate(dilations):
        w1, b1, pad1 = w1_list[i]
        w2, b2, dil2, pad2 = w2_list[i]
        write_tensor(OUT / f"resblock_w1_{i}.bin", w1.numpy())
        write_tensor(OUT / f"resblock_b1_{i}.bin", b1.numpy())
        write_tensor(OUT / f"resblock_w2_{i}.bin", w2.numpy())
        write_tensor(OUT / f"resblock_b2_{i}.bin", b2.numpy())
        write_tensor(OUT / f"resblock_a1_{i}.bin", a1_list[i].numpy())
        write_tensor(OUT / f"resblock_a2_{i}.bin", a2_list[i].numpy())
        write_tensor(OUT / f"resblock_meta_{i}.bin",
                     np.array([dil, pad1, pad2], dtype=np.int64))
    write_tensor(OUT / "resblock_global_meta.bin",
                 np.array([B, C_r, L_r, K_r, len(dilations)], dtype=np.int64))
    print(f"[resblock] x{tuple(x_r.shape)} dilations={dilations} -> y{tuple(y_r.shape)}")


if __name__ == "__main__":
    main()
