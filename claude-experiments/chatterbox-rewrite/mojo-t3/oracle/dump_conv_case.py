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


if __name__ == "__main__":
    main()
