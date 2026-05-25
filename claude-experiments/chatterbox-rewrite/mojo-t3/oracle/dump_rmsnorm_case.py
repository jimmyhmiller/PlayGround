"""
Produce self-contained RMSNorm parity cases for the Mojo test.

We pick layer 0 (input_layernorm) and layer 0's residual stream as input.
For RMSNorm specifically:
  input  = layer.0 input residual  ==  input_embeds
  weight = state_dict["layers.0.input_layernorm.weight"]
  output = activations["layer.0.input_layernorm_out"]

Two fixture sets:
  fp32 — input/weight/output sourced from the fp32 oracle pass.
  bf16 — sourced from the bf16 oracle pass (already rounded to bf16 precision
         by the HF bf16 forward), then re-encoded to actual bf16 raw bits.

Binary format (little-endian, packed):
  i64        rank
  i64[rank]  shape
  i32        dtype_tag    (0=fp32, 1=bf16-as-uint16)
  payload    raw element bytes (4 per fp32, 2 per bf16)
"""

from __future__ import annotations

import struct
from pathlib import Path

import numpy as np
import torch
from safetensors.torch import load_file

ROOT = Path(__file__).resolve().parents[1]
FIX = ROOT / "tests" / "fixtures"
OUT = FIX / "rmsnorm"
OUT.mkdir(parents=True, exist_ok=True)


def write_tensor(path: Path, arr: np.ndarray) -> None:
    if arr.dtype == np.float32:
        tag = 0
        raw = arr.astype(np.float32, copy=False).tobytes()
    elif arr.dtype == np.uint16:
        tag = 1
        raw = arr.astype(np.uint16, copy=False).tobytes()
    else:
        raise TypeError(f"unsupported dtype {arr.dtype}")
    shape = arr.shape
    with path.open("wb") as f:
        f.write(struct.pack("<q", len(shape)))
        f.write(struct.pack(f"<{len(shape)}q", *shape))
        f.write(struct.pack("<i", tag))
        f.write(raw)


def fp32_to_bf16_bits(arr_fp32: np.ndarray) -> np.ndarray:
    """Convert fp32 → bf16 with truncation-to-zero rounding semantics, returning
    raw uint16 bits. We round-trip through torch.bfloat16 to get the exact same
    rounding HF used during its bf16 forward pass."""
    t = torch.from_numpy(np.ascontiguousarray(arr_fp32)).to(torch.bfloat16)
    # torch.bfloat16 -> uint16 raw bits: view as int16, take unsigned twos-comp bits.
    return t.view(torch.int16).cpu().numpy().astype(np.uint16, copy=False)


def main() -> None:
    acts_fp32 = np.load(FIX / "activations_fp32.npz")
    acts_bf16 = np.load(FIX / "activations_bf16.npz")
    weights = load_file(str(FIX / "llama_t3_weights_fp32.safetensors"))

    # ---- fp32 case ----
    inp_fp32 = acts_fp32["input_embeds"].astype(np.float32)
    out_fp32 = acts_fp32["layer.0.input_layernorm_out"].astype(np.float32)
    wgt_fp32 = weights["layers.0.input_layernorm.weight"].cpu().numpy().astype(np.float32)

    write_tensor(OUT / "input_fp32.bin", inp_fp32)
    write_tensor(OUT / "weight_fp32.bin", wgt_fp32)
    write_tensor(OUT / "expected_fp32.bin", out_fp32)

    print(f"[fp32] input    {inp_fp32.shape}  -> {OUT/'input_fp32.bin'}")
    print(f"[fp32] weight   {wgt_fp32.shape}  -> {OUT/'weight_fp32.bin'}")
    print(f"[fp32] expected {out_fp32.shape}  -> {OUT/'expected_fp32.bin'}")
    print(f"  first input :  {inp_fp32.flatten()[:4]}")
    print(f"  first expect:  {out_fp32.flatten()[:4]}")

    # ---- bf16 case ----
    # Input is the bf16-rounded version of the embed (matches what the bf16
    # forward pass actually fed to layer 0).
    inp_bf16_fp = acts_bf16["input_embeds"].astype(np.float32)
    out_bf16_fp = acts_bf16["layer.0.input_layernorm_out"].astype(np.float32)
    inp_bf16_bits = fp32_to_bf16_bits(inp_bf16_fp)
    out_bf16_bits = fp32_to_bf16_bits(out_bf16_fp)
    wgt_bf16_bits = fp32_to_bf16_bits(wgt_fp32)

    write_tensor(OUT / "input_bf16.bin", inp_bf16_bits)
    write_tensor(OUT / "weight_bf16.bin", wgt_bf16_bits)
    write_tensor(OUT / "expected_bf16.bin", out_bf16_bits)

    print(f"[bf16] input    {inp_bf16_bits.shape}  -> {OUT/'input_bf16.bin'}")
    print(f"[bf16] weight   {wgt_bf16_bits.shape}  -> {OUT/'weight_bf16.bin'}")
    print(f"[bf16] expected {out_bf16_bits.shape}  -> {OUT/'expected_bf16.bin'}")
    print(f"  first input :  {inp_bf16_fp.flatten()[:4]}  (decoded fp32)")
    print(f"  first expect:  {out_bf16_fp.flatten()[:4]}  (decoded fp32)")
    print(f"  first input bits:  {[hex(int(b)) for b in inp_bf16_bits.flatten()[:4]]}")


if __name__ == "__main__":
    main()
