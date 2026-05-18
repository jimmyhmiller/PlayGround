"""Dump VoiceEncoder fixture: mels + weights + expected embed.

Standalone reimplementation of VoiceEncoder forward to avoid pulling
librosa/scipy into the env. Matches the upstream forward() exactly.
"""
import os, struct
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

OUT_DIR = os.path.join(os.path.dirname(__file__), "..", "tests", "fixtures", "ve_forward")
os.makedirs(OUT_DIR, exist_ok=True)


def write_tensor(path, arr):
    arr = np.ascontiguousarray(np.asarray(arr, dtype=np.float32))
    with open(path, "wb") as f:
        f.write(struct.pack("<q", len(arr.shape)))
        f.write(struct.pack(f"<{len(arr.shape)}q", *arr.shape))
        f.write(struct.pack("<i", 0))   # fp32 tag
        f.write(arr.tobytes())


class VoiceEncoder(nn.Module):
    """Matches chatterbox/src/chatterbox/models/voice_encoder/voice_encoder.py"""
    def __init__(self, num_mels=40, hidden=256, embed=256, final_relu=True):
        super().__init__()
        self.final_relu = final_relu
        self.lstm = nn.LSTM(num_mels, hidden, num_layers=3, batch_first=True)
        self.proj = nn.Linear(hidden, embed)

    def forward(self, mels):
        _, (h, _) = self.lstm(mels)
        x = self.proj(h[-1])
        if self.final_relu:
            x = F.relu(x)
        return x / torch.linalg.norm(x, dim=1, keepdim=True)


def save(name, t):
    write_tensor(os.path.join(OUT_DIR, name), t)


def main():
    torch.manual_seed(0)
    ve = VoiceEncoder().eval()

    B, T, M = 4, 160, 40
    mels = torch.rand(B, T, M, dtype=torch.float32)

    with torch.inference_mode():
        embed = ve(mels)
    print("mels:", mels.shape, "embed:", embed.shape)
    print("embed[0, :4]:", embed[0, :4].tolist())

    save("mels.bin", mels.numpy())
    save("embed.bin", embed.numpy())

    for i in range(3):
        save(f"weight_ih_l{i}.bin", getattr(ve.lstm, f"weight_ih_l{i}").detach().numpy())
        save(f"weight_hh_l{i}.bin", getattr(ve.lstm, f"weight_hh_l{i}").detach().numpy())
        save(f"bias_ih_l{i}.bin",   getattr(ve.lstm, f"bias_ih_l{i}").detach().numpy())
        save(f"bias_hh_l{i}.bin",   getattr(ve.lstm, f"bias_hh_l{i}").detach().numpy())

    save("proj_weight.bin", ve.proj.weight.detach().numpy())
    save("proj_bias.bin",   ve.proj.bias.detach().numpy())
    print("dumped to", OUT_DIR)


if __name__ == "__main__":
    main()
