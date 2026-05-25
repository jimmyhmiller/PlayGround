"""Dump upstream source-path intermediates for parity testing.

f0_predictor(mel) → f0 (B, T_mel)
  → upsample by 480 → f0_up (B, T_audio)
  → SourceModuleHnNSF → sine_merge (B, 1, T_audio)
  → STFT (n_fft=16, hop=4) → cat(real, imag) → s_stft (B, 18, T_s_frames)

Inputs: a small synthetic mel (B=1, 80, 60) — to keep memory reasonable while
running upstream's full source path.

This dump uses the same mel my test uses (well — uses sine-deterministic mel
of small size to avoid huge T_audio).
"""
import os, struct
import numpy as np
import torch

CKPT = "/home/jimmyhmiller/.cache/huggingface/hub/models--ResembleAI--chatterbox/snapshots/ef85ce7bef2f3f1a74d0d837d379d2fcb68203cd"
OUT = "weights/source_path_parity"


def write_tensor(path, arr):
    arr = np.ascontiguousarray(np.asarray(arr, dtype=np.float32))
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "wb") as f:
        f.write(struct.pack("<q", len(arr.shape)))
        f.write(struct.pack(f"<{len(arr.shape)}q", *arr.shape))
        f.write(struct.pack("<i", 0))
        f.write(arr.tobytes())


def main():
    from chatterbox.models.s3gen.hifigan import HiFTGenerator
    from chatterbox.models.s3gen.f0_predictor import ConvRNNF0Predictor

    torch.manual_seed(0)

    f0_predictor = ConvRNNF0Predictor()
    hift = HiFTGenerator(
        sampling_rate=24000,
        upsample_rates=[8, 5, 3],
        upsample_kernel_sizes=[16, 11, 7],
        source_resblock_kernel_sizes=[7, 7, 11],
        source_resblock_dilation_sizes=[[1, 3, 5], [1, 3, 5], [1, 3, 5]],
        f0_predictor=f0_predictor,
    )

    from safetensors.torch import safe_open
    p = os.path.join(CKPT, "s3gen.safetensors")
    hift_state = {}
    f0_state = {}
    with safe_open(p, framework="pt") as f:
        for k in f.keys():
            if k.startswith("hift."):
                hift_state[k[len("hift."):]] = f.get_tensor(k)
            elif k.startswith("mel2wav."):
                # mel2wav.f0_predictor.* belongs to f0_predictor; rest to hift.
                if k.startswith("mel2wav.f0_predictor."):
                    f0_state[k[len("mel2wav.f0_predictor."):]] = f.get_tensor(k)
                else:
                    hift_state[k[len("mel2wav."):]] = f.get_tensor(k)
    missing, unexpected = hift.load_state_dict(hift_state, strict=False)
    print(f"hift: missing={len(missing)} unexpected={len(unexpected)}")
    missing2, unexpected2 = f0_predictor.load_state_dict(f0_state, strict=False)
    print(f"f0_predictor: missing={len(missing2)} unexpected={len(unexpected2)}")
    hift.eval()
    f0_predictor.eval()

    B, T_MEL = 1, 60
    mel = torch.zeros(B, 80, T_MEL)
    for c in range(80):
        for ti in range(T_MEL):
            mel[0, c, ti] = np.sin(c * 0.05 + ti * 0.1) * 0.1

    # Hook intermediate condnet outputs.
    intermediates = {}
    def make_hook(name):
        def h(m, i, o):
            if isinstance(o, tuple): o = o[0]
            intermediates[name] = o.detach().cpu().numpy().copy()
        return h
    f0_predictor.condnet[0].register_forward_hook(make_hook("conv0"))
    f0_predictor.condnet[1].register_forward_hook(make_hook("elu0"))
    f0_predictor.condnet[2].register_forward_hook(make_hook("conv1"))
    f0_predictor.classifier.register_forward_hook(make_hook("classifier"))

    with torch.inference_mode():
        f0 = f0_predictor(mel)   # (B, T_MEL)
        # f0_upsamp = nn.Upsample(scale_factor = prod(ups) * hop_len = 8*5*3*4 = 480, mode='nearest')
        f0_up_seq = torch.nn.functional.interpolate(
            f0.unsqueeze(1), scale_factor=480.0, mode="nearest"
        ).transpose(1, 2)   # (B, T_audio, 1)
        # SourceModuleHnNSF takes (B, T_audio, 1) → sine_merge (B, T_audio, 1).
        # SineGen needs (B, 1, T_audio): SourceModuleHnNSF.forward transposes internally.
        sine_merge, noise, uv = hift.m_source(f0_up_seq)   # (B, T_audio, 1)
        sine_merge = sine_merge.transpose(1, 2)            # (B, 1, T_audio)
        # Forward STFT: hift._stft(s.squeeze(1)) → (B, n_freq, n_frames), (...)
        s_stft_real, s_stft_imag = hift._stft(sine_merge.squeeze(1))
        s_stft = torch.cat([s_stft_real, s_stft_imag], dim=1)

    print("f0 shape:", f0.shape, "mean-abs:", f0.abs().mean().item(),
          "max:", f0.abs().max().item())
    print("f0_up shape:", f0_up_seq.shape, "mean-abs:", f0_up_seq.abs().mean().item())
    print("sine_merge shape:", sine_merge.shape, "mean-abs:", sine_merge.abs().mean().item())
    print("s_stft shape:", s_stft.shape, "mean-abs:", s_stft.abs().mean().item())

    for name, v in intermediates.items():
        print(f"  {name}: shape={v.shape}")
        write_tensor(f"{OUT}/{name}.bin", v)

    write_tensor(f"{OUT}/mel.bin", mel.numpy())
    write_tensor(f"{OUT}/f0.bin", f0.numpy())
    write_tensor(f"{OUT}/f0_up.bin", f0_up_seq.squeeze(-1).numpy())   # (B, T_audio)
    write_tensor(f"{OUT}/sine_merge.bin", sine_merge.numpy())
    write_tensor(f"{OUT}/s_stft.bin", s_stft.numpy())
    print(f"wrote to {OUT}/")


if __name__ == "__main__":
    main()
