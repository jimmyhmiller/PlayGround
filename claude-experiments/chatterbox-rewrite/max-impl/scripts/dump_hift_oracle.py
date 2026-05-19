"""Dump upstream HiFTGenerator.decode() forward for fixed mel input + zero
source signal. Disable source path (s=zeros) so parity comparison matches our
Mojo `use_source=False` smoke test.

Writes:
  hift_parity/mel.bin       (B, 80, T_mel)
  hift_parity/expected_spec.bin   (B, 18, T_out)  conv_post output (pre-iSTFT)
  hift_parity/expected_audio.bin  (B, T_audio)    final audio after iSTFT + clamp
"""
import os, struct
import numpy as np
import torch

CKPT = "/home/jimmyhmiller/.cache/huggingface/hub/models--ResembleAI--chatterbox/snapshots/ef85ce7bef2f3f1a74d0d837d379d2fcb68203cd"
OUT = "weights/hift_parity"


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

    torch.manual_seed(0)

    hift = HiFTGenerator(
        sampling_rate=24000,
        upsample_rates=[8, 5, 3],
        upsample_kernel_sizes=[16, 11, 7],
        source_resblock_kernel_sizes=[7, 7, 11],
        source_resblock_dilation_sizes=[[1, 3, 5], [1, 3, 5], [1, 3, 5]],
    )

    from safetensors.torch import safe_open
    p = os.path.join(CKPT, "s3gen.safetensors")
    state = {}
    with safe_open(p, framework="pt") as f:
        for k in f.keys():
            if k.startswith("hift."):
                state[k[len("hift."):]] = f.get_tensor(k)
            elif k.startswith("mel2wav."):
                state[k[len("mel2wav."):]] = f.get_tensor(k)
    missing, unexpected = hift.load_state_dict(state, strict=False)
    print(f"hift: missing={len(missing)} unexpected={len(unexpected)}")
    if missing[:5]:
        for m in missing[:5]: print("  missing:", m)
    if unexpected[:5]:
        for u in unexpected[:5]: print("  unexpected:", u)
    hift.eval()

    # To match Mojo's use_source=False, zero out the source path weights and
    # biases so source_downs[i](s_stft) is the zero tensor and adding source_resblocks
    # (which leaves zero through Snake+conv chains, since Snake(0) = 0 and bias-less
    # convs are bias-less) keeps it zero.
    with torch.no_grad():
        for i in range(3):
            hift.source_downs[i].weight.zero_()
            hift.source_downs[i].bias.zero_()
            for j in range(3):
                hift.source_resblocks[i].convs1[j].weight.zero_()
                hift.source_resblocks[i].convs1[j].bias.zero_()
                hift.source_resblocks[i].convs2[j].weight.zero_()
                hift.source_resblocks[i].convs2[j].bias.zero_()
                hift.source_resblocks[i].activations1[j].alpha.zero_()
                hift.source_resblocks[i].activations2[j].alpha.zero_()
    # Note: even with alpha=0, the Snake formula has (1/alpha) which would NaN.
    # Set alpha to a non-zero small value so the Snake output of zero input
    # stays zero (since sin(0) = 0, regardless of alpha).
    with torch.no_grad():
        for i in range(3):
            for j in range(3):
                hift.source_resblocks[i].activations1[j].alpha.fill_(1.0)
                hift.source_resblocks[i].activations2[j].alpha.fill_(1.0)

    B, T_mel = 1, 4
    MEL = 80
    mel = torch.zeros(B, MEL, T_mel)
    for c in range(MEL):
        for ti in range(T_mel):
            mel[0, c, ti] = np.sin(c * 0.05 + ti * 0.1) * 0.1

    # Hook several intermediates to localize bugs.
    captures = {}

    def make_hook(name):
        def hook(m, i, o):
            if isinstance(o, tuple): o = o[0]
            captures[name] = o.detach().cpu().numpy().copy()
        return hook
    hift.conv_pre.register_forward_hook(make_hook("conv_pre_out"))
    hift.ups[0].register_forward_hook(make_hook("ups0_out"))
    hift.ups[1].register_forward_hook(make_hook("ups1_out"))
    hift.ups[2].register_forward_hook(make_hook("ups2_out"))
    hift.resblocks[0].register_forward_hook(make_hook("resblock0_out"))
    hift.resblocks[1].register_forward_hook(make_hook("resblock1_out"))
    hift.resblocks[2].register_forward_hook(make_hook("resblock2_out"))
    hift.conv_post.register_forward_hook(make_hook("conv_post_out"))
    hift.resblocks[0].convs1[0].register_forward_hook(make_hook("resblock0_conv1_0_out"))
    hift.resblocks[0].activations1[0].register_forward_hook(make_hook("resblock0_act1_0_out"))
    hift.resblocks[6].register_forward_hook(make_hook("resblock6_out"))
    hift.resblocks[7].register_forward_hook(make_hook("resblock7_out"))
    hift.resblocks[8].register_forward_hook(make_hook("resblock8_out"))
    hift.reflection_pad.register_forward_hook(make_hook("reflection_pad_out"))

    # ups_rates [8, 5, 3] = 120; hop=4 → T_audio = T_mel * 120 * 4 = T_mel*480.
    s = torch.zeros(B, 1, T_mel * 480)

    with torch.inference_mode():
        # The forward path needs s_stft, computed inside decode().
        # We use decode() with zero s — produces zero source contribution.
        audio = hift.decode(x=mel, s=s)

    print("audio shape:", audio.shape, "mean-abs:", audio.abs().mean().item(),
          "max-abs:", audio.abs().max().item())
    write_tensor(f"{OUT}/mel.bin", mel.numpy())
    write_tensor(f"{OUT}/expected_spec.bin", captures["conv_post_out"])
    write_tensor(f"{OUT}/expected_audio.bin", audio.numpy())
    for name in ["conv_pre_out", "ups0_out", "ups1_out", "ups2_out",
                  "resblock0_out", "resblock1_out",
                  "resblock2_out", "conv_post_out",
                  "resblock0_act1_0_out", "resblock0_conv1_0_out",
                  "resblock6_out", "resblock7_out", "resblock8_out",
                  "reflection_pad_out"]:
        arr = captures[name]
        print(f"  {name}: shape={arr.shape} mean-abs={np.abs(arr).mean():.4f}")
        write_tensor(f"{OUT}/{name}.bin", arr)
    print(f"wrote oracle to {OUT}/")


if __name__ == "__main__":
    main()
