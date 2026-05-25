"""Dump a single forward pass of the upstream Chatterbox CFM estimator
(`ConditionalDecoder`) for fixed deterministic inputs, so the Mojo port can
parity-test against it.

Writes (each in the fixture .bin format the Mojo loader uses):
  cfm_parity/x.bin       (B, 80, T)
  cfm_parity/mu.bin      (B, 80, T)
  cfm_parity/spks.bin    (B, 80)
  cfm_parity/cond.bin    (B, 80, T)
  cfm_parity/mask.bin    (B, 1, T)
  cfm_parity/t_scalar.bin (B,)
  cfm_parity/expected.bin (B, 80, T)   # estimator output (velocity field)

The fixture format mirrors `scripts/convert_weights.py:write_tensor`.

Run via:
  ~/.cache/paper-audiobooks/venvs/chatterbox/bin/python \\
    scripts/dump_cfm_oracle.py
"""
import os, struct, sys
import numpy as np
import torch

# Locate the chatterbox checkpoint already on this machine.
CKPT = "/home/jimmyhmiller/.cache/huggingface/hub/models--ResembleAI--chatterbox/snapshots/ef85ce7bef2f3f1a74d0d837d379d2fcb68203cd"
OUT = "weights/cfm_parity"


def write_tensor(path, arr):
    arr = np.ascontiguousarray(np.asarray(arr, dtype=np.float32))
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "wb") as f:
        f.write(struct.pack("<q", len(arr.shape)))
        f.write(struct.pack(f"<{len(arr.shape)}q", *arr.shape))
        f.write(struct.pack("<i", 0))
        f.write(arr.tobytes())


def main():
    from chatterbox.models.s3gen.decoder import ConditionalDecoder

    torch.manual_seed(0)
    B, T = 1, 16

    # Build the estimator with the upstream config.
    decoder = ConditionalDecoder(
        in_channels=320,
        out_channels=80,
        causal=True,
        channels=[256],
        attention_head_dim=64,
        n_blocks=4,
        num_mid_blocks=12,
        num_heads=8,
        act_fn="gelu",
    )

    # Load weights from the .safetensors file. Restrict to flow.decoder.estimator.*
    from safetensors.torch import safe_open
    p = os.path.join(CKPT, "s3gen.safetensors")
    state = {}
    with safe_open(p, framework="pt") as f:
        for k in f.keys():
            if k.startswith("flow.decoder.estimator."):
                state[k[len("flow.decoder.estimator."):]] = f.get_tensor(k)
    missing, unexpected = decoder.load_state_dict(state, strict=False)
    print(f"missing keys: {len(missing)}, unexpected: {len(unexpected)}")
    if missing:
        for m in missing[:10]: print("  missing:", m)
    if unexpected:
        for u in unexpected[:10]: print("  unexpected:", u)
    decoder.eval()

    # Deterministic inputs matching the Mojo smoke test.
    x = torch.zeros(B, 80, T)
    mu = torch.zeros(B, 80, T)
    spks = torch.zeros(B, 80)
    cond = torch.zeros(B, 80, T)
    mask = torch.ones(B, 1, T)
    t_scalar = torch.tensor([0.5])

    for c in range(80):
        for ti in range(T):
            x[0, c, ti] = np.sin(c * 0.05 + ti * 0.1) * 0.1
            mu[0, c, ti] = np.sin(c * 0.07 + ti * 0.13) * 0.1
        spks[0, c] = np.sin(c * 0.11) * 0.1

    # Hook intermediate activations to compare step-by-step in Mojo.
    intermediates = {}

    def make_hook(name):
        def hook(module, inputs, output):
            if isinstance(output, tuple):
                output = output[0]
            intermediates[name] = output.detach().cpu().numpy().copy()
        return hook

    # Time embedding output (after time_mlp).
    decoder.time_mlp.register_forward_hook(make_hook("time_mlp_out"))
    # First down resnet output (pre-transformer).
    decoder.down_blocks[0][0].register_forward_hook(make_hook("down0_resnet_out"))
    # First down transformer block (post-attn+ff).
    decoder.down_blocks[0][1][0].register_forward_hook(make_hook("down0_transformer0_out"))
    # First down block downsampler output.
    decoder.down_blocks[0][2].register_forward_hook(make_hook("down0_downsample_out"))
    # Mid block 0 output.
    decoder.mid_blocks[0][0].register_forward_hook(make_hook("mid0_resnet_out"))
    # Final block output.
    decoder.final_block.register_forward_hook(make_hook("final_block_out"))

    with torch.inference_mode():
        out = decoder(x=x, mask=mask, mu=mu, t=t_scalar, spks=spks, cond=cond)

    print("out shape:", out.shape, "mean-abs:", out.abs().mean().item(),
          "max:", out.abs().max().item())

    for k, v in intermediates.items():
        print(f"  {k}: shape={v.shape} mean-abs={np.abs(v).mean():.4f}")
        write_tensor(f"{OUT}/{k}.bin", v)

    write_tensor(f"{OUT}/x.bin", x.numpy())
    write_tensor(f"{OUT}/mu.bin", mu.numpy())
    write_tensor(f"{OUT}/spks.bin", spks.numpy())
    write_tensor(f"{OUT}/cond.bin", cond.numpy())
    write_tensor(f"{OUT}/mask.bin", mask.numpy())
    write_tensor(f"{OUT}/t_scalar.bin", t_scalar.numpy())
    write_tensor(f"{OUT}/expected.bin", out.numpy())
    print(f"wrote oracle to {OUT}/")


if __name__ == "__main__":
    main()
