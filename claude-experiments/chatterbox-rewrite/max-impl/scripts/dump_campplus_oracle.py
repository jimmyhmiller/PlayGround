"""Dump upstream CAMPPlus xvector backbone output for fixed (B, 320, T) FCM-style input.

We construct just the xvector Sequential trunk and feed it directly the FCM output
shape (B, 320, T). Mojo's xvector_forward takes the same shape.

Writes:
  campplus_parity/x.bin              (B, 320, T_in)
  campplus_parity/expected.bin       (B, 192)  speaker embedding
  + intermediates (tdnn, block1, transit1, ...)
"""
import os, struct
import numpy as np
import torch

CKPT = "/home/jimmyhmiller/.cache/huggingface/hub/models--ResembleAI--chatterbox/snapshots/ef85ce7bef2f3f1a74d0d837d379d2fcb68203cd"
OUT = "weights/campplus_parity"


def write_tensor(path, arr):
    arr = np.ascontiguousarray(np.asarray(arr, dtype=np.float32))
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "wb") as f:
        f.write(struct.pack("<q", len(arr.shape)))
        f.write(struct.pack(f"<{len(arr.shape)}q", *arr.shape))
        f.write(struct.pack("<i", 0))
        f.write(arr.tobytes())


def main():
    from chatterbox.models.s3gen.xvector import CAMPPlus

    torch.manual_seed(0)

    cp = CAMPPlus(feat_dim=80, embedding_size=192)

    from safetensors.torch import safe_open
    p = os.path.join(CKPT, "s3gen.safetensors")
    state = {}
    with safe_open(p, framework="pt") as f:
        for k in f.keys():
            if k.startswith("speaker_encoder."):
                state[k[len("speaker_encoder."):]] = f.get_tensor(k)
    missing, unexpected = cp.load_state_dict(state, strict=False)
    print(f"missing={len(missing)} unexpected={len(unexpected)}")
    if missing[:5]:
        for m in missing[:5]: print("  missing:", m)
    if unexpected[:5]:
        for u in unexpected[:5]: print("  unexpected:", u)
    cp.eval()

    # The FCM 2D head takes mel and emits (B, m_channels * feat_dim/8, T) = (B, 320, T).
    # We can't easily call cp.xvector directly because it needs the post-FCM shape.
    # Build input matching Mojo's smoke test: (B, 320, T_in).
    B, T_IN = 1, 16
    x = torch.zeros(B, 320, T_IN)
    for c in range(320):
        for ti in range(T_IN):
            x[0, c, ti] = np.sin(c * 0.05 + ti * 0.1) * 0.1

    intermediates = {}
    def make_hook(name):
        def h(m, i, o):
            if isinstance(o, tuple): o = o[0]
            intermediates[name] = o.detach().cpu().numpy().copy()
        return h

    cp.xvector.tdnn.register_forward_hook(make_hook("tdnn_out"))
    cp.xvector.block1.tdnnd1.register_forward_hook(make_hook("tdnnd1_out"))
    cp.xvector.block1.tdnnd1.nonlinear1.register_forward_hook(make_hook("tdnnd1_nl1_out"))
    cp.xvector.block1.tdnnd1.linear1.register_forward_hook(make_hook("tdnnd1_lin1_out"))
    cp.xvector.block1.tdnnd1.nonlinear2.register_forward_hook(make_hook("tdnnd1_nl2_out"))
    cp.xvector.block1.tdnnd1.cam_layer.register_forward_hook(make_hook("tdnnd1_cam_out"))
    cp.xvector.block1.register_forward_hook(make_hook("block1_out"))
    cp.xvector.transit1.register_forward_hook(make_hook("transit1_out"))
    cp.xvector.block2.register_forward_hook(make_hook("block2_out"))
    cp.xvector.block2.tdnnd1.cam_layer.register_forward_hook(make_hook("block2_tdnnd1_cam_out"))
    cp.xvector.block2.tdnnd1.cam_layer.linear_local.register_forward_hook(make_hook("block2_tdnnd1_cam_linloc_out"))
    cp.xvector.transit2.register_forward_hook(make_hook("transit2_out"))
    cp.xvector.block3.register_forward_hook(make_hook("block3_out"))
    cp.xvector.transit3.register_forward_hook(make_hook("transit3_out"))
    cp.xvector.out_nonlinear.register_forward_hook(make_hook("out_nonlinear_out"))
    cp.xvector.stats.register_forward_hook(make_hook("stats_out"))
    cp.xvector.dense.register_forward_hook(make_hook("dense_out"))

    with torch.inference_mode():
        # Manually run xvector trunk on our (B, 320, T) input.
        embed = cp.xvector(x)

    print("embed shape:", embed.shape, "mean-abs:", embed.abs().mean().item())

    write_tensor(f"{OUT}/x.bin", x.numpy())
    write_tensor(f"{OUT}/expected.bin", embed.numpy())
    for name, v in intermediates.items():
        print(f"  {name}: shape={v.shape} mean-abs={np.abs(v).mean():.4f}")
        write_tensor(f"{OUT}/{name}.bin", v)
    print(f"wrote oracle to {OUT}/")


if __name__ == "__main__":
    main()
