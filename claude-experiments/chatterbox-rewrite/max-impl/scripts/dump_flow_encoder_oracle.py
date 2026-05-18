"""Dump a single forward pass of the upstream UpsampleConformerEncoder for
fixed deterministic inputs.

Writes:
  flow_enc_parity/token_ids.bin    (B, T_in)   int64
  flow_enc_parity/expected.bin     (B, T_out, D)  encoder output (B, 2T_in, 512)
  flow_enc_parity/expected_mu.bin  (B, 80, T_out) encoder_proj output (post-transpose)

Also dumps several intermediates so we can isolate bugs:
  embed_out.bin           output of embed(LinearNoSubsampling)
  pre_lookahead_out.bin   output of pre_lookahead
  enc_layer0_out.bin      output of first encoder layer
  up_layer_out.bin        output of up_layer (interpolate + conv)
  after_norm_out.bin      output of after_norm
"""
import os, struct
import numpy as np
import torch

CKPT = "/home/jimmyhmiller/.cache/huggingface/hub/models--ResembleAI--chatterbox/snapshots/ef85ce7bef2f3f1a74d0d837d379d2fcb68203cd"
OUT = "weights/flow_enc_parity"


def write_tensor(path, arr):
    arr = np.ascontiguousarray(np.asarray(arr, dtype=np.float32))
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "wb") as f:
        f.write(struct.pack("<q", len(arr.shape)))
        f.write(struct.pack(f"<{len(arr.shape)}q", *arr.shape))
        f.write(struct.pack("<i", 0))
        f.write(arr.tobytes())


def write_i64(path, arr):
    arr = np.ascontiguousarray(np.asarray(arr, dtype=np.int64))
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "wb") as f:
        f.write(struct.pack("<q", len(arr.shape)))
        f.write(struct.pack(f"<{len(arr.shape)}q", *arr.shape))
        # Tag = 2 = i64, matching src/fixture.mojo:load_i64
        f.write(struct.pack("<i", 2))
        f.write(arr.tobytes())


def main():
    from chatterbox.models.s3gen.transformer.upsample_encoder import UpsampleConformerEncoder

    torch.manual_seed(0)

    enc = UpsampleConformerEncoder(
        input_size=512,
        output_size=512,
        attention_heads=8,
        linear_units=2048,
        num_blocks=6,
        positionwise_conv_kernel_size=1,
    )
    # Build input_embedding manually since the encoder doesn't own it.
    input_embedding = torch.nn.Embedding(6561, 512)
    encoder_proj = torch.nn.Linear(512, 80)

    from safetensors.torch import safe_open
    p = os.path.join(CKPT, "s3gen.safetensors")
    enc_state = {}
    ie_state = {}
    ep_state = {}
    with safe_open(p, framework="pt") as f:
        for k in f.keys():
            if k == "flow.input_embedding.weight":
                ie_state["weight"] = f.get_tensor(k)
            elif k == "flow.encoder_proj.weight":
                ep_state["weight"] = f.get_tensor(k)
            elif k == "flow.encoder_proj.bias":
                ep_state["bias"] = f.get_tensor(k)
            elif k.startswith("flow.encoder."):
                enc_state[k[len("flow.encoder."):]] = f.get_tensor(k)
    missing, unexpected = enc.load_state_dict(enc_state, strict=False)
    print(f"encoder: missing={len(missing)} unexpected={len(unexpected)}")
    if missing[:5]:
        for m in missing[:5]: print("  missing:", m)
    if unexpected[:5]:
        for u in unexpected[:5]: print("  unexpected:", u)
    input_embedding.load_state_dict(ie_state)
    encoder_proj.load_state_dict(ep_state)
    enc.eval()

    B, T_in = 1, 4
    token_ids = torch.tensor([[(i * 137 + 42) % 6561 for i in range(T_in)]], dtype=torch.long)
    xs_lens = torch.tensor([T_in], dtype=torch.long)

    intermediates = {}
    def make_hook(name):
        def h(m, i, o):
            if isinstance(o, tuple): o = o[0]
            intermediates[name] = o.detach().cpu().numpy().copy()
        return h

    enc.embed.register_forward_hook(make_hook("embed_out"))
    enc.pre_lookahead_layer.register_forward_hook(make_hook("pre_lookahead_out"))
    enc.encoders[0].register_forward_hook(make_hook("enc_layer0_out"))
    enc.up_layer.register_forward_hook(make_hook("up_layer_out"))
    enc.up_embed.register_forward_hook(make_hook("up_embed_out"))
    enc.up_encoders[0].register_forward_hook(make_hook("up_layer0_out"))
    enc.after_norm.register_forward_hook(make_hook("after_norm_out"))

    with torch.inference_mode():
        x_emb = input_embedding(token_ids)  # (B, T, D)
        out, masks = enc(x_emb, xs_lens)
        # encoder_proj applied on (B, T_up, D) → (B, T_up, 80).
        mu = encoder_proj(out)
        # Transpose to (B, 80, T_up).
        mu = mu.transpose(1, 2).contiguous()

    print("encoder out shape:", out.shape, "mean-abs:", out.abs().mean().item())
    print("mu shape:", mu.shape, "mean-abs:", mu.abs().mean().item())

    write_i64(f"{OUT}/token_ids.bin", token_ids.numpy())
    write_tensor(f"{OUT}/expected_enc.bin", out.numpy())
    write_tensor(f"{OUT}/expected_mu.bin", mu.numpy())
    for k, v in intermediates.items():
        if isinstance(v, tuple): v = v[0]
        print(f"  {k}: shape={v.shape} mean-abs={np.abs(v).mean():.4f}")
        write_tensor(f"{OUT}/{k}.bin", v)
    print(f"wrote oracle to {OUT}/")


if __name__ == "__main__":
    main()
