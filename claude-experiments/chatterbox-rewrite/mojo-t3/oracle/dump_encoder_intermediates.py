"""
Dump per-stage intermediates inside the UpsampleConformerEncoder so the Mojo
port can do parity testing one sub-block at a time.

Loads the existing flow_token_emb.bin fixture (which is the encoder input) and
runs flow.encoder stage by stage, dumping after each stage.
"""
from __future__ import annotations
import os, struct
from pathlib import Path
import numpy as np
import torch

os.environ.setdefault("MIOPEN_DEBUG_CONV_WINOGRAD", "0")
from chatterbox.tts import ChatterboxTTS

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "tests" / "fixtures" / "s3gen"
OUT.mkdir(parents=True, exist_ok=True)


def write_tensor(p: Path, arr: np.ndarray) -> None:
    if arr.dtype == np.float32:
        tag, raw = 0, arr.astype(np.float32, copy=False).tobytes()
    elif arr.dtype == np.int64:
        tag, raw = 2, arr.astype(np.int64, copy=False).tobytes()
    else:
        raise TypeError(arr.dtype)
    with p.open("wb") as f:
        f.write(struct.pack("<q", len(arr.shape)))
        f.write(struct.pack(f"<{len(arr.shape)}q", *arr.shape))
        f.write(struct.pack("<i", tag))
        f.write(raw)


def read_fixture(p: Path):
    with p.open("rb") as f:
        ndim = struct.unpack("<q", f.read(8))[0]
        shape = struct.unpack(f"<{ndim}q", f.read(ndim * 8))
        tag = struct.unpack("<i", f.read(4))[0]
        raw = f.read()
    if tag == 0:
        return np.frombuffer(raw, dtype=np.float32).reshape(shape).copy()
    elif tag == 2:
        return np.frombuffer(raw, dtype=np.int64).reshape(shape).copy()
    raise ValueError(tag)


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = ChatterboxTTS.from_pretrained(device=device)

    # Strip parametrizations.
    import torch.nn.utils.parametrize as parametrize
    def strip(m):
        if hasattr(m, "parametrizations"):
            for n in list(m.parametrizations.keys()):
                parametrize.remove_parametrizations(m, n, leave_parametrized=True)
        for c in m.children():
            strip(c)
    strip(model.s3gen)

    encoder = model.s3gen.flow.encoder
    encoder.eval()

    # Load saved flow_token_emb (1, T_in, 512) from previous dump.
    token_emb = torch.from_numpy(read_fixture(OUT / "flow_token_emb.bin")).to(device)
    T_in = token_emb.shape[1]
    print(f"[encoder] input shape: {tuple(token_emb.shape)}")

    # We need to construct the mask from token_len. Since flow_token_emb was
    # produced from the full token tensor (no padding), all positions are valid.
    xs_lens = torch.tensor([T_in], device=device, dtype=torch.long)

    with torch.inference_mode():
        # Reproduce encoder.forward but capture intermediates.

        # In CosyVoice encoder.embed forward:
        #   xs is (B, T, D=512) input
        #   produces (xs_subsampled, pos_emb, masks)
        # In LinearNoSubsampling: xs = out(xs) where out = Sequential(Linear, LayerNorm, Dropout)
        #   pos_emb = pos_enc(xs) → relative positional encoding
        from chatterbox.models.s3gen.utils.mask import make_pad_mask
        masks = ~make_pad_mask(xs_lens, T_in).unsqueeze(1)
        xs_emb, pos_emb, masks2 = encoder.embed(token_emb, masks)
        print(f"  embed out: xs={tuple(xs_emb.shape)} pos_emb={tuple(pos_emb.shape)} masks={tuple(masks2.shape)}")
        write_tensor(OUT / "enc_embed_xs.bin", xs_emb.cpu().numpy().astype(np.float32))
        write_tensor(OUT / "enc_embed_pos.bin", pos_emb.cpu().numpy().astype(np.float32))
        write_tensor(OUT / "enc_embed_mask.bin", masks2.float().cpu().numpy().astype(np.float32))

        # pre_lookahead.
        xs_pl = encoder.pre_lookahead_layer(xs_emb)
        write_tensor(OUT / "enc_pre_lookahead.bin", xs_pl.cpu().numpy().astype(np.float32))

        # Run each encoder layer separately to get sub-intermediates.
        xs = xs_pl
        for i, layer in enumerate(encoder.encoders):
            # For layer 0 only, dump MHA internal pieces.
            if i == 0:
                # Hook self_attn to capture q, k, v, scores, attn, attn_out, etc.
                attn = layer.self_attn
                captured_attn = {}
                orig_qkv = attn.forward_qkv
                def hook_qkv(q_in, k_in, v_in):
                    q, k, v = orig_qkv(q_in, k_in, v_in)
                    captured_attn['q'] = q.detach()
                    captured_attn['k'] = k.detach()
                    captured_attn['v'] = v.detach()
                    return q, k, v
                attn.forward_qkv = hook_qkv

                # Run norm_mha on input and capture.
                x_after_norm_mha = layer.norm_mha(xs)
                captured_attn['norm_mha_out'] = x_after_norm_mha.detach()

                # Run the attention forward manually so we can capture every step.
                import math as _math
                import torch as _torch
                q, k, v = orig_qkv(x_after_norm_mha, x_after_norm_mha, x_after_norm_mha)
                # forward_qkv already transposed to (B, H, T, D_k). Now in RelPos:
                # q = q.transpose(1, 2)  → (B, T, H, D_k)
                q_btHd = q.transpose(1, 2).contiguous()
                captured_attn['q_btHd'] = q_btHd.detach()
                # linear_pos(pos_emb).view → (B, 2T-1, H, D_k).transpose → (B, H, 2T-1, D_k)
                p = attn.linear_pos(pos_emb).view(pos_emb.size(0), -1, attn.h, attn.d_k)
                p = p.transpose(1, 2).contiguous()
                captured_attn['p'] = p.detach()
                # q_with_bias_u = (q + pos_bias_u).transpose(1, 2)  → (B, H, T, D_k)
                q_u = (q_btHd + attn.pos_bias_u).transpose(1, 2).contiguous()
                q_v = (q_btHd + attn.pos_bias_v).transpose(1, 2).contiguous()
                captured_attn['q_u'] = q_u.detach()
                captured_attn['q_v'] = q_v.detach()
                # matrix_ac = q_u @ k^T  → (B, H, T, T)
                matrix_ac = _torch.matmul(q_u, k.transpose(-2, -1))
                captured_attn['matrix_ac'] = matrix_ac.detach()
                # matrix_bd = q_v @ p^T  → (B, H, T, 2T-1)
                matrix_bd = _torch.matmul(q_v, p.transpose(-2, -1))
                captured_attn['matrix_bd_pre_shift'] = matrix_bd.detach()
                # rel_shift if shapes differ
                if matrix_ac.shape != matrix_bd.shape:
                    matrix_bd_shifted = attn.rel_shift(matrix_bd)
                else:
                    matrix_bd_shifted = matrix_bd
                captured_attn['matrix_bd'] = matrix_bd_shifted.detach()
                # scores = (ac + bd) / sqrt(d_k)
                scores = (matrix_ac + matrix_bd_shifted) / _math.sqrt(attn.d_k)
                captured_attn['scores'] = scores.detach()
                # mask path. masks2 is (B, 1, T). forward_attention does mask.unsqueeze(1).eq(0)
                mask_used = masks2
                if mask_used.size(2) > 0:
                    mfill = mask_used.unsqueeze(1).eq(0)
                    mfill = mfill[:, :, :, :scores.size(-1)]
                    scores_masked = scores.masked_fill(mfill, float('-inf'))
                    attn_w = _torch.softmax(scores_masked, dim=-1).masked_fill(mfill, 0.0)
                else:
                    attn_w = _torch.softmax(scores, dim=-1)
                captured_attn['attn'] = attn_w.detach()
                # attn @ v
                ctx_av = _torch.matmul(attn_w, v)  # (B, H, T, D_k)
                captured_attn['ctx_av'] = ctx_av.detach()
                # (B, H, T, D_k) → (B, T, H*D_k)
                ctx_merged = ctx_av.transpose(1, 2).contiguous().view(ctx_av.size(0), -1, attn.h * attn.d_k)
                captured_attn['ctx_merged'] = ctx_merged.detach()
                # linear_out
                attn_out = attn.linear_out(ctx_merged)
                captured_attn['attn_out'] = attn_out.detach()
                # Capture norm_ff input/output and ff sub-steps.
                # After self_attn the layer does: x = residual + attn_out (then dropout).
                x_post_mha = xs + attn_out  # residual is xs (the input to norm_mha, NOT norm_mha_out)
                captured_attn['x_post_mha'] = x_post_mha.detach()
                x_norm_ff = layer.norm_ff(x_post_mha)
                captured_attn['norm_ff_out'] = x_norm_ff.detach()
                # feed_forward = w_1, activation, w_2
                ff = layer.feed_forward
                w1_out = ff.w_1(x_norm_ff)
                captured_attn['ff_w1_out'] = w1_out.detach()
                act_out = ff.activation(w1_out)
                captured_attn['ff_act_out'] = act_out.detach()
                w2_out = ff.w_2(act_out)
                captured_attn['ff_w2_out'] = w2_out.detach()
                # Final layer out = x_post_mha + w2_out (residual)
                final_layer_out = x_post_mha + w2_out
                captured_attn['layer_out_manual'] = final_layer_out.detach()

            xs, _, _, _ = layer(xs, masks2, pos_emb, masks2)
            write_tensor(OUT / f"enc_layer_{i}_out.bin", xs.cpu().numpy().astype(np.float32))

            if i == 0:
                attn.forward_qkv = orig_qkv
                # Sanity: layer_out_manual should match the actual layer 0 output.
                d_manual = (captured_attn['layer_out_manual'] - xs).abs().max().item()
                print(f"  layer0 manual reconstruction vs actual: max abs diff = {d_manual:.6e}")
                for name, t in captured_attn.items():
                    write_tensor(OUT / f"enc_layer_0_{name}.bin", t.cpu().numpy().astype(np.float32))
                    print(f"  layer0 {name}: {tuple(t.shape)}")
            print(f"  encoder {i}: {tuple(xs.shape)}")

        # up_layer: takes (B, T, D) -> .transpose(1,2) -> conv1d(stride=2) -> .transpose
        xs_pre_up = xs.transpose(1, 2).contiguous()
        xs_up, xs_lens2 = encoder.up_layer(xs_pre_up, xs_lens)
        xs_up = xs_up.transpose(1, 2).contiguous()
        T_up = xs_up.shape[1]
        write_tensor(OUT / "enc_up_layer_out.bin", xs_up.cpu().numpy().astype(np.float32))
        print(f"  up_layer: {tuple(xs_up.shape)}")

        # up_embed.
        masks_up = ~make_pad_mask(xs_lens2, T_up).unsqueeze(1)
        xs_ue, pos_emb_up, masks_ue = encoder.up_embed(xs_up, masks_up)
        write_tensor(OUT / "enc_up_embed_xs.bin", xs_ue.cpu().numpy().astype(np.float32))
        write_tensor(OUT / "enc_up_embed_pos.bin", pos_emb_up.cpu().numpy().astype(np.float32))

        # up_encoders.
        xs2 = xs_ue
        for i, layer in enumerate(encoder.up_encoders):
            xs2, _, _, _ = layer(xs2, masks_ue, pos_emb_up, masks_ue)
            write_tensor(OUT / f"enc_up_layer_{i}_out.bin", xs2.cpu().numpy().astype(np.float32))

        # after_norm.
        if encoder.normalize_before:
            xs_final = encoder.after_norm(xs2)
        else:
            xs_final = xs2
        write_tensor(OUT / "enc_after_norm.bin", xs_final.cpu().numpy().astype(np.float32))
        print(f"  after_norm: {tuple(xs_final.shape)}")

        # Sanity: compare against encoder_h.bin.
        expected_h = torch.from_numpy(read_fixture(OUT / "encoder_h.bin")).to(device)
        diff = (xs_final - expected_h).abs().max().item()
        print(f"  reproduced vs encoder_h: max abs diff = {diff:.6e}")


if __name__ == "__main__":
    main()
