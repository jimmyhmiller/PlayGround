"""Reproduce the LCG noise in Python, feed it to upstream's CFM, see if the
output mel is sane. Isolates whether the LCG distribution itself is the problem
or whether the Mojo CFM solver diverges given different noise.

LCG implementation matches max-impl/src/cfm_estimator_new.mojo gaussian_noise_fill.
"""
import os, struct
import numpy as np
import torch


def lcg_normal(n: int, seed: int = 0xC0FFEE, sigma: float = 1.0) -> np.ndarray:
    """Mirror the Mojo LCG Box-Muller used by gaussian_noise_fill."""
    out = np.empty(n, dtype=np.float32)
    A = np.uint64(6364136223846793005)
    C = np.uint64(1442695040888963407)
    KNUTH = np.uint64(2654435761)
    for i in range(n):
        s = np.uint64(seed) ^ (np.uint64(i) * KNUTH)
        s = s * A + C
        bits1 = (s >> np.uint64(40)) & np.uint64(0xFFFFFF)
        u1 = (float(int(bits1)) + 0.5) / 16777216.0
        s = s * A + C
        bits2 = (s >> np.uint64(40)) & np.uint64(0xFFFFFF)
        u2 = (float(int(bits2)) + 0.5) / 16777216.0
        r = np.sqrt(-2.0 * np.log(u1))
        z = r * np.cos(2.0 * np.pi * u2) * sigma
        out[i] = z
    return out


def main():
    from chatterbox.tts import ChatterboxTTS

    print("Loading Chatterbox (cpu)...")
    m = ChatterboxTTS.from_pretrained("cpu")
    m.prepare_conditionals("/home/jimmyhmiller/.config/paper-audiobooks/default-voice.wav")
    s3gen_ref = m.conds.gen

    pt = s3gen_ref["prompt_token"]            # (1, 250)
    pf = s3gen_ref["prompt_feat"]             # (1, 500, 80)
    emb = s3gen_ref["embedding"]              # (1, 192)

    # Use SAME speech_tokens as the prior dump so we're testing only the CFM-noise effect.
    with open("weights/s3gen_prompt/speech_tokens.bin", "rb") as f:
        rank = struct.unpack("<q", f.read(8))[0]
        shape = struct.unpack(f"<{rank}q", f.read(8 * rank))
        tag = struct.unpack("<i", f.read(4))[0]
        st_np = np.frombuffer(f.read(), dtype=np.int64).reshape(shape)
    st = torch.from_numpy(st_np.copy()).long()
    print(f"loaded speech_tokens shape={st.shape}")

    # Build the same flow_inference inputs upstream uses.
    s3gen = m.s3gen
    flow = s3gen.flow

    # Replicate flow.inference up to right before randn_like.
    import torch.nn.functional as F
    embedding_normed = F.normalize(torch.atleast_2d(emb), dim=1)
    spks = flow.spk_embed_affine_layer(embedding_normed)  # (1, 80)
    token = torch.cat([pt, st.unsqueeze(0) if st.dim() == 1 else st], dim=1)  # (1, T_total_token)
    print(f"token shape={token.shape}")

    token_len = torch.tensor([token.shape[1]], dtype=torch.long)
    h, _ = flow.encoder(flow.input_embedding(token).contiguous(), token_len)
    h = flow.encoder_proj(h)
    mu = h.transpose(1, 2).contiguous()
    print(f"mu shape={mu.shape}")

    B, _, T_total = mu.shape

    cond = torch.zeros_like(mu)
    cond[:, :, : pf.shape[1]] = pf.transpose(1, 2)

    mask = torch.ones(B, 1, T_total, dtype=mu.dtype)

    # ── Run CFM with LCG-derived noise (matches Mojo)
    n_elem = B * 80 * T_total
    z_lcg = torch.from_numpy(lcg_normal(n_elem, seed=0xC0FFEE)).reshape(B, 80, T_total)
    print(f"LCG noise: mean={z_lcg.mean().item():.4f} std={z_lcg.std().item():.4f} "
          f"min={z_lcg.min().item():.3f} max={z_lcg.max().item():.3f}")

    cfm = flow.decoder
    print("Running upstream CFM with LCG noise...")
    with torch.inference_mode():
        # Replicate CausalConditionalCFM.forward but inject z manually.
        t_span = torch.linspace(0, 1, 11, dtype=mu.dtype)
        t_span = 1 - torch.cos(t_span * 0.5 * torch.pi)
        out = cfm.solve_euler(z_lcg, t_span=t_span, mu=mu, mask=mask, spks=spks, cond=cond)
    print(f"CFM out (full): shape={out.shape} mean-abs={out.abs().mean().item():.4f} "
          f"min={out.min().item():.3f} max={out.max().item():.3f}")
    trimmed = out[:, :, pf.shape[1]:]
    print(f"CFM out (trimmed): shape={trimmed.shape} mean-abs={trimmed.abs().mean().item():.4f} "
          f"min={trimmed.min().item():.3f} max={trimmed.max().item():.3f}")

    # Run HiFT on it.
    # Dump LCG noise + upstream CFM output for cross-check against Mojo.
    os.makedirs("weights/s3gen_prompt/lcg_diag", exist_ok=True)
    def _dump_fp32(path, arr):
        arr = np.ascontiguousarray(arr.astype(np.float32))
        with open(path, "wb") as f:
            f.write(struct.pack("<q", len(arr.shape)))
            f.write(struct.pack(f"<{len(arr.shape)}q", *arr.shape))
            f.write(struct.pack("<i", 0))
            f.write(arr.tobytes())
    _dump_fp32("weights/s3gen_prompt/lcg_diag/lcg_noise.bin", z_lcg.cpu().numpy())
    _dump_fp32("weights/s3gen_prompt/lcg_diag/upstream_cfm_mel_full.bin", out.cpu().numpy())
    _dump_fp32("weights/s3gen_prompt/lcg_diag/upstream_cfm_mel_trim.bin", trimmed.cpu().numpy())
    print(f"Dumped LCG noise + upstream CFM output to weights/s3gen_prompt/lcg_diag/")

    print("Running upstream HiFT on trimmed mel...")
    hift = s3gen.mel2wav
    with torch.inference_mode():
        wav, _ = hift.inference(trimmed)
    print(f"audio max-abs={wav.abs().max().item():.4f} mean-abs={wav.abs().mean().item():.4f}")

    # And compare with torch.randn noise of the same shape.
    print("--- Reference: torch.randn noise ---")
    torch.manual_seed(0)
    z_torch = torch.randn_like(mu)
    print(f"torch.randn: mean={z_torch.mean().item():.4f} std={z_torch.std().item():.4f} "
          f"min={z_torch.min().item():.3f} max={z_torch.max().item():.3f}")
    with torch.inference_mode():
        out2 = cfm.solve_euler(z_torch, t_span=t_span, mu=mu, mask=mask, spks=spks, cond=cond)
    trimmed2 = out2[:, :, pf.shape[1]:]
    print(f"CFM out (trimmed): mean-abs={trimmed2.abs().mean().item():.4f} "
          f"min={trimmed2.min().item():.3f} max={trimmed2.max().item():.3f}")
    with torch.inference_mode():
        wav2, _ = hift.inference(trimmed2)
    print(f"audio max-abs={wav2.abs().max().item():.4f} mean-abs={wav2.abs().mean().item():.4f}")

    # Save both wavs for listening.
    from scipy.io import wavfile
    pcm = (wav.squeeze(0).cpu().numpy() * 32767.0).clip(-32768, 32767).astype(np.int16)
    wavfile.write("weights/s3gen_prompt/upstream_cfm_with_lcg.wav", 24000, pcm)
    pcm2 = (wav2.squeeze(0).cpu().numpy() * 32767.0).clip(-32768, 32767).astype(np.int16)
    wavfile.write("weights/s3gen_prompt/upstream_cfm_with_randn.wav", 24000, pcm2)
    print("Wrote upstream_cfm_with_{lcg,randn}.wav")


if __name__ == "__main__":
    main()
