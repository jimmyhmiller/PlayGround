"""Feed upstream's CFM the EXACT noise that Mojo's gaussian_noise_fill produces
(dumped via tests/dump_mojo_lcg_noise.mojo). If upstream's CFM produces clean
audio with this noise, the LCG distribution itself is fine and the bug is
elsewhere in the Mojo pipeline. If upstream also clips, the noise is broken.
"""
import os, struct
import numpy as np
import torch
import torch.nn.functional as F


def read_fp32(path):
    with open(path, "rb") as f:
        rank = struct.unpack("<q", f.read(8))[0]
        shape = struct.unpack(f"<{rank}q", f.read(8 * rank))
        tag = struct.unpack("<i", f.read(4))[0]
        assert tag == 0
        data = np.frombuffer(f.read(), dtype=np.float32).copy()
    return data.reshape(shape) if rank > 0 else data


def main():
    from chatterbox.tts import ChatterboxTTS

    print("Loading Chatterbox (cpu)...")
    m = ChatterboxTTS.from_pretrained("cpu")
    m.prepare_conditionals("/home/jimmyhmiller/.config/paper-audiobooks/default-voice.wav")
    s3gen_ref = m.conds.gen

    pt = s3gen_ref["prompt_token"]
    pf = s3gen_ref["prompt_feat"]
    emb = s3gen_ref["embedding"]

    # speech_tokens is i64; load it that way
    with open("weights/s3gen_prompt/speech_tokens.bin", "rb") as f:
        rank = struct.unpack("<q", f.read(8))[0]
        shape = struct.unpack(f"<{rank}q", f.read(8 * rank))
        tag = struct.unpack("<i", f.read(4))[0]
        st_arr = np.frombuffer(f.read(), dtype=np.int64).reshape(shape).copy()
    st = torch.from_numpy(st_arr).long()

    s3gen = m.s3gen
    flow = s3gen.flow

    embedding_normed = F.normalize(torch.atleast_2d(emb), dim=1)
    spks = flow.spk_embed_affine_layer(embedding_normed)
    token = torch.cat([pt, st.unsqueeze(0) if st.dim() == 1 else st], dim=1)
    token_len = torch.tensor([token.shape[1]], dtype=torch.long)

    with torch.inference_mode():
        h, _ = flow.encoder(flow.input_embedding(token).contiguous(), token_len)
        h = flow.encoder_proj(h)
    mu = h.transpose(1, 2).contiguous()
    B, _, T_total = mu.shape
    print(f"mu shape={mu.shape}")

    cond = torch.zeros_like(mu)
    cond[:, :, : pf.shape[1]] = pf.transpose(1, 2)
    mask = torch.ones(B, 1, T_total, dtype=mu.dtype)

    # Load the EXACT mojo LCG noise.
    z_mojo = read_fp32("weights/s3gen_prompt/lcg_diag/mojo_lcg_noise.bin")
    print(f"mojo LCG noise: shape={z_mojo.shape} mean={z_mojo.mean():.4f} "
          f"std={z_mojo.std():.4f} min={z_mojo.min():.3f} max={z_mojo.max():.3f}")

    # Reshape to (B, 80, T_total)
    assert z_mojo.size == B * 80 * T_total, f"size mismatch: {z_mojo.size} vs {B*80*T_total}"
    z = torch.from_numpy(z_mojo).reshape(B, 80, T_total)

    cfm = flow.decoder
    print("Running upstream CFM with mojo LCG noise...")
    with torch.inference_mode():
        t_span = torch.linspace(0, 1, 11, dtype=mu.dtype)
        t_span = 1 - torch.cos(t_span * 0.5 * torch.pi)
        out = cfm.solve_euler(z, t_span=t_span, mu=mu, mask=mask, spks=spks, cond=cond)
    trimmed = out[:, :, pf.shape[1]:]
    print(f"CFM out (trimmed): shape={trimmed.shape} "
          f"mean-abs={trimmed.abs().mean().item():.4f} "
          f"min={trimmed.min().item():.3f} max={trimmed.max().item():.3f}")

    # Dump upstream's mel for this exact noise (1D fp32 fixture format).
    def _dump_fp32(path, arr):
        arr = np.ascontiguousarray(arr.astype(np.float32))
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "wb") as f:
            f.write(struct.pack("<q", len(arr.shape)))
            f.write(struct.pack(f"<{len(arr.shape)}q", *arr.shape))
            f.write(struct.pack("<i", 0))
            f.write(arr.tobytes())
    _dump_fp32("weights/s3gen_prompt/lcg_diag/upstream_mel_from_mojo_lcg.bin", out.cpu().numpy())
    _dump_fp32("weights/s3gen_prompt/lcg_diag/upstream_mel_trim_from_mojo_lcg.bin", trimmed.cpu().numpy())

    print("Running upstream HiFT on trimmed mel...")
    hift = s3gen.mel2wav

    # Hook conv_post + capture sine_merge / s_stft.
    captures = {}
    def cap(name):
        def hook(mod, inp, out):
            if isinstance(out, tuple):
                captures[name] = tuple(o.detach().clone() if torch.is_tensor(o) else o for o in out)
            else:
                captures[name] = out.detach().clone()
        return hook
    h_post = hift.conv_post.register_forward_hook(cap("conv_post"))
    h_src = hift.m_source.register_forward_hook(cap("m_source"))

    orig_decode = hift.decode
    def my_decode(x, s):
        sr, si = hift._stft(s.squeeze(1))
        captures["s_stft"] = torch.cat([sr, si], dim=1).detach().clone()
        return orig_decode(x, s)
    hift.decode = my_decode

    with torch.inference_mode():
        wav, _ = hift.inference(trimmed)
    h_post.remove(); h_src.remove()

    src = captures["m_source"]
    sine = src[0] if isinstance(src, tuple) else src
    s_stft = captures["s_stft"]
    print(f"sine_merge: shape={sine.shape} min={sine.min().item():.4f} "
          f"max={sine.max().item():.4f} mean-abs={sine.abs().mean().item():.5f}")
    print(f"s_stft: shape={s_stft.shape} min={s_stft.min().item():.4f} "
          f"max={s_stft.max().item():.4f} mean-abs={s_stft.abs().mean().item():.5f}")
    _dump_fp32("weights/s3gen_prompt/lcg_diag/upstream_sine_merge.bin", sine.cpu().numpy())
    _dump_fp32("weights/s3gen_prompt/lcg_diag/upstream_s_stft.bin", s_stft.cpu().numpy())

    cp = captures["conv_post"]
    print(f"conv_post_out: min={cp.min().item():.4f} max={cp.max().item():.4f} "
          f"mean={cp.mean().item():.4f}")
    _dump_fp32("weights/s3gen_prompt/lcg_diag/upstream_conv_post.bin", cp.cpu().numpy())
    # split into log-magnitude and phase
    log_mag = cp[:, :9, :]
    phase = cp[:, 9:, :]
    print(f"  log_mag: min={log_mag.min().item():.4f} max={log_mag.max().item():.4f}")
    print(f"  phase:   min={phase.min().item():.4f} max={phase.max().item():.4f}")
    print(f"audio max-abs={wav.abs().max().item():.4f}")

    from scipy.io import wavfile
    pcm = (wav.squeeze(0).cpu().numpy() * 32767.0).clip(-32768, 32767).astype(np.int16)
    wavfile.write("weights/s3gen_prompt/lcg_diag/upstream_with_mojo_lcg.wav", 24000, pcm)
    print("Wrote upstream_with_mojo_lcg.wav")


if __name__ == "__main__":
    main()
