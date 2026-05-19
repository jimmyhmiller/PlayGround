"""Dump s3tokenizer mel + tokens for parity testing."""
import os, struct
import numpy as np
import librosa
import torch


def write_tensor(path, arr):
    arr = np.ascontiguousarray(arr.astype(np.float32))
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "wb") as f:
        f.write(struct.pack("<q", len(arr.shape)))
        f.write(struct.pack(f"<{len(arr.shape)}q", *arr.shape))
        f.write(struct.pack("<i", 0))
        f.write(arr.tobytes())


def write_i64(path, arr):
    arr = np.ascontiguousarray(arr.astype(np.int64))
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "wb") as f:
        f.write(struct.pack("<q", len(arr.shape)))
        f.write(struct.pack(f"<{len(arr.shape)}q", *arr.shape))
        f.write(struct.pack("<i", 2))
        f.write(arr.tobytes())


def main():
    OUT = "weights/s3gen_prompt/s3tok_diag"
    os.makedirs(OUT, exist_ok=True)

    from chatterbox.tts import ChatterboxTTS
    m = ChatterboxTTS.from_pretrained("cpu")
    s3tok = m.s3gen.tokenizer
    s3tok.eval()

    ref_path = "/home/jimmyhmiller/.config/paper-audiobooks/default-voice.wav"
    wav_24, _ = librosa.load(ref_path, sr=24000)
    wav_16 = librosa.resample(wav_24, orig_sr=24000, target_sr=16000)
    print(f"wav_16 shape: {wav_16.shape}")
    write_tensor(f"{OUT}/wav_16k.bin", wav_16)

    # Hook intermediates inside encoder.
    captures = {}
    def cap(name):
        def hook(m, inp, out):
            captures[name] = out.detach().clone() if torch.is_tensor(out) else out
        return hook
    enc = s3tok.encoder
    enc.conv1.register_forward_hook(cap("conv1"))
    enc.conv2.register_forward_hook(cap("conv2"))
    enc.blocks[0].register_forward_hook(cap("block0"))
    enc.blocks[-1].register_forward_hook(cap("block_last"))

    # Capture block0's attention layer-norm input and output.
    b0 = enc.blocks[0]
    b0.attn_ln.register_forward_hook(cap("b0_attn_ln"))
    b0.attn.query.register_forward_hook(cap("b0_q"))
    b0.attn.key.register_forward_hook(cap("b0_k"))
    b0.attn.value.register_forward_hook(cap("b0_v"))
    b0.attn.out.register_forward_hook(cap("b0_attn_out"))
    b0.attn.fsmn_block.register_forward_hook(cap("b0_fsmn_block"))
    # MLP layers.
    b0.mlp_ln.register_forward_hook(cap("b0_mlp_ln"))

    # Monkey-patch forward_fsmn to capture its output (which is what gets added to attn).
    orig_fsmn_fwd = b0.attn.forward_fsmn
    def capture_fsmn_fwd(inputs, mask=None):
        out = orig_fsmn_fwd(inputs, mask)
        captures["b0_fsm_memory"] = out.detach().clone()
        return out
    b0.attn.forward_fsmn = capture_fsmn_fwd

    with torch.inference_mode():
        wav_16_t = torch.from_numpy(wav_16).unsqueeze(0)
        # log_mel_spectrogram
        log_mel = s3tok.log_mel_spectrogram(wav_16_t)  # (1, 128, T)
        # full forward
        tokens, tok_lens = s3tok(wav_16_t)

    print(f"log_mel shape: {log_mel.shape} mean={log_mel.mean().item():.4f}")
    print(f"tokens shape: {tokens.shape} (first 20: {tokens[0, :20].tolist()})")
    write_tensor(f"{OUT}/log_mel_16k.bin", log_mel.numpy())
    write_i64(f"{OUT}/tokens.bin", tokens.long().cpu().numpy())

    for k, v in captures.items():
        if isinstance(v, tuple):
            v = v[0]
        if torch.is_tensor(v):
            print(f"  capture {k}: shape={v.shape} mean-abs={v.abs().mean().item():.4f}")
            write_tensor(f"{OUT}/{k}.bin", v.cpu().numpy())

    # Dump the precomputed freqs_cis so we can compare RoPE tables.
    freqs_cis = enc.freqs_cis  # (1024*2, 64) complex
    real = torch.view_as_real(freqs_cis)
    cos_full = real[:, :, 0]
    sin_full = real[:, :, 1]
    print(f"  freqs_cis_cos: shape={cos_full.shape}")
    write_tensor(f"{OUT}/rope_cos.bin", cos_full.cpu().numpy())
    write_tensor(f"{OUT}/rope_sin.bin", sin_full.cpu().numpy())

    print(f"Wrote to {OUT}/")


if __name__ == "__main__":
    main()
