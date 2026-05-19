"""Run upstream HiFT on the dumped mel and capture every intermediate.

We already have the bit-exact upstream mel in weights/s3gen_prompt/expected_mel.bin.
This script feeds it to upstream's HiFTGenerator.inference and dumps:
  - f0 (B, T_mel)
  - s_after_source_module (B, 1, T_audio_full)
  - s_stft (B, 2*(n_fft//2 + 1), T_s_frames)   = source STFT used inside decode
  - x_after_conv_pre (B, channels, T_mel)
  - generated_speech (B, T_audio)
"""
import os, struct, sys
import numpy as np
import torch

def write_tensor(path, arr):
    arr = np.ascontiguousarray(np.asarray(arr, dtype=np.float32))
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "wb") as f:
        f.write(struct.pack("<q", len(arr.shape)))
        f.write(struct.pack(f"<{len(arr.shape)}q", *arr.shape))
        f.write(struct.pack("<i", 0))
        f.write(arr.tobytes())

def read_tensor(path):
    with open(path, "rb") as f:
        rank = struct.unpack("<q", f.read(8))[0]
        shape = struct.unpack(f"<{rank}q", f.read(8 * rank))
        tag = struct.unpack("<i", f.read(4))[0]
        assert tag == 0
        data = np.frombuffer(f.read(), dtype=np.float32).reshape(shape)
    return data

def main():
    from chatterbox.tts import ChatterboxTTS

    OUT = "weights/s3gen_prompt/hift_dump"
    os.makedirs(OUT, exist_ok=True)

    print("Loading Chatterbox...")
    m = ChatterboxTTS.from_pretrained("cpu")

    mel = read_tensor("weights/s3gen_prompt/expected_mel.bin")  # (1, 80, T)
    print(f"loaded mel shape={mel.shape}")
    mel_t = torch.from_numpy(mel)

    hift = m.s3gen.mel2wav
    hift.eval()

    captured = {}
    def cap(name):
        def hook(mod, inputs, output):
            if isinstance(output, tuple):
                captured[name] = tuple(o.detach().clone() if torch.is_tensor(o) else o for o in output)
            else:
                captured[name] = output.detach().clone() if torch.is_tensor(output) else output
        return hook

    h_f0 = hift.f0_predictor.register_forward_hook(cap("f0"))
    h_src = hift.m_source.register_forward_hook(cap("m_source"))
    h_pre = hift.conv_pre.register_forward_hook(cap("conv_pre"))
    h_post = hift.conv_post.register_forward_hook(cap("conv_post"))

    # Patch decode to capture s_stft
    orig_decode = hift.decode
    s_stft_capture = {}
    def my_decode(x, s):
        sr, si = hift._stft(s.squeeze(1))
        s_stft_capture["s_stft"] = torch.cat([sr, si], dim=1).detach().clone()
        return orig_decode(x, s)
    hift.decode = my_decode

    with torch.inference_mode():
        wav, s_after = hift.inference(mel_t)

    h_f0.remove(); h_src.remove(); h_pre.remove(); h_post.remove()

    f0 = captured["f0"]
    src_out = captured["m_source"]  # tuple (sine_merge, ...)
    sine = src_out[0] if isinstance(src_out, tuple) else src_out
    conv_pre_out = captured["conv_pre"]
    conv_post_out = captured["conv_post"]
    s_stft = s_stft_capture["s_stft"]

    print(f"f0 shape={f0.shape}  mean-abs={f0.abs().mean().item():.4f}")
    print(f"sine_merge shape={sine.shape}  mean-abs={sine.abs().mean().item():.4f}")
    print(f"s (transposed, post-source) shape={s_after.shape}  mean-abs={s_after.abs().mean().item():.4f}")
    print(f"s_stft shape={s_stft.shape}  mean-abs={s_stft.abs().mean().item():.4f}")
    print(f"conv_pre_out shape={conv_pre_out.shape}  mean-abs={conv_pre_out.abs().mean().item():.4f}")
    print(f"conv_post_out shape={conv_post_out.shape}  mean-abs={conv_post_out.abs().mean().item():.4f}")
    print(f"wav shape={wav.shape}  max-abs={wav.abs().max().item():.4f}")

    write_tensor(f"{OUT}/f0.bin", f0.cpu().numpy())
    write_tensor(f"{OUT}/sine_merge.bin", sine.cpu().numpy())
    write_tensor(f"{OUT}/s_after_source.bin", s_after.cpu().numpy())
    write_tensor(f"{OUT}/s_stft.bin", s_stft.cpu().numpy())
    write_tensor(f"{OUT}/conv_pre_out.bin", conv_pre_out.cpu().numpy())
    write_tensor(f"{OUT}/conv_post_out.bin", conv_post_out.cpu().numpy())
    write_tensor(f"{OUT}/audio.bin", wav.cpu().numpy())

    from scipy.io import wavfile
    pcm = (wav.squeeze(0).cpu().numpy() * 32767.0).clip(-32768, 32767).astype(np.int16)
    wavfile.write(f"{OUT}/upstream_hift_from_mel.wav", 24000, pcm)
    print(f"wrote {OUT}/upstream_hift_from_mel.wav")

if __name__ == "__main__":
    main()
