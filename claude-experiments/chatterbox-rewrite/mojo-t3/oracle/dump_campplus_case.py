"""
Dump CAMPPlus inputs + outputs for parity testing the pure-Mojo port.

Captures:
  ref_wav_16k.bin    (T_samples,)         16kHz mono audio
  fbank_feat.bin     (T_frames, 80)        Kaldi fbank features
  fbank_mean.bin     (80,)                 per-utterance mean (for centering)
  xvector.bin        (1, 192)              CAMPPlus output (the cloned-voice embedding)
  weights/*.bin      flat dump of every parameter
  weights_manifest.txt
"""
from __future__ import annotations
import os, sys, struct
from pathlib import Path
import numpy as np
import torch

os.environ.setdefault("MIOPEN_DEBUG_CONV_WINOGRAD", "0")
os.environ.setdefault("PYTORCH_HIP_ALLOC_CONF", "expandable_segments:True")

from chatterbox.tts import ChatterboxTTS

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "tests" / "fixtures" / "campplus"
OUT.mkdir(parents=True, exist_ok=True)
WDIR = OUT / "weights"
WDIR.mkdir(exist_ok=True)

REF_WAV = "/home/jimmyhmiller/.config/paper-audiobooks/default-voice.wav"


def write_tensor(p, arr):
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


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(42)
    model = ChatterboxTTS.from_pretrained(device=device)
    spk = model.s3gen.speaker_encoder

    # Strip weight_norm if any (CAMPPlus mostly doesn't have it but be safe).
    import torch.nn.utils.parametrize as parametrize
    def strip(m):
        if hasattr(m, "parametrizations"):
            for n in list(m.parametrizations.keys()):
                parametrize.remove_parametrizations(m, n, leave_parametrized=True)
        for c in m.children():
            strip(c)
    strip(spk)

    # Load reference WAV at 16kHz mono.
    import wave
    with wave.open(REF_WAV, "rb") as wf:
        sr = wf.getframerate()
        n = wf.getnframes()
        nc = wf.getnchannels()
        raw = wf.readframes(n)
    pcm = np.frombuffer(raw, dtype=np.int16).reshape(-1, nc).mean(axis=1).astype(np.float32) / 32768.0
    if sr != 16000:
        import scipy.signal as sig
        pcm = sig.resample_poly(pcm, 16000, sr).astype(np.float32)
    # Truncate to 10s for fixture size.
    pcm = pcm[: 16000 * 10]
    wav = torch.from_numpy(pcm).unsqueeze(0).to(device)
    print(f"[campplus] wav: {tuple(wav.shape)} sr=16000")
    write_tensor(OUT / "ref_wav_16k.bin", wav.cpu().numpy().astype(np.float32))

    # Capture the Kaldi fbank features that extract_feature() produces.
    from chatterbox.models.s3gen.xvector import extract_feature
    feats, lens, times = extract_feature(wav)
    # feats is (B, T_frames, 80), but extract_feature centers per-utterance.
    print(f"[campplus] fbank: {tuple(feats.shape)}")
    write_tensor(OUT / "fbank_feat.bin", feats.cpu().numpy().astype(np.float32))

    # Run CAMPPlus.
    with torch.inference_mode():
        xvec = spk(feats.to(device).to(torch.float32))
    print(f"[campplus] xvector: {tuple(xvec.shape)} (first 8): {xvec.flatten()[:8].cpu().numpy()}")
    write_tensor(OUT / "xvector.bin", xvec.cpu().numpy().astype(np.float32))

    # Also try CAMPPlus.inference() to be the same path s3gen uses.
    with torch.inference_mode():
        xvec_inf = spk.inference([wav.squeeze(0)])
    print(f"[campplus] inference xvector: {tuple(xvec_inf.shape)} first 8: {xvec_inf.flatten()[:8].cpu().numpy()}")
    write_tensor(OUT / "xvector_inference.bin", xvec_inf.cpu().numpy().astype(np.float32))

    # Dump every state_dict tensor.
    sd = spk.state_dict()
    print(f"[campplus] {len(sd)} weights")
    for k, v in sd.items():
        fname = k.replace(".", "__") + ".bin"
        write_tensor(WDIR / fname, v.detach().cpu().numpy().astype(np.float32))
    with (OUT / "weights_manifest.txt").open("w") as f:
        for k in sorted(sd.keys()):
            f.write(f"{k}\t{tuple(sd[k].shape)}\n")


if __name__ == "__main__":
    main()
