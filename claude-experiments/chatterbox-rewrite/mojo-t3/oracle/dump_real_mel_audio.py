"""
Produce a REAL mel + REAL audio from upstream Chatterbox on a real text prompt
and a real reference voice. Saves:
  real_mel.bin          (1, 80, T_mel)
  real_audio.bin        (1, T_audio)         upstream's full audio output
  real_s_stft_cat.bin   (1, 18, T_mel*240)   STFT of the real source signal

Run this with the paper-audiobooks chatterbox venv:
  /home/jimmyhmiller/.cache/paper-audiobooks/venvs/chatterbox/bin/python \
      oracle/dump_real_mel_audio.py

Then the Mojo full-HiFiGAN test can run on this real mel and we have a true
side-by-side audio comparison.
"""
from __future__ import annotations

import struct
import sys
from pathlib import Path

import numpy as np
import torch

# Disable MIOpen Winograd to match paper-audiobooks env (avoids the bug while
# we're producing the reference upstream output).
import os
os.environ.setdefault("MIOPEN_DEBUG_CONV_WINOGRAD", "0")
os.environ.setdefault("PYTORCH_HIP_ALLOC_CONF", "expandable_segments:True")

from chatterbox.tts import ChatterboxTTS

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "tests" / "fixtures" / "real"
OUT.mkdir(parents=True, exist_ok=True)

REF_VOICE = Path("/home/jimmyhmiller/.config/paper-audiobooks/default-voice.wav")
PROMPT = "Hello, this is a Mojo-based Chatterbox vocoder running on AMD."


def write_tensor(path: Path, arr: np.ndarray) -> None:
    if arr.dtype == np.float32:
        tag, raw = 0, arr.astype(np.float32, copy=False).tobytes()
    elif arr.dtype == np.uint16:
        tag, raw = 1, arr.astype(np.uint16, copy=False).tobytes()
    elif arr.dtype == np.int64:
        tag, raw = 2, arr.astype(np.int64, copy=False).tobytes()
    else:
        raise TypeError(f"unsupported dtype {arr.dtype}")
    with path.open("wb") as f:
        f.write(struct.pack("<q", len(arr.shape)))
        f.write(struct.pack(f"<{len(arr.shape)}q", *arr.shape))
        f.write(struct.pack("<i", tag))
        f.write(raw)


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[real] device={device}")
    model = ChatterboxTTS.from_pretrained(device=device)
    model.s3gen.eval()
    s3gen = model.s3gen

    # Strip weight_norm parametrizations so we can call hift.inference directly
    # without the same issue we hit in dump_hifigan_case.py.
    import torch.nn.utils.parametrize as parametrize
    def _strip(mod):
        if hasattr(mod, "parametrizations"):
            for name in list(mod.parametrizations.keys()):
                parametrize.remove_parametrizations(mod, name, leave_parametrized=True)
        for child in mod.children():
            _strip(child)
    _strip(s3gen.mel2wav)

    # ---- Run T3 → S3Gen flow to produce a mel + final audio ----
    with torch.inference_mode():
        # Use the synthesis path that paper-audiobooks uses.
        wav = model.generate(text=PROMPT, audio_prompt_path=str(REF_VOICE))
        # `wav` is a torch tensor (1, T_audio) at S3GEN_SR (24000).
        # We also want the mel that was produced internally. The simplest
        # way is to monkey-patch s3gen.flow_inference to capture it.
    print(f"[real] full inference wav shape: {tuple(wav.shape)}")

    # Re-run with capture hooks for mel.
    captured = {}
    orig_flow_inference = s3gen.flow_inference

    @torch.inference_mode()
    def capture_flow_inference(*args, **kwargs):
        result = orig_flow_inference(*args, **kwargs)
        # flow_inference returns the mel; the exact shape depends on the API.
        captured["mel"] = result.detach() if isinstance(result, torch.Tensor) else result
        return result

    s3gen.flow_inference = capture_flow_inference

    orig_hift_inference = s3gen.mel2wav.inference

    @torch.inference_mode()
    def capture_hift(speech_feat, **kwargs):
        captured["mel"] = speech_feat.detach()
        # also capture the source signal s
        f0 = s3gen.mel2wav.f0_predictor(speech_feat)
        s = s3gen.mel2wav.f0_upsamp(f0[:, None]).transpose(1, 2)
        s, _, _ = s3gen.mel2wav.m_source(s)
        s = s.transpose(1, 2)
        s_r, s_i = s3gen.mel2wav._stft(s.squeeze(1))
        captured["s_stft_cat"] = torch.cat([s_r, s_i], dim=1).detach()
        return orig_hift_inference(speech_feat, **kwargs)

    s3gen.mel2wav.inference = capture_hift

    with torch.inference_mode():
        wav2 = model.generate(text=PROMPT, audio_prompt_path=str(REF_VOICE))
    print(f"[real] re-run wav: {tuple(wav2.shape)}")
    print(f"[real] captured mel: {tuple(captured['mel'].shape)}")
    print(f"[real] captured s_stft_cat: {tuple(captured['s_stft_cat'].shape)}")

    # Save.
    write_tensor(OUT / "real_mel.bin",
                 captured["mel"].cpu().numpy().astype(np.float32))
    write_tensor(OUT / "real_s_stft_cat.bin",
                 captured["s_stft_cat"].cpu().numpy().astype(np.float32))
    write_tensor(OUT / "real_audio_upstream.bin",
                 wav2.cpu().numpy().astype(np.float32))

    print("[real] wrote real_mel.bin, real_s_stft_cat.bin, real_audio_upstream.bin")


if __name__ == "__main__":
    main()
