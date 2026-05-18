"""
Full paper-audiobooks-style integration test.

Loads real Chatterbox, monkey-patches s3gen.mel2wav.inference with the Mojo
implementation, runs model.generate() on a real prompt + cloned voice, and
verifies we get back real audio.

Run with the paper-audiobooks chatterbox venv:
  /home/jimmyhmiller/.cache/paper-audiobooks/venvs/chatterbox/bin/python \
      oracle/test_full_integration.py
"""
from __future__ import annotations
import os
import sys
import wave
from pathlib import Path

# paper-audiobooks env hardening.
os.environ.setdefault("MIOPEN_DEBUG_CONV_WINOGRAD", "0")
os.environ.setdefault("PYTORCH_HIP_ALLOC_CONF", "expandable_segments:True")

import numpy as np
import torch

# Add this directory so we can import the mojo_hifigan_py wrapper.
sys.path.insert(0, ".")
from mojo_hifigan_py import install_mojo_hifigan

from chatterbox.tts import ChatterboxTTS


REF_VOICE = "/home/jimmyhmiller/.config/paper-audiobooks/default-voice.wav"
PROMPT = "Hello, this is a Mojo-based Chatterbox vocoder running on AMD."


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[full] device={device}")
    model = ChatterboxTTS.from_pretrained(device=device)
    s3gen = model.s3gen

    # Strip weight_norm so we can call .inference directly without errors.
    import torch.nn.utils.parametrize as parametrize
    def _strip(mod):
        if hasattr(mod, "parametrizations"):
            for name in list(mod.parametrizations.keys()):
                parametrize.remove_parametrizations(mod, name, leave_parametrized=True)
        for child in mod.children():
            _strip(child)
    _strip(s3gen)

    # Install the Mojo HiFiGAN patch.
    install_mojo_hifigan(s3gen, work_dir=".")

    # Run real Chatterbox generation.
    with torch.inference_mode():
        torch.manual_seed(0)
        wav = model.generate(text=PROMPT, audio_prompt_path=REF_VOICE)
    print(f"[full] generated wav: {tuple(wav.shape)}")

    # Save the result for listening.
    OUT = Path("tests/fixtures/real_wavs")
    OUT.mkdir(parents=True, exist_ok=True)
    audio = wav.detach().cpu().numpy().flatten()
    audio = np.clip(audio, -0.99, 0.99)
    pcm = (audio * 32767.0).astype(np.int16)
    out_path = OUT / "mojo_full_pipeline.wav"
    with wave.open(str(out_path), "wb") as w:
        w.setnchannels(1); w.setsampwidth(2); w.setframerate(24000)
        w.writeframes(pcm.tobytes())
    print(f"[full] wrote {out_path}")
    print(f"[full] duration: {len(audio)/24000:.2f}s  peak: {np.abs(audio).max():.3f}")

    if np.abs(audio).max() < 0.01:
        print("✗ Audio is silent — Mojo HiFiGAN likely failed or returned zeros")
        return 1
    print("✓ Real Chatterbox + Mojo HiFiGAN integration: PRODUCED REAL AUDIO")
    return 0


if __name__ == "__main__":
    sys.exit(main())
