"""Smoke test for ChatterboxTTS wrapper. Synthesizes two utterances reusing
cached conditionals to confirm the prepare/generate split works."""
import sys
from pathlib import Path

import numpy as np

from chatterbox_mojo import ChatterboxTTS

import mojo.importer  # noqa: F401
import op_write_wav
from max.driver import Buffer


REF_WAV = "/home/jimmyhmiller/.config/paper-audiobooks/default-voice.wav"


def main():
    print("[test] loading model...")
    tts = ChatterboxTTS.from_pretrained(device="gpu")
    print("[test] preparing conditionals from ref voice...")
    tts.prepare_conditionals(REF_WAV, exaggeration=0.5)
    assert tts.conds is not None

    texts = [
        ("/tmp/wrap_one.wav", "hello world from the wrapped chatterbox."),
        ("/tmp/wrap_two.wav", "and this is the second utterance, reusing the cached voice."),
    ]
    for out_path, text in texts:
        print(f"[test] generating {out_path!r}: {text!r}")
        wav = tts.generate(text, cfg_weight=0.5, temperature=0.8, top_p=0.95,
                           repetition_penalty=1.2, exaggeration=0.5)
        arr = wav.squeeze(0).cpu().numpy().astype(np.float32)
        rms = float(np.sqrt((arr ** 2).mean()))
        peak = float(np.abs(arr).max())
        buf = Buffer.from_numpy(arr.reshape(1, -1))
        op_write_wav.write_wav(buf, arr.size, tts.sr, out_path)
        print(f"[test]   wrote {out_path} ({len(arr)/tts.sr:.2f}s, rms={rms:.3f}, peak={peak:.3f})")

    print("[PASS] wrapper smoke test")
    return 0


if __name__ == "__main__":
    sys.exit(main())
