"""Re-synthesize the exact 361-char chunk that was distorted in the audiobook."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import soundfile as sf
from scipy.signal import welch

from paper_audiobooks.tts import get_backend, SAMPLE_RATE


VOICE_REF = "/home/jimmyhmiller/.config/paper-audiobooks/default-voice.wav"

# The exact 361-char chunk that landed at t=830-849 in the audiobook.
CHUNK19 = (
    "The objection goes as follows: according to Christian belief, we human "
    "beings have been created by an all-powerful, all-knowing God who loves us "
    "enough to send his son, the second person of the divine Trinity, to "
    "suffer and die on our account; but given the devastating amount and "
    "variety of human suffering and evil in our sad world, this simply can't "
    "be true."
)


def stats(audio: np.ndarray, sr: int = SAMPLE_RATE) -> dict:
    f, p = welch(audio, sr, nperseg=2048)
    tot = float(np.sum(p)) + 1e-12
    return {
        "centroid_hz": float(np.sum(f * p) / tot),
        "below_300hz": float(np.sum(p[f < 300]) / tot),
        "300_to_2k": float(np.sum(p[(f >= 300) & (f < 2000)]) / tot),
        "rms": float(np.sqrt(np.mean(audio**2))),
        "duration_s": len(audio) / sr,
    }


def main() -> None:
    out_dir = Path("repro_clips")
    out_dir.mkdir(exist_ok=True)
    backend = get_backend("chatterbox")
    print(f"Synthesizing chunk19 ({len(CHUNK19)} chars), 5 trials")
    for trial in range(5):
        audio = backend.synthesize_chunk(CHUNK19, voice=VOICE_REF)
        path = out_dir / f"chunk19_trial{trial + 1}.wav"
        sf.write(path, audio, SAMPLE_RATE)
        s = stats(audio)
        print(f"  trial {trial + 1}: dur={s['duration_s']:.1f}s rms={s['rms']:.4f} "
              f"centroid={s['centroid_hz']:.0f}Hz <300Hz={s['below_300hz']:.3f} 300-2k={s['300_to_2k']:.3f}")


if __name__ == "__main__":
    main()
