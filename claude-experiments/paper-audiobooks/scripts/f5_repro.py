"""F5-TTS reliability test on the same 361-char chunk that fails on chatterbox.

Synthesizes via the same default-voice.wav reference. We expect F5 to:
1. Successfully clone the user's voice
2. Have a different (hopefully lower) anomaly rate than chatterbox

Uses the f5 venv directly to bypass our subprocess wrapper so we can score
each output's spectrum.
"""
from __future__ import annotations

import json
import subprocess
import tempfile
import time
from pathlib import Path

import numpy as np
import soundfile as sf
from scipy.signal import welch


VOICE_REF = "/home/jimmyhmiller/.config/paper-audiobooks/default-voice.wav"
F5_PYTHON = Path("/home/jimmyhmiller/.cache/paper-audiobooks/venvs/f5/bin/python")
CHUNK = (
    "The objection goes as follows: according to Christian belief, we human "
    "beings have been created by an all-powerful, all-knowing God who loves us "
    "enough to send his son, the second person of the divine Trinity, to "
    "suffer and die on our account; but given the devastating amount and "
    "variety of human suffering and evil in our sad world, this simply can't "
    "be true."
)
TRIALS = 6


CHILD_SCRIPT = r"""
import json, sys, time
from pathlib import Path
import numpy as np
import soundfile as sf
import torch

from f5_tts.api import F5TTS

print(f"loading F5 on {'cuda' if torch.cuda.is_available() else 'cpu'}", file=sys.stderr, flush=True)
t0 = time.time()
tts = F5TTS(device="cuda" if torch.cuda.is_available() else "cpu")
print(f"loaded in {time.time()-t0:.1f}s", file=sys.stderr, flush=True)

# Auto-transcribe the reference audio with whisper (F5 does this internally if
# ref_text is empty). Pass empty ref_text to trigger that path.
req = json.loads(sys.stdin.read())
audio, sr, _ = tts.infer(
    ref_file=req["ref_file"],
    ref_text="",
    gen_text=req["text"],
)
audio = np.asarray(audio, dtype="float32")
sf.write(req["out_path"], audio, sr)
print(json.dumps({"sample_rate": int(sr), "samples": int(audio.shape[0])}))
"""


def stats(audio: np.ndarray, sr: int) -> dict:
    f, p = welch(audio, sr, nperseg=2048)
    tot = float(np.sum(p)) + 1e-12
    return {
        "centroid": float(np.sum(f * p) / tot),
        "below_300": float(np.sum(p[f < 300]) / tot),
        "mid": float(np.sum(p[(f >= 300) & (f < 2000)]) / tot),
        "rms": float(np.sqrt(np.mean(audio**2))),
        "dur": len(audio) / sr,
    }


def is_anomalous(s: dict) -> bool:
    return (s["centroid"] < 700 and s["below_300"] > 0.5) or s["rms"] < 0.04


def main() -> None:
    out = Path("f5_repro")
    out.mkdir(exist_ok=True)

    print(f"\nTesting F5 on the chunk19 361-char text, {TRIALS} trials\n", flush=True)
    anomalies = 0
    for trial in range(1, TRIALS + 1):
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            tmp_path = Path(tmp.name)
        req = {"ref_file": VOICE_REF, "text": CHUNK, "out_path": str(tmp_path)}
        t0 = time.time()
        r = subprocess.run(
            [str(F5_PYTHON), "-c", CHILD_SCRIPT],
            input=json.dumps(req), text=True, capture_output=True,
        )
        if r.returncode != 0:
            print(f"  trial {trial}: FAILED rc={r.returncode}")
            print(f"    stderr (last 500): {r.stderr[-500:]}")
            continue
        elapsed = time.time() - t0
        audio, sr = sf.read(tmp_path)
        s = stats(audio.astype(np.float32), sr)
        anom = is_anomalous(s)
        if anom:
            anomalies += 1
        path = out / f"trial{trial}.wav"
        sf.write(path, audio, sr)
        flag = " ANOMALY" if anom else ""
        print(f"  trial {trial}: gen={elapsed:5.1f}s dur={s['dur']:5.1f}s rms={s['rms']:.4f} "
              f"cen={s['centroid']:5.0f}Hz <300={s['below_300']:.3f} "
              f"mid={s['mid']:.3f}{flag}", flush=True)
        tmp_path.unlink(missing_ok=True)

    print(f"\n=== F5 RESULT: {anomalies}/{TRIALS} anomalies ===")
    print("(chatterbox baseline: 4/24 = ~17% across temperatures)")


if __name__ == "__main__":
    main()
