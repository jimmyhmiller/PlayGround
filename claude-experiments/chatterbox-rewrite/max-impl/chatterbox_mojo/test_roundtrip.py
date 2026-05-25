"""Round-trip TTS→STT comparison: Mojo wrapper vs upstream chatterbox.

For each test phrase:
  1. Synthesize via Mojo ChatterboxTTS                  → mojo.wav
  2. Synthesize via upstream chatterbox (subprocess)    → upstream.wav
  3. Transcribe both via HF Whisper                     → mojo_text, upstream_text
  4. Compute WER (word error rate) vs the input phrase

Prints a side-by-side table at the end. STT runs only AFTER all synthesis is
done so we don't shuttle whisper + our model in/out of GPU memory repeatedly.
"""
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import torch

from chatterbox_mojo import ChatterboxTTS
import mojo.importer  # noqa: F401
import op_write_wav
from max.driver import Buffer


REF_WAV = "/home/jimmyhmiller/.config/paper-audiobooks/default-voice.wav"
OUT_DIR = Path("/tmp/rt")
OUT_DIR.mkdir(exist_ok=True)

UPSTREAM_PY = os.path.expanduser("~/.cache/paper-audiobooks/venvs/chatterbox/bin/python")

# Test phrases. Mix of short, medium, long; everyday vocabulary so whisper
# isn't the bottleneck.
PHRASES = [
    "Hello world.",
    "The quick brown fox jumps over the lazy dog.",
    "She sells seashells by the seashore.",
    "How much wood would a woodchuck chuck if a woodchuck could chuck wood?",
    "I would like a cup of coffee with milk and sugar, please.",
    "The weather today is sunny with a light breeze from the south.",
    "Testing one, two, three. This is a simple sentence.",
    "Artificial intelligence is changing how we write software.",
]


# ── Mojo synthesis ────────────────────────────────────────────────────────

def synth_mojo(tts: ChatterboxTTS, text: str, out_path: Path) -> None:
    # Match upstream defaults exactly (cfg=0.5, T=0.8, top_p=0.95, min_p=0.05, rep=1.2)
    wav = tts.generate(
        text,
        cfg_weight=0.5, temperature=0.8, top_p=0.95, min_p=0.05,
        repetition_penalty=1.2, exaggeration=0.5,
    )
    arr = wav.squeeze(0).cpu().numpy().astype(np.float32)
    buf = Buffer.from_numpy(arr.reshape(1, -1))
    op_write_wav.write_wav(buf, arr.size, tts.sr, str(out_path))


# ── Upstream synthesis (subprocess into upstream venv) ────────────────────

UPSTREAM_SCRIPT = '''
import sys, json, torch, numpy as np
from chatterbox.tts import ChatterboxTTS

ref, out_dir, phrases_json = sys.argv[1], sys.argv[2], sys.argv[3]
phrases = json.loads(phrases_json)

device = "cuda" if torch.cuda.is_available() else "cpu"
tts = ChatterboxTTS.from_pretrained(device=device)
tts.prepare_conditionals(ref, exaggeration=0.5)

import wave, struct
for i, text in enumerate(phrases):
    wav = tts.generate(text, cfg_weight=0.5, temperature=0.8, top_p=0.95,
                       repetition_penalty=1.2, exaggeration=0.5, min_p=0.05)
    arr = wav.squeeze(0).cpu().numpy().astype(np.float32)
    arr = np.clip(arr, -1.0, 1.0)
    pcm16 = (arr * 32767.0).astype(np.int16)
    path = f"{out_dir}/upstream_{i:02d}.wav"
    with wave.open(path, "wb") as w:
        w.setnchannels(1); w.setsampwidth(2); w.setframerate(tts.sr)
        w.writeframes(pcm16.tobytes())
    print(f"[upstream] {i}: {path}", flush=True)
print("[upstream] DONE", flush=True)
'''


def synth_upstream_all(phrases: list[str]) -> None:
    """Run upstream synthesis for all phrases in ONE subprocess (model loads once)."""
    # Strip pixi/conda env vars so the upstream venv's python finds its own libs.
    clean_env = {k: v for k, v in os.environ.items()
                 if not k.startswith(("PIXI_", "CONDA_", "PYTHON"))
                 and k not in ("VIRTUAL_ENV", "PYTHONHOME", "PYTHONPATH")}
    clean_env["PATH"] = "/usr/bin:/usr/local/bin"
    clean_env["HOME"] = os.environ.get("HOME", "/home/jimmyhmiller")
    proc = subprocess.run(
        [UPSTREAM_PY, "-c", UPSTREAM_SCRIPT, REF_WAV, str(OUT_DIR), json.dumps(phrases)],
        capture_output=True, text=True, timeout=1800, env=clean_env,
    )
    print(proc.stdout)
    if proc.returncode != 0:
        print("--- upstream stderr ---", file=sys.stderr)
        print(proc.stderr, file=sys.stderr)
        raise RuntimeError(f"upstream synth failed (rc={proc.returncode})")


# ── STT ───────────────────────────────────────────────────────────────────

WHISPER_SCRIPT = '''
import sys, json
from transformers import pipeline
import torch
paths = json.loads(sys.argv[1])
asr = pipeline("automatic-speech-recognition", model="openai/whisper-small.en",
               device=0 if torch.cuda.is_available() else -1, chunk_length_s=30)
out = {p: asr(p)["text"].strip() for p in paths}
print("===RESULTS===")
print(json.dumps(out))
'''


def transcribe_all(paths: list[str]) -> dict[str, str]:
    """Run whisper in upstream venv on all paths in one subprocess."""
    clean_env = {k: v for k, v in os.environ.items()
                 if not k.startswith(("PIXI_", "CONDA_", "PYTHON"))
                 and k not in ("VIRTUAL_ENV", "PYTHONHOME", "PYTHONPATH")}
    clean_env["PATH"] = "/usr/bin:/usr/local/bin"
    clean_env["HOME"] = os.environ.get("HOME", "/home/jimmyhmiller")
    proc = subprocess.run(
        [UPSTREAM_PY, "-c", WHISPER_SCRIPT, json.dumps(paths)],
        capture_output=True, text=True, timeout=1800, env=clean_env,
    )
    if proc.returncode != 0:
        print("--- whisper stderr ---", file=sys.stderr)
        print(proc.stderr[-2000:], file=sys.stderr)
        raise RuntimeError(f"whisper failed (rc={proc.returncode})")
    # Find the json line after ===RESULTS===
    marker = proc.stdout.rfind("===RESULTS===")
    return json.loads(proc.stdout[marker:].split("\n", 1)[1].strip())


# ── Metrics ───────────────────────────────────────────────────────────────

import re, string

def normalize(s: str) -> list[str]:
    s = s.lower()
    s = re.sub(rf"[{re.escape(string.punctuation)}]", "", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s.split()


def edit_distance(a: list, b: list) -> int:
    n, m = len(a), len(b)
    dp = list(range(m + 1))
    for i in range(1, n + 1):
        prev, dp[0] = dp[0], i
        for j in range(1, m + 1):
            cur = dp[j]
            if a[i-1] == b[j-1]:
                dp[j] = prev
            else:
                dp[j] = 1 + min(prev, dp[j], dp[j-1])
            prev = cur
    return dp[m]


def wer(ref: str, hyp: str) -> float:
    r, h = normalize(ref), normalize(hyp)
    if not r:
        return 0.0 if not h else 1.0
    return edit_distance(r, h) / len(r)


# ── Main ──────────────────────────────────────────────────────────────────

def main():
    # 1. Mojo synthesis (skip if all wavs already exist).
    mojo_paths = [OUT_DIR / f"mojo_{i:02d}.wav" for i in range(len(PHRASES))]
    if all(p.exists() for p in mojo_paths) and "--reuse" in sys.argv:
        print("[rt] reusing existing mojo wavs")
    else:
        print("[rt] loading Mojo ChatterboxTTS...")
        tts = ChatterboxTTS.from_pretrained(device="gpu")
        tts.prepare_conditionals(REF_WAV, exaggeration=0.5)
        for i, text in enumerate(PHRASES):
            print(f"[mojo] {i}: {text!r}")
            synth_mojo(tts, text, mojo_paths[i])
        del tts
        import gc; gc.collect()

    # 2. Upstream synthesis (separate venv → separate process → separate GPU lifetime).
    print("[rt] synthesizing upstream...")
    synth_upstream_all(PHRASES)
    upstream_paths = [OUT_DIR / f"upstream_{i:02d}.wav" for i in range(len(PHRASES))]

    # 3. STT both sets in one subprocess.
    print("[rt] running whisper on all wavs...")
    all_paths = [str(p) for p in mojo_paths] + [str(p) for p in upstream_paths]
    results = transcribe_all(all_paths)
    mojo_txt = [results[str(p)] for p in mojo_paths]
    upstream_txt = [results[str(p)] for p in upstream_paths]

    # 4. WER + report
    print()
    print("=" * 100)
    print(f"{'#':<3} {'ref':<55} {'mojo WER':>10} {'upstream WER':>14}")
    print("-" * 100)
    mojo_wers, up_wers = [], []
    rows = []
    for i, (text, mt, ut) in enumerate(zip(PHRASES, mojo_txt, upstream_txt)):
        m_wer, u_wer = wer(text, mt), wer(text, ut)
        mojo_wers.append(m_wer); up_wers.append(u_wer)
        print(f"{i:<3} {text[:54]:<55} {m_wer*100:>9.1f}% {u_wer*100:>13.1f}%")
        rows.append((text, mt, ut, m_wer, u_wer))
    print("-" * 100)
    print(f"{'AVG':<3} {'':<55} {sum(mojo_wers)/len(mojo_wers)*100:>9.1f}% {sum(up_wers)/len(up_wers)*100:>13.1f}%")
    print("=" * 100)
    print()

    # Detailed transcripts so we can eyeball what went wrong.
    for i, (text, mt, ut, mw, uw) in enumerate(rows):
        if mw > 0.01 or uw > 0.01:
            print(f"\n[{i}] ref:      {text}")
            print(f"    mojo:     {mt}   (WER {mw*100:.1f}%)")
            print(f"    upstream: {ut}   (WER {uw*100:.1f}%)")

    return 0


if __name__ == "__main__":
    sys.exit(main())
