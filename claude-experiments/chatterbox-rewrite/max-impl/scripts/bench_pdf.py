"""Throughput benchmark: Mojo wrapper vs upstream chatterbox.

Synthesizes a curated subset of the Gettier paper through both engines and
reports per-chunk wall-clock latency + total real-time factor. Both engines
get the same reference voice and identical generation kwargs.

We hand-curated the sentences (avoiding PDF artifacts like "fo1lowing",
"sttficietzt"); the goal is throughput parity, not transcript verification —
that was done by test_roundtrip.py.
"""
import json
import os
import subprocess
import sys
import time
import wave
from pathlib import Path

import numpy as np

# Add max-impl root so `chatterbox_mojo` package is importable.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

REF_WAV = "/home/jimmyhmiller/.config/paper-audiobooks/default-voice.wav"
OUT = Path("/tmp/gettier_bench")
OUT.mkdir(exist_ok=True)
UPSTREAM_PY = os.path.expanduser("~/.cache/paper-audiobooks/venvs/chatterbox/bin/python")

# Curated sentences from Gettier's "Is Justified True Belief Knowledge?" (1963).
# Hand-cleaned to remove OCR artifacts.
CHUNKS = [
    "Various attempts have been made in recent years to state necessary and sufficient conditions for someone's knowing a given proposition.",
    "I shall argue that the standard analysis of knowledge as justified true belief is false.",
    "It is possible for a person to be justified in believing a proposition that is in fact false.",
    "Suppose that Smith and Jones have applied for a certain job.",
    "And suppose that Smith has strong evidence for the conjunctive proposition that Jones is the man who will get the job, and Jones has ten coins in his pocket.",
    "But imagine, further, that unknown to Smith, he himself, not Jones, will get the job.",
    "And, also, unknown to Smith, he himself has ten coins in his pocket.",
    "All of the following are true: the proposition is true, Smith believes that it is true, and Smith is justified in believing that it is true.",
    "But it is equally clear that Smith does not know that the proposition is true.",
    "These two examples show that there is justified true belief that is not knowledge.",
]


def wav_duration(path: str) -> float:
    with wave.open(path, "rb") as w:
        return w.getnframes() / w.getframerate()


# ── Mojo timing ──────────────────────────────────────────────────────────

def bench_mojo() -> list[dict]:
    print("[bench] loading Mojo ChatterboxTTS (this includes model+weight load)...")
    from chatterbox_mojo import ChatterboxTTS
    import mojo.importer  # noqa: F401
    import op_write_wav
    from max.driver import Buffer

    load_t0 = time.perf_counter()
    tts = ChatterboxTTS.from_pretrained(device="gpu")
    tts.prepare_conditionals(REF_WAV, exaggeration=0.5)
    load_dt = time.perf_counter() - load_t0
    print(f"[bench] mojo load+prepare: {load_dt:.2f}s")

    results = []
    for i, text in enumerate(CHUNKS):
        t0 = time.perf_counter()
        wav = tts.generate(
            text,
            cfg_weight=0.5, temperature=0.8, top_p=0.95, min_p=0.05,
            repetition_penalty=1.2, exaggeration=0.5,
        )
        wall = time.perf_counter() - t0
        arr = wav.squeeze(0).cpu().numpy().astype(np.float32)
        out_path = str(OUT / f"mojo_{i:02d}.wav")
        buf = Buffer.from_numpy(arr.reshape(1, -1))
        op_write_wav.write_wav(buf, arr.size, tts.sr, out_path)
        dur = wav_duration(out_path)
        rtf = wall / dur if dur > 0 else float("inf")
        print(f"  [mojo {i}] wall={wall:6.2f}s  audio={dur:5.2f}s  rtf={rtf:5.3f}x  {text[:60]!r}")
        results.append({"i": i, "wall": wall, "audio": dur, "rtf": rtf})
    return results


# ── Upstream timing (subprocess) ─────────────────────────────────────────

UPSTREAM_SCRIPT = '''
import json, sys, time, wave
import numpy as np, torch
from chatterbox.tts import ChatterboxTTS

ref, out_dir, chunks_json = sys.argv[1], sys.argv[2], sys.argv[3]
chunks = json.loads(chunks_json)
device = "cuda" if torch.cuda.is_available() else "cpu"
load_t0 = time.perf_counter()
tts = ChatterboxTTS.from_pretrained(device=device)
tts.prepare_conditionals(ref, exaggeration=0.5)
load_dt = time.perf_counter() - load_t0
print(json.dumps({"event": "load", "wall": load_dt}), flush=True)

for i, text in enumerate(chunks):
    t0 = time.perf_counter()
    wav = tts.generate(text, cfg_weight=0.5, temperature=0.8, top_p=0.95,
                       min_p=0.05, repetition_penalty=1.2, exaggeration=0.5)
    wall = time.perf_counter() - t0
    arr = wav.squeeze(0).cpu().numpy().astype(np.float32)
    arr = np.clip(arr, -1.0, 1.0)
    pcm = (arr * 32767.0).astype(np.int16)
    path = f"{out_dir}/upstream_{i:02d}.wav"
    with wave.open(path, "wb") as w:
        w.setnchannels(1); w.setsampwidth(2); w.setframerate(tts.sr)
        w.writeframes(pcm.tobytes())
    dur = pcm.size / tts.sr
    print(json.dumps({"event": "chunk", "i": i, "wall": wall, "audio": dur}), flush=True)
'''


def bench_upstream() -> tuple[float, list[dict]]:
    print("[bench] launching upstream...")
    clean_env = {k: v for k, v in os.environ.items()
                 if not k.startswith(("PIXI_", "CONDA_", "PYTHON"))
                 and k not in ("VIRTUAL_ENV", "PYTHONHOME", "PYTHONPATH")}
    clean_env["PATH"] = "/usr/bin:/usr/local/bin"
    clean_env["HOME"] = os.environ.get("HOME", "/home/jimmyhmiller")
    proc = subprocess.Popen(
        [UPSTREAM_PY, "-c", UPSTREAM_SCRIPT, REF_WAV, str(OUT), json.dumps(CHUNKS)],
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, env=clean_env, text=True, bufsize=1,
    )
    load_wall = None
    results = []
    for line in proc.stdout:
        try:
            ev = json.loads(line.strip())
        except Exception:
            continue
        if ev["event"] == "load":
            load_wall = ev["wall"]
            print(f"[bench] upstream load+prepare: {load_wall:.2f}s")
        elif ev["event"] == "chunk":
            i, wall, dur = ev["i"], ev["wall"], ev["audio"]
            rtf = wall / dur if dur > 0 else float("inf")
            print(f"  [up   {i}] wall={wall:6.2f}s  audio={dur:5.2f}s  rtf={rtf:5.3f}x  {CHUNKS[i][:60]!r}")
            results.append({"i": i, "wall": wall, "audio": dur, "rtf": rtf})
    rc = proc.wait()
    if rc != 0:
        print("--- upstream stderr ---", file=sys.stderr)
        print(proc.stderr.read()[-2000:], file=sys.stderr)
        raise RuntimeError(f"upstream failed rc={rc}")
    return load_wall, results


# ── Main ─────────────────────────────────────────────────────────────────

def main():
    print(f"[bench] {len(CHUNKS)} chunks, total chars={sum(len(c) for c in CHUNKS)}, words={sum(len(c.split()) for c in CHUNKS)}")
    print()
    mojo_results = bench_mojo()
    print()
    upstream_load, upstream_results = bench_upstream()

    # Summary.
    def total(rs, key):
        return sum(r[key] for r in rs)

    print()
    print("=" * 80)
    print(f"{'engine':<10} {'wall':>10} {'audio':>10} {'RTF':>8} {'speedup':>10}")
    print("-" * 80)
    mw, ma = total(mojo_results, "wall"), total(mojo_results, "audio")
    uw, ua = total(upstream_results, "wall"), total(upstream_results, "audio")
    print(f"{'mojo':<10} {mw:>9.2f}s {ma:>9.2f}s {mw/ma:>7.3f}x {'(baseline)':>10}")
    print(f"{'upstream':<10} {uw:>9.2f}s {ua:>9.2f}s {uw/ua:>7.3f}x {uw/mw:>9.2f}x")
    print("=" * 80)
    print()
    print(f"mojo:     {ma:.1f}s of audio in {mw:.1f}s wall = {ma/mw:.2f}x real-time")
    print(f"upstream: {ua:.1f}s of audio in {uw:.1f}s wall = {ua/uw:.2f}x real-time")
    if uw > mw:
        print(f"\nMojo is {uw/mw:.2f}x faster than upstream.")
    else:
        print(f"\nUpstream is {mw/uw:.2f}x faster than Mojo.")


if __name__ == "__main__":
    sys.exit(main() or 0)
