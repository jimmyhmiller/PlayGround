"""Run N sequential generate() calls, log GTT/VRAM usage after each.

A monotonic upward trend means we leak GPU buffers per call — would OOM
during a long audiobook run. Stable usage means we're safe.
"""
import os, sys, subprocess, time, re
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
from chatterbox_mojo import ChatterboxTTS

REF = "/home/jimmyhmiller/.config/paper-audiobooks/default-voice.wav"

PHRASES = [
    "Hello world.",
    "The quick brown fox jumps over the lazy dog.",
    "She sells seashells by the seashore.",
    "How much wood would a woodchuck chuck if a woodchuck could chuck wood.",
    "I would like a cup of coffee with milk and sugar, please.",
    "The weather today is sunny with a light breeze from the south.",
    "Testing one, two, three. This is a simple sentence.",
    "Artificial intelligence is changing how we write software.",
]


def gpu_mem_mb():
    """Return (vram_used_MB, gtt_used_MB)."""
    out = subprocess.run(["rocm-smi", "--showmeminfo", "all"],
                         capture_output=True, text=True).stdout
    vram = int(re.search(r"VRAM Total Used Memory \(B\): (\d+)", out).group(1))
    gtt = int(re.search(r"GTT Total Used Memory \(B\): (\d+)", out).group(1))
    return vram / 1024**2, gtt / 1024**2


def main():
    n_iters = int(os.environ.get("N", "60"))
    print(f"[mem-test] running {n_iters} sequential generate() calls")

    v0, g0 = gpu_mem_mb()
    print(f"[mem-test] before model load:    vram={v0:7.1f}MB  gtt={g0:7.1f}MB")

    tts = ChatterboxTTS.from_pretrained(use_bf16=True)
    tts.prepare_conditionals(REF, exaggeration=0.5)

    v1, g1 = gpu_mem_mb()
    print(f"[mem-test] after model + condit: vram={v1:7.1f}MB  gtt={g1:7.1f}MB  (Δvram={v1-v0:+.1f}, Δgtt={g1-g0:+.1f})")

    print()
    print(f"{'iter':>4} {'vram_MB':>10} {'gtt_MB':>10} {'Δvram':>8} {'Δgtt':>8} {'wall_s':>8}  text")
    print("-" * 100)

    samples = []
    for i in range(n_iters):
        t0 = time.perf_counter()
        text = PHRASES[i % len(PHRASES)]
        _ = tts.generate(text, cfg_weight=0.5, temperature=0.8, top_p=0.95,
                          min_p=0.05, repetition_penalty=1.2, exaggeration=0.5)
        wall = time.perf_counter() - t0
        v, g = gpu_mem_mb()
        dv = v - v1
        dg = g - g1
        samples.append((i, v, g, wall))
        print(f"{i:>4} {v:>10.1f} {g:>10.1f} {dv:>+8.1f} {dg:>+8.1f} {wall:>8.2f}  {text[:55]!r}")

    print()
    # Trend: did GTT/VRAM keep growing past iter ~5?
    early_g = np.mean([s[2] for s in samples[2:7]])      # iters 2-6 (after warm-up)
    late_g = np.mean([s[2] for s in samples[-5:]])       # last 5
    early_v = np.mean([s[1] for s in samples[2:7]])
    late_v = np.mean([s[1] for s in samples[-5:]])
    print(f"[mem-test] GTT early-iter avg = {early_g:.1f}MB,  late-iter avg = {late_g:.1f}MB  (Δ={late_g-early_g:+.1f})")
    print(f"[mem-test] VRAM early-iter avg = {early_v:.1f}MB,  late-iter avg = {late_v:.1f}MB  (Δ={late_v-early_v:+.1f})")

    if (late_g - early_g) > 200:
        print("[mem-test] WARNING: GTT grew by >200 MB across iterations — likely a leak.")
    else:
        print("[mem-test] OK: GTT is stable (no leak detected).")


if __name__ == "__main__":
    sys.exit(main() or 0)
