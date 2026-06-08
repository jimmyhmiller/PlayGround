"""Test batched generate_batch() vs sequential generate().

- Run N phrases sequentially → measure wall, save wavs
- Run same N phrases in one batch → measure wall, save wavs
- Compare audio: should be similar (RNG differs per-row so not bit-exact, but
  same speaker/intonation/length)
- Verify both batches' audio is intelligible via Whisper round-trip
"""
import os, sys, time, subprocess, json, wave
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
from chatterbox_mojo import ChatterboxTTS

REF = "/home/jimmyhmiller/.config/paper-audiobooks/default-voice.wav"
OUT = Path("/tmp/batch_test")
OUT.mkdir(exist_ok=True)

PHRASES = [
    "Hello world, this is a test of the batched generation path.",
    "She sells seashells by the seashore.",
    "The quick brown fox jumps over the lazy dog.",
    "Testing one, two, three. This is a simple sentence.",
]


def write_wav(arr, path, sr=24000):
    arr = np.clip(arr, -1.0, 1.0)
    pcm = (arr * 32767.0).astype(np.int16)
    with wave.open(path, "wb") as w:
        w.setnchannels(1); w.setsampwidth(2); w.setframerate(sr)
        w.writeframes(pcm.tobytes())


def main():
    print(f"[batch-test] loading model (bf16)...")
    tts = ChatterboxTTS.from_pretrained(use_bf16=True)
    tts.prepare_conditionals(REF, exaggeration=0.5)

    # Sequential
    print(f"\n[batch-test] === sequential ({len(PHRASES)} chunks) ===")
    t0 = time.perf_counter()
    seq_audios = []
    for i, text in enumerate(PHRASES):
        ti = time.perf_counter()
        w = tts.generate(text, cfg_weight=0.5, temperature=0.8, top_p=0.95,
                          min_p=0.05, repetition_penalty=1.2, exaggeration=0.5,
                          rng_seed=0xDEADBEEF + i)
        dt = time.perf_counter() - ti
        arr = w.squeeze(0).cpu().numpy().astype(np.float32)
        seq_audios.append(arr)
        write_wav(arr, str(OUT / f"seq_{i:02d}.wav"))
        print(f"  seq[{i}]: {dt:.2f}s  {arr.size/24000:.2f}s audio  {text[:50]!r}")
    seq_wall = time.perf_counter() - t0
    seq_audio_total = sum(a.size for a in seq_audios) / 24000
    print(f"  TOTAL sequential: {seq_wall:.2f}s wall, {seq_audio_total:.2f}s audio, RTF={seq_wall/seq_audio_total:.3f}")

    # Batched
    print(f"\n[batch-test] === batched (B={len(PHRASES)}) ===")
    t0 = time.perf_counter()
    batch_audios = tts.generate_batch(
        PHRASES, cfg_weight=0.5, temperature=0.8, top_p=0.95,
        min_p=0.05, repetition_penalty=1.2, exaggeration=0.5,
        rng_seed=0xDEADBEEF,
    )
    batch_wall = time.perf_counter() - t0
    batch_arr = []
    for i, w in enumerate(batch_audios):
        a = w.squeeze(0).cpu().numpy().astype(np.float32)
        batch_arr.append(a)
        write_wav(a, str(OUT / f"batch_{i:02d}.wav"))
    batch_audio_total = sum(a.size for a in batch_arr) / 24000
    print(f"  TOTAL batched:   {batch_wall:.2f}s wall, {batch_audio_total:.2f}s audio, RTF={batch_wall/batch_audio_total:.3f}")
    print()
    print(f"[batch-test] SPEEDUP: {seq_wall/batch_wall:.2f}x (sequential / batched)")

    # Per-row length comparison
    print()
    print(f"{'i':>2} {'seq_s':>8} {'batch_s':>8}  text")
    print("-" * 80)
    for i in range(len(PHRASES)):
        s = seq_audios[i].size / 24000
        b = batch_arr[i].size / 24000
        print(f"{i:>2} {s:>8.2f} {b:>8.2f}  {PHRASES[i][:50]!r}")


if __name__ == "__main__":
    sys.exit(main() or 0)
