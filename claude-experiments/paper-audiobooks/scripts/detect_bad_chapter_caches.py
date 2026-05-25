"""Run the chatterbox muffled-output detector across cached chapter .npy files.

Each cached chapter is one big concatenated waveform — we slice it into
~12s windows and apply the same detector rule used in chatterbox.py
(_is_anomalous). A chapter is "bad" if any window fires.

Usage:
    python scripts/detect_bad_chapter_caches.py <cache_dir> [--delete]

Without --delete it only reports. With --delete it removes any chapter
.npy that has at least one bad window so the next pipeline run will
re-render it.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
from scipy.signal import welch

SAMPLE_RATE = 24000

# Chunks are concatenated with a 0.25s silence gap (see tts/__init__.py
# render_audio). We detect those silences to recover per-chunk boundaries
# and run the same detector the live pipeline uses on each chunk.
SILENCE_RMS_THRESHOLD = 0.005
MIN_SILENCE_S = 0.20  # the inserted gap is 0.25s, give some margin
MIN_CHUNK_S = 1.0     # ignore tiny fragments


def is_window_anomalous(audio: np.ndarray, sr: int) -> tuple[bool, dict]:
    """Same rule as chatterbox.py _is_anomalous, plus a severity tier.
    bad = (centroid < 700Hz AND <300Hz energy > 0.5) OR rms < 0.04.
    severity = "strong" if rms<0.04 (the unambiguous muffled signal),
               "weak" if only the spectral clause fires."""
    if audio.ndim > 1:
        audio = audio.mean(axis=-1)
    f, p = welch(audio, sr, nperseg=2048)
    tot = float(np.sum(p)) + 1e-12
    s = {
        "centroid_hz": float(np.sum(f * p) / tot),
        "below_300hz": float(np.sum(p[f < 300]) / tot),
        "rms": float(np.sqrt(np.mean(audio ** 2))),
    }
    spectral_bad = s["centroid_hz"] < 700 and s["below_300hz"] > 0.5
    rms_bad = s["rms"] < 0.04
    bad = spectral_bad or rms_bad
    s["severity"] = "strong" if rms_bad else ("weak" if spectral_bad else "ok")
    return bad, s


def split_into_chunks(audio: np.ndarray, sr: int) -> list[tuple[int, int]]:
    """Find chunk boundaries by detecting the inserted ~0.25s silences.
    Returns list of (start_sample, end_sample) for each speech chunk."""
    # Compute short-frame RMS to detect silence regions
    frame = int(sr * 0.05)  # 50ms frames
    n_frames = len(audio) // frame
    if n_frames == 0:
        return [(0, len(audio))]
    rms = np.array([
        np.sqrt(np.mean(audio[i * frame:(i + 1) * frame] ** 2))
        for i in range(n_frames)
    ])
    is_silent = rms < SILENCE_RMS_THRESHOLD
    min_silent_frames = int(MIN_SILENCE_S / 0.05)

    # Find runs of silence long enough to be inter-chunk gaps
    chunks: list[tuple[int, int]] = []
    chunk_start = 0
    i = 0
    while i < n_frames:
        if is_silent[i]:
            run_start = i
            while i < n_frames and is_silent[i]:
                i += 1
            run_end = i
            if run_end - run_start >= min_silent_frames:
                # Real inter-chunk gap; close current chunk
                speech_end = run_start * frame
                if speech_end - chunk_start >= int(sr * MIN_CHUNK_S):
                    chunks.append((chunk_start, speech_end))
                chunk_start = run_end * frame
        else:
            i += 1
    # Final chunk
    if len(audio) - chunk_start >= int(sr * MIN_CHUNK_S):
        chunks.append((chunk_start, len(audio)))
    return chunks


def scan_chapter(path: Path, sr: int = SAMPLE_RATE) -> dict:
    audio = np.load(path).astype(np.float32)
    n_total = len(audio)
    chunks = split_into_chunks(audio, sr)
    bad_chunks: list[tuple[int, float, dict]] = []
    n_strong = 0
    for idx, (start, end) in enumerate(chunks):
        clip = audio[start:end]
        bad, s = is_window_anomalous(clip, sr)
        if bad:
            bad_chunks.append((idx, start / sr, s))
            if s["severity"] == "strong":
                n_strong += 1
    return {
        "path": path,
        "duration_min": n_total / sr / 60,
        "n_windows": len(chunks),
        "n_bad": len(bad_chunks),
        "n_strong": n_strong,
        "bad_windows": bad_chunks,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("cache_dir", type=Path)
    ap.add_argument("--delete", action="store_true",
                    help="Delete chapter .npy files with any bad window")
    ap.add_argument("--threshold", type=int, default=1,
                    help="Minimum bad-window count to flag a chapter (default 1)")
    ap.add_argument("--strong-only", action="store_true",
                    help="Only flag chapters with at least one rms<0.04 chunk "
                         "(unambiguous muffled signal). Use this to filter "
                         "out spectral-clause false positives.")
    args = ap.parse_args()

    if not args.cache_dir.is_dir():
        print(f"not a directory: {args.cache_dir}", file=sys.stderr)
        return 2

    chapters = sorted(args.cache_dir.glob("chapter-*.npy"))
    if not chapters:
        print(f"no chapter-*.npy files in {args.cache_dir}", file=sys.stderr)
        return 2

    print(f"scanning {len(chapters)} chapter(s) in {args.cache_dir}")
    print(f"{'chapter':<22}{'dur_min':>9}{'chunks':>9}{'weak':>6}{'strong':>8}  status")
    print("-" * 76)
    bad_chapters: list[Path] = []
    for p in chapters:
        r = scan_chapter(p)
        if args.strong_only:
            flagged = r["n_strong"] >= args.threshold
        else:
            flagged = r["n_bad"] >= args.threshold
        status = "BAD" if flagged else "ok"
        n_weak = r["n_bad"] - r["n_strong"]
        print(f"{p.name:<22}{r['duration_min']:>9.1f}{r['n_windows']:>9}"
              f"{n_weak:>6}{r['n_strong']:>8}  {status}")
        if flagged:
            bad_chapters.append(p)
            # Show strong hits first, then up to 3 weak hits
            strong = [w for w in r["bad_windows"] if w[2]["severity"] == "strong"]
            weak = [w for w in r["bad_windows"] if w[2]["severity"] == "weak"]
            for idx, t, s in strong:
                print(f"    [STRONG] chunk {idx:4d} @{t:7.1f}s: "
                      f"centroid={s['centroid_hz']:.0f}Hz "
                      f"low={s['below_300hz']:.2f} rms={s['rms']:.3f}")
            for idx, t, s in weak[:3]:
                print(f"    [weak]   chunk {idx:4d} @{t:7.1f}s: "
                      f"centroid={s['centroid_hz']:.0f}Hz "
                      f"low={s['below_300hz']:.2f} rms={s['rms']:.3f}")
            if len(weak) > 3:
                print(f"    [weak]   ... +{len(weak) - 3} more")

    print("-" * 70)
    print(f"flagged {len(bad_chapters)}/{len(chapters)} chapter(s)")

    if bad_chapters and args.delete:
        for p in bad_chapters:
            print(f"deleting {p}")
            p.unlink()
    elif bad_chapters:
        print("re-run with --delete to remove these so the pipeline regenerates them")

    return 0


if __name__ == "__main__":
    sys.exit(main())
