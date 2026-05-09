"""Scan an m4b for likely TTS distortion windows.

Approach: decode to mono float32 with ffmpeg, walk in fixed windows, score each
window on several signals, print the windows that exceed thresholds. Designed
for chatterbox-style failures — clipping, sudden short energy spikes (pops),
and high-frequency hash from broken phoneme generation.

Usage:
  uv run python scripts/audio_glitch_finder.py output/foo.m4b
  uv run python scripts/audio_glitch_finder.py output/foo.m4b --top 30
  uv run python scripts/audio_glitch_finder.py output/foo.m4b --extract-clips suspect_clips/
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import soundfile as sf
from scipy.signal import butter, sosfiltfilt


SR = 24000  # we resample to this for analysis
WIN_SEC = 0.5
HOP_SEC = 0.25


@dataclass
class Window:
    start: float
    score: float
    reasons: list[str]
    rms: float
    clip_frac: float
    hf_ratio: float
    peak_burst: float


def load_audio(path: Path) -> np.ndarray:
    """Decode m4b to mono float32 at SR via ffmpeg."""
    proc = subprocess.run(
        ["ffmpeg", "-v", "error", "-i", str(path),
         "-ac", "1", "-ar", str(SR), "-f", "f32le", "-"],
        check=True, capture_output=True,
    )
    return np.frombuffer(proc.stdout, dtype=np.float32)


def get_chapters(path: Path) -> list[tuple[float, float, str]]:
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-show_chapters", "-of", "json", str(path)],
        check=True, capture_output=True, text=True,
    ).stdout
    d = json.loads(out)
    chapters = []
    for c in d.get("chapters", []):
        chapters.append((float(c["start_time"]), float(c["end_time"]),
                         c.get("tags", {}).get("title", "")))
    return chapters


def hf_energy_ratio(window: np.ndarray, sr: int = SR) -> float:
    """Energy above 4kHz / total energy. Speech mostly lives below 4kHz; high
    values signal hiss, buzz, or broken phonemes."""
    if len(window) < 64:
        return 0.0
    sos = butter(4, 4000.0 / (sr / 2), btype="highpass", output="sos")
    hp = sosfiltfilt(sos, window)
    total = float(np.sum(window**2)) + 1e-12
    return float(np.sum(hp**2) / total)


def peak_burst_score(window: np.ndarray) -> float:
    """Detect short high-amplitude bursts (pops/clicks): max instantaneous
    sample / median |sample| in the window."""
    a = np.abs(window)
    med = float(np.median(a)) + 1e-6
    return float(np.max(a) / med)


def clip_fraction(window: np.ndarray, thresh: float = 0.99) -> float:
    return float(np.mean(np.abs(window) >= thresh))


def scan(audio: np.ndarray) -> list[Window]:
    win = int(WIN_SEC * SR)
    hop = int(HOP_SEC * SR)
    out: list[Window] = []
    # global stats for adaptive thresholds
    rms_full = np.sqrt(np.mean(audio**2)) + 1e-9
    for start in range(0, len(audio) - win, hop):
        seg = audio[start:start + win]
        rms = float(np.sqrt(np.mean(seg**2)))
        if rms < 0.005:
            continue  # silence, skip
        clip = clip_fraction(seg)
        hf = hf_energy_ratio(seg)
        burst = peak_burst_score(seg)
        reasons: list[str] = []
        score = 0.0
        if clip > 0.001:  # >0.1% of samples clipping
            reasons.append(f"clip={clip:.3%}")
            score += clip * 1000
        if hf > 0.35:  # speech is normally <0.15
            reasons.append(f"hf={hf:.2f}")
            score += (hf - 0.35) * 50
        if burst > 25:  # peak >> median = pop
            reasons.append(f"burst={burst:.1f}")
            score += (burst - 25) / 5
        if rms > 5 * rms_full:  # window much louder than book average
            reasons.append(f"loud={rms/rms_full:.1f}x")
            score += rms / rms_full
        if reasons:
            out.append(Window(
                start=start / SR, score=score, reasons=reasons,
                rms=rms, clip_frac=clip, hf_ratio=hf, peak_burst=burst,
            ))
    return out


def cluster(windows: list[Window], gap_sec: float = 1.0) -> list[Window]:
    """Merge adjacent suspect windows into single events; keep the highest-scoring."""
    if not windows:
        return []
    windows = sorted(windows, key=lambda w: w.start)
    merged: list[Window] = [windows[0]]
    for w in windows[1:]:
        if w.start - (merged[-1].start + WIN_SEC) <= gap_sec:
            if w.score > merged[-1].score:
                merged[-1] = w
        else:
            merged.append(w)
    return merged


def chapter_at(t: float, chapters: list[tuple[float, float, str]]) -> tuple[int, str, float]:
    for i, (s, e, title) in enumerate(chapters):
        if s <= t < e:
            return i, title, t - s
    return -1, "", 0.0


def fmt_time(t: float) -> str:
    return f"{int(t//3600):02d}:{int((t%3600)//60):02d}:{t%60:06.3f}"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("m4b", type=Path)
    ap.add_argument("--top", type=int, default=20, help="show top-N suspect events")
    ap.add_argument("--extract-clips", type=Path, default=None,
                    help="directory to write 3s wav clips of each suspect event")
    args = ap.parse_args()

    print(f"loading {args.m4b}", file=sys.stderr)
    audio = load_audio(args.m4b)
    print(f"  {len(audio)/SR:.1f}s, peak={np.max(np.abs(audio)):.3f}, "
          f"rms={np.sqrt(np.mean(audio**2)):.3f}", file=sys.stderr)
    chapters = get_chapters(args.m4b)
    print(f"  {len(chapters)} chapter(s)", file=sys.stderr)

    windows = scan(audio)
    events = cluster(windows)
    events.sort(key=lambda w: w.score, reverse=True)
    print(f"\n{len(events)} suspect event(s); showing top {min(args.top, len(events))}:\n")

    if args.extract_clips:
        args.extract_clips.mkdir(parents=True, exist_ok=True)

    for rank, w in enumerate(events[:args.top], 1):
        ch_idx, ch_title, ch_off = chapter_at(w.start, chapters)
        print(f"#{rank:3d} score={w.score:6.1f}  t={fmt_time(w.start)}  "
              f"ch[{ch_idx}]+{fmt_time(ch_off)}  {ch_title[:50]!r}")
        print(f"     {' '.join(w.reasons)}")
        if args.extract_clips:
            clip_start = max(0.0, w.start - 1.0)
            clip_end = min(len(audio) / SR, w.start + 2.0)
            seg = audio[int(clip_start * SR):int(clip_end * SR)]
            fname = args.extract_clips / f"event{rank:03d}_t{int(w.start)}_ch{ch_idx}.wav"
            sf.write(fname, seg, SR)


if __name__ == "__main__":
    main()
