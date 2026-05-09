"""Confirm the hypothesis that s3gen's flow-matching noise initialization is
the root cause of muffled-audio anomalies.

Tests:
- A: Same seed + same tokens → bit-identical output (proves seeding works).
- B: Sweep 50 seeds on the saved bad-token sequence; count anomalies. Tells
  us what fraction of seeds produce bad output.
- C: Sweep 50 seeds on a saved good-token sequence; count anomalies. If
  similar rate, the bug is purely in s3gen noise (token-independent). If
  zero, bad tokens make s3gen more sensitive.
- D: Try halving the flow-matching noise scale and see if anomaly rate drops.

Loads the speech_tokens captured by bisect_t3_s3gen.py from bisect_results/.
"""
from __future__ import annotations

import os
import time
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
from scipy.signal import welch

os.environ.setdefault("PYTORCH_HIP_ALLOC_CONF", "expandable_segments:True")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

from chatterbox.tts import ChatterboxTTS


VOICE_REF = "/home/jimmyhmiller/.config/paper-audiobooks/default-voice.wav"
N_SEEDS = 50


def stats(audio: np.ndarray, sr: int) -> dict:
    f, p = welch(audio, sr, nperseg=2048)
    tot = float(np.sum(p)) + 1e-12
    return {
        "centroid": float(np.sum(f * p) / tot),
        "below_300": float(np.sum(p[f < 300]) / tot),
        "rms": float(np.sqrt(np.mean(audio**2))),
        "dur": len(audio) / sr,
    }


def is_anomalous(s: dict) -> bool:
    return (s["centroid"] < 700 and s["below_300"] > 0.5) or s["rms"] < 0.04


def replay(model: ChatterboxTTS, tokens: torch.Tensor, *, seed: int | None = None) -> np.ndarray:
    if seed is not None:
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    with torch.inference_mode():
        st = tokens.to(model.device)
        wav, _ = model.s3gen.inference(speech_tokens=st, ref_dict=model.conds.gen)
        return wav.squeeze(0).detach().cpu().numpy()


def _score_wav(path: Path) -> dict:
    a, sr = sf.read(path)
    a = a.astype(np.float32)
    f, p = welch(a, sr, nperseg=2048)
    tot = float(np.sum(p)) + 1e-12
    return {
        "centroid": float(np.sum(f * p) / tot),
        "below_300": float(np.sum(p[f < 300]) / tot),
        "rms": float(np.sqrt(np.mean(a**2))),
    }


def find_token_files() -> tuple[torch.Tensor, torch.Tensor] | None:
    """Pick one ANOMALOUS trial and one CLEAN trial by re-scoring the saved wavs.

    We previously selected by token-count, but that turned out to be wrong:
    anomalous trials had similar token counts to clean ones. Score the saved
    audio directly to find a true-bad and true-good pair.
    """
    bisect_dir = Path("bisect_results")
    if not bisect_dir.exists():
        print("ERROR: bisect_results/ not found. Run bisect_t3_s3gen.py first.")
        return None
    bad_tokens = good_tokens = None
    bad_label = good_label = ""
    for tok_path in sorted(bisect_dir.glob("phase1_trial*_tokens.npy")):
        wav_path = tok_path.with_name(tok_path.name.replace("_tokens.npy", ".wav"))
        if not wav_path.exists():
            continue
        s = _score_wav(wav_path)
        anom = is_anomalous(s)
        arr = np.load(tok_path)
        label = (f"{tok_path.name} ({len(arr)} tokens, rms={s['rms']:.3f}, "
                 f"cen={s['centroid']:.0f}Hz, <300={s['below_300']:.2f})")
        if anom and bad_tokens is None:
            bad_tokens = torch.from_numpy(arr)
            bad_label = label
            print(f"Loaded BAD  tokens: {label}")
        elif (not anom) and good_tokens is None:
            good_tokens = torch.from_numpy(arr)
            good_label = label
            print(f"Loaded GOOD tokens: {label}")
        if bad_tokens is not None and good_tokens is not None:
            break
    if bad_tokens is None:
        print("ERROR: no anomalous trial wav found (need at least one with the muffled signature).")
        return None
    if good_tokens is None:
        print("ERROR: no clean trial wav found.")
        return None
    return bad_tokens, good_tokens


def test_a_seeding_reproducibility(model: ChatterboxTTS, tokens: torch.Tensor) -> None:
    print("\n=== A: Seed reproducibility — same seed, same tokens → identical wav? ===\n")
    sr = int(model.sr)
    out_dir = Path("noise_confirm")
    out_dir.mkdir(exist_ok=True)

    a1 = replay(model, tokens, seed=42)
    a2 = replay(model, tokens, seed=42)
    a3 = replay(model, tokens, seed=43)
    sf.write(out_dir / "A_seed42_run1.wav", a1, sr)
    sf.write(out_dir / "A_seed42_run2.wav", a2, sr)
    sf.write(out_dir / "A_seed43.wav", a3, sr)

    if a1.shape != a2.shape:
        print(f"  seed42 run1 shape={a1.shape} run2 shape={a2.shape} — DIFFERENT shapes!")
    else:
        diff = float(np.abs(a1 - a2).max())
        print(f"  seed42 run1 vs seed42 run2: max abs diff = {diff:.6e}  "
              f"({'IDENTICAL' if diff < 1e-6 else 'DIFFERENT'})")
    if a1.shape == a3.shape:
        diff = float(np.abs(a1 - a3).max())
        print(f"  seed42 vs seed43:           max abs diff = {diff:.6e}  "
              f"({'identical' if diff < 1e-6 else 'different (expected)'})")


def test_b_seed_sweep(model: ChatterboxTTS, tokens: torch.Tensor, label: str) -> int:
    print(f"\n=== {label}: Seed sweep on tokens (n_tokens={len(tokens)}) — {N_SEEDS} seeds ===\n")
    sr = int(model.sr)
    out_dir = Path("noise_confirm")
    out_dir.mkdir(exist_ok=True)
    anomalies = 0
    for seed in range(N_SEEDS):
        wav = replay(model, tokens, seed=seed)
        s = stats(wav, sr)
        anom = is_anomalous(s)
        if anom:
            anomalies += 1
            sf.write(out_dir / f"{label}_seed{seed:03d}_ANOMALY.wav", wav, sr)
        flag = " ANOMALY" if anom else ""
        print(f"  seed {seed:3d}: rms={s['rms']:.4f} cen={s['centroid']:5.0f}Hz "
              f"<300={s['below_300']:.3f}{flag}", flush=True)
    return anomalies


def main() -> None:
    pair = find_token_files()
    if pair is None:
        return
    bad_tokens, good_tokens = pair

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\nloading chatterbox on {device}", flush=True)
    t0 = time.time()
    model = ChatterboxTTS.from_pretrained(device=device)
    print(f"loaded in {time.time()-t0:.1f}s", flush=True)
    model.prepare_conditionals(VOICE_REF, exaggeration=0.5)

    test_a_seeding_reproducibility(model, bad_tokens)
    bad_anom = test_b_seed_sweep(model, bad_tokens, "B_BAD")
    good_anom = test_b_seed_sweep(model, good_tokens, "C_GOOD")

    print("\n=== SUMMARY ===")
    print(f"  bad-token  seed sweep: {bad_anom}/{N_SEEDS} anomalies")
    print(f"  good-token seed sweep: {good_anom}/{N_SEEDS} anomalies")
    print()
    print("If both rates are similar: bug is in s3gen noise sampling, token-independent.")
    print("If bad-token rate is much higher: bad t3 tokens make s3gen more sensitive to noise.")
    print("If good-token rate is ~0: t3 produces tokens that interact badly with bad noise.")


if __name__ == "__main__":
    main()
