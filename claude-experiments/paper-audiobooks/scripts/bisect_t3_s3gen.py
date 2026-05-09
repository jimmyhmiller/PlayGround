"""Bisect chatterbox: is the muffled-audio bug in t3 (token generation) or
s3gen (token-to-audio decoding)?

Procedure:
1. Run model.generate() many times capturing the intermediate speech_tokens.
2. Score each run's audio.
3. For each captured (tokens, original_audio), re-run s3gen on the SAME tokens
   to see if it produces the same audio reproducibly.
4. Then swap: take a bad run's tokens and run s3gen → audio. Take a good run's
   tokens and run s3gen → audio. Compare.

Outcomes:
- If bad-tokens always make bad audio (and good-tokens always make good audio):
  the bug is upstream, in t3. The transformer is choosing bad token sequences.
- If bad-tokens sometimes make clean audio (s3gen non-deterministic):
  the bug is in s3gen.
- If bad-tokens replayed always make bad audio AND good-tokens replayed always
  make good audio: ambiguous between t3 and a deterministic-given-tokens s3gen.
  Then we ask: do the token sequences themselves look different?
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
from chatterbox.models.s3tokenizer import drop_invalid_tokens


VOICE_REF = "/home/jimmyhmiller/.config/paper-audiobooks/default-voice.wav"
CHUNK = (
    "The objection goes as follows: according to Christian belief, we human "
    "beings have been created by an all-powerful, all-knowing God who loves us "
    "enough to send his son, the second person of the divine Trinity, to "
    "suffer and die on our account; but given the devastating amount and "
    "variety of human suffering and evil in our sad world, this simply can't "
    "be true."
)
INITIAL_TRIALS = 12
RERUN_S3GEN_TRIALS = 5  # how many times to re-run s3gen on the same tokens


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


def generate_capture(model: ChatterboxTTS, text: str) -> tuple[torch.Tensor, np.ndarray, int]:
    """Replicate ChatterboxTTS.generate() but return both the speech_tokens
    AND the audio so we can later replay s3gen on the tokens."""
    import torch.nn.functional as F

    text = __import__("chatterbox.tts", fromlist=["punc_norm"]).punc_norm(text)
    text_tokens = model.tokenizer.text_to_tokens(text).to(model.device)

    cfg_weight = 0.5
    if cfg_weight > 0.0:
        text_tokens = torch.cat([text_tokens, text_tokens], dim=0)

    sot = model.t3.hp.start_text_token
    eot = model.t3.hp.stop_text_token
    text_tokens = F.pad(text_tokens, (1, 0), value=sot)
    text_tokens = F.pad(text_tokens, (0, 1), value=eot)

    with torch.inference_mode():
        speech_tokens = model.t3.inference(
            t3_cond=model.conds.t3,
            text_tokens=text_tokens,
            max_new_tokens=1000,
            temperature=0.8,
            cfg_weight=cfg_weight,
            repetition_penalty=1.2,
            min_p=0.05,
            top_p=1.0,
        )
        speech_tokens = speech_tokens[0]
        speech_tokens = drop_invalid_tokens(speech_tokens)
        speech_tokens = speech_tokens[speech_tokens < 6561]
        speech_tokens = speech_tokens.to(model.device)

        wav, _ = model.s3gen.inference(
            speech_tokens=speech_tokens,
            ref_dict=model.conds.gen,
        )
        wav = wav.squeeze(0).detach().cpu().numpy()
    sr = int(model.sr)
    # NOTE: skipping the watermarker — we want raw s3gen output for comparison.
    return speech_tokens.detach().cpu(), wav, sr


def replay_s3gen(model: ChatterboxTTS, speech_tokens: torch.Tensor) -> np.ndarray:
    """Re-run s3gen on a saved speech_tokens tensor."""
    with torch.inference_mode():
        st = speech_tokens.to(model.device)
        wav, _ = model.s3gen.inference(speech_tokens=st, ref_dict=model.conds.gen)
        return wav.squeeze(0).detach().cpu().numpy()


def main() -> None:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"loading chatterbox on {device}", flush=True)
    t0 = time.time()
    model = ChatterboxTTS.from_pretrained(device=device)
    print(f"loaded in {time.time()-t0:.1f}s", flush=True)

    out = Path("bisect_results")
    out.mkdir(exist_ok=True)
    model.prepare_conditionals(VOICE_REF, exaggeration=0.5)

    print(f"\n=== Phase 1: run {INITIAL_TRIALS} initial trials ===\n", flush=True)
    runs: list[dict] = []  # each: {trial, tokens, audio, stats, anomalous}
    for trial in range(1, INITIAL_TRIALS + 1):
        t0 = time.time()
        tokens, audio, sr = generate_capture(model, CHUNK)
        elapsed = time.time() - t0
        s = stats(audio, sr)
        anom = is_anomalous(s)
        runs.append({"trial": trial, "tokens": tokens, "audio": audio, "sr": sr,
                     "stats": s, "anomalous": anom})
        sf.write(out / f"phase1_trial{trial:02d}.wav", audio, sr)
        np.save(out / f"phase1_trial{trial:02d}_tokens.npy", tokens.numpy())
        flag = " ANOMALY" if anom else ""
        print(f"  trial {trial:2d}: gen={elapsed:5.1f}s "
              f"n_tokens={len(tokens):4d} dur={s['dur']:5.1f}s "
              f"rms={s['rms']:.4f} cen={s['centroid']:5.0f}Hz "
              f"<300={s['below_300']:.3f}{flag}", flush=True)

    bad = [r for r in runs if r["anomalous"]]
    good = [r for r in runs if not r["anomalous"]]
    print(f"\nPhase 1: {len(bad)} bad, {len(good)} good (out of {INITIAL_TRIALS})", flush=True)

    if not bad or not good:
        print("Need at least one bad and one good run — aborting bisect.", flush=True)
        return

    # Pick representatives.
    bad_run = bad[0]
    good_run = good[0]
    print(f"\nUsing bad=trial{bad_run['trial']} ({len(bad_run['tokens'])} tokens), "
          f"good=trial{good_run['trial']} ({len(good_run['tokens'])} tokens)\n", flush=True)

    # Compare token sequences quickly.
    bad_tokens = bad_run["tokens"]
    good_tokens = good_run["tokens"]
    print(f"Token comparison:", flush=True)
    print(f"  bad:  len={len(bad_tokens)}, min={int(bad_tokens.min())}, "
          f"max={int(bad_tokens.max())}, mean={float(bad_tokens.float().mean()):.1f}", flush=True)
    print(f"  good: len={len(good_tokens)}, min={int(good_tokens.min())}, "
          f"max={int(good_tokens.max())}, mean={float(good_tokens.float().mean()):.1f}", flush=True)

    print(f"\n=== Phase 2: replay s3gen on bad tokens × {RERUN_S3GEN_TRIALS} ===\n", flush=True)
    for i in range(1, RERUN_S3GEN_TRIALS + 1):
        wav = replay_s3gen(model, bad_tokens)
        s = stats(wav, model.sr)
        anom = is_anomalous(s)
        sf.write(out / f"phase2_bad_replay{i}.wav", wav, model.sr)
        flag = " ANOMALY" if anom else ""
        print(f"  replay {i}: rms={s['rms']:.4f} cen={s['centroid']:5.0f}Hz "
              f"<300={s['below_300']:.3f}{flag}", flush=True)

    print(f"\n=== Phase 3: replay s3gen on good tokens × {RERUN_S3GEN_TRIALS} ===\n", flush=True)
    for i in range(1, RERUN_S3GEN_TRIALS + 1):
        wav = replay_s3gen(model, good_tokens)
        s = stats(wav, model.sr)
        anom = is_anomalous(s)
        sf.write(out / f"phase3_good_replay{i}.wav", wav, model.sr)
        flag = " ANOMALY" if anom else ""
        print(f"  replay {i}: rms={s['rms']:.4f} cen={s['centroid']:5.0f}Hz "
              f"<300={s['below_300']:.3f}{flag}", flush=True)

    print("\n=== INTERPRETATION ===", flush=True)
    print("- If phase 2 (bad tokens) is consistently anomalous → t3 produced bad tokens; t3 is the bug.", flush=True)
    print("- If phase 2 sometimes clean → s3gen is non-deterministic; s3gen is the bug.", flush=True)
    print("- If phase 2 always anomalous AND phase 3 always clean: bug is in t3 (token quality matters).", flush=True)


if __name__ == "__main__":
    main()
