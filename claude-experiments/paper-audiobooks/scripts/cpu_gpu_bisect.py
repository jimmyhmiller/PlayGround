"""Locate where chatterbox s3gen diverges between CPU (always clean) and GPU
(intermittently muffled).

Strategy
--------
The bug only fires on GPU and only ~17-25% of the time, so we:
  1. Phase 1 (GPU): run model.generate() many times, capture all s3gen
     intermediates per trial — speech tokens, the initial flow-matching
     noise z, mu/mask/spks/cond, every ODE step x, the final mel, the
     final wav. Keep dumps for trials whose audio meets the muffled
     detector; discard the rest (their wavs are kept for reference).
  2. Phase 2 (CPU): for each captured bad GPU trial, reload the model on
     CPU and replay s3gen with the GPU's saved tokens and initial z
     injected so the inputs to solve_euler are byte-identical. Capture
     CPU intermediates the same way.
  3. Phase 3 (diff): for each bad-GPU vs paired-CPU pair, compare per
     stage: input tensors should be identical; the divergence stage tells
     us whether mu/cond differs (encoder bug), the trajectory diverges
     gradually (kernel-precision drift), or jumps suddenly (one bad
     kernel call).

Run inside the chatterbox venv:
    ~/.cache/paper-audiobooks/venvs/chatterbox/bin/python scripts/cpu_gpu_bisect.py

Outputs go to ./bisect_results/.
"""
from __future__ import annotations

import gc
import json
import os
import sys
import time
import types
from contextlib import contextmanager
from pathlib import Path

# Default to GPU phase; flip to CPU later via env. We do NOT touch
# CUDA_VISIBLE_DEVICES yet because phase 1 needs the GPU.
import numpy as np
import soundfile as sf
import torch
from scipy.signal import welch

VOICE_REF = "/home/jimmyhmiller/.config/paper-audiobooks/default-voice.wav"

# Chunks to bisect. Read by env BISECT_CHUNK_NAME (default: chunk19), or
# pass a path to a .txt file via BISECT_CHUNK_FILE.
CHUNKS = {
    # Original synthetic-repro chunk from CHATTERBOX_DEBUG.md (361 chars).
    "chunk19": (
        "The objection goes as follows: according to Christian belief, we human "
        "beings have been created by an all-powerful, all-knowing God who loves us "
        "enough to send his son, the second person of the divine Trinity, to "
        "suffer and die on our account; but given the devastating amount and "
        "variety of human suffering and evil in our sad world, this simply can't "
        "be true."
    ),
    # Captured halt 1 from Warrant and Proper Function chapter 1 (411 chars,
    # muffled / low-frequency dominant). Highest-confidence bad case.
    "halt1": (
        "The things we are most sure of, simple logical and arithmetical truths, "
        "such beliefs as that I now have a mild ache in my knee, that indeed I "
        "have knees, obvious perceptual truths, these are the sorts of beliefs "
        "we hold most firmly, perhaps with the maximum degree of firmness, and "
        "the ones such that we associate a very high degree of reliability with "
        "the modules of the design plan governing their production."
    ),
    # Captured halt 2 (232 chars, low-RMS / quiet failure mode).
    "halt2": (
        "Warrant: A First Approximation. One thought emerging from our survey of "
        "contemporary accounts of warrant, which are epistemic states of affairs "
        "or epistemic values, is that there are many different ways to be "
        "epistemically virtuous."
    ),
}
_NAME = os.environ.get("BISECT_CHUNK_NAME", "halt1")
if os.environ.get("BISECT_CHUNK_FILE"):
    CHUNK19 = Path(os.environ["BISECT_CHUNK_FILE"]).read_text(encoding="utf-8").strip()
    _NAME = "file"
else:
    CHUNK19 = CHUNKS[_NAME]
print(f"[bisect] using chunk {_NAME!r} ({len(CHUNK19)} chars)", flush=True)

OUT_DIR = Path(f"bisect_results_{_NAME}")
GPU_DIR = OUT_DIR / "gpu"
CPU_DIR = OUT_DIR / "cpu"
DIFF_DIR = OUT_DIR / "diff"

# GPU phase: keep going until we have at least this many bad trials, capped
# at MAX_GPU_TRIALS. Bug is process-state dependent, so a low-rate process
# might never produce one — accept that and stop. With known-bad chunks we
# can stop earlier (target 1) since each captured trial is enough to bisect.
TARGET_BAD_GPU_TRIALS = int(os.environ.get("BISECT_TARGET_BAD", "1"))
MAX_GPU_TRIALS = int(os.environ.get("BISECT_MAX_TRIALS", "30"))


def stats(audio: np.ndarray, sr: int) -> dict:
    f, p = welch(audio, sr, nperseg=2048)
    tot = float(np.sum(p)) + 1e-12
    return {
        "centroid_hz": float(np.sum(f * p) / tot),
        "below_300hz": float(np.sum(p[f < 300]) / tot),
        "300_to_2k": float(np.sum(p[(f >= 300) & (f < 2000)]) / tot),
        "rms": float(np.sqrt(np.mean(audio**2))),
        "duration_s": len(audio) / sr,
    }


def is_anomaly(s: dict) -> bool:
    return (s["centroid_hz"] < 700 and s["below_300hz"] > 0.5) or s["rms"] < 0.04


# ---------------------------------------------------------------------------
# Capture harness — patches s3gen at the instance level (no source edits).
# ---------------------------------------------------------------------------

class Capture:
    """Per-trial capture buffer. After each trial, materialize() writes
    every captured tensor to disk under prefix; clear() resets state."""

    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        self.cfm_inputs: dict | None = None  # mu, mask, spks, cond, n_timesteps, noised_mels
        self.z_init: torch.Tensor | None = None
        self.steps: list[torch.Tensor] = []  # x after each ODE step
        self.mel: torch.Tensor | None = None
        self.wav: torch.Tensor | None = None
        self.tokens: torch.Tensor | None = None
        self.ref_dict: dict | None = None

    def materialize(self, prefix: Path) -> None:
        prefix.mkdir(parents=True, exist_ok=True)
        if self.cfm_inputs is not None:
            torch.save({k: (v.detach().cpu() if torch.is_tensor(v) else v)
                        for k, v in self.cfm_inputs.items()},
                       prefix / "cfm_inputs.pt")
        if self.z_init is not None:
            torch.save(self.z_init.detach().cpu(), prefix / "z_init.pt")
        if self.steps:
            torch.save([s.detach().cpu() for s in self.steps], prefix / "steps.pt")
        if self.mel is not None:
            torch.save(self.mel.detach().cpu(), prefix / "mel.pt")
        if self.wav is not None:
            torch.save(self.wav.detach().cpu(), prefix / "wav.pt")
        if self.tokens is not None:
            torch.save(self.tokens.detach().cpu(), prefix / "tokens.pt")
        if self.ref_dict is not None:
            torch.save({k: (v.detach().cpu() if torch.is_tensor(v) else v)
                        for k, v in self.ref_dict.items()},
                       prefix / "ref_dict.pt")


def install_capture(model, cap: Capture, *, inject: dict | None = None) -> None:
    """Monkeypatch s3gen on `model` so:
      - The CFM's forward() captures (mu, mask, spks, cond, ...) and the
        z it samples (or, if inject['z_init'] is provided, uses that
        instead of torch.randn_like).
      - solve_euler captures x after every step.
      - flow_inference captures the final mel.
      - inference captures the final wav.

    Also captures speech tokens by patching s3gen.inference's first arg.

    inject (optional): {'z_init': Tensor, 'tokens': Tensor, 'ref_dict': dict}.
      If 'tokens' is set, model.s3gen.inference receives those tokens instead
      of whatever the caller passed.
    """
    s3gen = model.s3gen
    cfm = s3gen.flow.decoder  # CausalConditionalCFM
    inject = inject or {}

    # ---- CFM.forward: capture inputs + z, optionally inject z ----
    orig_cfm_forward = cfm.forward

    @torch.inference_mode()
    def patched_cfm_forward(self_cfm, mu, mask, n_timesteps, temperature=1.0,
                            spks=None, cond=None, noised_mels=None, meanflow=False):
        cap.cfm_inputs = {
            "mu": mu, "mask": mask, "spks": spks, "cond": cond,
            "n_timesteps": n_timesteps, "noised_mels": noised_mels,
            "meanflow": meanflow, "temperature": temperature,
            "t_scheduler": self_cfm.t_scheduler,
            "inference_cfg_rate": self_cfm.inference_cfg_rate,
        }

        if "z_init" in inject:
            z = inject["z_init"].to(mu.device).to(mu.dtype)
            assert z.shape == mu.shape, f"injected z shape {z.shape} != mu shape {mu.shape}"
        else:
            z = torch.randn_like(mu)

        if noised_mels is not None:
            prompt_len = mu.size(2) - noised_mels.size(2)
            z = z.clone()
            z[..., prompt_len:] = noised_mels

        cap.z_init = z

        t_span = torch.linspace(0, 1, n_timesteps + 1, device=mu.device, dtype=mu.dtype)
        if (not meanflow) and (self_cfm.t_scheduler == "cosine"):
            t_span = 1 - torch.cos(t_span * 0.5 * torch.pi)

        if meanflow:
            return self_cfm.basic_euler(z, t_span=t_span, mu=mu, mask=mask,
                                        spks=spks, cond=cond), None

        x = self_cfm.solve_euler(z, t_span=t_span, mu=mu, mask=mask,
                                 spks=spks, cond=cond, meanflow=meanflow)
        return x, None

    cfm.forward = types.MethodType(patched_cfm_forward, cfm)

    # ---- CFM.solve_euler: capture x after every step. We re-implement
    # rather than wrap because the original doesn't return intermediates.
    @torch.inference_mode()
    def patched_solve_euler(self_cfm, x, t_span, mu, mask, spks, cond, meanflow=False):
        from chatterbox.models.s3gen.flow_matching import cast_all
        in_dtype = x.dtype
        x, t_span, mu, mask, spks, cond = cast_all(
            x, t_span, mu, mask, spks, cond, dtype=self_cfm.estimator.dtype
        )

        B, T = mu.size(0), x.size(2)
        x_in    = torch.zeros([2 * B, 80, T], device=x.device, dtype=x.dtype)
        mask_in = torch.zeros([2 * B,  1, T], device=x.device, dtype=x.dtype)
        mu_in   = torch.zeros([2 * B, 80, T], device=x.device, dtype=x.dtype)
        t_in    = torch.zeros([2 * B       ], device=x.device, dtype=x.dtype)
        spks_in = torch.zeros([2 * B, 80   ], device=x.device, dtype=x.dtype)
        cond_in = torch.zeros([2 * B, 80, T], device=x.device, dtype=x.dtype)
        r_in    = torch.zeros([2 * B       ], device=x.device, dtype=x.dtype)

        for t, r in zip(t_span[:-1], t_span[1:]):
            t = t.unsqueeze(dim=0)
            r = r.unsqueeze(dim=0)
            x_in[:B] = x_in[B:] = x
            mask_in[:B] = mask_in[B:] = mask
            mu_in[:B] = mu
            t_in[:B] = t_in[B:] = t
            spks_in[:B] = spks
            cond_in[:B] = cond
            r_in[:B] = r_in[B:] = r
            dxdt = self_cfm.estimator.forward(
                x=x_in, mask=mask_in, mu=mu_in, t=t_in, spks=spks_in, cond=cond_in,
                r=r_in if meanflow else None,
            )
            dxdt, cfg_dxdt = torch.split(dxdt, [B, B], dim=0)
            dxdt = ((1.0 + self_cfm.inference_cfg_rate) * dxdt
                    - self_cfm.inference_cfg_rate * cfg_dxdt)
            dt = r - t
            x = x + dt * dxdt
            cap.steps.append(x.clone())

        return x.to(in_dtype)

    cfm.solve_euler = types.MethodType(patched_solve_euler, cfm)

    # ---- s3gen.flow_inference: capture mel ----
    orig_flow_inf = s3gen.flow_inference

    @torch.inference_mode()
    def patched_flow_inference(self_s3, speech_tokens, **kwargs):
        mel = orig_flow_inf(speech_tokens, **kwargs)
        cap.mel = mel
        return mel

    s3gen.flow_inference = types.MethodType(patched_flow_inference, s3gen)

    # ---- s3gen.inference: capture tokens, ref_dict, wav (and inject if asked) ----
    orig_s3_inf = s3gen.inference

    @torch.inference_mode()
    def patched_s3_inference(self_s3, speech_tokens, ref_wav=None, ref_sr=None,
                             ref_dict=None, drop_invalid_tokens=True,
                             n_cfm_timesteps=None, speech_token_lens=None):
        if "tokens" in inject:
            speech_tokens = inject["tokens"].to(self_s3.device)
            speech_token_lens = None  # let it recompute
        if "ref_dict" in inject:
            ref_wav = None
            ref_sr = None
            ref_dict = {k: (v.to(self_s3.device) if torch.is_tensor(v) else v)
                        for k, v in inject["ref_dict"].items()}

        cap.tokens = speech_tokens.clone() if torch.is_tensor(speech_tokens) else speech_tokens

        # We need ref_dict captured for replay; if the caller passed ref_wav,
        # let the underlying code embed it, then capture afterwards via the
        # flow_inference patch path. But here we need it BEFORE the call to
        # save it. Simplest: compute embed_ref ourselves and pass ref_dict.
        if ref_dict is None and ref_wav is not None:
            ref_dict = self_s3.embed_ref(ref_wav, ref_sr)
            ref_wav = None
            ref_sr = None
        cap.ref_dict = ref_dict

        wav = orig_s3_inf(speech_tokens, ref_wav=ref_wav, ref_sr=ref_sr,
                          ref_dict=ref_dict,
                          drop_invalid_tokens=drop_invalid_tokens,
                          n_cfm_timesteps=n_cfm_timesteps,
                          speech_token_lens=speech_token_lens)
        cap.wav = wav
        return wav

    s3gen.inference = types.MethodType(patched_s3_inference, s3gen)


# ---------------------------------------------------------------------------
# Phase runners
# ---------------------------------------------------------------------------

def run_gpu_phase() -> list[int]:
    """Return list of trial indices that produced anomalies."""
    assert torch.cuda.is_available(), "Phase 1 requires GPU"
    from chatterbox.tts import ChatterboxTTS
    print(f"[gpu] torch={torch.__version__}", flush=True)
    t0 = time.time()
    model = ChatterboxTTS.from_pretrained(device="cuda")
    print(f"[gpu] model loaded in {time.time()-t0:.1f}s", flush=True)

    cap = Capture()
    install_capture(model, cap)

    bad_trials: list[int] = []
    summary = []

    for trial in range(1, MAX_GPU_TRIALS + 1):
        cap.reset()
        t0 = time.time()
        wav = model.generate(CHUNK19, audio_prompt_path=VOICE_REF,
                             exaggeration=0.5, cfg_weight=0.5)
        gen_dt = time.time() - t0
        audio = wav.squeeze().cpu().numpy().astype("float32")
        s = stats(audio, model.sr)
        bad = is_anomaly(s)

        trial_dir = GPU_DIR / f"trial{trial:02d}"
        trial_dir.mkdir(parents=True, exist_ok=True)
        sf.write(trial_dir / "audio.wav", audio, model.sr)
        with open(trial_dir / "stats.json", "w") as f:
            json.dump({**s, "anomaly": bad, "gen_seconds": gen_dt}, f, indent=2)

        flag = "ANOMALY" if bad else "ok"
        print(f"[gpu trial {trial:2d}] gen={gen_dt:5.1f}s rms={s['rms']:.4f} "
              f"centroid={s['centroid_hz']:4.0f}Hz <300={s['below_300hz']:.3f}  [{flag}]",
              flush=True)
        summary.append({"trial": trial, **s, "anomaly": bad})

        if bad:
            cap.materialize(trial_dir / "dumps")
            bad_trials.append(trial)

        if len(bad_trials) >= TARGET_BAD_GPU_TRIALS:
            print(f"[gpu] reached {len(bad_trials)} bad trial(s) at trial {trial}, stopping", flush=True)
            break

    with open(GPU_DIR / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print(f"[gpu] done: {len(bad_trials)}/{len(summary)} anomalies "
          f"(trials: {bad_trials})", flush=True)
    return bad_trials


def run_cpu_phase(bad_trials: list[int]) -> None:
    """For each bad GPU trial, replay s3gen on CPU with GPU's tokens + z_init."""
    # New process is the right way to switch device, but we're in one
    # process. Force CPU by hiding GPU and reloading torch state. Easier:
    # spawn a child via subprocess. We'll exec ourselves with an env flag.
    # But for simplicity here, reload the model on CPU in the same process —
    # torch lets you create cpu tensors fine even with cuda visible.
    from chatterbox.tts import ChatterboxTTS
    print(f"\n[cpu] loading model on cpu...", flush=True)
    t0 = time.time()
    model_cpu = ChatterboxTTS.from_pretrained(device="cpu")
    print(f"[cpu] model loaded in {time.time()-t0:.1f}s", flush=True)

    cap = Capture()

    for trial_idx in bad_trials:
        gpu_trial = GPU_DIR / f"trial{trial_idx:02d}"
        dumps = gpu_trial / "dumps"
        if not (dumps / "tokens.pt").exists():
            print(f"[cpu] skip trial {trial_idx}: no GPU dumps", flush=True)
            continue

        tokens = torch.load(dumps / "tokens.pt", map_location="cpu")
        z_init = torch.load(dumps / "z_init.pt", map_location="cpu")
        ref_dict = torch.load(dumps / "ref_dict.pt", map_location="cpu")

        # Re-install capture each replay so tokens/ref_dict get re-injected.
        install_capture(model_cpu, cap, inject={
            "tokens": tokens,
            "z_init": z_init,
            "ref_dict": ref_dict,
        })
        cap.reset()

        print(f"[cpu replay] trial {trial_idx}: tokens={tuple(tokens.shape)} "
              f"z_init={tuple(z_init.shape)}", flush=True)
        t0 = time.time()
        wav = model_cpu.s3gen.inference(tokens.to("cpu"), ref_dict=ref_dict)
        gen_dt = time.time() - t0
        audio = wav.squeeze().cpu().numpy().astype("float32")
        s = stats(audio, model_cpu.sr)
        bad = is_anomaly(s)
        flag = "ANOMALY" if bad else "ok"
        print(f"[cpu replay] trial {trial_idx} done in {gen_dt:.1f}s "
              f"rms={s['rms']:.4f} centroid={s['centroid_hz']:.0f}Hz  [{flag}]",
              flush=True)

        out = CPU_DIR / f"trial{trial_idx:02d}"
        out.mkdir(parents=True, exist_ok=True)
        sf.write(out / "audio.wav", audio, model_cpu.sr)
        with open(out / "stats.json", "w") as f:
            json.dump({**s, "anomaly": bad, "gen_seconds": gen_dt}, f, indent=2)
        cap.materialize(out / "dumps")


def diff_phase(bad_trials: list[int]) -> None:
    print("\n[diff] per-stage CPU vs GPU comparison", flush=True)
    DIFF_DIR.mkdir(parents=True, exist_ok=True)

    for trial_idx in bad_trials:
        g = GPU_DIR / f"trial{trial_idx:02d}" / "dumps"
        c = CPU_DIR / f"trial{trial_idx:02d}" / "dumps"
        if not g.exists() or not c.exists():
            print(f"[diff] skip trial {trial_idx}: missing dumps", flush=True)
            continue

        report = {"trial": trial_idx, "stages": {}}
        print(f"\n--- trial {trial_idx} ---", flush=True)

        # Compare cfm inputs
        gi = torch.load(g / "cfm_inputs.pt", map_location="cpu")
        ci = torch.load(c / "cfm_inputs.pt", map_location="cpu")
        for k in ("mu", "mask", "spks", "cond"):
            gv, cv = gi.get(k), ci.get(k)
            if gv is None or cv is None:
                continue
            d = (gv.float() - cv.float()).abs()
            stat = {"max": float(d.max()), "mean": float(d.mean()),
                    "shape": tuple(gv.shape), "gpu_dtype": str(gv.dtype),
                    "cpu_dtype": str(cv.dtype)}
            report["stages"][f"cfm_input_{k}"] = stat
            print(f"  cfm_input.{k:6s} shape={stat['shape']} "
                  f"max_abs_diff={stat['max']:.3e} mean={stat['mean']:.3e}",
                  flush=True)

        # Compare z_init
        gz = torch.load(g / "z_init.pt", map_location="cpu")
        cz = torch.load(c / "z_init.pt", map_location="cpu")
        d = (gz.float() - cz.float()).abs()
        report["stages"]["z_init"] = {"max": float(d.max()), "mean": float(d.mean())}
        print(f"  z_init                max_abs_diff={d.max():.3e} mean={d.mean():.3e}",
              flush=True)

        # Compare each step
        gs = torch.load(g / "steps.pt", map_location="cpu")
        cs = torch.load(c / "steps.pt", map_location="cpu")
        n_steps = min(len(gs), len(cs))
        step_stats = []
        for i in range(n_steps):
            d = (gs[i].float() - cs[i].float()).abs()
            step_stats.append({"step": i + 1, "max": float(d.max()),
                               "mean": float(d.mean())})
            print(f"  step {i+1:2d}              max_abs_diff={d.max():.3e} "
                  f"mean={d.mean():.3e}", flush=True)
        report["stages"]["steps"] = step_stats

        # Compare mel
        gm = torch.load(g / "mel.pt", map_location="cpu")
        cm = torch.load(c / "mel.pt", map_location="cpu")
        d = (gm.float() - cm.float()).abs()
        report["stages"]["mel"] = {"max": float(d.max()), "mean": float(d.mean())}
        print(f"  mel                  max_abs_diff={d.max():.3e} mean={d.mean():.3e}",
              flush=True)

        # Compare wav
        gw = torch.load(g / "wav.pt", map_location="cpu")
        cw = torch.load(c / "wav.pt", map_location="cpu")
        n = min(gw.shape[-1], cw.shape[-1])
        d = (gw[..., :n].float() - cw[..., :n].float()).abs()
        report["stages"]["wav"] = {"max": float(d.max()), "mean": float(d.mean())}
        print(f"  wav (first {n} samples)  max_abs_diff={d.max():.3e} mean={d.mean():.3e}",
              flush=True)

        with open(DIFF_DIR / f"trial{trial_idx:02d}.json", "w") as f:
            json.dump(report, f, indent=2)


def main() -> None:
    OUT_DIR.mkdir(exist_ok=True)
    bad_trials = run_gpu_phase()
    if not bad_trials:
        print("\n[main] no anomalies on GPU this run — process may be in low-rate "
              "state. Try again or accept and exit.", flush=True)
        return

    # Free GPU memory before loading CPU model.
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    run_cpu_phase(bad_trials)
    diff_phase(bad_trials)
    print("\n[main] done. Inspect ./bisect_results/", flush=True)


if __name__ == "__main__":
    main()
