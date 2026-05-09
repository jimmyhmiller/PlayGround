"""Replay a captured GPU anomaly on CPU and diff every s3gen stage.

Input: a halt-<slug>-dumps/ directory produced by the chatterbox backend
when CHATTERBOX_HALT_ON_ANOMALY=1 fires. That directory must contain
tokens.pt, ref_dict.pt, z_init.pt, cfm_inputs.pt, steps.pt, mel.pt, wav.pt.

What this script does:
  1. Load the chatterbox model on CPU (no GPU touched).
  2. Install a copy of the in-process capture harness (same patches as
     the production backend) so we record CPU intermediates the same way.
  3. Inject the GPU-captured speech tokens, ref_dict, and initial noise z
     into model.s3gen.inference and model.s3gen.flow.decoder.forward, so
     the input to solve_euler is byte-identical to what the GPU saw.
  4. Run inference. Save CPU dumps next to the GPU dumps.
  5. Compare per stage: mu/mask/spks/cond should be identical (CPU
     re-derives them from tokens+ref_dict, so any drift here means the
     CPU encoder differs); each ODE step x; final mel; final wav.

Run inside the chatterbox venv:
    ~/.cache/paper-audiobooks/venvs/chatterbox/bin/python \\
        scripts/replay_halt_on_cpu.py bisect_results/halt/halt-<slug>-dumps
"""
from __future__ import annotations

import os
import sys
import time
import types
from pathlib import Path

# Hide GPU before torch starts so no HIP context is created.
os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["HIP_VISIBLE_DEVICES"] = ""

import json
import numpy as np
import soundfile as sf
import torch

assert not torch.cuda.is_available(), "must run with no visible GPU"

from chatterbox.tts import ChatterboxTTS


def _install_capture(model, cap, *, inject):
    """Same patches as the production capture harness, but with optional
    injection of tokens / ref_dict / z_init for replay determinism."""
    s3gen = model.s3gen
    cfm = s3gen.flow.decoder

    @torch.inference_mode()
    def patched_cfm_forward(self_cfm, mu, mask, n_timesteps, temperature=1.0,
                            spks=None, cond=None, noised_mels=None, meanflow=False):
        cap["cfm_inputs"] = {
            "mu": mu, "mask": mask, "spks": spks, "cond": cond,
            "n_timesteps": n_timesteps, "noised_mels": noised_mels,
            "meanflow": meanflow, "temperature": temperature,
            "t_scheduler": self_cfm.t_scheduler,
            "inference_cfg_rate": self_cfm.inference_cfg_rate,
        }
        if "z_init" in inject:
            z = inject["z_init"].to(mu.device).to(mu.dtype)
            assert z.shape == mu.shape, f"injected z {tuple(z.shape)} != mu {tuple(mu.shape)}"
        else:
            z = torch.randn_like(mu)
        if noised_mels is not None:
            prompt_len = mu.size(2) - noised_mels.size(2)
            z = z.clone()
            z[..., prompt_len:] = noised_mels
        cap["z_init"] = z
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
            t = t.unsqueeze(dim=0); r = r.unsqueeze(dim=0)
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
            cap["steps"].append(x.clone())
        return x.to(in_dtype)
    cfm.solve_euler = types.MethodType(patched_solve_euler, cfm)

    orig_flow_inf = s3gen.flow_inference
    @torch.inference_mode()
    def patched_flow_inference(self_s3, speech_tokens, **kwargs):
        mel = orig_flow_inf(speech_tokens, **kwargs)
        cap["mel"] = mel
        return mel
    s3gen.flow_inference = types.MethodType(patched_flow_inference, s3gen)


def _diff(name, gpu, cpu):
    if gpu is None or cpu is None:
        return None
    if torch.is_tensor(gpu) and torch.is_tensor(cpu):
        if gpu.shape != cpu.shape:
            return {"name": name, "gpu_shape": tuple(gpu.shape),
                    "cpu_shape": tuple(cpu.shape), "note": "shape mismatch"}
        d = (gpu.float() - cpu.float()).abs()
        return {"name": name, "shape": tuple(gpu.shape),
                "max_abs": float(d.max()), "mean_abs": float(d.mean()),
                "gpu_dtype": str(gpu.dtype), "cpu_dtype": str(cpu.dtype),
                "gpu_norm": float(gpu.float().norm()),
                "cpu_norm": float(cpu.float().norm())}
    return None


def main():
    if len(sys.argv) != 2:
        print(f"usage: {sys.argv[0]} <halt-dumps-dir>", file=sys.stderr)
        sys.exit(2)

    dumps = Path(sys.argv[1])
    assert dumps.is_dir(), f"not a dir: {dumps}"
    out_dir = dumps.parent / (dumps.name + "-cpu-replay")
    out_dir.mkdir(exist_ok=True)
    print(f"[replay] dumps from: {dumps}", flush=True)
    print(f"[replay] cpu output:  {out_dir}", flush=True)

    gpu_tokens = torch.load(dumps / "tokens.pt", map_location="cpu")
    gpu_ref = torch.load(dumps / "ref_dict.pt", map_location="cpu")
    gpu_z = torch.load(dumps / "z_init.pt", map_location="cpu")
    gpu_cfm = torch.load(dumps / "cfm_inputs.pt", map_location="cpu")
    gpu_steps = torch.load(dumps / "steps.pt", map_location="cpu")
    gpu_mel = torch.load(dumps / "mel.pt", map_location="cpu")
    gpu_wav = torch.load(dumps / "wav.pt", map_location="cpu")

    print(f"[replay] gpu tokens={tuple(gpu_tokens.shape)}", flush=True)
    print(f"[replay] gpu z_init={tuple(gpu_z.shape)}", flush=True)
    print(f"[replay] gpu mel={tuple(gpu_mel.shape)}", flush=True)
    print(f"[replay] gpu wav={tuple(gpu_wav.shape)}", flush=True)

    print(f"[replay] loading model on cpu...", flush=True)
    t0 = time.time()
    model = ChatterboxTTS.from_pretrained(device="cpu")
    print(f"[replay] model loaded in {time.time()-t0:.1f}s", flush=True)

    cap = {"cfm_inputs": None, "z_init": None, "steps": [], "mel": None}
    _install_capture(model, cap, inject={"z_init": gpu_z})

    # Move ref_dict tensors to CPU explicitly.
    ref_dict_cpu = {k: (v.to("cpu") if torch.is_tensor(v) else v)
                    for k, v in gpu_ref.items()}

    print(f"[replay] running s3gen.inference on cpu...", flush=True)
    t0 = time.time()
    result = model.s3gen.inference(
        gpu_tokens.to("cpu"),
        ref_dict=ref_dict_cpu,
    )
    cpu_wav = result[0] if isinstance(result, tuple) else result
    cpu_dt = time.time() - t0
    print(f"[replay] done in {cpu_dt:.1f}s, wav shape {tuple(cpu_wav.shape)}", flush=True)

    # Save CPU dumps
    torch.save(cap["cfm_inputs"], out_dir / "cfm_inputs.pt")
    torch.save(cap["z_init"], out_dir / "z_init.pt")
    torch.save(cap["steps"], out_dir / "steps.pt")
    torch.save(cap["mel"], out_dir / "mel.pt")
    torch.save(cpu_wav, out_dir / "wav.pt")

    # Also write a wav we can listen to.
    audio = cpu_wav.squeeze().cpu().numpy().astype("float32")
    sr = int(model.sr)
    sf.write(out_dir / "audio.wav", audio, sr)
    print(f"[replay] wrote {out_dir / 'audio.wav'}", flush=True)

    # Per-stage diff
    print(f"\n=== diff: GPU bad vs CPU replay ===", flush=True)
    diffs = []
    diffs.append(_diff("z_init", gpu_z, cap["z_init"]))
    for k in ("mu", "mask", "spks", "cond"):
        diffs.append(_diff(f"cfm_input.{k}", gpu_cfm.get(k), cap["cfm_inputs"].get(k)))
    n_steps = min(len(gpu_steps), len(cap["steps"]))
    for i in range(n_steps):
        diffs.append(_diff(f"step_{i+1:02d}", gpu_steps[i], cap["steps"][i]))
    diffs.append(_diff("mel", gpu_mel, cap["mel"]))
    n_w = min(gpu_wav.shape[-1], cpu_wav.shape[-1])
    diffs.append(_diff("wav", gpu_wav[..., :n_w], cpu_wav[..., :n_w]))

    for d in diffs:
        if d is None:
            continue
        if "max_abs" in d:
            rel = d["max_abs"] / (max(d["gpu_norm"], d["cpu_norm"]) + 1e-12)
            print(f"  {d['name']:18s} shape={d['shape']!s:25s} "
                  f"max={d['max_abs']:.3e} mean={d['mean_abs']:.3e} "
                  f"|gpu|={d['gpu_norm']:.2e} |cpu|={d['cpu_norm']:.2e}", flush=True)
        else:
            print(f"  {d['name']:18s} {d}", flush=True)

    with open(out_dir / "diff.json", "w") as f:
        json.dump(diffs, f, indent=2)


if __name__ == "__main__":
    main()
