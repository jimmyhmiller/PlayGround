"""Replay just the s3gen flow-matching ODE on CPU using captured GPU
inputs. No tokens needed — we inject (mu, mask, spks, cond, z_init)
directly into solve_euler. Then compare every step + the final mel.

Use after capture from halt-on-anomaly when tokens.pt / ref_dict.pt
weren't saved (e.g. the hifigan-cpu patch overrode the inference patch).

Usage:
    ~/.cache/paper-audiobooks/venvs/chatterbox/bin/python \\
        scripts/replay_ode_only.py bisect_results/halt/halt-<slug>-dumps
"""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["HIP_VISIBLE_DEVICES"] = ""

import numpy as np
import soundfile as sf
import torch

assert not torch.cuda.is_available()

from chatterbox.tts import ChatterboxTTS


def main():
    if len(sys.argv) != 2:
        print(f"usage: {sys.argv[0]} <halt-dumps-dir>", file=sys.stderr)
        sys.exit(2)

    dumps = Path(sys.argv[1])
    out_dir = dumps.parent / (dumps.name + "-cpu-ode-replay")
    out_dir.mkdir(exist_ok=True)
    print(f"[ode-replay] dumps: {dumps}", flush=True)
    print(f"[ode-replay] out:   {out_dir}", flush=True)

    gpu_z = torch.load(dumps / "z_init.pt", map_location="cpu")
    gpu_cfm = torch.load(dumps / "cfm_inputs.pt", map_location="cpu")
    gpu_steps = torch.load(dumps / "steps.pt", map_location="cpu")
    gpu_mel = torch.load(dumps / "mel.pt", map_location="cpu")

    print(f"[ode-replay] gpu z_init: {tuple(gpu_z.shape)}", flush=True)
    print(f"[ode-replay] gpu mu:     {tuple(gpu_cfm['mu'].shape)}", flush=True)
    print(f"[ode-replay] gpu mel:    {tuple(gpu_mel.shape)}", flush=True)

    print(f"[ode-replay] loading model on cpu...", flush=True)
    t0 = time.time()
    model = ChatterboxTTS.from_pretrained(device="cpu")
    print(f"[ode-replay] loaded in {time.time()-t0:.1f}s", flush=True)

    cfm = model.s3gen.flow.decoder

    # Run solve_euler directly with GPU's tensors. Use the un-patched
    # original implementation (no monkeypatching needed — we don't need
    # to capture, we'll just compare cfm output to gpu_mel).
    # But we DO need step-by-step output for the diff. Reuse the solve
    # logic with a manual loop.
    from chatterbox.models.s3gen.flow_matching import cast_all

    n_timesteps = gpu_cfm["n_timesteps"]
    t_scheduler = gpu_cfm["t_scheduler"]
    inference_cfg_rate = gpu_cfm["inference_cfg_rate"]
    meanflow = gpu_cfm.get("meanflow", False)

    mu = gpu_cfm["mu"].to("cpu")
    mask = gpu_cfm["mask"].to("cpu")
    spks = gpu_cfm["spks"].to("cpu")
    cond = gpu_cfm["cond"].to("cpu")
    z = gpu_z.to("cpu")

    t_span = torch.linspace(0, 1, n_timesteps + 1, device="cpu", dtype=mu.dtype)
    if (not meanflow) and (t_scheduler == "cosine"):
        t_span = 1 - torch.cos(t_span * 0.5 * torch.pi)

    print(f"[ode-replay] running solve_euler on cpu...", flush=True)
    t0 = time.time()

    in_dtype = z.dtype
    x, t_span_c, mu_c, mask_c, spks_c, cond_c = cast_all(
        z, t_span, mu, mask, spks, cond, dtype=cfm.estimator.dtype
    )

    B, T = mu_c.size(0), x.size(2)
    x_in    = torch.zeros([2*B, 80, T], device=x.device, dtype=x.dtype)
    mask_in = torch.zeros([2*B,  1, T], device=x.device, dtype=x.dtype)
    mu_in   = torch.zeros([2*B, 80, T], device=x.device, dtype=x.dtype)
    t_in    = torch.zeros([2*B       ], device=x.device, dtype=x.dtype)
    spks_in = torch.zeros([2*B, 80   ], device=x.device, dtype=x.dtype)
    cond_in = torch.zeros([2*B, 80, T], device=x.device, dtype=x.dtype)
    r_in    = torch.zeros([2*B       ], device=x.device, dtype=x.dtype)

    cpu_steps = []
    with torch.inference_mode():
        for t, r in zip(t_span_c[:-1], t_span_c[1:]):
            t = t.unsqueeze(dim=0); r = r.unsqueeze(dim=0)
            x_in[:B] = x_in[B:] = x
            mask_in[:B] = mask_in[B:] = mask_c
            mu_in[:B] = mu_c
            t_in[:B] = t_in[B:] = t
            spks_in[:B] = spks_c
            cond_in[:B] = cond_c
            r_in[:B] = r_in[B:] = r
            dxdt = cfm.estimator.forward(
                x=x_in, mask=mask_in, mu=mu_in, t=t_in, spks=spks_in, cond=cond_in,
                r=r_in if meanflow else None,
            )
            dxdt, cfg_dxdt = torch.split(dxdt, [B, B], dim=0)
            dxdt = ((1.0 + inference_cfg_rate) * dxdt - inference_cfg_rate * cfg_dxdt)
            dt = r - t
            x = x + dt * dxdt
            cpu_steps.append(x.clone())

    cpu_x = x.to(in_dtype)
    print(f"[ode-replay] solved in {time.time()-t0:.1f}s, {len(cpu_steps)} steps", flush=True)

    # The CPU final ODE output is the unpadded mel. The original flow.inference
    # crops it: feat = feat[:, :, mel_len1:]. We don't know mel_len1 here,
    # but we know the gpu_mel shape — figure out the prefix to strip by
    # matching shapes.
    #
    # gpu_mel shape: (1, 80, M_out). cpu_x shape: (1, 80, T_full). Then M_out
    # = T_full - mel_len1.
    if cpu_x.shape[-1] != gpu_mel.shape[-1]:
        prefix = cpu_x.shape[-1] - gpu_mel.shape[-1]
        cpu_mel = cpu_x[:, :, prefix:]
        print(f"[ode-replay] stripped {prefix}-frame prompt prefix from cpu_x")
    else:
        cpu_mel = cpu_x

    print(f"\n=== diff: GPU vs CPU ODE replay ===")
    def diff(name, g, c):
        if g.shape != c.shape:
            print(f"  {name:18s} shape mismatch: gpu {tuple(g.shape)} cpu {tuple(c.shape)}")
            return
        d = (g.float() - c.float()).abs()
        print(f"  {name:18s} shape={tuple(g.shape)!s:25s} max={d.max():.3e} mean={d.mean():.3e} "
              f"|gpu|={g.float().norm():.2e} |cpu|={c.float().norm():.2e}")

    n_steps = min(len(gpu_steps), len(cpu_steps))
    for i in range(n_steps):
        diff(f"step_{i+1:02d}", gpu_steps[i], cpu_steps[i])
    diff("mel(unpadded)", gpu_mel, cpu_mel)

    # Save cpu intermediates.
    torch.save(cpu_steps, out_dir / "steps.pt")
    torch.save(cpu_mel, out_dir / "mel.pt")
    torch.save(cpu_x, out_dir / "ode_full.pt")

    # Run hifigan on the CPU mel so we can listen.
    print(f"\n[ode-replay] running cpu hifigan on cpu mel...", flush=True)
    t0 = time.time()
    out_wavs, _ = model.s3gen.mel2wav.inference(
        speech_feat=cpu_mel.to(model.s3gen.dtype),
        cache_source=torch.zeros(1,1,0),
    )
    audio = out_wavs.squeeze().cpu().numpy().astype("float32")
    sf.write(out_dir / "audio.wav", audio, int(model.sr))
    print(f"  done in {time.time()-t0:.1f}s, rms={np.sqrt(np.mean(audio**2)):.4f}", flush=True)
    print(f"  wrote {out_dir / 'audio.wav'}")


if __name__ == "__main__":
    main()
