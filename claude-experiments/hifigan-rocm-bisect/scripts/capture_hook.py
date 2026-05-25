"""Phase A capture hook.

Loaded by the chatterbox child process when HIFIGAN_BISECT_CAPTURE=1 is set.
Wraps HiFTGenerator.inference so that on each call we:

  1. Run the GPU inference normally (whatever the user already configured).
  2. Deep-copy the module to CPU and re-run the same inference there.
  3. Compute spectral metrics (rms, centroid, <300 Hz energy ratio).
  4. If the GPU output looks bad (gpu_rms < BAD_RMS_RATIO * cpu_rms),
     persist a bundle to captures/run_<id>/.

The bundle contents are documented in PLAN_LAYER1.md Phase A.

Activation:
    HIFIGAN_BISECT_CAPTURE=1               # enable
    HIFIGAN_BISECT_OUT=/abs/captures        # override output dir (optional)
    HIFIGAN_BISECT_BAD_RATIO=0.6           # override threshold (optional)
    HIFIGAN_BISECT_FORCE=1                 # save every call, not only bad ones

Inject via either:

    python -c "import scripts.capture_hook as _; _.install()" -m chatterbox ...

or by setting PYTHONSTARTUP or sitecustomize to call install() early. The
function is idempotent.

This file does NOT import chatterbox at module-import time. It only patches
HiFTGenerator after install() is called and chatterbox has been imported.
"""

from __future__ import annotations

import copy
import datetime as _dt
import hashlib
import json
import os
import socket
import subprocess
import sys
import traceback
import uuid
from pathlib import Path

import numpy as np
import torch


# ----------------------------- config -----------------------------

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
_DEFAULT_OUT = _PROJECT_ROOT / "captures"

_INSTALLED = False


def _bool_env(name: str, default: bool = False) -> bool:
    val = os.environ.get(name)
    if val is None:
        return default
    return val.strip().lower() in ("1", "true", "yes", "on")


def _float_env(name: str, default: float) -> float:
    val = os.environ.get(name)
    if val is None:
        return default
    try:
        return float(val)
    except ValueError:
        return default


# --------------------------- spectral ------------------------------


def _spectral_metrics(wav: torch.Tensor, sr: int) -> dict:
    """Welch-PSD spectral metrics.

    Same metrics paper-audiobooks/src/paper_audiobooks/tts/backends/chatterbox.py
    `_is_anomalous` computes: rms, centroid_hz, below_300hz energy ratio,
    300-2k energy ratio. Using the same code shape so detector thresholds
    transfer 1:1.

    `wav` is expected to be a tensor; reshaped to 1-D on CPU.
    """
    from scipy.signal import welch  # imported lazily; not needed for CPU-only torch envs

    arr = wav.detach().to(torch.float32).cpu().reshape(-1).numpy()
    if arr.size == 0:
        return {"rms": 0.0, "centroid_hz": 0.0, "below_300hz": 0.0,
                "300_to_2k": 0.0, "duration_s": 0.0, "n_samples": 0}

    f, p = welch(arr, sr, nperseg=min(2048, arr.size))
    tot = float(np.sum(p)) + 1e-12
    return {
        "rms": float(np.sqrt(np.mean(arr ** 2))),
        "centroid_hz": float(np.sum(f * p) / tot),
        "below_300hz": float(np.sum(p[f < 300]) / tot),
        "300_to_2k": float(np.sum(p[(f >= 300) & (f < 2000)]) / tot),
        "duration_s": float(arr.size / sr),
        "n_samples": int(arr.size),
    }


def _is_anomalous(metrics: dict) -> bool:
    """Real-pipeline detector from paper-audiobooks chatterbox backend:
        bad = (centroid_hz < 700 AND below_300hz > 0.5) OR rms < 0.04
    A chunk-level absolute detector — doesn't need a CPU reference.
    """
    return (metrics["centroid_hz"] < 700 and metrics["below_300hz"] > 0.5) \
        or metrics["rms"] < 0.04


# ------------------------- env / metadata --------------------------


def _miopen_cache_hash() -> str:
    cache = Path.home() / ".cache" / "miopen"
    if not cache.exists():
        return "<absent>"
    h = hashlib.sha256()
    for p in sorted(cache.rglob("*")):
        if not p.is_file():
            continue
        try:
            st = p.stat()
        except OSError:
            continue
        h.update(str(p.relative_to(cache)).encode())
        h.update(str(st.st_size).encode())
        h.update(str(int(st.st_mtime)).encode())
    return h.hexdigest()


def _collect_env_metadata() -> dict:
    env_keys = [
        k
        for k in os.environ
        if k.startswith("MIOPEN_") or k.startswith("HSA_") or k.startswith("ROCR_")
        or k.startswith("HIP_") or k.startswith("PYTORCH_") or k.startswith("ROCM_")
    ]
    info: dict = {
        "timestamp": _dt.datetime.now().isoformat(),
        "hostname": socket.gethostname(),
        "pid": os.getpid(),
        "python": sys.version,
        "torch_version": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "cuda_device_count": torch.cuda.device_count() if torch.cuda.is_available() else 0,
        "cuda_device_name": (
            torch.cuda.get_device_name(0) if torch.cuda.is_available() and torch.cuda.device_count() else None
        ),
        "hip_version": getattr(torch.version, "hip", None),
        "miopen_cache_hash": _miopen_cache_hash(),
        "env": {k: os.environ[k] for k in env_keys},
    }
    # rocm-smi snapshot if present
    try:
        out = subprocess.run(
            ["rocm-smi", "--showproductname", "--showuse", "--showtemp"],
            capture_output=True, text=True, timeout=5,
        )
        info["rocm_smi"] = out.stdout
    except (FileNotFoundError, subprocess.TimeoutExpired, subprocess.SubprocessError):
        pass
    return info


# --------------------------- the hook ------------------------------


def _bundle_dir(root: Path) -> Path:
    stamp = _dt.datetime.now().strftime("%Y%m%dT%H%M%S")
    rid = f"{stamp}_{uuid.uuid4().hex[:8]}"
    out = root / f"run_{rid}"
    out.mkdir(parents=True, exist_ok=False)
    return out


def _save_wav(path: Path, wav: torch.Tensor, sr: int) -> None:
    """Save a 1-D or (1, T) tensor to a 16-bit PCM .wav.

    Avoids depending on torchaudio/scipy beyond what's already present.
    """
    import wave

    arr = wav.detach().to(torch.float32).cpu().reshape(-1).numpy()
    arr = np.clip(arr, -1.0, 1.0)
    pcm = (arr * 32767.0).astype(np.int16)
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sr)
        w.writeframes(pcm.tobytes())


def _extract_init_kwargs(model) -> dict:
    """Best-effort reconstruction of HiFTGenerator constructor kwargs from the
    instantiated module. The class doesn't store them directly, so we infer
    from attributes and submodule shapes. Phase B uses these to reconstruct
    the module without importing chatterbox."""
    # Discover upsample rates / kernels from self.ups
    upsample_rates = []
    upsample_kernel_sizes = []
    for m in model.ups:
        # weight_norm wraps ConvTranspose1d; underlying conv is m itself
        upsample_rates.append(int(m.stride[0]))
        upsample_kernel_sizes.append(int(m.kernel_size[0]))

    # ResBlock kernel sizes / dilations come from the first num_kernels resblocks
    num_kernels = int(model.num_kernels)
    resblock_kernel_sizes = []
    resblock_dilation_sizes = []
    for j in range(num_kernels):
        rb = model.resblocks[j]
        # kernel_size of first conv in convs1 (all share kernel)
        first = rb.convs1[0]
        resblock_kernel_sizes.append(int(first.kernel_size[0]))
        dils = [int(c.dilation[0]) for c in rb.convs1]
        resblock_dilation_sizes.append(dils)

    # Source resblocks
    source_resblock_kernel_sizes = []
    source_resblock_dilation_sizes = []
    for rb in model.source_resblocks:
        first = rb.convs1[0]
        source_resblock_kernel_sizes.append(int(first.kernel_size[0]))
        source_resblock_dilation_sizes.append([int(c.dilation[0]) for c in rb.convs1])

    # in_channels / base_channels come from conv_pre
    conv_pre = model.conv_pre
    in_channels = int(conv_pre.in_channels)
    base_channels = int(conv_pre.out_channels)

    return {
        "in_channels": in_channels,
        "base_channels": base_channels,
        "nb_harmonics": int(model.nb_harmonics),
        "sampling_rate": int(model.sampling_rate),
        "nsf_alpha": float(model.m_source.sine_amp),
        "nsf_sigma": float(model.m_source.noise_std),
        "nsf_voiced_threshold": float(model.m_source.l_sin_gen.voiced_threshold),
        "upsample_rates": upsample_rates,
        "upsample_kernel_sizes": upsample_kernel_sizes,
        "istft_params": {
            "n_fft": int(model.istft_params["n_fft"]),
            "hop_len": int(model.istft_params["hop_len"]),
        },
        "resblock_kernel_sizes": resblock_kernel_sizes,
        "resblock_dilation_sizes": resblock_dilation_sizes,
        "source_resblock_kernel_sizes": source_resblock_kernel_sizes,
        "source_resblock_dilation_sizes": source_resblock_dilation_sizes,
        "lrelu_slope": float(model.lrelu_slope),
        "audio_limit": float(model.audio_limit),
    }


def _layer_names(model) -> list[str]:
    """Probe set matching scripts/bisect_layers.py."""
    names = ["conv_pre"]
    for i in range(len(model.ups)):
        names.append(f"ups.{i}")
    names.append("reflection_pad")
    for i in range(len(model.source_downs)):
        names.append(f"source_downs.{i}")
        names.append(f"source_resblocks.{i}")
    for i in range(len(model.resblocks)):
        names.append(f"resblocks.{i}")
    names.append("conv_post")
    return names


def _register_layer_hooks(model, names, buf, *, sync=True):
    """Register forward hooks that append (name, tensor) to buf.

    sync=True (default): also moves tensor to CPU as fp32 immediately —
    convenient for offline analysis but introduces a per-layer GPU sync.
    Use this on the CPU module or when you want tensors ready to diff.

    sync=False: clone on-device, leave on GPU as original dtype. No sync.
    Use this for GPU intermediates during a *real* call so the hook does
    not perturb MIOpen autotune timing. Caller must `.cpu()` later.
    """
    handles = []
    name_to_mod = dict(model.named_modules())
    for name in names:
        if name not in name_to_mod:
            continue
        mod = name_to_mod[name]

        if sync:
            def make_hook(n):
                def fn(_m, _inp, out):
                    t = out[0] if isinstance(out, tuple) else out
                    buf.append((n, t.detach().to(torch.float32).cpu().clone()))
                return fn
        else:
            def make_hook(n):
                def fn(_m, _inp, out):
                    t = out[0] if isinstance(out, tuple) else out
                    # GPU-resident clone, no sync, original dtype.
                    buf.append((n, t.detach().clone()))
                return fn

        handles.append(mod.register_forward_hook(make_hook(name)))
    return handles


def _diff(a: torch.Tensor, b: torch.Tensor) -> dict:
    if a.shape != b.shape:
        return {"shape_mismatch": True,
                "shape_a": list(a.shape), "shape_b": list(b.shape)}
    d = (a - b).abs()
    rel = d / (a.abs() + 1e-8)
    return {
        "max_abs": float(d.max().item()),
        "mean_abs": float(d.mean().item()),
        "max_rel": float(rel.max().item()),
        "shape": list(a.shape),
    }


def _bisect_against_saved_gpu(out: Path, *, gpu_module, mel_in, cache_source,
                                gpu_layer_buf, original_inference) -> None:
    """The hook captured GPU layer outputs DURING the bad call (gpu_layer_buf).
    Now run CPU once (deterministic) and diff each layer against the saved
    GPU intermediates. CPU is reproducible so this is a faithful comparison
    of the actual bad GPU pass to the correct CPU reference.

    The earlier "run gpu a second time with hooks" approach FAILED because
    the bug is per-call nondeterministic: a second forward in the same
    process produces clean output. We learned that from inspecting hit3 —
    inline_gpu_out.pt rms was 0.061 (clean) while the saved gpu_out.pt rms
    was 0.028 (bad). So the only faithful per-layer trace is one captured
    during the bad call itself.
    """
    cpu_module = copy.deepcopy(gpu_module).to("cpu").eval()
    # Strip inherited GPU hooks before registering fresh CPU ones.
    for _m in cpu_module.modules():
        _m._forward_hooks.clear()
    names = _layer_names(cpu_module)

    cpu_buf: list = []
    cpu_handles = _register_layer_hooks(cpu_module, names, cpu_buf)
    try:
        with torch.inference_mode():
            cpu_wav, _ = original_inference(
                cpu_module, mel_in.detach().cpu(), cache_source.detach().cpu()
            )
    finally:
        for h in cpu_handles:
            h.remove()
    del cpu_module

    cpu_dict = {n: t for n, t in cpu_buf}
    # GPU intermediates were saved on-device, async; sync them to CPU now.
    gpu_dict = {n: t.detach().to(dtype=torch.float32, device="cpu")
                for n, t in gpu_layer_buf}

    per_probe = {}
    ordered = [n for n, _ in gpu_layer_buf if n in cpu_dict]
    for name in ordered:
        per_probe[name] = _diff(cpu_dict[name], gpu_dict[name])

    record = {
        "captured_inline": True,
        "captured_during_bad_pass": True,
        "ordered_probes": ordered,
        "per_probe": per_probe,
    }
    (out / "bisect_diffs.json").write_text(json.dumps(record, indent=2))

    # Persist the per-layer GPU intermediates from the bad pass + the CPU
    # intermediates we just generated. These let us re-analyze the bug after
    # the fact (e.g. find the first divergent op) without re-running.
    torch.save(gpu_dict, out / "gpu_intermediates.pt")
    torch.save({n: t for n, t in cpu_buf}, out / "cpu_intermediates.pt")
    torch.save(cpu_wav.detach().cpu(), out / "inline_cpu_out.pt")


def _snapshot_miopen_state(out: Path) -> dict:
    """Snapshot the on-disk MIOpen state that *might* be relevant to the
    bad call's solver picks. We copy:
      - $MIOPEN_USER_DB_PATH or ~/.config/miopen/ (user find-DBs, *.ufdb.txt)
      - $MIOPEN_CUSTOM_CACHE_DIR or ~/.cache/miopen/ (compiled .co kernels)
    These are the only on-disk MIOpen state that varies between processes.
    Some MIOpen state is in-RAM only (load-time find-DB, in-process per-shape
    solver picks, HIP allocator layout); we cannot snapshot those from here.

    Returns a manifest for metadata.json.
    """
    import shutil

    miopen_user = Path(os.environ.get(
        "MIOPEN_USER_DB_PATH", str(Path.home() / ".config" / "miopen")))
    miopen_cache = Path(os.environ.get(
        "MIOPEN_CUSTOM_CACHE_DIR", str(Path.home() / ".cache" / "miopen")))

    snap = out / "miopen_snapshot"
    snap.mkdir(exist_ok=True)
    manifest = {
        "user_db_src": str(miopen_user),
        "cache_src": str(miopen_cache),
        "user_db_files": [],
        "cache_bytes": 0,
    }
    if miopen_user.exists():
        dst = snap / "user_db"
        shutil.copytree(miopen_user, dst, dirs_exist_ok=True)
        manifest["user_db_files"] = [str(p.relative_to(dst))
                                     for p in dst.rglob("*") if p.is_file()]
    if miopen_cache.exists():
        dst = snap / "cache"
        shutil.copytree(miopen_cache, dst, dirs_exist_ok=True,
                        ignore=shutil.ignore_patterns("*.lock"))
        # Only count, don't list (kernel cache can be huge).
        manifest["cache_bytes"] = sum(
            p.stat().st_size for p in dst.rglob("*") if p.is_file()
        )
    return manifest


def _save_bundle(out: Path, *, model, mel_in, cache_source, gpu_wav, cpu_wav,
                  gpu_metrics, cpu_metrics, sr) -> None:
    torch.save(mel_in.detach().cpu(), out / "mel_in.pt")
    torch.save(cache_source.detach().cpu(), out / "s_cache.pt")
    torch.save({k: v.detach().cpu() for k, v in model.state_dict().items()},
               out / "hifigan_state_dict.pt")
    with (out / "hifigan_init_kwargs.json").open("w") as f:
        json.dump(_extract_init_kwargs(model), f, indent=2)
    torch.save(gpu_wav.detach().cpu(), out / "gpu_out.pt")
    torch.save(cpu_wav.detach().cpu(), out / "cpu_out.pt")
    _save_wav(out / "gpu_out.wav", gpu_wav, sr)
    _save_wav(out / "cpu_out.wav", cpu_wav, sr)
    meta = _collect_env_metadata()
    meta["sampling_rate"] = sr
    meta["gpu_metrics"] = gpu_metrics
    meta["cpu_metrics"] = cpu_metrics
    meta["mel_shape"] = list(mel_in.shape)
    meta["cache_source_shape"] = list(cache_source.shape)
    # Snapshot MIOpen state next to the bundle so we can attempt to
    # reproduce the bad call by restoring this exact on-disk state.
    try:
        meta["miopen_snapshot"] = _snapshot_miopen_state(out)
    except Exception as e:
        meta["miopen_snapshot_error"] = str(e)
    with (out / "metadata.json").open("w") as f:
        json.dump(meta, f, indent=2, default=str)


def install(*, out_dir: str | Path | None = None,
            bad_ratio: float | None = None,
            force_save: bool | None = None) -> None:
    """Patch HiFTGenerator.inference to capture bad-run bundles.

    Idempotent. Reads env vars if args are None.
    """
    global _INSTALLED
    if _INSTALLED:
        return

    out_root = Path(out_dir) if out_dir is not None else Path(
        os.environ.get("HIFIGAN_BISECT_OUT", _DEFAULT_OUT)
    )
    out_root = out_root.resolve()
    out_root.mkdir(parents=True, exist_ok=True)

    threshold = bad_ratio if bad_ratio is not None else _float_env("HIFIGAN_BISECT_BAD_RATIO", 0.6)
    force = force_save if force_save is not None else _bool_env("HIFIGAN_BISECT_FORCE", False)

    try:
        from chatterbox.models.s3gen.hifigan import HiFTGenerator  # type: ignore
    except Exception as e:
        print(f"[capture_hook] could not import HiFTGenerator: {e}", file=sys.stderr)
        return

    original_inference = HiFTGenerator.inference

    def patched_inference(self, speech_feat: torch.Tensor,
                          cache_source: torch.Tensor = torch.zeros(1, 1, 0)):
        # Register persistent forward hooks on this live instance the first
        # time we see it, so EVERY GPU forward records per-layer outputs.
        # The bug is per-call nondeterministic: we cannot re-run a bad call.
        # The only faithful per-layer trace is one captured during the call
        # that turned out bad. Hooks are cheap (~50MB of intermediates per
        # call, freed unless persisted).
        if not getattr(self, "_capture_hooks_attached", False):
            self._capture_hook_buf = []
            # sync=False so the hook does not introduce GPU->CPU sync points
            # during the call. Earlier we discovered hooks WITH sync make the
            # bug stop firing (it has a temporal/scheduling component).
            _register_layer_hooks(self, _layer_names(self), self._capture_hook_buf,
                                   sync=False)
            self._capture_hooks_attached = True

        # Reset buffer at the start of each call.
        self._capture_hook_buf.clear()

        # Snapshot RNG state right before the call so we can re-draw the
        # exact same SineGen randoms during replay. CPU + all GPU devices.
        rng_state = {
            "cpu": torch.get_rng_state().clone(),
            "cuda": [torch.cuda.get_rng_state(i).clone()
                     for i in range(torch.cuda.device_count())]
            if torch.cuda.is_available() else [],
        }

        gpu_wav, gpu_s = original_inference(self, speech_feat, cache_source)

        try:
            sr = int(self.sampling_rate)
            gpu_metrics = _spectral_metrics(gpu_wav, sr)

            # Deep-copy the module to CPU and rerun. Using deepcopy avoids
            # mutating self (it must continue serving subsequent calls on GPU).
            # IMPORTANT: clear inherited forward hooks on the copy so they
            # don't append to self._capture_hook_buf (the closures share that
            # list reference).
            cpu_module = copy.deepcopy(self).to("cpu").eval()
            for _m in cpu_module.modules():
                _m._forward_hooks.clear()
            with torch.inference_mode():
                cpu_wav, _cpu_s = original_inference(
                    cpu_module,
                    speech_feat.detach().cpu(),
                    cache_source.detach().cpu(),
                )
            cpu_metrics = _spectral_metrics(cpu_wav, sr)
            del cpu_module

            ratio = (gpu_metrics["rms"] / cpu_metrics["rms"]) if cpu_metrics["rms"] > 0 else 1.0
            gpu_anom = _is_anomalous(gpu_metrics)
            cpu_anom = _is_anomalous(cpu_metrics)
            # Bug-of-interest: GPU muffled, CPU clean — pure HiFiGAN divergence.
            divergence = gpu_anom and not cpu_anom
            # Coarse fallback: GPU much quieter than CPU even if absolute detector misses.
            quiet = ratio < threshold
            bad = divergence or quiet

            print(
                f"[capture_hook] mel={tuple(speech_feat.shape)} "
                f"gpu[rms={gpu_metrics['rms']:.4f} cent={gpu_metrics['centroid_hz']:.0f}Hz "
                f"<300={gpu_metrics['below_300hz']:.2f}] "
                f"cpu[rms={cpu_metrics['rms']:.4f} cent={cpu_metrics['centroid_hz']:.0f}Hz "
                f"<300={cpu_metrics['below_300hz']:.2f}] "
                f"ratio={ratio:.3f} "
                f"{'DIVERGENCE' if divergence else ('QUIET' if quiet else 'ok')}"
                f"{' (gpu_anom)' if gpu_anom else ''}"
                f"{' (cpu_anom!)' if cpu_anom else ''}",
                file=sys.stderr,
            )

            if bad or force:
                out = _bundle_dir(out_root)
                _save_bundle(
                    out,
                    model=self,
                    mel_in=speech_feat,
                    cache_source=cache_source,
                    gpu_wav=gpu_wav,
                    cpu_wav=cpu_wav,
                    gpu_metrics=gpu_metrics,
                    cpu_metrics=cpu_metrics,
                    sr=sr,
                )
                # Also persist RNG state captured BEFORE the call, so we
                # can reproduce SineGen randomness.
                torch.save(rng_state, out / "rng_state.pt")
                print(f"[capture_hook] saved bundle: {out}", file=sys.stderr)

                # Bisect against the GPU intermediates we already captured
                # during the bad call. Diff each layer against a CPU re-run
                # (CPU is deterministic so this is faithful).
                try:
                    print("[capture_hook] running post-hoc bisect on captured GPU intermediates ...",
                          file=sys.stderr)
                    _bisect_against_saved_gpu(
                        out,
                        gpu_module=self,
                        mel_in=speech_feat,
                        cache_source=cache_source,
                        gpu_layer_buf=list(self._capture_hook_buf),
                        original_inference=original_inference,
                    )
                    print(f"[capture_hook] bisect_diffs.json saved to {out}", file=sys.stderr)
                except Exception:
                    print("[capture_hook] bisect failed:", file=sys.stderr)
                    traceback.print_exc()

        except Exception:
            # Capture must NEVER break the host pipeline.
            print("[capture_hook] capture failed (continuing):", file=sys.stderr)
            traceback.print_exc()

        return gpu_wav, gpu_s

    HiFTGenerator.inference = patched_inference  # type: ignore[assignment]
    _INSTALLED = True
    print(
        f"[capture_hook] installed; out={out_root} threshold={threshold} force={force}",
        file=sys.stderr,
    )


# Auto-install if this module is imported and the env var is set.
if _bool_env("HIFIGAN_BISECT_CAPTURE", False):
    try:
        install()
    except Exception:
        traceback.print_exc()
