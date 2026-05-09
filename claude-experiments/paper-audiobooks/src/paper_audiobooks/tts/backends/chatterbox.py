"""Chatterbox TTS backend (Resemble AI).

Beats ElevenLabs in blind tests at 63.75% preference. Voice-cloning + emotion
control. Has its own venv to avoid version conflicts.

Setup:
    uv venv ~/.cache/paper-audiobooks/venvs/chatterbox --python 3.11
    VIRTUAL_ENV=~/.cache/paper-audiobooks/venvs/chatterbox uv pip install chatterbox-tts soundfile

The child process is long-lived: model loads once, then we feed it one JSON
request per line on stdin and read one JSON ack per line on stdout. This avoids
paying the ~4s model load per chunk.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
import threading
from pathlib import Path

import numpy as np
import soundfile as sf

from .base import Backend, BackendInfo

VENV_DIR = Path(os.path.expanduser("~/.cache/paper-audiobooks/venvs/chatterbox"))
VENV_PYTHON = VENV_DIR / "bin" / "python"

# Opt-in anomaly detection: when CHATTERBOX_HALT_ON_ANOMALY=1 the backend
# scores every generated chunk and raises ChatterboxAnomalyHalt on the first
# muffled-output match, after dumping the bad wav + the text that produced it
# to CHATTERBOX_HALT_DIR (default: bisect_results/halt). Used to capture the
# first in-the-wild anomaly during a real pipeline run.
HALT_ENV = "CHATTERBOX_HALT_ON_ANOMALY"
HALT_DIR_ENV = "CHATTERBOX_HALT_DIR"
DEFAULT_HALT_DIR = Path("bisect_results") / "halt"


class ChatterboxAnomalyHalt(RuntimeError):
    """Raised by the chatterbox backend when CHATTERBOX_HALT_ON_ANOMALY=1
    and a generated chunk matches the muffled-output detector. The message
    includes paths to the dumped wav and the input text."""


def _is_anomalous(audio: np.ndarray, sr: int) -> tuple[bool, dict]:
    """Same detector rule used throughout CHATTERBOX_DEBUG.md: a chunk is
    flagged when (centroid<700Hz AND <300Hz energy>0.5) OR rms<0.04."""
    from scipy.signal import welch
    if audio.ndim > 1:
        audio = audio.mean(axis=-1)
    f, p = welch(audio, sr, nperseg=2048)
    tot = float(np.sum(p)) + 1e-12
    s = {
        "centroid_hz": float(np.sum(f * p) / tot),
        "below_300hz": float(np.sum(p[f < 300]) / tot),
        "300_to_2k": float(np.sum(p[(f >= 300) & (f < 2000)]) / tot),
        "rms": float(np.sqrt(np.mean(audio ** 2))),
        "duration_s": float(len(audio) / sr),
    }
    bad = (s["centroid_hz"] < 700 and s["below_300hz"] > 0.5) or s["rms"] < 0.04
    return bad, s


def _die_with_parent() -> None:
    """preexec_fn: ask the kernel to SIGKILL this process if the parent dies.
    Linux-only (PR_SET_PDEATHSIG = 1, SIGKILL = 9). Without this, killing the
    pipeline parent leaves orphan chatterbox children spinning on the GPU."""
    try:
        import ctypes, signal
        libc = ctypes.CDLL("libc.so.6", use_errno=True)
        PR_SET_PDEATHSIG = 1
        libc.prctl(PR_SET_PDEATHSIG, signal.SIGKILL, 0, 0, 0)
    except Exception:
        pass  # best-effort; don't block child startup

# Per-chunk timeout once the model is loaded. ~250-char chunk on GPU is ~10-30s
# typical; we allow 5min as a hard ceiling so a real GPU hang doesn't block forever.
CHUNK_TIMEOUT_SECONDS = 300

_CHILD_SCRIPT = r"""
import json, os, sys, time, traceback, types

# Reduce HIP/CUDA allocator fragmentation — same fix as extract_worker.py.
os.environ.setdefault("PYTORCH_HIP_ALLOC_CONF", "expandable_segments:True")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

# Save the real stdout for protocol acks, then redirect fd 1 to stderr so any
# library that prints to stdout during import/model-load (chatterbox does:
# "loaded PerthNet (Implicit) at step 250,000") doesn't corrupt our JSON channel.
_protocol_stdout = os.fdopen(os.dup(1), "w", buffering=1)
os.dup2(2, 1)
sys.stdout = os.fdopen(1, "w", buffering=1)

print(f"[child] booting at {time.strftime('%H:%M:%S')}", file=sys.stderr, flush=True)
import numpy as np
import soundfile as sf
import torch

print(f"[child] torch={torch.__version__} cuda_avail={torch.cuda.is_available()}", file=sys.stderr, flush=True)

from chatterbox.tts import ChatterboxTTS
print("[child] loading model...", file=sys.stderr, flush=True)
t0 = time.time()
device = "cuda" if torch.cuda.is_available() else "cpu"
model = ChatterboxTTS.from_pretrained(device=device)
print(f"[child] model loaded in {time.time()-t0:.1f}s on {device}", file=sys.stderr, flush=True)

# ---- Optional s3gen capture harness (CHATTERBOX_HALT_ON_ANOMALY=1) -----
# When enabled, every generate() call records intermediates of the s3gen
# flow-matching ODE in a per-chunk in-memory buffer. The parent never reads
# these unless an anomaly fires; on anomaly it sends a {dump_intermediates}
# follow-up and we materialize the most recent buffer to disk. This avoids
# disk thrash on the thousands of clean chunks per book.
_capture_enabled = os.environ.get("CHATTERBOX_HALT_ON_ANOMALY") == "1"
_cap = {"cfm_inputs": None, "z_init": None, "steps": [], "mel": None,
        "wav": None, "tokens": None, "ref_dict": None, "text": None}

def _reset_cap():
    _cap["cfm_inputs"] = None
    _cap["z_init"] = None
    _cap["steps"] = []
    _cap["mel"] = None
    _cap["wav"] = None
    _cap["tokens"] = None
    _cap["ref_dict"] = None
    _cap["text"] = None

def _materialize_cap(prefix):
    # Best-effort: write every captured artifact independently. If one
    # artifact is malformed (e.g. a future bug stores the wrong type), log
    # and keep going so the rest of the dump still lands on disk.
    import os as _os
    _os.makedirs(prefix, exist_ok=True)
    def _to_cpu(v):
        if torch.is_tensor(v):
            return v.detach().cpu()
        return v
    def _try_save(name, fn):
        try:
            fn()
        except Exception as exc:
            print(f"[child] dump {name}: {exc}", file=sys.stderr, flush=True)
    if _cap["cfm_inputs"] is not None:
        _try_save("cfm_inputs", lambda: torch.save(
            {k: _to_cpu(v) for k, v in _cap["cfm_inputs"].items()},
            _os.path.join(prefix, "cfm_inputs.pt"),
        ))
    if _cap["z_init"] is not None:
        _try_save("z_init", lambda: torch.save(
            _cap["z_init"].detach().cpu(),
            _os.path.join(prefix, "z_init.pt"),
        ))
    if _cap["steps"]:
        _try_save("steps", lambda: torch.save(
            [s.detach().cpu() for s in _cap["steps"]],
            _os.path.join(prefix, "steps.pt"),
        ))
    if _cap["mel"] is not None:
        _try_save("mel", lambda: torch.save(
            _cap["mel"].detach().cpu(),
            _os.path.join(prefix, "mel.pt"),
        ))
    if _cap["wav"] is not None:
        _try_save("wav", lambda: torch.save(
            _cap["wav"].detach().cpu(),
            _os.path.join(prefix, "wav.pt"),
        ))
    if _cap["tokens"] is not None:
        _try_save("tokens", lambda: torch.save(
            _cap["tokens"].detach().cpu(),
            _os.path.join(prefix, "tokens.pt"),
        ))
    if _cap["ref_dict"] is not None:
        _try_save("ref_dict", lambda: torch.save(
            {k: _to_cpu(v) for k, v in _cap["ref_dict"].items()},
            _os.path.join(prefix, "ref_dict.pt"),
        ))
    if _cap["text"] is not None:
        _try_save("text", lambda: open(
            _os.path.join(prefix, "text.txt"), "w", encoding="utf-8",
        ).write(_cap["text"]))

def _install_capture(model):
    s3gen = model.s3gen
    cfm = s3gen.flow.decoder  # CausalConditionalCFM

    @torch.inference_mode()
    def patched_cfm_forward(self_cfm, mu, mask, n_timesteps, temperature=1.0,
                            spks=None, cond=None, noised_mels=None, meanflow=False):
        _cap["cfm_inputs"] = {
            "mu": mu, "mask": mask, "spks": spks, "cond": cond,
            "n_timesteps": n_timesteps, "noised_mels": noised_mels,
            "meanflow": meanflow, "temperature": temperature,
            "t_scheduler": self_cfm.t_scheduler,
            "inference_cfg_rate": self_cfm.inference_cfg_rate,
        }
        z = torch.randn_like(mu)
        if noised_mels is not None:
            prompt_len = mu.size(2) - noised_mels.size(2)
            z = z.clone()
            z[..., prompt_len:] = noised_mels
        _cap["z_init"] = z
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
            _cap["steps"].append(x.clone())
        return x.to(in_dtype)
    cfm.solve_euler = types.MethodType(patched_solve_euler, cfm)

    orig_flow_inf = s3gen.flow_inference
    @torch.inference_mode()
    def patched_flow_inference(self_s3, speech_tokens, **kwargs):
        mel = orig_flow_inf(speech_tokens, **kwargs)
        _cap["mel"] = mel
        return mel
    s3gen.flow_inference = types.MethodType(patched_flow_inference, s3gen)

    orig_s3_inf = s3gen.inference
    @torch.inference_mode()
    def patched_s3_inference(self_s3, speech_tokens, ref_wav=None, ref_sr=None,
                             ref_dict=None, drop_invalid_tokens=True,
                             n_cfm_timesteps=None, speech_token_lens=None):
        _cap["tokens"] = speech_tokens.clone() if torch.is_tensor(speech_tokens) else speech_tokens
        if ref_dict is None and ref_wav is not None:
            ref_dict = self_s3.embed_ref(ref_wav, ref_sr)
            ref_wav = None
            ref_sr = None
        _cap["ref_dict"] = ref_dict
        result = orig_s3_inf(speech_tokens, ref_wav=ref_wav, ref_sr=ref_sr,
                             ref_dict=ref_dict,
                             drop_invalid_tokens=drop_invalid_tokens,
                             n_cfm_timesteps=n_cfm_timesteps,
                             speech_token_lens=speech_token_lens)
        # s3gen.inference returns (output_wavs, output_sources) — only the
        # first is the audio tensor we want to capture.
        _cap["wav"] = result[0] if isinstance(result, tuple) else result
        return result
    s3gen.inference = types.MethodType(patched_s3_inference, s3gen)

if _capture_enabled:
    print("[child] anomaly-capture harness enabled", file=sys.stderr, flush=True)
    _install_capture(model)

# ---- Optional hifigan-on-CPU patch (CHATTERBOX_HIFIGAN_CPU=1) ----------
# The s3gen vocoder (HiFiGAN, ~1s/chunk on cpu) was identified as the source
# of the muffled-output bug on AMD ROCm. Bisect proved CPU and GPU produce
# identical mels (max abs diff ~5e-4) but the GPU vocoder intermittently
# yields ~half-amplitude / spectrally-shifted audio. Forcing only this stage
# onto CPU keeps the rest of the pipeline at full GPU speed.
if os.environ.get("CHATTERBOX_HIFIGAN_CPU") == "1":
    print("[child] forcing hifigan onto cpu", file=sys.stderr, flush=True)
    _s3 = model.s3gen
    _s3.mel2wav = _s3.mel2wav.to("cpu")
    if hasattr(_s3, "trim_fade") and torch.is_tensor(_s3.trim_fade):
        _s3.trim_fade = _s3.trim_fade.to("cpu")

    _orig_inference = _s3.inference

    @torch.inference_mode()
    def _patched_inference(self_s3, speech_tokens, ref_wav=None, ref_sr=None,
                           ref_dict=None, drop_invalid_tokens=True,
                           n_cfm_timesteps=None, speech_token_lens=None):
        # Run the flow stage normally — it stays on the configured device
        # (cuda) since we only moved mel2wav.
        output_mels = self_s3.flow_inference(
            speech_tokens,
            speech_token_lens=speech_token_lens,
            ref_wav=ref_wav, ref_sr=ref_sr, ref_dict=ref_dict,
            n_cfm_timesteps=n_cfm_timesteps, finalize=True,
        )
        # Move mel to cpu and run hifigan there. dtype cast happens after
        # the move so it uses cpu hifigan's expected dtype, matching the
        # original code's `output_mels.to(dtype=self.dtype)` semantics.
        output_mels = output_mels.to(device="cpu", dtype=torch.float32)
        output_wavs, output_sources = self_s3.mel2wav.inference(
            speech_feat=output_mels,
            cache_source=torch.zeros(1, 1, 0, device="cpu", dtype=torch.float32),
        )
        output_wavs[:, :len(self_s3.trim_fade)] *= self_s3.trim_fade
        return output_wavs, output_sources
    _s3.inference = types.MethodType(_patched_inference, _s3)

# Signal ready so the parent can start sending requests.
_protocol_stdout.write(json.dumps({"ready": True}) + "\n")
_protocol_stdout.flush()

for line in sys.stdin:
    line = line.strip()
    if not line:
        continue
    try:
        req = json.loads(line)
        if req.get("shutdown"):
            print("[child] shutdown requested", file=sys.stderr, flush=True)
            break
        # Out-of-band: parent saw an anomaly and asks for the most recent
        # capture to be flushed to disk.
        if req.get("dump_intermediates"):
            target = req["dump_intermediates"]
            try:
                _materialize_cap(target)
                print(f"[child] dumped capture to {target}", file=sys.stderr, flush=True)
                _protocol_stdout.write(json.dumps({"dumped": True, "path": target}) + "\n")
            except Exception as exc:
                traceback.print_exc()
                _protocol_stdout.write(json.dumps({"error": f"dump failed: {exc}"}) + "\n")
            _protocol_stdout.flush()
            continue

        text = req["text"]
        out_path = req["out_path"]
        audio_prompt = req.get("voice_ref_path") or None
        exaggeration = float(req.get("exaggeration", 0.5))
        cfg_weight = float(req.get("cfg_weight", 0.5))

        if _capture_enabled:
            _reset_cap()
            _cap["text"] = text

        print(f"[child] generate(): {len(text)}chars voice_ref={'yes' if audio_prompt else 'no'}", file=sys.stderr, flush=True)
        t0 = time.time()
        wav = model.generate(
            text,
            audio_prompt_path=audio_prompt,
            exaggeration=exaggeration,
            cfg_weight=cfg_weight,
        )
        gen_dt = time.time() - t0
        audio = wav.squeeze().cpu().numpy().astype("float32")
        sr = int(model.sr)
        sf.write(out_path, audio, sr)
        print(f"[child] done in {gen_dt:.1f}s, {audio.shape[0]/sr:.1f}s audio", file=sys.stderr, flush=True)
        _protocol_stdout.write(json.dumps({"sample_rate": sr, "samples": int(audio.shape[0])}) + "\n")
        _protocol_stdout.flush()
    except Exception as exc:
        traceback.print_exc()
        _protocol_stdout.write(json.dumps({"error": str(exc)}) + "\n")
        _protocol_stdout.flush()
"""


class _ChildHandle:
    """Wraps the long-lived chatterbox child process."""

    def __init__(self) -> None:
        self.proc = subprocess.Popen(
            [str(VENV_PYTHON), "-c", _CHILD_SCRIPT],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=sys.stderr,
            text=True,
            bufsize=1,  # line-buffered
            preexec_fn=_die_with_parent,  # kill child if parent dies (no orphans)
        )
        # Wait for the {"ready": true} ack so we don't send chunks before the model is up.
        ready_line = self.proc.stdout.readline()
        if not ready_line:
            raise RuntimeError("chatterbox child exited before becoming ready")
        try:
            ack = json.loads(ready_line)
        except json.JSONDecodeError as exc:
            raise RuntimeError(f"chatterbox child sent bad ready line: {ready_line!r}") from exc
        if not ack.get("ready"):
            raise RuntimeError(f"chatterbox child failed to start: {ack!r}")
        self._lock = threading.Lock()

    def dump_intermediates(self, target_dir: Path) -> None:
        """Send a sideband request asking the child to flush the most recent
        capture buffer to target_dir. The child only has anything to flush
        when CHATTERBOX_HALT_ON_ANOMALY=1 is set in its environment (which
        the parent inherits — same env). Must be called immediately after
        the synth() that produced the bad audio, before any other synth(),
        because the buffer is reset on every generate()."""
        req = {"dump_intermediates": str(target_dir)}
        with self._lock:
            if self.proc.poll() is not None:
                raise RuntimeError(f"chatterbox child died (rc={self.proc.returncode})")
            self.proc.stdin.write(json.dumps(req) + "\n")
            self.proc.stdin.flush()
            ack_line = self.proc.stdout.readline()
            if not ack_line:
                raise RuntimeError("chatterbox child closed stdout during dump")
            ack = json.loads(ack_line)
            if "error" in ack:
                raise RuntimeError(f"chatterbox dump error: {ack['error']}")

    def synth(self, *, text: str, voice_ref_path: str | None, out_path: Path) -> None:
        req = {"text": text, "out_path": str(out_path)}
        if voice_ref_path:
            req["voice_ref_path"] = voice_ref_path
        with self._lock:
            if self.proc.poll() is not None:
                raise RuntimeError(f"chatterbox child died (rc={self.proc.returncode})")
            self.proc.stdin.write(json.dumps(req) + "\n")
            self.proc.stdin.flush()

            # Wait for the ack with a timeout. We can't easily timeout readline(),
            # so use a watchdog thread that kills the child if it overruns.
            timer = threading.Timer(CHUNK_TIMEOUT_SECONDS, self._kill_on_timeout)
            timer.start()
            try:
                ack_line = self.proc.stdout.readline()
            finally:
                timer.cancel()

            if not ack_line:
                raise RuntimeError("chatterbox child closed stdout unexpectedly")
            ack = json.loads(ack_line)
            if "error" in ack:
                raise RuntimeError(f"chatterbox child error: {ack['error']}")

    def _kill_on_timeout(self) -> None:
        print(
            f"[chatterbox] TIMEOUT after {CHUNK_TIMEOUT_SECONDS}s — killing child",
            file=sys.stderr,
            flush=True,
        )
        try:
            self.proc.kill()
        except Exception:
            pass

    def close(self) -> None:
        if self.proc.poll() is None:
            try:
                self.proc.stdin.write(json.dumps({"shutdown": True}) + "\n")
                self.proc.stdin.flush()
                self.proc.stdin.close()
            except Exception:
                pass
            try:
                self.proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                self.proc.kill()
                self.proc.wait()


class ChatterboxBackend(Backend):
    info = BackendInfo(
        name="chatterbox",
        default_voice="default",
        max_chunk_chars=250,
        description="Chatterbox (Resemble AI) — emotion control, voice clone, beats ElevenLabs in blind tests.",
    )

    def __init__(self) -> None:
        if not VENV_PYTHON.exists():
            raise RuntimeError(
                f"Chatterbox venv not found at {VENV_DIR}. See chatterbox.py docstring."
            )
        self._child: _ChildHandle | None = None

    def _ensure_child(self) -> _ChildHandle:
        if self._child is None or self._child.proc.poll() is not None:
            self._child = _ChildHandle()
        return self._child

    def synthesize_chunk(self, text: str, *, voice: str) -> np.ndarray:
        from .. import SAMPLE_RATE
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            tmp_path = Path(tmp.name)
        voice_ref = None
        if voice and voice != "default" and Path(voice).expanduser().is_file():
            voice_ref = str(Path(voice).expanduser())
        try:
            child = self._ensure_child()
            child.synth(text=text, voice_ref_path=voice_ref, out_path=tmp_path)
            if tmp_path.stat().st_size == 0:
                raise RuntimeError("chatterbox produced an empty wav")
            audio, sr = sf.read(tmp_path, dtype="float32")
        finally:
            tmp_path.unlink(missing_ok=True)

        if os.environ.get(HALT_ENV) == "1":
            bad, stats = _is_anomalous(audio, sr)
            if bad:
                halt_dir = Path(os.environ.get(HALT_DIR_ENV) or DEFAULT_HALT_DIR)
                halt_dir.mkdir(parents=True, exist_ok=True)
                # Use a timestamped slug so multiple halts don't overwrite.
                import time as _time
                slug = _time.strftime("%Y%m%d-%H%M%S")
                wav_path = halt_dir / f"halt-{slug}.wav"
                txt_path = halt_dir / f"halt-{slug}.txt"
                stats_path = halt_dir / f"halt-{slug}.json"
                dumps_dir = halt_dir / f"halt-{slug}-dumps"
                sf.write(wav_path, audio, sr)
                txt_path.write_text(text, encoding="utf-8")
                stats_path.write_text(json.dumps(
                    {**stats, "voice_ref": voice_ref, "n_chars": len(text)},
                    indent=2,
                ), encoding="utf-8")
                # Ask the child to flush its most recent s3gen capture buffer
                # before any subsequent generate() overwrites it. Failure to
                # dump (e.g. capture harness disabled) is non-fatal — we
                # still raise with the audio artifacts.
                dump_note = ""
                try:
                    child.dump_intermediates(dumps_dir)
                    dump_note = f"\n  dumps: {dumps_dir}"
                except Exception as exc:
                    dump_note = f"\n  dumps: <failed: {exc}>"
                raise ChatterboxAnomalyHalt(
                    f"chatterbox anomaly detected: {stats}\n"
                    f"  wav:   {wav_path}\n"
                    f"  text:  {txt_path}\n"
                    f"  stats: {stats_path}{dump_note}\n"
                    f"  chars: {len(text)}"
                )

        if sr != SAMPLE_RATE:
            from .higgs import _resample
            audio = _resample(audio, sr, SAMPLE_RATE)
        return audio.astype(np.float32)

    def __del__(self) -> None:
        try:
            if getattr(self, "_child", None) is not None:
                self._child.close()
        except Exception:
            pass
