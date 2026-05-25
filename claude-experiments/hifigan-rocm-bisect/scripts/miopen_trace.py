"""Trace MIOpen solver picks for one BAD and one ok call.

Runs s3gen.inference twice in a single process:
  - call 1: trial06 tokens (historically clean)
  - call 2: trial07 tokens (historically BAD)
with MIOPEN_LOG_LEVEL=6 forcing every solver evaluation to be logged.

We capture stderr of each call separately, then diff solver names.

Run as:
    MIOPEN_LOG_LEVEL=6 MIOPEN_ENABLE_LOGGING=1 MIOPEN_ENABLE_LOGGING_CMD=1 \\
        python3 scripts/miopen_trace.py
"""
from __future__ import annotations
import os, sys, time
from pathlib import Path
import numpy as np


MIOPEN_DEFAULT_LOG_ENV = {
    "MIOPEN_LOG_LEVEL": "6",
    "MIOPEN_ENABLE_LOGGING": "1",
    "MIOPEN_ENABLE_LOGGING_CMD": "1",
}


def main():
    for k, v in MIOPEN_DEFAULT_LOG_ENV.items():
        os.environ.setdefault(k, v)
    os.environ.setdefault("PYTORCH_HIP_ALLOC_CONF", "expandable_segments:True")

    here = Path(__file__).resolve().parents[1]
    cb_src = here.parent / "chatterbox-rewrite" / "chatterbox" / "src"
    sys.path.insert(0, str(cb_src))

    out_dir = here / "miopen_trace_logs"
    out_dir.mkdir(exist_ok=True)

    import torch
    from chatterbox.tts import ChatterboxTTS  # type: ignore

    print("[trace] loading model ...", flush=True, file=sys.stderr)
    t0 = time.time()
    model = ChatterboxTTS.from_pretrained(device="cuda")
    print(f"[trace] loaded in {time.time()-t0:.1f}s", flush=True, file=sys.stderr)
    model.prepare_conditionals("/home/jimmyhmiller/.config/paper-audiobooks/default-voice.wav")

    tokens_dir = here.parent / "paper-audiobooks" / "bisect_results"
    if not tokens_dir.exists():
        # alternate path resolution
        tokens_dir = Path("../paper-audiobooks/bisect_results").resolve()

    cases = [
        ("clean", "phase1_trial06_tokens.npy"),
        ("bad",   "phase1_trial07_tokens.npy"),
    ]

    from scipy.signal import welch
    sr = int(model.sr)

    # Switch to capture stderr per-call by redirecting fd 2.
    saved_stderr = os.dup(2)

    for label, fname in cases:
        tokens_np = np.load(tokens_dir / fname)
        tokens = torch.from_numpy(tokens_np).to("cuda")
        log_path = out_dir / f"trace_{label}_{fname.replace('.npy', '.log')}"
        print(f"[trace] case={label} fname={fname} log={log_path}", flush=True, file=sys.stderr)
        # Reopen fd 2 to log file.
        log_fd = os.open(str(log_path), os.O_WRONLY | os.O_CREAT | os.O_TRUNC)
        os.dup2(log_fd, 2)
        try:
            with torch.inference_mode():
                wav, _ = model.s3gen.inference(speech_tokens=tokens,
                                                ref_dict=model.conds.gen)
        finally:
            os.dup2(saved_stderr, 2)
            os.close(log_fd)
        arr = wav.detach().to("cpu", torch.float32).reshape(-1).numpy()
        rms = float((arr ** 2).mean() ** 0.5)
        f, p = welch(arr, sr, nperseg=2048)
        tot = float(p.sum()) + 1e-12
        cent = float((f * p).sum() / tot)
        below = float(p[f < 300].sum() / tot)
        is_bad = (cent < 700 and below > 0.5) or rms < 0.04
        print(f"[trace] {label}: rms={rms:.4f} cent={cent:.0f}Hz <300={below:.2f} "
              f"-> {'BAD' if is_bad else 'ok'}", flush=True, file=sys.stderr)


if __name__ == "__main__":
    main()
