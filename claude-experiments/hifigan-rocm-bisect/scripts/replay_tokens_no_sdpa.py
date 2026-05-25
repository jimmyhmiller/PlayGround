"""Replay tokens but disable experimental SDPA kernels.

CHATTERBOX_DEBUG.md hypothesized that ROCm's experimental Flash/Mem
Efficient SDPA kernels are the source of muffled-output anomalies. Test
that here by:
- forcing torch.backends.cuda.enable_flash_sdp(False)
- enable_mem_efficient_sdp(False)
- enable_math_sdp(True) — fall back to math
- TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL=0

If the anomaly rate drops to 0, SDPA is the cause. If it stays the same,
SDPA is NOT the cause and we need to keep digging.
"""
from __future__ import annotations
import argparse, json, os, sys, time
from pathlib import Path
import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=20)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--voice-ref", default="/home/jimmyhmiller/.config/paper-audiobooks/default-voice.wav")
    ap.add_argument("--tokens-dir", default="../paper-audiobooks/bisect_results")
    args = ap.parse_args()

    os.environ.setdefault("PYTORCH_HIP_ALLOC_CONF", "expandable_segments:True")
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    # Disable experimental ROCm SDPA kernels.
    os.environ["TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL"] = "0"

    here = Path(__file__).resolve().parents[1]
    cb_src = here.parent / "chatterbox-rewrite" / "chatterbox" / "src"
    sys.path.insert(0, str(cb_src))

    import torch
    print(f"[no_sdpa] torch={torch.__version__} cuda_avail={torch.cuda.is_available()}",
          file=sys.stderr)
    # Disable the SDPA backends in torch.
    if hasattr(torch.backends.cuda, "enable_flash_sdp"):
        torch.backends.cuda.enable_flash_sdp(False)
        torch.backends.cuda.enable_mem_efficient_sdp(False)
        torch.backends.cuda.enable_math_sdp(True)
        print("[no_sdpa] disabled flash + mem_efficient SDPA, kept math", file=sys.stderr)

    from chatterbox.tts import ChatterboxTTS  # type: ignore
    print("[no_sdpa] loading ChatterboxTTS ...", file=sys.stderr)
    t0 = time.time()
    model = ChatterboxTTS.from_pretrained(device=args.device)
    print(f"[no_sdpa] loaded in {time.time()-t0:.1f}s", file=sys.stderr)
    model.prepare_conditionals(args.voice_ref)

    tokens_dir = Path(args.tokens_dir).resolve()
    token_files = sorted(tokens_dir.glob("phase1_trial*_tokens.npy"))
    print(f"[no_sdpa] {len(token_files)} token files", file=sys.stderr)

    from scipy.signal import welch
    sr = int(model.sr)

    print(f"\n{'trial':>5s}  {'tokenfile':>30s}  {'rms':>8s}  {'cent':>6s}  "
          f"{'<300':>5s}  {'classify':>10s}")
    bad = 0
    rmses = []
    for i in range(args.n):
        tf = token_files[i % len(token_files)]
        tokens_np = np.load(tf)
        tokens = torch.from_numpy(tokens_np).to(args.device)
        with torch.inference_mode():
            wav, _ = model.s3gen.inference(speech_tokens=tokens, ref_dict=model.conds.gen)
        arr = wav.detach().to("cpu", torch.float32).reshape(-1).numpy()
        rms = float((arr ** 2).mean() ** 0.5)
        f, p = welch(arr, sr, nperseg=2048)
        tot = float(p.sum()) + 1e-12
        cent = float((f * p).sum() / tot)
        below = float(p[f < 300].sum() / tot)
        is_bad = (cent < 700 and below > 0.5) or rms < 0.04
        if is_bad: bad += 1
        rmses.append(rms)
        print(f"{i:>5d}  {tf.name:>30s}  {rms:.4f}  {cent:>5.0f}  {below:.2f}  "
              f"{'BAD' if is_bad else 'ok'}")

    print(f"\n[no_sdpa] bad={bad}/{args.n}, rms range "
          f"[{min(rmses):.4f}, {max(rmses):.4f}]")


if __name__ == "__main__":
    main()
