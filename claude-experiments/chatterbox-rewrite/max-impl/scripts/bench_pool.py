"""Throughput bench: Mojo WorkerPool (N workers) vs upstream sequential.

Set N_WORKERS env to change worker count (default 2).
"""
import json, os, subprocess, sys, time, wave
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

REF_WAV = "/home/jimmyhmiller/.config/paper-audiobooks/default-voice.wav"
OUT = Path("/tmp/gettier_pool_bench")
OUT.mkdir(exist_ok=True)
UPSTREAM_PY = os.path.expanduser("~/.cache/paper-audiobooks/venvs/chatterbox/bin/python")

# Same chunks as bench_pdf.py for apples-to-apples.
CHUNKS = [
    "Various attempts have been made in recent years to state necessary and sufficient conditions for someone's knowing a given proposition.",
    "I shall argue that the standard analysis of knowledge as justified true belief is false.",
    "It is possible for a person to be justified in believing a proposition that is in fact false.",
    "Suppose that Smith and Jones have applied for a certain job.",
    "And suppose that Smith has strong evidence for the conjunctive proposition that Jones is the man who will get the job, and Jones has ten coins in his pocket.",
    "But imagine, further, that unknown to Smith, he himself, not Jones, will get the job.",
    "And, also, unknown to Smith, he himself has ten coins in his pocket.",
    "All of the following are true: the proposition is true, Smith believes that it is true, and Smith is justified in believing that it is true.",
    "But it is equally clear that Smith does not know that the proposition is true.",
    "These two examples show that there is justified true belief which is not knowledge.",
]


def bench_mojo_pool(n_workers: int) -> tuple[float, list[float]]:
    from chatterbox_mojo.pool import WorkerPool
    print(f"[bench] launching mojo pool with {n_workers} workers")
    t_load0 = time.perf_counter()
    pool = WorkerPool(n_workers=n_workers, voice_ref=REF_WAV, use_bf16=True, cfm_steps=5)
    print(f"[bench] mojo pool loaded in {time.perf_counter()-t_load0:.2f}s")
    # Warmup one synth per worker to amortize JIT etc.
    pool.synthesize_many(["warmup"] * n_workers)
    t0 = time.perf_counter()
    audios = pool.synthesize_many(CHUNKS, cfg_weight=0.5, temperature=0.8, top_p=0.95,
                                  min_p=0.05, repetition_penalty=1.2, exaggeration=0.5)
    elapsed = time.perf_counter() - t0
    durs = [a.size / 24000 for a in audios]
    pool.close()
    return elapsed, durs


# Reuse bench_pdf upstream path.
UPSTREAM_SCRIPT = r'''
import sys, json, time, os, wave
import numpy as np
sys.path.insert(0, "/home/jimmyhmiller/Documents/Code/Playground/claude-experiments/chatterbox-rewrite/chatterbox/src")
import importlib.metadata as _im
_orig = _im.version
def _v(name):
    try: return _orig(name)
    except _im.PackageNotFoundError: return "0.0.0"
_im.version = _v
import torch
torch.manual_seed(0xDEADBEEF)
from chatterbox.tts import ChatterboxTTS
device = "cuda" if torch.cuda.is_available() else "cpu"
ref_wav, out_dir, chunks_json = sys.argv[1], sys.argv[2], sys.argv[3]
chunks = json.loads(chunks_json)
t0 = time.perf_counter()
tts = ChatterboxTTS.from_pretrained(device=device)
tts.prepare_conditionals(ref_wav, exaggeration=0.5)
print(json.dumps({"event": "load", "wall": time.perf_counter() - t0}), flush=True)
for i, text in enumerate(chunks):
    t0 = time.perf_counter()
    wav = tts.generate(text, cfg_weight=0.5, temperature=0.8, top_p=0.95,
                       min_p=0.05, repetition_penalty=1.2, exaggeration=0.5)
    wall = time.perf_counter() - t0
    arr = wav.squeeze(0).cpu().numpy().astype(np.float32)
    dur = arr.size / tts.sr
    print(json.dumps({"event": "chunk", "i": i, "wall": wall, "audio": dur}), flush=True)
'''


def bench_upstream() -> tuple[float, list[float]]:
    print(f"[bench] launching upstream sequential")
    clean_env = {k: v for k, v in os.environ.items()
                 if not k.startswith(("PIXI_", "CONDA_", "PYTHON"))
                 and k not in ("VIRTUAL_ENV", "PYTHONHOME", "PYTHONPATH")}
    clean_env["PATH"] = "/usr/bin:/usr/local/bin"
    clean_env["HOME"] = os.environ.get("HOME", "/home/jimmyhmiller")
    proc = subprocess.Popen(
        [UPSTREAM_PY, "-c", UPSTREAM_SCRIPT, REF_WAV, str(OUT), json.dumps(CHUNKS)],
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, env=clean_env, text=True, bufsize=1,
    )
    durs = []
    walls = []
    for line in proc.stdout:
        try:
            ev = json.loads(line.strip())
        except Exception:
            continue
        if ev["event"] == "chunk":
            walls.append(ev["wall"])
            durs.append(ev["audio"])
    rc = proc.wait()
    if rc != 0:
        print(proc.stderr.read()[-2000:], file=sys.stderr)
        raise RuntimeError(f"upstream rc={rc}")
    return sum(walls), durs


def main():
    print(f"[bench] {len(CHUNKS)} chunks, {sum(len(c) for c in CHUNKS)} chars")
    n_w = int(os.environ.get("N_WORKERS", "2"))
    mojo_wall, mojo_durs = bench_mojo_pool(n_workers=n_w)
    print()
    up_wall, up_durs = bench_upstream()
    ma = sum(mojo_durs); ua = sum(up_durs)
    print()
    print("=" * 80)
    print(f"{'engine':<14} {'wall':>9} {'audio':>9} {'RTF':>8} {'speedup':>10}")
    print("-" * 80)
    print(f"{f'mojo ({n_w}w)':<14} {mojo_wall:>8.2f}s {ma:>8.2f}s {mojo_wall/ma:>7.3f}x {'(baseline)':>10}")
    print(f"{'upstream':<14} {up_wall:>8.2f}s {ua:>8.2f}s {up_wall/ua:>7.3f}x {up_wall/mojo_wall:>9.2f}x")
    print("=" * 80)
    print()
    print(f"mojo:     {ma:.1f}s audio in {mojo_wall:.1f}s wall = {ma/mojo_wall:.2f}x real-time")
    print(f"upstream: {ua:.1f}s audio in {up_wall:.1f}s wall = {ua/up_wall:.2f}x real-time")
    if up_wall > mojo_wall:
        print(f"\nMojo (2 workers) is {up_wall/mojo_wall:.2f}x faster than upstream.")
    else:
        print(f"\nUpstream is {mojo_wall/up_wall:.2f}x faster than Mojo (2 workers).")


if __name__ == "__main__":
    sys.exit(main() or 0)
