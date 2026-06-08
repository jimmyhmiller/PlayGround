"""Run two ChatterboxTTS instances in parallel subprocesses; verify both
finish without HSA exceptions. Memory recall says upstream chatterbox
crashes on this hardware when run in parallel — let's see if we do too.

Each child: load model, prepare conditionals, run N generate() calls.
Parent: launches both children simultaneously, captures their output.
"""
import os, sys, subprocess, time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

CHILD_SCRIPT_TEMPLATE = '''
import sys, os, time
sys.path.insert(0, "__ROOT__")
from chatterbox_mojo import ChatterboxTTS

tag = sys.argv[1]
n = int(sys.argv[2])

print(f"[{tag}] loading model...", flush=True)
t0 = time.perf_counter()
tts = ChatterboxTTS.from_pretrained(use_bf16=True)
tts.prepare_conditionals("/home/jimmyhmiller/.config/paper-audiobooks/default-voice.wav", exaggeration=0.5)
print(f"[{tag}] loaded in {time.perf_counter()-t0:.1f}s", flush=True)

phrases = [
    "Hello world.",
    "The quick brown fox jumps over the lazy dog.",
    "She sells seashells by the seashore.",
    "Testing one, two, three.",
]
for i in range(n):
    t0 = time.perf_counter()
    try:
        wav = tts.generate(phrases[i % len(phrases)], cfg_weight=0.5,
                            temperature=0.8, top_p=0.95, min_p=0.05,
                            repetition_penalty=1.2, exaggeration=0.5)
        dt = time.perf_counter() - t0
        print(f"[{tag}] iter {i}: {dt:.2f}s  samples={wav.shape[-1]}", flush=True)
    except Exception as e:
        print(f"[{tag}] iter {i}: ERROR {type(e).__name__}: {e}", flush=True)
        sys.exit(1)
print(f"[{tag}] done", flush=True)
'''
CHILD_SCRIPT = CHILD_SCRIPT_TEMPLATE.replace("__ROOT__", str(ROOT))


def main():
    n_iters = int(os.environ.get("N", "5"))
    print(f"[parallel-test] spawning 2 children, {n_iters} iters each")

    env = os.environ.copy()
    env["CHATTERBOX_BF16"] = "1"
    env["CHATTERBOX_CFM_STEPS"] = "5"

    procs = []
    for tag in ["A", "B"]:
        p = subprocess.Popen(
            ["pixi", "run", "python", "-c", CHILD_SCRIPT, tag, str(n_iters)],
            cwd=str(ROOT),
            env=env,
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1,
        )
        procs.append(p)
        time.sleep(0.5)  # slight stagger so they don't race the cache

    print(f"[parallel-test] both children launched, waiting...")
    statuses = []
    for p in procs:
        out, _ = p.communicate(timeout=900)
        print(f"\n========= child rc={p.returncode} =========")
        print(out)
        statuses.append(p.returncode)

    if all(s == 0 for s in statuses):
        print("\n[parallel-test] OK — both children succeeded")
    else:
        print(f"\n[parallel-test] FAILED — rcs={statuses}")
        sys.exit(1)


if __name__ == "__main__":
    sys.exit(main() or 0)
