"""Bench WorkerPool(n=2) vs single-instance sequential on Gettier chunks.

Each child loads its own ChatterboxTTS (so 2× model footprint, ~10 GB extra).
On Strix Halo's 128 GB GTT pool this fits easily.
"""
import sys, time, wave
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
from chatterbox_mojo import ChatterboxTTS
from chatterbox_mojo.pool import WorkerPool

REF = "/home/jimmyhmiller/.config/paper-audiobooks/default-voice.wav"
OUT = Path("/tmp/pool_test")
OUT.mkdir(exist_ok=True)

# Same Gettier-paper subset as bench_pdf.py.
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
    "These two examples show that there is justified true belief that is not knowledge.",
]


def write_wav(arr, path, sr=24000):
    arr = np.clip(arr, -1.0, 1.0)
    pcm = (arr * 32767.0).astype(np.int16)
    with wave.open(path, "wb") as w:
        w.setnchannels(1); w.setsampwidth(2); w.setframerate(sr)
        w.writeframes(pcm.tobytes())


def run_pool(n_workers: int):
    print(f"=== pool (n_workers={n_workers}) ===")
    t0 = time.perf_counter()
    pool = WorkerPool(n_workers=n_workers, voice_ref=REF, use_bf16=True, cfm_steps=5)
    load_dt = time.perf_counter() - t0
    print(f"  load+prepare (all workers): {load_dt:.1f}s")

    t0 = time.perf_counter()
    pool_audios = pool.synthesize_many(
        CHUNKS, cfg_weight=0.5, temperature=0.8, top_p=0.95,
        min_p=0.05, repetition_penalty=1.2, exaggeration=0.5,
    )
    pool_wall = time.perf_counter() - t0
    pool_audio_total = sum(a.size for a in pool_audios) / 24000
    for i, a in enumerate(pool_audios):
        write_wav(a, str(OUT / f"pool{n_workers}_{i:02d}.wav"))
    print(f"  TOTAL pool({n_workers}): {pool_wall:.2f}s wall, {pool_audio_total:.2f}s audio, "
          f"RTF={pool_wall/pool_audio_total:.3f}  ({pool_audio_total/pool_wall:.2f}x RT)")
    pool.close()
    return pool_wall, pool_audio_total


def main():
    mode = sys.argv[1] if len(sys.argv) > 1 else "pool2"

    if mode == "seq":
        # Sequential, in-process.
        tts = ChatterboxTTS.from_pretrained(use_bf16=True)
        tts.prepare_conditionals(REF, exaggeration=0.5)
        t0 = time.perf_counter()
        audios = []
        for i, text in enumerate(CHUNKS):
            w = tts.generate(text, cfg_weight=0.5, temperature=0.8, top_p=0.95,
                              min_p=0.05, repetition_penalty=1.2, exaggeration=0.5,
                              rng_seed=0xDEADBEEF + i)
            audios.append(w.squeeze(0).cpu().numpy().astype(np.float32))
        wall = time.perf_counter() - t0
        audio = sum(a.size for a in audios) / 24000
        for i, a in enumerate(audios):
            write_wav(a, str(OUT / f"seq_{i:02d}.wav"))
        print(f"seq: {wall:.2f}s wall, {audio:.2f}s audio, RTF={wall/audio:.3f} ({audio/wall:.2f}x RT)")
    elif mode.startswith("pool"):
        n = int(mode[4:])
        run_pool(n)
    else:
        print("usage: test_pool.py {seq|pool1|pool2|pool3|...}")
        sys.exit(2)


if __name__ == "__main__":
    sys.exit(main() or 0)
