"""Find LOUD thumps (max_diff > 0.8 — guaranteed perceptual) in the long
Quine output, then inspect mel around them.
"""
import wave, numpy as np
src = "/tmp/quine_bench/chatterbox-mojo.wav"
with wave.open(src, "rb") as w:
    sr = w.getframerate()
    pcm = np.frombuffer(w.readframes(w.getnframes()), dtype=np.int16).astype(np.float32) / 32768

diff = np.abs(np.diff(pcm))
# Cluster.
big = np.flatnonzero(diff > 0.3)
gaps = np.diff(big)
starts = np.concatenate([[0], np.flatnonzero(gaps > sr // 20) + 1])

# Find the LOUD ones (max in cluster > 0.8).
loud_events = []
for k in range(len(starts)):
    s = starts[k]
    e = starts[k + 1] if k + 1 < len(starts) else len(big)
    idxs = big[s:e]
    md = diff[idxs].max()
    if md > 0.8:
        loud_events.append((idxs[0], md))

print(f"loud events (max_diff > 0.8): {len(loud_events)} / {len(starts)} total")
print()
print(f"{'#':>3} {'sample':>10} {'time (s)':>10} {'max|d|':>8}")
for k, (s, md) in enumerate(loud_events[:30]):
    print(f"{k:>3} {s:>10} {s/sr:>10.3f} {md:>8.3f}")
