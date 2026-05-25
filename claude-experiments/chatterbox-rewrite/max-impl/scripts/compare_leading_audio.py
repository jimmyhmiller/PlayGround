"""Compare leading samples of our Mojo audio vs upstream's audio for the same
speech_tokens. The user reports 'the' is cut off at start of Mojo output.
"""
import struct
import numpy as np


def read_fp32(path):
    with open(path, "rb") as f:
        rank = struct.unpack("<q", f.read(8))[0]
        shape = struct.unpack(f"<{rank}q", f.read(8 * rank))
        tag = struct.unpack("<i", f.read(4))[0]
        assert tag == 0, f"expected fp32, got tag {tag}"
        data = np.frombuffer(f.read(), dtype=np.float32).copy()
    if rank > 0:
        return data.reshape(shape)
    return data


def read_wav(path):
    from scipy.io import wavfile
    sr, data = wavfile.read(path)
    if data.dtype == np.int16:
        data = data.astype(np.float32) / 32768.0
    return sr, data


def main():
    # Upstream audio (from prior dump)
    up_audio = read_fp32("weights/s3gen_prompt/hift_dump/audio.bin")
    if up_audio.ndim > 1:
        up_audio = up_audio[0]
    print(f"upstream audio: shape={up_audio.shape} max-abs={np.abs(up_audio).max():.4f}")
    print(f"  first 30 samples: {up_audio[:30]}")
    print(f"  first 30 max-abs: {np.abs(up_audio[:30]).max():.4f}")
    print(f"  samples 0-2400 (first 100ms) max-abs: {np.abs(up_audio[:2400]).max():.4f}")
    print(f"  samples 2400-4800 (100-200ms) max-abs: {np.abs(up_audio[2400:4800]).max():.4f}")

    # Mojo audio
    sr, mj_audio = read_wav("max_impl_with_prompt.wav")
    print(f"\nmojo audio:     sr={sr} shape={mj_audio.shape} max-abs={np.abs(mj_audio).max():.4f}")
    print(f"  first 30 samples: {mj_audio[:30]}")
    print(f"  first 30 max-abs: {np.abs(mj_audio[:30]).max():.4f}")
    print(f"  samples 0-2400 (first 100ms) max-abs: {np.abs(mj_audio[:2400]).max():.4f}")
    print(f"  samples 2400-4800 (100-200ms) max-abs: {np.abs(mj_audio[2400:4800]).max():.4f}")

    # Cross-correlate to find any offset
    n_xc = min(20000, len(up_audio), len(mj_audio))
    up_seg = up_audio[:n_xc]
    mj_seg = mj_audio[:n_xc]
    # Search for best alignment by sliding mj
    best_corr = -1.0
    best_shift = 0
    for shift in range(-200, 201):
        if shift >= 0:
            a = mj_seg[shift:n_xc]
            b = up_seg[:n_xc - shift]
        else:
            a = mj_seg[:n_xc + shift]
            b = up_seg[-shift:n_xc]
        if len(a) < 1000: continue
        c = np.corrcoef(a, b)[0, 1]
        if c > best_corr:
            best_corr = c
            best_shift = shift
    print(f"\nBest alignment: shift mojo by +{best_shift} → corr={best_corr:.4f}")

    # Compare segment by segment
    n = min(len(up_audio), len(mj_audio))
    print(f"\nSegment comparison (max-abs of |mojo-upstream| per 480-sample window):")
    print("window  mojo_max  upstream_max   diff_max   diff_relative")
    for w in range(0, min(40, n // 480)):
        s = w * 480
        e = s + 480
        mj = mj_audio[s:e]
        up = up_audio[s:e] if e <= len(up_audio) else None
        if up is None: break
        diff = np.abs(mj - up).max()
        mj_max = np.abs(mj).max()
        up_max = np.abs(up).max()
        rel = diff / max(up_max, 1e-9)
        marker = "  <-- BAD" if rel > 0.5 else ""
        print(f"  {w:3d}    {mj_max:.4f}    {up_max:.4f}      {diff:.4f}      {rel:.3f}{marker}")


if __name__ == "__main__":
    main()
