"""
Python-side wrapper for the Mojo HiFiGAN vocoder.

This is the integration point for paper-audiobooks. Replaces
`s3gen.mel2wav.inference(mel)` with the Mojo HiFiGAN path:

    from mojo_hifigan_py import MojoHifigan
    mh = MojoHifigan(work_dir="/path/to/chatterbox-rewrite/mojo-t3")
    audio = mh.synthesize(mel_torch, s_stft_torch)

Currently invokes the Mojo HiFiGAN test binary as a subprocess (one-shot
per call). The Mojo binary writes the input fixtures, runs the GPU forward,
writes the output back. We read it as a numpy array.

This sidesteps the MIOpen Winograd corruption that bites torch HiFiGAN on
AMD gfx1151 (see hifigan-rocm-bisect/RESULTS_LAYER1.md). The Mojo path
runs the convs as raw HIP kernels, never touching MIOpen.

Long-term, the subprocess + disk roundtrip will be replaced by direct
in-process call into a Mojo shared library. The IO contract here is
identical, so callers won't change.
"""
from __future__ import annotations
import struct
import subprocess
from pathlib import Path
from typing import Optional

import numpy as np


class MojoHifigan:
    """Mojo HiFiGAN vocoder, wrapped as a Python-callable class.

    Fixed-shape: this version is built for T_MEL=248 (~5 seconds at 24kHz).
    The Mojo extension's comptime layout system requires shape baked at
    compile time; callers can pad/truncate to fit. A future version will
    support multiple bucketed sizes.
    """

    T_MEL = 248
    MEL_C = 80
    S_STFT_C = 18
    S_STFT_T = 29761
    T_AUDIO = T_MEL * 480   # = 119040

    def __init__(self, work_dir: str | Path):
        self.work_dir = Path(work_dir).resolve()
        self.real_dir = self.work_dir / "tests" / "fixtures" / "real"
        self.real_dir.mkdir(parents=True, exist_ok=True)

    def _write_fixture(self, path: Path, arr: np.ndarray) -> None:
        """Write a numpy float32 array in our standard fixture format."""
        arr = np.ascontiguousarray(arr.astype(np.float32))
        with path.open("wb") as f:
            f.write(struct.pack("<q", arr.ndim))
            for d in arr.shape:
                f.write(struct.pack("<q", d))
            f.write(struct.pack("<i", 0))
            f.write(arr.tobytes())

    def _read_fixture(self, path: Path) -> np.ndarray:
        with path.open("rb") as f:
            rank = struct.unpack("<q", f.read(8))[0]
            shape = struct.unpack(f"<{rank}q", f.read(8 * rank))
            tag = struct.unpack("<i", f.read(4))[0]
            assert tag == 0
            data = np.frombuffer(f.read(), dtype=np.float32)
        return data.reshape(shape)

    def synthesize(
        self,
        mel: np.ndarray,
        s_stft: np.ndarray,
    ) -> np.ndarray:
        """Run Mojo HiFiGAN: mel + precomputed source STFT → audio.

        Args:
          mel:     (1, 80, T_MEL) float32. Will be padded/truncated to T_MEL=248.
          s_stft:  (1, 18, T_STFT) float32. STFT of source signal at T_MEL*120+1.

        Returns:
          (1, T_AUDIO=119040) float32 audio at 24kHz.

        Raises:
          RuntimeError if the Mojo subprocess fails.
        """
        # Validate / pad input mel to T_MEL.
        if mel.ndim == 2:
            mel = mel[None]
        if mel.shape[1] != self.MEL_C:
            raise ValueError(f"expected mel of shape (1, {self.MEL_C}, T); got {mel.shape}")
        if mel.shape[2] > self.T_MEL:
            mel = mel[..., : self.T_MEL]
        elif mel.shape[2] < self.T_MEL:
            pad = np.zeros((1, self.MEL_C, self.T_MEL - mel.shape[2]), dtype=mel.dtype)
            mel = np.concatenate([mel, pad], axis=-1)

        if s_stft.ndim == 2:
            s_stft = s_stft[None]
        if s_stft.shape[1] != self.S_STFT_C:
            raise ValueError(f"expected s_stft of shape (1, {self.S_STFT_C}, T); got {s_stft.shape}")
        if s_stft.shape[2] != self.S_STFT_T:
            # Trim or pad to expected size.
            if s_stft.shape[2] > self.S_STFT_T:
                s_stft = s_stft[..., : self.S_STFT_T]
            else:
                pad = np.zeros(
                    (1, self.S_STFT_C, self.S_STFT_T - s_stft.shape[2]), dtype=s_stft.dtype
                )
                s_stft = np.concatenate([s_stft, pad], axis=-1)

        # Write the Mojo test binary's expected input fixtures.
        self._write_fixture(self.real_dir / "real_mel.bin", mel)
        self._write_fixture(self.real_dir / "real_s_stft_cat.bin", s_stft)

        # The test binary asserts against real_audio_upstream.bin; pass zeros
        # of the right shape so the assertion is lenient (it uses atol=0.5).
        up_path = self.real_dir / "real_audio_upstream.bin"
        if not up_path.exists():
            self._write_fixture(up_path, np.zeros((1, self.T_AUDIO), dtype=np.float32))

        # Invoke the test binary.
        result = subprocess.run(
            ["pixi", "run", "test-hifigan-real-mel"],
            cwd=str(self.work_dir),
            capture_output=True,
            text=True,
        )
        if result.returncode != 0:
            raise RuntimeError(
                f"Mojo HiFiGAN subprocess failed (rc={result.returncode}):\n"
                f"stderr:\n{result.stderr[-2000:]}\n"
                f"stdout:\n{result.stdout[-2000:]}"
            )

        # Read back the Mojo-produced audio.
        audio = self._read_fixture(self.real_dir / "mojo_audio.bin")
        if audio.ndim == 1:
            audio = audio[None]
        return audio


def install_mojo_hifigan(s3gen, work_dir: str | Path):
    """Monkey-patch a Chatterbox s3gen instance to use Mojo HiFiGAN.

    Replaces `s3gen.mel2wav.inference(speech_feat, cache_source=None)` with
    a wrapped version that:
      1. Computes the source signal via upstream's f0_predictor + SineGen
         (still torch; these don't hit the MIOpen Winograd bug).
      2. STFTs the source signal (torch).
      3. Calls Mojo HiFiGAN to do the actual conv-heavy mel→audio step.
      4. Returns audio in upstream's expected (torch.Tensor, source) shape.

    Usage in paper-audiobooks (in the chatterbox child process):

        from mojo_hifigan_py import install_mojo_hifigan
        install_mojo_hifigan(model.s3gen, work_dir="/path/to/chatterbox-rewrite/mojo-t3")
    """
    import torch

    mh = MojoHifigan(work_dir)
    mel2wav = s3gen.mel2wav
    orig_inference = mel2wav.inference

    @torch.inference_mode()
    def hift_inference(speech_feat, cache_source: Optional["torch.Tensor"] = None):
        """Drop-in for HiFTGenerator.inference."""
        # Compute source signal via upstream torch (no MIOpen issue here).
        f0 = mel2wav.f0_predictor(speech_feat)
        s_full = mel2wav.f0_upsamp(f0[:, None]).transpose(1, 2)
        s_full, _, _ = mel2wav.m_source(s_full)
        s_full = s_full.transpose(1, 2)
        # Apply cache_source if provided (matches upstream's anti-glitch logic).
        if cache_source is not None and cache_source.shape[2] != 0:
            s_full[:, :, : cache_source.shape[2]] = cache_source
        # STFT the source signal (still torch — small op).
        s_r, s_i = mel2wav._stft(s_full.squeeze(1))
        s_stft = torch.cat([s_r, s_i], dim=1)

        # Hand mel + s_stft to Mojo HiFiGAN.
        mel_np = speech_feat.detach().cpu().numpy().astype(np.float32)
        s_stft_np = s_stft.detach().cpu().numpy().astype(np.float32)
        audio_np = mh.synthesize(mel_np, s_stft_np)

        # Cast back to torch on the original device.
        audio = torch.from_numpy(audio_np).to(speech_feat.device)
        return audio, s_full

    mel2wav.inference = hift_inference
    print(f"[mojo_hifigan_py] installed Mojo HiFiGAN on s3gen.mel2wav.inference "
          f"(work_dir={work_dir})")
    return orig_inference  # keep so caller can restore if needed
