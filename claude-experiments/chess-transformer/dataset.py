"""
Streaming dataset for bitboard format.
Memory-maps the file and reads batches sequentially, unpacking bitboards on the fly.
124 bytes/example: 12 × u64 bitboards (96 bytes) + 7 × f32 scalars (28 bytes).
"""

import numpy as np
import torch
from pathlib import Path

BITBOARD_BYTES = 12 * 8  # 96
SCALAR_BYTES = 7 * 4     # 28
EXAMPLE_BYTES = BITBOARD_BYTES + SCALAR_BYTES  # 124


def unpack_bitboards_to_tensor(raw_u64: np.ndarray) -> torch.Tensor:
    """Unpack (N, 12) uint64 bitboards to (N, 12, 8, 8) float32 tensor."""
    n = raw_u64.shape[0]
    bytes_view = raw_u64.view(np.uint8).reshape(n, 12, 8)
    bits = np.unpackbits(bytes_view, axis=2).reshape(n, 12, 8, 8).astype(np.float32)
    return torch.from_numpy(bits)


class StreamingBitboardDataset:
    """Streams batches from a memory-mapped bitboard binary file."""

    def __init__(self, path: str, batch_size: int = 8192, val_frac: float = 0.02):
        self.path = Path(path)
        file_size = self.path.stat().st_size
        assert file_size % EXAMPLE_BYTES == 0, f"File size {file_size} not divisible by {EXAMPLE_BYTES}"
        self.n_examples = file_size // EXAMPLE_BYTES
        self.batch_size = batch_size

        # Memory-map as raw bytes
        self.data = np.memmap(path, dtype=np.uint8, mode='r',
                              shape=(self.n_examples, EXAMPLE_BYTES))

        # Split: last val_frac is validation
        self.n_val = max(1, int(self.n_examples * val_frac))
        self.n_train = self.n_examples - self.n_val

        # Normalization stats from a sample
        sample_idx = np.random.default_rng(42).choice(
            self.n_train, min(100_000, self.n_train), replace=False)
        sample_scalars = self.data[sample_idx, BITBOARD_BYTES:].copy().view(np.float32).reshape(-1, 7)[:, :6]
        self.scalar_mean = torch.tensor(sample_scalars.mean(axis=0), dtype=torch.float32)
        self.scalar_std = torch.tensor(sample_scalars.std(axis=0) + 1e-8, dtype=torch.float32)

        self.n_train_batches = (self.n_train + batch_size - 1) // batch_size

        print(f"Dataset: {self.n_examples:,} total ({self.n_train:,} train, {self.n_val:,} val)", flush=True)
        print(f"File: {path} ({file_size / 1e9:.1f} GB, {EXAMPLE_BYTES} bytes/example)", flush=True)
        print(f"Batches per epoch: {self.n_train_batches}", flush=True)

    def _make_batch(self, start: int, end: int, device: torch.device):
        """Read a slice, unpack bitboards, normalize scalars, move to device."""
        chunk = self.data[start:end]

        # Unpack bitboards
        bb_raw = chunk[:, :BITBOARD_BYTES].copy().view(np.uint64).reshape(-1, 12)
        boards = unpack_bitboards_to_tensor(bb_raw).to(device)

        # Scalars and target
        scalar_raw = chunk[:, BITBOARD_BYTES:].copy().view(np.float32).reshape(-1, 7)
        scalars = torch.from_numpy(scalar_raw[:, :6].copy())
        scalars = ((scalars - self.scalar_mean) / self.scalar_std).to(device)
        targets = torch.from_numpy(scalar_raw[:, 6].copy()).to(device)

        return boards, scalars, targets

    def train_batches(self, device, max_batches=None):
        """Yield training batches sequentially."""
        n = 0
        for i in range(0, self.n_train, self.batch_size):
            if max_batches is not None and n >= max_batches:
                return
            end = min(i + self.batch_size, self.n_train)
            yield self._make_batch(i, end, device)
            n += 1

    def val_batches(self, device, max_batches=200):
        """Yield validation batches."""
        n = 0
        for i in range(self.n_train, self.n_examples, self.batch_size * 2):
            if n >= max_batches:
                return
            end = min(i + self.batch_size * 2, self.n_examples)
            yield self._make_batch(i, end, device)
            n += 1
