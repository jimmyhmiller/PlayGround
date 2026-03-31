"""Benchmark: Maia-style residual CNN vs our transformer."""

import os
os.environ["ROCBLAS_USE_HIPBLASLT"] = "1"
os.environ["PYTORCH_TUNABLEOP_ENABLED"] = "1"
os.environ["TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL"] = "1"

import torch
import torch.nn as nn
import time


class ResBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        residual = x
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.bn2(self.conv2(x))
        x = self.relu(x + residual)
        return x


class MoveTimeCNN(nn.Module):
    def __init__(self, in_channels=12, cnn_channels=128, n_blocks=6, n_scalars=6):
        super().__init__()
        self.cnn_channels = cnn_channels
        self.n_blocks = n_blocks

        # Input conv
        self.input_conv = nn.Sequential(
            nn.Conv2d(in_channels, cnn_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(cnn_channels),
            nn.ReLU(inplace=True),
        )

        # Residual blocks
        self.blocks = nn.Sequential(*[ResBlock(cnn_channels) for _ in range(n_blocks)])

        # Global average pool -> (B, cnn_channels)
        self.pool = nn.AdaptiveAvgPool2d(1)

        # Scalar feature encoder
        self.scalar_net = nn.Sequential(
            nn.Linear(n_scalars, 128),
            nn.GELU(),
            nn.Linear(128, 128),
        )

        # Regression head
        self.head = nn.Sequential(
            nn.Linear(cnn_channels + 128, 256),
            nn.GELU(),
            nn.Linear(256, 1),
        )

    def forward(self, board, scalars):
        # Board: (B, 12, 8, 8)
        x = self.input_conv(board)
        x = self.blocks(x)
        x = self.pool(x).squeeze(-1).squeeze(-1)  # (B, cnn_channels)

        # Scalars: (B, 6)
        s = self.scalar_net(scalars)  # (B, 128)

        # Combine and predict
        combined = torch.cat([x, s], dim=1)  # (B, cnn_channels + 128)
        return self.head(combined).squeeze(-1)  # (B,)


def bench(name, model, device, board, scalars, target, n_warmup=5, n_bench=50):
    print(f"\n{'='*60}", flush=True)
    print(f"{name}", flush=True)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  Parameters: {n_params:,}", flush=True)

    criterion = nn.MSELoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)
    scaler = torch.amp.GradScaler()

    # Warmup
    for _ in range(n_warmup):
        with torch.amp.autocast('cuda', dtype=torch.float16):
            pred = model(board, scalars)
            loss = criterion(pred, target)
        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
    torch.cuda.synchronize()

    # Benchmark
    t0 = time.time()
    for _ in range(n_bench):
        with torch.amp.autocast('cuda', dtype=torch.float16):
            pred = model(board, scalars)
            loss = criterion(pred, target)
        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(optimizer)
        scaler.update()
    torch.cuda.synchronize()
    elapsed = time.time() - t0

    ms = elapsed / n_bench * 1000
    epoch_batches = 87341
    epoch_h = elapsed / n_bench * epoch_batches / 3600
    print(f"  {ms:.0f} ms/batch, ~{epoch_h:.1f} hours/epoch", flush=True)

    del model, optimizer, scaler
    torch.cuda.empty_cache()


if __name__ == "__main__":
    torch.set_float32_matmul_precision('high')
    device = torch.device('cuda')
    print(f"PyTorch: {torch.__version__}", flush=True)

    board = torch.randn(8192, 12, 8, 8, device=device)
    scalars = torch.randn(8192, 6, device=device)
    target = torch.randn(8192, device=device)

    configs = [
        ("CNN 6 blocks x 64ch (~0.3M)", MoveTimeCNN(cnn_channels=64, n_blocks=6)),
        ("CNN 6 blocks x 128ch (~1.1M)", MoveTimeCNN(cnn_channels=128, n_blocks=6)),
        ("CNN 8 blocks x 128ch (~1.5M)", MoveTimeCNN(cnn_channels=128, n_blocks=8)),
        ("CNN 12 blocks x 128ch (~2.2M)", MoveTimeCNN(cnn_channels=128, n_blocks=12)),
        ("CNN 8 blocks x 256ch (~5.4M)", MoveTimeCNN(cnn_channels=256, n_blocks=8)),
        ("CNN 12 blocks x 256ch (~8.0M)", MoveTimeCNN(cnn_channels=256, n_blocks=12)),
    ]

    for name, model in configs:
        model = model.to(device)
        bench(name, model, device, board, scalars, target)

    print("\nDone!", flush=True)
