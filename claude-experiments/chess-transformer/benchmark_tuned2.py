"""Benchmark: hipBLASLt + TunableOp, d256 eager (no compile) and d128 compile."""

import os
os.environ["ROCBLAS_USE_HIPBLASLT"] = "1"
os.environ["PYTORCH_TUNABLEOP_ENABLED"] = "1"
os.environ["TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL"] = "1"

import torch
import torch._inductor.config as inductor_config
inductor_config.triton.cudagraphs = False
inductor_config.triton.cudagraph_trees = False

import torch.nn as nn
import time


class MoveTimeTransformer(nn.Module):
    def __init__(self, d_model=256, n_heads=8, n_layers=4, dropout=0.1):
        super().__init__()
        self.square_embed = nn.Linear(12, d_model)
        self.pos_embed = nn.Parameter(torch.randn(1, 65, d_model) * 0.02)
        self.cls_token = nn.Parameter(torch.randn(1, 1, d_model) * 0.02)
        self.scalar_proj = nn.Sequential(
            nn.Linear(6, d_model), nn.GELU(), nn.Linear(d_model, d_model))
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads, dim_feedforward=d_model * 4,
            dropout=dropout, activation='gelu', batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.head = nn.Sequential(
            nn.LayerNorm(d_model), nn.Linear(d_model, d_model),
            nn.GELU(), nn.Linear(d_model, 1))

    def forward(self, board, scalars):
        B = board.shape[0]
        x = board.reshape(B, 12, 64).permute(0, 2, 1)
        x = self.square_embed(x)
        cls = self.cls_token.expand(B, -1, -1)
        x = torch.cat([cls, x], dim=1)
        x = x + self.pos_embed
        x[:, 0] = x[:, 0] + self.scalar_proj(scalars)
        x = self.transformer(x)
        return self.head(x[:, 0]).squeeze(-1)


def bench(name, d_model, n_heads, n_layers, do_compile, device, board, scalars, target):
    print(f"\n{'='*60}", flush=True)
    print(f"Testing: {name}", flush=True)

    model = MoveTimeTransformer(d_model=d_model, n_heads=n_heads, n_layers=n_layers).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  Parameters: {n_params:,}", flush=True)

    if do_compile:
        print("  Compiling...", flush=True)
        model = torch.compile(model)

    criterion = nn.MSELoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)
    scaler = torch.amp.GradScaler()

    print("  Warmup...", flush=True)
    for i in range(10):
        with torch.amp.autocast('cuda', dtype=torch.float16):
            pred = model(board, scalars)
            loss = criterion(pred, target)
        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
    torch.cuda.synchronize()
    print("  Warmup done", flush=True)

    t0 = time.time()
    n = 30
    for _ in range(n):
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

    ms_per_batch = elapsed / n * 1000
    batches_per_epoch = 6_000_000 // 8192
    epoch_min = elapsed / n * batches_per_epoch / 60
    print(f"  {ms_per_batch:.0f} ms/batch, ~{epoch_min:.1f} min/epoch", flush=True)

    del model, optimizer, scaler
    torch.cuda.empty_cache()


if __name__ == "__main__":
    torch.set_float32_matmul_precision('high')
    device = torch.device('cuda')
    print(f"PyTorch: {torch.__version__}, HIP: {torch.version.hip}", flush=True)

    board = torch.randn(8192, 12, 8, 8, device=device)
    scalars = torch.randn(8192, 6, device=device)
    target = torch.randn(8192, device=device)

    bench("d256 eager+tuned fp16",     256, 8, 4, False, device, board, scalars, target)
    bench("d256 6L eager+tuned fp16",  256, 8, 6, False, device, board, scalars, target)

    print("\nDone!", flush=True)
