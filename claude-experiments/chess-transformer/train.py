"""
Training: CNN 6x64, streaming bitboard data, hipBLASLt + TunableOp.
With warmup + cosine LR schedule and Huber loss.
"""

import os
os.environ["ROCBLAS_USE_HIPBLASLT"] = "1"
os.environ["PYTORCH_TUNABLEOP_ENABLED"] = "1"
os.environ["TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL"] = "1"

import sys
import math
import time
import torch
import torch.nn as nn
from model import MoveTimeModel
from dataset import StreamingBitboardDataset

LOG_FILE = "data/progress.log"

def log(msg):
    print(msg, flush=True)
    with open(LOG_FILE, "a") as f:
        f.write(msg + "\n")


def train():
    open(LOG_FILE, "w").close()

    torch.set_float32_matmul_precision('high')
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log(f"Using device: {device}")
    if device.type == "cuda":
        log(f"GPU: {torch.cuda.get_device_name(0)}")
        log(f"PyTorch: {torch.__version__}, HIP: {torch.version.hip}")

    data_path = sys.argv[1] if len(sys.argv) > 1 else "data/train.bin"
    epochs = int(sys.argv[2]) if len(sys.argv) > 2 else 3

    dataset = StreamingBitboardDataset(data_path, batch_size=8192)

    # Model
    model = MoveTimeModel(cnn_channels=64, n_blocks=6).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    log(f"Model: CNN 6x64, params={n_params:,}")
    log(f"{epochs} epoch(s), {dataset.n_train_batches} batches/epoch")

    # LR schedule: warmup + cosine decay
    total_steps = dataset.n_train_batches * epochs
    warmup_steps = int(0.03 * total_steps)
    peak_lr = 5e-4

    optimizer = torch.optim.AdamW(model.parameters(), lr=peak_lr, weight_decay=0.01)

    def lr_lambda(step):
        if step < warmup_steps:
            return step / max(1, warmup_steps)
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return 0.01 + 0.99 * 0.5 * (1 + math.cos(math.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    criterion = nn.HuberLoss(delta=1.0)
    scaler = torch.amp.GradScaler()

    log(f"LR: {peak_lr} peak, {warmup_steps} warmup steps, {total_steps} total steps")
    log(f"Loss: HuberLoss(delta=1.0)")
    log(f"hipBLASLt + TunableOp + fp16")

    best_val_loss = float('inf')

    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        n_batches = 0
        t0 = time.time()

        for board, scalars, target in dataset.train_batches(device):
            with torch.amp.autocast('cuda', dtype=torch.float16):
                pred = model(board, scalars)
                loss = criterion(pred, target)

            optimizer.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()

            train_loss += loss.item()
            n_batches += 1

            if n_batches % 500 == 0:
                elapsed = time.time() - t0
                avg_loss = train_loss / n_batches
                ms_per_batch = elapsed / n_batches * 1000
                eta = (dataset.n_train_batches - n_batches) * elapsed / n_batches
                lr = optimizer.param_groups[0]['lr']
                log(f"  batch {n_batches}/{dataset.n_train_batches}, "
                    f"loss={loss.item():.4f}, avg={avg_loss:.4f}, "
                    f"{ms_per_batch:.0f}ms/batch, "
                    f"lr={lr:.2e}, "
                    f"eta={eta/60:.1f}m")

        train_loss /= n_batches
        train_time = time.time() - t0

        # Validate
        model.eval()
        val_loss = 0.0
        val_mse = 0.0
        val_batches = 0
        mse_criterion = nn.MSELoss()
        with torch.no_grad():
            for board, scalars, target in dataset.val_batches(device):
                with torch.amp.autocast('cuda', dtype=torch.float16):
                    pred = model(board, scalars)
                    loss = criterion(pred, target)
                    mse = mse_criterion(pred, target)
                val_loss += loss.item()
                val_mse += mse.item()
                val_batches += 1

        val_loss /= val_batches
        val_mse /= val_batches
        approx_factor = torch.tensor(val_mse).sqrt().exp().item()
        lr = optimizer.param_groups[0]['lr']
        log(f"Epoch {epoch+1}/{epochs} | "
            f"Train Huber: {train_loss:.4f} | Val Huber: {val_loss:.4f} | "
            f"Val MSE: {val_mse:.4f} | ~{approx_factor:.2f}x | "
            f"LR: {lr:.2e} | Time: {train_time/60:.1f}m")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), "data/best_model_cnn.pt")
            log(f"  -> Saved!")

    log(f"\nBest val Huber loss: {best_val_loss:.4f}")


if __name__ == "__main__":
    train()
