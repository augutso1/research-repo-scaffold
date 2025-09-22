from pathlib import Path
import os, sys, time, argparse, csv, json, subprocess, math, datetime, random
import yaml
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader

# ---------
# Utilities
# ---------
def get_git_hash():
    '''returns abbreviated hash of current git commit'''
    try:
        h = subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], stderr=subprocess.DEVNULL).decode().strip()
        return h
    except Exception:
        return "nogit"

def set_seed(seed: int, deterministic: bool = True):
    '''sets random seed for random number generation. And deterministic behaviour for PyTorch'''
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # cuDNN
    torch.backends.cudnn.deterministic = bool(deterministic)
    torch.backends.cudnn.benchmark = not bool(deterministic)
    torch.use_deterministic_algorithms(deterministic)

def device_from_config(cfg):
    '''sets torch.device based on the config'''
    dev = cfg["system"].get("device", "auto")
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif torch.backends.mps.is_available():
        return torch.device("mps")
    else:
        return torch.device("cpu")

# Verifies datetime and directory existance
def now_stamp():
    return datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)

# --------------------------------------
# Data (simple synthetic classification)
# --------------------------------------
def make_synthetic_split(n: int, n_features: int, n_classes: int, noise_std: float, seed: int, device):
    """
    Make a simple multi-class dataset:
    - Sample class centers from a fixed RNG (seed)
    - Each point = center + Gaussian noise
    """
    gen = torch.Generator(device='cpu').manual_seed(seed)
    # fixed centers in CPU for determinism
    centers = torch.randn(n_classes, n_features, generator=gen)
    # empty matrices
    x = torch.empty(n, n_features)
    y = torch.empty(n, dtype=torch.long)
    # distribute the classes (almost) evenly
    base = n // n_classes
    remainder = n % n_classes
    # generate synthetic data
    idx = 0
    for c in range(n_classes):
        k = base + (1 if c < remainder else 0)
        cx = centers[c]
        pts = cx + noise_std * torch.randn(k, n_features, generator=gen)
        x[idx:idx+k] = pts
        y[idx:idx+k] = c
        idx += k
    # shuffle deterministically to look more organic
    perm = torch.randperm(n, generator=gen)
    x = x[perm]
    y = y[perm]

    return x, y

def get_dataloaders(cfg, device):
    '''wraps the dataset into tensordatasets, config the seeding for reproducibility, returns dataloader objects for training and validations'''
    ds = cfg["dataset"]
    syscfg = cfg["system"]
    bs = cfg["train"]["batch_size"]
    nw = int(syscfg.get("num_workers", 0))
    seed = int(cfg["seed"])
    nfeat = int(ds["n_features"])
    ncls = int(ds["n_classes"])
    noise = float(ds["noise_std"])

    xtr, ytr = make_synthetic_split(ds["n_train"], nfeat, ncls, noise, seed=seed, device=device)
    xva, yva = make_synthetic_split(ds["n_val"],   nfeat, ncls, noise, seed=seed+1, device=device)

    train_ds = TensorDataset(xtr, ytr)
    val_ds   = TensorDataset(xva, yva)

    # Ensure dataloader workers are seeded deterministically
    def seed_worker(worker_id):
        '''seeding'''
        worker_seed = seed + worker_id
        np.random.seed(worker_seed)
        random.seed(worker_seed)

    g = torch.Generator()
    g.manual_seed(seed)

    train_loader = DataLoader(train_ds, batch_size=bs, shuffle=True, num_workers=nw, worker_init_fn=seed_worker, generator=g, drop_last=False)
    val_loader   = DataLoader(val_ds,   batch_size=bs, shuffle=False, num_workers=nw, worker_init_fn=seed_worker, generator=g, drop_last=False)
    return train_loader, val_loader

# ----------------------------
# Model
# ----------------------------
class MLP(nn.Module):
    def __init__(self, in_dim, hidden_sizes, out_dim, activation="relu"):
        super().__init__()
        act = nn.ReLU if activation.lower() == "relu" else nn.GELU
        layers = []
        prev = in_dim
        for h in hidden_sizes:
            layers += [nn.Linear(prev, h), act()]
            prev = h
        layers += [nn.Linear(prev, out_dim)]
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)

# ----------------------------
# Train / Eval
# ----------------------------
@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    total, correct, loss_sum = 0, 0, 0.0
    criterion = nn.CrossEntropyLoss()
    for xb, yb in loader:
        xb = xb.to(device)
        yb = yb.to(device)
        logits = model(xb.float())
        loss = criterion(logits, yb)
        loss_sum += loss.item() * yb.size(0)
        preds = logits.argmax(dim=1)
        correct += (preds == yb).sum().item()
        total += yb.size(0)
    return loss_sum / total, correct / total

def train_one_epoch(model, loader, optimizer, device, use_amp=False, scaler=None):
    model.train()
    total, correct, loss_sum = 0, 0, 0.0
    criterion = nn.CrossEntropyLoss()
    start = time.perf_counter()
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    for xb, yb in loader:
        xb = xb.to(device)
        yb = yb.to(device)
        optimizer.zero_grad(set_to_none=True)
        if use_amp:
            with torch.autocast(device_type="cuda", dtype=torch.float16 if torch.cuda.is_available() else torch.bfloat16, enabled=torch.cuda.is_available()):
                logits = model(xb.float())
                loss = criterion(logits, yb)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            logits = model(xb.float())
            loss = criterion(logits, yb)
            loss.backward()
            optimizer.step()
        loss_sum += loss.item() * yb.size(0)
        preds = logits.argmax(dim=1)
        correct += (preds == yb).sum().item()
        total += yb.size(0)
    elapsed = time.perf_counter() - start
    throughput = total / max(elapsed, 1e-8)
    vram_mb = 0.0
    if torch.cuda.is_available():
        vram_mb = torch.cuda.max_memory_allocated() / (1024**2)
    return loss_sum / total, correct / total, throughput, vram_mb, elapsed

# ----------------------------
# Main
# ----------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True, help="Path to YAML config")
    args = parser.parse_args()

    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)

    set_seed(int(cfg["seed"]), deterministic=bool(cfg["system"].get("deterministic", True)))
    device = device_from_config(cfg)

    # Build model
    in_dim = int(cfg["dataset"]["n_features"])
    out_dim = int(cfg["dataset"]["n_classes"])
    hidden = list(cfg["model"]["hidden_sizes"])
    act = cfg["model"].get("activation", "relu")
    model = MLP(in_dim, hidden, out_dim, activation=act).to(device)

    # Data
    train_loader, val_loader = get_dataloaders(cfg, device)

    # Optim
    opt = optim.Adam(model.parameters(), lr=float(cfg["train"]["lr"]), weight_decay=float(cfg["train"].get("weight_decay", 0.0)))

    # Precision / AMP
    precision = cfg["system"].get("precision", "fp32").lower()
    use_amp = (precision in ("fp16", "bf16")) and torch.cuda.is_available()
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    # Prepare run dir and logging
    stamp = now_stamp()
    run_name = f"{stamp}_{get_git_hash()}"
    out_dir = Path("runs") / run_name
    ensure_dir(out_dir)

    # Save a frozen config copy for provenance
    with open(out_dir / "config.yaml", "w") as f:
        yaml.safe_dump(cfg, f)

    # CSV logger
    log_path = out_dir / "logs.csv"
    with open(log_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["epoch", "split", "loss", "acc", "throughput", "vram_mb", "epoch_time_s"])
        writer.writeheader()

    best_val = float("inf")
    epochs = int(cfg["train"]["epochs"])

    for epoch in range(1, epochs + 1):
        tr_loss, tr_acc, tr_thr, tr_vram, tr_time = train_one_epoch(model, train_loader, opt, device, use_amp=use_amp, scaler=scaler)
        va_loss, va_acc = evaluate(model, val_loader, device)

        # Log to CSV
        with open(log_path, "a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["epoch", "split", "loss", "acc", "throughput", "vram_mb", "epoch_time_s"])
            writer.writerow({"epoch": epoch, "split": "train", "loss": tr_loss, "acc": tr_acc, "throughput": tr_thr, "vram_mb": tr_vram, "epoch_time_s": tr_time})
            writer.writerow({"epoch": epoch, "split": "val",   "loss": va_loss, "acc": va_acc, "throughput": "",      "vram_mb": "",       "epoch_time_s": ""})

        # Save checkpoints
        ckpt = {
            "epoch": epoch,
            "model_state": model.state_dict(),
            "optimizer_state": opt.state_dict(),
            "cfg": cfg,
            "val_loss": va_loss,
            "val_acc": va_acc,
        }
        torch.save(ckpt, out_dir / "last.pt")
        if va_loss < best_val:
            best_val = va_loss
            torch.save(ckpt, out_dir / "best.pt")

        # Console pretty print
        print(f"Epoch {epoch:02d} | train loss {tr_loss:.4f} acc {tr_acc:.3f} | val loss {va_loss:.4f} acc {va_acc:.3f} | thr {tr_thr:.1f} ex/s | vram {tr_vram:.1f} MB")

    # Summarize
    metrics = {"final_val_loss": va_loss, "final_val_acc": va_acc, "best_val_loss": best_val}
    with open(out_dir / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    print(f"\nSaved artifacts in: {out_dir}\n- logs.csv\n- config.yaml\n- last.pt / best.pt\n- metrics.json")

if __name__ == "__main__":
    main()
