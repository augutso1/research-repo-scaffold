# Reproducible Experiment Scaffold (Beginner-Friendly)

This is a tiny starter project that teaches you how to run ML experiments **reproducibly** (same setup → same results within tolerance).

### Quickstart

```bash
# 1) (Optional) Create and activate a virtual env
python -m venv .venv
# Windows: .venv\Scripts\activate
# macOS/Linux:
source .venv/bin/activate

# 2) Install deps
pip install -r requirements.txt

# 3) (Optional) Initialize git so the script logs your git hash
git init && git add . && git commit -m "init"

# 4) Run a baseline
python run.py --config configs/tiny_mlp.yaml

# 5) Re-run the SAME command; compare logs in ./runs/<timestamp>_<hash>/logs.csv
python run.py --config configs/tiny_mlp.yaml
```

### What you'll see
- A new folder under `runs/` for each execution.
- `config.yaml` copy, `logs.csv`, `metrics.json`, and checkpoints (`last.pt`, `best.pt`).
- Console output with epoch loss/acc, throughput, and (if on GPU) VRAM.

### How to test reproducibility
1. Run the same command twice.  
2. Open each run's `logs.csv` and compare the **final validation loss/accuracy**.  
3. They should match *exactly* on CPU; on some GPUs you may see tiny numeric drift. If drift is annoying:
   - Set `deterministic: true` in config (already true by default).
   - Avoid changing hardware/drivers between runs.
   - Try running on CPU to verify determinism first.

### Make a controlled change (non-reproducible on purpose)
- Change `train.batch_size` or `train.lr` in the config and re-run. Results should change—**that’s good.**
- Change only `seed`; results should change in *predictable* ways but be **repeatable** for that seed.

### Troubleshooting determinism
- DataLoader: we seed workers; keep `num_workers` low for now.
- CUDA: exact determinism can require environment flags; start on CPU if you want bitwise-stable repeats.
- If you use a GPU and see warnings, try:
  - `torch.use_deterministic_algorithms(True)` (not all ops supported)
  - Env var for cuBLAS determinism (varies by CUDA/toolkit)

### Project structure
```
repro_scaffold_starter/
├─ configs/
│  └─ tiny_mlp.yaml
├─ runs/                # outputs land here
├─ run.py               # main training script
├─ requirements.txt
└─ README.md
```

Have fun! Keep runs tidy and write down what changed between them. Small, disciplined experiments beat giant, messy ones.
