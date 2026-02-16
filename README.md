# MLP Hyperparameter Transfer & Architecture Sensitivity on MNIST

This project investigates two questions using MLP networks trained on MNIST:

1. **Hyperparameter Transfer** — Do AdamW hyperparameters tuned on one MLP architecture transfer well to others?
2. **Architecture Sensitivity** — After proper tuning, do different MLP architectures achieve essentially the same final test loss?

The experiment runs in two phases:

- **Search** — Run per-architecture hyperparameter search, select best, and retrain with multiple seeds
- **Transfer** — Evaluate every (target, source) architecture pair using transferred hyperparameters and compute a regret matrix

## Project Structure

```
├── configs/
│   ├── defaults.yaml          # Global training defaults (batch size, patience, device, etc.)
│   ├── search_space.yaml      # AdamW hyperparameter ranges (lr, betas, eps, weight decay)
│   ├── architectures.yaml     # Architecture grid (layer widths, dropouts, leaky ReLU alphas)
│   ├── search.yaml            # Search phase config (num_trials, retrain seeds)
│   └── transfer.yaml          # Transfer config (seeds, path to search results)
├── src/
│   ├── data.py                # MNIST loading, train/val/test split, dataloaders
│   ├── model.py               # MLP model definition
│   ├── train.py               # Training loop with early stopping
│   ├── search.py              # Random hyperparameter sampling
│   ├── experiments.py         # Search and transfer orchestration
│   ├── reporting.py           # CSV/plot saving and statistics
│   └── utils.py               # Seeding, YAML I/O, device resolution
├── run_experiment.py          # CLI entry point
├── requirements.txt
└── runs/                      # Output artifacts (created at runtime)
```

## Setup

Requires Python 3.10+.

```bash
pip install -r requirements.txt
```

Dependencies: `torch`, `torchvision`, `pyyaml`, `matplotlib`, `numpy`.

MNIST is downloaded automatically to the `data/` directory on first run.

## Configuration

All experiments are driven by YAML config files. The entry point merges a **defaults** file with a **phase-specific** config (phase-specific values override defaults).

Key settings in `configs/defaults.yaml`:

| Setting | Default | Description |
|---|---|---|
| `device` | `cuda` | `cuda`, `cpu`, or `auto` |
| `training.batch_size` | `4096` | Mini-batch size |
| `training.max_epochs` | `1000` | Maximum training epochs |
| `training.patience` | `7` | Early stopping patience (validation checks) |
| `training.val_interval` | `4` | Epochs between validation evaluations |
| `data.preload` | `cuda` | Preload dataset to GPU (`cuda`, `cpu`, or `false`) |

The hyperparameter search space is defined in `configs/search_space.yaml` (log-uniform sampling for lr/eps/weight_decay, uniform for betas).

Architecture combinations are defined in `configs/architectures.yaml` as a grid of layer widths, dropout rates, and leaky ReLU slopes.

## Running Experiments

All phases are launched via `run_experiment.py` with a `--phase` flag and a phase-specific `--config` file.

### Search Phase — Per-Architecture Tuning & Retrain

Runs hyperparameter search across all architectures, selects the best hyperparameters for each, and retrains with multiple seeds.

```bash
python run_experiment.py \
  --phase search \
  --defaults configs/defaults.yaml \
  --search-space configs/search_space.yaml \
  --architectures configs/architectures.yaml \
  --config configs/search.yaml
```

Edit `configs/search.yaml` to configure:

- `search.num_trials` — trials per architecture (start small, e.g., 10-30, and adjust based on results)
- `retrain.seeds` — list of random seeds for retraining (default: `[1, 2, 3, 4, 5]`)
- `output_dir` — where to write results

**Testing tip**: To test the pipeline quickly, temporarily reduce the architecture grid in `configs/architectures.yaml` to just 1-2 architectures and use a small `num_trials` value (e.g., 2-5).

**Outputs** (in `output_dir/<arch_name>/`):

- `search_results.csv` — all trials with validation and test loss
- `best_hyperparameters.yaml` — optimal hyperparameters for this architecture
- `best_trial/metrics.csv` and `best_trial/loss_curve.png` — best trial training curve
- `retrain/seed_N/config.yaml` — per-seed retrain configuration and test loss
- `retrain/seed_losses.csv` — test loss for each seed
- `summary.yaml` — test loss statistics (mean, variance, min, max) across seeds

### Transfer Phase — Hyperparameter Transfer & Regret Analysis

Trains every architecture with every other architecture's best hyperparameters and computes a regret matrix. Automatically reuses search phase retrain results for diagonal entries (same source and target) to avoid redundant computation.

```bash
python run_experiment.py \
  --phase transfer \
  --defaults configs/defaults.yaml \
  --search-space configs/search_space.yaml \
  --architectures configs/architectures.yaml \
  --config configs/transfer.yaml
```

Edit `configs/transfer.yaml` to configure:

- `transfer.seeds` — random seeds for each (target, source) pair
- `transfer.search_dir` — path to search phase output directory (for loading best hyperparameters and reusing results)
- `output_dir` — where to write results

**Outputs** (in `output_dir`):

- `transfer_results.csv` — mean/var/min/max test loss for each (target, source) pair
- `transfer_seed_losses.csv` — per-seed test loss for every combination
- `regret_matrix.csv` — regret values: `mean_test_loss(transferred) - mean_test_loss(own)`
- `relative_regret_matrix.csv` — relative regret: `transferred / own`
- `transfer_test_loss.png` — heatmap of mean test loss across all architecture pairs
- `regret_matrix_mean.png` — heatmap of mean regret
- `relative_regret_matrix.png` — heatmap of relative regret

## Typical Workflow

```bash
# 1. Run per-architecture hyperparameter search (start with num_trials=10-30, adjust if needed)
python run_experiment.py --phase search --config configs/search.yaml

# 2. Run transfer experiment (set search_dir in transfer.yaml to match search output_dir)
python run_experiment.py --phase transfer --config configs/transfer.yaml
```

**For testing**: Before running the full experiment, test with a subset of architectures and fewer trials to verify everything works.

## Reproducibility

- All random seeds are explicitly logged in config artifacts
- Seed control covers `torch`, `numpy`, `random`, and dataloader workers
- Validation data is used only for early stopping and model selection; test data is evaluated only after training completes
- Set `deterministic: true` in defaults to enable `torch.backends.cudnn.deterministic` (may reduce performance)
