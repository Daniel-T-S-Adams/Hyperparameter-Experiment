# MLP Hyperparameter Transfer & Architecture Sensitivity  
## Experiment Plan (Agent Instructions)

---

## 0. Overview

This experiment answers two questions:

1. **Hyperparameter Transfer**  
   Do AdamW hyperparameters tuned on one MLP architecture transfer well to others?

2. **Architecture Sensitivity**  
   After proper tuning, do different MLP architectures achieve essentially the same final test loss on MNIST?

The experiment is split into two phases:

- **Phase A — Pre-Test Configuration**  
  Establish sensible hyperparameter ranges and tuning budgets using a single architecture.

- **Phase B — Full Experiment**  
  Run architecture-specific tuning, hyperparameter transfer, and regret analysis across all architectures.

All code written in Phase A must be reusable in Phase B.

---

## 1. Global Fixed Setup (Shared Across All Phases)

### Dataset
- Dataset: MNIST
- Input: flattened 28×28 images
- Fixed split:
  - Train: 50,000
  - Validation: 10,000
  - Test: 10,000

### Training Procedure (Fixed Defaults)
> These values **must be configurable**, but default to:

- Optimizer: AdamW
- Batch size: **256**
- Max epochs: **1000**
- Early stopping:
  - Metric: validation loss
  - Patience: **7 validation checks**
  - Note: Patience is counted in validation checks (not epochs), regardless of validation interval.
- Loss: cross-entropy
- Activations: Leaky ReLU
- Output: linear layer with 10 logits

**Important rule:**  
Validation data is used for *early stopping* and *hyperparameter selection only*.  
Test data is *never* used for early stopping, model selection, or tuning decisions.

---

## 2. Hyperparameter Search Space (AdamW)

Use **random search** over the following ranges.

### AdamW Hyperparameters

| Hyperparameter | Range | Sampling |
|----------------|------|----------|
| Learning rate (`lr`) | [1e-5, 3e-2] | log-uniform |
| Beta 1 (`β₁`) | [0.85, 0.99] | uniform |
| Beta 2 (`β₂`) | [0.9, 0.9999] | uniform |
| Epsilon (`ε`) | [1e-9, 1e-6] | log-uniform |
| Weight decay (`λ`) | [1e-6, 1e-2] | log-uniform |

---

## 3. Architecture Definition

Each **unique combination** below defines a distinct architecture.

### Architecture Dimensions
- Hidden layers:
  - 2 layers
  - 3 layers
- Hidden widths:
  - 2 layers:  
    - [256, 128]  
    - [192, 192]
  - 3 layers:  
    - [256, 128, 128]  
    - [192, 192, 128]
- Dropout probability (after each hidden layer):
  - {0.1, 0.2, 0.3}
- Leaky ReLU negative slope α:
  - {0.015, 0.045}

**Final experiment uses all combinations (2 two-layer × 2 three-layer).**

---

## 4. Phase A — Pre-Test Configuration

### Goal
- Determine a reasonable **hyperparameter trial budget**
- Validate:
  - training stability
  - early stopping behavior
  - artifact generation
- Reuse all code in Phase B

---

### A1. Select Pre-Test Architecture
Choose **one fixed architecture** (configurable), e.g.:
- 2 layers: [256, 128]
- Dropout: 0.2
- Leaky ReLU α = 0.015

---

### A2. Hyperparameter Random Search (Single Architecture)

#### Procedure
1. Fix:
   - architecture
   - dataset split
   - random seed = **1**
2. Run random search for a **configurable number of trials**
3. For each trial:
   - Train model with early stopping **using validation loss only**
   - Track per-epoch:
     - training loss
     - validation loss
4. Select the trial with **minimum validation loss**

**Test data is not evaluated during hyperparameter search.**

---

### A3. Artifacts to Save (Per Trial)

#### CSV file
- Columns:
  - epoch
  - training_loss
  - validation_loss
- Metadata (in header or separate row):
  - hyperparameter values
  - best validation loss
  - epoch of best validation loss
  - terminating epoch (early stopping)

#### Config file (recommended)
- `config.json` (or `yaml`) per trial with:
  - architecture definition
  - hyperparameters
  - random seeds
  - dataset split info

#### Plot
- Training curve:
  - x-axis: epoch
  - y-axis: loss
  - lines:
    - training loss
    - validation loss

---

### A4. Pre-Test Outputs

- All trial CSVs and plots
- Summary:
  - best hyperparameters
  - number of trials required for convergence
- Decision:
  - choose trial budget for Phase B using
    running best-validation-loss vs. trials to detect diminishing returns

---

## 5. Phase B — Full Experiment

---

## 6. Architecture-Specific Hyperparameter Optimisation

### For each architecture `a ∈ A`:

1. Fix:
   - architecture `a`
   - dataset split
   - random seed = **1**
2. Run random search with budget determined in Phase A
3. Select optimal hyperparameters:
   - `h*_a = argmin(validation loss)`
4. Retrain architecture `a` from scratch with `h*_a` and early stopping
   - Use the **best-validation checkpoint** (not the last epoch)
   - Repeat for **5 random seeds**
5. Evaluate **final test loss** on the test dataset for each seed
6. Save:
   - CSV + training curve plot for **best trial only**
   - best validation loss
   - test loss: mean / variance / min / max across seeds

---

## 7. Hyperparameter Transfer Experiment

### For each ordered pair of architectures `(a, a′)`:

1. Load optimal hyperparameters `h*_{a′}`
2. Train architecture `a` using `h*_{a′}`
3. Use validation loss for early stopping
4. After training completes, evaluate **test loss**
5. Repeat steps 2–4 with **5 random seeds**
   - Seeds control:
     - weight initialization
     - mini-batch order
     - dropout randomness
6. Compute test-loss statistics:
   - mean
   - variance
   - min / max

---

## 8. Regret Computation

For each architecture pair `(a, a′)`:

```
Regret R(a, a′) =
  mean_test_loss(a trained with h*_{a′})
  − mean_test_loss(a trained with h*_a)
```

---

## 9. Final Outputs

### 9.1 Per-Architecture Outputs
- Best-trial CSV (training + validation only)
- Training curve plot
- Optimal hyperparameter vector
- Final test loss

### 9.2 Transfer Outputs
- Regret matrix:
  - rows = evaluated architecture `a`
  - columns = source hyperparameters `a′`
- For each matrix entry:
  - mean regret
  - variance
  - min / max across seeds
- Saved as:
  - CSV
  - optional heatmap plot

---

## 10. Architecture Sensitivity Analysis

- Compare optimal test losses `{ L̂_a(h*_a) }`
- Report:
  - mean
  - variance
  - range across architectures
- Interpretation:
  - small spread ⇒ architecture differences are not materially important on MNIST

---

## 11. Reproducibility Requirements

- All configs must be serializable
- Random seeds explicitly logged
- Seed control covers:
  - `torch`, `numpy`, and `random`
  - data loader worker seeds
  - dropout randomness
  - backend determinism flags (if used)
- Clear separation between:
  - training data
  - validation data (model selection)
  - test data (final evaluation only)
- Directory structure must allow:
  - Phase A → Phase B reuse
  - per-architecture isolation
  - per-seed tracking
