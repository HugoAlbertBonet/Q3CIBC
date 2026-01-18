# Q3CIBC

**Q3C for Imitation with Behavioral Cloning** – an offline imitation learning method that combines control-point generation with wire-fitting Q-value approximation and Implicit Behavioral Clonning (IBC).

---

## Overview

Q3CIBC learns a policy from offline expert demonstrations by:

1. **Control Point Generator** – produces multiple wire-fitting control points per state.
2. **Q-Estimator** – estimates the Q-values of each candidate action.
3. **Wire-Fitting Normalization** – smoothly interpolates Q-values using inverse-distance weighting, making the learned Q-function structurally maximizable.
4. **Implicit Behavioral Clonning** – instead of explicit models trained on MSE, we train Q3C as an implicit model, maximizing the probability of the expert's action.

---

## Project Structure

```
Q3CIBC/
├── main.py            # Training loop
├── models.py          # ControlPointGenerator & QEstimator networks
├── loss.py            # InfoNCE, MSE, and separation losses
├── normalizations.py  # Wire-fitting Q-value normalization
├── datasets.py        # D4RL dataset loader (via Minari)
├── d4rl.py            # (optional) D4RL utilization example
├── pyproject.toml     # Project metadata & dependencies
└── README.md
```

---

## Installation

Requires **Python ≥ 3.12**.

```bash
# Clone the repository
git clone https://github.com/HugoAlbertBonet/Q3CIBC.git
cd Q3CIBC

# Install dependencies (using uv, pip, or your preferred tool)
uv sync          # or: pip install -e .
```

---

## Usage

Run training on the default D4RL environment (`pen/human-v2`):

```bash
uv run main.py
# or
python main.py
```

### Configuration

Edit the constants at the top of `main.py`:

| Parameter        | Default   | Description                        |
|------------------|-----------|------------------------------------|
| `epochs`         | 100       | Number of training epochs          |
| `learning_rate`  | 1e-5      | AdamW learning rate                |
| `batch_size`     | 64        | Mini-batch size                    |
| `control_points` | 30        | Candidate actions per state        |

---

## Key Components

### `ControlPointGenerator`
Fully-connected network mapping states to `N` candidate action vectors.

### `QEstimator`
Fully-connected network mapping actions to scalar Q-values.

### `wireFittingNorm`
Vectorized wire-fitting normalization:

$$
Q(s,a) = \frac{\sum_i w_i(s,a)\,\hat{Q}_i(s)}{\sum_i w_i(s,a)}, \quad w_i(s,a) = \frac{1}{|a - \hat{a}_i(s)|^2 + c\,(\hat{Q}_{\max} - \hat{Q}_i(s))}
$$

### Losses
- **InfoNCE** – contrastive loss treating expert action as positive.
- **MSE** – distance from nearest control point to expert.
- **Separation** – encourages control-point diversity via inverse pairwise distances.

---

## Dependencies

- `torch >= 2.9`
- `minari[all] >= 0.5`
- `gymnasium-robotics >= 1.4`
- `numpy >= 2.4`

