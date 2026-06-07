"""Inference-time benchmark for Q3CIBC vs IBC paper inference methods on
**Pushing single target, IMAGE observations**.

Sibling of bench_inference.py — same comparison principle (random weights,
forward-pass cost only, architectures sized to the published configs), but
the obs is now a (3*frame_stack=6, H=240, W=320) uint8 image instead of a
20-D state vector.

Methods benchmarked
-------------------
1. **Q3CIBC + CP-DFO (stack recipe)** — our best multi-seed-robust pixel
   recipe (5-seed mean 0.948, std 0.015):
     - PixelControlPointGenerator (ConvMaxpoolEncoder + MLP head 256×2)
       → 20 CPs.
     - PixelQEstimator (ConvMaxpoolEncoder + DenseResnetValue 1024×1).
     - DFO refinement: 10 iters, std=0.005, decay=0.5, +16 uniform safety
       samples. Late fusion: encoder runs ONCE per env step, all DFO
       scoring uses cached 256-D features.

2. **IBC + DFO (paper Simulated Pushing, Pixels, EBM)** — paper-faithful
   per Florence et al. 2021 (Implicit BC), Appendix D.2 hyperparameter
   table for "Simulated Pushing, Pixels, Implicit BC (EBM)":
     - EBM variant: **DFO** (NOT Langevin — paper uses Langevin for states,
       DFO for pixels).
     - Conv. Net: 4-layer ConvMaxPool → 256-D features.
     - MLP network: 1024×4 (4 dense layers at width 1024, regular dense
       layers — NOT ResNetDenseBlock).
     - Activation: ReLU.
     - DFO samples: 4096.
     - DFO iterations: 3.
     - Action boundary buffer: 0.05.
     - Image size: 240×180.
     - Late fusion: encoder once per env step, DFO inner loop on cached
       256-D features.

Random weights note
-------------------
The user asked to confirm: inference wall-clock depends on tensor shapes
and the algorithm graph, NOT on weight values. Random vs trained weights
produce identical FLOPs/second. The only practical impact of random
weights would be if the algorithm had data-dependent early-exit branches
(neither method here does) or if numerical degeneracy stalled cuDNN's
algorithm picks (negligible after warmup).

Success-rate sources
--------------------
* Q3C row uses our 5-seed mean from the pushingPixelsG/H stack recipe
  trials (trials.jsonl: G5, G6, H1, H2, H3 — std=0.005, 16 unif, 10 iters,
  decay=0.5, 150k + LR=3e-4).
* IBC row uses values from Florence et al. 2021 (Implicit BC) Table 3
  "Block Pushing — Pixels (EBM, DFO)" row: 100% ± 0% across 3 seeds.
  NOT measured here.

CSV output
----------
results/hyperparam_search/combinedv2_cpascounter_training/pushing_pixels/single_target_pixels.csv

Columns: method, success_rate_pct, success_rate_std_pct, num_seeds, inference_time_ms

Usage
-----
    uv run --managed-python --extra pushing python bench_inference_pixels.py \
        --num-steps 50 --warmup 5 --device auto
"""

from __future__ import annotations

import argparse
import csv
import statistics
import time
from contextlib import contextmanager
from pathlib import Path

import torch

from torch import nn

from utils.models import (
    PixelControlPointGenerator,
    PixelQEstimator,
    ConvMaxpoolEncoder,
    DenseResnetValue,
)


RESULTS_DIR = (
    Path(__file__).parent
    / "results"
    / "hyperparam_search"
    / "combinedv2_cpascounter_training"
    / "pushing_pixels"
)
CSV_PATH = RESULTS_DIR / "single_target_pixels.csv"


# Task constants (matches PushingPixelsEnv + PushingPixelsDataset).
FRAME_STACK = 2
IMAGE_CHANNELS = 3 * FRAME_STACK   # 6
IMAGE_HEIGHT = 240                  # env native
IMAGE_WIDTH = 320                   # env native
ENCODER_TARGET_HEIGHT = 180         # IBC pixel_ebm_langevin.gin
ENCODER_TARGET_WIDTH = 240          # IBC pixel_ebm_langevin.gin
ACTION_DIM = 2
ACTION_MIN = -1.0
ACTION_MAX = 1.0


# ─── Success-rate constants ───────────────────────────────────────────────
# Q3C stack 5-seed mean (G5, G6, H1, H2, H3 in
# results/.../pushing_pixels/trials.jsonl):
#   0.93, 0.94, 0.96, 0.94, 0.97 → mean 0.948, std 0.015
# 100 eval episodes/seed × 5 seeds.
Q3C_STACK_SR_MEAN = 0.948
Q3C_STACK_SR_STD = 0.015
Q3C_STACK_SEEDS = 5

# IBC paper Block Pushing — Pixels (Implicit BC EBM, DFO inference).
# Per Florence et al. 2021 Table 3: 100% ± 0% across 3 seeds.
# NOT from this repo's experiments.
IBC_SR_MEAN = 1.00
IBC_SR_STD = 0.00
IBC_SEEDS = 3


# ─── Timing harness ───────────────────────────────────────────────────────

@contextmanager
def cuda_sync(device: torch.device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    yield
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def time_block(fn, num_steps: int, warmup: int, device: torch.device) -> dict:
    for _ in range(warmup):
        fn()
    times: list[float] = []
    for _ in range(num_steps):
        with cuda_sync(device):
            t0 = time.perf_counter()
            fn()
            with cuda_sync(device):
                pass
            t1 = time.perf_counter()
        times.append(t1 - t0)
    return {
        "mean_ms": 1000.0 * statistics.mean(times),
        "median_ms": 1000.0 * statistics.median(times),
        "stdev_ms": 1000.0 * (statistics.stdev(times) if len(times) > 1 else 0.0),
        "min_ms": 1000.0 * min(times),
        "max_ms": 1000.0 * max(times),
        "n": len(times),
    }


# ─── Method 1: Q3C + CP-DFO stack recipe ──────────────────────────────────

def make_q3c_stack(device: torch.device):
    """Stack recipe: encoder once → 20 CPs + 16 uniform → 10 DFO iters."""
    cp_gen = PixelControlPointGenerator(
        output_dim=ACTION_DIM,
        control_points=20,
        hidden_dims=[256, 256],
        action_bounds=(ACTION_MIN, ACTION_MAX),
        network_kind="mlp",
        in_channels=IMAGE_CHANNELS,
        encoder_target_height=ENCODER_TARGET_HEIGHT,
        encoder_target_width=ENCODER_TARGET_WIDTH,
    ).to(device).eval()

    q_net = PixelQEstimator(
        action_dim=ACTION_DIM,
        in_channels=IMAGE_CHANNELS,
        encoder_target_height=ENCODER_TARGET_HEIGHT,
        encoder_target_width=ENCODER_TARGET_WIDTH,
        value_width=1024,
        value_num_blocks=1,
    ).to(device).eval()

    # Stack recipe hyperparams
    dfo_iters = 10
    dfo_std = 0.005
    dfo_decay = 0.5
    n_uniform = 16

    def select_action():
        # Random uint8 image batch (1, C, H, W) — matches PushingPixelsEnv
        # obs shape after PushingPixelsSimulation._obs_to_tensor.
        obs = torch.randint(0, 256, (1, IMAGE_CHANNELS, IMAGE_HEIGHT, IMAGE_WIDTH),
                            dtype=torch.uint8, device=device)

        with torch.no_grad():
            # Encode once (cached for entire DFO loop — this is the win)
            features = q_net.encode(obs)        # (1, 256)
            cps = cp_gen(obs)                    # (1, 20, 2)

            # Mix uniform safety samples into candidate pool
            unif = torch.empty(1, n_uniform, ACTION_DIM, device=device).uniform_(ACTION_MIN, ACTION_MAX)
            candidates = torch.cat([cps, unif], dim=1)  # (1, 36, 2)
            N = candidates.shape[1]

            std = dfo_std
            log_probs = None
            for it in range(dfo_iters):
                # All scoring uses cached features — encoder NOT re-run.
                log_probs = q_net.score(features, candidates).squeeze(-1)  # (1, N)
                probs = torch.softmax(log_probs.squeeze(0), dim=-1)
                idx = torch.multinomial(probs, N, replacement=True)
                counts = torch.bincount(idx, minlength=N)
                repeat_idx = torch.repeat_interleave(
                    torch.arange(N, device=device), counts
                )
                candidates = candidates[:, repeat_idx, :]
                if it < dfo_iters - 1:
                    candidates = candidates + torch.randn_like(candidates) * std
                    candidates = candidates.clamp(ACTION_MIN, ACTION_MAX)
                    std *= dfo_decay
            # Re-score and pick best on the final reordered candidates
            final = q_net.score(features, candidates).squeeze(-1)
            sel = final.argmax(dim=1)
            return candidates[0, sel[0], :]

    return "q3c", select_action, (cp_gen, q_net)


# ─── Method 2: IBC + DFO (paper Simulated Pushing Pixels EBM) ─────────────

class _IBCPixelEBM(nn.Module):
    """IBC paper architecture for Simulated Pushing, Pixels, EBM.

    Per Florence et al. 2021 Appendix D.2:
      - Conv. Net: 4-layer ConvMaxPool (same as our ConvMaxpoolEncoder).
      - MLP network: 1024×4 (4 dense layers width 1024 — REGULAR dense,
        not ResNet blocks).
      - Activation: ReLU.
      - Late fusion: encode image once, MLP processes [features, action].
    """

    def __init__(self, in_channels: int, target_height: int, target_width: int,
                 feature_dim: int = 256, mlp_width: int = 1024, mlp_depth: int = 4):
        super().__init__()
        self.encoder = ConvMaxpoolEncoder(
            in_channels=in_channels,
            target_height=target_height,
            target_width=target_width,
            feature_dim=feature_dim,
        )
        in_dim = feature_dim + ACTION_DIM
        layers = []
        prev = in_dim
        for _ in range(mlp_depth):
            layers.append(nn.Linear(prev, mlp_width))
            layers.append(nn.ReLU(inplace=True))
            prev = mlp_width
        layers.append(nn.Linear(mlp_width, 1))
        self.mlp = nn.Sequential(*layers)
        self.feature_dim = feature_dim

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x)

    def score(self, features: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        """features: (B, F).  action: (B, N, A) or (B, A).  Returns (B, N, 1) or (B, 1)."""
        if action.ndim == 3:
            B, N, _ = action.shape
            features = features.unsqueeze(1).expand(B, N, -1)
        x = torch.cat([features, action], dim=-1)
        return self.mlp(x)


def make_ibc_dfo(device: torch.device):
    """IBC PixelEBM + DFO inference (paper Pushing Pixels EBM hyperparameters).

    Per Florence et al. 2021 Appendix D.2 "Simulated Pushing, Pixels, EBM":
      - Architecture: ConvMaxpoolEncoder + 4-layer MLP (1024×4, regular dense).
      - DFO samples: 4096.
      - DFO iterations: 3.
      - Action boundary buffer: 0.05.
    Late fusion: encoder runs once per env step; DFO inner loop on cached features.
    """
    q_net = _IBCPixelEBM(
        in_channels=IMAGE_CHANNELS,
        target_height=ENCODER_TARGET_HEIGHT,
        target_width=ENCODER_TARGET_WIDTH,
        feature_dim=256,
        mlp_width=1024,
        mlp_depth=4,
    ).to(device).eval()

    num_samples = 4096
    num_iters = 3
    boundary_buffer = 0.05
    iteration_std = 0.33  # std init for iterative_dfo (paper doesn't specify;
                          # 0.33 matches IBC state-task default in mcmc.iterative_dfo)

    def select_action():
        obs = torch.randint(0, 256, (1, IMAGE_CHANNELS, IMAGE_HEIGHT, IMAGE_WIDTH),
                            dtype=torch.uint8, device=device)
        with torch.no_grad():
            features = q_net.encode(obs)  # (1, 256) — cached for entire DFO loop

            # Initial uniform sample with 5% boundary buffer (paper-faithful).
            buf = (ACTION_MAX - ACTION_MIN) * boundary_buffer
            sample_min = ACTION_MIN - buf
            sample_max = ACTION_MAX + buf
            actions = torch.empty(1, num_samples, ACTION_DIM, device=device).uniform_(sample_min, sample_max)

            std = iteration_std
            log_probs = None
            for it in range(num_iters):
                log_probs = q_net.score(features, actions).squeeze(-1)  # (1, N)
                probs = torch.softmax(log_probs.squeeze(0), dim=-1)
                idx = torch.multinomial(probs, num_samples, replacement=True)
                counts = torch.bincount(idx, minlength=num_samples)
                repeat_idx = torch.repeat_interleave(
                    torch.arange(num_samples, device=device), counts
                )
                actions = actions[:, repeat_idx, :]
                if it < num_iters - 1:
                    actions = actions + torch.randn_like(actions) * std
                    actions = actions.clamp(ACTION_MIN, ACTION_MAX)
                    std *= 0.5
            sel = log_probs.argmax(dim=1)
            return actions[0, sel[0], :]

    return "ibc", select_action, (q_net,)


# ─── Driver ───────────────────────────────────────────────────────────────

def param_count(*modules: torch.nn.Module) -> int:
    return sum(p.numel() for m in modules for p in m.parameters())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--num-steps", type=int, default=30,
                        help="Timed calls per method (default 30).")
    parser.add_argument("--warmup", type=int, default=5,
                        help="Warmup calls per method (discarded).")
    parser.add_argument("--device", type=str, default="auto",
                        choices=["auto", "cpu", "cuda"])
    args = parser.parse_args()

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    print(f"Device: {device}")
    if device.type == "cuda":
        print(f"  {torch.cuda.get_device_name(device)}")
    print(f"Image shape: ({IMAGE_CHANNELS}, {IMAGE_HEIGHT}, {IMAGE_WIDTH}) uint8")
    print(f"Encoder target: {ENCODER_TARGET_HEIGHT}×{ENCODER_TARGET_WIDTH}")
    print(f"Action dim: {ACTION_DIM}")
    print(f"Timed steps: {args.num_steps}  (warmup={args.warmup})")
    print()

    rows = []  # (method, sr_mean_pct, sr_std_pct, num_seeds, scoring_evals, timings_dict)
    # Scoring evals = number of (state, action) → Q forward passes per env step.
    # Q3C stack: 10 DFO iters (each scores 36 candidates) + 1 final re-score = 11×36 = 396.
    # IBC DFO: 3 iters × 4096 samples = 12,288. No final re-score (IBC convention).
    Q3C_SCORING_EVALS = (10 + 1) * (20 + 16)  # 396
    IBC_SCORING_EVALS = 3 * 4096               # 12288

    print("=" * 78)
    print("Method 1: Q3CIBC + CP-DFO (stack recipe)")
    print("  20 CPs + 16 uniform safety, 10 iters, std=0.005, decay=0.5")
    print("  Late fusion: encoder runs ONCE per env step")
    method, fn, modules = make_q3c_stack(device)
    n_params = param_count(*modules)
    print(f"  Param count: {n_params:,}")
    t = time_block(fn, args.num_steps, args.warmup, device)
    print(f"  Mean:    {t['mean_ms']:7.3f} ms  (median {t['median_ms']:7.3f} ms, std {t['stdev_ms']:6.3f} ms)")
    print(f"  Range:   [{t['min_ms']:7.3f}, {t['max_ms']:7.3f}] ms over {t['n']} runs")
    print(f"  SR:      {Q3C_STACK_SR_MEAN*100:.1f}% ± {Q3C_STACK_SR_STD*100:.1f}% over {Q3C_STACK_SEEDS} seeds (from G5/G6/H1/H2/H3 stack)")
    rows.append(("q3c", Q3C_STACK_SR_MEAN * 100, Q3C_STACK_SR_STD * 100, Q3C_STACK_SEEDS, Q3C_SCORING_EVALS, t))
    print()

    print("=" * 78)
    print("Method 2: IBC + DFO (paper Simulated Pushing Pixels EBM, Appendix D.2)")
    print("  ConvMaxPool encoder + 1024×4 regular MLP value net")
    print("  DFO: 4096 samples × 3 iters, action boundary buffer=0.05, late fusion")
    method, fn, modules = make_ibc_dfo(device)
    n_params = param_count(*modules)
    print(f"  Param count: {n_params:,}")
    t = time_block(fn, args.num_steps, args.warmup, device)
    print(f"  Mean:    {t['mean_ms']:7.3f} ms  (median {t['median_ms']:7.3f} ms, std {t['stdev_ms']:6.3f} ms)")
    print(f"  Range:   [{t['min_ms']:7.3f}, {t['max_ms']:7.3f}] ms over {t['n']} runs")
    print(f"  SR:      {IBC_SR_MEAN*100:.1f}% ± {IBC_SR_STD*100:.1f}% over {IBC_SEEDS} seeds (from Florence et al. 2021 Table 3, NOT measured here)")
    rows.append(("ibc", IBC_SR_MEAN * 100, IBC_SR_STD * 100, IBC_SEEDS, IBC_SCORING_EVALS, t))
    print()

    # ─── CSV write ───────────────────────────────────────────────────────
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    with open(CSV_PATH, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "method",
            "success_rate_pct",
            "success_rate_std_pct",
            "num_seeds",
            "scoring_evals",
            "inference_time_ms",
        ])
        for method, sr_mean, sr_std, seeds, scoring_evals, timings in rows:
            writer.writerow([
                method,
                f"{sr_mean:.2f}",
                f"{sr_std:.2f}",
                seeds,
                scoring_evals,
                f"{timings['mean_ms']:.3f}",
            ])
    print("=" * 78)
    print(f"CSV written: {CSV_PATH}")
    print()
    # Echo table
    with open(CSV_PATH) as f:
        for line in f:
            print(f"  {line.rstrip()}")


if __name__ == "__main__":
    main()
