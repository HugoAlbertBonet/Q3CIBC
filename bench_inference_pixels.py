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


# ─── Q3C success-rate constants per iter count (best config each) ────────
# All from results/.../pushing_pixels/trials.jsonl. SR % expressed as
# (mean, std, n_seeds).
#
# 0 iters — I0A: 200k training + 30 CPs, pure argmax (no DFO).
#   Trials #99 (seed=0=0.98), #100 (seed=1=0.93), #98 (seed=2=0.91).
Q3C_0ITER  = (0.940, 0.030, 3)
# 3 iters — J3: 200k+30CPs base, std=0.02, decay=0.5, 16 uniform safety.
#   Trials #133 (seed=0=0.97), #134 (seed=1=0.90), #135 (seed=2=0.95).
Q3C_3ITER  = (0.940, 0.029, 3)
# 5 iters — J5 ⭐: 200k+30CPs base, std=0.005, decay=0.5, 24 uniform safety.
#   Trials #136 (seed=0=0.98), #137 (seed=1=0.94), #138 (seed=2=0.95).
Q3C_5ITER  = (0.957, 0.017, 3)
# 10 iters — Stack: 150k+20CPs base, std=0.005, decay=0.5, 16 uniform safety.
#   Trials G5/G6/H1/H2/H3 (seeds 0/1/2/3/4 = 0.93/0.94/0.96/0.94/0.97).
Q3C_10ITER = (0.948, 0.015, 5)

# IBC paper Block Pushing — Pixels (Implicit BC EBM, DFO inference).
# Per Florence et al. 2021 Table 3: 100% ± 0% across 3 seeds.
# NOT from this repo's experiments.
IBC_SR_MEAN = 1.00
IBC_SR_STD = 0.00
IBC_SEEDS = 3

# IBC paper Block Pushing — Pixels (MDN-BC, Appendix D.2 / Table 3).
# 10.0% ± 4.3% across 3 seeds. MDN is the weakest pixel baseline IBC reports.
MDN_SR_MEAN = 0.100
MDN_SR_STD = 0.043
MDN_SEEDS = 3

# IBC paper Block Pushing — Pixels (MSE-BC, Appendix D.2 / Table 3).
# 87.0% ± 4.1% across 3 seeds.
MSE_SR_MEAN = 0.870
MSE_SR_STD = 0.041
MSE_SEEDS = 3


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


# ─── Method 1: Q3C + CP-DFO (parameterized by best-config per iter count) ─

def make_q3c(device: torch.device, control_points: int, n_uniform: int,
             dfo_iters: int, dfo_std: float, dfo_decay: float, method_name: str):
    """Q3C inference: encoder once → CPs (+ uniform safety) → DFO loop.

    For dfo_iters=0 → pure CP argmax (skip DFO loop).
    For dfo_iters>0 → resample + jitter inner loop on cached features.
    """
    cp_gen = PixelControlPointGenerator(
        output_dim=ACTION_DIM,
        control_points=control_points,
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

    def select_action():
        obs = torch.randint(0, 256, (1, IMAGE_CHANNELS, IMAGE_HEIGHT, IMAGE_WIDTH),
                            dtype=torch.uint8, device=device)

        with torch.no_grad():
            features = q_net.encode(obs)        # (1, 256)
            cps = cp_gen(obs)                    # (1, control_points, 2)

            if n_uniform > 0:
                unif = torch.empty(1, n_uniform, ACTION_DIM, device=device).uniform_(ACTION_MIN, ACTION_MAX)
                candidates = torch.cat([cps, unif], dim=1)
            else:
                candidates = cps
            N = candidates.shape[1]

            if dfo_iters == 0:
                # Pure argmax over candidates — no DFO.
                log_probs = q_net.score(features, candidates).squeeze(-1)  # (1, N)
                sel = log_probs.argmax(dim=1)
                return candidates[0, sel[0], :]

            std = dfo_std
            log_probs = None
            for it in range(dfo_iters):
                log_probs = q_net.score(features, candidates).squeeze(-1)
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
            final = q_net.score(features, candidates).squeeze(-1)
            sel = final.argmax(dim=1)
            return candidates[0, sel[0], :]

    return method_name, select_action, (cp_gen, q_net)


# Backward-compat shim: stack = 10-iter variant
def make_q3c_stack(device: torch.device):
    return make_q3c(device, control_points=20, n_uniform=16,
                    dfo_iters=10, dfo_std=0.005, dfo_decay=0.5,
                    method_name="q3c_10iter")


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


# ─── Method 3: IBC + MDN-BC (paper Simulated Pushing Pixels MDN) ──────────

class _CoordConvEncoder(nn.Module):
    """Standard 4-layer ConvMaxpool encoder with CoordConv channel prepend.

    IBC paper Appendix D.2 marks `coord conv = True` for the MSE-BC pixels
    config. Adds (x, y) coordinate channels to the input image before the
    first conv. Coords are linspace [0, 1] over H and W, broadcast to (B, H, W).
    Underlying encoder is built with `in_channels + 2`.
    """

    def __init__(self, in_channels: int, target_height: int, target_width: int,
                 feature_dim: int = 256):
        super().__init__()
        self.encoder = ConvMaxpoolEncoder(
            in_channels=in_channels + 2,
            target_height=target_height,
            target_width=target_width,
            feature_dim=feature_dim,
        )

    def _add_coords(self, x: torch.Tensor) -> torch.Tensor:
        # Match ConvMaxpoolEncoder's preprocessing: uint8 → float in [0, 1].
        if x.dtype == torch.uint8:
            x = x.float() / 255.0
        elif x.max() > 1.5:
            x = x / 255.0
        B, C, H, W = x.shape
        ys = torch.linspace(0.0, 1.0, H, device=x.device).view(1, 1, H, 1).expand(B, 1, H, W)
        xs = torch.linspace(0.0, 1.0, W, device=x.device).view(1, 1, 1, W).expand(B, 1, H, W)
        return torch.cat([x, xs, ys], dim=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self._add_coords(x)
        # Now float in [0, 1]; encoder's normalization is a no-op (max ≤ 1).
        return self.encoder(x)


class _IBCPixelMDN(nn.Module):
    """IBC paper Simulated Pushing, Pixels, MDN-BC.

    Per Florence et al. 2021 Appendix D.2:
      - Image size: 120×90 (smaller than EBM's 240×180).
      - Conv Net: 4-layer ConvMaxPool.
      - MLP: 512×8 (width × depth = 8 dense layers).
      - Num components: 26 Gaussian mixture components.
      - Dropout 0.1 (MLP only).
      - Activation: ReLU.
      - Test temperature 2.0, test variance exponent 4.0 (consumed by sampling;
        for argmax-mode inference the temperature only affects mixing weights).

    Inference path (paper-faithful argmax-mode):
      1. Encoder → 256-D features.
      2. MLP → (K, 2·A + 1) outputs per sample.
      3. Mixing logits → argmax → pick best component.
      4. Return that component's mean (tanh-bounded).
    """

    def __init__(self, in_channels: int, target_height: int, target_width: int,
                 mlp_width: int = 512, mlp_depth: int = 8, n_components: int = 26,
                 dropout: float = 0.1):
        super().__init__()
        self.encoder = ConvMaxpoolEncoder(
            in_channels=in_channels,
            target_height=target_height,
            target_width=target_width,
            feature_dim=256,
        )
        self.n_components = n_components
        # Per component: mean (A), log_std (A), mixing logit (1).
        self.out_dim = n_components * (2 * ACTION_DIM + 1)
        layers = []
        prev = 256
        for _ in range(mlp_depth):
            layers.append(nn.Linear(prev, mlp_width))
            layers.append(nn.ReLU(inplace=True))
            layers.append(nn.Dropout(dropout))
            prev = mlp_width
        layers.append(nn.Linear(mlp_width, self.out_dim))
        self.mlp = nn.Sequential(*layers)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        feats = self.encoder(obs)                       # (B, 256)
        out = self.mlp(feats)                           # (B, K·(2A+1))
        B = out.shape[0]
        out = out.view(B, self.n_components, 2 * ACTION_DIM + 1)
        means = out[:, :, :ACTION_DIM]                  # (B, K, A)
        mix_logits = out[:, :, -1]                      # (B, K)
        best = mix_logits.argmax(dim=1)                 # (B,)
        idx = torch.arange(B, device=obs.device)
        action = means[idx, best]                       # (B, A)
        return action.tanh()


def make_ibc_mdn(device: torch.device):
    """IBC PixelMDN per Florence et al. 2021 Appendix D.2 (Pushing Pixels MDN-BC).

    Inference is a single forward pass through encoder + MDN head (no DFO,
    no Langevin, no sampling — argmax-mode picks the best mixture component
    and returns its mean).
    """
    # MDN config: image size 120×90 (per paper Appendix D.2 table).
    mdn_target_h = 90
    mdn_target_w = 120
    net = _IBCPixelMDN(
        in_channels=IMAGE_CHANNELS,
        target_height=mdn_target_h,
        target_width=mdn_target_w,
        mlp_width=512,
        mlp_depth=8,
        n_components=26,
        dropout=0.1,
    ).to(device).eval()

    def select_action():
        obs = torch.randint(0, 256, (1, IMAGE_CHANNELS, IMAGE_HEIGHT, IMAGE_WIDTH),
                            dtype=torch.uint8, device=device)
        with torch.no_grad():
            action = net(obs)
        return action[0]

    return "ibc_mdn", select_action, (net,)


# ─── Method 4: IBC + MSE-BC (paper Simulated Pushing Pixels MSE) ──────────

class _IBCPixelMSE(nn.Module):
    """IBC paper Simulated Pushing, Pixels, MSE-BC.

    Per Florence et al. 2021 Appendix D.2:
      - Image size: 240×180.
      - Conv Net: 4-layer ConvMaxPool with CoordConv (2 extra coord channels).
      - MLP: 512×4 (width × depth = 4 dense layers).
      - Dropout 0.1 (MLP only).
      - Activation: ReLU.

    Inference is a single forward pass.
    """

    def __init__(self, in_channels: int, target_height: int, target_width: int,
                 mlp_width: int = 512, mlp_depth: int = 4, dropout: float = 0.1):
        super().__init__()
        # CoordConv prepends 2 channels before encoder.
        self.encoder = _CoordConvEncoder(
            in_channels=in_channels,
            target_height=target_height,
            target_width=target_width,
            feature_dim=256,
        )
        layers = []
        prev = 256
        for _ in range(mlp_depth):
            layers.append(nn.Linear(prev, mlp_width))
            layers.append(nn.ReLU(inplace=True))
            layers.append(nn.Dropout(dropout))
            prev = mlp_width
        layers.append(nn.Linear(mlp_width, ACTION_DIM))
        self.mlp = nn.Sequential(*layers)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        feats = self.encoder(obs)
        action = self.mlp(feats)
        return action.tanh()


def make_ibc_mse(device: torch.device):
    """IBC PixelMSE per Florence et al. 2021 Appendix D.2 (Pushing Pixels MSE-BC).

    Inference is a single forward pass through CoordConv encoder + MLP +
    tanh. No iterative refinement.
    """
    net = _IBCPixelMSE(
        in_channels=IMAGE_CHANNELS,
        target_height=ENCODER_TARGET_HEIGHT,   # 180
        target_width=ENCODER_TARGET_WIDTH,     # 240
        mlp_width=512,
        mlp_depth=4,
        dropout=0.1,
    ).to(device).eval()

    def select_action():
        obs = torch.randint(0, 256, (1, IMAGE_CHANNELS, IMAGE_HEIGHT, IMAGE_WIDTH),
                            dtype=torch.uint8, device=device)
        with torch.no_grad():
            action = net(obs)
        return action[0]

    return "ibc_mse", select_action, (net,)


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
    # Q3C variants:
    #   0 iters  (I0A):   1 score call × 30 CPs                            =   30
    #   3 iters  (J3):    (3 + 1 final rescore) × (30 CPs + 16 unif)       =  184
    #   5 iters  (J5):    (5 + 1 final rescore) × (30 CPs + 24 unif)       =  324
    #   10 iters (stack): (10 + 1 final rescore) × (20 CPs + 16 unif)      =  396
    # IBC DFO: 3 iters × 4096 samples = 12,288. No final re-score (IBC convention).
    IBC_SCORING_EVALS = 3 * 4096

    # ── Q3C variants: best config per iter count ────────────────────────
    q3c_variants = [
        # (method_name, cps, n_uniform, iters, std, decay, sr_const, source_note)
        ("q3c_0iter",  30,  0,  0,  0.000, 0.5, Q3C_0ITER,  "I0A: 200k+30CPs, pure argmax"),
        ("q3c_3iter",  30, 16,  3,  0.020, 0.5, Q3C_3ITER,  "J3:  200k+30CPs, std=0.02,  16 unif"),
        ("q3c_5iter",  30, 24,  5,  0.005, 0.5, Q3C_5ITER,  "J5:  200k+30CPs, std=0.005, 24 unif"),
        ("q3c_10iter", 20, 16, 10, 0.005, 0.5, Q3C_10ITER, "Stack: 150k+20CPs, std=0.005, 16 unif"),
    ]
    for idx, (name, cps, n_uni, iters, std, decay, sr_const, note) in enumerate(q3c_variants, start=1):
        print("=" * 78)
        print(f"Method {idx}: Q3CIBC + CP-DFO ({iters} iters)  [{note}]")
        method, fn, modules = make_q3c(device, control_points=cps, n_uniform=n_uni,
                                       dfo_iters=iters, dfo_std=std, dfo_decay=decay,
                                       method_name=name)
        n_params = param_count(*modules)
        print(f"  Param count: {n_params:,}")
        t = time_block(fn, args.num_steps, args.warmup, device)
        print(f"  Mean:    {t['mean_ms']:7.3f} ms  (median {t['median_ms']:7.3f} ms, std {t['stdev_ms']:6.3f} ms)")
        print(f"  Range:   [{t['min_ms']:7.3f}, {t['max_ms']:7.3f}] ms over {t['n']} runs")
        sr_mean, sr_std, n_seeds = sr_const
        print(f"  SR:      {sr_mean*100:.1f}% ± {sr_std*100:.1f}% over {n_seeds} seeds")
        # Scoring evals: 1*cps if iters=0, (iters+1)*(cps+n_uni) otherwise.
        if iters == 0:
            scoring_evals = cps
        else:
            scoring_evals = (iters + 1) * (cps + n_uni)
        rows.append((name, sr_mean * 100, sr_std * 100, n_seeds, scoring_evals, t))
        print()

    print("=" * 78)
    print("Method 5: IBC + DFO (paper Simulated Pushing Pixels EBM, Appendix D.2)")
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

    print("=" * 78)
    print("Method 6: IBC + MDN-BC (paper Simulated Pushing Pixels MDN, Appendix D.2)")
    print("  ConvMaxPool encoder (target 120×90) + 512×8 MLP head + 26-component MDN")
    print("  Inference: single forward pass, argmax(mixing_logits) → component mean → tanh")
    method, fn, modules = make_ibc_mdn(device)
    n_params = param_count(*modules)
    print(f"  Param count: {n_params:,}")
    t = time_block(fn, args.num_steps, args.warmup, device)
    print(f"  Mean:    {t['mean_ms']:7.3f} ms  (median {t['median_ms']:7.3f} ms, std {t['stdev_ms']:6.3f} ms)")
    print(f"  Range:   [{t['min_ms']:7.3f}, {t['max_ms']:7.3f}] ms over {t['n']} runs")
    print(f"  SR:      {MDN_SR_MEAN*100:.1f}% ± {MDN_SR_STD*100:.1f}% over {MDN_SEEDS} seeds (from Florence et al. 2021 Table 3, NOT measured here)")
    # MDN: 1 forward pass total (encoder + MLP head). "Scoring evals" doesn't
    # apply (no iterative search); record 1 for consistency.
    MDN_SCORING_EVALS = 1
    rows.append(("ibc_mdn", MDN_SR_MEAN * 100, MDN_SR_STD * 100, MDN_SEEDS, MDN_SCORING_EVALS, t))
    print()

    print("=" * 78)
    print("Method 7: IBC + MSE-BC (paper Simulated Pushing Pixels MSE, Appendix D.2)")
    print("  CoordConv (+2 channels) ConvMaxPool encoder (target 240×180) + 512×4 MLP + tanh")
    print("  Inference: single forward pass (no iterative refinement)")
    method, fn, modules = make_ibc_mse(device)
    n_params = param_count(*modules)
    print(f"  Param count: {n_params:,}")
    t = time_block(fn, args.num_steps, args.warmup, device)
    print(f"  Mean:    {t['mean_ms']:7.3f} ms  (median {t['median_ms']:7.3f} ms, std {t['stdev_ms']:6.3f} ms)")
    print(f"  Range:   [{t['min_ms']:7.3f}, {t['max_ms']:7.3f}] ms over {t['n']} runs")
    print(f"  SR:      {MSE_SR_MEAN*100:.1f}% ± {MSE_SR_STD*100:.1f}% over {MSE_SEEDS} seeds (from Florence et al. 2021 Table 3, NOT measured here)")
    MSE_SCORING_EVALS = 1
    rows.append(("ibc_mse", MSE_SR_MEAN * 100, MSE_SR_STD * 100, MSE_SEEDS, MSE_SCORING_EVALS, t))
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
