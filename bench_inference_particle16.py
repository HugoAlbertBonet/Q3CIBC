"""Inference-time benchmark for particle 16-D — IBC paper Langevin vs. our recipes.

Why this script exists
----------------------
The pushing benchmark at `bench_inference.py` measures algorithmic compute cost
on the 2-D Pushing task with random weights. This file does the same comparison
for the **Particle, n_dim=16** task — the one the hyperparameter search was
chasing — and writes a CSV row per method.

Three configurations are compared, all on networks shaped for particle 16-D
(obs_dim = 4·n_dim·frame_stack = 4·16·2 = 128, action_dim = 16,
action_bounds = [0, 1]):

  1. IBC + Langevin (paper config)
       ResNetPreAct Q-net width=128, depth=8 blocks (= IBC `MLPEBM.depth=16`).
       512 uniform chains; 100 iters + 100 `optimize_again` iters; lr=0.1,
       lr_final=1e-5, decay_power=2.0, delta_clip=0.1, noise_scale=1.0.
       Source: `google-research/ibc/configs/.../mlp_ebm_langevin.gin`.
       Success rate: not in our trial log (we don't train the IBC paper
       recipe directly); cell left blank.

  2. Q3CIBC best  (trial 231 recipe — peak 94% success)
       MLP Q-net width=256 depth=2, MLP CP-generator width=256 depth=2 with
       control_points=30. ONE chain initialized from the argmax-Q control
       point. 300 inference iterations; lr_init=0.015, lr_final=1e-4,
       decay_power=2.0, delta_clip=0.015, noise_scale=0.05.
       This is the highest single-trial success rate we observed on
       particle 16-D (n=1 / 15 at inf=300; mean of inf=300 group ≈ 87%).

  3. Q3CIBC fastest ≥90% (HH11-HH15 recipe — mean 89.2%, 3/5 trials at 90%)
       Same architecture as #2 but cheaper inference: 150 iters,
       lr_init=0.025, delta_clip=0.020 (~67% bigger per-step movement to
       compensate for halved iteration budget). Empirically the most
       reliable recipe we found — 4/5 trials in HH11-HH15 hit ≥88%,
       none below 88%, none above 90%.

What it measures: wall-clock per `select_action` call.
What it does NOT measure: training cost, task performance — random weights
are used throughout; success rates are filled from the trial log.

Output
------
Appends one CSV row per method to:
    results/particle/inference_benchmark.csv

The CSV is created with a header if missing. Stdout also prints a short
summary table so timings can be eyeballed at a glance.

Usage
-----
    uv run --managed-python --extra pushing python bench_inference_particle16.py \\
        --num-steps 50 --warmup 5 --device auto
"""

from __future__ import annotations

import argparse
import csv
import os
import platform
import statistics
import time
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch

from utils.models import ControlPointGenerator, QEstimator


# ─── Particle n_dim=16 task constants ─────────────────────────────────────────
# From config_json/config.json (particle entry): n_dim=16, state_dim=64,
# action_dim=16, frame_stack=2, action_bounds=[0, 1].
N_DIM = 16
FRAME_STACK = 2
OBS_DIM = 4 * N_DIM * FRAME_STACK          # 4·16·2 = 128
ACTION_DIM = N_DIM                          # 16
ACTION_MIN = 0.0
ACTION_MAX = 1.0

RESULTS_PATH = Path(__file__).parent / "results" / "particle" / "inference_benchmark.csv"

CSV_COLUMNS = [
    "timestamp",
    "method",
    "approach",            # "ibc" or "q3c"
    "dimension",           # particle n_dim
    "success_rate",        # from trial log (blank for IBC-not-trained)
    "inference_mean_ms",
    "inference_median_ms",
    "inference_stdev_ms",
    "inference_min_ms",
    "inference_max_ms",
    "num_params",
    "langevin_iters",
    "num_chains",
    "q_net_kind",
    "q_width",
    "q_depth_blocks",
    "device",
    "notes",
]


@contextmanager
def cuda_sync(device: torch.device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    yield
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def time_block(fn, num_steps: int, warmup: int, device: torch.device) -> dict:
    """Run `fn()` num_steps + warmup times, return ms-scale timing stats."""
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


# ─── Network builders ─────────────────────────────────────────────────────────

def build_q_resnet(width: int, depth_blocks: int, device: torch.device) -> QEstimator:
    """Paper-shaped ResNetPreAct Q-net (IBC MLPEBM)."""
    return QEstimator(
        state_dim=OBS_DIM,
        action_dim=ACTION_DIM,
        hidden_dims=[width] * depth_blocks,
        use_spectral_norm=False,
        dropout_rate=0.0,
        network_kind="resnet",
        width=width,
        depth=depth_blocks,
    ).to(device).eval()


def build_q_mlp(width: int, depth: int, device: torch.device) -> QEstimator:
    """Q3CIBC's flat-MLP Q-net (matches the locked particle-16 recipe)."""
    return QEstimator(
        state_dim=OBS_DIM,
        action_dim=ACTION_DIM,
        hidden_dims=[width] * depth,
        use_spectral_norm=False,
        dropout_rate=0.0,
        network_kind="mlp",
        width=width,
        depth=depth,
    ).to(device).eval()


def build_cp_gen(width: int, depth: int, control_points: int,
                 device: torch.device) -> ControlPointGenerator:
    """Q3CIBC's CP generator — unused by IBC methods."""
    return ControlPointGenerator(
        input_dim=OBS_DIM,
        output_dim=ACTION_DIM,
        control_points=control_points,
        hidden_dims=[width] * depth,
        action_bounds=(ACTION_MIN, ACTION_MAX),
        network_kind="mlp",
        width=width,
        depth=depth,
        use_spectral_norm=False,
    ).to(device).eval()


# ─── Langevin primitive ───────────────────────────────────────────────────────

def langevin_pass(
    q_net: QEstimator,
    obs: torch.Tensor,            # (1, OBS_DIM)
    actions: torch.Tensor,        # (1, N, ACTION_DIM)
    num_iterations: int,
    lr_init: float,
    lr_final: float,
    decay_power: float,
    delta_clip: float,
    noise_scale: float,
) -> torch.Tensor:
    """Single Langevin bank pass (mirrors google-research/ibc + our sample_langevin)."""
    obs_expanded = obs.unsqueeze(1).expand(-1, actions.shape[1], -1)
    a = actions.detach().clone().requires_grad_(True)
    for it in range(num_iterations):
        frac = it / max(1, num_iterations - 1)
        stepsize = (lr_init - lr_final) * (1.0 - frac) ** decay_power + lr_final
        energy = -q_net(obs_expanded, a).squeeze(-1)
        grad = torch.autograd.grad(energy.sum(), a, create_graph=False)[0]
        step = -stepsize * grad
        step = torch.clamp(step, -delta_clip, delta_clip)
        noise = torch.randn_like(a) * stepsize * noise_scale
        a = (a + step + noise).clamp(ACTION_MIN, ACTION_MAX).detach().requires_grad_(True)
    return a.detach()


# ─── Method 1: IBC paper Langevin ─────────────────────────────────────────────

def make_method_ibc_langevin(device: torch.device):
    """Paper config: ResNetPreAct width=128 depth=8 blocks, 512 chains × 200 iters total."""
    q_net = build_q_resnet(width=128, depth_blocks=8, device=device)
    name = "IBC + Langevin (paper: ResNet 128×8, 512 chains, 100+100 iters)"
    cfg = dict(num_iterations=100, lr_init=0.1, lr_final=1e-5, decay_power=2.0,
               delta_clip=0.1, noise_scale=1.0)

    def select_action():
        obs = torch.randn(1, OBS_DIM, device=device)
        actions = torch.empty(1, 512, ACTION_DIM, device=device).uniform_(ACTION_MIN, ACTION_MAX)
        refined = langevin_pass(q_net, obs, actions, **cfg)
        refined = langevin_pass(q_net, obs, refined, **cfg)  # optimize_again
        with torch.no_grad():
            obs_exp = obs.unsqueeze(1).expand(-1, refined.shape[1], -1)
            q = q_net(obs_exp, refined).squeeze(-1)
            sel = q.argmax(dim=1)
        return refined[0, sel[0], :]

    return name, select_action, q_net, None


# ─── Method 2: Q3CIBC best (trial 231 — 94% success) ──────────────────────────

def make_method_q3c_best(device: torch.device):
    """Trial 231 recipe — single chain from argmax-Q CP, 300 iters, default step."""
    q_net = build_q_mlp(width=256, depth=2, device=device)
    cp_gen = build_cp_gen(width=256, depth=2, control_points=30, device=device)
    name = "Q3CIBC best (MLP 256×2, CPs=30, 1 chain, 300 iters)"
    lv_cfg = dict(num_iterations=300, lr_init=0.015, lr_final=1e-4, decay_power=2.0,
                  delta_clip=0.015, noise_scale=0.05)

    def select_action():
        obs = torch.randn(1, OBS_DIM, device=device)
        with torch.no_grad():
            cps = cp_gen(obs)  # (1, 30, 16)
            obs_exp = obs.unsqueeze(1).expand(-1, cps.shape[1], -1)
            q = q_net(obs_exp, cps).squeeze(-1)
            best = q.argmax(dim=1)
            best_cp = cps[0, best[0], :].view(1, 1, -1).clone()
        refined = langevin_pass(q_net, obs, best_cp, **lv_cfg)
        return refined[0, 0, :]

    return name, select_action, q_net, cp_gen


# ─── Method 3: Q3CIBC fastest ≥90% (HH11-HH15 — 89.2% mean, 90% peak) ────────

def make_method_q3c_fastest(device: torch.device):
    """HH11-HH15 recipe — single chain from argmax-Q CP, 150 iters, ~67% bigger step."""
    q_net = build_q_mlp(width=256, depth=2, device=device)
    cp_gen = build_cp_gen(width=256, depth=2, control_points=30, device=device)
    name = "Q3CIBC fastest ≥90% (MLP 256×2, CPs=30, 1 chain, 150 iters, lr=0.025, clip=0.020)"
    lv_cfg = dict(num_iterations=150, lr_init=0.025, lr_final=1e-4, decay_power=2.0,
                  delta_clip=0.020, noise_scale=0.05)

    def select_action():
        obs = torch.randn(1, OBS_DIM, device=device)
        with torch.no_grad():
            cps = cp_gen(obs)
            obs_exp = obs.unsqueeze(1).expand(-1, cps.shape[1], -1)
            q = q_net(obs_exp, cps).squeeze(-1)
            best = q.argmax(dim=1)
            best_cp = cps[0, best[0], :].view(1, 1, -1).clone()
        refined = langevin_pass(q_net, obs, best_cp, **lv_cfg)
        return refined[0, 0, :]

    return name, select_action, q_net, cp_gen


# ─── Driver ───────────────────────────────────────────────────────────────────

@dataclass
class MethodSpec:
    name: str
    approach: str          # "ibc" or "q3c"
    success_rate: str      # string so blank cells stay blank
    langevin_iters: int
    num_chains: int
    q_net_kind: str
    q_width: int
    q_depth_blocks: int
    notes: str
    make: callable


def param_count(*modules) -> int:
    return sum(p.numel() for m in modules if m is not None for p in m.parameters())


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--num-steps", type=int, default=50, help="Timed calls per method")
    parser.add_argument("--warmup", type=int, default=5, help="Untimed warmup calls")
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cuda", "cpu"])
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() and args.device != "cpu" else "cpu") \
        if args.device == "auto" else torch.device(args.device)

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    methods = [
        MethodSpec(
            name="IBC + Langevin (paper)",
            approach="ibc",
            success_rate="",  # IBC paper recipe not trained in our pipeline at n_dim=16
            langevin_iters=200,  # 100 + 100 optimize_again
            num_chains=512,
            q_net_kind="resnet",
            q_width=128,
            q_depth_blocks=8,
            notes="MLPEBM ResNetPreAct; lr=0.1, lr_final=1e-5, clip=0.1, noise=1.0; "
                  "512 uniform chains; argmax of final scores.",
            make=make_method_ibc_langevin,
        ),
        MethodSpec(
            name="Q3CIBC best (trial 231)",
            approach="q3c",
            success_rate="0.94",
            langevin_iters=300,
            num_chains=1,
            q_net_kind="mlp",
            q_width=256,
            q_depth_blocks=2,
            notes="MLP Q + MLP CP-gen (CPs=30); 1 chain from argmax-Q CP; "
                  "lr=0.015, lr_final=1e-4, clip=0.015, noise=0.05. "
                  "Best particle-16 trial in `particle/16/trials.jsonl`.",
            make=make_method_q3c_best,
        ),
        MethodSpec(
            name="Q3CIBC fastest ≥90% (HH11-HH15)",
            approach="q3c",
            success_rate="0.892",
            langevin_iters=150,
            num_chains=1,
            q_net_kind="mlp",
            q_width=256,
            q_depth_blocks=2,
            notes="Same architecture as `best` but 150 iters, lr=0.025, clip=0.020. "
                  "Mean over 5 HH11-HH15 trials = 0.892; 3/5 hit 0.90 exactly, "
                  "none below 0.88. Halves inference cost vs the `best` recipe.",
            make=make_method_q3c_fastest,
        ),
    ]

    print("=" * 78)
    print(f"Inference benchmark — Particle n_dim={N_DIM} (random weights)")
    print(f"  device = {device}")
    print(f"  warmup={args.warmup}, timed_steps={args.num_steps}")
    print(f"  OBS_DIM={OBS_DIM} (4·{N_DIM}·{FRAME_STACK} frame-stack), ACTION_DIM={ACTION_DIM}")
    print("=" * 78)

    timestamp = datetime.now(timezone.utc).isoformat()
    device_str = (torch.cuda.get_device_name(device) if device.type == "cuda"
                  else platform.processor() or platform.machine())
    rows = []

    for spec in methods:
        name, fn, q_net, cp_gen = spec.make(device)
        params = param_count(q_net, cp_gen)
        print(f"\n-- {name}  ({params:,} params)")
        try:
            timings = time_block(fn, args.num_steps, args.warmup, device)
        except Exception as exc:
            print(f"   FAILED: {exc!r}")
            continue
        print(f"   mean = {timings['mean_ms']:7.2f} ms   "
              f"median = {timings['median_ms']:7.2f} ms   "
              f"std = {timings['stdev_ms']:6.2f} ms   "
              f"min = {timings['min_ms']:7.2f} ms   "
              f"max = {timings['max_ms']:7.2f} ms")

        rows.append({
            "timestamp": timestamp,
            "method": spec.name,
            "approach": spec.approach,
            "dimension": N_DIM,
            "success_rate": spec.success_rate,
            "inference_mean_ms": f"{timings['mean_ms']:.3f}",
            "inference_median_ms": f"{timings['median_ms']:.3f}",
            "inference_stdev_ms": f"{timings['stdev_ms']:.3f}",
            "inference_min_ms": f"{timings['min_ms']:.3f}",
            "inference_max_ms": f"{timings['max_ms']:.3f}",
            "num_params": params,
            "langevin_iters": spec.langevin_iters,
            "num_chains": spec.num_chains,
            "q_net_kind": spec.q_net_kind,
            "q_width": spec.q_width,
            "q_depth_blocks": spec.q_depth_blocks,
            "device": device_str,
            "notes": spec.notes,
        })

    # ── Summary table sorted by mean inference time ───────────────────────────
    print()
    print("=" * 78)
    print("Sorted by mean inference time (cheapest first):")
    print("=" * 78)
    sorted_rows = sorted(rows, key=lambda r: float(r["inference_mean_ms"]))
    if sorted_rows:
        fastest = float(sorted_rows[0]["inference_mean_ms"])
        for r in sorted_rows:
            mean_ms = float(r["inference_mean_ms"])
            rel = mean_ms / fastest
            print(f"  {r['method']:<55}  {mean_ms:>8.2f} ms   {rel:>5.2f}×")

    # ── Append/create CSV ─────────────────────────────────────────────────────
    RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    file_exists = RESULTS_PATH.exists()
    with open(RESULTS_PATH, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_COLUMNS)
        if not file_exists:
            writer.writeheader()
        for r in rows:
            writer.writerow(r)
    print()
    print(f"Appended {len(rows)} rows to {RESULTS_PATH.relative_to(Path(__file__).parent)}")


if __name__ == "__main__":
    main()
