"""Inference-time benchmark for Q3CIBC vs. IBC paper on D4RL pen-human.

Compares wall-clock per env-step of two inference paths at the same Q
architecture (IBC paper recipe: 512×8 ResNet/MLP with spectral norm), random
weights — pure timing comparison. Reward/SEM/std come from already-recorded
training+eval runs in trials.jsonl (no retraining needed).

Methods benchmarked
-------------------
1. **Q3C-B4 (CP-argmax)** — Q3CIBC headline path.
   `cp_gen(obs)` → 100 CPs → `q_net(obs, cps)` → argmax. NO refinement.
   1 fwd through CP gen + 1 fwd through Q over 100 candidates.
2. **IBC paper-exact (Langevin)** — Florence et al. 2021 App. D.1.
   Uniform initial sampling (`UNIFORM_BOUNDARY_BUFFER=0.05`), 100 Langevin
   iters, step_init=0.5 → 1e-5 polynomial decay, noise=0.5, delta_clip=0.5.
   512 candidates per step (`INFERENCE_NUM_SAMPLES`). Autograd at every iter.

Reward / std / SEM / cross-seed std
-----------------------------------
Loaded from existing trial records:
  Q3C-B4   : results/hyperparam_search/combinedv2_cpascounter_training/d4rl/pen/trials.jsonl
             (filter: cp=100, top_k=30, inf_lit=0, training_steps=100000, GP=1.0/hinge)
  IBC-paper: results/hyperparam_search/ibc_dfo_pen/trials.jsonl
             (filter: NUM_COUNTER_EXAMPLES=8, GRADIENT_MARGIN=1.0,
              SOFTMAX_TEMPERATURE=1.0, LANGEVIN_TRAIN_ITERATIONS=100,
              INFERENCE_NUM_ITERATIONS=100; reeval_only=true so eval works)

Output
------
CSV at results/hyperparam_search/combinedv2_cpascounter_training/d4rl/pen/
     pen_inference_results.csv
columns: method, n_seeds, avg_reward, std (σ_ep avg), SEM, cross_seed_std,
         inference_time_mean_ms, inference_time_std_ms, inference_time_median_ms

Usage
-----
    uv run --managed-python python bench_inference_pen.py
    uv run --managed-python python bench_inference_pen.py --num-steps 200 --warmup 20 --device cuda
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
import time
from contextlib import contextmanager
from pathlib import Path

import numpy as np
import torch

from utils.models import QEstimator, ControlPointGenerator


ROOT = Path(__file__).parent
Q3C_TRIALS = ROOT / "results" / "hyperparam_search" / "combinedv2_cpascounter_training" / "d4rl" / "pen" / "trials.jsonl"
IBC_TRIALS = ROOT / "results" / "hyperparam_search" / "ibc_dfo_pen" / "trials.jsonl"
OUT_CSV = ROOT / "results" / "hyperparam_search" / "combinedv2_cpascounter_training" / "d4rl" / "pen" / "pen_inference_results.csv"

# Pen env / arch constants (AdroitHandPen-v1, IBC paper App. D.1).
OBS_DIM = 45
ACTION_DIM = 24
ACTION_MIN = -1.0
ACTION_MAX = 1.0

# IBC paper-exact recipe (Q + inference).
IBC_Q_WIDTH = 512
IBC_Q_DEPTH = 8                 # paper "depth=8" linear layers count
IBC_INF_NUM_SAMPLES = 512
IBC_INF_NUM_ITERS = 100
IBC_INF_LR_INIT = 0.5
IBC_INF_LR_FINAL = 1e-5
IBC_INF_DECAY = 2.0
IBC_INF_DELTA_CLIP = 0.5
IBC_INF_NOISE = 0.5
IBC_UNIFORM_BUFFER = 0.05

# Q3C-B4 recipe (pure CP-argmax inference, no refinement).
Q3C_CP = 100
Q3C_TOPK = 30
Q3C_CP_WIDTH = 512
Q3C_CP_DEPTH = 8
Q3C_Q_WIDTH = 512
Q3C_Q_DEPTH = 8

# Q3C + DFO inference recipe ("#70" recipe — penE/F best DFO config).
# Train differences vs B4: num_uniform_negatives=64, gradient_penalty_weight=5.
# Inference: 10 DFO iters + 32 uniform safety samples.
Q3C_DFO_CP = 100
Q3C_DFO_TOPK = 30
Q3C_DFO_ITERS = 10
Q3C_DFO_ITER_STD = 0.05
Q3C_DFO_STD_DECAY = 0.7
Q3C_DFO_NUM_UNIFORM = 32

# Q3C + gentle Langevin inference recipe (penFlangevin best — cp=80 + very-gentle 25 iters).
# Train differences vs B4: control_points=80, top_k=25.
# Inference: 25 Langevin iters at very-gentle settings (walk envelope ~0.05).
Q3C_LANG_CP = 80
Q3C_LANG_TOPK = 25
Q3C_LANG_NUM_ITERS = 25
Q3C_LANG_LR_INIT = 0.01
Q3C_LANG_LR_FINAL = 1e-6
Q3C_LANG_DECAY = 2.0
Q3C_LANG_DELTA_CLIP = 0.01
Q3C_LANG_NOISE = 0.1


# ─── CUDA sync helpers (mirror bench_inference.py) ───────────────────────────

@contextmanager
def cuda_sync(device: torch.device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    yield
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def time_block(fn, num_steps: int, warmup: int, device: torch.device) -> dict:
    """Run fn() num_steps + warmup times, return timing stats in ms."""
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


# ─── Network builders (random weights, paper-faithful shapes) ────────────────

def build_q_paper(device: torch.device) -> QEstimator:
    """IBC paper Q-net: 512×8 MLP with spectral norm. Random weights."""
    return QEstimator(
        state_dim=OBS_DIM,
        action_dim=ACTION_DIM,
        hidden_dims=[IBC_Q_WIDTH] * IBC_Q_DEPTH,
        use_spectral_norm=True,
        network_kind="mlp",
        width=IBC_Q_WIDTH,
        depth=IBC_Q_DEPTH,
    ).to(device).eval()


def build_q3c_q(device: torch.device) -> QEstimator:
    return QEstimator(
        state_dim=OBS_DIM,
        action_dim=ACTION_DIM,
        hidden_dims=[Q3C_Q_WIDTH] * Q3C_Q_DEPTH,
        use_spectral_norm=True,
        network_kind="mlp",
        width=Q3C_Q_WIDTH,
        depth=Q3C_Q_DEPTH,
    ).to(device).eval()


def build_q3c_cpgen(device: torch.device) -> ControlPointGenerator:
    return ControlPointGenerator(
        input_dim=OBS_DIM,
        output_dim=ACTION_DIM,
        control_points=Q3C_CP,
        hidden_dims=[Q3C_CP_WIDTH] * Q3C_CP_DEPTH,
        action_bounds=(ACTION_MIN, ACTION_MAX),
        network_kind="mlp",
        width=Q3C_CP_WIDTH,
        depth=Q3C_CP_DEPTH,
        use_spectral_norm=False,
    ).to(device).eval()


# ─── Inference primitives ────────────────────────────────────────────────────

def langevin_pass_ibc(q_net, obs, actions, num_iterations, lr_init, lr_final,
                     decay_power, delta_clip, noise_scale):
    """IBC paper-style Langevin. Mirrors hyperparam_search_dfo.langevin loop."""
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


# ─── Method 1: Q3C-B4 CP-argmax (no refinement) ──────────────────────────────

def make_method_q3c(device: torch.device):
    cp_gen = build_q3c_cpgen(device)
    q_net = build_q3c_q(device)
    name = f"Q3C-B4 CP-argmax (cp={Q3C_CP}, no refinement)"

    def select_action():
        obs = torch.randn(1, OBS_DIM, device=device)
        with torch.no_grad():
            cps = cp_gen(obs)                                          # (1, N, A)
            obs_exp = obs.unsqueeze(1).expand(-1, cps.shape[1], -1)
            q = q_net(obs_exp, cps).squeeze(-1)                        # (1, N)
            best = q.argmax(dim=1)
            action = cps[0, best[0], :]
        return action

    return name, select_action


# ─── Method 2: Q3C + CP-DFO inference (#70 recipe — penE/F best DFO config) ──

def build_q3c_cpgen_n(device: torch.device, cp: int) -> ControlPointGenerator:
    """Same arch as build_q3c_cpgen but parameterised CP count for cp!=100."""
    return ControlPointGenerator(
        input_dim=OBS_DIM,
        output_dim=ACTION_DIM,
        control_points=cp,
        hidden_dims=[Q3C_CP_WIDTH] * Q3C_CP_DEPTH,
        action_bounds=(ACTION_MIN, ACTION_MAX),
        network_kind="mlp",
        width=Q3C_CP_WIDTH,
        depth=Q3C_CP_DEPTH,
        use_spectral_norm=False,
    ).to(device).eval()


def make_method_q3c_dfo(device: torch.device):
    """Q3C + DFO inference. Mirrors hyperparam_search.evaluate_q3c DFO path:
    CP cloud (+optional uniform safety samples) → iterative softmax-resample
    + Gaussian jitter + clip → argmax final scoring. No autograd.
    """
    cp_gen = build_q3c_cpgen_n(device, Q3C_DFO_CP)
    q_net = build_q3c_q(device)
    name = (f"Q3C+CP-DFO (cp={Q3C_DFO_CP}, {Q3C_DFO_ITERS} iters, "
            f"+{Q3C_DFO_NUM_UNIFORM} uniform safety)")

    def select_action():
        obs = torch.randn(1, OBS_DIM, device=device)
        with torch.no_grad():
            cps = cp_gen(obs)                                          # (1, cp, A)
            if Q3C_DFO_NUM_UNIFORM > 0:
                unif = torch.empty(
                    1, Q3C_DFO_NUM_UNIFORM, ACTION_DIM, device=device
                ).uniform_(ACTION_MIN, ACTION_MAX)
                candidates = torch.cat([cps, unif], dim=1)
            else:
                candidates = cps.clone()
            N = candidates.shape[1]
            obs_exp = obs.unsqueeze(1).expand(-1, N, -1)
            std = Q3C_DFO_ITER_STD
            for it in range(Q3C_DFO_ITERS):
                log_probs = q_net(obs_exp, candidates).squeeze(-1)     # (1, N)
                probs = torch.softmax(log_probs.squeeze(0), dim=-1)
                idx = torch.multinomial(probs, N, replacement=True)
                counts = torch.bincount(idx, minlength=N)
                repeat_idx = torch.repeat_interleave(
                    torch.arange(N, device=device), counts
                )
                candidates = candidates[:, repeat_idx, :]
                if it < Q3C_DFO_ITERS - 1:
                    candidates = candidates + torch.randn_like(candidates) * std
                    candidates = candidates.clamp(ACTION_MIN, ACTION_MAX)
                    std *= Q3C_DFO_STD_DECAY
            final = q_net(obs_exp, candidates).squeeze(-1)
            sel = final.argmax(dim=1)
            action = candidates[0, sel[0], :]
        return action

    return name, select_action


# ─── Method 3: Q3C + gentle Langevin inference (penFlangevin best) ───────────

def make_method_q3c_langevin(device: torch.device):
    """Q3C + gentle Langevin inference. Initialises chain at argmax-CP, then
    runs `num_iters` Langevin steps with very-gentle hypers (walk envelope ~0.05).
    Uses autograd. Mirrors hyperparam_search.evaluate_q3c LangevinRefined path.
    """
    cp_gen = build_q3c_cpgen_n(device, Q3C_LANG_CP)
    q_net = build_q3c_q(device)
    name = (f"Q3C+gentle Langevin (cp={Q3C_LANG_CP}, {Q3C_LANG_NUM_ITERS} iters, "
            f"very-gentle)")

    def select_action():
        obs = torch.randn(1, OBS_DIM, device=device)
        with torch.no_grad():
            cps = cp_gen(obs)                                          # (1, cp, A)
            obs_exp = obs.unsqueeze(1).expand(-1, cps.shape[1], -1)
            q = q_net(obs_exp, cps).squeeze(-1)                        # (1, cp)
            best = q.argmax(dim=1)
            best_cp = cps[0, best[0], :].view(1, 1, -1).clone()        # (1, 1, A)

        # Gentle Langevin refinement. Same loop as langevin_pass_ibc but
        # applied to the single argmax-CP starting point.
        refined = langevin_pass_ibc(
            q_net, obs, best_cp,
            num_iterations=Q3C_LANG_NUM_ITERS,
            lr_init=Q3C_LANG_LR_INIT,
            lr_final=Q3C_LANG_LR_FINAL,
            decay_power=Q3C_LANG_DECAY,
            delta_clip=Q3C_LANG_DELTA_CLIP,
            noise_scale=Q3C_LANG_NOISE,
        )
        return refined[0, 0, :]

    return name, select_action


# ─── Method 4: IBC paper-exact Langevin inference ────────────────────────────

def make_method_ibc_paper(device: torch.device):
    q_net = build_q_paper(device)
    name = (f"IBC paper-exact Langevin ({IBC_INF_NUM_ITERS} iters × "
            f"{IBC_INF_NUM_SAMPLES} samples)")

    buf = (ACTION_MAX - ACTION_MIN) * IBC_UNIFORM_BUFFER
    samp_lo = ACTION_MIN - buf
    samp_hi = ACTION_MAX + buf

    def select_action():
        obs = torch.randn(1, OBS_DIM, device=device)
        # Initial uniform candidates with IBC's 5% boundary buffer.
        actions = torch.empty(1, IBC_INF_NUM_SAMPLES, ACTION_DIM, device=device).uniform_(
            samp_lo, samp_hi
        )
        refined = langevin_pass_ibc(
            q_net, obs, actions,
            num_iterations=IBC_INF_NUM_ITERS,
            lr_init=IBC_INF_LR_INIT,
            lr_final=IBC_INF_LR_FINAL,
            decay_power=IBC_INF_DECAY,
            delta_clip=IBC_INF_DELTA_CLIP,
            noise_scale=IBC_INF_NOISE,
        )
        with torch.no_grad():
            obs_exp = obs.unsqueeze(1).expand(-1, refined.shape[1], -1)
            q = q_net(obs_exp, refined).squeeze(-1)
            best = q.argmax(dim=1)
            action = refined[0, best[0], :]
        return action

    return name, select_action


# ─── Reward / std / SEM / cross-seed std from existing trials ────────────────

def _load_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    out = []
    with open(path) as f:
        for line in f:
            if line.strip():
                out.append(json.loads(line))
    return out


# Trial-id cutoff: penC onwards uses corrected 100-step protocol +
# standardize obs + per-dim action norm. Earlier trials (200-step regime)
# share the same `params` dict but are NOT comparable.
Q3C_PROTOCOL_MIN_ID = 16


def q3c_b4_stats() -> dict:
    """B4 multi-seed aggregates from Q3C trials.jsonl.

    Filter: cp=100, top_k=30, inf_lit=0, dfo=0, training_steps=100000,
    LR=5e-4, GP=1.0/hinge, num_langevin_negatives=8, num_uniform_negatives=0,
    Q=512×8 SN, langevin_init_kind=uniform. Trials before
    Q3C_PROTOCOL_MIN_ID are dropped (different env protocol). When the same
    seed has multiple matching trials, the LATEST trial wins.
    """
    trials = _load_jsonl(Q3C_TRIALS)
    keep = []
    for t in trials:
        if t.get("training_failed") or t.get("eval_error"):
            continue
        if int(t.get("trial_id", 0)) < Q3C_PROTOCOL_MIN_ID:
            continue
        p = t.get("params", {}) or {}
        if (p.get("control_points") == 100
                and p.get("top_k_control_points") == 30
                and p.get("inference_langevin_iterations", 0) == 0
                and p.get("inference_dfo_iterations", 0) == 0
                and p.get("training_steps") == 100000
                and p.get("learning_rate") == 5e-4
                and p.get("gradient_penalty_weight") == 1.0
                and p.get("gradient_penalty_form") == "hinge"
                and p.get("num_langevin_negatives") == 8
                and p.get("num_uniform_negatives", 0) == 0
                and p.get("q_width") == 512
                and p.get("q_depth") == 8
                and p.get("langevin_init_kind", "uniform") == "uniform"
                and p.get("entropy_bandwidth", 0.2) == 0.2
                and p.get("cp_use_spectral_norm", False) is False
                and p.get("noisy_expert_count", 0) == 0
                and p.get("cp_width") == 512
                and p.get("cp_depth") == 8):
            keep.append(t)
    # Dedupe by seed — keep the latest trial per seed.
    keep.sort(key=lambda t: int(t.get("trial_id", 0)))
    by_seed: dict[int, dict] = {}
    for t in keep:
        by_seed[(t.get("params") or {}).get("trial_seed")] = t
    deduped = list(by_seed.values())
    return _aggregate(deduped, label="Q3C-B4")


def q3c_dfo_stats() -> dict:
    """Q3C+DFO multi-seed aggregates (#70 recipe).

    Filter: cp=100, top_k=30, inference_dfo_iterations=10, inference_dfo_num_uniform=32,
    num_uniform_negatives=64, gradient_penalty_weight=5, training_steps=100k,
    LR=5e-4, GP form=hinge, num_langevin_negatives=8, Q=512×8 SN.
    Dedupe by seed (latest wins).
    """
    trials = _load_jsonl(Q3C_TRIALS)
    keep = []
    for t in trials:
        if t.get("training_failed") or t.get("eval_error"):
            continue
        if int(t.get("trial_id", 0)) < Q3C_PROTOCOL_MIN_ID:
            continue
        p = t.get("params", {}) or {}
        if (p.get("control_points") == 100
                and p.get("top_k_control_points") == 30
                and p.get("inference_langevin_iterations", 0) == 0
                and p.get("inference_dfo_iterations") == 10
                and p.get("inference_dfo_num_uniform") == 32
                and p.get("training_steps") == 100000
                and p.get("learning_rate") == 5e-4
                and p.get("gradient_penalty_weight") == 5.0
                and p.get("gradient_penalty_form") == "hinge"
                and p.get("num_langevin_negatives") == 8
                and p.get("num_uniform_negatives") == 64
                and p.get("q_width") == 512
                and p.get("q_depth") == 8
                and p.get("langevin_init_kind", "uniform") == "uniform"
                and p.get("entropy_bandwidth", 0.2) == 0.2
                and p.get("cp_use_spectral_norm", False) is False
                and p.get("noisy_expert_count", 0) == 0
                and p.get("cp_width") == 512
                and p.get("cp_depth") == 8):
            keep.append(t)
    keep.sort(key=lambda t: int(t.get("trial_id", 0)))
    by_seed: dict[int, dict] = {}
    for t in keep:
        by_seed[(t.get("params") or {}).get("trial_seed")] = t
    deduped = list(by_seed.values())
    return _aggregate(deduped, label="Q3C+DFO")


def q3c_langevin_stats() -> dict:
    """Q3C+gentle Langevin aggregates (penFlangevin best — cp=80 + very-gentle 25).

    Filter: cp=80, top_k=25, inference_langevin_iterations=25,
    inference_langevin_lr_init=0.01, inference_langevin_delta_clip=0.01,
    inference_langevin_noise_scale=0.1. Dedupe by seed.
    """
    trials = _load_jsonl(Q3C_TRIALS)
    keep = []
    for t in trials:
        if t.get("training_failed") or t.get("eval_error"):
            continue
        if int(t.get("trial_id", 0)) < Q3C_PROTOCOL_MIN_ID:
            continue
        p = t.get("params", {}) or {}
        if (p.get("control_points") == 80
                and p.get("top_k_control_points") == 25
                and p.get("inference_langevin_iterations") == 25
                and p.get("inference_dfo_iterations", 0) == 0
                and float(p.get("inference_langevin_lr_init", 0)) == 0.01
                and float(p.get("inference_langevin_delta_clip", 0)) == 0.01
                and float(p.get("inference_langevin_noise_scale", 0)) == 0.1
                and p.get("training_steps") == 100000
                and p.get("learning_rate") == 5e-4
                and p.get("num_langevin_negatives") == 8
                and p.get("num_uniform_negatives", 0) == 0
                and p.get("q_width") == 512
                and p.get("q_depth") == 8
                and p.get("cp_width") == 512
                and p.get("cp_depth") == 8):
            keep.append(t)
    keep.sort(key=lambda t: int(t.get("trial_id", 0)))
    by_seed: dict[int, dict] = {}
    for t in keep:
        by_seed[(t.get("params") or {}).get("trial_seed")] = t
    deduped = list(by_seed.values())
    return _aggregate(deduped, label="Q3C+gentleLangevin")


def ibc_paper_stats() -> dict:
    """IBC paper-exact multi-seed aggregates from ibc_dfo_pen reevals.

    Filter: paper-exact recipe (NUM_COUNTER_EXAMPLES=8, GRADIENT_MARGIN=1.0,
    SOFTMAX_TEMPERATURE=1.0, LANGEVIN_TRAIN_ITERATIONS=100,
    INFERENCE_NUM_ITERATIONS=100, INFERENCE_LR_INIT=0.5, HIDDEN_DIMS=[512]*8).
    Only includes reeval_only trials (originals had SN reload bug).
    """
    trials = _load_jsonl(IBC_TRIALS)
    keep = []
    for t in trials:
        if t.get("training_failed") or t.get("eval_error"):
            continue
        h = t.get("hparams") or t.get("params") or {}
        if (h.get("NUM_COUNTER_EXAMPLES") == 8
                and h.get("GRADIENT_MARGIN") == 1.0
                and h.get("SOFTMAX_TEMPERATURE") == 1.0
                and h.get("LANGEVIN_TRAIN_ITERATIONS") == 100
                and h.get("INFERENCE_NUM_ITERATIONS") == 100
                and float(h.get("INFERENCE_LR_INIT", 0)) == 0.5
                and h.get("HIDDEN_DIMS") == [IBC_Q_WIDTH] * IBC_Q_DEPTH):
            keep.append(t)
    # Dedupe by seed — keep the latest trial per seed (so e.g. reeval
    # supersedes the original failed-eval row).
    keep.sort(key=lambda t: int(t.get("trial_id", 0)))
    by_seed: dict[int, dict] = {}
    for t in keep:
        h = t.get("hparams") or t.get("params") or {}
        by_seed[h.get("trial_seed")] = t
    deduped = list(by_seed.values())
    return _aggregate(deduped, label="IBC-paper-exact")


def _aggregate(trials: list[dict], label: str) -> dict:
    if not trials:
        return {"label": label, "n_seeds": 0, "avg_reward": None,
                "std": None, "SEM": None, "cross_seed_std": None,
                "seeds": [], "trial_ids": []}
    means = [float(t["avg_reward"]) for t in trials]
    sigma_eps = [float(t.get("std_reward") or 0) for t in trials]
    n = len(means)
    mean = statistics.mean(means)
    if n > 1:
        cross_std = statistics.stdev(means)
        sem = cross_std / math.sqrt(n)
    else:
        cross_std = 0.0
        sem = 0.0
    avg_sigma_ep = statistics.mean(sigma_eps)
    # Pull seeds from whichever param dict is populated.
    seeds = sorted({
        (t.get("params") or t.get("hparams") or {}).get("trial_seed")
        for t in trials
    })
    tids = sorted(t.get("trial_id", 0) for t in trials)
    return {
        "label": label, "n_seeds": n, "avg_reward": mean,
        "std": avg_sigma_ep, "SEM": sem, "cross_seed_std": cross_std,
        "seeds": seeds, "trial_ids": tids,
    }


# ─── Main ────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Pen inference benchmark.")
    parser.add_argument("--num-steps", type=int, default=100,
                        help="Timed env-steps per method (after warmup).")
    parser.add_argument("--warmup", type=int, default=10,
                        help="Warmup steps discarded from timing.")
    parser.add_argument("--device", type=str, default="auto",
                        choices=("auto", "cpu", "cuda"))
    args = parser.parse_args()

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    print(f"Device: {device}")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(device)}")

    # Seed for reproducibility of random weights.
    torch.manual_seed(0)
    np.random.seed(0)

    # ── Build + time all four methods ───────────────────────────────────────
    methods = []
    builders = (
        make_method_q3c,
        make_method_q3c_dfo,
        make_method_q3c_langevin,
        make_method_ibc_paper,
    )
    for builder in builders:
        name, fn = builder(device)
        print(f"\nTiming: {name}")
        stats = time_block(fn, num_steps=args.num_steps, warmup=args.warmup, device=device)
        print(f"  mean={stats['mean_ms']:.3f}ms  median={stats['median_ms']:.3f}ms  "
              f"stdev={stats['stdev_ms']:.3f}ms  min={stats['min_ms']:.3f}ms  max={stats['max_ms']:.3f}ms")
        methods.append((name, stats))

    q3c_time = methods[0][1]
    q3c_dfo_time = methods[1][1]
    q3c_lang_time = methods[2][1]
    ibc_time = methods[3][1]

    # ── Pull reward stats from existing trials ──────────────────────────────
    q3c_stats = q3c_b4_stats()
    q3c_dfo_st = q3c_dfo_stats()
    q3c_lang_st = q3c_langevin_stats()
    ibc_stats = ibc_paper_stats()
    all_stats = (q3c_stats, q3c_dfo_st, q3c_lang_st, ibc_stats)
    print("\nReward stats (from existing trials):")
    for s in all_stats:
        if s["n_seeds"] == 0:
            print(f"  {s['label']}: NO MATCHING TRIALS FOUND")
            continue
        print(f"  {s['label']}: n_seeds={s['n_seeds']} seeds={s['seeds']} "
              f"trial_ids={s['trial_ids']}")
        print(f"    avg_reward={s['avg_reward']:.1f}  σ_ep(avg)={s['std']:.1f}  "
              f"SEM={s['SEM']:.2f}  cross_seed_std={s['cross_seed_std']:.2f}")

    # ── Speed-vs-reward summary ─────────────────────────────────────────────
    if q3c_stats["avg_reward"] is not None and ibc_stats["avg_reward"] is not None:
        speedup = ibc_time["mean_ms"] / q3c_time["mean_ms"]
        print(f"\nQ3C CP-argmax inference {speedup:.1f}× faster than IBC paper.")
        print(f"Q3C CP-argmax reward       = {q3c_stats['avg_reward']:.1f}  ({q3c_time['mean_ms']:.2f} ms)")
        if q3c_dfo_st["avg_reward"] is not None:
            print(f"Q3C + DFO reward           = {q3c_dfo_st['avg_reward']:.1f}  ({q3c_dfo_time['mean_ms']:.2f} ms)")
        if q3c_lang_st["avg_reward"] is not None:
            print(f"Q3C + gentle Langevin      = {q3c_lang_st['avg_reward']:.1f}  ({q3c_lang_time['mean_ms']:.2f} ms)")
        print(f"IBC paper Langevin reward  = {ibc_stats['avg_reward']:.1f}  ({ibc_time['mean_ms']:.2f} ms)")

    # ── Write CSV ────────────────────────────────────────────────────────────
    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    cols = [
        "method", "n_seeds", "avg_reward", "std", "SEM", "cross_seed_std",
        "inference_time_mean_ms", "inference_time_std_ms", "inference_time_median_ms",
        "inference_time_min_ms", "inference_time_max_ms", "timed_steps", "seeds",
    ]
    rows = []
    for ((name, t), s) in zip(methods, all_stats):
        rows.append({
            "method": name,
            "n_seeds": s["n_seeds"],
            "avg_reward": f"{s['avg_reward']:.4f}" if s["avg_reward"] is not None else "",
            "std": f"{s['std']:.4f}" if s["std"] is not None else "",
            "SEM": f"{s['SEM']:.4f}" if s["SEM"] is not None else "",
            "cross_seed_std": f"{s['cross_seed_std']:.4f}" if s["cross_seed_std"] is not None else "",
            "inference_time_mean_ms": f"{t['mean_ms']:.4f}",
            "inference_time_std_ms": f"{t['stdev_ms']:.4f}",
            "inference_time_median_ms": f"{t['median_ms']:.4f}",
            "inference_time_min_ms": f"{t['min_ms']:.4f}",
            "inference_time_max_ms": f"{t['max_ms']:.4f}",
            "timed_steps": t["n"],
            "seeds": ",".join(str(x) for x in s["seeds"]),
        })

    with open(OUT_CSV, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for r in rows:
            w.writerow(r)
    print(f"\nWrote {OUT_CSV}")


if __name__ == "__main__":
    main()
