"""Inference-time benchmark for Q3CIBC vs. the IBC paper's inference methods,
configured for the **Simulated Pushing, single target, states** task.

Why this script exists
----------------------
Performance metrics (success rate) are tracked in `results/hyperparam_search/`.
This file is about *raw wall-clock per env step* — the cost of running each
inference algorithm on the same hardware, with networks of the same shape
the paper trained, but **random weights**. Nothing is trained; only the
forward-pass cost of the inference procedure is measured. Comparing apples
to apples for compute efficiency, not for task performance.

Methods benchmarked
-------------------
1. **Q3CIBC (CP-anchored, "pushingA")** — single Langevin chain initialized
   from the highest-Q control point. This is the path that hit 100% in
   `pushing/trials.jsonl` trial #9.
2. **IBC + Langevin** — exact `mlp_ebm_langevin.gin` config:
   MLPEBM ResNetPreAct width=128 depth=16 (= 8 blocks for us),
   `num_action_samples=512`, 100 iters + `optimize_again` (200 total iters
   per chain), `noise_scale=1.0`.
3. **IBC + DFO** — exact `mlp_ebm_best.gin` config:
   MLPEBM ResNetPreAct width=128 depth=8 (= 4 blocks),
   `iterative_dfo(num_action_samples=16384, num_iterations=3)`.
4. **IBC + DFO Autoregressive** — paper §4.3 protocol. Each action dim is
   sampled sequentially. 256 candidates per dim, scored conditional on
   already-selected dims; argmax per dim. (No `pushing_states/*.gin`
   config exists for autoregressive specifically; this uses the same
   MLPEBM as DFO-best for fair compute comparison.)

Results
-------
Per-run record is appended as one JSON line to
    results/hyperparam_search/combinedv2_cpascounter_training/pushing/inference_benchmark.jsonl
with timestamp, device, per-method timings, and param counts. Plain-text
table is still printed to stdout so quick comparisons stay convenient.

What it does NOT measure
------------------------
- Training cost (out of scope).
- Task performance (success rate) — we use random weights.
- CPU→GPU transfer overhead beyond a single warm-up.
- Backwards pass (inference only).

Usage
-----
    uv run --managed-python --extra pushing python bench_inference.py \
        --num-steps 50 \
        --warmup 5 \
        --batch-size 1 \
        --device auto
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import statistics
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch

from utils.models import ControlPointGenerator, QEstimator


RESULTS_PATH = (
    Path(__file__).parent
    / "results"
    / "hyperparam_search"
    / "combinedv2_cpascounter_training"
    / "pushing"
    / "inference_benchmark.jsonl"
)


# ─── Constants matching the Pushing single-target states task ─────────────────
# Observation: 10D × frame_stack=2 = 20D (block_translation, block_orientation,
# effector_translation, effector_target_translation, target_translation,
# target_orientation; tiled for the 2-step history).
OBS_DIM = 20
ACTION_DIM = 2
ACTION_MIN = -1.0  # normalized action space, both Q3CIBC and IBC train here
ACTION_MAX = 1.0


@contextmanager
def cuda_sync(device: torch.device):
    """Synchronize CUDA before and after to get accurate wall-clock timings.

    On CPU this is a no-op. On CUDA, kernel launches are asynchronous so a
    naive `time.perf_counter` call returns before the GPU is done.
    """
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    yield
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def time_block(fn, num_steps: int, warmup: int, device: torch.device) -> dict:
    """Run `fn()` num_steps + warmup times, return timing stats.

    Warmup runs are discarded; they cover lazy CUDA init, cuDNN heuristics
    selecting an algo, and the first JIT compile.
    """
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


# ─── Network builders (random weights, matching paper architectures) ──────────

def build_mlpebm(width: int, depth_blocks: int, device: torch.device) -> QEstimator:
    """Build a Q-network shaped like IBC's MLPEBM (ResNetPreActivation).

    Args:
        width: paper's `MLPEBM.width` (128 for both DFO-best and Langevin).
        depth_blocks: number of residual BLOCKS (each containing 2 Linear
            layers — this is Q3CIBC's `q_depth` convention). To match
            IBC's `MLPEBM.depth=N` (which counts Linears), pass `N // 2`.
            E.g. IBC depth=16 → depth_blocks=8; IBC depth=8 → depth_blocks=4.

    Notes:
        - No spectral norm (paper gin: `ResNetLayer.normalizer = None`).
        - No dropout (paper gin: `MLPEBM.rate = 0.0`).
        - ReLU (paper gin: `MLPEBM.activation = 'relu'`).
    """
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


def build_q3cibc_cpgen(device: torch.device, control_points: int,
                       cp_width: int, cp_depth: int) -> ControlPointGenerator:
    """The CP generator unique to Q3CIBC.

    Sized by the shared --cp-width / --cp-depth CLI args so all Q3CIBC
    methods see the same generator. IBC methods (DFO, Langevin,
    Autoregressive) do not use the CP generator at all.
    """
    return ControlPointGenerator(
        input_dim=OBS_DIM,
        output_dim=ACTION_DIM,
        control_points=control_points,
        hidden_dims=[cp_width] * cp_depth,
        action_bounds=(ACTION_MIN, ACTION_MAX),
        network_kind="mlp",
        width=cp_width,
        depth=cp_depth,
        use_spectral_norm=False,
    ).to(device).eval()


# ─── Inference primitives ─────────────────────────────────────────────────────

def langevin_pass(
    q_net: QEstimator,
    obs: torch.Tensor,            # (1, obs_dim)
    actions: torch.Tensor,        # (1, N, action_dim) — IN PLACE refined
    num_iterations: int,
    lr_init: float,
    lr_final: float,
    decay_power: float,
    delta_clip: float,
    noise_scale: float,
) -> torch.Tensor:
    """Single Langevin chain bank pass.

    Mirrors `mcmc.langevin_actions_given_obs` + `langevin_step` in
    google-research/ibc. Polynomial schedule + per-step Gaussian noise
    scaled by `noise_scale * step_size`.
    """
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


def iterative_dfo_pass(
    q_net: QEstimator,
    obs: torch.Tensor,        # (1, obs_dim)
    num_samples: int,
    num_iterations: int = 3,
    iteration_std: float = 0.33,
    uniform_boundary_buffer: float = 0.05,
) -> torch.Tensor:
    """IBC's `iterative_dfo`: score N samples, resample softmax-weighted, repeat.

    Faithful PyTorch port of `mcmc.iterative_dfo` in google-research/ibc.
    Each iteration:
      1. Forward Q on all N action samples → log_probs.
      2. Resample N from the softmax distribution, ordering the gathered
         samples by category index (mirrors IBC's
         `tf.gather(samples, tf.repeat(arange(N), counts))`).
      3. Add Gaussian noise (std halves per iter).
      4. Clip to action box.

    Two fidelity details matched against `mcmc.iterative_dfo`:
      - Initial samples are drawn from `[min - buffer*range, max + buffer*range]`
        (`tensor_spec.sample_spec_nest` + `uniform_boundary_buffer = 0.05`).
      - Resample output ordering matches IBC's `bincount → tf.repeat`
        convention, so the post-resample multiset is at the same positions as
        IBC's. This matters because the policy's final argmax indexes into
        `action_samples[argmax(log_probs)]` — see `MappedCategorical.mode()`
        in `IbcPolicy._distribution`.
    """
    # ── 1. Initial uniform samples with the 5% boundary buffer ─────────────
    buf = (ACTION_MAX - ACTION_MIN) * uniform_boundary_buffer
    sample_min = ACTION_MIN - buf
    sample_max = ACTION_MAX + buf
    actions = torch.empty(1, num_samples, ACTION_DIM, device=obs.device).uniform_(sample_min, sample_max)
    obs_expanded = obs.unsqueeze(1).expand(-1, num_samples, -1)
    std = iteration_std
    log_probs = None

    for it in range(num_iterations):
        with torch.no_grad():
            energies = q_net(obs_expanded, actions).squeeze(-1)  # (1, N)
        log_probs = energies  # IBC treats `net_logits` as log-probs directly.

        # ── 2. Resample with replacement, in IBC's category-ordered fashion ─
        # `torch.multinomial` draws N indices. We then sort + bincount + repeat
        # to put the gathered samples in the same order tf.gather+tf.repeat
        # produces for IBC: index i in the output is a copy of input index i
        # whenever counts[i] > 0 (otherwise it's the next non-zero category).
        idx = torch.multinomial(
            torch.softmax(log_probs.squeeze(0), dim=-1),
            num_samples,
            replacement=True,
        )
        counts = torch.bincount(idx, minlength=num_samples)  # (N,)
        repeat_indices = torch.repeat_interleave(
            torch.arange(num_samples, device=obs.device), counts
        )
        actions = actions[:, repeat_indices, :]

        # ── 3. Noise + clip for the next iteration (skipped after the last) ─
        if it < num_iterations - 1:
            actions = actions + torch.randn_like(actions) * std
            actions = actions.clamp(ACTION_MIN, ACTION_MAX)
            std *= 0.5

    return actions, log_probs


@dataclass
class BenchConfig:
    """All sizes shared across approaches.

    The point of having one config (driven by CLI args) is that every
    method below uses **the same Q-network and the same CP-generator**.
    That makes the timings directly comparable: any speed difference is
    due to the *algorithm*, not architecture inflation. The paper used
    different MLPEBM depths for DFO (depth=8 = 4 blocks for us) and
    Langevin (depth=16 = 8 blocks), so a strictly paper-faithful
    comparison would use unequal nets — but then "DFO is faster" partly
    just means "smaller net". This script gives the algorithm side a fair
    shake by holding the net constant.
    """
    q_width: int = 128
    q_depth: int = 8
    cp_width: int = 128
    cp_depth: int = 2
    control_points: int = 20
    # DFO refinement defaults match the empirically-validated 100%-success
    # recipe from `pushing/trials.jsonl` (trials 36, 39, 41 — three runs
    # across seeds 0/1/2, training 150k–200k, all hit 50/50 eval seeds).
    # This is THE fastest inference setup that reached 100% on Pushing
    # single-target states; only training budget/seed varies between those
    # three trials. Inference cost: 5 iters × 20 CPs = 100 no-autograd
    # forward passes (vs IBC DFO's 16384 × 3 = 49,152).
    #
    # NOTE on the comparison: IBC's paper Table 3 reports their EBM with
    # DFO inference at 100 ± 0 on this same task (paper used DFO for 2D
    # pushing per Appendix D.2 / p19). So this benchmark compares two
    # methods that BOTH reach 100% on pushing-states; the differentiator
    # is wall-clock per env step at the same Q-net size, not task
    # performance — see the table in the script's stdout.
    dfo_num_iters: int = 5
    dfo_iter_std: float = 0.02
    dfo_std_decay: float = 0.5
    dfo_num_uniform: int = 0  # safety-valve uniform samples mixed in; 0 = pure CP-DFO


# ─── Method 1: Q3CIBC pushingA-style — CP argmax + single Langevin chain ─────

def make_method_q3cibc_pushingA(device: torch.device, cfg: BenchConfig):
    cp_gen = build_q3cibc_cpgen(device, control_points=cfg.control_points,
                                cp_width=cfg.cp_width, cp_depth=cfg.cp_depth)
    q_net = build_mlpebm(width=cfg.q_width, depth_blocks=cfg.q_depth, device=device)
    name = "Q3CIBC pushingA (1 chain from best CP)"

    def select_action():
        obs = torch.randn(1, OBS_DIM, device=device)
        with torch.no_grad():
            cps = cp_gen(obs)  # (1, N_cp, A)
            obs_exp = obs.unsqueeze(1).expand(-1, cps.shape[1], -1)
            q = q_net(obs_exp, cps).squeeze(-1)
            best = q.argmax(dim=1)
            best_cp = cps[0, best[0], :].view(1, 1, -1).clone()
        refined = langevin_pass(
            q_net, obs, best_cp,
            num_iterations=100,
            lr_init=0.1, lr_final=1e-5, decay_power=2.0,
            delta_clip=0.1, noise_scale=0.1,
        )
        return refined[0, 0, :]

    return name, select_action


# ─── Method 2: Q3CIBC + CP-DFO refinement (new, this script) ─────────────────

def make_method_q3cibc_cp_dfo(device: torch.device, cfg: BenchConfig):
    cp_gen = build_q3cibc_cpgen(device, control_points=cfg.control_points,
                                cp_width=cfg.cp_width, cp_depth=cfg.cp_depth)
    q_net = build_mlpebm(width=cfg.q_width, depth_blocks=cfg.q_depth, device=device)
    name = (f"Q3CIBC + CP-DFO ({cfg.dfo_num_iters} iters, std={cfg.dfo_iter_std}, "
            f"decay={cfg.dfo_std_decay}, +{cfg.dfo_num_uniform} uniform)")

    def select_action():
        obs = torch.randn(1, OBS_DIM, device=device)
        with torch.no_grad():
            cps = cp_gen(obs)  # (1, N_cp, A)
        # Optional safety-valve: extra uniform samples for the first iter.
        if cfg.dfo_num_uniform > 0:
            unif = torch.empty(1, cfg.dfo_num_uniform, ACTION_DIM, device=device).uniform_(ACTION_MIN, ACTION_MAX)
            candidates = torch.cat([cps, unif], dim=1)
        else:
            candidates = cps.clone()
        N = candidates.shape[1]
        obs_expanded = obs.unsqueeze(1).expand(-1, N, -1)
        std = cfg.dfo_iter_std
        log_probs = None
        for it in range(cfg.dfo_num_iters):
            with torch.no_grad():
                log_probs = q_net(obs_expanded, candidates).squeeze(-1)  # (1, N)
            # IBC-style category-ordered resampling (see iterative_dfo_pass).
            probs = torch.softmax(log_probs.squeeze(0), dim=-1)
            idx = torch.multinomial(probs, N, replacement=True)
            counts = torch.bincount(idx, minlength=N)
            repeat_idx = torch.repeat_interleave(torch.arange(N, device=device), counts)
            candidates = candidates[:, repeat_idx, :]
            if it < cfg.dfo_num_iters - 1:
                candidates = candidates + torch.randn_like(candidates) * std
                candidates = candidates.clamp(ACTION_MIN, ACTION_MAX)
                std *= cfg.dfo_std_decay
        sel = log_probs.argmax(dim=1)
        return candidates[0, sel[0], :]

    return name, select_action


# ─── Method 3: IBC + Langevin (`mlp_ebm_langevin.gin`) ───────────────────────

def make_method_ibc_langevin(device: torch.device, cfg: BenchConfig):
    # Q-net sized by --q-width / --q-depth (shared). For strictly paper-faithful
    # comparison set them to 128 / 8 (= IBC's depth=16 in their convention).
    q_net = build_mlpebm(width=cfg.q_width, depth_blocks=cfg.q_depth, device=device)
    name = "IBC + Langevin (paper config: 512 chains, 100+100 iters)"

    def select_action():
        obs = torch.randn(1, OBS_DIM, device=device)
        actions = torch.empty(1, 512, ACTION_DIM, device=device).uniform_(ACTION_MIN, ACTION_MAX)
        refined = langevin_pass(
            q_net, obs, actions,
            num_iterations=100,
            lr_init=0.1, lr_final=1e-5, decay_power=2.0,
            delta_clip=0.1, noise_scale=1.0,
        )
        refined = langevin_pass(
            q_net, obs, refined,
            num_iterations=100,
            lr_init=0.1, lr_final=1e-5, decay_power=2.0,
            delta_clip=0.1, noise_scale=1.0,
        )
        with torch.no_grad():
            obs_exp = obs.unsqueeze(1).expand(-1, refined.shape[1], -1)
            q = q_net(obs_exp, refined).squeeze(-1)
            sel = q.argmax(dim=1)
        return refined[0, sel[0], :]

    return name, select_action


# ─── Method 4: IBC + DFO (`mlp_ebm_best.gin`) ────────────────────────────────

def make_method_ibc_dfo(device: torch.device, cfg: BenchConfig):
    # Paper used depth=8 (=4 blocks) for DFO; shared --q-depth overrides.
    q_net = build_mlpebm(width=cfg.q_width, depth_blocks=cfg.q_depth, device=device)
    name = "IBC + DFO (paper best config: 16384 samples × 3 iters)"

    def select_action():
        obs = torch.randn(1, OBS_DIM, device=device)
        actions, log_probs = iterative_dfo_pass(
            q_net, obs, num_samples=16384, num_iterations=3, iteration_std=0.33,
        )
        sel = log_probs.argmax(dim=1)
        return actions[0, sel[0], :]

    return name, select_action


# ─── Method 5: IBC + DFO Autoregressive ──────────────────────────────────────

def make_method_ibc_dfo_autoregressive(device: torch.device, cfg: BenchConfig,
                                       samples_per_dim: int = 256):
    """Autoregressive DFO: dim-by-dim greedy selection.

    The IBC paper (§4.3) introduces autoregressive sampling as an alternative
    inference method, used for Sweeping (3D) and Bi-Manual Sweeping (12D).
    The paper's `pushing_states` task does NOT use autoregressive — it uses
    standard DFO. Included here for completeness as a compute reference.

    Per dim k: sample N candidates, score conditional on already-fixed dims,
    argmax. Total forward passes: action_dim × samples_per_dim.
    """
    q_net = build_mlpebm(width=cfg.q_width, depth_blocks=cfg.q_depth, device=device)
    name = f"IBC + DFO Autoregressive ({samples_per_dim} samples × {ACTION_DIM} dims)"

    def select_action():
        obs = torch.randn(1, OBS_DIM, device=device)
        fixed = torch.zeros(1, ACTION_DIM, device=device)
        for k in range(ACTION_DIM):
            candidates = torch.empty(samples_per_dim, device=device).uniform_(ACTION_MIN, ACTION_MAX)
            action_batch = fixed.unsqueeze(1).expand(-1, samples_per_dim, -1).clone()
            action_batch[0, :, k] = candidates
            obs_exp = obs.unsqueeze(1).expand(-1, samples_per_dim, -1)
            with torch.no_grad():
                q = q_net(obs_exp, action_batch).squeeze(-1)
            sel = q.argmax(dim=1)
            fixed[0, k] = candidates[sel[0]]
        return fixed[0]

    return name, select_action


# ─── Driver ───────────────────────────────────────────────────────────────────

@dataclass
class BenchResult:
    name: str
    timings: dict
    params: int
    notes: str = ""


def param_count(*modules: torch.nn.Module) -> int:
    return sum(p.numel() for m in modules for p in m.parameters())


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--num-steps", type=int, default=50, help="Timed calls per method (default 50)")
    parser.add_argument("--warmup", type=int, default=5, help="Untimed warmup calls (default 5)")
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cuda", "cpu"])
    parser.add_argument("--seed", type=int, default=0)
    # Shared architecture (used by ALL methods so timings reflect algorithm, not net size).
    parser.add_argument("--q-width", type=int, default=128, help="Q-net width (default 128, matches paper)")
    parser.add_argument("--q-depth", type=int, default=8,
                        help="Q-net residual BLOCKS — 8 blocks = paper's MLPEBM.depth=16 (default 8)")
    parser.add_argument("--cp-width", type=int, default=128, help="CP-gen width, Q3CIBC only (default 128)")
    parser.add_argument("--cp-depth", type=int, default=2, help="CP-gen hidden layers, Q3CIBC only (default 2)")
    parser.add_argument("--control-points", type=int, default=20,
                        help="Number of CPs the Q3CIBC generator emits per state (default 20)")
    # CP-DFO refinement knobs. Defaults below come from the FASTEST inference
    # recipe in our trial log that reached 100% success on Pushing (trials 36,
    # 39, 41 in `results/.../pushing/trials.jsonl`). All three differed only
    # in training budget / seed; the inference triple (5 iters, std=0.02,
    # decay=0.5) is the empirical sweet spot.
    parser.add_argument("--dfo-num-iters", type=int, default=5,
                        help="Q3CIBC CP-DFO iterations (default 5 — 100%-recipe value; IBC uses 3 over 16k uniform)")
    parser.add_argument("--dfo-iter-std", type=float, default=0.02,
                        help="Q3CIBC CP-DFO initial Gaussian noise std (default 0.02 — 100%-recipe value; IBC uses 0.33)")
    parser.add_argument("--dfo-std-decay", type=float, default=0.5,
                        help="Q3CIBC CP-DFO noise decay per iter (default 0.5 — 100%-recipe value; IBC also uses 0.5)")
    parser.add_argument("--dfo-num-uniform", type=int, default=0,
                        help="Extra uniform samples mixed into CP-DFO's first iter for safety (default 0)")
    args = parser.parse_args()

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    cfg = BenchConfig(
        q_width=args.q_width,
        q_depth=args.q_depth,
        cp_width=args.cp_width,
        cp_depth=args.cp_depth,
        control_points=args.control_points,
        dfo_num_iters=args.dfo_num_iters,
        dfo_iter_std=args.dfo_iter_std,
        dfo_std_decay=args.dfo_std_decay,
        dfo_num_uniform=args.dfo_num_uniform,
    )

    print("=" * 78)
    print("Inference-time benchmark — Pushing single-target states (random weights)")
    print(f"  device = {device}")
    print(f"  warmup={args.warmup}, timed_steps={args.num_steps}")
    print(f"  Q-net  (shared): resnet width={cfg.q_width} depth_blocks={cfg.q_depth}")
    print(f"  CP-gen (shared): mlp    width={cfg.cp_width} depth={cfg.cp_depth} "
          f"control_points={cfg.control_points}")
    print(f"  CP-DFO knobs: iters={cfg.dfo_num_iters} std0={cfg.dfo_iter_std} "
          f"decay={cfg.dfo_std_decay} +uniform={cfg.dfo_num_uniform}")
    print("=" * 78)
    print()

    methods = [
        make_method_q3cibc_pushingA,
        make_method_q3cibc_cp_dfo,
        make_method_ibc_langevin,
        make_method_ibc_dfo,
        make_method_ibc_dfo_autoregressive,
    ]

    results: list[BenchResult] = []
    for make in methods:
        name, fn = make(device, cfg)
        # Param count: ask the closure's captured modules
        # by walking the closure cells (the dataclass approach is overkill).
        params = sum(p.numel()
                     for cell in fn.__closure__ or ()
                     if hasattr(cell.cell_contents, "parameters")
                     for p in cell.cell_contents.parameters())
        print(f"-- {name}  ({params:,} params)")
        try:
            timings = time_block(fn, args.num_steps, args.warmup, device)
        except Exception as exc:
            print(f"   FAILED: {exc!r}")
            continue
        print(
            f"   mean = {timings['mean_ms']:7.2f} ms   "
            f"median = {timings['median_ms']:7.2f} ms   "
            f"std = {timings['stdev_ms']:6.2f} ms   "
            f"min = {timings['min_ms']:7.2f} ms   "
            f"max = {timings['max_ms']:7.2f} ms"
        )
        results.append(BenchResult(name=name, timings=timings, params=params))

    # Final comparison table sorted by mean.
    print()
    print("=" * 78)
    print("Sorted by mean inference time (cheapest first):")
    print("=" * 78)
    sorted_r = sorted(results, key=lambda r: r.timings["mean_ms"])
    width_name = max(len(r.name) for r in sorted_r)
    fastest = sorted_r[0].timings["mean_ms"]
    print(f"  {'Method':<{width_name}}  {'mean (ms)':>10}  {'×fastest':>9}  {'params':>11}")
    print("  " + "─" * (width_name + 36))
    for r in sorted_r:
        rel = r.timings["mean_ms"] / fastest
        print(f"  {r.name:<{width_name}}  {r.timings['mean_ms']:>10.2f}  {rel:>8.2f}×  {r.params:>11,}")
    print()

    # ── Persist run as one JSONL record alongside the trial logs ────────────
    # Co-located with `pushing/trials.jsonl` so all pushing-task artifacts
    # share a directory. Append-only so successive runs can be compared
    # (e.g. before/after a Q-net architecture change).
    record = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "device": str(device),
        "device_name": (
            torch.cuda.get_device_name(device) if device.type == "cuda" else platform.processor() or platform.machine()
        ),
        "torch_version": torch.__version__,
        "num_steps": args.num_steps,
        "warmup": args.warmup,
        "seed": args.seed,
        "config": {
            "q_width": cfg.q_width, "q_depth": cfg.q_depth,
            "cp_width": cfg.cp_width, "cp_depth": cfg.cp_depth,
            "control_points": cfg.control_points,
            "dfo_num_iters": cfg.dfo_num_iters, "dfo_iter_std": cfg.dfo_iter_std,
            "dfo_std_decay": cfg.dfo_std_decay, "dfo_num_uniform": cfg.dfo_num_uniform,
        },
        "results": [
            {
                "method": r.name,
                "params": r.params,
                **r.timings,
            }
            for r in results
        ],
    }
    RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(RESULTS_PATH, "a") as f:
        f.write(json.dumps(record, sort_keys=True) + "\n")
    print(f"Appended run to {RESULTS_PATH.relative_to(Path(__file__).parent)}")


if __name__ == "__main__":
    main()
