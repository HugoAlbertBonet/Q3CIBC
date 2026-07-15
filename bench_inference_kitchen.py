"""Inference-time benchmark for Q3CIBC vs. IBC on D4RL kitchen-complete.

Same-machine wall-clock per env-step for each inference path, at the SAME
IBC-identical Q architecture (ResNetPreActivation, 8 dense layers, width 512,
spectral norm, no trailing activation — the corrected arch from the IBC
audit). Random weights — pure timing. Quality numbers (avg_tasks_completed)
come from already-recorded trials.jsonl rows, filtered to the FINAL stack
(cp200/tk50 + resnet generator + kitchen_qpos_only) so timing and quality
describe the same configs.

Methods
-------
1. **Q3C CP-argmax**            — cp_gen(obs) -> 200 CPs -> Q -> argmax.
2. **Q3C + Langevin 30+20**     — corrected chain (noise linear in stepsize)
                                  on the full CP cloud, lr 0.1, + 20-iter
                                  polish at const 1e-5, argmax re-score.
3. **Q3C + Langevin 50+30**     — same, longer schedule (kcI6 best).
4. **Q3C + CP-DFO (it5 std.05)**— forward-only softmax-resample refinement
                                  (quality rows appear once kcJ9 lands).
5. **IBC full-faithful**        — 512 uniform samples x 100 Langevin iters
                                  (lr 0.5, noise 1.0) + optimize_again 100
                                  iters @1e-5 (kcCibc protocol), argmax.

Rerun after each batch lands to refresh quality columns:
    uv run --managed-python python bench_inference_kitchen.py
    uv run --managed-python python bench_inference_kitchen.py --num-steps 200 --device cuda
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

from utils.models import QEstimator, ControlPointGenerator, _build_backbone
from utils.sampling import sample_langevin


ROOT = Path(__file__).parent
Q3C_TRIALS = ROOT / "results" / "hyperparam_search" / "combinedv2_cpascounter_training" / "d4rl" / "kitchen" / "trials.jsonl"
IBC_TRIALS = ROOT / "results" / "hyperparam_search" / "ibc_dfo_kitchen" / "trials.jsonl"
OUT_CSV = ROOT / "results" / "hyperparam_search" / "combinedv2_cpascounter_training" / "d4rl" / "kitchen" / "kitchen_inference_results.csv"

# Kitchen env constants (FrankaKitchen-v1, qpos-only input per IBC paper).
OBS_DIM = 30            # KITCHEN_QPOS_ONLY: robot qpos(9) + object qpos(21)
ACTION_DIM = 9
ACTION_MIN = -1.0
ACTION_MAX = 1.0

# Shared IBC-identical Q architecture (corrected: resnet, no trailing act).
Q_WIDTH = 512
Q_BLOCKS = 4            # ResNetPreActivationBlock count = 8 dense layers

# Q3C final stack.
Q3C_CP = 200
Q3C_TOPK = 50
CP_WIDTH = 512
CP_BLOCKS = 4           # resnet generator

# Q3C faithful-chain inference settings (kcI winners).
LV_LR_INIT = 0.1
LV_LR_FINAL = 1e-5
LV_DECAY = 2.0
LV_DELTA_CLIP = 0.5
LV_NOISE = 1.0
AGAIN_NOISE = 0.5

# Q3C DFO refinement (kcJ9 config).
DFO_ITERS = 5
DFO_STD = 0.05
DFO_DECAY = 0.5

# Explicit BC (MSE) baseline — IBC paper's own explicit-policy comparison.
# Arch from configs/d4rl/mlp_mse_best.gin: MLPMSE = ResNetPreActivation,
# width 2048, depth 8 dense layers (4 of our 2-layer blocks), dropout 0.1
# (inactive at eval), single forward obs -> action. We never trained MSE-BC
# in our env, so its quality is the PAPER-REPORTED kitchen-complete value
# (legacy env; kitchen's tasks-metric is port-robust, see ibc-repro-fixes).
MSE_WIDTH = 2048
MSE_BLOCKS = 4          # = 8 dense layers
MSE_PAPER_TASKS = 1.76  # IBC paper Table 2, kitchen-complete, BC(MSE)
MSE_PAPER_SEM = 0.04    # paper's +/- (SEM interpretation, 3 seeds)
MSE_PAPER_NSEEDS = 3

# IBC full-faithful inference (kcCibc protocol).
IBC_NUM_SAMPLES = 512
IBC_INF_ITERS = 100
IBC_LR_INIT = 0.5
IBC_LR_FINAL = 1e-5
IBC_DELTA_CLIP = 0.5
IBC_NOISE = 1.0
IBC_AGAIN_ITERS = 100
IBC_AGAIN_NOISE = 0.5
IBC_UNIFORM_BUFFER = 0.05


# ─── Timing helpers (mirror bench_inference_pen.py) ──────────────────────────

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


# ─── Network builders (random weights, corrected shapes) ─────────────────────

def build_q(device: torch.device) -> QEstimator:
    """IBC-identical corrected Q/EBM: resnet d4 (8 dense), SN, no final act."""
    return QEstimator(
        state_dim=OBS_DIM,
        action_dim=ACTION_DIM,
        hidden_dims=[Q_WIDTH] * Q_BLOCKS,
        use_spectral_norm=True,
        network_kind="resnet",
        width=Q_WIDTH,
        depth=Q_BLOCKS,
        resnet_final_activation=False,
    ).to(device).eval()


def build_cpgen(device: torch.device) -> ControlPointGenerator:
    """Final-stack resnet generator."""
    return ControlPointGenerator(
        input_dim=OBS_DIM,
        output_dim=ACTION_DIM,
        control_points=Q3C_CP,
        hidden_dims=[CP_WIDTH] * CP_BLOCKS,
        action_bounds=(ACTION_MIN, ACTION_MAX),
        network_kind="resnet",
        width=CP_WIDTH,
        depth=CP_BLOCKS,
        use_spectral_norm=False,
    ).to(device).eval()


def _bounds(device):
    lo = torch.full((ACTION_DIM,), ACTION_MIN, device=device)
    hi = torch.full((ACTION_DIM,), ACTION_MAX, device=device)
    return lo, hi


def _neg_q(q_net):
    def f(obs_exp, actions):
        return -q_net(obs_exp, actions).squeeze(-1)
    return f


# ─── Methods ─────────────────────────────────────────────────────────────────

def make_method_argmax(device):
    cp_gen, q_net = build_cpgen(device), build_q(device)
    name = f"Q3C CP-argmax (cp={Q3C_CP}, resnet gen, no refinement)"

    def select_action():
        obs = torch.randn(1, OBS_DIM, device=device)
        with torch.no_grad():
            cps = cp_gen(obs)
            obs_exp = obs.unsqueeze(1).expand(-1, cps.shape[1], -1)
            q = q_net(obs_exp, cps).squeeze(-1)
            return cps[0, q.argmax(dim=1)[0], :]

    return name, select_action


def make_method_langevin(device, iters: int, again_iters: int, lr: float = LV_LR_INIT):
    """Faithful chain on the full CP cloud — same sample_langevin the eval uses."""
    cp_gen, q_net = build_cpgen(device), build_q(device)
    lo, hi = _bounds(device)
    energy = _neg_q(q_net)
    name = f"Q3C+Langevin {iters}+{again_iters} (cp={Q3C_CP}, lr {lr}, faithful chain)"

    def select_action():
        obs = torch.randn(1, OBS_DIM, device=device)
        with torch.no_grad():
            cps = cp_gen(obs)
        for p in q_net.parameters():
            p.requires_grad_(False)
        refined = sample_langevin(
            energy_function=energy, observations=obs,
            num_samples=cps.shape[1], action_min=lo, action_max=hi,
            num_iterations=iters, lr_init=lr, lr_final=LV_LR_FINAL,
            polynomial_decay_power=LV_DECAY, delta_action_clip=LV_DELTA_CLIP,
            noise_scale=LV_NOISE, initial_actions=cps.clone(), device=device,
            noise_via_stepsize=True,
        )
        if again_iters > 0:
            refined = sample_langevin(
                energy_function=energy, observations=obs,
                num_samples=cps.shape[1], action_min=lo, action_max=hi,
                num_iterations=again_iters, lr_init=1e-5, lr_final=1e-5,
                polynomial_decay_power=LV_DECAY, delta_action_clip=LV_DELTA_CLIP,
                noise_scale=AGAIN_NOISE, initial_actions=refined, device=device,
                noise_via_stepsize=True,
            )
        with torch.no_grad():
            obs_exp = obs.unsqueeze(1).expand(-1, refined.shape[1], -1)
            q = q_net(obs_exp, refined).squeeze(-1)
            return refined[0, q.argmax(dim=1)[0], :]

    return name, select_action


def make_method_dfo(device):
    cp_gen, q_net = build_cpgen(device), build_q(device)
    name = f"Q3C+CP-DFO (cp={Q3C_CP}, it{DFO_ITERS} std{DFO_STD} dec{DFO_DECAY})"

    def select_action():
        obs = torch.randn(1, OBS_DIM, device=device)
        with torch.no_grad():
            candidates = cp_gen(obs).clone()
            N = candidates.shape[1]
            obs_exp = obs.unsqueeze(1).expand(-1, N, -1)
            std = DFO_STD
            for it in range(DFO_ITERS):
                log_probs = q_net(obs_exp, candidates).squeeze(-1)
                probs = torch.softmax(log_probs.squeeze(0), dim=-1)
                idx = torch.multinomial(probs, N, replacement=True)
                counts = torch.bincount(idx, minlength=N)
                repeat_idx = torch.repeat_interleave(torch.arange(N, device=device), counts)
                candidates = candidates[:, repeat_idx, :]
                if it < DFO_ITERS - 1:
                    candidates = (candidates + torch.randn_like(candidates) * std).clamp(ACTION_MIN, ACTION_MAX)
                    std *= DFO_DECAY
            final = q_net(obs_exp, candidates).squeeze(-1)
            return candidates[0, final.argmax(dim=1)[0], :]

    return name, select_action


def make_method_mse(device):
    """Explicit BC: one forward pass through the paper's MLPMSE arch."""
    net = _build_backbone(
        input_dim=OBS_DIM, output_dim=ACTION_DIM,
        network_kind="resnet", hidden_dims=[MSE_WIDTH] * MSE_BLOCKS,
        width=MSE_WIDTH, depth=MSE_BLOCKS,
        activation=torch.nn.ReLU, use_spectral_norm=False,
        resnet_final_activation=False,
    ).to(device).eval()
    name = f"Explicit BC MSE (paper arch {MSE_WIDTH}x8-dense; quality: paper-reported)"

    def select_action():
        obs = torch.randn(1, OBS_DIM, device=device)
        with torch.no_grad():
            return net(obs)[0].clamp(ACTION_MIN, ACTION_MAX)

    return name, select_action


def make_method_ibc(device):
    """IBC full-faithful (kcCibc protocol): 512 uniform x (100 + again 100)."""
    q_net = build_q(device)
    lo, hi = _bounds(device)
    energy = _neg_q(q_net)
    buf = (ACTION_MAX - ACTION_MIN) * IBC_UNIFORM_BUFFER
    name = (f"IBC full-faithful Langevin ({IBC_INF_ITERS}+{IBC_AGAIN_ITERS} iters x "
            f"{IBC_NUM_SAMPLES} samples)")

    def select_action():
        obs = torch.randn(1, OBS_DIM, device=device)
        for p in q_net.parameters():
            p.requires_grad_(False)
        init = torch.empty(1, IBC_NUM_SAMPLES, ACTION_DIM, device=device).uniform_(
            ACTION_MIN - buf, ACTION_MAX + buf
        )
        refined = sample_langevin(
            energy_function=energy, observations=obs,
            num_samples=IBC_NUM_SAMPLES, action_min=lo, action_max=hi,
            num_iterations=IBC_INF_ITERS, lr_init=IBC_LR_INIT, lr_final=IBC_LR_FINAL,
            polynomial_decay_power=LV_DECAY, delta_action_clip=IBC_DELTA_CLIP,
            noise_scale=IBC_NOISE, initial_actions=init, device=device,
            noise_via_stepsize=True,
        )
        refined = sample_langevin(
            energy_function=energy, observations=obs,
            num_samples=IBC_NUM_SAMPLES, action_min=lo, action_max=hi,
            num_iterations=IBC_AGAIN_ITERS, lr_init=1e-5, lr_final=1e-5,
            polynomial_decay_power=LV_DECAY, delta_action_clip=IBC_DELTA_CLIP,
            noise_scale=IBC_AGAIN_NOISE, initial_actions=refined, device=device,
            noise_via_stepsize=True,
        )
        with torch.no_grad():
            obs_exp = obs.unsqueeze(1).expand(-1, refined.shape[1], -1)
            q = q_net(obs_exp, refined).squeeze(-1)
            return refined[0, q.argmax(dim=1)[0], :]

    return name, select_action


# ─── Quality stats from recorded trials ──────────────────────────────────────

def _load_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(l) for l in open(path) if l.strip()]


def _final_stack(p: dict) -> bool:
    """Q3C final-stack training config (constant across kcI/kcJ)."""
    return (p.get("control_points") == 200
            and p.get("top_k_control_points") == 50
            and p.get("cp_network_kind") == "resnet"
            and p.get("cp_depth") == 4
            and p.get("q_network_kind") == "resnet"
            and p.get("q_depth") == 4
            and p.get("q_resnet_final_activation") is False
            and p.get("kitchen_qpos_only") is True
            and p.get("num_langevin_negatives") == 16
            and p.get("training_steps") == 150000
            and p.get("noisy_expert_count", 0) == 0
            # chunked trials (action_chunk>1) are separate configs — keep the
            # CSV rows single-step so quality matches the timed methods
            and p.get("action_chunk", 1) == 1)


def _dedupe_by_seed(trials: list[dict]) -> list[dict]:
    trials.sort(key=lambda t: int(t.get("trial_id", 0)))
    by_seed: dict = {}
    for t in trials:
        by_seed[(t.get("params") or t.get("hparams") or {}).get("trial_seed")] = t
    return list(by_seed.values())


def _aggregate(trials: list[dict], label: str) -> dict:
    if not trials:
        return {"label": label, "n_seeds": 0, "avg_tasks_completed": None,
                "SEM": None, "cross_seed_std": None, "avg_reward": None,
                "seeds": [], "trial_ids": []}
    vals = [float(t["avg_tasks_completed"]) for t in trials]
    rews = [float(t.get("avg_reward") or 0) for t in trials]
    n = len(vals)
    cross = statistics.stdev(vals) if n > 1 else 0.0
    return {
        "label": label, "n_seeds": n,
        "avg_tasks_completed": statistics.mean(vals),
        "SEM": cross / math.sqrt(n) if n > 1 else 0.0,
        "cross_seed_std": cross,
        "avg_reward": statistics.mean(rews),
        "seeds": sorted((t.get("params") or t.get("hparams") or {}).get("trial_seed") for t in trials),
        "trial_ids": sorted(t.get("trial_id", 0) for t in trials),
    }


def q3c_stats(mode: str) -> dict:
    """mode: 'argmax' | 'lv30' | 'lv50' | 'dfo'."""
    keep = []
    for t in _load_jsonl(Q3C_TRIALS):
        if t.get("training_failed") or t.get("eval_error"):
            continue
        p = t.get("params", {}) or {}
        if not _final_stack(p):
            continue
        lv = p.get("inference_langevin_iterations", 0)
        ag = p.get("inference_langevin_again_iterations", 0)
        lr = float(p.get("inference_langevin_lr_init", 0) or 0)
        dfo = p.get("inference_dfo_iterations", 0)
        if mode == "argmax" and lv == 0 and dfo == 0:
            keep.append(t)
        elif mode == "lv30" and lv == 30 and ag == 20 and lr == 0.1 and dfo == 0:
            keep.append(t)
        elif mode == "lv50" and lv == 50 and ag == 30 and lr == 0.1 and dfo == 0:
            keep.append(t)
        elif (mode == "lv50g" and lv == 50 and ag == 30 and lr == 0.05 and dfo == 0
              and p.get("action_chunk", 1) == 1):
            keep.append(t)
        elif (mode == "dfo" and lv == 0 and dfo == DFO_ITERS
              and float(p.get("inference_dfo_iteration_std", 0)) == DFO_STD):
            keep.append(t)
    return _aggregate(_dedupe_by_seed(keep), label=f"Q3C-{mode}")


def ibc_stats() -> dict:
    """kcCibc full-faithful paper-exact rows (all 5 fixes, ncx=8)."""
    keep = []
    for t in _load_jsonl(IBC_TRIALS):
        if t.get("training_failed") or t.get("eval_error"):
            continue
        h = t.get("hparams") or t.get("params") or {}
        if (h.get("NETWORK_KIND") == "resnet"
                and h.get("RESNET_FINAL_ACTIVATION") is False
                and h.get("KITCHEN_QPOS_ONLY") is True
                and h.get("USE_SQRT_STD") is True
                and h.get("NUM_COUNTER_EXAMPLES") == 8
                and h.get("LANGEVIN_TRAIN_ITERATIONS") == 100
                and h.get("INFERENCE_NUM_ITERATIONS") == 100
                and h.get("INFERENCE_NOISE_VIA_STEPSIZE") is True
                and h.get("INFERENCE_OPTIMIZE_AGAIN") is True
                and h.get("GRADIENT_MARGIN") == 1.0
                and h.get("SOFTMAX_TEMPERATURE") == 1.0):
            keep.append(t)
    return _aggregate(_dedupe_by_seed(keep), label="IBC-full-faithful")


# ─── Diffusion Policy (this repo), matched to Q3C's final kitchen stack ──────

def make_method_dp(device, sampler: str, ddim_steps, name: str):
    """DP inference on kitchen: flat denoiser matched to Q3C's final stack
    (resnet 512x4, qpos-only 30-D obs). SN OFF — it is an IBC energy regularizer
    that chokes eps/v-regression (penDPA->DPB). DDPM full chain or DDIM sub-sampled.
    """
    from utils.diffusion import build_denoiser, build_diffusion, resolve_dp_params
    dp = resolve_dp_params({"model": {"diffusion": {
        "denoiser_network_kind": "resnet", "denoiser_width": 512, "denoiser_depth": 4,
        "denoiser_use_spectral_norm": False, "time_emb_dim": 128,
        "num_train_timesteps": 100, "beta_schedule": "cosine",
    }}})
    model = build_denoiser(OBS_DIM, ACTION_DIM, dp, device).eval()
    diffusion = build_diffusion(dp, device, (ACTION_MIN, ACTION_MAX))

    def select_action():
        obs = torch.randn(1, OBS_DIM, device=device)
        with torch.no_grad():
            if sampler == "ddpm":
                a = diffusion.ddpm_sample(model, obs, ACTION_DIM)
            else:
                a = diffusion.ddim_sample(model, obs, ACTION_DIM, num_steps=ddim_steps)
        return a[0]

    return name, select_action


def _dp_kitchen_trials() -> list[dict]:
    """kitchenDPC: epsilon, qpos-only 30-D, resnet 512x4, 150k — the run matched
    to Q3C's final stack. (Unmatched DPA/DPB rows are excluded on purpose.)"""
    path = (ROOT / "results" / "hyperparam_search" / "diffusion_policy_training"
            / "d4rl" / "kitchen" / "trials.jsonl")
    out = []
    for t in _load_jsonl(path):
        p = t.get("params") or {}
        if (not t.get("training_failed")
                and p.get("kitchen_qpos_only") is True
                and p.get("denoiser_network_kind") == "resnet"
                and p.get("denoiser_width") == 512
                and p.get("denoiser_depth") == 4
                and p.get("training_steps") == 150000
                and p.get("prediction_type") == "epsilon"):
            out.append(t)
    return out


def dp_stats(key: str, label: str) -> dict:
    """Aggregate per-sampler avg_tasks_completed over the matched DP kitchen seeds."""
    remapped = []
    for t in _dp_kitchen_trials():
        v = t.get(f"{key}_avg_tasks_completed")
        if v is None:
            continue
        remapped.append({"avg_tasks_completed": v,
                         "avg_reward": t.get(f"{key}_avg_reward"),
                         "params": t.get("params"), "trial_id": t.get("trial_id")})
    return _aggregate(_dedupe_by_seed(remapped), label)


# ─── Main ────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Kitchen inference benchmark.")
    parser.add_argument("--num-steps", type=int, default=100)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--device", type=str, default="auto", choices=("auto", "cpu", "cuda"))
    args = parser.parse_args()

    device = torch.device(
        ("cuda" if torch.cuda.is_available() else "cpu") if args.device == "auto" else args.device
    )
    print(f"Device: {device}")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(device)}")

    torch.manual_seed(0)
    np.random.seed(0)

    builders = (
        ("mse", make_method_mse(device)),
        ("argmax", make_method_argmax(device)),
        ("lv30", make_method_langevin(device, 30, 20)),
        ("lv50", make_method_langevin(device, 50, 30)),
        ("lv50g", make_method_langevin(device, 50, 30, lr=0.05)),
        ("dfo", make_method_dfo(device)),
        ("ibc", make_method_ibc(device)),
        ("dp_ddpm100", make_method_dp(device, "ddpm", None, "DP + DDPM (100 steps, eps, resnet 512x4)")),
        ("dp_ddim5", make_method_dp(device, "ddim", 5, "DP + DDIM (5 steps, eps, resnet 512x4)")),
        ("dp_ddim10", make_method_dp(device, "ddim", 10, "DP + DDIM (10 steps, eps, resnet 512x4)")),
        ("dp_ddim25", make_method_dp(device, "ddim", 25, "DP + DDIM (25 steps, eps, resnet 512x4)")),
    )
    timed = []
    for key, (name, fn) in builders:
        print(f"\nTiming: {name}")
        st = time_block(fn, num_steps=args.num_steps, warmup=args.warmup, device=device)
        print(f"  mean={st['mean_ms']:.3f}ms  median={st['median_ms']:.3f}ms  "
              f"stdev={st['stdev_ms']:.3f}ms  min={st['min_ms']:.3f}ms  max={st['max_ms']:.3f}ms")
        timed.append((key, name, st))

    quality = {
        # Paper-reported (legacy env): we never trained MSE-BC in our env.
        "mse": {"label": "BC-MSE-paper", "n_seeds": MSE_PAPER_NSEEDS,
                "avg_tasks_completed": MSE_PAPER_TASKS, "SEM": MSE_PAPER_SEM,
                "cross_seed_std": MSE_PAPER_SEM * math.sqrt(MSE_PAPER_NSEEDS),
                "avg_reward": None, "seeds": ["paper"], "trial_ids": ["paper"]},
        "argmax": q3c_stats("argmax"),
        "lv30": q3c_stats("lv30"),
        "lv50": q3c_stats("lv50"),
        "lv50g": q3c_stats("lv50g"),
        "dfo": q3c_stats("dfo"),
        "ibc": ibc_stats(),
        "dp_ddpm100": dp_stats("ddpm", "DP-ddpm100"),
        "dp_ddim5": dp_stats("ddim5", "DP-ddim5"),
        "dp_ddim10": dp_stats("ddim10", "DP-ddim10"),
        "dp_ddim25": dp_stats("ddim25", "DP-ddim25"),
    }
    print("\nQuality (avg_tasks_completed /4, from recorded trials):")
    for k, s in quality.items():
        if s["n_seeds"] == 0:
            print(f"  {k}: NO MATCHING TRIALS (fills in after the batch lands)")
        else:
            print(f"  {k}: n={s['n_seeds']} seeds={s['seeds']} ids={s['trial_ids']} "
                  f"tasks={s['avg_tasks_completed']:.3f} SEM={s['SEM']:.3f}")

    ibc_ms = next(st for k, _, st in timed if k == "ibc")["mean_ms"]
    print("\nSpeedups vs IBC full-faithful:")
    for k, name, st in timed:
        if k != "ibc":
            print(f"  {name}: {ibc_ms / st['mean_ms']:.1f}x faster")

    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    cols = ["method", "n_seeds", "avg_tasks_completed", "SEM", "cross_seed_std",
            "avg_reward", "inference_time_mean_ms", "inference_time_std_ms",
            "inference_time_median_ms", "inference_time_min_ms",
            "inference_time_max_ms", "timed_steps", "seeds", "trial_ids"]
    with open(OUT_CSV, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for k, name, st in timed:
            s = quality[k]
            w.writerow({
                "method": name,
                "n_seeds": s["n_seeds"],
                "avg_tasks_completed": f"{s['avg_tasks_completed']:.4f}" if s["avg_tasks_completed"] is not None else "",
                "SEM": f"{s['SEM']:.4f}" if s["SEM"] is not None else "",
                "cross_seed_std": f"{s['cross_seed_std']:.4f}" if s["cross_seed_std"] is not None else "",
                "avg_reward": f"{s['avg_reward']:.2f}" if s["avg_reward"] is not None else "",
                "inference_time_mean_ms": f"{st['mean_ms']:.4f}",
                "inference_time_std_ms": f"{st['stdev_ms']:.4f}",
                "inference_time_median_ms": f"{st['median_ms']:.4f}",
                "inference_time_min_ms": f"{st['min_ms']:.4f}",
                "inference_time_max_ms": f"{st['max_ms']:.4f}",
                "timed_steps": st["n"],
                "seeds": ",".join(str(x) for x in s["seeds"]),
                "trial_ids": ",".join(str(x) for x in s["trial_ids"]),
            })
    print(f"\nWrote {OUT_CSV}")


if __name__ == "__main__":
    main()
