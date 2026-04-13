"""Evaluate IBC-DFO checkpoints with configurable inference-time Langevin params.

Runs DFO-only evaluation over 50 seeds for each (checkpoint, langevin_config) combo.
Much faster than run_particle_success_rate_comparison since it skips Q3C evaluation.

Usage:
    python eval_dfo_sweep.py
"""

import json
import os
import sys
import time
from collections import deque
from pathlib import Path

import numpy as np
import torch

from simulations.particle_env import ParticleEnv
from utils.models import QEstimator
from utils.normalizations import ObservationNormalizer
from utils.sampling import sample_langevin

CONFIG_PATH = Path("config_json/config.json")


def load_config():
    with open(CONFIG_PATH) as f:
        return json.load(f)


def infer_hidden_dims(state_dict):
    indices = sorted({
        int(k.split(".")[1])
        for k in state_dict
        if k.startswith("network.") and k.endswith(".weight")
    })
    return [int(state_dict[f"network.{i}.weight"].shape[0]) for i in indices[:-1]]


def load_dfo_model(checkpoint_path, state_dim, action_dim, device):
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    norm_stats = None
    if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
        sd = ckpt["model_state_dict"]
        norm_stats = ckpt.get("norm_stats")
    else:
        sd = ckpt
    try:
        hidden = infer_hidden_dims(sd)
    except Exception:
        hidden = [256, 256]
    model = QEstimator(state_dim=state_dim, action_dim=action_dim, hidden_dims=hidden)
    model.load_state_dict(sd)
    model.to(device).eval()
    return model, norm_stats


def denormalize_action(action_norm, norm_stats):
    if norm_stats is None:
        return action_norm
    act_min = np.asarray(norm_stats["act_min"], dtype=np.float32)
    act_max = np.asarray(norm_stats["act_max"], dtype=np.float32)
    rng = np.where((act_max - act_min) == 0, 1.0, act_max - act_min)
    return action_norm * rng + act_min


def select_action_dfo(stacked_obs, model, obs_normalizer, norm_stats,
                      action_bounds, action_dim, device, langevin_cfg):
    state_t = torch.from_numpy(stacked_obs).float().unsqueeze(0).to(device)
    state_n = obs_normalizer.normalize(state_t)
    buf = 0.05
    norm_min = torch.full((action_dim,), -buf, device=device)
    norm_max = torch.full((action_dim,), 1.0 + buf, device=device)

    samples = sample_langevin(
        energy_function=model,
        observations=state_n,
        num_samples=langevin_cfg["num_samples"],
        action_min=norm_min,
        action_max=norm_max,
        num_iterations=langevin_cfg["num_iterations"],
        lr_init=langevin_cfg["lr_init"],
        lr_final=langevin_cfg["lr_final"],
        polynomial_decay_power=langevin_cfg.get("polynomial_decay_power", 2.0),
        delta_action_clip=langevin_cfg.get("delta_action_clip", 0.1),
        noise_scale=langevin_cfg.get("noise_scale", 1.0),
        device=device,
    )

    with torch.no_grad():
        se = state_n.unsqueeze(1).expand(-1, samples.shape[1], -1)
        energies = model(se, samples).squeeze(-1)
        best = samples[0, energies.argmin(dim=-1)[0]].cpu().numpy()

    action = denormalize_action(best, norm_stats)
    return np.clip(action, action_bounds[0], action_bounds[1])


def evaluate_dfo(model, norm_stats, obs_normalizer, seeds, env_kwargs,
                 frame_stack, action_dim, action_bounds, device, langevin_cfg):
    successes = 0
    for seed in seeds:
        env = ParticleEnv(**env_kwargs)
        obs, _ = env.reset(seed=seed)
        buf = deque(maxlen=frame_stack)
        for _ in range(frame_stack):
            buf.append(obs.copy())

        done = False
        success = False
        while not done:
            stacked = np.concatenate(list(buf)) if frame_stack > 1 else buf[-1]
            action = select_action_dfo(
                stacked, model, obs_normalizer, norm_stats,
                action_bounds, action_dim, device, langevin_cfg,
            )
            obs, _, terminated, truncated, info = env.step(action)
            buf.append(obs.copy())
            done = terminated or truncated
            success = bool(info.get("success", False))
        if success:
            successes += 1
        env.close()
    return successes / max(len(seeds), 1)


def main():
    config = load_config()
    env_cfg = config["environments"]["particle"]
    sim_cfg = config.get("simulation", {})

    n_dim = int(env_cfg["n_dim"])
    state_dim = int(env_cfg["state_dim"])
    action_dim = int(env_cfg["action_dim"])
    frame_stack = int(env_cfg.get("frame_stack", 2))
    action_bounds = tuple(env_cfg.get("action_bounds", [0.0, 1.0]))
    max_steps = int(sim_cfg.get("max_episode_steps", 50))
    num_seeds = int(sim_cfg.get("num_seeds", 50))
    seeds = list(range(num_seeds))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    env_kwargs = {"n_dim": n_dim, "n_steps": max_steps, "render_mode": None}

    obs_normalizer = ObservationNormalizer(
        env_id=env_cfg["env_id"], device=device,
        frame_stack=frame_stack, particle_n_dim=n_dim,
    )

    checkpoints = [
        ("run_01_baseline", "checkpoints/ibc_dfo/particle/run_01_baseline_paper/q_estimator.pt"),
        ("run_02_CE16", "checkpoints/ibc_dfo/particle/run_02_counter_examples_16/q_estimator.pt"),
        ("run_03_CE32", "checkpoints/ibc_dfo/particle/run_03_counter_examples_32/q_estimator.pt"),
        ("run_04_CE64", "checkpoints/ibc_dfo/particle/run_04_counter_examples_64/q_estimator.pt"),
    ]

    DEFAULT_LANGEVIN = {
        "num_samples": 512, "num_iterations": 100,
        "lr_init": 0.1, "lr_final": 1e-5,
        "polynomial_decay_power": 2.0, "delta_action_clip": 0.1, "noise_scale": 1.0,
    }

    # ── Phase 1: find best checkpoint with default inference ──
    print("=" * 75)
    print("PHASE 1: Evaluate all checkpoints with default inference")
    print("=" * 75)
    results = []
    best_ckpt_name = None
    best_ckpt_sr = -1.0
    best_ckpt_path = None

    for i, (ckpt_name, ckpt_path) in enumerate(checkpoints):
        if not os.path.exists(ckpt_path):
            print(f"SKIP {ckpt_name}: not found")
            continue
        model, norm_stats = load_dfo_model(ckpt_path, state_dim * frame_stack, action_dim, device)
        print(f"[{i+1}/{len(checkpoints)}] {ckpt_name} + default ...", end=" ", flush=True)
        t0 = time.time()
        sr = evaluate_dfo(
            model, norm_stats, obs_normalizer, seeds, env_kwargs,
            frame_stack, action_dim, action_bounds, device, DEFAULT_LANGEVIN,
        )
        elapsed = time.time() - t0
        print(f"success_rate={sr:.2%}  ({elapsed:.0f}s)")
        results.append({"checkpoint": ckpt_name, "inference": "default",
                        "success_rate": sr, "time_s": elapsed})
        if sr > best_ckpt_sr:
            best_ckpt_sr = sr
            best_ckpt_name = ckpt_name
            best_ckpt_path = ckpt_path

    print(f"\nBest checkpoint: {best_ckpt_name} ({best_ckpt_sr:.2%})")

    # ── Phase 2: tune inference on best checkpoint ──
    print("\n" + "=" * 75)
    print(f"PHASE 2: Inference tuning on {best_ckpt_name}")
    print("=" * 75)

    inference_variants = [
        ("samples_1024", {**DEFAULT_LANGEVIN, "num_samples": 1024}),
        ("samples_2048", {**DEFAULT_LANGEVIN, "num_samples": 2048}),
        ("iters_200", {**DEFAULT_LANGEVIN, "num_iterations": 200}),
        ("low_noise_0.5", {**DEFAULT_LANGEVIN, "noise_scale": 0.5}),
        ("low_noise_0.1", {**DEFAULT_LANGEVIN, "noise_scale": 0.1}),
        ("combo_1024_i200_n0.5", {
            **DEFAULT_LANGEVIN, "num_samples": 1024,
            "num_iterations": 200, "noise_scale": 0.5,
        }),
    ]

    model, norm_stats = load_dfo_model(best_ckpt_path, state_dim * frame_stack, action_dim, device)

    for i, (lcfg_name, lcfg) in enumerate(inference_variants):
        print(f"[{i+1}/{len(inference_variants)}] {best_ckpt_name} + {lcfg_name} ...", end=" ", flush=True)
        t0 = time.time()
        sr = evaluate_dfo(
            model, norm_stats, obs_normalizer, seeds, env_kwargs,
            frame_stack, action_dim, action_bounds, device, lcfg,
        )
        elapsed = time.time() - t0
        print(f"success_rate={sr:.2%}  ({elapsed:.0f}s)")
        results.append({"checkpoint": best_ckpt_name, "inference": lcfg_name,
                        "success_rate": sr, "time_s": elapsed})

    print("\n" + "=" * 75)
    print(f"{'Checkpoint':<20} {'Inference':<16} {'Success Rate':>12} {'Time':>8}")
    print("-" * 75)
    for r in sorted(results, key=lambda x: -x["success_rate"]):
        print(f"{r['checkpoint']:<20} {r['inference']:<16} {r['success_rate']:>11.2%} {r['time_s']:>7.0f}s")
    print("=" * 75)

    out_path = Path("results/ibc_dfo/particle/inference_sweep.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()
