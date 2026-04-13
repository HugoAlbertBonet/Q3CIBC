"""Run IBC-DFO quick experiments (20k steps each), evaluate, and report results.

Runs experiments 5-9 sequentially (each 20k steps instead of 100k for speed),
evaluates each on 50 seeds using the best inference config (noise_scale=0.1),
and prints a comparison table.
"""

import json
import os
import subprocess
import sys
import time
from pathlib import Path

EVAL_SEEDS = 50


def evaluate_checkpoint(ckpt_path, config_path="config_json/config.json"):
    """Evaluate a DFO checkpoint using the eval_dfo_sweep module."""
    from eval_dfo_sweep import (
        load_config, load_dfo_model, evaluate_dfo,
        ObservationNormalizer,
    )
    import torch

    config = load_config()
    env_cfg = config["environments"]["particle"]
    sim_cfg = config.get("simulation", {})
    n_dim = int(env_cfg["n_dim"])
    state_dim = int(env_cfg["state_dim"])
    action_dim = int(env_cfg["action_dim"])
    frame_stack = int(env_cfg.get("frame_stack", 2))
    action_bounds = tuple(env_cfg.get("action_bounds", [0.0, 1.0]))
    max_steps = int(sim_cfg.get("max_episode_steps", 50))
    seeds = list(range(EVAL_SEEDS))
    device = torch.device("cpu")
    env_kwargs = {"n_dim": n_dim, "n_steps": max_steps, "render_mode": None}

    obs_normalizer = ObservationNormalizer(
        env_id=env_cfg["env_id"], device=device,
        frame_stack=frame_stack, particle_n_dim=n_dim,
    )

    model, norm_stats = load_dfo_model(
        ckpt_path, state_dim * frame_stack, action_dim, device,
    )

    langevin_cfg = {
        "num_samples": 512, "num_iterations": 100,
        "lr_init": 0.1, "lr_final": 1e-5,
        "polynomial_decay_power": 2.0, "delta_action_clip": 0.1,
        "noise_scale": 0.1,
    }

    sr = evaluate_dfo(
        model, norm_stats, obs_normalizer, seeds, env_kwargs,
        frame_stack, action_dim, action_bounds, device, langevin_cfg,
    )
    return sr


def main():
    run_ids = [5, 6, 7, 8, 9]
    results = []

    from ibc.ibc_dfo_particle_training import get_run_schedule
    schedule = get_run_schedule()

    for run_id in run_ids:
        trial = schedule[run_id]
        print(f"\n{'='*70}")
        print(f"Training run {run_id}: {trial['name']}")
        print(f"Overrides: {trial['overrides']}")
        print(f"{'='*70}", flush=True)

        t0 = time.time()
        result = subprocess.run(
            [sys.executable, "-m", "ibc.ibc_dfo_particle_training", "--run_id", str(run_id)],
            env={**os.environ, "PYTHONUNBUFFERED": "1"},
        )
        train_time = time.time() - t0

        if result.returncode != 0:
            print(f"  FAILED (exit {result.returncode})")
            results.append({
                "run_id": run_id, "name": trial["name"],
                "success_rate": 0.0, "train_time": train_time,
            })
            continue

        ckpt_dir = f"checkpoints/ibc_dfo/particle/run_{run_id:02d}_{trial['name']}"
        ckpt_path = os.path.join(ckpt_dir, "q_estimator.pt")

        if not os.path.exists(ckpt_path):
            print(f"  Checkpoint not found: {ckpt_path}")
            results.append({
                "run_id": run_id, "name": trial["name"],
                "success_rate": 0.0, "train_time": train_time,
            })
            continue

        print(f"\n  Evaluating on {EVAL_SEEDS} seeds (noise_scale=0.1)...", flush=True)
        t1 = time.time()
        sr = evaluate_checkpoint(ckpt_path)
        eval_time = time.time() - t1

        print(f"  success_rate={sr:.2%}  (train={train_time:.0f}s, eval={eval_time:.0f}s)")
        results.append({
            "run_id": run_id, "name": trial["name"],
            "success_rate": sr, "train_time": train_time, "eval_time": eval_time,
        })

    # Also evaluate existing best (run_01)
    print(f"\n{'='*70}")
    print("Evaluating baseline (run_01) for comparison...")
    print(f"{'='*70}", flush=True)
    sr_baseline = evaluate_checkpoint(
        "checkpoints/ibc_dfo/particle/run_01_baseline_paper/q_estimator.pt"
    )
    print(f"  baseline success_rate={sr_baseline:.2%}")

    print(f"\n{'='*75}")
    print(f"{'Run':<6} {'Name':<28} {'Success Rate':>12} {'Train Time':>12}")
    print("-" * 75)
    print(f"{'1':<6} {'baseline_paper (100k)':<28} {sr_baseline:>11.2%} {'(pretrained)':>12}")
    for r in sorted(results, key=lambda x: -x["success_rate"]):
        print(f"{r['run_id']:<6} {r['name']:<28} {r['success_rate']:>11.2%} {r['train_time']:>11.0f}s")
    print("=" * 75)

    out_path = Path("results/ibc_dfo/particle/quick_experiments.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump({"baseline_sr": sr_baseline, "experiments": results}, f, indent=2)
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()
