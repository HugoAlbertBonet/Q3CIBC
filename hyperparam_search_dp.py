"""Diffusion Policy trial driver — DDPM + DDIM eval from ONE trained denoiser.

Sibling of hyperparam_search.py, but for the Diffusion Policy baseline
(`diffusion_policy_training.py` + `utils.diffusion`). Same CLI contract as the
batches expect:

  uv run python hyperparam_search_dp.py diffusion_policy_training.py --run \
      --active-env pushing --fixed-params '{"trial_seed":0, ...}'

Key difference vs hyperparam_search.evaluate_q3c: DDPM and DDIM are NOT separate
trainings — one denoiser checkpoint is rolled out under BOTH samplers in a single
trial, and the trial record carries per-sampler success_rate / avg_reward /
inference-time. Top-level success_rate/avg_reward mirror the DDPM sampler.

Trial bookkeeping (run ids, per-trial config, jsonl logging) is reused verbatim
from hyperparam_search.py so DP trials live in the same results tree, under the
`diffusion_policy_training` script stem.
"""

from __future__ import annotations

import argparse
import inspect
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch

import hyperparam_search as hps  # reuse trial plumbing (config, run ids, logging)
from utils.diffusion import build_denoiser, build_diffusion, resolve_dp_params

ROOT_DIR = hps.ROOT_DIR


# ── Param application (DP: route ALL fixed-params into env_config['training']) ──

def apply_dp_params_to_config(config: dict, params: dict) -> dict:
    from copy import deepcopy
    config = deepcopy(config)
    active_env = config.get("active_env", "pushing")
    tr = config["environments"][active_env].setdefault("training", {})
    for k, v in params.items():
        tr[k] = v
    return config


# ── Simulation factory: override select_action with diffusion sampling ─────────

def _make_dp_simulation_cls(SimulationCls, denoiser, diffusion, action_dim,
                            action_bounds, sampler, ddim_steps, ddim_eta, device):
    """Subclass `SimulationCls`, replacing select_action with a DDPM/DDIM sample.

    The denoiser emits a normalized action in [-1, 1]; we denormalize via the
    base sim's `_denormalize_action` (set up from norm_stats) before env.step,
    exactly as the Q3C eval path does.
    """

    class _DPSimulation(SimulationCls):
        def select_action(self, observation, return_q_range: bool = False):
            obs_tensor = torch.tensor(observation, dtype=torch.float32).unsqueeze(0).to(device)
            obs_tensor = self.obs_normalizer.normalize(obs_tensor)
            with torch.no_grad():
                if sampler == "ddpm":
                    a = diffusion.ddpm_sample(denoiser, obs_tensor, action_dim)
                else:
                    a = diffusion.ddim_sample(denoiser, obs_tensor, action_dim,
                                              num_steps=ddim_steps, eta=ddim_eta)
            action_normalized = a[0].cpu().numpy()
            action_normalized = np.clip(action_normalized, action_bounds[0], action_bounds[1])
            action = self._denormalize_action(action_normalized)
            if return_q_range:
                return action, (0.0, 0.0)
            return action

    return _DPSimulation


def _resolve_simulation_cls(active_env: str):
    if active_env == "pushing":
        from simulations.pushing_simulation import PushingSimulation
        return PushingSimulation
    if active_env == "pushing_multi":
        from simulations.pushing_multi_simulation import PushingMultiSimulation
        return PushingMultiSimulation
    if active_env == "pen":
        from simulations.pen_human_v2_simulation import PenHumanV2Simulation
        return PenHumanV2Simulation
    if active_env == "door":
        from simulations.door_human_v2_simulation import DoorHumanV2Simulation
        return DoorHumanV2Simulation
    if active_env == "kitchen":
        from simulations.kitchen_simulation import KitchenSimulation
        return KitchenSimulation
    if active_env == "libero_goal":
        from simulations.libero_goal_simulation import LiberoGoalSimulation
        return LiberoGoalSimulation
    from simulations.particle_simulation import ParticleSimulation
    return ParticleSimulation


def _summarize(all_results: list[dict]) -> dict:
    successes = [bool(r.get("success", r.get("terminated", False))) for r in all_results]
    rewards = [float(r.get("total_reward", 0.0)) for r in all_results]
    ep_len = [int(r.get("episode_length", 0)) for r in all_results]
    return {
        "success_rate": float(np.mean(successes)) if successes else 0.0,
        "success_rate_std": float(np.std(successes)) if successes else 0.0,
        "avg_reward": float(np.mean(rewards)) if rewards else 0.0,
        "std_reward": float(np.std(rewards)) if rewards else 0.0,
        "median_reward": float(np.median(rewards)) if rewards else 0.0,
        "avg_episode_length": float(np.mean(ep_len)) if ep_len else 0.0,
    }


def evaluate_dp(checkpoint_dir: str, config: dict) -> dict:
    """Load denoiser; roll out DDPM + each DDIM step count; return per-sampler metrics."""
    active_env = config.get("active_env", "pushing")
    env_config = config["environments"][active_env]
    sim_config = config.get("simulation", {})
    training_shared = config.get("training_shared", {})

    SimulationCls = _resolve_simulation_cls(active_env)

    action_dim = env_config["action_dim"]
    frame_stack = env_config.get("frame_stack", 1)
    action_bounds = tuple(env_config.get("action_bounds", [-1.0, 1.0]))
    max_episode_steps = env_config.get(
        "max_episode_steps", sim_config.get("max_episode_steps", 50)
    )
    num_seeds = int(env_config.get(
        "num_eval_seeds",
        sim_config.get("num_seeds", len(sim_config.get("default_seeds", [0]))),
    ))
    if num_seeds <= 0:
        raise ValueError("num_eval_seeds must be >= 1")
    seeds = list(range(num_seeds))

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dp = resolve_dp_params(env_config, training_shared)

    # norm_stats — written by diffusion_policy_training (same schema as Q3C).
    norm_stats_path = os.path.join(checkpoint_dir, "norm_stats.pt")
    norm_stats = torch.load(norm_stats_path, weights_only=False) if os.path.exists(norm_stats_path) else None
    if norm_stats is not None and "state_shape" in norm_stats:
        state_dim = int(norm_stats["state_shape"])
    else:
        state_dim = int(env_config["state_dim"]) * frame_stack

    # Prefer EMA weights when present.
    ema_path = os.path.join(checkpoint_dir, "denoiser_ema.pt")
    den_path = os.path.join(checkpoint_dir, "denoiser.pt")
    load_path = ema_path if os.path.exists(ema_path) else den_path
    if not os.path.exists(load_path):
        return {"success_rate": 0.0, "avg_reward": 0.0,
                "error": f"denoiser checkpoint not found in {checkpoint_dir}"}

    denoiser = build_denoiser(state_dim, action_dim, dp, device)
    denoiser.load_state_dict(torch.load(load_path, map_location=device, weights_only=True))
    denoiser.eval()
    diffusion = build_diffusion(dp, device, action_bounds)

    # Base kwargs filtered to whatever the sim's __init__ actually accepts.
    base_kwargs = {
        "control_point_generator": denoiser,  # unused (select_action overridden)
        "q_estimator": denoiser,              # unused
        "device": device,
        "max_episode_steps": max_episode_steps,
        "frame_stack": frame_stack,
        "norm_stats": norm_stats,
        "goal_dist_tolerance": env_config.get("goal_dist_tolerance", 0.02),
        "particle_n_dim": env_config.get("n_dim"),
    }
    sig = inspect.signature(SimulationCls.__init__)
    sim_kwargs = {k: v for k, v in base_kwargs.items() if k in sig.parameters}

    # Sampler specs: DDPM (full chain) + each requested DDIM step count.
    specs: list[tuple[str, str, int | None]] = []
    if dp["eval_ddpm"]:
        specs.append(("ddpm", "ddpm", None))
    for n in dp["ddim_eval_steps"]:
        specs.append((f"ddim{int(n)}", "ddim", int(n)))

    combined: dict = {"num_seeds": num_seeds}
    primary: dict | None = None
    per_seed_primary: list = []

    for name, sampler, n_steps in specs:
        DPSim = _make_dp_simulation_cls(
            SimulationCls, denoiser, diffusion, action_dim, action_bounds,
            sampler, n_steps, dp["ddim_eta"], device,
        )
        sim = DPSim(**sim_kwargs)
        results = []
        t0 = time.perf_counter()
        for seed in seeds:
            results.append(sim.run_episode(seed=seed))
        elapsed = time.perf_counter() - t0
        if hasattr(sim, "close"):
            sim.close()

        summ = _summarize(results)
        total_steps = sum(int(r.get("episode_length", 0)) for r in results)
        ms_per_step = (elapsed * 1000.0 / total_steps) if total_steps else 0.0

        combined[f"{name}_success_rate"] = summ["success_rate"]
        combined[f"{name}_avg_reward"] = summ["avg_reward"]
        combined[f"{name}_std_reward"] = summ["std_reward"]
        combined[f"{name}_avg_episode_length"] = summ["avg_episode_length"]
        combined[f"{name}_ms_per_step"] = round(ms_per_step, 3)
        print(f"    [{name}] success_rate={summ['success_rate']:.2%} "
              f"avg_reward={summ['avg_reward']:.3f} {ms_per_step:.2f} ms/step")

        if primary is None:  # DDPM if present, else first DDIM
            primary = summ
            per_seed_primary = [
                {"seed": seeds[i], "success": bool(r.get("success", r.get("terminated", False))),
                 "reward": float(r.get("total_reward", 0.0)),
                 "episode_length": int(r.get("episode_length", 0))}
                for i, r in enumerate(results)
            ]

    combined["success_rate"] = primary["success_rate"] if primary else 0.0
    combined["avg_reward"] = primary["avg_reward"] if primary else 0.0
    combined["std_reward"] = primary["std_reward"] if primary else 0.0
    combined["per_seed"] = per_seed_primary
    return combined


# ── Single trial (mirrors hyperparam_search.run_single_trial) ─────────────────

def run_single_trial_dp(script_path: Path, params: dict, active_env_override: str | None,
                        training_steps_override: int | None, timeout: int | None) -> dict:
    script_name = script_path.name
    run_id = hps._new_run_id()
    print(f"\n{'=' * 80}\nDP RUN {run_id} — {script_name}\n{'=' * 80}")
    print(f"Parameters:\n{json.dumps(params, indent=2)}")

    config = hps.load_config()
    if active_env_override is not None:
        if active_env_override not in config.get("environments", {}):
            raise ValueError(f"--active-env {active_env_override!r} not in config.json")
        config["active_env"] = active_env_override
    config = apply_dp_params_to_config(config, params)
    checkpoint_dir = hps.set_run_checkpoint_dir(config, run_id)
    active_env = config.get("active_env", "pushing")

    if training_steps_override is not None:
        config["environments"][active_env].setdefault("training", {})["training_steps"] = training_steps_override
    actual_steps = (config["environments"][active_env].get("training", {})
                    .get("training_steps", config.get("training_shared", {}).get("training_steps", 100000)))

    os.makedirs(checkpoint_dir, exist_ok=True)
    trial_config_path = Path(checkpoint_dir) / "config.json"
    with open(trial_config_path, "w") as f:
        json.dump(config, f, indent=4)

    print(f"\n  Training ({actual_steps} steps) — config at {trial_config_path}")
    success, stdout, duration = hps.run_training(
        script_path, timeout=timeout, env_extras={"Q3C_CONFIG_PATH": str(trial_config_path)},
    )
    timestamp = datetime.now(timezone.utc).isoformat()

    if not success:
        print(f"\n  Training FAILED after {duration:.0f}s")
        record = {
            "run_id": run_id, "script": script_name, "active_env": active_env,
            "params": params, "training_steps": actual_steps,
            "duration_seconds": round(duration, 1), "success_rate": 0.0, "avg_reward": 0.0,
            "training_failed": True, "error": "\n".join(stdout.strip().splitlines()[-5:])[-300:],
            "checkpoint_dir": checkpoint_dir, "timestamp": timestamp,
        }
        trial_id = hps.append_trial(script_name, record, active_env=active_env)
        print(f"  Recorded as trial #{trial_id}")
        return record

    print(f"\n  Training completed in {duration:.0f}s\n  Evaluating (DDPM + DDIM)...")
    train_metrics = hps.extract_final_metrics(stdout)
    try:
        eval_results = evaluate_dp(checkpoint_dir, config)
    except Exception as exc:
        eval_results = {"success_rate": 0.0, "avg_reward": 0.0,
                        "error": f"Evaluation failed: {exc}", "per_seed": []}
        print(f"  Evaluation failed: {exc}")

    env_specific = {k: v for k, v in eval_results.items()
                    if k not in ("per_seed", "error", "success_rate", "avg_reward")}
    record = {
        "run_id": run_id, "script": script_name, "active_env": active_env,
        "params": params, "training_steps": actual_steps,
        "duration_seconds": round(duration, 1),
        "success_rate": eval_results.get("success_rate", 0.0),
        "avg_reward": eval_results.get("avg_reward", 0.0),
        **env_specific, **train_metrics,
        "eval_details": eval_results.get("per_seed", []),
        "eval_error": eval_results.get("error"),
        "checkpoint_dir": checkpoint_dir, "timestamp": timestamp,
    }
    trial_id = hps.append_trial(script_name, record, active_env=active_env)
    print(f"\n  Result (trial #{trial_id}): "
          f"DDPM success_rate={record['success_rate']:.2%}, avg_reward={record['avg_reward']:.3f}")
    return record


def main() -> None:
    parser = argparse.ArgumentParser(description="Diffusion Policy trial driver (DDPM + DDIM).")
    parser.add_argument("script", type=str, help="DP training script (diffusion_policy_training.py)")
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--run", action="store_true", help="Run a single trial")
    mode.add_argument("--analyze", action="store_true", help="Print summary of past trials")
    parser.add_argument("--params", type=str, default=None)
    parser.add_argument("--fixed-params", type=str, default=None)
    parser.add_argument("--reduced-steps", type=int, default=None)
    parser.add_argument("--timeout", type=int, default=None)
    parser.add_argument("--num-reps", type=int, default=1)
    parser.add_argument("--active-env", type=str, default=None)
    parser.add_argument("--min-trial-id", type=int, default=0)
    args = parser.parse_args()

    script_path = ROOT_DIR / args.script
    if not script_path.exists():
        print(f"Error: script not found at {script_path}")
        sys.exit(1)
    script_name = Path(args.script).name

    if args.analyze:
        hps.print_analysis(script_name, active_env=args.active_env, min_trial_id=args.min_trial_id)
        return

    params: dict = {}
    if args.params:
        params.update(json.loads(args.params))
    if args.fixed_params:
        params.update(json.loads(args.fixed_params))

    seed_pinned = "trial_seed" in params
    for rep in range(max(1, args.num_reps)):
        rep_params = dict(params)
        if not seed_pinned:
            rep_params["trial_seed"] = rep
        if args.num_reps > 1:
            print(f"\n[rep {rep + 1}/{args.num_reps}] trial_seed={rep_params['trial_seed']}")
        run_single_trial_dp(
            script_path=script_path, params=rep_params,
            active_env_override=args.active_env,
            training_steps_override=args.reduced_steps, timeout=args.timeout,
        )


if __name__ == "__main__":
    main()
