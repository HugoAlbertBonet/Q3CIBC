"""Agent-assisted hyperparameter search for Q3C-IBC training scripts.

Runs training trials with different hyperparameter configurations, evaluates
success rates, and supports iterative refinement guided by AI analysis.

Modes:
    --run               Run a single trial (with --params or auto-suggested)
    --auto              Run multiple trials with adaptive exploration
    --analyze           Print summary table of all past trials

Usage:
    python hyperparam_search.py combinedv2_cpascounter_training.py --run
    python hyperparam_search.py combinedv2_cpascounter_training.py --run --params '{"learning_rate": 5e-4}'
    python hyperparam_search.py combinedv2_cpascounter_training.py --auto --max-trials 5
    python hyperparam_search.py combinedv2_cpascounter_training.py --analyze
    python hyperparam_search.py combinedv2_cpascounter_training.py --auto --max-trials 3 --reduced-steps 20000
"""

from __future__ import annotations

import argparse
import json
import os
import random
import re
import shutil
import subprocess
import sys
import time
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch

ROOT_DIR = Path(__file__).parent
sys.path.insert(0, str(ROOT_DIR))

CONFIG_PATH = ROOT_DIR / "config_json" / "config.json"
CONFIG_BACKUP_PATH = ROOT_DIR / "config_json" / "config.json.bak"
RESULTS_BASE_DIR = ROOT_DIR / "results" / "hyperparam_search"
CHECKPOINTS_BASE_DIR = ROOT_DIR / "checkpoints" / "hpsearch"

# ─── Search space: param_name -> {values, type, location} ────────────────────
# location: "env_training" = environments.<active_env>.training
#            "training_shared" = training_shared
#            "env_model"       = environments.<active_env>.model
SEARCH_SPACE: dict[str, dict] = {
    "control_points": {
        "values": [20, 30, 50, 75, 100],
        "type": "int",
        "location": "env_model",
    },
    "learning_rate": {
        "values": [1e-4, 3e-4, 5e-4, 1e-3, 2e-3, 5e-3],
        "type": "float",
        "location": "env_training",
    },
    "batch_size": {
        "values": [128, 256, 512],
        "type": "int",
        "location": "env_training",
    },
    "counter_examples": {
        "values": [8, 16, 32, 64],
        "type": "int",
        "location": "env_training",
    },
    "top_k_control_points": {
        "values": [10, 20, 50, 70],
        "type": "int",
        "location": "env_training",
    },
    "separation_weight": {
        "values": [0.01, 0.05, 0.1, 0.2],
        "type": "float",
        "location": "training_shared",
    },
    "mse_weight": {
        "values": [1.0, 3.0, 5.0, 10.0],
        "type": "float",
        "location": "training_shared",
    },
    "info_nce_weight": {
        "values": [0.5, 1.0, 2.0],
        "type": "float",
        "location": "training_shared",
    },
    "generator_infonce_weight": {
        "values": [0.01, 0.05, 0.1, 0.2],
        "type": "float",
        "location": "training_shared",
    },
    "training_steps": {
        "values": [50000, 100000, 200000],
        "type": "int",
        "location": "env_training",
    },
    "separation_loss": {
        "values": ["separation", "entropy"],
        "type": "str",
        "location": "env_training",
    },
    "entropy_bandwidth": {
        "values": [0.05, 0.1, 0.2],
        "type": "float",
        "location": "env_training",
    },
    "num_hidden_layers": {
        "values": [2, 4, 8],
        "type": "int",
        "location": "env_model",
    },
    "num_neurons": {
        "values": [128, 256, 512],
        "type": "int",
        "location": "env_model",
    },
    "estimator_learning_rate": {
        "values": [1e-4, 3e-4, 5e-4, 1e-3],
        "type": "float",
        "location": "env_training",
    },
    "scheduler_type": {
        "values": ["cosine", "cosine_warm_restarts"],
        "type": "str",
        "location": "env_training",
    },
    "cosine_t0": {
        "values": [25000, 50000, 100000],
        "type": "int",
        "location": "env_training",
    },
    "infonce_logit_clamp": {
        "values": [10.0, 20.0, 30.0, 50.0],
        "type": "float",
        "location": "env_training",
    },
    "use_spectral_norm": {
        "values": [True, False],
        "type": "bool",
        "location": "env_model",
    },
    "cosine_t_max": {
        "values": [100000, 150000, 200000],
        "type": "int",
        "location": "env_training",
    },
}


# ─── Config I/O ──────────────────────────────────────────────────────────────

def load_config() -> dict:
    with open(CONFIG_PATH, "r") as f:
        return json.load(f)


def save_config(config: dict) -> None:
    with open(CONFIG_PATH, "w") as f:
        json.dump(config, f, indent=4)


def backup_config() -> None:
    shutil.copy2(CONFIG_PATH, CONFIG_BACKUP_PATH)


def restore_config() -> None:
    if CONFIG_BACKUP_PATH.exists():
        shutil.copy2(CONFIG_BACKUP_PATH, CONFIG_PATH)
        CONFIG_BACKUP_PATH.unlink()


# ─── Trials I/O ──────────────────────────────────────────────────────────────

def _results_dir(script_name: str) -> Path:
    config = load_config()
    active_env = config.get("active_env", "particle")
    env_cfg = config.get("environments", {}).get(active_env, {})
    results_dir = RESULTS_BASE_DIR / Path(script_name).stem / active_env

    # For particle experiments, partition trials by n_dim to avoid mixing runs.
    if "n_dim" in env_cfg:
        results_dir = results_dir / str(env_cfg["n_dim"])

    return results_dir


def _trials_path(script_name: str) -> Path:
    return _results_dir(script_name) / "trials.jsonl"


def load_trials(script_name: str) -> list[dict]:
    path = _trials_path(script_name)
    if not path.exists():
        return []
    trials = []
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if line:
                trials.append(json.loads(line))
    return trials


def append_trial(script_name: str, record: dict) -> None:
    path = _trials_path(script_name)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a") as f:
        f.write(json.dumps(record, sort_keys=True, default=str) + "\n")


def next_trial_id(script_name: str) -> int:
    trials = load_trials(script_name)
    if not trials:
        return 1
    return max(t["trial_id"] for t in trials) + 1


# ─── Hyperparameter detection ────────────────────────────────────────────────

def detect_script_params(script_path: Path) -> list[str]:
    """Scan training script source to find which search-space params it reads."""
    with open(script_path, "r") as f:
        source = f.read()
    detected = []
    for param_name in SEARCH_SPACE:
        if re.search(rf'["\']({re.escape(param_name)})["\']', source):
            detected.append(param_name)
    return detected


def get_baseline_params(config: dict, detected_params: list[str]) -> dict:
    """Read current config values for the detected params."""
    active_env = config.get("active_env", "particle")
    env_training = config["environments"][active_env].get("training", {})
    env_model = config["environments"][active_env].get("model", {})
    training_shared = config.get("training_shared", {})

    baseline: dict = {}
    for param in detected_params:
        space = SEARCH_SPACE[param]
        if space["location"] == "env_model":
            val = env_model.get(param)
        elif space["location"] == "env_training":
            val = env_training.get(param, training_shared.get(param))
        else:
            val = training_shared.get(param)
        if val is not None:
            baseline[param] = val
    return baseline


def apply_params_to_config(config: dict, params: dict) -> dict:
    """Return a deep copy of *config* with hyperparameter overrides applied."""
    config = deepcopy(config)
    active_env = config.get("active_env", "particle")
    for param, value in params.items():
        if param not in SEARCH_SPACE:
            continue
        space = SEARCH_SPACE[param]
        if space["location"] == "env_model":
            config["environments"][active_env].setdefault("model", {})[param] = value
        elif space["location"] == "env_training":
            config["environments"][active_env].setdefault("training", {})[param] = value
        else:
            config.setdefault("training_shared", {})[param] = value
    return config


def set_trial_checkpoint_dir(config: dict, trial_id: int) -> str:
    """Point model_save_dir to a trial-specific directory and return the path."""
    trial_dir = str(CHECKPOINTS_BASE_DIR / f"trial_{trial_id:03d}")
    config.setdefault("training_shared", {})["model_save_dir"] = trial_dir
    return trial_dir


# ─── Training subprocess ─────────────────────────────────────────────────────

def run_training(
    script_path: Path,
    timeout: int | None = None,
) -> tuple[bool, str, float]:
    """Run a training script as a subprocess, streaming output live.

    Returns (success, captured_stdout, duration_seconds).
    """
    start = time.time()
    stdout_lines: list[str] = []
    env = {**os.environ, "WANDB_MODE": "disabled", "PYTHONUNBUFFERED": "1"}

    try:
        proc = subprocess.Popen(
            [sys.executable, str(script_path)],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            cwd=str(ROOT_DIR),
            env=env,
        )
        assert proc.stdout is not None
        for line in proc.stdout:
            sys.stdout.write(f"  | {line}")
            sys.stdout.flush()
            stdout_lines.append(line)
        proc.wait(timeout=timeout)
        duration = time.time() - start
        return proc.returncode == 0, "".join(stdout_lines), duration
    except subprocess.TimeoutExpired:
        proc.kill()  # type: ignore[possibly-undefined]
        duration = time.time() - start
        return False, "".join(stdout_lines) + "\n[TIMED OUT]", duration
    except Exception as exc:
        duration = time.time() - start
        return False, "".join(stdout_lines) + f"\n[ERROR: {exc}]", duration


def extract_final_metrics(stdout: str) -> dict:
    """Parse the last log line of training stdout for loss/accuracy."""
    metrics: dict = {}
    for line in reversed(stdout.strip().splitlines()):
        m_total = re.search(r"Total:\s*([\d.]+)", line)
        m_loss = re.search(r"Loss:\s*([\d.]+)", line)
        m_acc = re.search(r"Acc:\s*([\d.]+)", line)
        if m_total:
            metrics["final_train_loss"] = float(m_total.group(1))
        elif m_loss:
            metrics["final_train_loss"] = float(m_loss.group(1))
        if m_acc:
            metrics["final_train_accuracy"] = float(m_acc.group(1))
        if metrics:
            break
    return metrics


# ─── Evaluation ──────────────────────────────────────────────────────────────

def evaluate_q3c(checkpoint_dir: str, config: dict) -> dict:
    """Load Q3C models from *checkpoint_dir* and measure success rate."""
    from utils.models import ControlPointGenerator, QEstimator
    from simulations.particle_simulation import ParticleSimulation

    active_env = config.get("active_env", "particle")
    env_config = config["environments"][active_env]
    sim_config = config.get("simulation", {})

    state_dim = env_config["state_dim"]
    action_dim = env_config["action_dim"]
    frame_stack = env_config.get("frame_stack", 1)
    action_bounds = tuple(env_config.get("action_bounds", [0, 1]))
    n_dim = env_config.get("n_dim", 2)
    control_points = env_config["model"]["control_points"]
    num_hidden_layers = env_config["model"]["num_hidden_layers"]
    num_neurons = env_config["model"]["num_neurons"]
    use_spectral_norm = env_config["model"].get("use_spectral_norm", False)
    hidden_dims = [num_neurons] * num_hidden_layers
    max_episode_steps = sim_config.get("max_episode_steps", 50)
    num_seeds = int(sim_config.get("num_seeds", len(sim_config.get("default_seeds", [0]))))
    if num_seeds <= 0:
        raise ValueError("simulation.num_seeds must be >= 1")
    seeds = list(range(num_seeds))

    cp_path = os.path.join(checkpoint_dir, "control_point_generator.pt")
    q_path = os.path.join(checkpoint_dir, "q_estimator.pt")
    norm_stats_path = os.path.join(checkpoint_dir, "norm_stats.pt")

    if not os.path.exists(cp_path) or not os.path.exists(q_path):
        return {
            "success_rate": 0.0,
            "avg_reward": 0.0,
            "error": f"Checkpoints not found in {checkpoint_dir}",
        }

    # Presence of norm_stats.pt = ibc_with_cps (actions normalized to [0,1]
    # before the Q estimator sees them).
    norm_stats = None
    if os.path.exists(norm_stats_path):
        norm_stats = torch.load(norm_stats_path, weights_only=False)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    cp_gen = ControlPointGenerator(
        input_dim=state_dim * frame_stack,
        output_dim=action_dim,
        control_points=control_points,
        hidden_dims=hidden_dims,
        action_bounds=action_bounds,
    )
    cp_gen.load_state_dict(
        torch.load(cp_path, map_location=device, weights_only=True)
    )
    cp_gen.to(device).eval()

    q_est = QEstimator(
        state_dim=state_dim * frame_stack,
        action_dim=action_dim,
        hidden_dims=hidden_dims,
        use_spectral_norm=use_spectral_norm,
    )
    q_est.load_state_dict(
        torch.load(q_path, map_location=device, weights_only=True)
    )
    q_est.to(device).eval()

    sim = ParticleSimulation(
        control_point_generator=cp_gen,
        q_estimator=q_est,
        n_dim=n_dim,
        device=device,
        max_episode_steps=max_episode_steps,
        render_mode=None,
        frame_stack=frame_stack,
        norm_stats=norm_stats,
    )

    all_results = []
    for seed in seeds:
        result = sim.run_episode(seed=seed)
        all_results.append(result)
    sim.close()

    successes = [r.get("success", False) for r in all_results]
    rewards = [r.get("total_reward", 0.0) for r in all_results]

    return {
        "success_rate": float(np.mean(successes)),
        "avg_reward": float(np.mean(rewards)),
        "num_seeds": len(seeds),
        "per_seed": [
            {
                "seed": seeds[i],
                "success": bool(successes[i]),
                "reward": float(rewards[i]),
            }
            for i in range(len(seeds))
        ],
    }


# ─── Auto-suggest strategy ───────────────────────────────────────────────────

def suggest_next_params(
    trials: list[dict],
    detected_params: list[str],
    baseline: dict,
) -> tuple[dict, str]:
    """Adaptively pick the next hyperparameter configuration to try.

    Strategy:
      Phase 1 (trial 1): Run the baseline.
      Phase 2 (trials 2..len(detected)+1): Vary one parameter at a time from
              the current best, choosing the parameter not yet explored.
      Phase 3 (afterwards): Combine best-per-param values with one random
              perturbation. Avoid exact duplicate configs.
    """
    if not trials:
        return baseline.copy(), "baseline"

    best_trial = max(trials, key=lambda t: t.get("success_rate", -1))
    best_params: dict = best_trial["params"]
    tried_signatures = {
        json.dumps(t["params"], sort_keys=True) for t in trials
    }

    # Phase 2: one-at-a-time exploration from best-so-far
    explored_params = set()
    for t in trials:
        for p in detected_params:
            if t["params"].get(p) != baseline.get(p):
                explored_params.add(p)

    unexplored = [p for p in detected_params if p not in explored_params and p in SEARCH_SPACE]
    if unexplored:
        param = unexplored[0]
        space = SEARCH_SPACE[param]
        current_val = best_params.get(param, baseline.get(param))
        candidates = [v for v in space["values"] if v != current_val]
        if candidates:
            new_val = random.choice(candidates)
            suggested = best_params.copy()
            suggested[param] = new_val
            return suggested, f"varying {param}={new_val} (from {current_val})"

    # Phase 3: combine best-per-param + perturbation
    combined: dict = {}
    for param in detected_params:
        if param not in SEARCH_SPACE:
            continue
        param_scores: dict = {}
        for t in trials:
            val = t["params"].get(param)
            if val is not None:
                sr = t.get("success_rate", 0)
                if val not in param_scores or sr > param_scores[val]:
                    param_scores[val] = sr
        if param_scores:
            combined[param] = max(param_scores, key=param_scores.get)  # type: ignore[arg-type]
        elif param in baseline:
            combined[param] = baseline[param]

    # Perturb one random parameter to avoid stagnation
    perturb_param = random.choice(detected_params)
    if perturb_param in SEARCH_SPACE:
        combined[perturb_param] = random.choice(SEARCH_SPACE[perturb_param]["values"])

    sig = json.dumps(combined, sort_keys=True)
    if sig in tried_signatures:
        for _ in range(20):
            p = random.choice(detected_params)
            if p in SEARCH_SPACE:
                combined[p] = random.choice(SEARCH_SPACE[p]["values"])
            sig = json.dumps(combined, sort_keys=True)
            if sig not in tried_signatures:
                break

    return combined, "combining best-per-param + perturbation"


# ─── Trial runner ─────────────────────────────────────────────────────────────

def run_single_trial(
    script_path: Path,
    params: dict,
    trial_id: int,
    training_steps_override: int | None = None,
    timeout: int | None = None,
) -> dict:
    """Modify config, train, evaluate, record and restore."""
    script_name = script_path.name
    print(f"\n{'=' * 80}")
    print(f"TRIAL {trial_id} — {script_name}")
    print(f"{'=' * 80}")
    print(f"Parameters:\n{json.dumps(params, indent=2)}")

    config = load_config()
    config = apply_params_to_config(config, params)
    checkpoint_dir = set_trial_checkpoint_dir(config, trial_id)

    active_env = config.get("active_env", "particle")
    if training_steps_override is not None:
        config["environments"][active_env].setdefault("training", {})[
            "training_steps"
        ] = training_steps_override

    actual_steps = (
        config["environments"][active_env]
        .get("training", {})
        .get(
            "training_steps",
            config.get("training_shared", {}).get("training_steps", 100000),
        )
    )

    backup_config()
    try:
        save_config(config)

        print(f"\n  Training ({actual_steps} steps)...")
        success, stdout, duration = run_training(script_path, timeout=timeout)

        if not success:
            print(f"\n  Training FAILED after {duration:.0f}s")
            last_lines = "\n".join(stdout.strip().splitlines()[-5:])
            record = {
                "trial_id": trial_id,
                "script": script_name,
                "params": params,
                "training_steps": actual_steps,
                "duration_seconds": round(duration, 1),
                "success_rate": 0.0,
                "avg_reward": 0.0,
                "training_failed": True,
                "error": last_lines[-300:],
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }
            append_trial(script_name, record)
            return record

        print(f"\n  Training completed in {duration:.0f}s")
        train_metrics = extract_final_metrics(stdout)

        print("  Evaluating...")
        eval_results: dict
        try:
            eval_results = evaluate_q3c(checkpoint_dir, config)
        except Exception as exc:
            eval_results = {
                "success_rate": 0.0,
                "avg_reward": 0.0,
                "error": f"Evaluation failed: {exc}",
                "per_seed": [],
            }
            print(f"  Evaluation failed: {exc}")

        record = {
            "trial_id": trial_id,
            "script": script_name,
            "params": params,
            "training_steps": actual_steps,
            "duration_seconds": round(duration, 1),
            "success_rate": eval_results.get("success_rate", 0.0),
            "avg_reward": eval_results.get("avg_reward", 0.0),
            **train_metrics,
            "eval_details": eval_results.get("per_seed", []),
            "eval_error": eval_results.get("error"),
            "checkpoint_dir": checkpoint_dir,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
        append_trial(script_name, record)

        sr = record["success_rate"]
        rw = record["avg_reward"]
        print(f"\n  Result: success_rate={sr:.2%}, avg_reward={rw:.3f}")
        return record
    finally:
        restore_config()


# ─── Analyze / summary ───────────────────────────────────────────────────────

def print_analysis(script_name: str) -> None:
    """Print a formatted results table sorted by success rate."""
    trials = load_trials(script_name)
    if not trials:
        print(f"No trials found for {script_name}.")
        return

    all_param_names: list[str] = []
    seen: set[str] = set()
    for t in trials:
        for p in t.get("params", {}):
            if p not in seen:
                all_param_names.append(p)
                seen.add(p)

    # Column definitions: (header, width)
    cols: list[tuple[str, int]] = [("Trial", 6)]
    for p in all_param_names:
        cols.append((p[:12], max(len(p[:12]), 10)))
    cols += [("Steps", 8), ("Success", 8), ("Reward", 8), ("Loss", 8), ("Time", 7)]

    header = " | ".join(f"{name:>{w}}" for name, w in cols)
    separator = "-+-".join("-" * w for _, w in cols)

    print(f"\n{'=' * len(header)}")
    print(f"  Hyperparameter search results: {script_name}")
    print(f"{'=' * len(header)}")
    print(header)
    print(separator)

    sorted_trials = sorted(
        trials, key=lambda t: t.get("success_rate", -1), reverse=True
    )
    for t in sorted_trials:
        row_vals: list[str] = [f"{t['trial_id']:>6}"]
        for p in all_param_names:
            val = t.get("params", {}).get(p, "")
            if isinstance(val, float):
                row_vals.append(f"{val:>{cols[len(row_vals)][1]}.4g}")
            else:
                row_vals.append(f"{str(val):>{cols[len(row_vals)][1]}}")

        steps = t.get("training_steps", "?")
        row_vals.append(f"{steps:>8}")

        failed = t.get("training_failed", False)
        sr = t.get("success_rate", 0)
        rw = t.get("avg_reward", 0)
        loss = t.get("final_train_loss")
        dur = t.get("duration_seconds", 0)

        row_vals.append("  FAILED" if failed else f"{sr:>7.0%}")
        row_vals.append(f"{rw:>8.3f}")
        row_vals.append(f"{loss:>8.4f}" if loss is not None else f"{'—':>8}")
        row_vals.append(f"{dur / 60:>6.1f}m")

        print(" | ".join(row_vals))

    print(f"{'=' * len(header)}")

    valid = [t for t in trials if not t.get("training_failed")]
    if valid:
        best = max(valid, key=lambda t: t.get("success_rate", 0))
        print(f"\nBest trial: #{best['trial_id']}  "
              f"success_rate={best['success_rate']:.2%}  "
              f"avg_reward={best['avg_reward']:.3f}")
        print(f"  Params: {json.dumps(best['params'], indent=4)}")

    print(f"\nTotal trials: {len(trials)} ({len(valid)} completed, "
          f"{len(trials) - len(valid)} failed)\n")


# ─── CLI ──────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Agent-assisted hyperparameter search for Q3C-IBC training scripts.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  python hyperparam_search.py combinedv2_cpascounter_training.py --run\n"
            "  python hyperparam_search.py combinedv2_cpascounter_training.py --run "
            "--params '{\"learning_rate\": 5e-4}'\n"
            "  python hyperparam_search.py combinedv2_cpascounter_training.py --auto "
            "--max-trials 5 --reduced-steps 20000\n"
            "  python hyperparam_search.py combinedv2_cpascounter_training.py --analyze\n"
        ),
    )
    parser.add_argument(
        "script",
        type=str,
        help="Training script to optimize (e.g. combinedv2_cpascounter_training.py)",
    )

    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--run", action="store_true", help="Run a single trial")
    mode.add_argument(
        "--auto",
        action="store_true",
        help="Run multiple trials with adaptive exploration",
    )
    mode.add_argument(
        "--analyze", action="store_true", help="Print summary of past trials"
    )

    parser.add_argument(
        "--params",
        type=str,
        default=None,
        help='JSON string of param overrides for --run (e.g. \'{"learning_rate": 5e-4}\')',
    )
    parser.add_argument(
        "--fixed-params",
        type=str,
        default=None,
        help=(
            "JSON string of params to lock in all trials (works with --run and --auto), "
            "e.g. '{\"counter_examples\": 0}'"
        ),
    )
    parser.add_argument(
        "--max-trials",
        type=int,
        default=5,
        help="Number of trials for --auto mode (default: 5)",
    )
    parser.add_argument(
        "--reduced-steps",
        type=int,
        default=None,
        help="Override training_steps for faster exploration",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=None,
        help="Per-trial timeout in seconds",
    )
    args = parser.parse_args()

    fixed_params: dict = {}
    if args.fixed_params:
        fixed_params = json.loads(args.fixed_params)

    script_path = ROOT_DIR / args.script
    if not script_path.exists():
        print(f"Error: script not found at {script_path}")
        sys.exit(1)

    script_name = Path(args.script).name

    # ── Analyze mode ──────────────────────────────────────────────────────
    if args.analyze:
        print_analysis(script_name)
        return

    # ── Detect params and baseline ────────────────────────────────────────
    detected_params = detect_script_params(script_path)
    if not detected_params:
        print(f"Warning: no tunable hyperparameters detected in {script_name}.")
        print("The script may use hardcoded values instead of config.json.")
    else:
        print(f"Detected tunable params in {script_name}:")
        for p in detected_params:
            print(f"  - {p}  (search space: {SEARCH_SPACE[p]['values']})")

    config = load_config()
    baseline = get_baseline_params(config, detected_params)
    print(f"\nBaseline (current config):")
    for k, v in baseline.items():
        print(f"  {k} = {v}")
    print()

    # ── Run mode ──────────────────────────────────────────────────────────
    if args.run:
        trial_id = next_trial_id(script_name)
        if args.params:
            user_params = json.loads(args.params)
            params = baseline.copy()
            params.update(user_params)
        else:
            trials = load_trials(script_name)
            params, reason = suggest_next_params(trials, detected_params, baseline)
            print(f"Auto-suggested ({reason})")

        if fixed_params:
            params.update(fixed_params)
            print(f"Applied fixed params: {json.dumps(fixed_params)}")

        run_single_trial(
            script_path=script_path,
            params=params,
            trial_id=trial_id,
            training_steps_override=args.reduced_steps,
            timeout=args.timeout,
        )
        return

    # ── Auto mode ─────────────────────────────────────────────────────────
    if args.auto:
        for i in range(args.max_trials):
            trial_id = next_trial_id(script_name)
            trials = load_trials(script_name)
            params, reason = suggest_next_params(trials, detected_params, baseline)
            print(f"\n[Auto {i + 1}/{args.max_trials}] Strategy: {reason}")

            if fixed_params:
                params.update(fixed_params)
                print(f"[Auto {i + 1}/{args.max_trials}] Fixed params: {json.dumps(fixed_params)}")

            run_single_trial(
                script_path=script_path,
                params=params,
                trial_id=trial_id,
                training_steps_override=args.reduced_steps,
                timeout=args.timeout,
            )

        print("\n\nAuto-exploration complete. Full results:")
        print_analysis(script_name)


if __name__ == "__main__":
    main()
