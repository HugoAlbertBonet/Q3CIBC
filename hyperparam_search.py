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
import fcntl
import json
import os
import random
import re
import secrets
import subprocess
import sys
import time
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch

# torch>=2.6 defaults torch.load to weights_only=True, which rejects the numpy
# arrays inside our norm_stats.pt (obs mean/std, action min/max, and the LIBERO
# goal-embedding matrix). add_safe_globals doesn't reliably fix it under numpy
# 2.x: the pickle stores the global as `numpy.core.multiarray._reconstruct`,
# but on numpy 2.x the real callable's module is `numpy._core.multiarray`, so
# torch's allowlist match misses. We trust our own checkpoints, so force every
# torch.load in this process to weights_only=False.
_ORIG_TORCH_LOAD = torch.load


def _trusted_torch_load(*args, **kwargs):
    kwargs["weights_only"] = False
    return _ORIG_TORCH_LOAD(*args, **kwargs)


torch.load = _trusted_torch_load

ROOT_DIR = Path(__file__).parent
sys.path.insert(0, str(ROOT_DIR))

CONFIG_PATH = ROOT_DIR / "config_json" / "config.json"
RESULTS_BASE_DIR = ROOT_DIR / "results" / "hyperparam_search"
CHECKPOINTS_BASE_DIR = ROOT_DIR / "checkpoints" / "hpsearch"


def _new_run_id() -> str:
    """Unique identifier for a trial: timestamp + random suffix. Safe under concurrency."""
    return datetime.now().strftime("%Y%m%dT%H%M%S") + "_" + secrets.token_hex(4)

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
        "values": [50000, 100000, 150000, 200000, 250000, 300000, 350000, 400000, 500000],
        "type": "int",
        "location": "env_training",
    },
    "separation_loss": {
        "values": ["separation", "entropy", "chamfer"],
        "type": "str",
        "location": "env_training",
    },
    "exclude_top_from_separation": {
        "values": [False, True],
        "type": "bool",
        "location": "training_shared",
    },
    "noisy_expert_count": {
        "values": [0, 4, 8, 16],
        "type": "int",
        "location": "training_shared",
    },
    "noisy_expert_std": {
        "values": [0.02, 0.05, 0.1, 0.2],
        "type": "float",
        "location": "training_shared",
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
    # Per-net architecture (added 2026-05-07 — Q-estimator capacity probe).
    # Mirrors IBC paper's ResNetPreActivation. "mlp" preserves legacy behavior.
    "q_network_kind": {
        "values": ["mlp", "resnet"],
        "type": "str",
        "location": "env_model",
    },
    "q_width": {
        "values": [128, 256, 512, 1024, 2048],
        "type": "int",
        "location": "env_model",
    },
    "q_depth": {
        "values": [2, 4, 8, 16],
        "type": "int",
        "location": "env_model",
    },
    "q_use_spectral_norm": {
        "values": [True, False],
        "type": "bool",
        "location": "env_model",
    },
    "cp_network_kind": {
        "values": ["mlp", "resnet"],
        "type": "str",
        "location": "env_model",
    },
    "cp_width": {
        "values": [128, 256, 512, 1024],
        "type": "int",
        "location": "env_model",
    },
    "cp_depth": {
        "values": [2, 4, 8],
        "type": "int",
        "location": "env_model",
    },
    "cp_use_spectral_norm": {
        "values": [True, False],
        "type": "bool",
        "location": "env_model",
    },
    "cosine_t_max": {
        "values": [
            100000,
            150000,
            200000,
            250000,
            300000,
            350000,
            400000,
            500000,
        ],
        "type": "int",
        "location": "env_training",
    },
    "target_update_interval": {
        "values": [200, 500, 1000, 2000, 5000],
        "type": "int",
        "location": "training_shared",
    },
    "inference_langevin_iterations": {
        "values": [0, 10, 25, 50, 100, 150, 200, 250, 300],
        "type": "int",
        "location": "env_training",
    },
    # Inference-time Langevin hyperparam overrides. When set, these REPLACE
    # the corresponding training-Langevin values (langevin_lr_init, etc.)
    # ONLY for the eval-time refinement chain — training Langevin negs still
    # use the paper-faithful aggressive values. Use to test gentle inference
    # refinement on Q3C's narrow-trained Q surface.
    "inference_langevin_lr_init": {
        "values": [0.005, 0.01, 0.05, 0.1, 0.5],
        "type": "float",
        "location": "env_training",
    },
    "inference_langevin_lr_final": {
        "values": [1e-7, 1e-6, 1e-5, 1e-4],
        "type": "float",
        "location": "env_training",
    },
    "inference_langevin_decay_power": {
        "values": [1.0, 2.0, 4.0],
        "type": "float",
        "location": "env_training",
    },
    "inference_langevin_delta_clip": {
        "values": [0.005, 0.01, 0.02, 0.05, 0.1, 0.5],
        "type": "float",
        "location": "env_training",
    },
    "inference_langevin_noise_scale": {
        "values": [0.0, 0.05, 0.1, 0.3, 1.0],
        "type": "float",
        "location": "env_training",
    },
    # ── CP-DFO refinement at inference (Q3CIBC-specific) ────────────────────
    # When > 0, replaces inference-time Langevin with a DFO-style iterative
    # refinement starting from the CP cloud (optionally with a few extra
    # uniform samples for safety). Cheaper than Langevin (~5 forward passes
    # vs ~100, no autograd) and matches DFO's quality whenever the CP cloud
    # already covers the right action mode — which pushingA showed it does on
    # Pushing. If both `inference_dfo_iterations > 0` and
    # `inference_langevin_iterations > 0` are set, DFO takes precedence.
    "inference_dfo_iterations": {
        "values": [0, 3, 5, 10, 15],
        "type": "int",
        "location": "env_training",
    },
    "inference_dfo_iteration_std": {
        "values": [0.005, 0.01, 0.015, 0.03, 0.05, 0.1, 0.2],
        "type": "float",
        "location": "env_training",
    },
    "inference_dfo_iteration_std_decay": {
        "values": [0.5, 0.7, 0.9],
        "type": "float",
        "location": "env_training",
    },
    "inference_dfo_num_uniform": {
        "values": [0, 16, 32, 64],
        "type": "int",
        "location": "env_training",
    },
    "langevin_num_iterations": {
        "values": [0, 10, 25, 50, 100],
        "type": "int",
        "location": "env_training",
    },
    # Langevin refinement hyperparameters. These override the per-env
    # env_model.langevin_config defaults at both training and inference time.
    "langevin_lr_init": {
        "values": [0.005, 0.01, 0.02, 0.05, 0.1],
        "type": "float",
        "location": "env_training",
    },
    "langevin_lr_final": {
        "values": [1e-6, 1e-5, 1e-4],
        "type": "float",
        "location": "env_training",
    },
    "langevin_noise_scale": {
        "values": [0.0, 0.05, 0.1, 0.3, 1.0],
        "type": "float",
        "location": "env_training",
    },
    "langevin_delta_clip": {
        "values": [0.01, 0.02, 0.05, 0.1],
        "type": "float",
        "location": "env_training",
    },
    "langevin_decay_power": {
        "values": [1.0, 2.0, 4.0],
        "type": "float",
        "location": "env_training",
    },
    # IBC negative mixture (Florence et al., 2021).
    "num_uniform_negatives": {
        "values": [0, 16, 32, 64, 128],
        "type": "int",
        "location": "env_training",
    },
    "num_langevin_negatives": {
        "values": [0, 16, 32, 64],
        "type": "int",
        "location": "env_training",
    },
    # Langevin negative starting distribution. "uniform" = paper-faithful;
    # "cps" = start from CP cloud, find Q-peaks in CP neighbourhoods.
    "langevin_init_kind": {
        "values": ["uniform", "cps"],
        "type": "str",
        "location": "env_training",
    },
    "langevin_init_jitter": {
        "values": [0.0, 0.01, 0.03, 0.05, 0.1],
        "type": "float",
        "location": "env_training",
    },
    # Noisy-expert hard negatives (estimator-only). σ linearly interpolates
    # from sigma_start to sigma_final over training.
    "noisy_expert_count": {
        "values": [0, 8, 16, 32, 64],
        "type": "int",
        "location": "training_shared",
    },
    "noisy_expert_sigma_start": {
        "values": [0.05, 0.1, 0.2, 0.3, 0.5],
        "type": "float",
        "location": "training_shared",
    },
    "noisy_expert_sigma_final": {
        "values": [0.005, 0.01, 0.02, 0.05],
        "type": "float",
        "location": "training_shared",
    },
    # IBC gradient penalty (Florence et al., 2021, App. B).
    "gradient_penalty_weight": {
        "values": [0.0, 0.1, 1.0, 10.0],
        "type": "float",
        "location": "training_shared",
    },
    "gradient_penalty_margin": {
        # 0.05–0.2 is the firing range for our 2x256 MLP on [0,1]^8;
        # 0.5–2.0 stays in line with the IBC paper at larger scales.
        "values": [0.05, 0.1, 0.2, 0.5, 1.0, 2.0],
        "type": "float",
        "location": "training_shared",
    },
    "gradient_penalty_form": {
        # "hinge"  = IBC-faithful one-sided: penalty = max(0, |grad|-margin)^2
        # "target" = WGAN-GP two-sided:      penalty = (|grad|-margin)^2
        "values": ["hinge", "target"],
        "type": "str",
        "location": "training_shared",
    },
    # Deterministic seed — searchable so we can run reps with seed=0,1,2,...
    "trial_seed": {
        "values": [0, 1, 2, 3, 4],
        "type": "int",
        "location": "env_training",
    },
}


def effective_langevin_config(env_config: dict) -> dict:
    """Merge env_training langevin_* overrides onto env_model.langevin_config defaults.

    Returns a dict keyed by the native sample_langevin arg names (lr_init, etc.),
    so callers in both training and evaluation share one source of truth.
    """
    base = dict(env_config.get("model", {}).get("langevin_config", {}))
    training = env_config.get("training", {})
    overrides = {
        "num_iterations": "langevin_num_iterations",
        "lr_init": "langevin_lr_init",
        "lr_final": "langevin_lr_final",
        "noise_scale": "langevin_noise_scale",
        "delta_action_clip": "langevin_delta_clip",
        "polynomial_decay_power": "langevin_decay_power",
    }
    for native_key, training_key in overrides.items():
        if training_key in training:
            base[native_key] = training[training_key]
    return base


def effective_inference_langevin_config(env_config: dict) -> dict:
    """Inference-time Langevin config: training defaults overridden by inference_* keys.

    Lets callers run aggressive paper-faithful Langevin during training
    (for hard negatives) while using a gentler inference chain to refine
    actions on Q3C's narrow-trained Q surface. Falls back to training values
    for any key not overridden, so an empty inference_* set = same as training.
    """
    cfg = effective_langevin_config(env_config)
    training = env_config.get("training", {})
    overrides = {
        "lr_init": "inference_langevin_lr_init",
        "lr_final": "inference_langevin_lr_final",
        "noise_scale": "inference_langevin_noise_scale",
        "delta_action_clip": "inference_langevin_delta_clip",
        "polynomial_decay_power": "inference_langevin_decay_power",
    }
    for native_key, inf_key in overrides.items():
        if inf_key in training:
            cfg[native_key] = training[inf_key]
    return cfg


# ─── Config I/O ──────────────────────────────────────────────────────────────

def load_config() -> dict:
    """Read the on-disk default config. Never mutated by parallel trials."""
    with open(CONFIG_PATH, "r") as f:
        return json.load(f)


# ─── Trials I/O ──────────────────────────────────────────────────────────────

def _results_dir(script_name: str, active_env: str | None = None) -> Path:
    """Resolve the trials directory for a (script, env) pair.

    When `active_env` is provided (the CLI override path) we use it directly
    and DO NOT touch config.json. This avoids the long-standing race where
    config.json was flipped between trial submit and trial record, which
    caused pushing trials to be logged into the particle folder and vice
    versa. The disk-read fallback is kept so old call paths (`--analyze`
    without --active-env) still work.
    """
    if active_env is None:
        config = load_config()
        active_env = config.get("active_env", "particle")
        env_cfg = config.get("environments", {}).get(active_env, {})
    else:
        # Still need env_cfg to detect particle n_dim partitioning; read from
        # disk but use the override for the env selection.
        config = load_config()
        env_cfg = config.get("environments", {}).get(active_env, {})

    # Env → results-subpath mapping. D4RL-family envs go under d4rl/<env>
    # so they can be grouped together as the codebase grows (kitchen, hammer,
    # door, etc. all share the AdroitHand-like protocol).
    _ENV_PATH_MAP: dict[str, str] = {
        "pen": "d4rl/pen",
        "door": "d4rl/door",
        "kitchen": "d4rl/kitchen",
    }
    env_subpath = _ENV_PATH_MAP.get(active_env, active_env)
    results_dir = RESULTS_BASE_DIR / Path(script_name).stem / env_subpath

    # For particle experiments, partition trials by n_dim to avoid mixing runs.
    if "n_dim" in env_cfg:
        results_dir = results_dir / str(env_cfg["n_dim"])

    return results_dir


def _trials_path(script_name: str, active_env: str | None = None) -> Path:
    return _results_dir(script_name, active_env=active_env) / "trials.jsonl"


def load_trials(script_name: str, active_env: str | None = None) -> list[dict]:
    path = _trials_path(script_name, active_env=active_env)
    if not path.exists():
        return []
    trials = []
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if line:
                trials.append(json.loads(line))
    return trials


def append_trial(script_name: str, record: dict, active_env: str | None = None) -> int:
    """Atomically assign a monotonically-increasing trial_id and append the record.

    Uses fcntl.flock for an exclusive lock over the jsonl for the short read-max +
    write-line section. Safe under parallel sbatch submissions. Returns the id.
    """
    path = _trials_path(script_name, active_env=active_env)
    path.parent.mkdir(parents=True, exist_ok=True)

    with open(path, "a") as f:
        fcntl.flock(f.fileno(), fcntl.LOCK_EX)
        try:
            max_id = 0
            try:
                with open(path, "r") as rf:
                    for line in rf:
                        line = line.strip()
                        if not line:
                            continue
                        try:
                            max_id = max(max_id, int(json.loads(line).get("trial_id", 0)))
                        except (json.JSONDecodeError, ValueError):
                            continue
            except FileNotFoundError:
                pass
            trial_id = max_id + 1
            record["trial_id"] = trial_id
            f.write(json.dumps(record, sort_keys=True, default=str) + "\n")
            f.flush()
            os.fsync(f.fileno())
        finally:
            fcntl.flock(f.fileno(), fcntl.LOCK_UN)
    return trial_id


# ─── Hyperparameter detection ────────────────────────────────────────────────

# Params consumed at evaluation time only (never referenced by training scripts).
# They must still appear in the search so they get tuned.
INFERENCE_ONLY_PARAMS: set[str] = {
    "inference_langevin_iterations",
    "inference_langevin_lr_init",
    "inference_langevin_lr_final",
    "inference_langevin_decay_power",
    "inference_langevin_delta_clip",
    "inference_langevin_noise_scale",
    "inference_dfo_iterations",
    "inference_dfo_iteration_std",
    "inference_dfo_iteration_std_decay",
    "inference_dfo_num_uniform",
}


def detect_script_params(script_path: Path) -> list[str]:
    """Scan training script source to find which search-space params it reads."""
    with open(script_path, "r") as f:
        source = f.read()
    detected = []
    for param_name in SEARCH_SPACE:
        if param_name in INFERENCE_ONLY_PARAMS:
            detected.append(param_name)
            continue
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


def set_run_checkpoint_dir(config: dict, run_id: str) -> str:
    """Point model_save_dir to a per-run directory (unique even under parallel runs)."""
    run_dir = str(CHECKPOINTS_BASE_DIR / f"run_{run_id}")
    config.setdefault("training_shared", {})["model_save_dir"] = run_dir
    return run_dir


# ─── Training subprocess ─────────────────────────────────────────────────────

def run_training(
    script_path: Path,
    timeout: int | None = None,
    env_extras: dict[str, str] | None = None,
) -> tuple[bool, str, float]:
    """Run a training script as a subprocess, streaming output live.

    Returns (success, captured_stdout, duration_seconds).
    `env_extras` is layered on top of the inherited env (e.g., Q3C_CONFIG_PATH).
    """
    start = time.time()
    stdout_lines: list[str] = []
    env = {**os.environ, "WANDB_MODE": "disabled", "PYTHONUNBUFFERED": "1"}
    if env_extras:
        env.update(env_extras)

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
    from utils.sampling import sample_langevin

    active_env = config.get("active_env", "particle")
    env_config = config["environments"][active_env]
    sim_config = config.get("simulation", {})

    # Pick the right simulation class. `pushing` uses the vendored IBC env
    # (PyBullet + gym), which lives behind the `pushing` optional-extras and
    # is NOT installed on SLURM nodes running particle/dummy/pen trials. Keep
    # the import lazy so a particle SLURM job never touches pushing deps.
    if active_env == "pushing":
        from simulations.pushing_simulation import PushingSimulation
        SimulationCls = PushingSimulation
    elif active_env == "pushing_multi":
        from simulations.pushing_multi_simulation import PushingMultiSimulation
        SimulationCls = PushingMultiSimulation
    elif active_env == "pushing_pixels":
        from simulations.pushing_pixels_simulation import PushingPixelsSimulation
        SimulationCls = PushingPixelsSimulation
    elif active_env == "pen":
        from simulations.pen_human_v2_simulation import PenHumanV2Simulation
        SimulationCls = PenHumanV2Simulation
    elif active_env == "door":
        from simulations.door_human_v2_simulation import DoorHumanV2Simulation
        SimulationCls = DoorHumanV2Simulation
    elif active_env == "kitchen":
        from simulations.kitchen_simulation import KitchenSimulation
        SimulationCls = KitchenSimulation
    elif active_env == "libero_goal":
        from simulations.libero_goal_simulation import LiberoGoalSimulation
        SimulationCls = LiberoGoalSimulation
    else:
        from simulations.particle_simulation import ParticleSimulation
        SimulationCls = ParticleSimulation

    state_dim = env_config["state_dim"]
    action_dim = env_config["action_dim"]
    frame_stack = env_config.get("frame_stack", 1)
    action_bounds = tuple(env_config.get("action_bounds", [0, 1]))
    n_dim = env_config.get("n_dim", 2)
    em = env_config["model"]
    control_points = em["control_points"]
    num_hidden_layers = em["num_hidden_layers"]
    num_neurons = em["num_neurons"]
    use_spectral_norm = em.get("use_spectral_norm", False)
    hidden_dims = [num_neurons] * num_hidden_layers
    # Per-net architecture (mirrors combinedv2_cpascounter_training.py).
    q_network_kind = em.get("q_network_kind", "mlp")
    q_width = em.get("q_width", num_neurons)
    q_depth = em.get("q_depth", num_hidden_layers)
    q_use_spectral_norm = em.get("q_use_spectral_norm", use_spectral_norm)
    cp_network_kind = em.get("cp_network_kind", "mlp")
    cp_width = em.get("cp_width", num_neurons)
    cp_depth = em.get("cp_depth", num_hidden_layers)
    cp_use_spectral_norm = em.get("cp_use_spectral_norm", False)
    # Per-env override wins over the shared simulation.max_episode_steps.
    # Pushing needs 100 (IBC paper BlockPush-v0); particle uses the global 50.
    max_episode_steps = env_config.get(
        "max_episode_steps", sim_config.get("max_episode_steps", 50)
    )
    # IBC Table 3 reports simulated pushing over 100 evaluation episodes per
    # training seed. Keep this env-scoped so particle's established eval count
    # does not change.
    num_seeds = int(
        env_config.get(
            "num_eval_seeds",
            sim_config.get("num_seeds", len(sim_config.get("default_seeds", [0]))),
        )
    )
    if num_seeds <= 0:
        raise ValueError("simulation.num_seeds must be >= 1")
    seeds = list(range(num_seeds))

    inference_langevin_iterations = int(
        env_config.get("training", {}).get("inference_langevin_iterations", 0)
    )
    # CP-DFO refinement (Q3CIBC-specific, no IBC analog). Takes precedence
    # over inference Langevin when > 0, so a trial can opt into either path
    # without changing the rest of the recipe.
    inference_dfo_iterations = int(
        env_config.get("training", {}).get("inference_dfo_iterations", 0)
    )
    inference_dfo_iteration_std = float(
        env_config.get("training", {}).get("inference_dfo_iteration_std", 0.1)
    )
    inference_dfo_iteration_std_decay = float(
        env_config.get("training", {}).get("inference_dfo_iteration_std_decay", 0.7)
    )
    inference_dfo_num_uniform = int(
        env_config.get("training", {}).get("inference_dfo_num_uniform", 0)
    )
    # Effective langevin hyperparams for INFERENCE chain. Starts from training
    # Langevin config (env_model.langevin_config + langevin_* training overrides),
    # then applies any inference_langevin_* overrides on top. Lets eval use
    # gentler step sizes than training while keeping training paper-faithful.
    langevin_cfg = effective_inference_langevin_config(env_config)

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

    if active_env == "pushing_pixels":
        from utils.models import PixelControlPointGenerator, PixelQEstimator
        # state_dim is [C, H, W]. frame_stack is already baked into C
        # (PushingPixelsDataset reports state_shape=(3*frame_stack, H, W)),
        # so DON'T multiply by frame_stack here.
        in_channels = int(state_dim[0])
        enc_h = int(env_config.get("encoder_target_height", 180))
        enc_w = int(env_config.get("encoder_target_width", 240))
        value_width = int(em.get("value_width", 1024))
        value_num_blocks = int(em.get("value_num_blocks", 1))
        cp_gen = PixelControlPointGenerator(
            output_dim=action_dim,
            control_points=control_points,
            hidden_dims=[cp_width] * cp_depth,
            action_bounds=action_bounds,
            network_kind=cp_network_kind,
            width=cp_width,
            depth=cp_depth,
            use_spectral_norm=cp_use_spectral_norm,
            in_channels=in_channels,
            encoder_target_height=enc_h,
            encoder_target_width=enc_w,
        )
        cp_gen.load_state_dict(torch.load(cp_path, map_location=device, weights_only=True))
        cp_gen.to(device).eval()

        q_est = PixelQEstimator(
            action_dim=action_dim,
            in_channels=in_channels,
            encoder_target_height=enc_h,
            encoder_target_width=enc_w,
            value_width=value_width,
            value_num_blocks=value_num_blocks,
        )
        q_est.load_state_dict(torch.load(q_path, map_location=device, weights_only=True))
        q_est.to(device).eval()
    else:
        # libero_goal bakes the goal embedding into the state AFTER frame-stacking,
        # so its input dim is NOT state_dim*frame_stack. Read the exact length
        # the dataset used straight from norm_stats.
        if active_env == "libero_goal" and norm_stats is not None and "state_shape" in norm_stats:
            flat_input_dim = int(norm_stats["state_shape"])
        else:
            flat_input_dim = state_dim * frame_stack
        cp_gen = ControlPointGenerator(
            input_dim=flat_input_dim,
            output_dim=action_dim,
            control_points=control_points,
            hidden_dims=[cp_width] * cp_depth,
            action_bounds=action_bounds,
            network_kind=cp_network_kind,
            width=cp_width,
            depth=cp_depth,
            use_spectral_norm=cp_use_spectral_norm,
        )
        cp_gen.load_state_dict(
            torch.load(cp_path, map_location=device, weights_only=True)
        )
        cp_gen.to(device).eval()

        q_est = QEstimator(
            state_dim=flat_input_dim,
            action_dim=action_dim,
            hidden_dims=[q_width] * q_depth,
            use_spectral_norm=q_use_spectral_norm,
            network_kind=q_network_kind,
            width=q_width,
            depth=q_depth,
        )
        q_est.load_state_dict(
            torch.load(q_path, map_location=device, weights_only=True)
        )
        q_est.to(device).eval()

    # ── Pixel envs: dedicated late-fused DFO / Langevin refinement ────────
    # The flat-state wrappers below assume `obs.unsqueeze(1).expand(-1, N, -1)`
    # is cheap — that's true for vector obs, but for images it would re-encode
    # the (1, C, H, W) tensor N (DFO) or 100 (Langevin) times PER ENV STEP.
    # Instead we encode ONCE per step, cache the 256-D features, and run the
    # refinement inner loop against PixelQEstimator.score(features, actions).
    # This is what IBC's `late_fusion = True` config flag does upstream.
    if active_env == "pushing_pixels":
        if inference_dfo_iterations > 0:
            _dfo_iters = inference_dfo_iterations
            _dfo_std0 = inference_dfo_iteration_std
            _dfo_decay = inference_dfo_iteration_std_decay
            _dfo_n_uniform = inference_dfo_num_uniform

            class PixelDFORefinedSimulation(SimulationCls):
                """Pixel-aware CP-DFO refinement (encode once per env step).

                Same algorithm as DFORefinedSimulation below — initial pop = CP
                cloud (+ optional uniform safety samples); each iter resamples
                via category-ordered softmax(Q) and jitters — but the Q forward
                calls run against cached image features instead of re-encoding.
                """

                def select_action(self, observation, return_q_range: bool = False):
                    obs_tensor = self._obs_to_tensor(observation)  # (1, C, H, W) uint8

                    with torch.no_grad():
                        features = self.q_estimator.encode(obs_tensor)  # (1, F)
                        cps = self.control_point_generator(obs_tensor)  # (1, N_cp, A)

                        if _dfo_n_uniform > 0:
                            unif = torch.empty(
                                1, _dfo_n_uniform, cps.shape[-1], device=self.device
                            ).uniform_(float(action_bounds[0]), float(action_bounds[1]))
                            candidates = torch.cat([cps, unif], dim=1)
                        else:
                            candidates = cps.clone()

                        N = candidates.shape[1]
                        std = float(_dfo_std0)
                        for it in range(_dfo_iters):
                            log_probs = self.q_estimator.score(features, candidates).squeeze(-1)  # (1, N)
                            probs = torch.softmax(log_probs.squeeze(0), dim=-1)
                            idx = torch.multinomial(probs, N, replacement=True)
                            counts = torch.bincount(idx, minlength=N)
                            repeat_idx = torch.repeat_interleave(
                                torch.arange(N, device=self.device), counts
                            )
                            candidates = candidates[:, repeat_idx, :]
                            if it < _dfo_iters - 1:
                                candidates = candidates + torch.randn_like(candidates) * std
                                candidates = candidates.clamp(
                                    float(action_bounds[0]), float(action_bounds[1])
                                )
                                std *= _dfo_decay
                        # Re-score after the final reorder so argmax index aligns
                        # with the (reordered) candidates tensor — same fix as
                        # the flat-state DFORefinedSimulation.
                        final_log_probs = self.q_estimator.score(features, candidates).squeeze(-1)
                        sel = final_log_probs.argmax(dim=1)
                        action_normalized = candidates[0, sel[0], :].cpu().numpy()
                        q_range = (final_log_probs.min().item(), final_log_probs.max().item())

                    action = np.clip(action_normalized, action_bounds[0], action_bounds[1])
                    action = self._denormalize_action(action)
                    if return_q_range:
                        return action, q_range
                    return action

            sim_cls = PixelDFORefinedSimulation

        elif inference_langevin_iterations > 0:
            class PixelLangevinRefinedSimulation(SimulationCls):
                """Pixel-aware Langevin refinement (encode once per env step).

                Encodes the (1, C, H, W) image once into 256-D features, picks
                the argmax-Q CP as the starting action, then runs Langevin MCMC
                on actions against the cached features. The energy_function
                ignores `sample_langevin`'s expanded-obs argument and uses the
                closed-over `features` tensor instead — that's how we get the
                speedup vs the flat-state wrapper.
                """

                def select_action(self, observation, return_q_range: bool = False):
                    obs_tensor = self._obs_to_tensor(observation)  # (1, C, H, W) uint8

                    with torch.no_grad():
                        features = self.q_estimator.encode(obs_tensor)  # (1, F)
                        cps = self.control_point_generator(obs_tensor)  # (1, N_cp, A)
                        q_values = self.q_estimator.score(features, cps).squeeze(-1)  # (1, N)
                        best_idx = q_values.argmax(dim=1)
                        q_range = (q_values.min().item(), q_values.max().item())
                        best_cp = cps[0, best_idx[0], :].view(1, 1, -1).clone()  # (1, 1, A)

                    act_min_t = torch.full(
                        (cps.shape[-1],), float(action_bounds[0]), device=self.device
                    )
                    act_max_t = torch.full(
                        (cps.shape[-1],), float(action_bounds[1]), device=self.device
                    )

                    for p in self.q_estimator.parameters():
                        p.requires_grad_(False)

                    # Closed over `features` — the loop uses the cached encoding,
                    # not sample_langevin's expanded `obs_lv` arg (we just need
                    # to accept its signature).
                    def _neg_energy_fn(obs_lv, actions_lv):
                        return -self.q_estimator.score(features, actions_lv).squeeze(-1)

                    refined = sample_langevin(
                        energy_function=_neg_energy_fn,
                        observations=features,  # (1, F) — expanded internally, ignored by our fn
                        num_samples=1,
                        action_min=act_min_t,
                        action_max=act_max_t,
                        num_iterations=inference_langevin_iterations,
                        lr_init=float(langevin_cfg.get("lr_init", 0.1)),
                        lr_final=float(langevin_cfg.get("lr_final", 1e-5)),
                        polynomial_decay_power=float(
                            langevin_cfg.get("polynomial_decay_power", 2.0)
                        ),
                        delta_action_clip=float(
                            langevin_cfg.get("delta_action_clip", 0.1)
                        ),
                        noise_scale=float(langevin_cfg.get("noise_scale", 1.0)),
                        initial_actions=best_cp,
                        device=self.device,
                    )

                    for p in self.q_estimator.parameters():
                        p.requires_grad_(True)

                    action = refined[0, 0, :].cpu().numpy()
                    action = np.clip(action, action_bounds[0], action_bounds[1])
                    action = self._denormalize_action(action)
                    if return_q_range:
                        return action, q_range
                    return action

            sim_cls = PixelLangevinRefinedSimulation

        else:
            sim_cls = SimulationCls

    elif inference_dfo_iterations > 0:
        # ── CP-DFO refinement (Q3CIBC inference). Cheaper than Langevin: no
        # autograd, only N small-batch forward passes through the Q-net.
        # Initial population = CP cloud (+ optional N_uniform random
        # samples). Each iter: score → category-ordered resample with
        # softmax(Q) → small Gaussian jitter → clip. Mirrors IBC's
        # `iterative_dfo` mechanics (see `bench_inference.iterative_dfo_pass`)
        # but with a model-trained initial population.
        _dfo_iters = inference_dfo_iterations
        _dfo_std0 = inference_dfo_iteration_std
        _dfo_decay = inference_dfo_iteration_std_decay
        _dfo_n_uniform = inference_dfo_num_uniform

        class DFORefinedSimulation(SimulationCls):
            """Refines the CP cloud with iterative DFO before acting."""

            def select_action(self, observation, return_q_range: bool = False):
                obs_tensor = (
                    torch.tensor(observation, dtype=torch.float32)
                    .unsqueeze(0)
                    .to(self.device)
                )
                obs_tensor = self.obs_normalizer.normalize(obs_tensor)

                with torch.no_grad():
                    cps = self.control_point_generator(obs_tensor)  # (1, N_cp, D)

                    # Action normalization helper (matches the Langevin path).
                    def _norm(a):
                        if self._act_min_t is not None:
                            return (a - self._act_min_t) / self._act_rng_t
                        return a

                    # Mix in uniform safety samples if requested.
                    if _dfo_n_uniform > 0:
                        unif = torch.empty(
                            1, _dfo_n_uniform, cps.shape[-1], device=self.device
                        ).uniform_(float(action_bounds[0]), float(action_bounds[1]))
                        candidates = torch.cat([cps, unif], dim=1)
                    else:
                        candidates = cps.clone()

                    N = candidates.shape[1]
                    obs_expanded = obs_tensor.unsqueeze(1).expand(-1, N, -1)
                    std = float(_dfo_std0)
                    for it in range(_dfo_iters):
                        log_probs = self.q_estimator(obs_expanded, _norm(candidates)).squeeze(-1)
                        probs = torch.softmax(log_probs.squeeze(0), dim=-1)
                        # IBC-style category-ordered resample.
                        idx = torch.multinomial(probs, N, replacement=True)
                        counts = torch.bincount(idx, minlength=N)
                        repeat_idx = torch.repeat_interleave(
                            torch.arange(N, device=self.device), counts
                        )
                        candidates = candidates[:, repeat_idx, :]
                        if it < _dfo_iters - 1:
                            candidates = candidates + torch.randn_like(candidates) * std
                            candidates = candidates.clamp(
                                float(action_bounds[0]), float(action_bounds[1])
                            )
                            std *= _dfo_decay
                    # FIX: re-score AFTER the final reorder so argmax index lines
                    # up with the (now reordered) candidates tensor. The previous
                    # version used log_probs from the iteration's pre-reorder
                    # scoring and indexed into the reordered candidates, picking
                    # the wrong action when softmax mass was spread (the bug was
                    # masked on pushing where Q is sharply peaked).
                    final_log_probs = self.q_estimator(obs_expanded, _norm(candidates)).squeeze(-1)
                    sel = final_log_probs.argmax(dim=1)
                    action_normalized = candidates[0, sel[0], :].cpu().numpy()
                    q_range = (final_log_probs.min().item(), final_log_probs.max().item())

                action = np.clip(action_normalized, action_bounds[0], action_bounds[1])
                # _denormalize_action maps model-space (e.g., [-1, 1] for pushing)
                # back to env-action space. It's a no-op for particle where
                # _raw_act_min is None, but REQUIRED for pushing.
                action = self._denormalize_action(action)
                if return_q_range:
                    return action, q_range
                return action

        sim_cls = DFORefinedSimulation
    elif inference_langevin_iterations > 0:
        class LangevinRefinedParticleSimulation(SimulationCls):
            """Refines the highest-Q control point with Langevin MCMC before acting."""

            def select_action(self, observation, return_q_range: bool = False):
                obs_tensor = (
                    torch.tensor(observation, dtype=torch.float32)
                    .unsqueeze(0)
                    .to(self.device)
                )
                obs_tensor = self.obs_normalizer.normalize(obs_tensor)

                with torch.no_grad():
                    cps = self.control_point_generator(obs_tensor)  # (1, N, D)
                    obs_expanded = obs_tensor.unsqueeze(1).expand(-1, cps.shape[1], -1)
                    if self._act_min_t is not None:
                        cp_for_q = (cps - self._act_min_t) / self._act_rng_t
                    else:
                        cp_for_q = cps
                    q_values = self.q_estimator(obs_expanded, cp_for_q).squeeze(-1)
                    best_idx = q_values.argmax(dim=1)
                    q_range = (q_values.min().item(), q_values.max().item())
                    best_cp = cps[0, best_idx[0], :].view(1, 1, -1).clone()

                act_min_t = torch.full(
                    (cps.shape[-1],), float(action_bounds[0]), device=self.device
                )
                act_max_t = torch.full(
                    (cps.shape[-1],), float(action_bounds[1]), device=self.device
                )

                for p in self.q_estimator.parameters():
                    p.requires_grad_(False)

                _norm_min = self._act_min_t
                _norm_rng = self._act_rng_t

                def _neg_energy_fn(obs_lv, actions_lv):
                    if _norm_min is not None:
                        a_in = (actions_lv - _norm_min) / _norm_rng
                    else:
                        a_in = actions_lv
                    return -self.q_estimator(obs_lv, a_in).squeeze(-1)

                refined = sample_langevin(
                    energy_function=_neg_energy_fn,
                    observations=obs_tensor,
                    num_samples=1,
                    action_min=act_min_t,
                    action_max=act_max_t,
                    num_iterations=inference_langevin_iterations,
                    lr_init=float(langevin_cfg.get("lr_init", 0.1)),
                    lr_final=float(langevin_cfg.get("lr_final", 1e-5)),
                    polynomial_decay_power=float(
                        langevin_cfg.get("polynomial_decay_power", 2.0)
                    ),
                    delta_action_clip=float(
                        langevin_cfg.get("delta_action_clip", 0.1)
                    ),
                    noise_scale=float(langevin_cfg.get("noise_scale", 1.0)),
                    initial_actions=best_cp,
                    device=self.device,
                )

                for p in self.q_estimator.parameters():
                    p.requires_grad_(True)

                action = refined[0, 0, :].cpu().numpy()
                action = np.clip(action, action_bounds[0], action_bounds[1])
                # Denormalize to the env's native action box when the
                # simulation declares a non-identity inverse (Pushing). For
                # ParticleSimulation this is a no-op (action_bounds = [0, 1]
                # is already the env action box).
                action = self._denormalize_action(action)
                if return_q_range:
                    return action, q_range
                return action

        sim_cls = LangevinRefinedParticleSimulation
    else:
        sim_cls = SimulationCls

    # PushingSimulation has no n_dim arg (1-block/1-target, fixed schema)
    # but has its own goal_dist_tolerance knob (IBC paper used 0.02).
    sim_kwargs: dict = dict(
        control_point_generator=cp_gen,
        q_estimator=q_est,
        device=device,
        max_episode_steps=max_episode_steps,
        render_mode=None,
        frame_stack=frame_stack,
        norm_stats=norm_stats,
    )
    if active_env == "pushing":
        sim_kwargs["goal_dist_tolerance"] = float(
            env_config.get("goal_dist_tolerance", 0.02)
        )
    elif active_env == "pushing_multi":
        # IBC class default for the multimodal variant is 0.04 (looser than
        # single-target because both blocks must satisfy the criterion).
        sim_kwargs["goal_dist_tolerance"] = float(
            env_config.get("goal_dist_tolerance", 0.04)
        )
    elif active_env == "pushing_pixels":
        # Single-target physics — same 0.02 tolerance as states variant.
        sim_kwargs["goal_dist_tolerance"] = float(
            env_config.get("goal_dist_tolerance", 0.02)
        )
    elif active_env in ("pen", "door", "kitchen"):
        # Adroit D4RL + FrankaKitchen — no goal_dist_tolerance / n_dim knobs.
        pass
    elif active_env == "libero_goal":
        # Multi-task language-conditioned eval — obs schema + goal embeddings
        # come from norm_stats; no n_dim / tolerance knobs.
        pass
    else:
        sim_kwargs["n_dim"] = n_dim
    sim = sim_cls(**sim_kwargs)

    all_results = []
    for seed in seeds:
        result = sim.run_episode(seed=seed)
        all_results.append(result)
    sim.close()

    def _finite(x: float) -> float | None:
        """JSON-safe: inf/nan → None so trials.jsonl stays strictly valid JSON."""
        xf = float(x)
        return xf if np.isfinite(xf) else None

    successes = [bool(r.get("success", False)) for r in all_results]
    rewards = [float(r.get("total_reward", 0.0)) for r in all_results]
    ep_lengths = [int(r.get("episode_length", 0)) for r in all_results]
    terminated_flags = [bool(r.get("terminated", False)) for r in all_results]

    if active_env == "kitchen":
        # FrankaKitchen headline metric = avg_tasks_completed (0..N), matching
        # IBC Table 2 (kitchen-complete = 3.37/4). success = solved ALL tasks.
        tasks_done = [int(r.get("tasks_completed", 0)) for r in all_results]
        return {
            "success_rate": float(np.mean(successes)),
            "success_rate_std": float(np.std(successes)),
            "avg_tasks_completed": float(np.mean(tasks_done)),
            "std_tasks_completed": float(np.std(tasks_done)),
            "median_tasks_completed": float(np.median(tasks_done)),
            "avg_reward": float(np.mean(rewards)),
            "std_reward": float(np.std(rewards)),
            "median_reward": float(np.median(rewards)),
            "avg_episode_length": float(np.mean(ep_lengths)),
            "num_seeds": len(seeds),
            "per_seed": [
                {
                    "seed": seeds[i],
                    "success": successes[i],
                    "tasks_completed": tasks_done[i],
                    "reward": rewards[i],
                    "episode_length": ep_lengths[i],
                    "terminated": terminated_flags[i],
                }
                for i in range(len(seeds))
            ],
        }

    if active_env in ("pen", "door", "libero_goal"):
        # Adroit D4RL human tasks AND LIBERO-Goal report success_rate as the
        # headline metric (LIBERO's canonical number is per-suite success rate;
        # the env emits a binary success info bit). avg_reward is logged too but
        # is secondary for libero_goal. per_seed here is per-eval-episode; for
        # libero_goal the sim cycles tasks across episodes (see LiberoGoalSimulation).
        return {
            "success_rate": float(np.mean(successes)),
            "success_rate_std": float(np.std(successes)),
            "avg_reward": float(np.mean(rewards)),
            "std_reward": float(np.std(rewards)),
            "median_reward": float(np.median(rewards)),
            "avg_episode_length": float(np.mean(ep_lengths)),
            "num_seeds": len(seeds),
            "per_seed": [
                {
                    "seed": seeds[i],
                    "success": successes[i],
                    "reward": rewards[i],
                    "episode_length": ep_lengths[i],
                    "terminated": terminated_flags[i],
                }
                for i in range(len(seeds))
            ],
        }

    if active_env in ("pushing", "pushing_pixels"):
        # Single-target pushing (states OR pixels) — single goal, same metric
        # layout so the trial logs / analyzer queries stay uniform across the
        # two observation modalities.
        dists_target = [float(r.get("min_dist_to_target", np.inf)) for r in all_results]
        finite_target = [d for d in dists_target if np.isfinite(d)]
        return {
            "success_rate": float(np.mean(successes)),
            "success_rate_std": float(np.std(successes)),
            "avg_reward": float(np.mean(rewards)),
            "std_reward": float(np.std(rewards)),
            "median_reward": float(np.median(rewards)),
            "avg_min_dist_to_target": float(np.mean(finite_target)) if finite_target else None,
            "std_min_dist_to_target": float(np.std(finite_target)) if finite_target else None,
            "median_min_dist_to_target": float(np.median(finite_target)) if finite_target else None,
            "avg_episode_length": float(np.mean(ep_lengths)),
            "num_seeds": len(seeds),
            "per_seed": [
                {
                    "seed": seeds[i],
                    "success": successes[i],
                    "reward": rewards[i],
                    "min_dist_to_target": _finite(dists_target[i]),
                    "episode_length": ep_lengths[i],
                    "terminated": terminated_flags[i],
                }
                for i in range(len(seeds))
            ],
        }

    if active_env == "pushing_multi":
        # Multimodal pushing: 2 blocks, 2 targets. Each block is independently
        # assigned to its closest target (mirrors IBC's _get_reward). We log
        # per-block min distances + the mean so trial logs surface partial
        # progress when only one block lands.
        d_mean = [float(r.get("min_mean_dist_to_target", np.inf)) for r in all_results]
        d_b0 = [float(r.get("min_block0_dist_to_target", np.inf)) for r in all_results]
        d_b1 = [float(r.get("min_block1_dist_to_target", np.inf)) for r in all_results]
        finite_mean = [d for d in d_mean if np.isfinite(d)]
        finite_b0 = [d for d in d_b0 if np.isfinite(d)]
        finite_b1 = [d for d in d_b1 if np.isfinite(d)]
        return {
            "success_rate": float(np.mean(successes)),
            "success_rate_std": float(np.std(successes)),
            "avg_reward": float(np.mean(rewards)),
            "std_reward": float(np.std(rewards)),
            "median_reward": float(np.median(rewards)),
            "avg_min_mean_dist_to_target": float(np.mean(finite_mean)) if finite_mean else None,
            "std_min_mean_dist_to_target": float(np.std(finite_mean)) if finite_mean else None,
            "median_min_mean_dist_to_target": float(np.median(finite_mean)) if finite_mean else None,
            "avg_min_block0_dist_to_target": float(np.mean(finite_b0)) if finite_b0 else None,
            "avg_min_block1_dist_to_target": float(np.mean(finite_b1)) if finite_b1 else None,
            "avg_episode_length": float(np.mean(ep_lengths)),
            "num_seeds": len(seeds),
            "per_seed": [
                {
                    "seed": seeds[i],
                    "success": successes[i],
                    "reward": rewards[i],
                    "min_mean_dist_to_target": _finite(d_mean[i]),
                    "min_block0_dist_to_target": _finite(d_b0[i]),
                    "min_block1_dist_to_target": _finite(d_b1[i]),
                    "episode_length": ep_lengths[i],
                    "terminated": terminated_flags[i],
                }
                for i in range(len(seeds))
            ],
        }

    dists_first = [float(r.get("min_dist_to_first_goal", np.inf)) for r in all_results]
    dists_second = [float(r.get("min_dist_to_second_goal", np.inf)) for r in all_results]
    finite_first = [d for d in dists_first if np.isfinite(d)]
    finite_second = [d for d in dists_second if np.isfinite(d)]

    return {
        "success_rate": float(np.mean(successes)),
        "avg_reward": float(np.mean(rewards)),
        "std_reward": float(np.std(rewards)),
        "median_reward": float(np.median(rewards)),
        "avg_min_dist_first_goal": float(np.mean(finite_first)) if finite_first else None,
        "avg_min_dist_second_goal": float(np.mean(finite_second)) if finite_second else None,
        "median_min_dist_first_goal": float(np.median(finite_first)) if finite_first else None,
        "median_min_dist_second_goal": float(np.median(finite_second)) if finite_second else None,
        "avg_episode_length": float(np.mean(ep_lengths)),
        "num_seeds": len(seeds),
        "per_seed": [
            {
                "seed": seeds[i],
                "success": successes[i],
                "reward": rewards[i],
                "min_dist_first_goal": _finite(dists_first[i]),
                "min_dist_second_goal": _finite(dists_second[i]),
                "episode_length": ep_lengths[i],
                "terminated": terminated_flags[i],
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
    training_steps_override: int | None = None,
    timeout: int | None = None,
    active_env_override: str | None = None,
) -> dict:
    """Write a per-run config, train, evaluate, and atomically append the trial record.

    Every trial gets a unique run_id. Its config is written to a unique path and the
    training subprocess reads it via Q3C_CONFIG_PATH. The checkpoint directory is also
    run-id-scoped. No shared config.json mutation occurs, so parallel sbatch jobs do
    not collide on either config state or checkpoint files.

    `active_env_override`, when set, pins the env for this trial across training,
    evaluation, AND result-logging. This is the supported way to dispatch envs
    from the CLI — flipping config.json's active_env is racy.
    """
    script_name = script_path.name
    run_id = _new_run_id()

    print(f"\n{'=' * 80}")
    print(f"RUN {run_id} — {script_name}")
    print(f"{'=' * 80}")
    print(f"Parameters:\n{json.dumps(params, indent=2)}")

    config = load_config()
    if active_env_override is not None:
        if active_env_override not in config.get("environments", {}):
            raise ValueError(
                f"--active-env {active_env_override!r} is not in config.json's "
                f"environments. Known: {list(config.get('environments', {}).keys())}"
            )
        config["active_env"] = active_env_override
        print(f"  active_env override → {active_env_override}")
    config = apply_params_to_config(config, params)
    checkpoint_dir = set_run_checkpoint_dir(config, run_id)

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

    # Per-run config lives next to the checkpoints so you can always reconstruct a run.
    os.makedirs(checkpoint_dir, exist_ok=True)
    trial_config_path = Path(checkpoint_dir) / "config.json"
    with open(trial_config_path, "w") as f:
        json.dump(config, f, indent=4)

    print(f"\n  Training ({actual_steps} steps) — config at {trial_config_path}")
    success, stdout, duration = run_training(
        script_path,
        timeout=timeout,
        env_extras={"Q3C_CONFIG_PATH": str(trial_config_path)},
    )

    timestamp = datetime.now(timezone.utc).isoformat()

    if not success:
        print(f"\n  Training FAILED after {duration:.0f}s")
        last_lines = "\n".join(stdout.strip().splitlines()[-5:])
        record = {
            "run_id": run_id,
            "script": script_name,
            "active_env": active_env,
            "params": params,
            "training_steps": actual_steps,
            "duration_seconds": round(duration, 1),
            "success_rate": 0.0,
            "avg_reward": 0.0,
            "training_failed": True,
            "error": last_lines[-300:],
            "checkpoint_dir": checkpoint_dir,
            "timestamp": timestamp,
        }
        trial_id = append_trial(script_name, record, active_env=active_env)
        print(f"  Recorded as trial #{trial_id}")
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

    # Env-agnostic record schema. Particle eval returns `*_first_goal` /
    # `*_second_goal` keys; pushing returns `*_to_target` keys. We spread
    # ALL non-private eval scalars into the top level so analyzers don't
    # silently drop env-specific metrics. Particle-style keys are kept as
    # explicit fields for backward compatibility with the legacy analyzer.
    env_specific = {
        k: v for k, v in eval_results.items()
        if k not in ("per_seed", "error", "success_rate", "avg_reward")
    }
    record = {
        "run_id": run_id,
        "script": script_name,
        "active_env": active_env,
        "params": params,
        "training_steps": actual_steps,
        "duration_seconds": round(duration, 1),
        "success_rate": eval_results.get("success_rate", 0.0),
        "avg_reward": eval_results.get("avg_reward", 0.0),
        **env_specific,
        **train_metrics,
        "eval_details": eval_results.get("per_seed", []),
        "eval_error": eval_results.get("error"),
        "checkpoint_dir": checkpoint_dir,
        "timestamp": timestamp,
    }
    trial_id = append_trial(script_name, record, active_env=active_env)

    import math as _math
    sr = record["success_rate"]
    rw = record["avg_reward"]
    rw_std = record.get("std_reward")
    n_eval = int(record.get("num_seeds") or 1)
    # Print SEM = σ_ep / √n_eval — standard error of THIS trial's mean over
    # its n_eval episodes. Cross-seed SEM (over multiple training seeds) is
    # what print_analysis aggregates; this single-trial SEM is the
    # within-trial counterpart.
    if rw_std is not None and n_eval > 0:
        sem = rw_std / _math.sqrt(n_eval)
        rw_str = f"{rw:.3f} ± {sem:.3f} (SEM, n={n_eval}; σ_ep={rw_std:.1f})"
    else:
        rw_str = f"{rw:.3f}"
    tc = record.get("avg_tasks_completed")
    tc_str = f", avg_tasks_completed={tc:.3f}" if tc is not None else ""
    print(
        f"\n  Result (trial #{trial_id}): success_rate={sr:.2%}{tc_str}, avg_reward={rw_str}"
    )
    return record


# ─── Analyze / summary ───────────────────────────────────────────────────────

def print_analysis(
    script_name: str,
    active_env: str | None = None,
    min_trial_id: int = 0,
) -> None:
    """Print a formatted results table sorted by success rate.

    `min_trial_id`: skip trials with id below this value — useful when env
    config has changed (e.g. pen `max_episode_steps` 200→100) and earlier
    trials have the same `params` dict but were trained under a different
    protocol. Defaults to 0 (no filter).
    """
    trials = load_trials(script_name, active_env=active_env)
    if min_trial_id > 0:
        trials = [t for t in trials if int(t.get("trial_id", 0)) >= min_trial_id]
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
    cols += [("Steps", 8), ("Success", 8), ("Reward", 18), ("Loss", 8), ("Time", 7)]

    header = " | ".join(f"{name:>{w}}" for name, w in cols)
    separator = "-+-".join("-" * w for _, w in cols)

    print(f"\n{'=' * len(header)}")
    print(f"  Hyperparameter search results: {script_name}")
    print(f"{'=' * len(header)}")
    print(header)
    print(separator)

    env_names_for_sort = {t.get("active_env") for t in trials}
    if env_names_for_sort <= {"pen", "door"}:
        sorted_trials = sorted(
            trials, key=lambda t: t.get("avg_reward", float("-inf")), reverse=True
        )
    else:
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
        rw_std = t.get("std_reward")
        loss = t.get("final_train_loss")
        dur = t.get("duration_seconds", 0)

        row_vals.append("  FAILED" if failed else f"{sr:>7.0%}")
        rw_cell = (
            f"{rw:.3f} ± {rw_std:.3f}" if rw_std is not None else f"{rw:.3f}"
        )
        row_vals.append(f"{rw_cell:>18}")
        row_vals.append(f"{loss:>8.4f}" if loss is not None else f"{'—':>8}")
        row_vals.append(f"{dur / 60:>6.1f}m")

        print(" | ".join(row_vals))

    print(f"{'=' * len(header)}")

    valid = [t for t in trials if not t.get("training_failed")]
    if valid:
        # D4RL Adroit objectives are reward (IBC paper Table 2 reports raw
        # returns for pen/door). Other envs still rank by success_rate.
        env_names = {t.get("active_env") for t in valid}
        if env_names <= {"pen", "door"}:
            best = max(valid, key=lambda t: t.get("avg_reward", float("-inf")))
        else:
            best = max(valid, key=lambda t: t.get("success_rate", 0))
        best_std = best.get("std_reward")
        rw_str = (
            f"{best['avg_reward']:.3f} ± {best_std:.3f}"
            if best_std is not None
            else f"{best['avg_reward']:.3f}"
        )
        print(f"\nBest trial: #{best['trial_id']}  "
              f"success_rate={best['success_rate']:.2%}  "
              f"avg_reward={rw_str}")
        print(f"  Params: {json.dumps(best['params'], indent=4)}")

    print(f"\nTotal trials: {len(trials)} ({len(valid)} completed, "
          f"{len(trials) - len(valid)} failed)\n")

    # ── Cross-seed aggregation table ──────────────────────────────────────
    # Groups trials with identical config but different `trial_seed`. For
    # each group computes:
    #   - mean of per-seed means
    #   - σ_ep_avg : average per-episode std across the group's seeds
    #     (env-intrinsic spread of episode rewards — bimodal on pen)
    #   - cross_std : sample stdev (ddof=1) of per-seed means
    #     (cross-seed variability — comparable to IBC paper's "± std")
    #   - cross_sem : cross_std / √n  (standard error of the mean across
    #     seeds — what IBC paper Table 2's ±65 most likely reports)
    import math
    from collections import defaultdict

    # Skip trials with eval errors (those have avg_reward=0 and would distort
    # cross-seed means if mixed with successful trials of the same config).
    eval_ok = [t for t in valid if not t.get("eval_error")]

    sig_groups: dict[str, list[dict]] = defaultdict(list)
    for t in eval_ok:
        p = dict(t.get("params") or {})
        p.pop("trial_seed", None)
        sig = json.dumps(p, sort_keys=True, default=str)
        sig_groups[sig].append(t)

    multi = [
        (sig, ts) for sig, ts in sig_groups.items() if len(ts) >= 2
    ]
    if multi:
        rows = []
        for sig, ts in multi:
            means = [float(t.get("avg_reward", 0)) for t in ts]
            per_ep = [float(t.get("std_reward") or 0) for t in ts]
            srs = [float(t.get("success_rate", 0)) for t in ts]
            n = len(means)
            mean_ = sum(means) / n
            var = sum((m - mean_) ** 2 for m in means) / (n - 1)
            cross_std = math.sqrt(var)
            cross_sem = cross_std / math.sqrt(n)
            avg_per_ep = sum(per_ep) / n
            avg_sr = sum(srs) / n
            seeds = sorted(
                {(t.get("params") or {}).get("trial_seed") for t in ts}
            )
            tid_list = sorted(t.get("trial_id", 0) for t in ts)
            rows.append((mean_, cross_std, cross_sem, avg_per_ep, avg_sr, n, seeds, tid_list))
        rows.sort(key=lambda r: -r[0])

        print("=" * 110)
        print("Cross-seed aggregates (groups with ≥2 trials of same config, different trial_seed)")
        print("=" * 110)
        print(
            f"{'n':>3} {'seeds':<14} {'trial_ids':<22} {'mean_R':>10} "
            f"{'cross_std':>10} {'SEM':>8} {'σ_ep(avg)':>11} {'SR(avg)':>8}"
        )
        print("-" * 110)
        for mean_, cstd, csem, pep, sr, n, seeds, tids in rows[:25]:
            seed_str = ",".join(str(s) for s in seeds)
            tid_str = ",".join(str(t) for t in tids[:6]) + ("…" if len(tids) > 6 else "")
            print(
                f"{n:>3} {seed_str:<14} {tid_str:<22} {mean_:>10.1f} "
                f"{cstd:>10.2f} {csem:>8.2f} {pep:>11.1f} {sr*100:>7.1f}%"
            )
        print("=" * 110)
        print(
            "  σ_ep(avg)  : intrinsic per-episode reward spread, averaged across the group's seeds.\n"
            "  cross_std  : sample stdev of per-seed means (sometimes printed as ± in papers).\n"
            "  SEM        : cross_std / √n  — IBC paper Table 2 ±65 best matches this interpretation.\n"
        )


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
        "--min-trial-id", type=int, default=0,
        help="When analyzing, skip trials with id below this value. Useful "
             "to scope cross-seed aggregation to a recent batch when env "
             "protocol has changed (e.g. pen max_episode_steps 200→100).",
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
    parser.add_argument(
        "--num-reps",
        type=int,
        default=1,
        help=(
            "Number of repetitions per config (each rep gets trial_seed=0,1,...)."
            " Use this to measure variance honestly (default: 1)."
        ),
    )
    parser.add_argument(
        "--active-env",
        type=str,
        default=None,
        help=(
            "Override `active_env` from config.json for this invocation only. "
            "When set, the trial trains, evaluates, AND is logged under this "
            "env — no race with concurrent config.json edits. Recommended for "
            "all SLURM batches. Choices: any key under environments.* in config.json."
        ),
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

    active_env_cli = args.active_env

    # ── Analyze mode ──────────────────────────────────────────────────────
    if args.analyze:
        print_analysis(
            script_name, active_env=active_env_cli,
            min_trial_id=args.min_trial_id,
        )
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
    if active_env_cli is not None:
        # Make baseline reflect the override so the baseline params come from
        # the right env's training/model blocks.
        if active_env_cli not in config.get("environments", {}):
            print(
                f"Error: --active-env {active_env_cli!r} not found in config.json. "
                f"Known: {list(config.get('environments', {}).keys())}"
            )
            sys.exit(1)
        config["active_env"] = active_env_cli
        print(f"Active env override (CLI): {active_env_cli}")
    baseline = get_baseline_params(config, detected_params)
    print(f"\nBaseline (current config):")
    for k, v in baseline.items():
        print(f"  {k} = {v}")
    print()

    # ── Run mode ──────────────────────────────────────────────────────────
    if args.run:
        if args.params:
            user_params = json.loads(args.params)
            params = baseline.copy()
            params.update(user_params)
        else:
            trials = load_trials(script_name, active_env=active_env_cli)
            params, reason = suggest_next_params(trials, detected_params, baseline)
            print(f"Auto-suggested ({reason})")

        if fixed_params:
            params.update(fixed_params)
            print(f"Applied fixed params: {json.dumps(fixed_params)}")

        seed_pinned = "trial_seed" in params
        for rep in range(max(1, args.num_reps)):
            rep_params = dict(params)
            if not seed_pinned:
                rep_params["trial_seed"] = rep
            if args.num_reps > 1:
                print(f"\n[rep {rep + 1}/{args.num_reps}] trial_seed={rep_params['trial_seed']}")
            run_single_trial(
                script_path=script_path,
                params=rep_params,
                training_steps_override=args.reduced_steps,
                timeout=args.timeout,
                active_env_override=active_env_cli,
            )
        return

    # ── Auto mode ─────────────────────────────────────────────────────────
    if args.auto:
        for i in range(args.max_trials):
            trials = load_trials(script_name, active_env=active_env_cli)
            params, reason = suggest_next_params(trials, detected_params, baseline)
            print(f"\n[Auto {i + 1}/{args.max_trials}] Strategy: {reason}")

            if fixed_params:
                params.update(fixed_params)
                print(f"[Auto {i + 1}/{args.max_trials}] Fixed params: {json.dumps(fixed_params)}")

            seed_pinned = "trial_seed" in params
            for rep in range(max(1, args.num_reps)):
                rep_params = dict(params)
                if not seed_pinned:
                    rep_params["trial_seed"] = rep
                if args.num_reps > 1:
                    print(f"  [rep {rep + 1}/{args.num_reps}] trial_seed={rep_params['trial_seed']}")
                run_single_trial(
                    script_path=script_path,
                    params=rep_params,
                    training_steps_override=args.reduced_steps,
                    timeout=args.timeout,
                    active_env_override=active_env_cli,
                )

        print("\n\nAuto-exploration complete. Full results:")
        print_analysis(script_name, active_env=active_env_cli)


if __name__ == "__main__":
    main()
