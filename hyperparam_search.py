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
        "values": [50000, 100000, 200000],
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
        "values": [100000, 150000, 200000],
        "type": "int",
        "location": "env_training",
    },
    "target_update_interval": {
        "values": [200, 500, 1000, 2000, 5000],
        "type": "int",
        "location": "training_shared",
    },
    "inference_langevin_iterations": {
        "values": [0, 10, 25, 50, 100],
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


# ─── Config I/O ──────────────────────────────────────────────────────────────

def load_config() -> dict:
    """Read the on-disk default config. Never mutated by parallel trials."""
    with open(CONFIG_PATH, "r") as f:
        return json.load(f)


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


def append_trial(script_name: str, record: dict) -> int:
    """Atomically assign a monotonically-increasing trial_id and append the record.

    Uses fcntl.flock for an exclusive lock over the jsonl for the short read-max +
    write-line section. Safe under parallel sbatch submissions. Returns the id.
    """
    path = _trials_path(script_name)
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
INFERENCE_ONLY_PARAMS: set[str] = {"inference_langevin_iterations"}


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
    from simulations.particle_simulation import ParticleSimulation
    from utils.sampling import sample_langevin

    active_env = config.get("active_env", "particle")
    env_config = config["environments"][active_env]
    sim_config = config.get("simulation", {})

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
    max_episode_steps = sim_config.get("max_episode_steps", 50)
    num_seeds = int(sim_config.get("num_seeds", len(sim_config.get("default_seeds", [0]))))
    if num_seeds <= 0:
        raise ValueError("simulation.num_seeds must be >= 1")
    seeds = list(range(num_seeds))

    inference_langevin_iterations = int(
        env_config.get("training", {}).get("inference_langevin_iterations", 0)
    )
    # Effective langevin hyperparams: env_training.langevin_* overrides env_model.langevin_config.
    langevin_cfg = effective_langevin_config(env_config)

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
        state_dim=state_dim * frame_stack,
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

    if inference_langevin_iterations > 0:
        class LangevinRefinedParticleSimulation(ParticleSimulation):
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
                if return_q_range:
                    return action, q_range
                return action

        sim_cls = LangevinRefinedParticleSimulation
    else:
        sim_cls = ParticleSimulation

    sim = sim_cls(
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

    def _finite(x: float) -> float | None:
        """JSON-safe: inf/nan → None so trials.jsonl stays strictly valid JSON."""
        xf = float(x)
        return xf if np.isfinite(xf) else None

    successes = [bool(r.get("success", False)) for r in all_results]
    rewards = [float(r.get("total_reward", 0.0)) for r in all_results]
    dists_first = [float(r.get("min_dist_to_first_goal", np.inf)) for r in all_results]
    dists_second = [float(r.get("min_dist_to_second_goal", np.inf)) for r in all_results]
    ep_lengths = [int(r.get("episode_length", 0)) for r in all_results]
    terminated_flags = [bool(r.get("terminated", False)) for r in all_results]

    finite_first = [d for d in dists_first if np.isfinite(d)]
    finite_second = [d for d in dists_second if np.isfinite(d)]

    return {
        "success_rate": float(np.mean(successes)),
        "avg_reward": float(np.mean(rewards)),
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
) -> dict:
    """Write a per-run config, train, evaluate, and atomically append the trial record.

    Every trial gets a unique run_id. Its config is written to a unique path and the
    training subprocess reads it via Q3C_CONFIG_PATH. The checkpoint directory is also
    run-id-scoped. No shared config.json mutation occurs, so parallel sbatch jobs do
    not collide on either config state or checkpoint files.
    """
    script_name = script_path.name
    run_id = _new_run_id()

    print(f"\n{'=' * 80}")
    print(f"RUN {run_id} — {script_name}")
    print(f"{'=' * 80}")
    print(f"Parameters:\n{json.dumps(params, indent=2)}")

    config = load_config()
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
        trial_id = append_trial(script_name, record)
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

    record = {
        "run_id": run_id,
        "script": script_name,
        "params": params,
        "training_steps": actual_steps,
        "duration_seconds": round(duration, 1),
        "success_rate": eval_results.get("success_rate", 0.0),
        "avg_reward": eval_results.get("avg_reward", 0.0),
        "avg_min_dist_first_goal": eval_results.get("avg_min_dist_first_goal"),
        "avg_min_dist_second_goal": eval_results.get("avg_min_dist_second_goal"),
        "median_min_dist_first_goal": eval_results.get("median_min_dist_first_goal"),
        "median_min_dist_second_goal": eval_results.get("median_min_dist_second_goal"),
        "avg_episode_length": eval_results.get("avg_episode_length"),
        **train_metrics,
        "eval_details": eval_results.get("per_seed", []),
        "eval_error": eval_results.get("error"),
        "checkpoint_dir": checkpoint_dir,
        "timestamp": timestamp,
    }
    trial_id = append_trial(script_name, record)

    sr = record["success_rate"]
    rw = record["avg_reward"]
    print(f"\n  Result (trial #{trial_id}): success_rate={sr:.2%}, avg_reward={rw:.3f}")
    return record


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
    parser.add_argument(
        "--num-reps",
        type=int,
        default=1,
        help=(
            "Number of repetitions per config (each rep gets trial_seed=0,1,...)."
            " Use this to measure variance honestly (default: 1)."
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
            )
        return

    # ── Auto mode ─────────────────────────────────────────────────────────
    if args.auto:
        for i in range(args.max_trials):
            trials = load_trials(script_name)
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
                )

        print("\n\nAuto-exploration complete. Full results:")
        print_analysis(script_name)


if __name__ == "__main__":
    main()
