"""Compare Particle success rate between IBC-DFO and Q3C-IBC.

This script runs evaluation without plotting and writes/updates one CSV row per n_dim.

Output CSV:
  results/particle/success_rates.csv
Columns:
  - n_dim
  - success_rate_dfo
  - success_rate_q3cibc
  - num_seeds

Seeding protocol follows the original IBC evaluation idea:
  seeds = [0, 1, ..., num_seeds - 1]
  one episode per seed

Usage:
  python -m simulations.run_particle_success_rate_comparison
"""

from __future__ import annotations

import csv
import json
import os
import sys
from collections import deque
from pathlib import Path

import numpy as np
import torch

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from simulations.particle_env import ParticleEnv
from utils.models import ControlPointGenerator, QEstimator
from utils.normalizations import ObservationNormalizer
from utils.sampling import sample_langevin


CONFIG_PATH = Path(__file__).parent.parent / "config_json" / "config.json"
RESULTS_CSV_PATH = Path(__file__).parent.parent / "results" / "particle" / "success_rates.csv"


def _load_config() -> dict:
    with open(CONFIG_PATH, "r") as f:
        return json.load(f)


def _to_tensor_bounds(action_bounds: tuple[float, float], action_dim: int, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    action_min = torch.full((action_dim,), action_bounds[0], dtype=torch.float32, device=device)
    action_max = torch.full((action_dim,), action_bounds[1], dtype=torch.float32, device=device)
    return action_min, action_max


def _infer_hidden_dims_from_state_dict(state_dict: dict[str, torch.Tensor]) -> list[int]:
    linear_indices: list[int] = []
    for key in state_dict.keys():
        if key.startswith("network.") and key.endswith(".weight"):
            try:
                idx = int(key.split(".")[1])
            except (ValueError, IndexError):
                continue
            linear_indices.append(idx)

    linear_indices = sorted(set(linear_indices))
    if len(linear_indices) < 2:
        raise ValueError("Could not infer hidden dims from state dict.")

    hidden_dims: list[int] = []
    for idx in linear_indices[:-1]:
        w = state_dict[f"network.{idx}.weight"]
        hidden_dims.append(int(w.shape[0]))
    return hidden_dims


def _load_q_estimator_from_checkpoint(
    checkpoint_path: str,
    state_dim: int,
    action_dim: int,
    default_hidden_dims: list[int],
    device: torch.device,
) -> tuple[QEstimator, dict | None]:
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)

    norm_stats = None
    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        state_dict = checkpoint["model_state_dict"]
        norm_stats = checkpoint.get("norm_stats")
    else:
        state_dict = checkpoint

    try:
        hidden_dims = _infer_hidden_dims_from_state_dict(state_dict)
    except Exception:
        hidden_dims = default_hidden_dims

    model = QEstimator(
        state_dim=state_dim,
        action_dim=action_dim,
        hidden_dims=hidden_dims,
    )
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    return model, norm_stats


def _load_control_point_generator(
    checkpoint_path: str,
    input_dim: int,
    action_dim: int,
    control_points: int,
    hidden_dims: list[int],
    action_bounds: tuple[float, float],
    device: torch.device,
) -> ControlPointGenerator:
    model = ControlPointGenerator(
        input_dim=input_dim,
        output_dim=action_dim,
        hidden_dims=hidden_dims,
        control_points=control_points,
        action_bounds=action_bounds,
    )
    state_dict = torch.load(checkpoint_path, map_location=device, weights_only=True)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    return model


def _denormalize_action(action_norm: np.ndarray, norm_stats: dict | None) -> np.ndarray:
    if norm_stats is None:
        return action_norm

    act_min = np.asarray(norm_stats["act_min"], dtype=np.float32)
    act_max = np.asarray(norm_stats["act_max"], dtype=np.float32)
    rng = act_max - act_min
    rng = np.where(rng == 0, np.ones_like(rng), rng)
    return action_norm * rng + act_min


def _select_action_dfo(
    stacked_obs: np.ndarray,
    q_estimator: QEstimator,
    obs_normalizer: ObservationNormalizer,
    dfo_norm_stats: dict | None,
    action_bounds: tuple[float, float],
    action_dim: int,
    device: torch.device,
    langevin_cfg: dict,
) -> np.ndarray:
    state_tensor = torch.from_numpy(stacked_obs).float().unsqueeze(0).to(device)
    state_norm = obs_normalizer.normalize(state_tensor)

    # Original IBC-style boundary buffer around normalized action range.
    boundary_buffer = 0.05
    norm_min = torch.full((action_dim,), -boundary_buffer, dtype=torch.float32, device=device)
    norm_max = torch.full((action_dim,), 1.0 + boundary_buffer, dtype=torch.float32, device=device)

    samples = sample_langevin(
        energy_function=q_estimator,
        observations=state_norm,
        num_samples=int(langevin_cfg.get("num_samples", 512)),
        action_min=norm_min,
        action_max=norm_max,
        num_iterations=int(langevin_cfg.get("num_iterations", 100)),
        lr_init=float(langevin_cfg.get("lr_init", 0.1)),
        lr_final=float(langevin_cfg.get("lr_final", 1e-5)),
        polynomial_decay_power=float(langevin_cfg.get("polynomial_decay_power", 2.0)),
        delta_action_clip=float(langevin_cfg.get("delta_action_clip", 0.1)),
        noise_scale=float(langevin_cfg.get("noise_scale", 1.0)),
        device=device,
    )

    with torch.no_grad():
        state_expanded = state_norm.unsqueeze(1).expand(-1, samples.shape[1], -1)
        energies = q_estimator(state_expanded, samples).squeeze(-1)
        best_idx = energies.argmin(dim=-1)
        best_action_norm = samples[0, best_idx[0]].cpu().numpy()

    action = _denormalize_action(best_action_norm, dfo_norm_stats)
    action = np.clip(action, action_bounds[0], action_bounds[1])
    return action


def _select_action_q3c(
    stacked_obs: np.ndarray,
    control_point_generator: ControlPointGenerator,
    q_estimator: QEstimator,
    obs_normalizer: ObservationNormalizer,
    action_bounds: tuple[float, float],
    device: torch.device,
) -> np.ndarray:
    state_tensor = torch.from_numpy(stacked_obs).float().unsqueeze(0).to(device)
    state_norm = obs_normalizer.normalize(state_tensor)

    with torch.no_grad():
        control_points = control_point_generator(state_norm)
        state_expanded = state_norm.unsqueeze(1).expand(-1, control_points.shape[1], -1)
        q_values = q_estimator(state_expanded, control_points).squeeze(-1)
        best_idx = q_values.argmax(dim=-1)
        action = control_points[0, best_idx[0]].cpu().numpy()

    action = np.clip(action, action_bounds[0], action_bounds[1])
    return action


def _run_method_success_rate(
    method: str,
    seeds: list[int],
    env_kwargs: dict,
    frame_stack: int,
    action_dim: int,
    action_bounds: tuple[float, float],
    obs_normalizer: ObservationNormalizer,
    device: torch.device,
    dfo_q_estimator: QEstimator | None = None,
    dfo_norm_stats: dict | None = None,
    q3c_control_point_generator: ControlPointGenerator | None = None,
    q3c_q_estimator: QEstimator | None = None,
    dfo_langevin_cfg: dict | None = None,
) -> float:
    successes = 0

    for seed in seeds:
        env = ParticleEnv(**env_kwargs)
        obs, _ = env.reset(seed=seed)

        frame_buffer: deque[np.ndarray] = deque(maxlen=frame_stack)
        for _ in range(frame_stack):
            frame_buffer.append(obs.copy())

        def _stacked() -> np.ndarray:
            if frame_stack <= 1:
                return frame_buffer[-1]
            return np.concatenate(list(frame_buffer))

        done = False
        final_success = False
        while not done:
            stacked_obs = _stacked()

            if method == "dfo":
                if dfo_q_estimator is None or dfo_langevin_cfg is None:
                    raise ValueError("DFO estimator/config not provided.")
                action = _select_action_dfo(
                    stacked_obs=stacked_obs,
                    q_estimator=dfo_q_estimator,
                    obs_normalizer=obs_normalizer,
                    dfo_norm_stats=dfo_norm_stats,
                    action_bounds=action_bounds,
                    action_dim=action_dim,
                    device=device,
                    langevin_cfg=dfo_langevin_cfg,
                )
            elif method == "q3cibc":
                if q3c_control_point_generator is None or q3c_q_estimator is None:
                    raise ValueError("Q3C models not provided.")
                action = _select_action_q3c(
                    stacked_obs=stacked_obs,
                    control_point_generator=q3c_control_point_generator,
                    q_estimator=q3c_q_estimator,
                    obs_normalizer=obs_normalizer,
                    action_bounds=action_bounds,
                    device=device,
                )
            else:
                raise ValueError(f"Unknown method: {method}")

            obs, _reward, terminated, truncated, info = env.step(action)
            frame_buffer.append(obs.copy())
            done = terminated or truncated
            final_success = bool(info.get("success", False))

        if final_success:
            successes += 1

        env.close()

    return successes / max(len(seeds), 1)


def _upsert_csv_row(
    csv_path: Path,
    n_dim: int,
    success_rate_dfo: float,
    success_rate_q3cibc: float,
    num_seeds: int,
    train_time_dfo: float | None = None,
    inference_time_dfo: float | None = None,
    train_time_q3cibc: float | None = None,
    inference_time_q3cibc: float | None = None,
) -> None:
    csv_path.parent.mkdir(parents=True, exist_ok=True)

    headers = [
        "n_dim",
        "success_rate_dfo", "train_time_dfo", "inference_time_dfo",
        "success_rate_q3cibc", "train_time_q3cibc", "inference_time_q3cibc",
        "num_seeds",
    ]
    rows: list[dict[str, str]] = []

    if csv_path.exists():
        with open(csv_path, "r", newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                filtered = {k: v for k, v in row.items() if k in headers}
                rows.append(filtered)

    def _fmt_time(t: float | None) -> str:
        if t is None:
            return ""
        return f"{t:.1f}"

    new_row = {
        "n_dim": str(n_dim),
        "success_rate_dfo": f"{success_rate_dfo:.6f}",
        "train_time_dfo": _fmt_time(train_time_dfo),
        "inference_time_dfo": _fmt_time(inference_time_dfo),
        "success_rate_q3cibc": f"{success_rate_q3cibc:.6f}",
        "train_time_q3cibc": _fmt_time(train_time_q3cibc),
        "inference_time_q3cibc": _fmt_time(inference_time_q3cibc),
        "num_seeds": str(num_seeds),
    }

    updated = False
    for row in rows:
        try:
            row_n_dim = int(row.get("n_dim", "-1"))
        except ValueError:
            continue

        if row_n_dim == n_dim:
            row.update(new_row)
            updated = True
            break

    if not updated:
        rows.append(new_row)

    rows.sort(key=lambda r: int(r.get("n_dim", "0")))

    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=headers)
        writer.writeheader()
        writer.writerows(rows)


import time as _time


def _get_dfo_train_time(checkpoint_path: str, device: torch.device) -> float | None:
    """Extract training duration from DFO checkpoint's sibling train_summary.json."""
    try:
        ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
        if not isinstance(ckpt, dict):
            return None
        run_id = ckpt.get("run_id") or (ckpt.get("hparams") or {}).get("RUN_ID")
        run_name = ckpt.get("run_name") or (ckpt.get("hparams") or {}).get("RUN_NAME")
        if run_id is not None and run_name:
            summary_path = os.path.join(
                os.path.dirname(checkpoint_path),
                f"run_{int(run_id):02d}_{run_name}",
                "train_summary.json",
            )
            if os.path.exists(summary_path):
                with open(summary_path) as f:
                    return json.load(f).get("duration_seconds")
    except Exception:
        pass
    return None


def _get_q3c_train_time(model_dir: str) -> float | None:
    """Try to find Q3C training duration from hpsearch trial results."""
    trials_path = os.path.join(
        "results", "hyperparam_search",
        "combinedv2_cpascounter_training", "trials.jsonl",
    )
    if not os.path.exists(trials_path):
        return None
    try:
        best_duration = None
        best_sr = -1.0
        with open(trials_path) as f:
            for line in f:
                trial = json.loads(line)
                sr = trial.get("success_rate", 0)
                if sr > best_sr:
                    best_sr = sr
                    best_duration = trial.get("duration_seconds")
        return best_duration
    except Exception:
        return None


def main() -> None:
    config = _load_config()
    env_config = config["environments"]["particle"]
    sim_config = config.get("simulation", {})
    training_shared = config.get("training_shared", {})

    n_dim = int(env_config["n_dim"])
    state_dim = int(env_config["state_dim"])
    action_dim = int(env_config["action_dim"])
    frame_stack = int(env_config.get("frame_stack", 1))
    action_bounds = tuple(env_config.get("action_bounds", [0.0, 1.0]))
    max_episode_steps = int(sim_config.get("max_episode_steps", 50))

    num_seeds = int(sim_config.get("num_seeds", len(sim_config.get("default_seeds", [0]))))
    if num_seeds <= 0:
        raise ValueError("simulation.num_seeds must be >= 1")

    seeds = list(range(num_seeds))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model_dir = training_shared.get("model_save_dir", "checkpoints")

    dfo_q_path = os.path.join(model_dir, "ibc_dfo", "particle", "q_estimator.pt")
    q3c_q_path = os.path.join(model_dir, "q_estimator.pt")
    q3c_cp_path = os.path.join(model_dir, "control_point_generator.pt")

    if not os.path.exists(dfo_q_path):
        raise FileNotFoundError(f"DFO checkpoint not found: {dfo_q_path}")
    if not os.path.exists(q3c_q_path):
        raise FileNotFoundError(f"Q3C Q-estimator checkpoint not found: {q3c_q_path}")
    if not os.path.exists(q3c_cp_path):
        raise FileNotFoundError(f"Q3C control-point checkpoint not found: {q3c_cp_path}")

    q3c_hidden_layers = int(env_config["model"]["num_hidden_layers"])
    q3c_hidden_neurons = int(env_config["model"]["num_neurons"])
    q3c_hidden_dims = [q3c_hidden_neurons for _ in range(q3c_hidden_layers)]

    dfo_q_estimator, dfo_norm_stats = _load_q_estimator_from_checkpoint(
        checkpoint_path=dfo_q_path,
        state_dim=state_dim * frame_stack,
        action_dim=action_dim,
        default_hidden_dims=[256, 256],
        device=device,
    )

    q3c_q_estimator, _ = _load_q_estimator_from_checkpoint(
        checkpoint_path=q3c_q_path,
        state_dim=state_dim * frame_stack,
        action_dim=action_dim,
        default_hidden_dims=q3c_hidden_dims,
        device=device,
    )

    q3c_control_point_generator = _load_control_point_generator(
        checkpoint_path=q3c_cp_path,
        input_dim=state_dim * frame_stack,
        action_dim=action_dim,
        control_points=int(env_config["model"]["control_points"]),
        hidden_dims=q3c_hidden_dims,
        action_bounds=(float(action_bounds[0]), float(action_bounds[1])),
        device=device,
    )

    obs_normalizer = ObservationNormalizer(
        env_id=env_config["env_id"],
        device=device,
        frame_stack=frame_stack,
        particle_n_dim=n_dim,
    )

    dfo_langevin_cfg = {
        "num_samples": 512,
        "num_iterations": 100,
        "lr_init": 0.1,
        "lr_final": 1e-5,
        "polynomial_decay_power": 2.0,
        "delta_action_clip": 0.1,
        "noise_scale": 1.0,
    }

    env_kwargs = {
        "n_dim": n_dim,
        "n_steps": max_episode_steps,
        "render_mode": None,
    }

    train_time_dfo = _get_dfo_train_time(dfo_q_path, device)
    train_time_q3cibc = _get_q3c_train_time(model_dir)

    print(f"Evaluating Particle success rate for n_dim={n_dim} over {num_seeds} seeds...")
    print(f"  DFO train time:  {train_time_dfo:.1f}s" if train_time_dfo else "  DFO train time:  N/A")
    print(f"  Q3C train time:  {train_time_q3cibc:.1f}s" if train_time_q3cibc else "  Q3C train time:  N/A")

    print("Method 1/2: IBC-DFO")
    t0 = _time.time()
    success_rate_dfo = _run_method_success_rate(
        method="dfo",
        seeds=seeds,
        env_kwargs=env_kwargs,
        frame_stack=frame_stack,
        action_dim=action_dim,
        action_bounds=(float(action_bounds[0]), float(action_bounds[1])),
        obs_normalizer=obs_normalizer,
        device=device,
        dfo_q_estimator=dfo_q_estimator,
        dfo_norm_stats=dfo_norm_stats,
        dfo_langevin_cfg=dfo_langevin_cfg,
    )
    inference_time_dfo = _time.time() - t0

    print("Method 2/2: Q3C-IBC")
    t0 = _time.time()
    success_rate_q3c = _run_method_success_rate(
        method="q3cibc",
        seeds=seeds,
        env_kwargs=env_kwargs,
        frame_stack=frame_stack,
        action_dim=action_dim,
        action_bounds=(float(action_bounds[0]), float(action_bounds[1])),
        obs_normalizer=obs_normalizer,
        device=device,
        q3c_control_point_generator=q3c_control_point_generator,
        q3c_q_estimator=q3c_q_estimator,
    )
    inference_time_q3cibc = _time.time() - t0

    _upsert_csv_row(
        csv_path=RESULTS_CSV_PATH,
        n_dim=n_dim,
        success_rate_dfo=success_rate_dfo,
        success_rate_q3cibc=success_rate_q3c,
        num_seeds=num_seeds,
        train_time_dfo=train_time_dfo,
        inference_time_dfo=inference_time_dfo,
        train_time_q3cibc=train_time_q3cibc,
        inference_time_q3cibc=inference_time_q3cibc,
    )

    print("Done.")
    print(f"  n_dim: {n_dim}")
    print(f"  success_rate_dfo:     {success_rate_dfo:.4f}  (inference: {inference_time_dfo:.1f}s, train: {train_time_dfo:.1f}s)" if train_time_dfo else f"  success_rate_dfo:     {success_rate_dfo:.4f}  (inference: {inference_time_dfo:.1f}s)")
    print(f"  success_rate_q3cibc:  {success_rate_q3c:.4f}  (inference: {inference_time_q3cibc:.1f}s, train: {train_time_q3cibc:.1f}s)" if train_time_q3cibc else f"  success_rate_q3cibc:  {success_rate_q3c:.4f}  (inference: {inference_time_q3cibc:.1f}s)")
    print(f"  results file: {RESULTS_CSV_PATH}")


if __name__ == "__main__":
    main()
