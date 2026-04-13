"""Diagnostic: Does the Q-estimator matter for action selection, or is the generator doing all the work?

Tests the best MSE=20 model with three inference strategies:
  1. Q-selected (normal): pick control point with highest Q-value
  2. Random: pick a random control point
  3. Closest-to-mean: pick the control point closest to the mean of all control points
  4. Q-worst: pick control point with LOWEST Q-value

If Q-selected >> Random, the implicit model matters.
If Q-selected ≈ Random, the generator is doing all the work (explicit).
"""

import json
import sys
from pathlib import Path
from collections import deque

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent))

from simulations.particle_env import ParticleEnv
from utils.models import ControlPointGenerator, QEstimator
from utils.normalizations import ObservationNormalizer


def load_config():
    with open("config_json/config.json") as f:
        return json.load(f)


def run_episode(env, generator, estimator, obs_normalizer, device, action_bounds, frame_stack, strategy="q_best"):
    obs, _ = env.reset()
    frame_buffer = deque([obs] * frame_stack, maxlen=frame_stack)
    stacked = np.concatenate(list(frame_buffer), axis=0)

    total_reward = 0.0
    done = False
    q_spreads = []

    while not done:
        state_tensor = torch.from_numpy(stacked).float().unsqueeze(0).to(device)
        state_norm = obs_normalizer.normalize(state_tensor)

        with torch.no_grad():
            control_points = generator(state_norm)  # (1, N, action_dim)
            state_expanded = state_norm.unsqueeze(1).expand(-1, control_points.shape[1], -1)
            q_values = estimator(state_expanded, control_points).squeeze(-1)  # (1, N)

            q_min = q_values.min().item()
            q_max = q_values.max().item()
            q_spreads.append(q_max - q_min)

            if strategy == "q_best":
                best_idx = q_values.argmax(dim=-1)
            elif strategy == "q_worst":
                best_idx = q_values.argmin(dim=-1)
            elif strategy == "random":
                best_idx = torch.randint(0, control_points.shape[1], (1,), device=device)
            elif strategy == "mean_closest":
                mean_cp = control_points.mean(dim=1, keepdim=True)  # (1, 1, action_dim)
                dists = (control_points - mean_cp).norm(dim=-1)  # (1, N)
                best_idx = dists.argmin(dim=-1)
            else:
                raise ValueError(f"Unknown strategy: {strategy}")

            action = control_points[0, best_idx[0]].cpu().numpy()

        action = np.clip(action, action_bounds[0], action_bounds[1])
        obs, reward, terminated, truncated, _ = env.step(action)
        frame_buffer.append(obs)
        stacked = np.concatenate(list(frame_buffer), axis=0)
        total_reward += reward
        done = terminated or truncated

    return total_reward, np.mean(q_spreads)


def main():
    cfg = load_config()
    env_cfg = cfg["environments"]["particle"]
    n_dim = env_cfg.get("n_dim", 5)
    state_dim = env_cfg["state_dim"]
    action_dim = env_cfg["action_dim"]
    action_bounds = tuple(env_cfg.get("action_bounds", [0, 1]))
    frame_stack = env_cfg.get("frame_stack", 2)
    model_cfg = env_cfg.get("model", {})
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load best MSE=20 models (trial_025 or trial_026)
    for trial_dir in ["checkpoints/hpsearch/trial_025", "checkpoints/hpsearch/trial_026"]:
        trial_path = Path(trial_dir)
        if not trial_path.exists():
            continue

        cp_path = trial_path / "control_point_generator.pt"
        qe_path = trial_path / "q_estimator.pt"
        if not cp_path.exists() or not qe_path.exists():
            continue

        # Infer control_points from state dict
        cp_sd = torch.load(cp_path, map_location=device, weights_only=True)
        for k, v in cp_sd.items():
            if k.endswith(".weight") and v.shape[0] % action_dim == 0:
                last_layer_key = k
        output_features = cp_sd[last_layer_key].shape[0]
        n_control_points = output_features // action_dim

        hidden_dims = [model_cfg.get("num_neurons", 512)] * model_cfg.get("num_hidden_layers", 8)

        generator = ControlPointGenerator(
            input_dim=state_dim * frame_stack,
            output_dim=action_dim,
            control_points=n_control_points,
            hidden_dims=hidden_dims,
            action_bounds=action_bounds,
        )
        generator.load_state_dict(cp_sd)
        generator.to(device).eval()

        qe_sd = torch.load(qe_path, map_location=device, weights_only=True)
        estimator = QEstimator(state_dim=state_dim * frame_stack, action_dim=action_dim, hidden_dims=hidden_dims)
        estimator.load_state_dict(qe_sd)
        estimator.to(device).eval()

        obs_normalizer = ObservationNormalizer(
            env_id=env_cfg["env_id"], device=device, frame_stack=frame_stack,
            particle_n_dim=n_dim,
        )

        print(f"\n{'='*70}")
        print(f"Diagnostic: {trial_dir} (n_dim={n_dim}, CP={n_control_points})")
        print(f"{'='*70}")

        seeds = list(range(50))
        strategies = ["q_best", "random", "mean_closest", "q_worst"]

        for strategy in strategies:
            successes = 0
            total_q_spread = 0
            for seed in seeds:
                env = ParticleEnv(n_dim=n_dim, n_steps=env_cfg.get("max_episode_steps", 50))
                env.reset(seed=seed)
                env.close()
                env = ParticleEnv(n_dim=n_dim, n_steps=env_cfg.get("max_episode_steps", 50))
                obs, _ = env.reset(seed=seed)
                env.close()

                env = ParticleEnv(n_dim=n_dim, n_steps=env_cfg.get("max_episode_steps", 50))
                env.reset(seed=seed)
                reward, q_spread = run_episode(env, generator, estimator, obs_normalizer, device, action_bounds, frame_stack, strategy)
                env.close()
                if reward > 0:
                    successes += 1
                total_q_spread += q_spread

            sr = successes / len(seeds) * 100
            avg_spread = total_q_spread / len(seeds)
            print(f"  {strategy:15s}: {sr:5.1f}% success ({successes}/{len(seeds)}) | avg Q-spread: {avg_spread:.4f}")


if __name__ == "__main__":
    main()
