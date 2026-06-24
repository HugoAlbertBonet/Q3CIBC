"""Diffusion Policy training (capacity-matched Q3C baseline).

Trains a single epsilon-prediction denoiser (`utils.diffusion.DiffusionDenoiser`,
which reuses the Q3C `QEstimator` trunk) on offline expert (state, action) pairs.
At eval time the SAME checkpoint is sampled with both DDPM and DDIM — see
`hyperparam_search_dp.evaluate_dp`.

Driven by hyperparam_search_dp.py exactly like combinedv2_cpascounter_training.py
is driven by hyperparam_search.py: config path comes from Q3C_CONFIG_PATH, all
--fixed-params are routed into env_config['training'].

Saves into MODEL_SAVE_DIR:
  - denoiser.pt        (denoiser state_dict)
  - denoiser_ema.pt    (EMA weights, if ema_decay > 0)
  - norm_stats.pt      (obs_mean/std, act_min/max, action_norm_range, state_shape)
                        — SAME schema combinedv2 writes, so PushingSimulation's
                        obs-standardize + action-denormalize works unchanged.
"""

import copy
import json
import os
import random
import time
from pathlib import Path

import numpy as np
import torch

from utils.diffusion import build_denoiser, build_diffusion, resolve_dp_params
from utils.normalizations import ObservationNormalizer

# ── Config (Q3C_CONFIG_PATH when driven by hyperparam_search_dp) ──────────────
config_path = Path(
    os.environ.get("Q3C_CONFIG_PATH")
    or (Path(__file__).parent / "config_json" / "config.json")
)
with open(config_path, "r") as f:
    config = json.load(f)

active_env = config.get("active_env", "pushing")
env_config = config["environments"][active_env]
training_shared = config.get("training_shared", {})
env_training = env_config.get("training", {})

frame_stack = env_config.get("frame_stack", 1)
action_bounds = tuple(env_config.get("action_bounds", [-1.0, 1.0]))
env_id = env_config.get("env_id", active_env)

training_steps = int(env_training.get("training_steps", training_shared.get("training_steps", 100000)))
batch_size = int(env_training.get("batch_size", training_shared.get("batch_size", 512)))
learning_rate = float(env_training.get("learning_rate", training_shared.get("learning_rate", 1e-3)))
trial_seed = int(env_training.get("trial_seed", 0))
num_workers = int(training_shared.get("num_workers", 0))
log_interval = int(training_shared.get("log_interval", 1000))
save_interval = int(training_shared.get("save_interval", 10000))
MODEL_SAVE_DIR = training_shared.get("model_save_dir", "checkpoints")

dp = resolve_dp_params(env_config, training_shared)


def load_dataset():
    """Mirror combinedv2_cpascounter_training.load_dataset (state-based envs)."""
    if active_env in ("pen", "door", "kitchen"):
        from utils.datasets import D4RLDataset
        return D4RLDataset(env_config["dataset_name"], download=True, frame_stack=frame_stack)
    elif active_env == "particle":
        from utils.datasets import ParticleDataset
        return ParticleDataset(env_config["data_dir"], n_dim=env_config.get("n_dim", 2), frame_stack=frame_stack)
    elif active_env == "pushing":
        from utils.datasets import PushingDataset
        return PushingDataset(data_dir=env_config["data_dir"], frame_stack=frame_stack)
    elif active_env == "pushing_multi":
        from utils.datasets import PushingMultiDataset
        return PushingMultiDataset(data_dir=env_config["data_dir"], frame_stack=frame_stack)
    elif active_env == "libero_goal":
        from utils.datasets import LiberoGoalDataset
        return LiberoGoalDataset(
            goal_embeddings_path=env_config["goal_embeddings_path"],
            frame_stack=frame_stack,
            max_demos_per_task=env_config.get("max_demos_per_task"),
        )
    elif active_env == "dummy":
        from utils.datasets import DummyDataset
        return DummyDataset(
            size=10000,
            step_size=env_config.get("step_size", 0.1),
            goal_radius=env_config.get("goal_radius", 0.05),
            n_dim=env_config.get("n_dim", 2),
            frame_stack=frame_stack,
        )
    raise ValueError(f"Unknown / unsupported environment for DP: {active_env}")


def build_obs_normalizer(dataset, device):
    """Standardize for the IBC-faithful envs (matches combinedv2)."""
    if active_env == "libero_goal":
        return ObservationNormalizer(env_id=env_id, device=device, frame_stack=1,
                                     obs_mean=dataset.obs_mean, obs_std=dataset.obs_std)
    if active_env in ("pushing", "pushing_multi", "pen", "door", "kitchen"):
        if not hasattr(dataset, "obs_mean") or not hasattr(dataset, "obs_std"):
            raise RuntimeError(f"{active_env} dataset must expose obs_mean/obs_std.")
        return ObservationNormalizer(env_id=env_id, device=device, frame_stack=frame_stack,
                                     obs_mean=dataset.obs_mean, obs_std=dataset.obs_std)
    particle_n_dim = env_config.get("n_dim") if active_env == "particle" else None
    return ObservationNormalizer(env_id=env_id, device=device, frame_stack=frame_stack,
                                 particle_n_dim=particle_n_dim)


def main():
    random.seed(trial_seed)
    np.random.seed(trial_seed)
    torch.manual_seed(trial_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(trial_seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"trial_seed={trial_seed} | device={device} | active_env={active_env}")
    print(f"Diffusion Policy (epsilon-pred, Q-estimator trunk). DP params: {json.dumps(dp)}")
    print(f"Training steps: {training_steps} | batch_size: {batch_size} | lr: {learning_rate}")

    print(f"Loading {active_env} dataset...")
    dataset = load_dataset()
    print(f"Dataset size: {len(dataset)} | state_shape={dataset.state_shape} | action_shape={dataset.action_shape}")

    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, persistent_workers=num_workers > 0,
    )
    obs_normalizer = build_obs_normalizer(dataset, device)

    denoiser = build_denoiser(dataset.state_shape, dataset.action_shape, dp, device)
    diffusion = build_diffusion(dp, device, action_bounds)
    n_params = sum(p.numel() for p in denoiser.parameters())
    print(f"Denoiser params: {n_params} (kind={dp['denoiser_network_kind']} "
          f"w={dp['denoiser_width']} d={dp['denoiser_depth']} t_emb={dp['time_emb_dim']})")

    optimizer = torch.optim.AdamW(denoiser.parameters(), lr=learning_rate)
    effective_t_max = env_training.get("cosine_t_max", training_steps)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=effective_t_max, eta_min=1e-6)

    ema_decay = dp["ema_decay"]
    ema_denoiser = copy.deepcopy(denoiser) if ema_decay > 0 else None
    if ema_denoiser is not None:
        for p in ema_denoiser.parameters():
            p.requires_grad_(False)

    os.makedirs(MODEL_SAVE_DIR, exist_ok=True)
    start_time = time.time()
    step = 0
    running_loss = 0.0
    running_n = 0

    denoiser.train()
    while step < training_steps:
        for batch in dataloader:
            if step >= training_steps:
                break
            states = batch["state"].float().to(device)
            states = obs_normalizer.normalize(states)
            actions = batch["action"].float().to(device)  # already in [-1, 1]

            loss = diffusion.training_loss(denoiser, states, actions)

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(denoiser.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()

            if ema_denoiser is not None:
                with torch.no_grad():
                    for ep, p in zip(ema_denoiser.parameters(), denoiser.parameters()):
                        ep.mul_(ema_decay).add_(p, alpha=1.0 - ema_decay)

            running_loss += float(loss.item())
            running_n += 1
            step += 1

            if step % log_interval == 0:
                avg = running_loss / max(running_n, 1)
                elapsed = time.time() - start_time
                # "Loss:" token parsed by hyperparam_search.extract_final_metrics.
                print(f"Step {step}/{training_steps} | Loss: {avg:.6f} | {elapsed:.0f}s")
                running_loss = 0.0
                running_n = 0

            if save_interval and step % save_interval == 0:
                torch.save(denoiser.state_dict(), os.path.join(MODEL_SAVE_DIR, "denoiser.pt"))
                if ema_denoiser is not None:
                    torch.save(ema_denoiser.state_dict(), os.path.join(MODEL_SAVE_DIR, "denoiser_ema.pt"))

    # Final checkpoint.
    torch.save(denoiser.state_dict(), os.path.join(MODEL_SAVE_DIR, "denoiser.pt"))
    if ema_denoiser is not None:
        torch.save(ema_denoiser.state_dict(), os.path.join(MODEL_SAVE_DIR, "denoiser_ema.pt"))

    # norm_stats — SAME schema as combinedv2 so eval's PushingSimulation reuses it.
    norm_stats = {
        "act_min": dataset.act_min,
        "act_max": dataset.act_max,
        "action_norm_range": getattr(dataset, "action_norm_range", (-1.0, 1.0)),
        "state_shape": dataset.state_shape,
    }
    if hasattr(dataset, "obs_mean"):
        norm_stats["obs_mean"] = dataset.obs_mean
        norm_stats["obs_std"] = dataset.obs_std
    if active_env == "libero_goal":
        norm_stats["libero_obs_keys"] = dataset.libero_obs_keys
        norm_stats["libero_obs_dims"] = dataset.libero_obs_dims
        norm_stats["goal_embeddings"] = dataset.goal_embeddings
        norm_stats["goal_task_names"] = dataset.goal_task_names
        norm_stats["goal_emb_dim"] = dataset.goal_emb_dim
    torch.save(norm_stats, os.path.join(MODEL_SAVE_DIR, "norm_stats.pt"))
    print(f"Saved denoiser.pt + norm_stats.pt to {MODEL_SAVE_DIR}")
    print(f"Done in {time.time() - start_time:.0f}s")


if __name__ == "__main__":
    main()
