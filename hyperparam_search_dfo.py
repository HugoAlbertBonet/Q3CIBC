"""Agent-assisted hyperparameter search for IBC-DFO training.

Trains IBC-DFO models with different hyperparameter configurations, evaluates
success rates via Langevin MCMC inference, and supports iterative refinement.

Modes:
    --run               Run a single trial (with --params or defaults)
    --analyze           Print summary table of all past trials

Usage:
    python hyperparam_search_dfo.py --run
    python hyperparam_search_dfo.py --run --params '{"SOFTMAX_TEMPERATURE": 0.5}'
    python hyperparam_search_dfo.py --run --params '{"INFERENCE_NUM_ITERATIONS": 50}'
    python hyperparam_search_dfo.py --run --quick
    python hyperparam_search_dfo.py --run --num-reps 3   # 3 reps with trial_seed=0,1,2
    python hyperparam_search_dfo.py --analyze

Mirrors `hyperparam_search.py` for the Q3C-IBC architecture: per-seed eval
details, aggregate distance metrics, deterministic seeding, NaN recovery,
concurrency-safe trial-id assignment via flock, and per-n_dim results path.
"""

from __future__ import annotations

import argparse
import fcntl
import json
import os
import random
import secrets
import sys
import time
from collections import deque
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch

ROOT_DIR = Path(__file__).parent
sys.path.insert(0, str(ROOT_DIR))

from utils.models import QEstimator
from utils.normalizations import ObservationNormalizer
from utils.sampling import sample_langevin

CONFIG_PATH = ROOT_DIR / "config_json" / "config.json"
RESULTS_BASE = ROOT_DIR / "results" / "hyperparam_search"
CHECKPOINTS_BASE = ROOT_DIR / "checkpoints" / "hpsearch_dfo"

# Env-specific results-dir slug. Particle path kept under the legacy
# "ibc_dfo_particle" so previous trials.jsonl files are untouched.
_RESULTS_SLUG = {
    "particle": "ibc_dfo_particle",
    "pen": "ibc_dfo_pen",
}

BASELINE_HPARAMS: dict = {
    # Training
    "TRAINING_STEPS": 100_000,
    "BATCH_SIZE": 512,
    "LEARNING_RATE": 1e-3,
    "LR_DECAY_RATE": 0.99,
    "LR_DECAY_STEPS": 100,
    "NUM_COUNTER_EXAMPLES": 16,
    "LANGEVIN_TRAIN_ITERATIONS": 100,
    "LANGEVIN_STEPSIZE_INIT": 0.1,
    "LANGEVIN_STEPSIZE_FINAL": 1e-5,
    "LANGEVIN_STEPSIZE_POWER": 2.0,
    "LANGEVIN_NOISE_SCALE": 1.0,
    "LANGEVIN_DELTA_ACTION_CLIP": 0.1,
    "GRADIENT_MARGIN": 1.0,
    "SOFTMAX_TEMPERATURE": 1.0,
    "UNIFORM_BOUNDARY_BUFFER": 0.05,
    "HIDDEN_DIMS": [256, 256],
    # Architecture (IBC paper uses spectral_norm=True for D4RL pen-human).
    "Q_USE_SPECTRAL_NORM": False,
    # Stability
    "trial_seed": 0,
    "nan_abort_threshold": 50,
    # Inference Langevin (paper-faithful defaults)
    "INFERENCE_NUM_SAMPLES": 512,
    "INFERENCE_NUM_ITERATIONS": 100,
    "INFERENCE_LR_INIT": 0.1,
    "INFERENCE_LR_FINAL": 1e-5,
    "INFERENCE_DECAY_POWER": 2.0,
    "INFERENCE_DELTA_CLIP": 0.1,
    "INFERENCE_NOISE_SCALE": 0.1,
}

# Particle uses 50 eval seeds (legacy); pen uses 100 (matches IBC Table 2).
_DEFAULT_NUM_EVAL_SEEDS = {"particle": 50, "pen": 100}
NUM_EVAL_SEEDS = 50  # kept for particle backcompat (used when active_env="particle")


# ─── Concurrency helpers (mirror hyperparam_search.py) ────────────────────────

def _new_run_id() -> str:
    """Unique identifier per trial: timestamp + random hex. Safe under concurrency."""
    return datetime.now().strftime("%Y%m%dT%H%M%S") + "_" + secrets.token_hex(4)


def _results_dir(active_env: str = "particle") -> Path:
    """Resolve per-env trials dir. Particle keeps the legacy `<n_dim>` partition."""
    slug = _RESULTS_SLUG.get(active_env, f"ibc_dfo_{active_env}")
    base = RESULTS_BASE / slug
    if active_env == "particle":
        cfg = load_config()
        n_dim = int(cfg["environments"]["particle"].get("n_dim", 2))
        return base / str(n_dim)
    return base


def _trials_path(active_env: str = "particle") -> Path:
    return _results_dir(active_env) / "trials.jsonl"


def append_trial(record: dict, active_env: str = "particle") -> int:
    """Atomically assign monotonically-increasing trial_id and append.

    Uses fcntl.flock for an exclusive lock during the read-max + write section.
    Safe under parallel sbatch submissions. Returns the assigned id.
    """
    path = _trials_path(active_env)
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


# ─── Config + dataset utilities ──────────────────────────────────────────────

def load_config():
    with open(CONFIG_PATH) as f:
        return json.load(f)


def compute_dataset_stats(dataset):
    acts = dataset.actions
    return {
        "act_min": acts.min(axis=0).astype(np.float32),
        "act_max": acts.max(axis=0).astype(np.float32),
    }


def normalize_tensor(x, x_min, x_max, device):
    rng = x_max - x_min
    rng = np.where(rng == 0, np.ones_like(rng), rng)
    return (x - torch.from_numpy(x_min).float().to(device)) / torch.from_numpy(rng).float().to(device)


def _finite(x: float) -> float | None:
    """JSON-safe: inf/nan → None so trials.jsonl stays strictly valid JSON."""
    xf = float(x)
    return xf if np.isfinite(xf) else None


# ─── Training-time Langevin counter-example sampler ──────────────────────────

def langevin_counter_examples(
    energy_model,
    obs_norm,
    device,
    hparams,
    action_dim,
    act_range_lo: float = 0.0,
    act_range_hi: float = 1.0,
):
    """Sample IBC paper-style Langevin counter-examples.

    `act_range_lo` / `act_range_hi` are the action-box bounds in the SAME
    space the model is trained on. Particle uses [0, 1] (dataset actions are
    pre-normalized to [0, 1]), pen uses [-1, 1] (D4RLDataset returns actions
    already in [-1, 1] via per-dim min-max). The UNIFORM_BOUNDARY_BUFFER
    hparam pads both sides identically.
    """
    B = obs_norm.shape[0]
    act_min = act_range_lo - hparams["UNIFORM_BOUNDARY_BUFFER"]
    act_max = act_range_hi + hparams["UNIFORM_BOUNDARY_BUFFER"]
    n_counter = hparams["NUM_COUNTER_EXAMPLES"]
    actions = torch.rand(B, n_counter, action_dim, device=device) * (act_max - act_min) + act_min
    delta_clip = hparams["LANGEVIN_DELTA_ACTION_CLIP"] * 0.5 * (act_max - act_min)
    obs_expanded = obs_norm.unsqueeze(1).expand(-1, n_counter, -1)

    for p in energy_model.parameters():
        p.requires_grad_(False)

    for k in range(hparams["LANGEVIN_TRAIN_ITERATIONS"]):
        frac = 1.0 - k / max(hparams["LANGEVIN_TRAIN_ITERATIONS"] - 1, 1)
        stepsize = (
            hparams["LANGEVIN_STEPSIZE_FINAL"]
            + (hparams["LANGEVIN_STEPSIZE_INIT"] - hparams["LANGEVIN_STEPSIZE_FINAL"])
            * (frac ** hparams["LANGEVIN_STEPSIZE_POWER"])
        )
        actions = actions.detach().requires_grad_(True)
        energies = energy_model(obs_expanded, actions).squeeze(-1)
        grad = torch.autograd.grad(energies.sum(), actions)[0].detach()
        noise = torch.randn_like(actions) * hparams["LANGEVIN_NOISE_SCALE"]
        delta = stepsize * (0.5 * grad + noise)
        delta = torch.clamp(delta, -delta_clip, delta_clip)
        actions = torch.clamp(actions.detach() - delta, act_min, act_max).detach()

    for p in energy_model.parameters():
        p.requires_grad_(True)
    return actions.detach()


# ─── Training ─────────────────────────────────────────────────────────────────

def train_dfo(hparams: dict, run_id: str, active_env: str = "particle") -> dict:
    """Train a DFO model and return metadata dict.

    Env branches:
      - particle: ParticleDataset, minmax obs normalization with JSON bounds,
        actions normalized per-dim to [0, 1] inside the train loop.
      - pen:      D4RLDataset, standardize obs normalization from dataset
        stats (paper-faithful), actions already in [-1, 1] from D4RLDataset.
    """
    # Deterministic seeding — same trial_seed ⇒ same training trajectory.
    seed = int(hparams.get("trial_seed", 0))
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    cfg = load_config()
    env_cfg = cfg["environments"][active_env]
    frame_stack = env_cfg.get("frame_stack", 1)
    action_dim = env_cfg["action_dim"]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"  device={device}, trial_seed={seed}, active_env={active_env}")
    print(f"  action_dim={action_dim}, frame_stack={frame_stack}")
    print(f"  Steps={hparams['TRAINING_STEPS']}, LR={hparams['LEARNING_RATE']}, "
          f"Temp={hparams['SOFTMAX_TEMPERATURE']}")
    print(f"  Counter-examples={hparams['NUM_COUNTER_EXAMPLES']}, "
          f"Langevin iters={hparams['LANGEVIN_TRAIN_ITERATIONS']}")
    print(f"  Model: {hparams['HIDDEN_DIMS']}, SN={hparams.get('Q_USE_SPECTRAL_NORM', False)}, "
          f"Grad margin={hparams['GRADIENT_MARGIN']}")

    if active_env == "particle":
        from utils.datasets import ParticleDataset
        n_dim = env_cfg.get("n_dim", 2)
        dataset = ParticleDataset(
            env_cfg["data_dir"], n_dim=n_dim, frame_stack=frame_stack,
        )
        # Particle keeps the legacy [0, 1] in-model action range; per-batch
        # normalize_tensor maps raw dataset actions into [0, 1].
        action_in_model_range = (0.0, 1.0)
        per_batch_action_norm = True
        norm_stats = compute_dataset_stats(dataset)
        obs_normalizer = ObservationNormalizer(
            env_id=env_cfg["env_id"], device=device,
            frame_stack=frame_stack, particle_n_dim=n_dim,
        )
    elif active_env == "pen":
        from utils.datasets import D4RLDataset
        dataset = D4RLDataset(
            env_cfg["dataset_name"],
            download=True,
            frame_stack=frame_stack,
            normalize_actions=True,
            action_norm_range=(-1.0, 1.0),
        )
        # D4RLDataset returns actions already per-dim min-max normalized to
        # [-1, 1] (IBC paper App. B.3). Skip per-batch action normalize.
        action_in_model_range = (-1.0, 1.0)
        per_batch_action_norm = False
        # Norm stats include obs mean/std AND raw act min/max so eval can
        # both standardize obs and denormalize the model's [-1, 1] action
        # back to the env's per-dim native range.
        norm_stats = {
            "obs_mean": dataset.obs_mean.astype(np.float32),
            "obs_std": dataset.obs_std.astype(np.float32),
            "act_min": dataset.act_min.astype(np.float32),
            "act_max": dataset.act_max.astype(np.float32),
            "action_norm_range": (-1.0, 1.0),
            "frame_stack": frame_stack,
            "env_id": env_cfg["env_id"],
        }
        obs_normalizer = ObservationNormalizer(
            env_id=env_cfg["env_id"], device=device,
            frame_stack=frame_stack,
            obs_mean=dataset.obs_mean,
            obs_std=dataset.obs_std,
        )
    else:
        raise ValueError(f"Unsupported active_env for DFO: {active_env}")

    print(f"  Dataset size: {len(dataset)}")

    obs_dim = dataset.state_shape
    act_dim = dataset.action_shape
    energy_model = QEstimator(
        state_dim=obs_dim,
        action_dim=act_dim,
        hidden_dims=hparams["HIDDEN_DIMS"],
        use_spectral_norm=bool(hparams.get("Q_USE_SPECTRAL_NORM", False)),
    ).to(device)

    optimizer = torch.optim.Adam(energy_model.parameters(), lr=hparams["LEARNING_RATE"])
    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=hparams["BATCH_SIZE"], shuffle=True, drop_last=True,
    )

    current_lr = hparams["LEARNING_RATE"]
    nan_abort_threshold = int(hparams.get("nan_abort_threshold", 50))
    consecutive_nan_batches = 0

    start_time = time.time()
    step = 0
    log_interval = 500

    last_loss = last_nce = last_gp = last_acc = None

    while step < hparams["TRAINING_STEPS"]:
        for batch in dataloader:
            if step >= hparams["TRAINING_STEPS"]:
                break

            states = batch["state"].float().to(device)
            actions = batch["action"].float().to(device)
            B = states.shape[0]

            states_norm = obs_normalizer.normalize(states)
            if per_batch_action_norm:
                actions_norm = normalize_tensor(
                    actions, norm_stats["act_min"], norm_stats["act_max"], device,
                )
            else:
                # D4RLDataset pre-normalized actions to action_in_model_range.
                actions_norm = actions

            counter_actions = langevin_counter_examples(
                energy_model, states_norm, device, hparams, act_dim,
                act_range_lo=action_in_model_range[0],
                act_range_hi=action_in_model_range[1],
            )

            n_counter = hparams["NUM_COUNTER_EXAMPLES"]
            all_actions = torch.cat([counter_actions, actions_norm.unsqueeze(1)], dim=1)
            states_expanded = states_norm.unsqueeze(1).expand(-1, n_counter + 1, -1)

            energies = energy_model(states_expanded, all_actions).squeeze(-1)
            logits = -energies / hparams["SOFTMAX_TEMPERATURE"]
            log_probs = logits - torch.logsumexp(logits, dim=1, keepdim=True)
            loss_infonce = -log_probs[:, -1].mean()

            gp_actions = all_actions.detach().reshape(B * (n_counter + 1), -1).requires_grad_(True)
            gp_states = states_expanded.detach().reshape(B * (n_counter + 1), -1)
            gp_energies = energy_model(gp_states, gp_actions)
            grad_gp = torch.autograd.grad(gp_energies.sum(), gp_actions, create_graph=True)[0]
            grad_norms = grad_gp.abs().max(dim=-1).values
            grad_penalty = torch.clamp(grad_norms - hparams["GRADIENT_MARGIN"], min=0).pow(2).mean()

            loss = loss_infonce + grad_penalty
            if torch.isnan(loss):
                consecutive_nan_batches += 1
                # Wipe Adam moments so a single NaN batch doesn't poison subsequent steps.
                optimizer.state.clear()
                if consecutive_nan_batches >= nan_abort_threshold:
                    raise RuntimeError(
                        f"Training diverged: {consecutive_nan_batches} consecutive NaN batches"
                    )
                continue
            consecutive_nan_batches = 0

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            with torch.no_grad():
                best_idx = logits.argmax(dim=1)
                accuracy = (best_idx == n_counter).float().mean().item()

            last_loss = loss.item()
            last_nce = loss_infonce.item()
            last_gp = grad_penalty.item()
            last_acc = accuracy
            step += 1

            if step % hparams["LR_DECAY_STEPS"] == 0:
                current_lr *= hparams["LR_DECAY_RATE"]
                for pg in optimizer.param_groups:
                    pg["lr"] = current_lr

            if step % log_interval == 0:
                elapsed = time.time() - start_time
                print(
                    f"  Step {step}/{hparams['TRAINING_STEPS']} | "
                    f"Loss: {loss.item():.4f} (NCE: {loss_infonce.item():.4f}, GP: {grad_penalty.item():.4f}) | "
                    f"Acc: {accuracy:.3f} | LR: {current_lr:.2e} | {elapsed:.1f}s",
                    flush=True,
                )

    total_time = time.time() - start_time
    print(f"  Training completed in {total_time:.1f}s ({total_time / 60:.2f} min)")

    save_dir = CHECKPOINTS_BASE / f"run_{run_id}"
    save_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = save_dir / "q_estimator.pt"
    torch.save({
        "model_state_dict": energy_model.state_dict(),
        "norm_stats": norm_stats,
        "step": hparams["TRAINING_STEPS"],
        "hparams": hparams,
        "run_id": run_id,
        "active_env": active_env,
        "action_in_model_range": action_in_model_range,
    }, ckpt_path)
    # Persist the exact hparams next to the checkpoint for traceability.
    with open(save_dir / "hparams.json", "w") as f:
        json.dump(hparams, f, indent=2, default=str)
    print(f"  Model saved to {ckpt_path}")

    return {
        "checkpoint_path": str(ckpt_path),
        "checkpoint_dir": str(save_dir),
        "duration_seconds": total_time,
        "final_train_loss": last_loss,
        "final_infonce": last_nce,
        "final_grad_penalty": last_gp,
        "final_accuracy": last_acc,
    }


# ─── Evaluation ──────────────────────────────────────────────────────────────

def evaluate_checkpoint(
    ckpt_path: str,
    langevin_cfg: dict,
    num_seeds: int | None = None,
    active_env: str | None = None,
) -> dict:
    """Evaluate a DFO checkpoint with paper-faithful Langevin inference.

    `active_env` is taken from the checkpoint metadata when not passed
    explicitly. Particle returns the same shape of dict as
    `hyperparam_search.py:evaluate_q3c`; pen returns reward-focused metrics
    (success_rate, avg/std/median reward, ep length) matching the pen branch
    in hyperparam_search.py.
    """
    cfg = load_config()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    sd = ckpt["model_state_dict"] if isinstance(ckpt, dict) and "model_state_dict" in ckpt else ckpt
    norm_stats = ckpt.get("norm_stats") if isinstance(ckpt, dict) else None
    if active_env is None:
        active_env = ckpt.get("active_env", "particle") if isinstance(ckpt, dict) else "particle"
    action_in_model_range = (
        tuple(ckpt.get("action_in_model_range", (0.0, 1.0)))
        if isinstance(ckpt, dict)
        else (0.0, 1.0)
    )

    env_cfg = cfg["environments"][active_env]
    action_dim = int(env_cfg["action_dim"])
    frame_stack = int(env_cfg.get("frame_stack", 1))
    action_bounds = tuple(env_cfg.get("action_bounds", [0, 1]))

    # Layer indices come from `.bias` keys (always present, unchanged under
    # spectral_norm — which renames `weight` → `weight_orig`). Using `.weight`
    # alone misses every layer when SN is on. Hidden dims are output dims of
    # each Linear EXCEPT the final one (which is the model output, dim=1 for
    # the energy estimator).
    bias_indices = sorted(
        {int(k.split(".")[1]) for k in sd if k.startswith("network.") and k.endswith(".bias")}
    )
    if not bias_indices:
        raise RuntimeError(
            f"Cannot infer architecture from checkpoint keys: {list(sd.keys())[:5]}..."
        )
    hidden = [int(sd[f"network.{i}.bias"].shape[0]) for i in bias_indices[:-1]]

    # Detect spectral norm under either API:
    #   - old (torch.nn.utils.spectral_norm): keys `weight_orig`, `weight_u`, `weight_v`
    #   - new (torch.nn.utils.parametrizations.spectral_norm): `parametrizations.weight.*`
    use_sn = any(
        ("weight_orig" in k) or ("parametrizations.weight" in k)
        for k in sd.keys()
    )

    if active_env == "particle":
        state_dim = int(env_cfg["state_dim"])
        max_steps = int(cfg.get("simulation", {}).get("max_episode_steps", 50))
        n_dim = int(env_cfg["n_dim"])
        model = QEstimator(
            state_dim=state_dim * frame_stack, action_dim=action_dim,
            hidden_dims=hidden, use_spectral_norm=use_sn,
        )
        model.load_state_dict(sd)
        model.to(device).eval()
        obs_normalizer = ObservationNormalizer(
            env_id=env_cfg["env_id"], device=device,
            frame_stack=frame_stack, particle_n_dim=n_dim,
        )
    elif active_env == "pen":
        state_dim = int(env_cfg["state_dim"])
        max_steps = int(env_cfg.get("max_episode_steps", 100))
        if norm_stats is None or "obs_mean" not in norm_stats:
            raise RuntimeError(
                "Pen DFO eval requires norm_stats with obs_mean/obs_std. "
                "Train with a fresh checkpoint."
            )
        model = QEstimator(
            state_dim=state_dim * frame_stack, action_dim=action_dim,
            hidden_dims=hidden, use_spectral_norm=use_sn,
        )
        model.load_state_dict(sd)
        model.to(device).eval()
        obs_normalizer = ObservationNormalizer(
            env_id=env_cfg["env_id"], device=device,
            frame_stack=frame_stack,
            obs_mean=np.asarray(norm_stats["obs_mean"], dtype=np.float32),
            obs_std=np.asarray(norm_stats["obs_std"], dtype=np.float32),
        )
    else:
        raise ValueError(f"Unsupported active_env for DFO eval: {active_env}")

    if num_seeds is None:
        num_seeds = int(env_cfg.get("num_eval_seeds", _DEFAULT_NUM_EVAL_SEEDS.get(active_env, 50)))

    buf_sz = 0.05
    lo, hi = float(action_in_model_range[0]), float(action_in_model_range[1])
    norm_min = torch.full((action_dim,), lo - buf_sz, device=device)
    norm_max = torch.full((action_dim,), hi + buf_sz, device=device)

    def denorm(a):
        """Map model-space action in `action_in_model_range` back to env-native.

        Particle: model and env share [0, 1]; norm_stats is a per-dim min-max
        rescale (legacy). Pen: model emits in [-1, 1] and per-dim min-max
        maps to env-native (env actually wraps to [-1, 1] but per-dim min/max
        is tighter — paper's per-dim min-max protocol).
        """
        if norm_stats is None:
            return a
        rng = np.where(
            (norm_stats["act_max"] - norm_stats["act_min"]) == 0, 1.0,
            norm_stats["act_max"] - norm_stats["act_min"],
        )
        if active_env == "pen":
            scale = rng / (hi - lo)
            return norm_stats["act_min"] + (a - lo) * scale
        return a * rng + norm_stats["act_min"]

    seeds = list(range(num_seeds))
    successes: list[bool] = []
    rewards: list[float] = []
    dists_first: list[float] = []
    dists_second: list[float] = []
    ep_lengths: list[int] = []
    terminated_flags: list[bool] = []

    # Lazy env factory keeps the gymnasium-robotics import out of particle runs.
    def _make_env():
        if active_env == "particle":
            from simulations.particle_env import ParticleEnv
            return ParticleEnv(n_dim=n_dim, n_steps=max_steps, render_mode=None)
        # pen
        import gymnasium as gym
        import gymnasium_robotics
        gym.register_envs(gymnasium_robotics)
        return gym.make(
            env_cfg["env_id"], reward_type="dense",
            max_episode_steps=max_steps, render_mode=None,
        )

    t0 = time.time()
    for seed in seeds:
        env = _make_env()
        obs, _ = env.reset(seed=seed)
        frame_buf = deque(maxlen=frame_stack)
        for _ in range(frame_stack):
            frame_buf.append(obs.copy())

        total_reward = 0.0
        ep_len = 0
        terminated = False
        truncated = False
        info: dict = {}
        any_step_success = False
        while not (terminated or truncated):
            stacked = np.concatenate(list(frame_buf)) if frame_stack > 1 else frame_buf[-1]
            st = torch.from_numpy(stacked).float().unsqueeze(0).to(device)
            st_n = obs_normalizer.normalize(st)

            samples = sample_langevin(
                energy_function=model, observations=st_n,
                num_samples=int(langevin_cfg["num_samples"]),
                action_min=norm_min, action_max=norm_max,
                num_iterations=int(langevin_cfg["num_iterations"]),
                lr_init=float(langevin_cfg["lr_init"]),
                lr_final=float(langevin_cfg["lr_final"]),
                polynomial_decay_power=float(langevin_cfg.get("polynomial_decay_power", 2.0)),
                delta_action_clip=float(langevin_cfg.get("delta_action_clip", 0.1)),
                noise_scale=float(langevin_cfg["noise_scale"]),
                device=device,
            )
            with torch.no_grad():
                se = st_n.unsqueeze(1).expand(-1, samples.shape[1], -1)
                e = model(se, samples).squeeze(-1)
                best_a = samples[0, e.argmin(dim=-1)[0]].cpu().numpy()
            action = np.clip(denorm(best_a), action_bounds[0], action_bounds[1])

            obs, reward, terminated, truncated, info = env.step(action)
            frame_buf.append(obs.copy())
            total_reward += float(reward)
            ep_len += 1
            # AdroitHandPen emits info["success"] each step while goal pose
            # tolerance holds — track sticky any-step success for pen.
            if active_env == "pen" and bool(info.get("success", info.get("is_success", False))):
                any_step_success = True

        if active_env == "pen":
            successes.append(any_step_success)
        else:
            successes.append(bool(info.get("success", False)))
        rewards.append(total_reward)
        dists_first.append(float(info.get("min_dist_to_first_goal", np.inf)))
        dists_second.append(float(info.get("min_dist_to_second_goal", np.inf)))
        ep_lengths.append(ep_len)
        terminated_flags.append(bool(terminated))
        env.close()

    eval_time = time.time() - t0

    # Env-branched metrics: pen has no first/second-goal distance; particle
    # records them. Both record reward stats (paper-comparable for pen).
    if active_env == "pen":
        return {
            "success_rate": float(np.mean(successes)),
            "success_rate_std": float(np.std(successes)),
            "avg_reward": float(np.mean(rewards)),
            "std_reward": float(np.std(rewards)),
            "median_reward": float(np.median(rewards)),
            "avg_episode_length": float(np.mean(ep_lengths)),
            "num_seeds": len(seeds),
            "eval_time_s": eval_time,
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
        "eval_time_s": eval_time,
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


# ─── Trial runner ─────────────────────────────────────────────────────────────

def run_trial(
    params_override: dict | None = None,
    quick: bool = False,
    reeval_checkpoint: str | None = None,
    active_env: str = "particle",
):
    """If `reeval_checkpoint` is given, skip training and just evaluate that
    checkpoint with the inference params from `params_override`. Useful for
    sweeping inference Langevin without paying for retraining (with deterministic
    seeds, retrained models are bit-exact identical to existing checkpoints)."""
    run_id = _new_run_id()
    hparams = deepcopy(BASELINE_HPARAMS)

    if quick:
        hparams["TRAINING_STEPS"] = 10_000
        hparams["LANGEVIN_TRAIN_ITERATIONS"] = 50

    if params_override:
        hparams.update(params_override)

    print(f"\n{'='*70}")
    print(f"DFO RUN {run_id}")
    print(f"{'='*70}")
    if reeval_checkpoint:
        print(f"REEVAL ONLY — checkpoint: {reeval_checkpoint}")
        print(f"Inference Langevin params:\n"
              f"  num_samples={hparams['INFERENCE_NUM_SAMPLES']}, "
              f"num_iterations={hparams['INFERENCE_NUM_ITERATIONS']}, "
              f"lr_init={hparams['INFERENCE_LR_INIT']}, "
              f"noise_scale={hparams['INFERENCE_NOISE_SCALE']}")
    else:
        print(f"Params:\n{json.dumps(hparams, indent=2, default=str)}")

    timestamp = datetime.now(timezone.utc).isoformat()

    train_meta: dict
    if reeval_checkpoint:
        ckpt = Path(reeval_checkpoint)
        if not ckpt.exists():
            # Allow passing a run_id; resolve to its q_estimator.pt
            candidate = CHECKPOINTS_BASE / reeval_checkpoint / "q_estimator.pt"
            if candidate.exists():
                ckpt = candidate
            else:
                raise FileNotFoundError(
                    f"Checkpoint not found: tried {reeval_checkpoint} and {candidate}"
                )
        train_meta = {
            "checkpoint_path": str(ckpt),
            "checkpoint_dir": str(ckpt.parent),
            "duration_seconds": 0.0,
            "final_train_loss": None,
            "final_infonce": None,
            "final_grad_penalty": None,
            "final_accuracy": None,
        }
        training_failed = False
        train_error = None
    else:
        print(f"\n  Training ({hparams['TRAINING_STEPS']} steps) on active_env={active_env}...")
        try:
            train_meta = train_dfo(hparams, run_id, active_env=active_env)
            training_failed = False
            train_error = None
        except RuntimeError as exc:
            # Likely the NaN-abort path; record the failure and skip eval.
            print(f"\n  Training FAILED: {exc}")
            train_meta = {
                "checkpoint_path": None,
                "checkpoint_dir": str(CHECKPOINTS_BASE / f"run_{run_id}"),
                "duration_seconds": 0.0,
                "final_train_loss": None,
                "final_infonce": None,
                "final_grad_penalty": None,
                "final_accuracy": None,
            }
            training_failed = True
            train_error = str(exc)

    inference_cfg = {
        "num_samples": int(hparams["INFERENCE_NUM_SAMPLES"]),
        "num_iterations": int(hparams["INFERENCE_NUM_ITERATIONS"]),
        "lr_init": float(hparams["INFERENCE_LR_INIT"]),
        "lr_final": float(hparams["INFERENCE_LR_FINAL"]),
        "polynomial_decay_power": float(hparams["INFERENCE_DECAY_POWER"]),
        "delta_action_clip": float(hparams["INFERENCE_DELTA_CLIP"]),
        "noise_scale": float(hparams["INFERENCE_NOISE_SCALE"]),
    }

    eval_results: dict = {}
    eval_error = None
    if not training_failed and train_meta["checkpoint_path"] is not None:
        # Env-driven eval seed count: pen=100 (paper), particle=50 (legacy).
        cfg_for_eval = load_config()
        eval_env_cfg = cfg_for_eval["environments"].get(active_env, {})
        num_eval = int(
            eval_env_cfg.get("num_eval_seeds", _DEFAULT_NUM_EVAL_SEEDS.get(active_env, NUM_EVAL_SEEDS))
        )
        print(f"\n  Evaluating on {num_eval} seeds...")
        try:
            eval_results = evaluate_checkpoint(
                train_meta["checkpoint_path"], inference_cfg, num_eval, active_env=active_env,
            )
            if active_env == "pen":
                import math
                ne = int(eval_results.get("num_seeds") or 1)
                sem = float(eval_results["std_reward"]) / math.sqrt(ne) if ne > 0 else 0.0
                print(
                    f"\n  Result: avg_reward={eval_results['avg_reward']:.1f} "
                    f"± {sem:.2f} (SEM, n={ne}) "
                    f"(median={eval_results['median_reward']:.1f}, "
                    f"σ_ep={eval_results['std_reward']:.1f}), "
                    f"success_rate={eval_results['success_rate']*100:.2f}%, "
                    f"eval_time={eval_results['eval_time_s']:.1f}s"
                )
            else:
                print(
                    f"\n  Result: success_rate={eval_results['success_rate']*100:.2f}%, "
                    f"eval_time={eval_results['eval_time_s']:.1f}s, "
                    f"d1_med={eval_results.get('median_min_dist_first_goal')}, "
                    f"d2_med={eval_results.get('median_min_dist_second_goal')}"
                )
        except Exception as exc:
            eval_error = f"Evaluation failed: {exc}"
            print(f"\n  {eval_error}")

    record = {
        "run_id": run_id,
        "timestamp": timestamp,
        "active_env": active_env,
        "hparams": hparams,
        "params": hparams,                          # alias so analysis tools that look for `params` work
        "training_steps": hparams["TRAINING_STEPS"],
        "duration_seconds": train_meta["duration_seconds"],
        "train_duration_s": train_meta["duration_seconds"],   # kept for back-compat with old analyze()
        "success_rate": eval_results.get("success_rate", 0.0),
        "avg_reward": eval_results.get("avg_reward", 0.0),
        "std_reward": eval_results.get("std_reward"),
        "median_reward": eval_results.get("median_reward"),
        "num_seeds": eval_results.get("num_seeds"),
        "avg_min_dist_first_goal": eval_results.get("avg_min_dist_first_goal"),
        "avg_min_dist_second_goal": eval_results.get("avg_min_dist_second_goal"),
        "median_min_dist_first_goal": eval_results.get("median_min_dist_first_goal"),
        "median_min_dist_second_goal": eval_results.get("median_min_dist_second_goal"),
        "avg_episode_length": eval_results.get("avg_episode_length"),
        "eval_time_s": eval_results.get("eval_time_s"),
        "final_train_loss": train_meta["final_train_loss"],
        "final_infonce": train_meta["final_infonce"],
        "final_grad_penalty": train_meta["final_grad_penalty"],
        "final_train_accuracy": train_meta["final_accuracy"],
        "checkpoint_path": train_meta["checkpoint_path"],
        "checkpoint_dir": train_meta["checkpoint_dir"],
        "inference_cfg": inference_cfg,
        "eval_details": eval_results.get("per_seed", []),
        "eval_error": eval_error,
        "training_failed": training_failed,
        "error": train_error,
        "quick": quick,
        "reeval_only": reeval_checkpoint is not None,
        "reeval_checkpoint": reeval_checkpoint,
    }

    trial_id = append_trial(record, active_env=active_env)
    print(f"  Logged as trial #{trial_id} (run_id={run_id}) → {_trials_path(active_env)}")
    return record


# ─── Analysis ─────────────────────────────────────────────────────────────────

def analyze(active_env: str = "particle", min_trial_id: int = 0):
    """Show trials + cross-seed aggregates.

    `min_trial_id`: skip trials with id below this — scope to a recent batch
    when env protocol has changed (e.g. fixed eval bug after trial N).
    """
    path = _trials_path(active_env)
    if not path.exists():
        print(f"No trials found at {path}.")
        return

    trials = []
    with open(path) as f:
        for line in f:
            if line.strip():
                trials.append(json.loads(line))
    if min_trial_id > 0:
        trials = [t for t in trials if int(t.get("trial_id", 0)) >= min_trial_id]

    # Sort by reward for pen (paper-aligned objective), success_rate elsewhere.
    sort_key = (
        (lambda t: -t.get("avg_reward", -1e9)) if active_env == "pen"
        else (lambda t: -t.get("success_rate", 0))
    )
    trials.sort(key=sort_key)

    print(f"\n{'=' * 140}")
    print(f"  Hyperparameter search results: ibc_dfo_{active_env} (path: {path})")
    print(f"{'=' * 140}")
    print(
        f"{'ID':>4} {'SR%':>6} {'Reward±std':>18} {'Q':>2} {'Steps':>7} {'Temp':>5} {'CE':>4} "
        f"{'LR':>8} {'Arch':>10} {'SN':>3} {'Margin':>6} {'TrainT':>7} {'EvalT':>7} "
        f"{'NCE':>7} {'Acc':>5} {'inf_iters':>9} {'inf_smp':>7}"
    )
    print("-" * 140)
    for t in trials:
        h = t.get("hparams") or t.get("params") or {}
        q = "Y" if t.get("quick") else "N"
        arch = "x".join(str(d) for d in h.get("HIDDEN_DIMS", []))
        sr = t.get("success_rate", 0)
        rw = t.get("avg_reward", 0)
        rw_std = t.get("std_reward")
        rw_cell = f"{rw:.1f} ± {rw_std:.1f}" if rw_std is not None else f"{rw:.1f}"
        sn = "Y" if h.get("Q_USE_SPECTRAL_NORM") else "N"
        inf_cfg = t.get("inference_cfg") or {}
        print(
            f"{t.get('trial_id', '?'):>4} {sr * 100:>5.1f}% {rw_cell:>18} {q:>2} "
            f"{h.get('TRAINING_STEPS', 0):>7} {h.get('SOFTMAX_TEMPERATURE', 0):>5.2f} "
            f"{h.get('NUM_COUNTER_EXAMPLES', 0):>4} {h.get('LEARNING_RATE', 0):>8.1e} "
            f"{arch:>10} {sn:>3} {h.get('GRADIENT_MARGIN', 0):>6.1f} "
            f"{t.get('duration_seconds', 0) or 0:>6.0f}s {(t.get('eval_time_s') or 0):>6.0f}s "
            f"{t.get('final_infonce', 0) or 0:>7.4f} {t.get('final_train_accuracy', 0) or 0:>5.3f} "
            f"{inf_cfg.get('num_iterations', '—'):>9} {inf_cfg.get('num_samples', '—'):>7}"
        )
    print("=" * 140)

    valid = [t for t in trials if not t.get("training_failed")]
    if valid:
        if active_env == "pen":
            best = max(valid, key=lambda t: t.get("avg_reward", float("-inf")))
        else:
            best = max(valid, key=lambda t: t.get("success_rate", 0))
        rw_std = best.get("std_reward")
        rw_str = (
            f"{best.get('avg_reward', 0):.1f} ± {rw_std:.1f}"
            if rw_std is not None
            else f"{best.get('avg_reward', 0):.1f}"
        )
        print(f"\nBest trial: #{best.get('trial_id')}  "
              f"success_rate={best.get('success_rate', 0):.2%}  "
              f"avg_reward={rw_str}")
        print(f"  hparams: {json.dumps(best.get('hparams', {}), indent=2, default=str)}")

    print(f"\nTotal trials: {len(trials)} ({len(valid)} completed, "
          f"{len(trials) - len(valid)} failed)\n")

    # ── Cross-seed aggregation ────────────────────────────────────────────
    # See hyperparam_search.py print_analysis for full semantics:
    #   - σ_ep(avg): intrinsic per-episode reward spread per seed (avg).
    #   - cross_std: sample stdev of per-seed means.
    #   - SEM:        cross_std/√n — IBC paper Table 2 ±65 best matches this.
    import math
    from collections import defaultdict

    # Filter out trials with eval errors (e.g. earlier penAibc batch hit a
    # spectral-norm reload bug that left avg_reward=0 — those rows would
    # corrupt the cross-seed mean if mixed with reevals).
    eval_ok = [t for t in valid if not t.get("eval_error")]

    sig_groups: dict[str, list[dict]] = defaultdict(list)
    for t in eval_ok:
        h = dict(t.get("hparams") or t.get("params") or {})
        h.pop("trial_seed", None)
        sig = json.dumps(h, sort_keys=True, default=str)
        sig_groups[sig].append(t)

    multi = [(sig, ts) for sig, ts in sig_groups.items() if len(ts) >= 2]
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
                {(t.get("hparams") or t.get("params") or {}).get("trial_seed") for t in ts}
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

def main():
    parser = argparse.ArgumentParser(description="DFO hyperparameter search")
    parser.add_argument("--run", action="store_true", help="Run a single trial")
    parser.add_argument("--analyze", action="store_true", help="Analyze past trials")
    parser.add_argument(
        "--params", type=str, default=None,
        help='JSON of hparam overrides. Same keys as BASELINE_HPARAMS, plus '
             'INFERENCE_* keys for the eval-time Langevin config.',
    )
    parser.add_argument(
        "--quick", action="store_true",
        help="Quick trial: 10k steps, 50 Langevin training iters",
    )
    parser.add_argument(
        "--num-reps", type=int, default=1,
        help="Number of repetitions of the same config (each gets trial_seed=0,1,..). "
             "Use to measure variance honestly. Default 1.",
    )
    parser.add_argument(
        "--reeval-checkpoint", type=str, default=None,
        help="Path to a previously-saved q_estimator.pt (or a run_id under "
             "checkpoints/hpsearch_dfo/). Skips training and only runs eval "
             "with the inference Langevin params from --params. Cheap way to "
             "sweep inference cost without retraining the same model.",
    )
    parser.add_argument(
        "--active-env", type=str, default="particle",
        choices=list(_RESULTS_SLUG.keys()),
        help="Environment to train + eval on. Default: particle (legacy). "
             "Use 'pen' for IBC-paper-faithful pen-human-v2 runs.",
    )
    parser.add_argument(
        "--min-trial-id", type=int, default=0,
        help="When analyzing, skip trials with id below this value (scope "
             "to recent batch).",
    )
    args = parser.parse_args()

    if args.analyze:
        analyze(active_env=args.active_env, min_trial_id=args.min_trial_id)
        return

    if args.run:
        params = json.loads(args.params) if args.params else {}
        seed_pinned = "trial_seed" in params
        for rep in range(max(1, args.num_reps)):
            rep_params = dict(params)
            if not seed_pinned:
                rep_params["trial_seed"] = rep
            if args.num_reps > 1:
                print(f"\n[rep {rep + 1}/{args.num_reps}] trial_seed={rep_params['trial_seed']}")
            run_trial(
                rep_params,
                quick=args.quick,
                reeval_checkpoint=args.reeval_checkpoint,
                active_env=args.active_env,
            )
        return

    parser.print_help()


if __name__ == "__main__":
    main()
