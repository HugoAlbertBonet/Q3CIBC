"""Agent-assisted hyperparameter search for IBC-DFO training.

Trains IBC-DFO models with different hyperparameter configurations, evaluates
success rates via Langevin MCMC inference, and supports iterative refinement.

Modes:
    --run               Run a single trial (with --params or defaults)
    --analyze           Print summary table of all past trials

Usage:
    python hyperparam_search_dfo.py --run
    python hyperparam_search_dfo.py --run --params '{"SOFTMAX_TEMPERATURE": 0.5}'
    python hyperparam_search_dfo.py --run --quick   # 10k steps, 50 Langevin iters
    python hyperparam_search_dfo.py --analyze
"""

from __future__ import annotations

import argparse
import json
import os
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

from simulations.particle_env import ParticleEnv
from utils.models import QEstimator
from utils.normalizations import ObservationNormalizer
from utils.sampling import sample_langevin

CONFIG_PATH = ROOT_DIR / "config_json" / "config.json"
RESULTS_DIR = ROOT_DIR / "results" / "hyperparam_search" / "ibc_dfo_particle"
TRIALS_PATH = RESULTS_DIR / "trials.jsonl"
CHECKPOINTS_BASE = ROOT_DIR / "checkpoints" / "hpsearch_dfo"

BASELINE_HPARAMS = {
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
}

NUM_EVAL_SEEDS = 50
INFERENCE_LANGEVIN = {
    "num_samples": 512,
    "num_iterations": 100,
    "lr_init": 0.1,
    "lr_final": 1e-5,
    "polynomial_decay_power": 2.0,
    "delta_action_clip": 0.1,
    "noise_scale": 0.1,
}


def load_config():
    with open(CONFIG_PATH) as f:
        return json.load(f)


def next_trial_id() -> int:
    if not TRIALS_PATH.exists():
        return 1
    count = 0
    with open(TRIALS_PATH) as f:
        for _ in f:
            count += 1
    return count + 1


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


def langevin_counter_examples(energy_model, obs_norm, device, hparams, action_dim):
    B = obs_norm.shape[0]
    act_min = 0.0 - hparams["UNIFORM_BOUNDARY_BUFFER"]
    act_max = 1.0 + hparams["UNIFORM_BOUNDARY_BUFFER"]
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


def train_dfo(hparams: dict, trial_id: int) -> dict:
    """Train a DFO model and return metadata dict."""
    from utils.datasets import ParticleDataset

    cfg = load_config()
    env_cfg = cfg["environments"]["particle"]
    n_dim = env_cfg.get("n_dim", 2)
    frame_stack = env_cfg.get("frame_stack", 2)
    action_dim = env_cfg["action_dim"]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"  Using device: {device}")
    print(f"  n_dim={n_dim}, action_dim={action_dim}, frame_stack={frame_stack}")
    print(f"  Steps={hparams['TRAINING_STEPS']}, LR={hparams['LEARNING_RATE']}, "
          f"Temp={hparams['SOFTMAX_TEMPERATURE']}")
    print(f"  Counter-examples={hparams['NUM_COUNTER_EXAMPLES']}, "
          f"Langevin iters={hparams['LANGEVIN_TRAIN_ITERATIONS']}")
    print(f"  Model: {hparams['HIDDEN_DIMS']}, Grad margin={hparams['GRADIENT_MARGIN']}")

    dataset = ParticleDataset(env_cfg["data_dir"], n_dim=n_dim, frame_stack=frame_stack)
    print(f"  Dataset size: {len(dataset)}")
    norm_stats = compute_dataset_stats(dataset)

    obs_normalizer = ObservationNormalizer(
        env_id=env_cfg["env_id"], device=device,
        frame_stack=frame_stack, particle_n_dim=n_dim,
    )

    obs_dim = dataset.state_shape
    act_dim = dataset.action_shape
    energy_model = QEstimator(
        state_dim=obs_dim, action_dim=act_dim, hidden_dims=hparams["HIDDEN_DIMS"],
    ).to(device)

    optimizer = torch.optim.Adam(energy_model.parameters(), lr=hparams["LEARNING_RATE"])
    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=hparams["BATCH_SIZE"], shuffle=True, drop_last=True,
    )

    current_lr = hparams["LEARNING_RATE"]
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
            actions_norm = normalize_tensor(
                actions, norm_stats["act_min"], norm_stats["act_max"], device,
            )

            counter_actions = langevin_counter_examples(
                energy_model, states_norm, device, hparams, act_dim,
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
                continue

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

    save_dir = CHECKPOINTS_BASE / f"trial_{trial_id:03d}"
    save_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = save_dir / "q_estimator.pt"
    torch.save({
        "model_state_dict": energy_model.state_dict(),
        "norm_stats": norm_stats,
        "step": hparams["TRAINING_STEPS"],
        "hparams": hparams,
        "trial_id": trial_id,
    }, ckpt_path)
    print(f"  Model saved to {ckpt_path}")

    return {
        "checkpoint_path": str(ckpt_path),
        "duration_seconds": total_time,
        "final_train_loss": last_loss,
        "final_infonce": last_nce,
        "final_grad_penalty": last_gp,
        "final_accuracy": last_acc,
    }


def evaluate_checkpoint(ckpt_path: str, langevin_cfg: dict, num_seeds: int = 50) -> tuple[float, float]:
    """Evaluate a DFO checkpoint. Returns (success_rate, eval_time)."""
    cfg = load_config()
    env_cfg = cfg["environments"]["particle"]
    n_dim = int(env_cfg["n_dim"])
    state_dim = int(env_cfg["state_dim"])
    action_dim = int(env_cfg["action_dim"])
    frame_stack = int(env_cfg.get("frame_stack", 2))
    action_bounds = tuple(env_cfg.get("action_bounds", [0, 1]))
    max_steps = int(cfg.get("simulation", {}).get("max_episode_steps", 50))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    sd = ckpt["model_state_dict"] if isinstance(ckpt, dict) and "model_state_dict" in ckpt else ckpt
    norm_stats = ckpt.get("norm_stats") if isinstance(ckpt, dict) else None

    indices = sorted({int(k.split(".")[1]) for k in sd if k.startswith("network.") and k.endswith(".weight")})
    hidden = [int(sd[f"network.{i}.weight"].shape[0]) for i in indices[:-1]]

    model = QEstimator(state_dim=state_dim * frame_stack, action_dim=action_dim, hidden_dims=hidden)
    model.load_state_dict(sd)
    model.to(device).eval()

    obs_normalizer = ObservationNormalizer(
        env_id=env_cfg["env_id"], device=device,
        frame_stack=frame_stack, particle_n_dim=n_dim,
    )

    buf_sz = 0.05
    norm_min = torch.full((action_dim,), -buf_sz, device=device)
    norm_max = torch.full((action_dim,), 1.0 + buf_sz, device=device)

    def denorm(a):
        if norm_stats is None:
            return a
        rng = np.where((norm_stats["act_max"] - norm_stats["act_min"]) == 0, 1.0,
                       norm_stats["act_max"] - norm_stats["act_min"])
        return a * rng + norm_stats["act_min"]

    t0 = time.time()
    successes = 0
    for seed in range(num_seeds):
        env = ParticleEnv(n_dim=n_dim, n_steps=max_steps, render_mode=None)
        obs, _ = env.reset(seed=seed)
        frame_buf = deque(maxlen=frame_stack)
        for _ in range(frame_stack):
            frame_buf.append(obs.copy())

        done = False
        success = False
        while not done:
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

            obs, _, terminated, truncated, info = env.step(action)
            frame_buf.append(obs.copy())
            done = terminated or truncated
            success = bool(info.get("success", False))
        if success:
            successes += 1
        env.close()

    sr = successes / max(num_seeds, 1)
    eval_time = time.time() - t0
    return sr, eval_time


def run_trial(params_override: dict | None = None, quick: bool = False):
    trial_id = next_trial_id()
    hparams = deepcopy(BASELINE_HPARAMS)

    if quick:
        hparams["TRAINING_STEPS"] = 10_000
        hparams["LANGEVIN_TRAIN_ITERATIONS"] = 50

    if params_override:
        hparams.update(params_override)

    print(f"\n{'='*70}")
    print(f"DFO Trial {trial_id}")
    print(f"{'='*70}")
    print(f"Params: {json.dumps({k: v for k, v in hparams.items()}, indent=2)}")

    print(f"\n  Training ({hparams['TRAINING_STEPS']} steps)...")
    train_meta = train_dfo(hparams, trial_id)

    print(f"\n  Evaluating on {NUM_EVAL_SEEDS} seeds...")
    sr, eval_time = evaluate_checkpoint(
        train_meta["checkpoint_path"], INFERENCE_LANGEVIN, NUM_EVAL_SEEDS,
    )
    print(f"\n  Result: success_rate={sr*100:.2f}%, eval_time={eval_time:.1f}s")

    trial_record = {
        "trial_id": trial_id,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "hparams": hparams,
        "success_rate": sr,
        "eval_time_s": eval_time,
        "train_duration_s": train_meta["duration_seconds"],
        "final_train_loss": train_meta["final_train_loss"],
        "final_infonce": train_meta["final_infonce"],
        "final_grad_penalty": train_meta["final_grad_penalty"],
        "final_accuracy": train_meta["final_accuracy"],
        "checkpoint_path": train_meta["checkpoint_path"],
        "inference_cfg": INFERENCE_LANGEVIN,
        "quick": quick,
    }

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    with open(TRIALS_PATH, "a") as f:
        f.write(json.dumps(trial_record, default=str) + "\n")
    print(f"  Logged to {TRIALS_PATH}")
    return trial_record


def analyze():
    if not TRIALS_PATH.exists():
        print("No trials found.")
        return

    trials = []
    with open(TRIALS_PATH) as f:
        for line in f:
            if line.strip():
                trials.append(json.loads(line))

    trials.sort(key=lambda t: -t["success_rate"])

    print(f"\n{'='*100}")
    print(f"{'ID':>4} {'SR%':>6} {'Quick':>5} {'Steps':>7} {'Temp':>5} {'CE':>4} "
          f"{'LR':>8} {'Arch':>12} {'Margin':>6} {'TrainT':>7} {'EvalT':>7} {'NCE':>7} {'Acc':>5}")
    print("-" * 100)
    for t in trials:
        h = t["hparams"]
        q = "Y" if t.get("quick") else "N"
        arch = "x".join(str(d) for d in h["HIDDEN_DIMS"])
        print(
            f"{t['trial_id']:>4} {t['success_rate']*100:>5.1f}% {q:>5} "
            f"{h['TRAINING_STEPS']:>7} {h['SOFTMAX_TEMPERATURE']:>5.2f} "
            f"{h['NUM_COUNTER_EXAMPLES']:>4} {h['LEARNING_RATE']:>8.1e} "
            f"{arch:>12} {h['GRADIENT_MARGIN']:>6.1f} "
            f"{t['train_duration_s']:>6.0f}s {t['eval_time_s']:>6.0f}s "
            f"{t.get('final_infonce', 0):>7.4f} {t.get('final_accuracy', 0):>5.3f}"
        )
    print("=" * 100)


def main():
    parser = argparse.ArgumentParser(description="DFO hyperparameter search")
    parser.add_argument("--run", action="store_true", help="Run a single trial")
    parser.add_argument("--analyze", action="store_true", help="Analyze past trials")
    parser.add_argument("--params", type=str, default=None, help="JSON override params")
    parser.add_argument("--quick", action="store_true", help="Quick trial (10k steps, 50 Langevin iters)")
    args = parser.parse_args()

    if args.analyze:
        analyze()
        return

    if args.run:
        params = json.loads(args.params) if args.params else None
        run_trial(params, quick=args.quick)
        return

    parser.print_help()


if __name__ == "__main__":
    main()
