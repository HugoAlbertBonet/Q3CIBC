"""Analyze Langevin refinement trajectories at inference time.

Loads a trained Q3C checkpoint (generator + estimator + optional norm_stats +
config.json), runs a few particle episodes, and at every env step logs the
Langevin trajectory from the top-1 control point:

  iter | ||a_k - a_0||  ||grad Q||   Q(s,a_k)  d(a_k, goal_1)  d(a_k, goal_2)

This tells us whether Langevin is a useful local refiner (Q rises smoothly,
action drifts a little and the distance to the active goal shrinks) or a random
walk (Q oscillates, drift grows unbounded, distances grow).

Usage:
    uv run python analyze_langevin_trajectory.py <checkpoint_dir> \
        [--iterations 100] [--seeds 5] [--env-steps 10] [--verbose]
        [--lr-init 0.1] [--noise-scale 1.0] [--delta-clip 0.1]

If the Langevin tuning flags are omitted, values come from
<checkpoint_dir>/config.json's env_model.langevin_config.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT_DIR = Path(__file__).parent
sys.path.insert(0, str(ROOT_DIR))

from utils.models import ControlPointGenerator, QEstimator
from utils.sampling import _polynomial_decay
from simulations.particle_simulation import ParticleSimulation


def _langevin_trajectory(
    energy_fn,
    obs_tensor: torch.Tensor,
    start_action: torch.Tensor,
    action_min: torch.Tensor,
    action_max: torch.Tensor,
    num_iterations: int,
    lr_init: float,
    lr_final: float,
    decay_power: float,
    delta_clip: float,
    noise_scale: float,
) -> list[dict]:
    """Run Langevin MCMC while logging stats per iteration.

    Mirrors utils.sampling.sample_langevin but returns per-step diagnostics:
    {iter, action, drift, grad_norm, Q}
    """
    actions = start_action.clone().detach()
    start = start_action.detach().clone()
    obs_expanded = obs_tensor.unsqueeze(1)  # (1, 1, D_obs)

    log: list[dict] = []
    # Step 0: no update yet, just the starting point.
    with torch.no_grad():
        q0 = -energy_fn(obs_expanded, actions).item()
    log.append({
        "iter": 0,
        "action": actions.squeeze().cpu().numpy().copy(),
        "drift": 0.0,
        "grad_norm": 0.0,
        "Q": q0,
    })

    for k in range(num_iterations):
        actions = actions.detach().requires_grad_(True)
        energy = energy_fn(obs_expanded, actions)
        grad = torch.autograd.grad(energy.sum(), actions, create_graph=False)[0]
        grad_norm = float(grad.norm().item())

        lr_k = _polynomial_decay(lr_init, lr_final, decay_power, k, num_iterations)
        noise = torch.randn_like(actions) * noise_scale * (lr_k ** 0.5)
        delta = -(lr_k / 2.0) * grad + noise
        delta = torch.clamp(delta, -delta_clip, delta_clip)
        actions = (actions.detach() + delta).clamp(action_min, action_max)

        with torch.no_grad():
            q_k = -energy_fn(obs_expanded, actions).item()
        drift = float((actions - start).norm().item())

        log.append({
            "iter": k + 1,
            "action": actions.squeeze().cpu().numpy().copy(),
            "drift": drift,
            "grad_norm": grad_norm,
            "Q": q_k,
        })
    return log


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("checkpoint_dir", help="Path to run_<id> dir with .pt files and config.json")
    parser.add_argument("--iterations", type=int, default=100)
    parser.add_argument("--seeds", type=int, default=5)
    parser.add_argument("--env-steps", type=int, default=10,
                        help="How many env steps per seed to probe.")
    parser.add_argument("--verbose", action="store_true",
                        help="Print full per-iter table for a few env steps (costly).")
    parser.add_argument("--lr-init", type=float, default=None)
    parser.add_argument("--lr-final", type=float, default=None)
    parser.add_argument("--decay-power", type=float, default=None)
    parser.add_argument("--delta-clip", type=float, default=None)
    parser.add_argument("--noise-scale", type=float, default=None)
    args = parser.parse_args()

    ckpt_dir = Path(args.checkpoint_dir)
    config_path = ckpt_dir / "config.json"
    if not config_path.exists():
        config_path = ROOT_DIR / "config_json" / "config.json"
        print(f"No config.json in {ckpt_dir}; falling back to {config_path}")
    with open(config_path) as f:
        config = json.load(f)

    active_env = config.get("active_env", "particle")
    if active_env != "particle":
        raise SystemExit(f"This tool only supports active_env=particle (got {active_env}).")

    env_config = config["environments"]["particle"]
    state_dim = env_config["state_dim"]
    action_dim = env_config["action_dim"]
    frame_stack = env_config.get("frame_stack", 1)
    n_dim = env_config.get("n_dim", 2)
    action_bounds = tuple(env_config.get("action_bounds", [0, 1]))
    control_points = env_config["model"]["control_points"]
    num_hidden_layers = env_config["model"]["num_hidden_layers"]
    num_neurons = env_config["model"]["num_neurons"]
    use_spectral_norm = env_config["model"].get("use_spectral_norm", False)
    hidden_dims = [num_neurons] * num_hidden_layers

    lv_cfg = env_config["model"].get("langevin_config", {})
    lr_init = args.lr_init if args.lr_init is not None else float(lv_cfg.get("lr_init", 0.1))
    lr_final = args.lr_final if args.lr_final is not None else float(lv_cfg.get("lr_final", 1e-5))
    decay_power = args.decay_power if args.decay_power is not None else float(lv_cfg.get("polynomial_decay_power", 2.0))
    delta_clip = args.delta_clip if args.delta_clip is not None else float(lv_cfg.get("delta_action_clip", 0.1))
    noise_scale = args.noise_scale if args.noise_scale is not None else float(lv_cfg.get("noise_scale", 1.0))

    device = "cuda" if torch.cuda.is_available() else "cpu"

    cp_gen = ControlPointGenerator(
        input_dim=state_dim * frame_stack,
        output_dim=action_dim,
        control_points=control_points,
        hidden_dims=hidden_dims,
        action_bounds=action_bounds,
    )
    cp_gen.load_state_dict(torch.load(ckpt_dir / "control_point_generator.pt",
                                      map_location=device, weights_only=True))
    cp_gen.to(device).eval()

    q_est = QEstimator(
        state_dim=state_dim * frame_stack,
        action_dim=action_dim,
        hidden_dims=hidden_dims,
        use_spectral_norm=use_spectral_norm,
    )
    q_est.load_state_dict(torch.load(ckpt_dir / "q_estimator.pt",
                                     map_location=device, weights_only=True))
    q_est.to(device).eval()
    for p in q_est.parameters():
        p.requires_grad_(False)

    norm_stats_path = ckpt_dir / "norm_stats.pt"
    act_min_t, act_rng_t = None, None
    if norm_stats_path.exists():
        ns = torch.load(norm_stats_path, weights_only=False)
        act_min = np.asarray(ns["act_min"], dtype=np.float32)
        act_max = np.asarray(ns["act_max"], dtype=np.float32)
        rng = np.where(act_max - act_min == 0, 1.0, act_max - act_min)
        act_min_t = torch.from_numpy(act_min).to(device)
        act_rng_t = torch.from_numpy(rng.astype(np.float32)).to(device)

    def energy_fn(obs_expanded, a):
        if act_min_t is not None:
            a = (a - act_min_t) / act_rng_t
        return -q_est(obs_expanded, a).squeeze(-1)

    action_min_t = torch.full((action_dim,), float(action_bounds[0]), device=device)
    action_max_t = torch.full((action_dim,), float(action_bounds[1]), device=device)

    sim = ParticleSimulation(
        control_point_generator=cp_gen,
        q_estimator=q_est,
        n_dim=n_dim,
        device=device,
        max_episode_steps=config.get("simulation", {}).get("max_episode_steps", 50),
        render_mode=None,
        frame_stack=frame_stack,
        norm_stats={"act_min": act_min, "act_max": act_max} if norm_stats_path.exists() else None,
    )

    print(f"\nLoaded checkpoint: {ckpt_dir}")
    print(f"  control_points={control_points}, layers={num_hidden_layers}x{num_neurons}, "
          f"frame_stack={frame_stack}, n_dim={n_dim}")
    print(f"Langevin: iters={args.iterations}, lr_init={lr_init}, noise_scale={noise_scale}, "
          f"delta_clip={delta_clip}, lr_final={lr_final}, decay_power={decay_power}")
    print(f"Probing {args.seeds} seeds × up to {args.env_steps} env steps each.\n")

    all_trajs: list[list[dict]] = []
    for seed in range(args.seeds):
        if sim.env is None:
            sim.env = sim.create_env()
        obs, _ = sim.env.reset(seed=seed)
        stacked = sim._reset_frame_buffer(obs)
        g1 = np.asarray(sim.env.obs_log[0]["pos_first_goal"])
        g2 = np.asarray(sim.env.obs_log[0]["pos_second_goal"])

        for step in range(args.env_steps):
            obs_tensor = torch.tensor(stacked, dtype=torch.float32).unsqueeze(0).to(device)
            obs_tensor = sim.obs_normalizer.normalize(obs_tensor)
            with torch.no_grad():
                cps = cp_gen(obs_tensor)
                obs_exp_k = obs_tensor.unsqueeze(1).expand(-1, cps.shape[1], -1)
                if act_min_t is not None:
                    cp_q = (cps - act_min_t) / act_rng_t
                else:
                    cp_q = cps
                q_vals = q_est(obs_exp_k, cp_q).squeeze(-1)
                best_idx = q_vals.argmax(dim=1)
                best_cp = cps[0, best_idx[0], :].view(1, 1, -1).clone()

            traj = _langevin_trajectory(
                energy_fn=energy_fn,
                obs_tensor=obs_tensor,
                start_action=best_cp,
                action_min=action_min_t,
                action_max=action_max_t,
                num_iterations=args.iterations,
                lr_init=lr_init,
                lr_final=lr_final,
                decay_power=decay_power,
                delta_clip=delta_clip,
                noise_scale=noise_scale,
            )
            for rec in traj:
                rec["dist_goal1"] = float(np.linalg.norm(rec["action"] - g1))
                rec["dist_goal2"] = float(np.linalg.norm(rec["action"] - g2))
                rec["seed"] = seed
                rec["env_step"] = step
            all_trajs.append(traj)

            if args.verbose and seed == 0 and step < 2:
                print(f"  === seed {seed}, env_step {step} ===")
                print(f"  {'iter':>4} {'drift':>8} {'|grad|':>8} {'Q':>10} {'d(a,g1)':>9} {'d(a,g2)':>9}")
                for rec in traj[::max(1, args.iterations // 10)]:
                    print(f"  {rec['iter']:>4} {rec['drift']:>8.3f} {rec['grad_norm']:>8.3f} "
                          f"{rec['Q']:>10.3f} {rec['dist_goal1']:>9.3f} {rec['dist_goal2']:>9.3f}")

            # Advance env using the pre-Langevin top CP (policy under test) so we stay on the
            # same trajectory the evaluator would follow.
            a_np = np.clip(best_cp.squeeze().cpu().numpy(), action_bounds[0], action_bounds[1])
            obs, _, terminated, truncated, _ = sim.env.step(a_np)
            stacked = sim._update_frame_buffer(obs)
            if terminated or truncated:
                break
    sim.close()

    # ─── Aggregate ───────────────────────────────────────────────────────────
    iters = args.iterations
    drift_by_iter   = np.zeros(iters + 1)
    q_by_iter       = np.zeros(iters + 1)
    d1_by_iter      = np.zeros(iters + 1)
    d2_by_iter      = np.zeros(iters + 1)
    grad_by_iter    = np.zeros(iters + 1)
    count_by_iter   = np.zeros(iters + 1)
    for traj in all_trajs:
        for rec in traj:
            i = rec["iter"]
            drift_by_iter[i]   += rec["drift"]
            q_by_iter[i]       += rec["Q"]
            d1_by_iter[i]      += rec["dist_goal1"]
            d2_by_iter[i]      += rec["dist_goal2"]
            grad_by_iter[i]    += rec["grad_norm"]
            count_by_iter[i]   += 1
    for arr in (drift_by_iter, q_by_iter, d1_by_iter, d2_by_iter, grad_by_iter):
        arr /= np.maximum(count_by_iter, 1)

    print(f"\nAggregated over {int(count_by_iter[0])} (seed, env_step) probes:")
    print(f"{'iter':>5} {'drift':>8} {'|grad|':>9} {'Q':>10} {'d(a,g1)':>9} {'d(a,g2)':>9}")
    show = [0, 1, 2, 5, 10, 25, 50, 75, 100]
    for i in show:
        if i > iters:
            continue
        print(f"{i:>5} {drift_by_iter[i]:>8.3f} {grad_by_iter[i]:>9.4f} "
              f"{q_by_iter[i]:>10.3f} {d1_by_iter[i]:>9.3f} {d2_by_iter[i]:>9.3f}")

    # Quick diagnostic
    q_rise = q_by_iter[iters] - q_by_iter[0]
    drift_end = drift_by_iter[iters]
    print("\n─── Diagnosis ───")
    print(f"  Q change over {iters} iterations: {q_rise:+.3f}  "
          f"(positive → ascent is working)")
    print(f"  Final drift ||a_k - a_0||: {drift_end:.3f}  "
          f"(compare to action-space diameter √{action_dim} ≈ {action_dim**0.5:.2f})")
    print(f"  Did action stay near expected local max? "
          f"{'yes' if drift_end < 0.3 else 'no — action diffused'}")


if __name__ == "__main__":
    main()
