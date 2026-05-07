"""Numerical analysis of inference Langevin behavior — bypass the giant Plot 11.

For each (checkpoint, env_seed), run one episode and report per-step:
  - distance(best_cp_to_nearest_goal): pre-Langevin gap
  - distance(refined_action_to_nearest_goal): post-Langevin gap
  - delta_dist = pre - post  (positive = Langevin moved closer)
  - delta_q = Q_final - Q_initial  (should be > 0 if Langevin is climbing Q)
  - drift = ‖refined - initial‖ (how far did Langevin actually move)

Aggregates by step into a punchy summary so we can answer:
  - Is the CP cloud the bottleneck? (pre-Langevin gap >> threshold)
  - Is Langevin helping? (delta_dist > 0, delta_q > 0)
  - Is Langevin steering toward goal? (delta_dist correlated with drift)
"""
from __future__ import annotations
import json, os, sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))

from utils.models import ControlPointGenerator, QEstimator  # noqa: E402
from utils.sampling import sample_langevin  # noqa: E402
from simulations.particle_simulation import ParticleSimulation  # noqa: E402


def effective_langevin_config(env_cfg):
    base = dict(env_cfg.get("model", {}).get("langevin_config", {}))
    training = env_cfg.get("training", {})
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


def run_one(ckpt_dir: Path, env_seed: int, device: str = "cpu") -> dict:
    cfg = json.loads((ckpt_dir / "config.json").read_text())
    env_cfg = cfg["environments"]["particle"]
    n_dim = env_cfg["n_dim"]
    state_dim = env_cfg["state_dim"]
    action_dim = env_cfg["action_dim"]
    frame_stack = env_cfg.get("frame_stack", 1)
    action_bounds = tuple(env_cfg.get("action_bounds", [0.0, 1.0]))
    cps = env_cfg["model"]["control_points"]
    nh = env_cfg["model"]["num_hidden_layers"]
    nn_dim = env_cfg["model"]["num_neurons"]
    use_sn = env_cfg["model"].get("use_spectral_norm", False)
    inf_lv = int(env_cfg.get("training", {}).get("inference_langevin_iterations", 0))
    lv_cfg = effective_langevin_config(env_cfg)
    max_steps = cfg["simulation"].get("max_episode_steps", 50)
    threshold = float(env_cfg.get("goal_distance", 0.05))

    cp_gen = ControlPointGenerator(
        input_dim=state_dim * frame_stack,
        output_dim=action_dim,
        control_points=cps,
        hidden_dims=[nn_dim] * nh,
        action_bounds=action_bounds,
    )
    cp_gen.load_state_dict(torch.load(ckpt_dir / "control_point_generator.pt", map_location=device, weights_only=True))
    cp_gen.to(device).eval()

    q_est = QEstimator(
        state_dim=state_dim * frame_stack,
        action_dim=action_dim,
        hidden_dims=[nn_dim] * nh,
        use_spectral_norm=use_sn,
    )
    q_est.load_state_dict(torch.load(ckpt_dir / "q_estimator.pt", map_location=device, weights_only=True))
    q_est.to(device).eval()

    sim = ParticleSimulation(
        control_point_generator=cp_gen,
        q_estimator=q_est,
        n_dim=n_dim,
        device=device,
        max_episode_steps=max_steps,
        render_mode=None,
        frame_stack=frame_stack,
    )
    if sim.env is None:
        sim.env = sim.create_env()
    obs, info = sim.env.reset(seed=env_seed)
    stacked_obs = sim._reset_frame_buffer(obs)

    rows = []
    act_min_t = torch.full((action_dim,), float(action_bounds[0]), device=device)
    act_max_t = torch.full((action_dim,), float(action_bounds[1]), device=device)

    for step in range(max_steps):
        # Goals are last 2*n_dim of the current frame within the stacked obs.
        single_frame_dim = 4 * n_dim
        base = stacked_obs.shape[0] - single_frame_dim
        goal1 = stacked_obs[base + 2*n_dim: base + 3*n_dim]
        goal2 = stacked_obs[base + 3*n_dim: base + 4*n_dim]

        obs_tensor = torch.tensor(stacked_obs, dtype=torch.float32).unsqueeze(0).to(device)
        obs_tensor = sim.obs_normalizer.normalize(obs_tensor)

        with torch.no_grad():
            cps_t = cp_gen(obs_tensor)  # (1, N, A)
            obs_exp = obs_tensor.unsqueeze(1).expand(-1, cps_t.shape[1], -1)
            qv = q_est(obs_exp, cps_t).squeeze(-1)  # (1, N)
            best_idx = qv.argmax(dim=1).item()
            best_cp = cps_t[0, best_idx, :].clone()
            q_initial = qv[0, best_idx].item()

        # Closest CP to either goal (pre-Langevin)
        cps_np = cps_t[0].detach().cpu().numpy()  # (N, A)
        dists_to_g1 = np.linalg.norm(cps_np[:, :n_dim] - goal1[None, :], axis=1)
        dists_to_g2 = np.linalg.norm(cps_np[:, :n_dim] - goal2[None, :], axis=1)
        min_cp_to_g1 = float(dists_to_g1.min())
        min_cp_to_g2 = float(dists_to_g2.min())
        # Best CP's distance to nearest goal
        bcp = best_cp.cpu().numpy()
        d_bcp_g1 = float(np.linalg.norm(bcp[:n_dim] - goal1))
        d_bcp_g2 = float(np.linalg.norm(bcp[:n_dim] - goal2))
        d_bcp_min = min(d_bcp_g1, d_bcp_g2)

        # Inference Langevin from best CP
        if inf_lv > 0:
            for p in q_est.parameters():
                p.requires_grad_(False)

            def neg_e(o, a):
                return -q_est(o, a).squeeze(-1)

            init = best_cp.view(1, 1, -1)
            refined, traj = sample_langevin(
                energy_function=neg_e,
                observations=obs_tensor,
                num_samples=1,
                action_min=act_min_t,
                action_max=act_max_t,
                num_iterations=inf_lv,
                lr_init=float(lv_cfg.get("lr_init", 0.1)),
                lr_final=float(lv_cfg.get("lr_final", 1e-5)),
                polynomial_decay_power=float(lv_cfg.get("polynomial_decay_power", 2.0)),
                delta_action_clip=float(lv_cfg.get("delta_action_clip", 0.1)),
                noise_scale=float(lv_cfg.get("noise_scale", 1.0)),
                initial_actions=init,
                return_trajectories=True,
                device=device,
            )
            for p in q_est.parameters():
                p.requires_grad_(True)
            traj_np = torch.stack(traj, dim=0).squeeze(1).squeeze(1).cpu().numpy()  # (T+1, A)
            with torch.no_grad():
                obs_exp_t = obs_tensor.unsqueeze(1).expand(-1, traj_np.shape[0], -1)
                t_t = torch.from_numpy(traj_np).to(device).unsqueeze(0)
                tq = q_est(obs_exp_t, t_t).squeeze(-1).squeeze(0).cpu().numpy()
            ref = refined[0, 0, :].cpu().numpy()
            drift = float(np.linalg.norm(ref - bcp))
            d_ref_g1 = float(np.linalg.norm(ref[:n_dim] - goal1))
            d_ref_g2 = float(np.linalg.norm(ref[:n_dim] - goal2))
            d_ref_min = min(d_ref_g1, d_ref_g2)
            q_final = float(tq[-1])
            q_max_along = float(tq.max())
            q_monotone_inc = bool(np.all(np.diff(tq) >= -1e-6))
            d_along = np.minimum(
                np.linalg.norm(traj_np[:, :n_dim] - goal1[None, :], axis=1),
                np.linalg.norm(traj_np[:, :n_dim] - goal2[None, :], axis=1),
            )
            d_monotone_dec = bool(np.all(np.diff(d_along) <= 1e-6))
            action_used = ref
        else:
            drift = 0.0
            d_ref_min = d_bcp_min
            d_ref_g1 = d_bcp_g1
            d_ref_g2 = d_bcp_g2
            q_final = q_initial
            q_max_along = q_initial
            q_monotone_inc = True
            d_monotone_dec = True
            action_used = bcp

        rows.append({
            "step": step,
            "min_cp_to_g1": min_cp_to_g1,
            "min_cp_to_g2": min_cp_to_g2,
            "best_cp_to_g1": d_bcp_g1,
            "best_cp_to_g2": d_bcp_g2,
            "best_cp_min": d_bcp_min,
            "refined_to_g1": d_ref_g1,
            "refined_to_g2": d_ref_g2,
            "refined_min": d_ref_min,
            "lv_drift": drift,
            "lv_q_initial": q_initial,
            "lv_q_final": q_final,
            "lv_q_max": q_max_along,
            "lv_q_delta": q_final - q_initial,
            "lv_dist_delta": d_bcp_min - d_ref_min,
            "lv_q_monotone_inc": q_monotone_inc,
            "lv_dist_monotone_dec": d_monotone_dec,
        })

        action = np.clip(action_used, action_bounds[0], action_bounds[1])
        obs, reward, terminated, truncated, info = sim.env.step(action)
        stacked_obs = sim._update_frame_buffer(obs)
        if terminated or truncated:
            break

    sim.close()
    return {
        "checkpoint": str(ckpt_dir),
        "env_seed": env_seed,
        "inf_lv": inf_lv,
        "lv_cfg": {k: lv_cfg.get(k) for k in ("lr_init", "lr_final", "noise_scale", "delta_action_clip", "polynomial_decay_power")},
        "threshold": threshold,
        "n_dim": n_dim,
        "control_points": cps,
        "rows": rows,
    }


def summarize(result: dict) -> dict:
    rows = result["rows"]
    if not rows:
        return {}
    arr = lambda key: np.array([r[key] for r in rows], dtype=np.float64)
    return {
        "n_steps": len(rows),
        "min_cp_to_g1_mean": float(arr("min_cp_to_g1").mean()),
        "min_cp_to_g2_mean": float(arr("min_cp_to_g2").mean()),
        "best_cp_min_mean": float(arr("best_cp_min").mean()),
        "refined_min_mean": float(arr("refined_min").mean()),
        "lv_drift_mean": float(arr("lv_drift").mean()),
        "lv_q_delta_mean": float(arr("lv_q_delta").mean()),
        "lv_dist_delta_mean": float(arr("lv_dist_delta").mean()),
        "lv_q_monotone_inc_frac": float(np.mean([r["lv_q_monotone_inc"] for r in rows])),
        "lv_dist_monotone_dec_frac": float(np.mean([r["lv_dist_monotone_dec"] for r in rows])),
    }


if __name__ == "__main__":
    ckpt_root = ROOT / "checkpoints" / "hpsearch"
    runs = [
        ("trial_0_cp20_seed1_lv75",   "run_20260506T124658_be247b43"),
        ("trial_2_cp30_seed1_lv75",   "run_20260506T140717_24e5f31a"),
        ("trial_3_cp30_seed2_lv75",   "run_20260506T144808_616a814f"),
        ("trial_7_cp30_seed1_lv120",  "run_20260506T155710_a0ca04f2"),
    ]

    print(f"{'tag':<28s} {'env_seed':>3s} | {'cp_g1':>6s} {'cp_g2':>6s} {'best':>6s} {'refn':>6s} {'drift':>6s} {'qΔ':>6s} {'dΔ':>7s} {'q↑':>5s} {'d↓':>5s}")
    for tag, run_id in runs:
        for env_seed in (0, 1, 2):
            res = run_one(ckpt_root / run_id, env_seed)
            s = summarize(res)
            print(
                f"{tag:<28s} {env_seed:>3d} | "
                f"{s['min_cp_to_g1_mean']:>6.3f} {s['min_cp_to_g2_mean']:>6.3f} "
                f"{s['best_cp_min_mean']:>6.3f} {s['refined_min_mean']:>6.3f} "
                f"{s['lv_drift_mean']:>6.3f} {s['lv_q_delta_mean']:>+6.3f} "
                f"{s['lv_dist_delta_mean']:>+7.4f} "
                f"{s['lv_q_monotone_inc_frac']:>4.0%} {s['lv_dist_monotone_dec_frac']:>4.0%}"
            )
