"""Standalone diagnostic runner for a trained Diffusion Policy checkpoint on
dummy_bimodal — the DP analog of `python -m simulations.run_simulation` for
Q3C/IBC checkpoints.

DP has no control-point cloud or Q function, so it doesn't fit
`plot_dummy_debug` at all. Instead, at each snapshot step this draws
`--n-samples` INDEPENDENT stochastic rollouts of the denoiser at the current
state (each a fresh noise->action denoising chain) and plots their headings
via `plot_dummy_dp_debug` — the direct visual test of whether the diffusion
sampler covers both detour headings near the obstacle, the same question
Q3C's CP-cloud plot answers for Q3C/IBC.

Usage:
    python -m simulations.run_dummy_bimodal_dp_diagnostic \\
        --checkpoint checkpoints/hpsearch/<run>/denoiser_ema.pt \\
        --seeds 0 1 2 --episodes 6 --n-samples 64 --output-dir plots/dummy_bimodal_dp
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent.parent))

from simulations.dummy_bimodal_env import DummyBimodalEnv
from utils.diffusion import build_denoiser, build_diffusion, resolve_dp_params
from utils.normalizations import ObservationNormalizer
from utils.vis_dummy import plot_dummy_dp_debug

CONFIG_PATH = Path(
    os.environ.get("Q3C_CONFIG_PATH")
    or (Path(__file__).parent.parent / "config_json" / "config.json")
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=str, required=True,
                         help="Path to denoiser.pt or denoiser_ema.pt")
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument("--episodes", type=int, default=6,
                         help="Episodes per seed")
    parser.add_argument("--n-samples", type=int, default=64,
                         help="Independent denoising chains drawn per snapshot")
    parser.add_argument("--sampler", type=str, default="ddpm", choices=["ddpm", "ddim"])
    parser.add_argument("--ddim-steps", type=int, default=10)
    parser.add_argument("--snapshot-steps", type=int, nargs="+",
                         default=[1, 5, 10, 20, 50, 100])
    parser.add_argument("--output-dir", type=str, default="plots/dummy_bimodal_dp")
    parser.add_argument("--device", type=str, default="cpu", choices=["cpu", "cuda"])
    args = parser.parse_args()

    with open(CONFIG_PATH) as f:
        config = json.load(f)
    env_config = config["environments"]["dummy_bimodal"]
    training_shared = config.get("training_shared", {})

    device = args.device
    dp = resolve_dp_params(env_config, training_shared)
    print(f"DP params: {json.dumps(dp)}")

    action_dim = 1
    frame_stack = int(env_config.get("frame_stack", 1))
    state_dim = int(env_config["state_dim"]) * frame_stack
    denoiser = build_denoiser(state_dim, action_dim, dp, device)
    denoiser.load_state_dict(torch.load(args.checkpoint, map_location=device, weights_only=True))
    denoiser.eval()
    diffusion = build_diffusion(dp, device, action_bounds=(-1.0, 1.0))

    obs_normalizer = ObservationNormalizer(
        env_id="DummyBimodal-v0", device=device, frame_stack=frame_stack,
    )

    def sample_actions(state_tensor: torch.Tensor, n: int) -> np.ndarray:
        """n independent denoising chains at the SAME state -> (n,) headings."""
        state_batch = state_tensor.repeat(n, 1)
        with torch.no_grad():
            if args.sampler == "ddpm":
                a = diffusion.ddpm_sample(denoiser, state_batch, action_dim)
            else:
                a = diffusion.ddim_sample(denoiser, state_batch, action_dim,
                                           num_steps=args.ddim_steps, eta=0.0)
        return a.squeeze(-1).cpu().numpy()

    def select_action(state_tensor: torch.Tensor) -> np.ndarray:
        return sample_actions(state_tensor, 1)  # one rollout action

    obstacle_radius = float(env_config.get("obstacle_radius", 0.25))
    obstacle_center = np.zeros(2, dtype=np.float32)
    os.makedirs(args.output_dir, exist_ok=True)

    all_results = []
    for seed in args.seeds:
        env = DummyBimodalEnv(
            step_size=env_config.get("step_size", 0.1),
            goal_radius=env_config.get("goal_radius", 0.1),
            obstacle_radius=obstacle_radius,
        )
        for ep in range(args.episodes):
            ep_seed = seed * 1000 + ep
            obs, _ = env.reset(seed=ep_seed)
            goal = obs[:2].copy()
            agent_pos = obs[2:4].copy()
            trajectory = [agent_pos.copy()]
            history_buffer = [obs.copy()]
            total_reward = 0.0
            step = 0
            done = False

            while not done:
                step += 1
                if frame_stack <= 1:
                    state_np = obs.copy()
                else:
                    frames = [obs] + [
                        history_buffer[-(i + 1)] if i < len(history_buffer) else
                        (history_buffer[0] if history_buffer else obs)
                        for i in range(frame_stack - 1)
                    ]
                    state_np = np.concatenate(frames).astype(np.float32)
                state_tensor = torch.from_numpy(state_np).float().unsqueeze(0).to(device)
                state_tensor = obs_normalizer.normalize(state_tensor)

                if step in args.snapshot_steps:
                    save_path = os.path.join(
                        args.output_dir, f"ep{ep}_step{step:03d}_seed{seed}.png"
                    )
                    plot_dummy_dp_debug(
                        sample_fn=lambda n, st=state_tensor: sample_actions(st, n),
                        save_path=save_path,
                        state=state_tensor,
                        trajectory=trajectory,
                        goal=goal,
                        agent_pos=agent_pos,
                        step_idx=step,
                        episode_idx=ep,
                        n_samples=args.n_samples,
                        title=f"Dummy Bimodal Nav (DP, seed {seed})",
                        obstacle_center=obstacle_center,
                        obstacle_radius=obstacle_radius,
                    )

                action = select_action(state_tensor)
                obs, reward, terminated, truncated, info = env.step(action)
                total_reward += reward
                agent_pos = obs[2:4].copy()
                trajectory.append(agent_pos.copy())
                history_buffer.append(obs.copy())
                done = terminated or truncated

            success = terminated
            all_results.append({"seed": seed, "episode": ep, "steps": step,
                                 "reward": total_reward, "success": success})
            print(f"  seed={seed} ep={ep}: steps={step} reward={total_reward:.2f} success={success}")
        env.close()

    successes = [r["success"] for r in all_results]
    print(f"\nSuccess rate over {len(all_results)} episodes: {np.mean(successes):.2%}")
    print(f"Plots saved to {args.output_dir}/")


if __name__ == "__main__":
    main()
