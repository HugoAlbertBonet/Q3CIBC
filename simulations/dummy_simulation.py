
"""Simulation wrapper for Dummy 2D Grid Navigation environment.

Runs actual episodes, collects diagnostic snapshots at configurable timesteps,
and generates per-snapshot visualization plots.
"""

import os
import numpy as np
import torch
from simulations.dummy_env import DummyEnv
from utils.vis_dummy import plot_dummy_debug
from utils.sampling import sample_langevin, sample_uniform


class DummySimulation:
    def __init__(
        self,
        control_point_generator,
        q_estimator,
        device="cpu",
        render_mode=None,
        save_dir="plots/dummy",
        step_size=0.1,
        goal_radius=0.05,
        max_episode_steps=200,
        frame_stack=1,
        snapshot_steps=None,
        **kwargs
    ):
        self.model = control_point_generator
        self.estimator = q_estimator
        self.device = device
        self.render_mode = render_mode
        self.save_dir = save_dir
        self.step_size = step_size
        self.goal_radius = goal_radius
        self.max_episode_steps = max_episode_steps
        self.frame_stack = frame_stack
        self.snapshot_steps = snapshot_steps or [1, 5, 10, 20, 50]
        self.langevin_config = kwargs.get("langevin_config", {})

        self.env = DummyEnv(
            step_size=step_size,
            goal_radius=goal_radius,
            max_steps=max_episode_steps,
            render_mode=render_mode,
        )

        self.all_results = []

    def _build_state_with_history(self, obs, history_buffer):
        """Build the full state including frame-stacked history.

        For frame_stack=1: state = obs (goal_x, goal_y, pos_x, pos_y)
        For frame_stack>1: state = concat(obs, prev_obs_1, ..., prev_obs_(n-1))
        """
        if self.frame_stack <= 1:
            return obs.copy()

        # Stack: current obs + (frame_stack - 1) previous observations
        frames = [obs]
        for i in range(self.frame_stack - 1):
            if i < len(history_buffer):
                frames.append(history_buffer[-(i + 1)])
            else:
                # Pad with first observation (or current if no history)
                frames.append(history_buffer[0] if history_buffer else obs)
        return np.concatenate(frames).astype(np.float32)

    def _make_langevin_fn(self):
        """Create a Langevin sampling function for visualization."""
        def langevin_fn(model, estimator, device, state):
            n_samples = 64
            mins = [-1.0]
            maxs = [1.0]

            initial_guess = torch.from_numpy(
                sample_uniform(n_samples, 1, mins, maxs)
            ).float().to(device)  # (1, 64, 1)

            def energy(o, a):
                return -estimator(o, a).squeeze(-1)

            samples, trajs = sample_langevin(
                energy_function=energy,
                observations=state,
                initial_actions=initial_guess,
                num_samples=n_samples,
                action_min=torch.tensor(mins).to(device),
                action_max=torch.tensor(maxs).to(device),
                num_iterations=self.langevin_config.get("num_iterations", 20),
                lr_init=self.langevin_config.get("lr_init", 0.1),
                lr_final=self.langevin_config.get("lr_final", 0.01),
                polynomial_decay_power=2.0,
                return_trajectories=True,
                device=device
            )

            # samples: (1, 64, 1) -> (64, 1)
            samples = samples.squeeze(0)
            # trajs: list of (1, 64, 1) -> (64, Steps, 1)
            trajs_tensor = torch.stack(trajs).squeeze(1).permute(1, 0, 2)

            return samples.cpu().numpy(), trajs_tensor.cpu().numpy()

        return langevin_fn

    def run_simulation(self, num_episodes=3, seed=None):
        """Run episodes and generate diagnostic snapshots."""
        print(f"Running Dummy 2D Navigation simulation ({num_episodes} episodes)...")
        os.makedirs(self.save_dir, exist_ok=True)

        langevin_fn = self._make_langevin_fn()
        self.all_results = []

        for ep in range(num_episodes):
            ep_seed = (seed or 0) * 1000 + ep
            obs, info = self.env.reset(seed=ep_seed)

            goal = obs[:2].copy()
            agent_pos = obs[2:4].copy()
            trajectory = [agent_pos.copy()]
            history_buffer = [obs.copy()]

            total_reward = 0.0
            done = False
            step = 0

            while not done:
                step += 1

                # Build state with frame stacking
                state_np = self._build_state_with_history(obs, history_buffer)
                state_tensor = torch.from_numpy(state_np).float().unsqueeze(0).to(self.device)

                # Check if we should save a snapshot at this step
                if step in self.snapshot_steps:
                    save_path = os.path.join(
                        self.save_dir,
                        f"ep{ep}_step{step:03d}_seed{seed}.png"
                    )
                    plot_dummy_debug(
                        model=self.model,
                        estimator=self.estimator,
                        device=self.device,
                        save_path=save_path,
                        state=state_tensor,
                        trajectory=trajectory,
                        goal=goal,
                        agent_pos=agent_pos,
                        step_idx=step,
                        episode_idx=ep,
                        langevin_fn=langevin_fn,
                        title=f"Dummy Nav (Seed {seed})",
                    )

                # Select action using the model
                with torch.no_grad():
                    cps = self.model(state_tensor)  # (1, N, 1)
                    # Expand state for Q evaluation
                    state_exp = state_tensor.unsqueeze(1).expand(-1, cps.shape[1], -1)
                    q_vals = self.estimator(state_exp, cps).squeeze(-1)  # (1, N)
                    # Pick action with highest Q
                    best_idx = q_vals.argmax(dim=1)
                    action = cps[0, best_idx[0]].cpu().numpy()  # (1,)

                obs, reward, terminated, truncated, info = self.env.step(action)
                total_reward += reward

                agent_pos = obs[2:4].copy()
                trajectory.append(agent_pos.copy())
                history_buffer.append(obs.copy())

                done = terminated or truncated

            success = terminated  # reached goal (not just truncated)
            self.all_results.append({
                "total_reward": total_reward,
                "episode_length": step,
                "success": success,
            })
            print(f"  Episode {ep}: steps={step}, reward={total_reward:.2f}, success={success}")

        return self.all_results

    def close(self):
        self.env.close()

    def get_summary(self):
        if not self.all_results:
            return {
                "num_episodes": 0,
                "reward_mean": 0.0,
                "reward_std": 0.0,
                "success_rate": 0.0,
            }
        rewards = [r["total_reward"] for r in self.all_results]
        successes = [r["success"] for r in self.all_results]
        return {
            "num_episodes": len(self.all_results),
            "reward_mean": np.mean(rewards),
            "reward_std": np.std(rewards),
            "success_rate": np.mean(successes),
        }
