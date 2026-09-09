"""Simulation class for the DummyBimodal-v0 environment.

Follows the same BaseSimulation contract as ParticleSimulation / PushingSimulation
so hyperparam_search.py's generic evaluate_q3c(...) dispatch works unchanged:
construct with (control_point_generator, q_estimator, device, max_episode_steps,
render_mode, frame_stack, norm_stats, ...), then call run_episode(seed) per seed.

`success` is not in DummyBimodalEnv's info dict — `terminated` (distance <
goal_radius) already IS the success signal, exactly like plain Dummy-v0.
"""

from __future__ import annotations

import os

import numpy as np
import torch

from .base_simulation import BaseSimulation
from .dummy_bimodal_env import DummyBimodalEnv
from utils.vis_dummy import plot_dummy_debug
from utils.sampling import sample_langevin, sample_uniform
from utils.normalizations import ObservationNormalizer


class DummyBimodalSimulation(BaseSimulation):
    """Simulation for testing trained policies on DummyBimodal-v0."""

    def __init__(
        self,
        control_point_generator: torch.nn.Module,
        q_estimator: torch.nn.Module,
        device: str = "cpu",
        max_episode_steps: int = 200,
        render_mode: str | None = None,
        frame_stack: int = 1,
        norm_stats: dict | None = None,
        step_size: float = 0.1,
        goal_radius: float = 0.1,
        obstacle_radius: float = 0.25,
    ) -> None:
        super().__init__(
            env_id="DummyBimodal-v0",
            control_point_generator=control_point_generator,
            q_estimator=q_estimator,
            device=device,
            max_episode_steps=max_episode_steps,
            frame_stack=frame_stack,
        )
        self.render_mode = render_mode
        self.norm_stats = norm_stats  # unused (dummy_bimodal trains unnormalized)
        self.step_size = step_size
        self.goal_radius = goal_radius
        self.obstacle_radius = obstacle_radius
        # No action renormalization needed (dummy_bimodal trains directly in
        # the env's [-1, 1] action box) — None here is what tells the generic
        # inference-Langevin-refinement wrapper in hyperparam_search.py
        # (LangevinRefinedParticleSimulation, used whenever
        # inference_langevin_iterations > 0) to skip the CP renormalization
        # step. Every other Simulation class sets these for the same reason.
        self._act_min_t = None
        self._act_rng_t = None

    def create_env(self) -> DummyBimodalEnv:
        return DummyBimodalEnv(
            step_size=self.step_size,
            goal_radius=self.goal_radius,
            max_steps=self.max_episode_steps,
            obstacle_radius=self.obstacle_radius,
            render_mode=self.render_mode,
        )

    def run_episode(self, seed: int | None = None) -> dict:
        result = super().run_episode(seed=seed)
        # `terminated` = reached the goal (vs. `truncated` = ran out of steps).
        result["success"] = result["terminated"]
        return result


class DummyBimodalDiagnosticSimulation:
    """Diagnostic runner for DummyBimodal-v0 — mirrors `DummySimulation`.

    Not part of the BaseSimulation/hyperparam_search contract (same as plain
    Dummy-v0's DummySimulation: it drives `plot_dummy_debug` snapshots rather
    than returning per-episode reward dicts for auto-eval). Use this to
    generate the visual "CP cloud splits into two clusters near the
    obstacle" figures described in Tier 1 of the multimodality writeup.
    """

    def __init__(
        self,
        control_point_generator,
        q_estimator,
        device="cpu",
        render_mode=None,
        save_dir="plots/dummy_bimodal",
        step_size=0.1,
        goal_radius=0.1,
        obstacle_radius=0.25,
        max_episode_steps=200,
        frame_stack=1,
        snapshot_steps=None,
        **kwargs,
    ):
        self.model = control_point_generator
        self.estimator = q_estimator
        self.device = device
        self.render_mode = render_mode
        self.save_dir = save_dir
        self.step_size = step_size
        self.goal_radius = goal_radius
        self.obstacle_radius = obstacle_radius
        self.obstacle_center = np.zeros(2, dtype=np.float32)
        self.max_episode_steps = max_episode_steps
        self.frame_stack = frame_stack
        self.snapshot_steps = snapshot_steps or [1, 5, 10, 20, 50, 100]
        self.langevin_config = kwargs.get("langevin_config", {})

        self.obs_normalizer = ObservationNormalizer(
            env_id="DummyBimodal-v0", device=self.device, frame_stack=self.frame_stack
        )

        self.env = DummyBimodalEnv(
            step_size=step_size,
            goal_radius=goal_radius,
            max_steps=max_episode_steps,
            obstacle_radius=obstacle_radius,
            render_mode=render_mode,
        )

        self.all_results = []

    def _build_state_with_history(self, obs, history_buffer):
        if self.frame_stack <= 1:
            return obs.copy()
        frames = [obs]
        for i in range(self.frame_stack - 1):
            if i < len(history_buffer):
                frames.append(history_buffer[-(i + 1)])
            else:
                frames.append(history_buffer[0] if history_buffer else obs)
        return np.concatenate(frames).astype(np.float32)

    def _make_langevin_fn(self):
        def langevin_fn(model, estimator, device, state):
            n_samples = 64
            mins = [-1.0]
            maxs = [1.0]

            initial_guess = torch.from_numpy(
                sample_uniform(n_samples, 1, mins, maxs)
            ).float().to(device)

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
                device=device,
            )

            samples = samples.squeeze(0)
            trajs_tensor = torch.stack(trajs).squeeze(1).permute(1, 0, 2)
            return samples.cpu().numpy(), trajs_tensor.cpu().numpy()

        return langevin_fn

    def run_simulation(self, num_episodes=6, seed=None):
        print(f"Running DummyBimodal simulation ({num_episodes} episodes)...")
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

                state_np = self._build_state_with_history(obs, history_buffer)
                state_tensor = torch.from_numpy(state_np).float().unsqueeze(0).to(self.device)
                state_tensor = self.obs_normalizer.normalize(state_tensor)

                if step in self.snapshot_steps:
                    save_path = os.path.join(
                        self.save_dir, f"ep{ep}_step{step:03d}_seed{seed}.png"
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
                        title=f"Dummy Bimodal Nav (Seed {seed})",
                        obstacle_center=self.obstacle_center,
                        obstacle_radius=self.obstacle_radius,
                    )

                with torch.no_grad():
                    cps = self.model(state_tensor)
                    state_exp = state_tensor.unsqueeze(1).expand(-1, cps.shape[1], -1)
                    q_vals = self.estimator(state_exp, cps).squeeze(-1)
                    best_idx = q_vals.argmax(dim=1)
                    action = cps[0, best_idx[0]].cpu().numpy()

                obs, reward, terminated, truncated, info = self.env.step(action)
                total_reward += reward

                agent_pos = obs[2:4].copy()
                trajectory.append(agent_pos.copy())
                history_buffer.append(obs.copy())

                done = terminated or truncated

            success = terminated
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
