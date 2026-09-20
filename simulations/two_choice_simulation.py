"""Simulation classes for TwoChoice-v0.

`TwoChoiceSimulation` follows the same BaseSimulation contract as every
other env's Simulation class so hyperparam_search.py's generic
evaluate_q3c(...) dispatch works unchanged. Since every episode is exactly
one step, run_episode IS select_action + step — there's no rollout loop.

`TwoChoiceDiagnosticSimulation` drives the trimmed 4-panel diagnostic plot
(utils.vis_two_choice.plot_two_choice_debug) the same way
DummyBimodalDiagnosticSimulation does for dummy_bimodal.
"""

from __future__ import annotations

import os

import numpy as np
import torch

from .base_simulation import BaseSimulation
from .two_choice_env import TwoChoiceEnv
from utils.vis_two_choice import plot_two_choice_debug
from utils.sampling import sample_langevin, sample_uniform
from utils.normalizations import ObservationNormalizer


class TwoChoiceSimulation(BaseSimulation):
    """Simulation for testing trained policies on TwoChoice-v0."""

    def __init__(
        self,
        control_point_generator: torch.nn.Module,
        q_estimator: torch.nn.Module,
        device: str = "cpu",
        max_episode_steps: int = 1,
        render_mode: str | None = None,
        frame_stack: int = 1,
        norm_stats: dict | None = None,
        min_separation: float = 0.3,
        success_tolerance: float = 0.05,
    ) -> None:
        super().__init__(
            env_id="TwoChoice-v0",
            control_point_generator=control_point_generator,
            q_estimator=q_estimator,
            device=device,
            max_episode_steps=max_episode_steps,
            frame_stack=frame_stack,
        )
        self.render_mode = render_mode
        self.norm_stats = norm_stats  # unused (trains unnormalized, like dummy_bimodal)
        self.min_separation = min_separation
        self.success_tolerance = success_tolerance
        self._act_min_t = None
        self._act_rng_t = None

    def create_env(self) -> TwoChoiceEnv:
        return TwoChoiceEnv(
            min_separation=self.min_separation,
            success_tolerance=self.success_tolerance,
            render_mode=self.render_mode,
        )

    def run_episode(self, seed: int | None = None) -> dict:
        if self.env is None:
            self.env = self.create_env()
        obs, _ = self.env.reset(seed=seed)
        stacked_obs = self._reset_frame_buffer(obs)
        action = self.select_action(stacked_obs)
        obs, reward, terminated, truncated, info = self.env.step(action)
        return {
            "episode_length": 1,
            "total_reward": float(reward),
            "terminated": bool(terminated),
            "truncated": bool(truncated),
            "success": bool(info.get("success", False)),
            "dist_to_mode_a": info.get("dist_to_mode_a"),
            "dist_to_mode_b": info.get("dist_to_mode_b"),
        }


class TwoChoiceDiagnosticSimulation:
    """Diagnostic runner for TwoChoice-v0 — drives plot_two_choice_debug.

    Not part of the BaseSimulation/hyperparam_search contract (same relation
    as DummyBimodalDiagnosticSimulation to DummyBimodalSimulation): this is
    for generating the visual CP-cloud-vs-two-modes figures, not for the
    auto-eval success-rate number.
    """

    def __init__(
        self,
        control_point_generator,
        q_estimator,
        device="cpu",
        save_dir="plots/two_choice",
        min_separation=0.3,
        success_tolerance=0.05,
        frame_stack=1,
        **kwargs,
    ):
        self.model = control_point_generator
        self.estimator = q_estimator
        self.device = device
        self.save_dir = save_dir
        self.min_separation = min_separation
        self.success_tolerance = success_tolerance
        self.frame_stack = frame_stack
        self.langevin_config = kwargs.get("langevin_config", {})

        self.obs_normalizer = ObservationNormalizer(
            env_id="TwoChoice-v0", device=self.device, frame_stack=self.frame_stack
        )
        self.env = TwoChoiceEnv(
            min_separation=min_separation, success_tolerance=success_tolerance,
        )
        self.all_results = []

    def _make_langevin_fn(self):
        def langevin_fn(model, estimator, device, state):
            n_samples = 64
            mins, maxs = [-1.0], [1.0]
            initial_guess = torch.from_numpy(
                sample_uniform(n_samples, 1, mins, maxs)
            ).float().to(device)

            def energy(o, a):
                return -estimator(o, a).squeeze(-1)

            samples, trajs = sample_langevin(
                energy_function=energy, observations=state,
                initial_actions=initial_guess, num_samples=n_samples,
                action_min=torch.tensor(mins).to(device),
                action_max=torch.tensor(maxs).to(device),
                num_iterations=self.langevin_config.get("num_iterations", 20),
                lr_init=self.langevin_config.get("lr_init", 0.1),
                lr_final=self.langevin_config.get("lr_final", 0.01),
                polynomial_decay_power=2.0, return_trajectories=True, device=device,
            )
            samples = samples.squeeze(0)
            trajs_tensor = torch.stack(trajs).squeeze(1).permute(1, 0, 2)
            return samples.cpu().numpy(), trajs_tensor.cpu().numpy()
        return langevin_fn

    def run_simulation(self, num_episodes=12, seed=None):
        print(f"Running TwoChoice diagnostic ({num_episodes} states)...")
        os.makedirs(self.save_dir, exist_ok=True)
        langevin_fn = self._make_langevin_fn()
        self.all_results = []

        for ep in range(num_episodes):
            ep_seed = (seed or 0) * 1000 + ep
            obs, _ = self.env.reset(seed=ep_seed)
            state_tensor = torch.from_numpy(obs).float().unsqueeze(0).to(self.device)
            state_tensor = self.obs_normalizer.normalize(state_tensor)

            save_path = os.path.join(self.save_dir, f"state{ep:03d}_seed{seed}.png")
            plot_two_choice_debug(
                model=self.model, estimator=self.estimator, device=self.device,
                save_path=save_path, state=state_tensor,
                mode_a=float(obs[0]), mode_b=float(obs[1]),
                state_idx=ep, langevin_fn=langevin_fn,
                title=f"TwoChoice (Seed {seed})",
            )

            with torch.no_grad():
                cps = self.model(state_tensor)
                state_exp = state_tensor.unsqueeze(1).expand(-1, cps.shape[1], -1)
                q_vals = self.estimator(state_exp, cps).squeeze(-1)
                best_idx = q_vals.argmax(dim=1)
                action = cps[0, best_idx[0]].cpu().numpy()

            _, reward, _, _, info = self.env.step(action)
            self.all_results.append({
                "total_reward": float(reward),
                "success": bool(info.get("success", False)),
            })
            print(f"  State {ep}: modes=({obs[0]:.2f},{obs[1]:.2f}) "
                  f"action={action[0]:.2f} reward={reward:.3f} success={info['success']}")

        return self.all_results

    def close(self):
        self.env.close()

    def get_summary(self):
        if not self.all_results:
            return {"num_episodes": 0, "reward_mean": 0.0, "success_rate": 0.0}
        rewards = [r["total_reward"] for r in self.all_results]
        successes = [r["success"] for r in self.all_results]
        return {
            "num_episodes": len(self.all_results),
            "reward_mean": np.mean(rewards),
            "reward_std": np.std(rewards),
            "success_rate": np.mean(successes),
        }
