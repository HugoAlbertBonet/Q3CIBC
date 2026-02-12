
"""Simulation wrapper for Dummy Verification."""

import os
import torch
import numpy as np
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
        **kwargs
    ):
        self.model = control_point_generator
        self.estimator = q_estimator
        self.device = device
        self.render_mode = render_mode
        self.save_dir = save_dir
        
        self.env = DummyEnv(render_mode=render_mode)
        self.langevin_config = kwargs.get("langevin_config", {})

    def run_simulation(self, num_episodes=1, seed=None):
        print(f"Generating Dummy Verification Plots in {self.save_dir}...")
        os.makedirs(self.save_dir, exist_ok=True)
        
        # Dataset not strictly needed for this plot as Expert is fixed at 0
        dataset = None 
        
        # Define Langevin function wrapper
        def langevin_fn(model, estimator, device):
            n_samples = 64
            # Action bounds [-pi, pi]
            # sample_uniform expects lists
            mins = [-np.pi]
            maxs = [np.pi]
            
            initial_guess = torch.from_numpy(
                sample_uniform(n_samples, 1, mins, maxs)
            ).float().to(device) # (1, 64, 1)
            
            # Fixed State obs
            obs = torch.tensor([[1.0, 0.0]]).to(device) # (1, 2)
            
            # Energy function: -Q(s, a)
            def energy(o, a):
                return -estimator(o, a).squeeze(-1)
            
            # Run Langevin
            samples, trajs = sample_langevin(
                energy_function=energy,
                observations=obs,
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
            
            # Reshape for plotting
            # samples: (1, 64, 1) -> (64, 1)
            samples = samples.squeeze(0)
            
            # trajs: list of (1, 64, 1) -> (64, Steps, 1)
            # Stack list -> (Steps, 1, 64, 1) -> squeeze dim 1 -> (Steps, 64, 1) -> permute -> (64, Steps, 1)
            trajs_tensor = torch.stack(trajs).squeeze(1).permute(1, 0, 2)
            
            return samples.cpu().numpy(), trajs_tensor.cpu().numpy()

        # Plot!
        save_path = os.path.join(self.save_dir, f"verification_seed{seed}.png")
        plot_dummy_debug(
            self.model,
            self.estimator,
            dataset,
            self.device,
            save_path,
            langevin_fn=langevin_fn,
            title=f"Dummy Verification (Seed {seed})"
        )
        
        return [{"total_reward": 1.0, "episode_length": 0, "success": True}]

    def close(self):
        self.env.close()

    def get_summary(self):
        return {
            "num_episodes": 1,
            "reward_mean": 1.0,
            "reward_std": 0.0,
            "success_rate": 1.0
        }
