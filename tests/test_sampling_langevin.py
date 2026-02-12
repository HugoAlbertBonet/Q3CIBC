"""Tests for Langevin MCMC sampling with polynomial LR decay."""

import pytest
import torch
import sys
sys.path.insert(0, str(__file__).rsplit('/', 2)[0])

from utils.sampling import sample_langevin


class QuadraticEnergy(torch.nn.Module):
    """Simple quadratic energy function for testing: E(x, y) = ||y||^2"""
    def forward(self, observations: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        # Energy is sum of squared actions, should be minimized at 0
        return (actions ** 2).sum(dim=-1)  # (B, N)


class TestSampleLangevin:
    """Test suite for sample_langevin function."""

    @pytest.fixture
    def energy_fn(self):
        return QuadraticEnergy()

    @pytest.fixture
    def action_bounds(self):
        action_dim = 4
        return (
            torch.full((action_dim,), -1.0),
            torch.full((action_dim,), 1.0)
        )

    def test_output_shape(self, energy_fn, action_bounds):
        """Test that output has correct shape."""
        batch_size = 8
        num_samples = 5
        obs_dim = 10
        action_min, action_max = action_bounds
        
        observations = torch.randn(batch_size, obs_dim)
        
        samples = sample_langevin(
            energy_function=energy_fn,
            observations=observations,
            num_samples=num_samples,
            action_min=action_min,
            action_max=action_max,
            num_iterations=10,
            lr_init=0.1,
            lr_final=1e-5,
            noise_scale=0.01
        )
        
        expected_shape = (batch_size, num_samples, action_min.shape[-1])
        assert samples.shape == expected_shape, f"Expected {expected_shape}, got {samples.shape}"

    def test_samples_within_bounds(self, energy_fn, action_bounds):
        """Test that samples are clipped to action bounds."""
        batch_size = 8
        num_samples = 5
        obs_dim = 10
        action_min, action_max = action_bounds
        
        observations = torch.randn(batch_size, obs_dim)
        
        samples = sample_langevin(
            energy_function=energy_fn,
            observations=observations,
            num_samples=num_samples,
            action_min=action_min,
            action_max=action_max,
            num_iterations=50,
            lr_init=0.1,
            lr_final=1e-5,
            noise_scale=1.0,
            delta_action_clip=0.1,
        )
        
        assert (samples >= action_min).all(), "Samples should be >= action_min"
        assert (samples <= action_max).all(), "Samples should be <= action_max"

    def test_samples_converge_to_minimum(self, energy_fn, action_bounds):
        """Test that samples converge toward energy minimum (0 for quadratic)."""
        batch_size = 16
        num_samples = 10
        obs_dim = 10
        action_min, action_max = action_bounds
        
        observations = torch.randn(batch_size, obs_dim)
        
        # Run with many iterations and low noise to allow convergence
        samples = sample_langevin(
            energy_function=energy_fn,
            observations=observations,
            num_samples=num_samples,
            action_min=action_min,
            action_max=action_max,
            num_iterations=200,
            lr_init=0.1,
            lr_final=1e-5,
            polynomial_decay_power=2.0,
            delta_action_clip=0.5,
            noise_scale=0.01,  # Small noise to allow convergence
        )
        
        # Samples should be close to 0 (the minimum of quadratic energy)
        mean_abs = samples.abs().mean()
        assert mean_abs < 0.5, f"Samples should converge near 0, got mean abs {mean_abs:.4f}"

    def test_no_gradients_in_output(self, energy_fn, action_bounds):
        """Test that returned samples are detached (no gradients)."""
        batch_size = 4
        num_samples = 3
        obs_dim = 10
        action_min, action_max = action_bounds
        
        observations = torch.randn(batch_size, obs_dim)
        
        samples = sample_langevin(
            energy_function=energy_fn,
            observations=observations,
            num_samples=num_samples,
            action_min=action_min,
            action_max=action_max,
            num_iterations=5,
            lr_init=0.1,
            lr_final=1e-5,
            noise_scale=1.0,
        )
        
        assert not samples.requires_grad, "Output samples should be detached"

    def test_polynomial_lr_decay(self, energy_fn, action_bounds):
        """Test that polynomial LR decay actually changes step sizes."""
        from utils.sampling import _polynomial_decay
        
        lr_init = 0.1
        lr_final = 1e-5
        power = 2.0
        num_steps = 50
        
        # First step should be close to lr_init
        lr_0 = _polynomial_decay(lr_init, lr_final, power, 0, num_steps)
        assert abs(lr_0 - lr_init) < 1e-6, f"First LR should be ~{lr_init}, got {lr_0}"
        
        # Last step should be close to lr_final
        lr_last = _polynomial_decay(lr_init, lr_final, power, num_steps - 1, num_steps)
        assert abs(lr_last - lr_final) < 1e-6, f"Last LR should be ~{lr_final}, got {lr_last}"
        
        # Middle should be between init and final
        lr_mid = _polynomial_decay(lr_init, lr_final, power, num_steps // 2, num_steps)
        assert lr_final < lr_mid < lr_init, f"Mid LR {lr_mid} should be between {lr_final} and {lr_init}"
