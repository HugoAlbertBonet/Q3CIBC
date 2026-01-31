"""Tests for the D4RL Pen environment (human-v2)."""

from tests.base_test import BaseEnvironmentTest


class TestPenHuman(BaseEnvironmentTest):
    """Test suite for D4RL/pen/human-v2 environment.
    
    The Adroit Hand Pen environment involves a robotic hand
    manipulating a pen to match a target orientation.
    
    Characteristics:
        - State dim: 45
        - Action dim: 24
        - Human demonstrations dataset
    """

    dataset_path = "D4RL/pen/human-v2"
    
    # Environment-specific configuration
    control_points = 30
    batch_size = 64
    
    # Optionally add environment-specific tests:
    
    def test_pen_environment_dimensions(self, dataset):
        """Verify expected dimensions for pen environment."""
        # Pen environment has specific known dimensions
        assert dataset.state_shape == 45, (
            f"Pen state should be 45-dim, got {dataset.state_shape}"
        )
        assert dataset.action_shape == 24, (
            f"Pen action should be 24-dim, got {dataset.action_shape}"
        )
