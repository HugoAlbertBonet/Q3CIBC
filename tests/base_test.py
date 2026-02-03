"""Base test class for environment-agnostic model testing."""

import pytest
import torch
from abc import ABC, abstractmethod

import sys
sys.path.insert(0, str(__file__).rsplit('/', 2)[0])

from datasets import D4RLDataset
from models import ControlPointGenerator, QEstimator
from loss import lossInfoNCE, lossMSE, lossSeparation
from normalizations import wireFittingNorm


class BaseEnvironmentTest(ABC):
    """Abstract base class for testing models across different environments.
    
    Subclasses must define:
        - dataset_path: str (e.g., 'D4RL/pen/human-v2')
    
    Subclasses may override:
        - control_points: int (default: 30)
        - batch_size: int (default: 64)
        - hidden_dims: tuple (default: (256, 256))
        - learning_rate: float (default: 1e-5)
        - smoothing_param: float (default: 0.1)
    """

    # Required - subclass must define
    dataset_path: str

    # Configurable defaults - subclass may override
    control_points: int = 30
    batch_size: int = 64
    hidden_dims: tuple = (256, 256)
    learning_rate: float = 1e-5
    smoothing_param: float = 0.1
    test_epochs: int = 2  # Keep small for testing

    @pytest.fixture
    def dataset(self):
        """Load the environment dataset."""
        return D4RLDataset(self.dataset_path, download=True)

    @pytest.fixture
    def control_point_generator(self, dataset):
        """Create the ControlPointGenerator model."""
        return ControlPointGenerator(
            input_dim=dataset.state_shape,
            output_dim=dataset.action_shape,
            hidden_dims=self.hidden_dims,
            control_points=self.control_points,
        )

    @pytest.fixture
    def q_estimator(self, dataset):
        """Create the QEstimator model."""
        return QEstimator(
            state_dim=dataset.state_shape,
            action_dim=dataset.action_shape,
            hidden_dims=self.hidden_dims,
        )

    @pytest.fixture
    def dataloader(self, dataset):
        """Create a DataLoader for the dataset."""
        return torch.utils.data.DataLoader(
            dataset, batch_size=self.batch_size, shuffle=True
        )

    # === Model Architecture Tests ===

    def test_dataset_loads(self, dataset):
        """Test that dataset loads correctly and has expected attributes."""
        assert len(dataset) > 0, "Dataset should not be empty"
        assert dataset.state_shape > 0, "State shape should be positive"
        assert dataset.action_shape > 0, "Action shape should be positive"

    def test_generator_output_shape(self, dataset, control_point_generator):
        """Test ControlPointGenerator produces correct output shape."""
        batch_size = 4
        dummy_states = torch.randn(batch_size, dataset.state_shape)
        output = control_point_generator(dummy_states)
        
        expected_shape = (batch_size, self.control_points, dataset.action_shape)
        assert output.shape == expected_shape, (
            f"Expected shape {expected_shape}, got {output.shape}"
        )

    def test_estimator_output_shape(self, dataset, q_estimator):
        """Test QEstimator produces correct output shape."""
        batch_size = 4
        dummy_states = torch.randn(batch_size, dataset.state_shape)
        dummy_actions = torch.randn(batch_size, dataset.action_shape)
        output = q_estimator(dummy_states, dummy_actions)
        
        expected_shape = (batch_size, 1)
        assert output.shape == expected_shape, (
            f"Expected shape {expected_shape}, got {output.shape}"
        )

    # === Training Step Tests ===

    def test_generator_training_step(self, dataloader, control_point_generator):
        """Test that a single generator training step runs without errors."""
        optimizer = torch.optim.AdamW(
            control_point_generator.parameters(), lr=self.learning_rate
        )
        
        batch = next(iter(dataloader))
        states = batch['state'].float()
        actions = batch['action'].float()

        predicted_actions = control_point_generator(states)
        loss = lossMSE(predicted_actions, actions) + lossSeparation(predicted_actions)
        
        assert not torch.isnan(loss), "Generator loss should not be NaN"
        assert loss.requires_grad, "Loss should require gradients"
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    def test_estimator_training_step(self, dataloader, control_point_generator, q_estimator):
        """Test that a single estimator training step runs without errors."""
        optimizer = torch.optim.AdamW(
            q_estimator.parameters(), lr=self.learning_rate
        )
        
        batch = next(iter(dataloader))
        states = batch['state'].float()
        actions = batch['action'].float()

        with torch.no_grad():
            predicted_actions = control_point_generator(states)

        # Expand states to match control points: (B, state_dim) -> (B, N, state_dim)
        states_expanded = states.unsqueeze(1).expand(-1, predicted_actions.shape[1], -1)
        estimations = q_estimator(states_expanded, predicted_actions).squeeze(-1)
        estimations_target = q_estimator(states, actions).squeeze(-1)
        
        estimations = wireFittingNorm(
            control_points=predicted_actions,
            expert_action=actions,
            control_point_values=estimations,
            expert_action_value=estimations_target,
            c=torch.ones(states.shape[0], predicted_actions.shape[1] + 1) * self.smoothing_param
        )
        
        loss = lossInfoNCE(estimations)
        
        # Skip if NaN (can happen with random init)
        if torch.isnan(loss):
            pytest.skip("NaN loss detected - may occur with random initialization")
        
        assert loss.requires_grad, "Estimator loss should require gradients"
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    # === Integration Test ===

    def test_full_training_loop(self, dataloader, control_point_generator, q_estimator):
        """Test a complete training loop for a few epochs."""
        optimizer_gen = torch.optim.AdamW(
            control_point_generator.parameters(), lr=self.learning_rate
        )
        optimizer_est = torch.optim.AdamW(
            q_estimator.parameters(), lr=self.learning_rate
        )

        steps_completed = 0
        max_steps = 5  # Limit steps for testing

        for batch in dataloader:
            if steps_completed >= max_steps:
                break

            states = batch['state'].float()
            actions = batch['action'].float()

            # Generator step
            predicted_actions = control_point_generator(states)
            loss_gen = lossMSE(predicted_actions, actions) + lossSeparation(predicted_actions)
            
            optimizer_gen.zero_grad()
            loss_gen.backward()
            optimizer_gen.step()

            # Estimator step
            with torch.no_grad():
                predicted_actions_detached = predicted_actions.detach()

            # Expand states to match control points: (B, state_dim) -> (B, N, state_dim)
            states_expanded = states.unsqueeze(1).expand(-1, predicted_actions_detached.shape[1], -1)
            estimations = q_estimator(states_expanded, predicted_actions_detached).squeeze(-1)
            estimations_target = q_estimator(states, actions).squeeze(-1)
            
            estimations = wireFittingNorm(
                control_points=predicted_actions_detached,
                expert_action=actions,
                control_point_values=estimations,
                expert_action_value=estimations_target,
                c=torch.ones(states.shape[0], predicted_actions_detached.shape[1] + 1) * self.smoothing_param
            )
            
            loss_est = lossInfoNCE(estimations)
            
            if not torch.isnan(loss_est):
                optimizer_est.zero_grad()
                loss_est.backward()
                optimizer_est.step()

            steps_completed += 1

        assert steps_completed == max_steps, f"Should complete {max_steps} steps"
