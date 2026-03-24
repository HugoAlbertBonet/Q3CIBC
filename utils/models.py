import torch
from torch import nn
from torch.nn.utils import spectral_norm
from typing import Sequence


class ControlPointGenerator(nn.Module):
	"""Produces multiple candidate action vectors per state."""

	def __init__(
		self,
		input_dim: int,
		output_dim: int,
		hidden_dims: Sequence[int] = (256, 256),
		activation: type[nn.Module] = nn.ReLU,
		control_points: int = 10,
		action_bounds: tuple[float, float] = (-1.0, 1.0),
	) -> None:
		super().__init__()
		self.output_dim = output_dim
		self.control_points = control_points
		self.action_min = action_bounds[0]
		self.action_max = action_bounds[1]

		layers = []
		prev_dim = input_dim
		for dim in hidden_dims:
			layers.append(nn.Linear(prev_dim, dim))
			layers.append(activation())
			prev_dim = dim

		#layers[-1] = nn.Tanh()  # Bound last hidden to [-1,1] to prevent sigmoid saturation

		layers.append(nn.Linear(prev_dim, output_dim * control_points))
		self.network = nn.Sequential(*layers)

	def forward(self, x: torch.Tensor) -> torch.Tensor:
		batch = x.shape[0]
		out = self.network(x)
		out = out.view(batch, self.control_points, self.output_dim)
		# Tanh maps to [-1, 1], then scale to [action_min, action_max]
		out = torch.tanh(out) * ((self.action_max - self.action_min) / 2) + (self.action_max + self.action_min) / 2
		return out

class QEstimator(nn.Module):
	"""State-conditioned Q-network that maps (state, action) pairs to Q-values."""

	def __init__(
		self,
		state_dim: int,
		action_dim: int,
		output_dim: int = 1,
		hidden_dims: Sequence[int] = (256, 256),
		activation: type[nn.Module] = nn.ReLU,
		dropout_rate: float = 0.0,
		use_spectral_norm: bool = False,
		init_mode: str = "default",
		init_std: float = 0.05,
	) -> None:
		super().__init__()
		self.state_dim = state_dim
		self.action_dim = action_dim
		self.dropout_rate = dropout_rate
		self.use_spectral_norm = use_spectral_norm
		self.init_mode = init_mode
		self.init_std = init_std

		layers = []
		prev_dim = state_dim + action_dim  # Concatenate state and action
		for dim in hidden_dims:
			layers.append(self._make_linear(prev_dim, dim))
			layers.append(activation())
			if dropout_rate > 0.0:
				layers.append(nn.Dropout(p=dropout_rate))
			prev_dim = dim

		layers.append(self._make_linear(prev_dim, output_dim))
		self.network = nn.Sequential(*layers)
		self._init_parameters()

	def _make_linear(self, in_dim: int, out_dim: int) -> nn.Module:
		"""Create a linear layer with optional spectral normalization."""
		layer = nn.Linear(in_dim, out_dim)
		if self.use_spectral_norm:
			layer = spectral_norm(layer)
		return layer

	def _init_parameters(self) -> None:
		"""Initialize model parameters.

		Modes:
		- default: keep PyTorch defaults
		- normal: Normal(0, init_std) for weights and biases
		"""
		if self.init_mode == "default":
			return

		if self.init_mode != "normal":
			raise ValueError(f"Unsupported init_mode: {self.init_mode}")

		for module in self.modules():
			if isinstance(module, nn.Linear):
				nn.init.normal_(module.weight, mean=0.0, std=self.init_std)
				if module.bias is not None:
					nn.init.normal_(module.bias, mean=0.0, std=self.init_std)

	def forward(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
		"""
		Args:
			state: State tensor of shape (B, state_dim) or (B, N, state_dim)
			action: Action tensor of shape (B, action_dim) or (B, N, action_dim)
		Returns:
			Q-values of shape (B, 1) or (B, N, 1)
		"""
		x = torch.cat([state, action], dim=-1)
		return self.network(x)

