import torch
from torch import nn
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
	) -> None:
		super().__init__()
		self.output_dim = output_dim
		self.control_points = control_points

		layers = []
		prev_dim = input_dim
		for dim in hidden_dims:
			layers.append(nn.Linear(prev_dim, dim))
			layers.append(activation())
			prev_dim = dim
		layers[-1] = nn.Tanh()  # Final activation to bound outputs

		layers.append(nn.Linear(prev_dim, output_dim * control_points))
		self.network = nn.Sequential(*layers)

	def forward(self, x: torch.Tensor) -> torch.Tensor:
		batch = x.shape[0]
		out = self.network(x)
		return out.view(batch, self.control_points, self.output_dim)

class QEstimator(nn.Module):
	"""Simple fully-connected network used to map states to actions."""

	def __init__(
		self,
		input_dim: int,
		output_dim: int,
		hidden_dims: Sequence[int] = (256, 256),
		activation: type[nn.Module] = nn.ReLU,
	) -> None:
		super().__init__()

		layers = []
		prev_dim = input_dim
		for dim in hidden_dims:
			layers.append(nn.Linear(prev_dim, dim))
			layers.append(activation())
			prev_dim = dim

		layers.append(nn.Linear(prev_dim, output_dim))
		self.network = nn.Sequential(*layers)

	def forward(self, x: torch.Tensor) -> torch.Tensor:
		return self.network(x)

