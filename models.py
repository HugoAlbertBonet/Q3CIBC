import torch
from torch import nn
from typing import Sequence


class FeedForwardNN(nn.Module):
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

