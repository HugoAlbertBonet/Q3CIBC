import torch
from torch import nn
from torch.nn.utils import spectral_norm
from typing import Sequence


def _make_linear(in_dim: int, out_dim: int, use_spectral_norm: bool = False) -> nn.Module:
	layer = nn.Linear(in_dim, out_dim)
	return spectral_norm(layer) if use_spectral_norm else layer


class ResNetPreActivationBlock(nn.Module):
	"""Pre-activation residual block (Florence et al., 2021, ResNetPreActivation).

	One block: y = activation(x); y = Linear(y); y = activation(y); y = Linear(y); return x + y.
	No normalization (the IBC paper's `.gin` configs set `ResNetLayer.normalizer = None`).
	"""

	def __init__(
		self,
		width: int,
		activation: type[nn.Module] = nn.ReLU,
		use_spectral_norm: bool = False,
	) -> None:
		super().__init__()
		self.act1 = activation()
		self.linear1 = _make_linear(width, width, use_spectral_norm)
		self.act2 = activation()
		self.linear2 = _make_linear(width, width, use_spectral_norm)

	def forward(self, x: torch.Tensor) -> torch.Tensor:
		y = self.act1(x)
		y = self.linear1(y)
		y = self.act2(y)
		y = self.linear2(y)
		return x + y


def _build_backbone(
	input_dim: int,
	output_dim: int,
	*,
	network_kind: str,
	hidden_dims: Sequence[int],
	width: int | None,
	depth: int | None,
	activation: type[nn.Module],
	use_spectral_norm: bool,
) -> nn.Sequential:
	"""Construct either a plain MLP or a ResNetPreActivation backbone.

	- network_kind == "mlp": stacks Linear/activation pairs with `hidden_dims`.
	- network_kind == "resnet": Linear(input -> width); `depth` ResNetPreActivation
	  blocks of width `width`; activation; Linear(width -> output).
	"""
	if network_kind == "mlp":
		layers = []
		prev = input_dim
		for dim in hidden_dims:
			layers.append(_make_linear(prev, dim, use_spectral_norm))
			layers.append(activation())
			prev = dim
		layers.append(_make_linear(prev, output_dim, use_spectral_norm))
		return nn.Sequential(*layers)

	if network_kind == "resnet":
		assert width is not None and depth is not None and depth >= 1, \
			"resnet kind requires width and depth >= 1"
		layers: list[nn.Module] = [_make_linear(input_dim, width, use_spectral_norm)]
		for _ in range(depth):
			layers.append(ResNetPreActivationBlock(width, activation, use_spectral_norm))
		layers.append(activation())
		layers.append(_make_linear(width, output_dim, use_spectral_norm))
		return nn.Sequential(*layers)

	raise ValueError(f"Unknown network_kind: {network_kind!r}. Expected 'mlp' or 'resnet'.")


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
		network_kind: str = "mlp",
		width: int | None = None,
		depth: int | None = None,
		use_spectral_norm: bool = False,
	) -> None:
		super().__init__()
		self.output_dim = output_dim
		self.control_points = control_points
		self.action_min = action_bounds[0]
		self.action_max = action_bounds[1]

		self.network = _build_backbone(
			input_dim=input_dim,
			output_dim=output_dim * control_points,
			network_kind=network_kind,
			hidden_dims=hidden_dims,
			width=width,
			depth=depth,
			activation=activation,
			use_spectral_norm=use_spectral_norm,
		)

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
		network_kind: str = "mlp",
		width: int | None = None,
		depth: int | None = None,
	) -> None:
		super().__init__()
		self.state_dim = state_dim
		self.action_dim = action_dim
		self.dropout_rate = dropout_rate
		self.use_spectral_norm = use_spectral_norm
		self.init_mode = init_mode
		self.init_std = init_std

		# ResNet pre-activation backbone has no native dropout slot; the IBC paper's
		# configs do not enable dropout on EBM. We honour dropout_rate only for MLP.
		if network_kind == "mlp" and dropout_rate > 0.0:
			layers: list[nn.Module] = []
			prev_dim = state_dim + action_dim
			for dim in hidden_dims:
				layers.append(_make_linear(prev_dim, dim, use_spectral_norm))
				layers.append(activation())
				layers.append(nn.Dropout(p=dropout_rate))
				prev_dim = dim
			layers.append(_make_linear(prev_dim, output_dim, use_spectral_norm))
			self.network = nn.Sequential(*layers)
		else:
			self.network = _build_backbone(
				input_dim=state_dim + action_dim,
				output_dim=output_dim,
				network_kind=network_kind,
				hidden_dims=hidden_dims,
				width=width,
				depth=depth,
				activation=activation,
				use_spectral_norm=use_spectral_norm,
			)
		self._init_parameters()

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

