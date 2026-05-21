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


# ────────────────────────────────────────────────────────────────────────────
# Pixel networks — ports of IBC's `networks/pixel_ebm.py`, `conv_maxpool.py`,
# and `dense_resnet_value.py` (Florence et al. 2021, `pushing_pixels/
# pixel_ebm_langevin.gin`). Used by active_env="pushing_pixels".
#
# Late fusion: the conv encoder runs ONCE per state, then the resulting
# feature vector is broadcast over the N candidate actions before late-fused
# into the value network. Critical for Langevin inner loops (many actions /
# state) — without it, you re-run the conv tower N times per step.
# ────────────────────────────────────────────────────────────────────────────


class ConvMaxpoolEncoder(nn.Module):
	"""4-layer Conv2D + MaxPool + GlobalAvgPool image encoder.

	Mirrors IBC's `get_conv_maxpool` (networks/layers/conv_maxpool.py): filters
	[32, 64, 128, 256], all kernel=3x3 padding=same ReLU, MaxPool2D(2,2) after
	each conv, then GlobalAveragePooling2D → 256-D feature vector.

	Input pipeline (IBC's `image_prepro.preprocess`):
	  1. uint8 → float32 in [0, 1].
	  2. bilinear resize to (target_h, target_w) = (180, 240) per the gin.

	`in_channels` is `3 * frame_stack` because frames are stacked channel-wise
	upstream — for sequence_length=2 we get a 6-channel image.
	"""

	def __init__(
		self,
		in_channels: int = 6,
		target_height: int = 180,
		target_width: int = 240,
		feature_dim: int = 256,
	) -> None:
		super().__init__()
		if feature_dim != 256:
			raise ValueError(
				"ConvMaxpoolEncoder mirrors IBC's get_conv_maxpool which outputs "
				"a 256-D vector (last conv = 256 filters → GlobalAvgPool). "
				f"Got feature_dim={feature_dim}."
			)
		self.target_height = target_height
		self.target_width = target_width

		self.conv1 = nn.Conv2d(in_channels, 32, kernel_size=3, padding=1)
		self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
		self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
		self.conv4 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
		self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
		self.relu = nn.ReLU(inplace=True)

	def forward(self, x: torch.Tensor) -> torch.Tensor:
		"""Args: x of shape (B, C, H, W), uint8 or float (any range).

		Returns: (B, 256) feature vector.
		"""
		if x.dtype == torch.uint8:
			x = x.float() / 255.0
		elif x.max() > 1.5:  # accept already-float but not-yet-scaled input
			x = x / 255.0
		if x.shape[-2:] != (self.target_height, self.target_width):
			x = nn.functional.interpolate(
				x, size=(self.target_height, self.target_width),
				mode="bilinear", align_corners=False,
			)
		x = self.pool(self.relu(self.conv1(x)))
		x = self.pool(self.relu(self.conv2(x)))
		x = self.pool(self.relu(self.conv3(x)))
		x = self.pool(self.relu(self.conv4(x)))
		# Global average pooling over spatial dims.
		x = x.mean(dim=[2, 3])  # (B, 256)
		return x


class _DenseResnetBlock(nn.Module):
	"""Bottleneck residual block: width/4 → width/4 → width, ReLU pre-activation.

	Port of IBC's ResNetDenseBlock (networks/layers/dense_resnet_value.py).
	No batch/layer norm (IBC's value config doesn't enable any).
	"""

	def __init__(self, width: int) -> None:
		super().__init__()
		self.dense0 = nn.Linear(width, width // 4)
		self.dense1 = nn.Linear(width // 4, width // 4)
		self.dense2 = nn.Linear(width // 4, width)
		self.dense3 = nn.Linear(width, width)  # projection if shapes mismatch
		self.act = nn.ReLU(inplace=True)

	def forward(self, x: torch.Tensor) -> torch.Tensor:
		y = self.dense0(self.act(x))
		y = self.dense1(self.act(y))
		y = self.dense2(self.act(y))
		# In IBC's impl, x's shape never mismatches y's once dense0 fixes width.
		# Defensive projection kept for parity with their `if x.shape != y.shape`.
		if x.shape[-1] != y.shape[-1]:
			x = self.dense3(self.act(x))
		return x + y


class DenseResnetValue(nn.Module):
	"""Dense + N ResNetBlocks + Dense(1). Port of IBC's DenseResnetValue.

	IBC's `pushing_pixels/pixel_ebm_langevin.gin` sets width=1024, num_blocks=1.
	`Normal(0, 0.05)` init on every Dense — matches IBC's `kernel_initializer='normal'`
	and `bias_initializer='normal'`, which in Keras both default to
	`RandomNormal(mean=0.0, stddev=0.05)`. Initial port used std=1.0; with a
	1024-wide hidden layer that produced ~32-std initial activations and
	prevented the Q estimator from learning (saw ~6% argmax-pick of the
	closest-to-expert CP). std=0.05 keeps activations in a sane range.
	"""

	_INIT_STD = 0.05

	def __init__(self, in_dim: int, width: int = 1024, num_blocks: int = 1) -> None:
		super().__init__()
		self.dense0 = nn.Linear(in_dim, width)
		self.blocks = nn.ModuleList(_DenseResnetBlock(width) for _ in range(num_blocks))
		self.dense1 = nn.Linear(width, 1)
		self._init_parameters()

	def _init_parameters(self) -> None:
		for m in self.modules():
			if isinstance(m, nn.Linear):
				nn.init.normal_(m.weight, mean=0.0, std=self._INIT_STD)
				if m.bias is not None:
					nn.init.normal_(m.bias, mean=0.0, std=self._INIT_STD)

	def forward(self, x: torch.Tensor) -> torch.Tensor:
		x = self.dense0(x)
		for block in self.blocks:
			x = block(x)
		return self.dense1(x)  # (..., 1)


class PixelControlPointGenerator(nn.Module):
	"""Image-conditioned CP generator.

	Encoder (ConvMaxpoolEncoder) → 256-D features → ControlPointGenerator-style
	MLP that emits N candidate actions per state. Encoder has its OWN weights
	(not shared with PixelQEstimator) — matches IBC's separate networks per
	loss head.
	"""

	def __init__(
		self,
		output_dim: int,
		control_points: int,
		hidden_dims: Sequence[int] = (256, 256),
		action_bounds: tuple[float, float] = (-1.0, 1.0),
		network_kind: str = "mlp",
		width: int | None = None,
		depth: int | None = None,
		use_spectral_norm: bool = False,
		activation: type[nn.Module] = nn.ReLU,
		in_channels: int = 6,
		encoder_target_height: int = 180,
		encoder_target_width: int = 240,
		encoder_feature_dim: int = 256,
	) -> None:
		super().__init__()
		self.encoder = ConvMaxpoolEncoder(
			in_channels=in_channels,
			target_height=encoder_target_height,
			target_width=encoder_target_width,
			feature_dim=encoder_feature_dim,
		)
		self.head = ControlPointGenerator(
			input_dim=encoder_feature_dim,
			output_dim=output_dim,
			hidden_dims=hidden_dims,
			activation=activation,
			control_points=control_points,
			action_bounds=action_bounds,
			network_kind=network_kind,
			width=width,
			depth=depth,
			use_spectral_norm=use_spectral_norm,
		)

	def forward(self, images: torch.Tensor) -> torch.Tensor:
		"""Args: images (B, C, H, W). Returns: (B, control_points, output_dim)."""
		features = self.encoder(images)
		return self.head(features)


class PixelQEstimator(nn.Module):
	"""Image-conditioned Q estimator with late fusion.

	Encoder (ConvMaxpoolEncoder) → 256-D features (run ONCE per state) →
	concat with each candidate action → DenseResnetValue → scalar Q.

	Late fusion accepts two input patterns:
	  - state=(B, C, H, W),    action=(B, A)         → returns (B, 1)
	  - state=(B, C, H, W),    action=(B, N, A)      → returns (B, N, 1)
	Encoder runs once in both cases; features are broadcast in the second.
	"""

	def __init__(
		self,
		action_dim: int,
		in_channels: int = 6,
		encoder_target_height: int = 180,
		encoder_target_width: int = 240,
		encoder_feature_dim: int = 256,
		value_width: int = 1024,
		value_num_blocks: int = 1,
	) -> None:
		super().__init__()
		self.encoder = ConvMaxpoolEncoder(
			in_channels=in_channels,
			target_height=encoder_target_height,
			target_width=encoder_target_width,
			feature_dim=encoder_feature_dim,
		)
		self.value = DenseResnetValue(
			in_dim=encoder_feature_dim + action_dim,
			width=value_width,
			num_blocks=value_num_blocks,
		)
		self.action_dim = action_dim
		self.encoder_feature_dim = encoder_feature_dim

	def encode(self, images: torch.Tensor) -> torch.Tensor:
		"""Run the conv encoder once per state. Returns (B, feature_dim)."""
		return self.encoder(images)

	def score(self, features: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
		"""Late-fuse pre-encoded features with action(s).

		Args:
		  features: (B, F) — output of `encode(...)`.
		  action:   (B, A) or (B, N, A).
		Returns: (B, 1) or (B, N, 1).
		"""
		if action.ndim == 3:
			B, N, _ = action.shape
			features = features.unsqueeze(1).expand(B, N, -1)
		x = torch.cat([features, action], dim=-1)
		return self.value(x)

	def forward(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
		"""Convenience: encode then score, in one call.

		Accepted state shapes:
		  - (B, C, H, W)       — raw image batch (preferred).
		  - (B, N, C, H, W)    — image broadcast over N candidates via
		    `.unsqueeze(1).expand(-1, N, -1, -1, -1)`. We collapse with
		    `state[:, 0]` since `expand` produces a stride-0 view (all N
		    slices share the same memory), so this is a SAFE drop, not a
		    drop of unique data. This lets call sites that were written for
		    flat states (`sample_langevin`, gradient-penalty paths) keep
		    working without per-call-site rewrites.

		Note: prefer the explicit `encode(...) → score(...)` pattern when
		evaluating many actions against the same state — it skips even the
		single slice op.
		"""
		if state.ndim == 5:
			state = state[:, 0]
		return self.score(self.encode(state), action)

