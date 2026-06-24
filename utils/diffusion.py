"""Diffusion Policy backbone + DDPM/DDIM samplers (self-contained, no `diffusers`).

This is the capacity-matched Diffusion Policy baseline for the Q3CIBC ablation:
the denoiser reuses the SAME MLP/ResNet trunk as the Q3C `QEstimator`
(`utils.models._build_backbone`) — only the I/O changes. Instead of
`(state, action) -> Q`, the denoiser computes `(state, noisy_action, t) -> eps`
(epsilon-prediction, the Diffusion Policy / DDPM default, Ho et al. 2020;
Chi et al. 2023).

Schedulers are implemented here directly (≈ DDPMScheduler / DDIMScheduler from
HF `diffusers`) to avoid adding `diffusers` to the locked server venv. One
trained denoiser is sampled with EITHER DDPM (stochastic, full T steps) or DDIM
(deterministic, sub-sampled steps) — the same checkpoint, two samplers. That is
exactly the DDPM-vs-DDIM axis of the study.

Action space convention: actions are normalized to [-1, 1] at the dataset level
(matches PushingDataset), so the samplers clamp the predicted clean sample x0 to
[-1, 1] each step (the `clip_sample=True` behaviour in Diffusion Policy).
"""

from __future__ import annotations

import math
from typing import Sequence

import torch
from torch import nn

from utils.models import _build_backbone


# ────────────────────────────────────────────────────────────────────────────
# Timestep embedding
# ────────────────────────────────────────────────────────────────────────────

class SinusoidalTimeEmbedding(nn.Module):
    """Standard transformer/DDPM sinusoidal embedding of the diffusion timestep."""

    def __init__(self, dim: int) -> None:
        super().__init__()
        if dim % 2 != 0:
            raise ValueError(f"time_emb_dim must be even, got {dim}")
        self.dim = dim

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        # t: (B,) float tensor of timestep indices.
        half = self.dim // 2
        device = t.device
        freqs = torch.exp(
            -math.log(10000.0) * torch.arange(half, device=device, dtype=torch.float32) / half
        )
        args = t.float().unsqueeze(-1) * freqs.unsqueeze(0)  # (B, half)
        return torch.cat([torch.sin(args), torch.cos(args)], dim=-1)  # (B, dim)


# ────────────────────────────────────────────────────────────────────────────
# Denoiser — Q-estimator trunk, epsilon-prediction head
# ────────────────────────────────────────────────────────────────────────────

class DiffusionDenoiser(nn.Module):
    """epsilon-predictor reusing the Q3C `QEstimator` trunk.

    Input  = concat(state, noisy_action, time_embedding)  -> `_build_backbone`
    Output = predicted noise, shape == action_dim.

    `network_kind`/`width`/`depth`/`use_spectral_norm` are passed straight to the
    same `_build_backbone` used by `QEstimator`, so the trunk is byte-for-byte the
    Q3C estimator architecture (capacity-matched ablation). The ONLY structural
    deltas are: input grows by (action_dim + time_emb_dim), output = action_dim.
    """

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        *,
        time_emb_dim: int = 128,
        network_kind: str = "mlp",
        width: int | None = None,
        depth: int | None = None,
        hidden_dims: Sequence[int] | None = None,
        use_spectral_norm: bool = False,
        activation: type[nn.Module] = nn.ReLU,
    ) -> None:
        super().__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.time_emb = SinusoidalTimeEmbedding(time_emb_dim)
        if hidden_dims is None:
            w = width if width is not None else 256
            d = depth if depth is not None else 2
            hidden_dims = [w] * d
        self.network = _build_backbone(
            input_dim=state_dim + action_dim + time_emb_dim,
            output_dim=action_dim,
            network_kind=network_kind,
            hidden_dims=hidden_dims,
            width=width,
            depth=depth,
            activation=activation,
            use_spectral_norm=use_spectral_norm,
        )

    def forward(
        self, state: torch.Tensor, noisy_action: torch.Tensor, t: torch.Tensor
    ) -> torch.Tensor:
        te = self.time_emb(t)
        x = torch.cat([state, noisy_action, te], dim=-1)
        return self.network(x)


# ────────────────────────────────────────────────────────────────────────────
# Beta schedules
# ────────────────────────────────────────────────────────────────────────────

def _make_betas(num_timesteps: int, schedule: str) -> torch.Tensor:
    if schedule == "linear":
        # DDPM linear schedule (Ho et al. 2020), scaled to any T.
        scale = 1000.0 / num_timesteps
        return torch.linspace(
            scale * 1e-4, scale * 0.02, num_timesteps, dtype=torch.float64
        ).float()
    if schedule == "cosine":
        # Nichol & Dhariwal (2021) cosine schedule.
        s = 0.008
        steps = num_timesteps + 1
        x = torch.linspace(0, num_timesteps, steps, dtype=torch.float64)
        acp = torch.cos(((x / num_timesteps) + s) / (1 + s) * math.pi / 2) ** 2
        acp = acp / acp[0]
        betas = 1 - (acp[1:] / acp[:-1])
        return betas.clamp(max=0.999).float()
    raise ValueError(f"Unknown beta_schedule: {schedule!r}. Expected 'linear' or 'cosine'.")


# ────────────────────────────────────────────────────────────────────────────
# Gaussian diffusion: training loss + DDPM / DDIM sampling
# ────────────────────────────────────────────────────────────────────────────

class GaussianDiffusion:
    """epsilon-prediction Gaussian diffusion with DDPM and DDIM samplers.

    Buffers (alphas_cumprod etc.) live on `device`. The same instance is used at
    train time (`training_loss`) and at eval time (`ddpm_sample` / `ddim_sample`).
    """

    def __init__(
        self,
        num_timesteps: int = 100,
        beta_schedule: str = "cosine",
        device: str | torch.device = "cpu",
        clip_sample: bool = True,
        action_low: float = -1.0,
        action_high: float = 1.0,
    ) -> None:
        self.num_timesteps = int(num_timesteps)
        self.device = torch.device(device)
        self.clip_sample = clip_sample
        self.action_low = action_low
        self.action_high = action_high

        betas = _make_betas(self.num_timesteps, beta_schedule).to(self.device)
        alphas = 1.0 - betas
        acp = torch.cumprod(alphas, dim=0)
        acp_prev = torch.cat([torch.ones(1, device=self.device), acp[:-1]])

        self.betas = betas
        self.alphas = alphas
        self.alphas_cumprod = acp
        self.alphas_cumprod_prev = acp_prev
        self.sqrt_acp = torch.sqrt(acp)
        self.sqrt_one_minus_acp = torch.sqrt(1.0 - acp)
        # Posterior q(x_{t-1} | x_t, x_0) coefficients.
        self.posterior_var = betas * (1.0 - acp_prev) / (1.0 - acp)
        self.posterior_mean_coef1 = betas * torch.sqrt(acp_prev) / (1.0 - acp)
        self.posterior_mean_coef2 = (1.0 - acp_prev) * torch.sqrt(alphas) / (1.0 - acp)

    # ── training ────────────────────────────────────────────────────────────
    def q_sample(self, x0: torch.Tensor, t: torch.Tensor, noise: torch.Tensor) -> torch.Tensor:
        sqrt_acp = self.sqrt_acp[t].unsqueeze(-1)
        sqrt_omacp = self.sqrt_one_minus_acp[t].unsqueeze(-1)
        return sqrt_acp * x0 + sqrt_omacp * noise

    def training_loss(
        self, model: nn.Module, state: torch.Tensor, x0: torch.Tensor
    ) -> torch.Tensor:
        B = x0.shape[0]
        t = torch.randint(0, self.num_timesteps, (B,), device=x0.device)
        noise = torch.randn_like(x0)
        xt = self.q_sample(x0, t, noise)
        eps_pred = model(state, xt, t.float())
        return torch.mean((eps_pred - noise) ** 2)

    # ── helpers ───────────────────────────────────────────────────────────────
    def _predict_x0(self, xt: torch.Tensor, t: int, eps: torch.Tensor) -> torch.Tensor:
        x0 = (xt - self.sqrt_one_minus_acp[t] * eps) / self.sqrt_acp[t]
        if self.clip_sample:
            x0 = x0.clamp(self.action_low, self.action_high)
        return x0

    # ── DDPM (stochastic, full chain) ─────────────────────────────────────────
    @torch.no_grad()
    def ddpm_sample(
        self, model: nn.Module, state: torch.Tensor, action_dim: int
    ) -> torch.Tensor:
        B = state.shape[0]
        x = torch.randn(B, action_dim, device=self.device)
        for t in reversed(range(self.num_timesteps)):
            t_batch = torch.full((B,), t, device=self.device, dtype=torch.float32)
            eps = model(state, x, t_batch)
            x0 = self._predict_x0(x, t, eps)
            mean = self.posterior_mean_coef1[t] * x0 + self.posterior_mean_coef2[t] * x
            if t > 0:
                noise = torch.randn_like(x)
                x = mean + torch.sqrt(self.posterior_var[t]) * noise
            else:
                x = mean
        return x

    # ── DDIM (deterministic when eta=0, sub-sampled) ──────────────────────────
    @torch.no_grad()
    def ddim_sample(
        self,
        model: nn.Module,
        state: torch.Tensor,
        action_dim: int,
        num_steps: int = 10,
        eta: float = 0.0,
    ) -> torch.Tensor:
        B = state.shape[0]
        x = torch.randn(B, action_dim, device=self.device)
        # Evenly-spaced sub-sequence of the training timesteps, descending.
        step_idx = torch.linspace(
            0, self.num_timesteps - 1, num_steps, device=self.device
        ).round().long()
        seq = list(reversed(step_idx.tolist()))
        for i, t in enumerate(seq):
            t_batch = torch.full((B,), t, device=self.device, dtype=torch.float32)
            eps = model(state, x, t_batch)
            x0 = self._predict_x0(x, t, eps)
            acp_t = self.alphas_cumprod[t]
            t_prev = seq[i + 1] if i + 1 < len(seq) else -1
            acp_prev = self.alphas_cumprod[t_prev] if t_prev >= 0 else torch.ones((), device=self.device)
            sigma = (
                eta
                * torch.sqrt((1 - acp_prev) / (1 - acp_t))
                * torch.sqrt(1 - acp_t / acp_prev)
            )
            dir_xt = torch.sqrt((1 - acp_prev - sigma**2).clamp(min=0.0)) * eps
            x = torch.sqrt(acp_prev) * x0 + dir_xt
            if eta > 0 and t_prev >= 0:
                x = x + sigma * torch.randn_like(x)
        if self.clip_sample:
            x = x.clamp(self.action_low, self.action_high)
        return x


# ────────────────────────────────────────────────────────────────────────────
# Config-driven factories (shared by training + eval so they stay in lock-step)
# ────────────────────────────────────────────────────────────────────────────

def resolve_dp_params(env_config: dict, training_shared: dict | None = None) -> dict:
    """Resolve DP hyperparameters with precedence: env training > model.diffusion > shared > default.

    The per-trial config written by hyperparam_search_dp routes ALL --fixed-params
    into env_config['training'], so that block wins. `model.diffusion` holds the
    standalone defaults.
    """
    training_shared = training_shared or {}
    tr = env_config.get("training", {})
    dp = env_config.get("model", {}).get("diffusion", {})

    def g(key, default):
        if key in tr:
            return tr[key]
        if key in dp:
            return dp[key]
        if key in training_shared:
            return training_shared[key]
        return default

    return {
        "num_train_timesteps": int(g("num_train_timesteps", 100)),
        "beta_schedule": str(g("beta_schedule", "cosine")),
        "time_emb_dim": int(g("time_emb_dim", 128)),
        "denoiser_network_kind": str(g("denoiser_network_kind", "mlp")),
        "denoiser_width": int(g("denoiser_width", 256)),
        "denoiser_depth": int(g("denoiser_depth", 2)),
        "denoiser_use_spectral_norm": bool(g("denoiser_use_spectral_norm", False)),
        "ema_decay": float(g("ema_decay", 0.0)),
        # Eval-only sampler knobs.
        "ddim_eval_steps": list(g("ddim_eval_steps", [10, 25])),
        "ddim_eta": float(g("ddim_eta", 0.0)),
        "eval_ddpm": bool(g("eval_ddpm", True)),
    }


def build_denoiser(
    state_dim: int, action_dim: int, dp: dict, device: str | torch.device = "cpu"
) -> DiffusionDenoiser:
    return DiffusionDenoiser(
        state_dim=state_dim,
        action_dim=action_dim,
        time_emb_dim=dp["time_emb_dim"],
        network_kind=dp["denoiser_network_kind"],
        width=dp["denoiser_width"],
        depth=dp["denoiser_depth"],
        use_spectral_norm=dp["denoiser_use_spectral_norm"],
    ).to(device)


def build_diffusion(
    dp: dict, device: str | torch.device = "cpu", action_bounds: tuple[float, float] = (-1.0, 1.0)
) -> GaussianDiffusion:
    return GaussianDiffusion(
        num_timesteps=dp["num_train_timesteps"],
        beta_schedule=dp["beta_schedule"],
        device=device,
        clip_sample=True,
        action_low=float(action_bounds[0]),
        action_high=float(action_bounds[1]),
    )
