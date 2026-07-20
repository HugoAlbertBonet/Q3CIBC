"""Shared loading + DFO inference for the real-robot IBC Push-T checkpoints.

Single source of truth for anything that has to agree bit-for-bit between the
robot client (scripts/deploy_pusht_real_ibc.py) and the offline diagnostic
(scripts/diagnose_pusht_actions_ibc.py): how the PixelEBM is rebuilt from a
seed directory, which weights are picked, how DFO selects an action, and how
that action is denormalized to metres.

The checkpoints are written by scripts/train_pusht_real_ibc.py, which follows
google-research/ibc's optimal Pushing-Pixels config
(configs/pushing_pixels/pixel_ebm_best.gin).
"""

from __future__ import annotations

from dataclasses import dataclass, field
import json
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import torch


# Fallbacks matching pixel_ebm_best.gin + mcmc.iterative_dfo defaults, used
# only when a per-run config predates a field.
DFO_DEFAULTS = {
    "samples": 2048,
    "iterations": 3,
    "iteration_std": 0.33,
    "std_decay": 0.5,
    "boundary_buffer": 0.05,
}


@dataclass
class IBCPolicy:
    """A loaded IBC checkpoint plus everything needed to act with it."""

    name: str
    seed_dir: Path
    checkpoint: Path
    ebm: Any
    act_min: np.ndarray
    act_max: np.ndarray
    norm_range: Tuple[float, float]
    frame_stack: int
    camera_streams: List[str]
    image_hw: Tuple[int, int]
    action_bounds: Tuple[float, float]
    dfo: Dict[str, Any] = field(default_factory=dict)

    @property
    def in_channels(self) -> int:
        return 3 * len(self.camera_streams) * self.frame_stack

    def nfe_info(self) -> Dict[str, Any]:
        """Function evaluations per action selection.

        DFO's cost is exact and static: each iteration scores the whole sample
        cloud through the value head once. The conv encoder runs a single time
        per step (late fusion) and is not part of the per-candidate count.
        """
        nfe = int(self.dfo["samples"]) * int(self.dfo["iterations"])
        return {
            "nfe": nfe,
            "policy_type": "ibc_ebm_dfo",
            "details": (
                f"IBC EBM + DFO: {self.dfo['iterations']} iterations x "
                f"{self.dfo['samples']} action samples = {nfe} value-head "
                f"evaluations (+1 conv encoder pass, late fusion)"
            ),
            "breakdown": {
                "dfo_samples": int(self.dfo["samples"]),
                "dfo_iterations": int(self.dfo["iterations"]),
                "encoder_passes": 1,
            },
        }


def load_run_config(seed_dir: Path) -> dict:
    cfg_path = Path(seed_dir) / "config.json"
    if not cfg_path.is_file():
        raise FileNotFoundError(f"missing per-run config: {cfg_path}")
    with cfg_path.open() as fh:
        config = json.load(fh)
    return config["environments"][config["active_env"]]


def resolve_checkpoint(seed_dir: Path, ckpt_step: int | None = None) -> Path:
    """Pick which weights to use.

    An explicit step wins. Otherwise prefer the final q_estimator.pt, falling
    back to the newest q_estimator_step*.pt snapshot so a seed that is still
    training (or whose job died) is still usable rather than failing outright.
    """
    seed_dir = Path(seed_dir)
    if ckpt_step is not None and ckpt_step > 0:
        return seed_dir / f"q_estimator_step{ckpt_step:06d}.pt"
    final = seed_dir / "q_estimator.pt"
    if final.is_file():
        return final
    snapshots = sorted(seed_dir.glob("q_estimator_step*.pt"))
    if not snapshots:
        raise FileNotFoundError(
            f"no q_estimator.pt and no q_estimator_step*.pt snapshots in {seed_dir}"
        )
    print(
        f"[warn] no final q_estimator.pt in {seed_dir}; using snapshot "
        f"{snapshots[-1].name} (training likely still in progress)"
    )
    return snapshots[-1]


def load_policy(
    seed_dir: Path,
    device: torch.device,
    ckpt_step: int | None = None,
    dfo_overrides: Dict[str, Any] | None = None,
) -> IBCPolicy:
    """Rebuild the PixelEBM from a seed directory and load its weights."""
    from utils.models import PixelQEstimator

    seed_dir = Path(seed_dir)
    env = load_run_config(seed_dir)
    # weights_only=False: our own trusted checkpoints (state dicts + numpy).
    norm_stats = torch.load(
        seed_dir / "norm_stats.pt", map_location="cpu", weights_only=False
    )

    frame_stack = int(norm_stats.get("frame_stack", env.get("frame_stack", 2)))
    camera_streams = list(
        norm_stats.get("camera_streams", env.get("camera_streams", ["images1"]))
    )
    image_hw = (int(env.get("image_height", 240)), int(env.get("image_width", 320)))
    a_lo, a_hi = env.get("action_bounds", [-1.0, 1.0])

    m = env["model"]
    ebm = (
        PixelQEstimator(
            action_dim=int(env.get("action_dim", 2)),
            in_channels=3 * len(camera_streams) * frame_stack,
            encoder_target_height=int(env.get("encoder_target_height", 180)),
            encoder_target_width=int(env.get("encoder_target_width", 240)),
            value_width=int(m.get("value_width", 1024)),
            value_num_blocks=int(m.get("value_num_blocks", 1)),
            cond_dim=0,
            encoder_kind=m.get("encoder_kind", "conv_maxpool"),
        )
        .to(device)
        .eval()
    )

    ckpt_path = resolve_checkpoint(seed_dir, ckpt_step)
    if not ckpt_path.is_file():
        raise FileNotFoundError(f"missing checkpoint weights: {ckpt_path}")
    ebm.load_state_dict(torch.load(ckpt_path, map_location=device, weights_only=False))

    inf = env.get("inference", {})
    dfo = {
        "samples": int(inf.get("dfo_samples", DFO_DEFAULTS["samples"])),
        "iterations": int(inf.get("dfo_iterations", DFO_DEFAULTS["iterations"])),
        "iteration_std": float(
            inf.get("dfo_iteration_std", DFO_DEFAULTS["iteration_std"])
        ),
        "std_decay": float(inf.get("dfo_std_decay", DFO_DEFAULTS["std_decay"])),
        "boundary_buffer": float(
            inf.get("uniform_boundary_buffer", DFO_DEFAULTS["boundary_buffer"])
        ),
    }
    for key, value in (dfo_overrides or {}).items():
        if value is not None:
            dfo[key] = value

    return IBCPolicy(
        name=f"{seed_dir.parent.name}/{seed_dir.name}/{ckpt_path.stem}",
        seed_dir=seed_dir,
        checkpoint=ckpt_path,
        ebm=ebm,
        act_min=np.asarray(norm_stats["act_min"], np.float32),
        act_max=np.asarray(norm_stats["act_max"], np.float32),
        norm_range=tuple(norm_stats.get("action_norm_range", (-1.0, 1.0))),
        frame_stack=frame_stack,
        camera_streams=camera_streams,
        image_hw=image_hw,
        action_bounds=(float(a_lo), float(a_hi)),
        dfo=dfo,
    )


@torch.no_grad()
def dfo_select(
    policy: IBCPolicy, obs_u8: torch.Tensor, return_initial_scores: bool = False
):
    """IBC's iterative_dfo over a batch of observations.

    Faithful to mcmc.iterative_dfo: the encoder runs once and the loop scores
    cached features (late fusion); samples start uniform in the action box
    widened by the boundary buffer; each iteration resamples from
    softmax(scores) and jitters with a halving Gaussian std; the final argmax
    picks the action.

    IBC's resample gathers with `tf.repeat(arange(N), bincount(idx))`, which is
    exactly the sampled index multiset in sorted order, so a per-row sort
    reproduces its ordering and batches without looping over rows.

    obs_u8: (B, C, H, W) uint8. Returns (B, A) actions clamped to the action
    box -- the boundary buffer admits candidates slightly outside the trained
    range and those must never reach hardware. With return_initial_scores,
    also returns the (B, N) scores over the initial uniform cloud, which is
    the honest picture of the learned energy surface.
    """
    ebm = policy.ebm
    a_lo, a_hi = policy.action_bounds
    n = int(policy.dfo["samples"])
    iterations = int(policy.dfo["iterations"])
    device = obs_u8.device
    B = obs_u8.shape[0]
    action_dim = ebm.action_dim

    features = ebm.encode(obs_u8)  # (B, F) — once per step

    buf = (a_hi - a_lo) * float(policy.dfo["boundary_buffer"])
    actions = torch.empty(B, n, action_dim, device=device).uniform_(
        a_lo - buf, a_hi + buf
    )
    std = float(policy.dfo["iteration_std"])
    initial_scores = None
    scores = None
    for it in range(iterations):
        scores = ebm.score(features, actions).squeeze(-1)  # (B, N)
        if it == 0:
            initial_scores = scores
        probs = torch.softmax(scores, dim=-1)
        idx = torch.multinomial(probs, n, replacement=True)
        idx, _ = idx.sort(dim=1)        # == IBC's bincount -> repeat ordering
        actions = torch.gather(
            actions, 1, idx.unsqueeze(-1).expand(-1, -1, action_dim)
        )
        if it < iterations - 1:
            actions = actions + torch.randn_like(actions) * std
            actions = actions.clamp(a_lo, a_hi)
            std *= float(policy.dfo["std_decay"])

    sel = scores.argmax(dim=1)
    chosen = actions[torch.arange(B, device=device), sel].clamp(a_lo, a_hi)
    if return_initial_scores:
        return chosen, initial_scores
    return chosen


def unnormalize(norm_action, policy: IBCPolicy) -> np.ndarray:
    """Normalized action -> metres, inverting the dataset's min-max scaling."""
    lo, hi = policy.norm_range
    scale = (policy.act_max - policy.act_min) / (hi - lo)
    return (
        policy.act_min + (np.asarray(norm_action, np.float32) - lo) * scale
    ).astype(np.float32)
