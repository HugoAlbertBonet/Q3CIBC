"""Loading + action selection for the real-robot Q3C Push-T checkpoints.

Sibling of `utils.ibc_policy`, exposing the same interface so one robot client
can drive either policy. Q3C differs from IBC in what it loads and how it picks
an action:

* Two networks, not one: a PixelControlPointGenerator that proposes a small
  cloud of candidate actions ("control points") from the image, plus a
  PixelQEstimator that scores them. IBC has no proposal network -- it scores a
  large uniform sample cloud with the EBM alone.
* Selection is an argmax over ~20 proposed control points, optionally softmax
  sampled at a temperature, against IBC's 2048-sample x 3-iteration DFO search.
  That makes Q3C roughly two orders of magnitude cheaper per step.
* EMA weights by default (`*_ema.pt`), matching how the trainer saves and how
  the seeds were evaluated. `--no_ema` selects the raw copy.

Optional CP-DFO refinement is supported and read from the per-run config's
`training` block (`inference_dfo_*`). The deployed Push-T seeds leave those at
0, i.e. pure CP-cloud argmax, which is what scripts/deploy_pusht_real.py did on
this hardware.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import json
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import torch


@dataclass
class Q3CPolicy:
    """A loaded Q3C checkpoint plus everything needed to act with it."""

    name: str
    seed_dir: Path
    checkpoints: List[Path]
    cp_gen: Any
    q_net: Any
    act_min: np.ndarray
    act_max: np.ndarray
    norm_range: Tuple[float, float]
    frame_stack: int
    camera_streams: List[str]
    image_hw: Tuple[int, int]
    action_bounds: Tuple[float, float]
    control_points: int
    cp_selection: str = "argmax"
    cp_temperature: float = 1.0
    dfo: Dict[str, Any] = field(default_factory=dict)

    kind = "q3c"

    @property
    def in_channels(self) -> int:
        return 3 * len(self.camera_streams) * self.frame_stack

    def describe(self) -> str:
        n_uniform = int(self.dfo.get("num_uniform", 0))
        iters = int(self.dfo.get("iterations", 0))
        base = (
            f"Q3C CP-cloud: {self.control_points} control points"
            + (f" + {n_uniform} uniform safety samples" if n_uniform else "")
            + f", selection={self.cp_selection}"
        )
        if self.cp_selection == "sample":
            base += f" (temp={self.cp_temperature})"
        if iters > 0:
            base += (
                f", CP-DFO refinement {iters} iters "
                f"(std={self.dfo['iteration_std']}, "
                f"x{self.dfo['std_decay']}/iter)"
            )
        return base

    def nfe_info(self) -> Dict[str, Any]:
        """Value-head evaluations per action selection.

        Pure argmax scores the candidate cloud once. With refinement each
        iteration rescores it, plus a final rescore before the argmax.
        """
        n_candidates = self.control_points + int(self.dfo.get("num_uniform", 0))
        iters = int(self.dfo.get("iterations", 0))
        nfe = n_candidates if iters == 0 else (iters + 1) * n_candidates
        detail = (
            f"Q3C: 1 scoring pass x {n_candidates} candidates = {nfe} "
            f"value-head evaluations"
            if iters == 0
            else (
                f"Q3C + CP-DFO: ({iters} iterations + 1 final rescore) x "
                f"{n_candidates} candidates = {nfe} value-head evaluations"
            )
        )
        return {
            "nfe": nfe,
            "policy_type": "q3c_cp_dfo" if iters else "q3c_cp_argmax",
            "details": detail + " (+1 conv encoder pass, late fusion)",
            "breakdown": {
                "control_points": self.control_points,
                "uniform_samples": int(self.dfo.get("num_uniform", 0)),
                "dfo_iterations": iters,
                "encoder_passes": 1,
            },
        }

    @torch.no_grad()
    def select(self, obs_u8: torch.Tensor) -> torch.Tensor:
        """(B, C, H, W) uint8 -> (B, A) normalized actions."""
        return cp_select(self, obs_u8)


def load_run_config(seed_dir: Path) -> dict:
    cfg_path = Path(seed_dir) / "config.json"
    if not cfg_path.is_file():
        raise FileNotFoundError(f"missing per-run config: {cfg_path}")
    with cfg_path.open() as fh:
        config = json.load(fh)
    return config["environments"][config["active_env"]]


def load_policy(
    seed_dir: Path,
    device: torch.device,
    no_ema: bool = False,
    cp_selection: str | None = None,
    dfo_overrides: Dict[str, Any] | None = None,
) -> Q3CPolicy:
    """Rebuild the CP generator + Q estimator and load their weights."""
    from utils.models import PixelControlPointGenerator, PixelQEstimator

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
    action_dim = int(env.get("action_dim", 2))
    in_channels = 3 * len(camera_streams) * frame_stack

    m = env["model"]
    enc_h = int(env.get("encoder_target_height", 180))
    enc_w = int(env.get("encoder_target_width", 240))
    control_points = int(m.get("control_points", 50))
    num_neurons = int(m.get("num_neurons", 512))
    num_hidden_layers = int(m.get("num_hidden_layers", 8))
    cp_width = int(m.get("cp_width", num_neurons))
    cp_depth = int(m.get("cp_depth", num_hidden_layers))
    cp_network_kind = m.get("cp_network_kind", "mlp")
    encoder_kind = m.get("encoder_kind", "conv_maxpool")

    cp_gen = (
        PixelControlPointGenerator(
            output_dim=action_dim,
            control_points=control_points,
            hidden_dims=[cp_width for _ in range(cp_depth)],
            action_bounds=(float(a_lo), float(a_hi)),
            network_kind=cp_network_kind,
            width=cp_width,
            depth=cp_depth,
            in_channels=in_channels,
            encoder_target_height=enc_h,
            encoder_target_width=enc_w,
            cond_dim=0,
            encoder_kind=encoder_kind,
            goal_dim=0,
        )
        .to(device)
        .eval()
    )
    q_net = (
        PixelQEstimator(
            action_dim=action_dim,
            in_channels=in_channels,
            encoder_target_height=enc_h,
            encoder_target_width=enc_w,
            value_width=int(m.get("value_width", 1024)),
            value_num_blocks=int(m.get("value_num_blocks", 1)),
            cond_dim=0,
            encoder_kind=encoder_kind,
            goal_dim=0,
        )
        .to(device)
        .eval()
    )

    suffix = "" if no_ema else "_ema"
    cp_path = seed_dir / f"control_point_generator{suffix}.pt"
    q_path = seed_dir / f"q_estimator{suffix}.pt"
    for path, model in ((cp_path, cp_gen), (q_path, q_net)):
        if not path.is_file():
            raise FileNotFoundError(
                f"missing checkpoint weights: {path}"
                + ("" if no_ema else "  (try --no_ema for the raw copy)")
            )
        model.load_state_dict(torch.load(path, map_location=device, weights_only=False))

    # Inference-time refinement lives in the per-run TRAINING block (see
    # README). Absent on the deployed Push-T seeds => 0 => pure CP argmax.
    train = env.get("training", {})
    dfo = {
        "iterations": int(train.get("inference_dfo_iterations", 0)),
        "iteration_std": float(train.get("inference_dfo_iteration_std", 0.1)),
        "std_decay": float(train.get("inference_dfo_iteration_std_decay", 0.5)),
        "num_uniform": int(train.get("inference_dfo_num_uniform", 0)),
    }
    for key, value in (dfo_overrides or {}).items():
        if value is not None:
            dfo[key] = value

    selection = cp_selection or str(norm_stats.get("cp_selection", "argmax"))
    if selection not in ("argmax", "sample"):
        raise ValueError(f"cp_selection must be argmax|sample, got {selection!r}")

    return Q3CPolicy(
        name=f"{seed_dir.parent.name}/{seed_dir.name}/"
        f"{'raw' if no_ema else 'ema'}",
        seed_dir=seed_dir,
        checkpoints=[cp_path, q_path],
        cp_gen=cp_gen,
        q_net=q_net,
        act_min=np.asarray(norm_stats["act_min"], np.float32),
        act_max=np.asarray(norm_stats["act_max"], np.float32),
        norm_range=tuple(norm_stats.get("action_norm_range", (-1.0, 1.0))),
        frame_stack=frame_stack,
        camera_streams=camera_streams,
        image_hw=image_hw,
        action_bounds=(float(a_lo), float(a_hi)),
        control_points=control_points,
        cp_selection=selection,
        cp_temperature=float(norm_stats.get("cp_selection_temperature", 1.0)),
        dfo=dfo,
    )


@torch.no_grad()
def cp_select(policy: Q3CPolicy, obs_u8: torch.Tensor) -> torch.Tensor:
    """Rank the control-point cloud and return the chosen action.

    Encoder runs once and the scoring reuses the cached features (late fusion),
    same as IBC. With `inference_dfo_iterations` > 0 the cloud is additionally
    refined by resample-and-jitter rounds before the final argmax.

    obs_u8: (B, C, H, W) uint8. Returns (B, A) clamped to the action box.
    """
    a_lo, a_hi = policy.action_bounds
    device = obs_u8.device
    B = obs_u8.shape[0]

    features = policy.q_net.encode(obs_u8)     # (B, F) — once per step
    candidates = policy.cp_gen(obs_u8)         # (B, P, A)
    action_dim = candidates.shape[-1]

    n_uniform = int(policy.dfo.get("num_uniform", 0))
    if n_uniform > 0:
        # Safety valve: a few uniform samples so selection is not confined to
        # whatever the proposal network happened to emit.
        unif = torch.empty(B, n_uniform, action_dim, device=device).uniform_(a_lo, a_hi)
        candidates = torch.cat([candidates, unif], dim=1)
    n = candidates.shape[1]

    iterations = int(policy.dfo.get("iterations", 0))
    if iterations == 0:
        logits = policy.q_net.score(features, candidates).squeeze(-1)   # (B, N)
        if policy.cp_selection == "sample":
            probs = torch.softmax(logits / max(policy.cp_temperature, 1e-6), dim=-1)
            idx = torch.multinomial(probs, 1).squeeze(-1)               # (B,)
        else:
            idx = logits.argmax(dim=1)
        chosen = candidates[torch.arange(B, device=device), idx]
        return chosen.clamp(a_lo, a_hi)

    # CP-DFO refinement. Same resample-and-jitter loop as IBC's iterative_dfo,
    # but seeded from the control-point cloud instead of a uniform one, so it
    # needs far fewer candidates. The per-row sort reproduces IBC's
    # bincount -> repeat_interleave gather ordering.
    std = float(policy.dfo["iteration_std"])
    scores = None
    for it in range(iterations):
        scores = policy.q_net.score(features, candidates).squeeze(-1)
        probs = torch.softmax(scores, dim=-1)
        idx = torch.multinomial(probs, n, replacement=True)
        idx, _ = idx.sort(dim=1)
        candidates = torch.gather(
            candidates, 1, idx.unsqueeze(-1).expand(-1, -1, action_dim)
        )
        candidates = candidates + torch.randn_like(candidates) * std
        candidates = candidates.clamp(a_lo, a_hi)
        std *= float(policy.dfo["std_decay"])
    scores = policy.q_net.score(features, candidates).squeeze(-1)
    chosen = candidates[torch.arange(B, device=device), scores.argmax(dim=1)]
    return chosen.clamp(a_lo, a_hi)


def unnormalize(norm_action, policy: Q3CPolicy) -> np.ndarray:
    """Normalized action -> metres, inverting the dataset's min-max scaling."""
    lo, hi = policy.norm_range
    scale = (policy.act_max - policy.act_min) / (hi - lo)
    return (
        policy.act_min + (np.asarray(norm_action, np.float32) - lo) * scale
    ).astype(np.float32)
