#!/usr/bin/env python3
"""Offline check: are the IBC (EBM + DFO) Push-T checkpoints healthy?

IBC counterpart of scripts/diagnose_pusht_actions.py. Runs each checkpoint
over the TRAINING dataset and compares its predicted (normalized) actions to
the ground-truth teleop actions, so a bad rollout can be attributed to the
policy itself rather than to the deploy path.

Per seed it reports, for each action dim:
  - predicted vs ground-truth mean/std and fraction negative,
  - corr(pred_dx, pred_dy)   (≈1 => the two dims move together => diagonal),
  - predicted std             (≈0 => mode collapse, ignores the image),
  - MAE(pred, gt),
  - quadrant histogram of predicted actions (how many land in --,-+,+-,++).

Plus two EBM-specific health checks that the q3c diagnostic has no analogue
for. An implicit policy is only as good as its energy surface, and a surface
that never learned looks *fine* on the action statistics above (uniform DFO
samples in, near-uniform actions out — responsive-looking, but random):

  - `energy_entropy`: normalized entropy of softmax(scores) over the initial
    uniform action cloud. ≈1.0 means every candidate scores the same, i.e. a
    flat/random energy surface (the InfoNCE-≈-ln(K) failure mode this repo
    hit in an earlier IBC reproduction). Well below 1.0 means the EBM
    discriminates.
  - `expert_percentile`: fraction of the uniform cloud that the ground-truth
    action outscores. ≈0.5 is chance (the EBM cannot tell the expert action
    from noise); ≈1.0 means the expert action sits at an energy minimum,
    which is exactly what training optimized for.

Action selection is the same DFO as the deploy client (2048 samples, 3
iterations, std 0.33 halving, boundary buffer 0.05), batched. DFO is
stochastic; --dfo-repeats > 1 additionally reports how much the selected
action moves across independent DFO runs on the *same* observation, which
separates "the policy is uncertain" from "the search is under-resolved".

Run on the cluster (needs the project env with torch + tf for JPEG decode):

    sbatch scripts/diagnose_pusht_actions_ibc.sbatch
    # or directly:
    .venv/bin/python scripts/diagnose_pusht_actions_ibc.py \
        --output-root checkpoints/pusht_real_ibc --seeds 29 47
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
import sys
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# utils.ibc_policy is the single source of truth for the model build, weight
# selection and DFO, shared with scripts/deploy_pusht_real_ibc.py, so this
# diagnostic and the robot run agree bit-for-bit.
from utils import ibc_policy


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--output-root", type=Path,
                   default=ROOT / "checkpoints" / "pusht_real_ibc")
    p.add_argument("--seeds", type=int, nargs="+", default=[11, 29, 47, 83])
    p.add_argument("--dataset", type=Path, default=ROOT / "data" / "03-23-pusht-data.zip")
    p.add_argument("--num-samples", type=int, default=3000,
                   help="random transitions sampled from the dataset")
    p.add_argument("--batch-size", type=int, default=32,
                   help="transitions per forward pass. Kept well below the q3c "
                        "diagnostic's 128 because DFO scores batch_size x 2048 "
                        "candidates through a 1024-wide value head at once.")
    p.add_argument("--ckpt-step", type=int, default=None,
                   help="diagnose q_estimator_step{N:06d}.pt (an intermediate "
                        "snapshot) instead of the final q_estimator.pt. Use this "
                        "to check a seed whose training has not finished yet.")
    p.add_argument("--dfo-samples", type=int, default=None,
                   help="override DFO sample count (default: per-run config)")
    p.add_argument("--dfo-iterations", type=int, default=None,
                   help="override DFO iteration count (default: per-run config)")
    p.add_argument("--dfo-repeats", type=int, default=1,
                   help="independent DFO runs per observation. >1 reports the "
                        "per-observation spread of the selected action across "
                        "runs (inference stochasticity), at proportional cost.")
    p.add_argument("--zero-motion", action="store_true",
                   help="replace the real (t-1,t) frame stack with the newest "
                        "frame duplicated across all slots, so the obs carries "
                        "zero inter-frame motion. Tests whether a deploy-only "
                        "collapse is caused by out-of-distribution "
                        "static/near-static stacks rather than image content.")
    p.add_argument("--seed", type=int, default=0, help="RNG seed for DFO + sampling")
    p.add_argument("--device", default="cuda")
    p.add_argument("--out", type=Path,
                   default=ROOT / "results" / "pusht_action_diagnostic_ibc.json")
    return p.parse_args()


@torch.no_grad()
def energy_health(ebm, obs_u8, gt_actions, initial_scores):
    """How discriminative is the energy surface? See the module docstring.

    Returns (normalized_entropy (B,), expert_percentile (B,)).
    """
    features = ebm.encode(obs_u8)
    expert = ebm.score(features, gt_actions.unsqueeze(1)).squeeze(-1).squeeze(-1)  # (B,)

    log_probs = torch.log_softmax(initial_scores, dim=-1)
    entropy = -(log_probs.exp() * log_probs).sum(dim=-1)
    normalized_entropy = entropy / np.log(initial_scores.shape[1])

    percentile = (initial_scores < expert.unsqueeze(1)).float().mean(dim=1)
    return normalized_entropy, percentile


def quadrant_hist(a: np.ndarray) -> dict:
    dx, dy = a[:, 0], a[:, 1]
    return {
        "--": int(np.sum((dx < 0) & (dy < 0))),
        "-+": int(np.sum((dx < 0) & (dy >= 0))),
        "+-": int(np.sum((dx >= 0) & (dy < 0))),
        "++": int(np.sum((dx >= 0) & (dy >= 0))),
    }


def diagnose_seed(seed: int, args, device) -> dict:
    from utils.datasets import PushTRealPixelsDataset

    seed_dir = (args.output_root / f"seed_{seed:04d}").resolve()
    policy = ibc_policy.load_policy(
        seed_dir,
        device,
        ckpt_step=args.ckpt_step,
        dfo_overrides={
            "samples": args.dfo_samples,
            "iterations": args.dfo_iterations,
        },
    )
    ebm = policy.ebm
    fs = policy.frame_stack
    ckpt_name = policy.checkpoint.name

    ds = PushTRealPixelsDataset(
        archive_path=str(args.dataset), frame_stack=fs,
        camera_streams=tuple(policy.camera_streams), resize_hw=policy.image_hw,
        normalize_actions=True, action_norm_range=(-1.0, 1.0),
    )

    n = len(ds)
    k = min(args.num_samples, n)
    rng = np.random.default_rng(args.seed)
    idxs = rng.choice(n, size=k, replace=False)

    preds, gts, ents, pcts, repeat_stds = [], [], [], [], []
    for start in range(0, k, args.batch_size):
        chunk = idxs[start:start + args.batch_size]
        states = np.stack([ds[int(i)]["state"] for i in chunk])          # (b,C,H,W) uint8
        if args.zero_motion:
            # Channels are oldest->newest (datasets.py __getitem__); the newest
            # frame is the last per_frame channels. Duplicate it across every
            # slot so the stack has no inter-frame motion, mimicking a static
            # deploy observation.
            per_frame = states.shape[1] // fs
            newest = states[:, -per_frame:, :, :]
            states = np.tile(newest, (1, fs, 1, 1))
        gt = np.stack([ds[int(i)]["action"] for i in chunk])             # (b,2) normalized
        obs_u8 = torch.from_numpy(np.ascontiguousarray(states)).to(device)
        gt_t = torch.from_numpy(gt).float().to(device)

        runs = []
        for _ in range(max(1, args.dfo_repeats)):
            pred_t, initial_scores = ibc_policy.dfo_select(
                policy, obs_u8, return_initial_scores=True
            )
            runs.append(pred_t)
        if len(runs) > 1:
            stacked = torch.stack(runs)                      # (R, b, A)
            repeat_stds.append(stacked.std(dim=0).cpu().numpy())
        pred_t = runs[0]

        ent, pct = energy_health(ebm, obs_u8, gt_t, initial_scores)
        preds.append(pred_t.cpu().numpy())
        gts.append(gt)
        ents.append(ent.cpu().numpy())
        pcts.append(pct.cpu().numpy())

    pred = np.concatenate(preds).astype(np.float64)
    gt = np.concatenate(gts).astype(np.float64)
    ent = np.concatenate(ents).astype(np.float64)
    pct = np.concatenate(pcts).astype(np.float64)

    def col_stats(a):
        return {"mean": a.mean(axis=0).tolist(), "std": a.std(axis=0).tolist(),
                "frac_neg": (a < 0).mean(axis=0).tolist()}

    corr = float(np.corrcoef(pred[:, 0], pred[:, 1])[0, 1]) if pred.std() > 0 else float("nan")
    result = {
        "seed": seed, "samples": int(k), "checkpoint": ckpt_name,
        "dfo": dict(policy.dfo),
        "zero_motion": bool(args.zero_motion),
        "act_min": np.asarray(ds.act_min).tolist(), "act_max": np.asarray(ds.act_max).tolist(),
        "pred": col_stats(pred), "gt": col_stats(gt),
        "pred_corr_dx_dy": corr,
        "mae": np.abs(pred - gt).mean(axis=0).tolist(),
        "pred_quadrants": quadrant_hist(pred),
        "gt_quadrants": quadrant_hist(gt),
        "energy_entropy": {"mean": float(ent.mean()), "std": float(ent.std())},
        "expert_percentile": {"mean": float(pct.mean()), "std": float(pct.std())},
    }
    if repeat_stds:
        rs = np.concatenate(repeat_stds).astype(np.float64)
        result["dfo_repeat_std"] = {"mean": rs.mean(axis=0).tolist(),
                                    "repeats": int(args.dfo_repeats)}
    return result


def main() -> int:
    args = parse_args()
    device = torch.device(args.device if (torch.cuda.is_available() or args.device == "cpu")
                          else "cpu")
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    print(f"device={device}  dataset={args.dataset}  samples={args.num_samples}")
    print(f"checkpoints={args.output_root}  seeds={args.seeds}")
    if args.zero_motion:
        print("ZERO-MOTION mode: frame stack = newest frame duplicated "
              "(no inter-frame motion). If a healthy policy now collapses to "
              "near-fixed actions, a deploy runaway is an OOD static-stack "
              "artifact, not a checkpoint or orientation bug.")
    results = []
    for seed in args.seeds:
        print(f"\n===== seed {seed:04d} =====")
        try:
            r = diagnose_seed(seed, args, device)
        except Exception as exc:  # keep going across seeds
            print(f"seed {seed} FAILED: {exc}")
            results.append({"seed": seed, "error": str(exc)})
            continue
        results.append(r)
        p, g = r["pred"], r["gt"]
        print(f"  checkpoint     {r['checkpoint']}")
        print(f"  samples        {r['samples']}  "
              f"(DFO {r['dfo']['samples']}x{r['dfo']['iterations']})")
        print(f"  act range      min={r['act_min']}  max={r['act_max']}")
        print(f"  pred mean/std  mean={np.round(p['mean'],4)} std={np.round(p['std'],4)} "
              f"frac_neg={np.round(p['frac_neg'],3)}")
        print(f"  gt   mean/std  mean={np.round(g['mean'],4)} std={np.round(g['std'],4)} "
              f"frac_neg={np.round(g['frac_neg'],3)}")
        print(f"  corr(dx,dy)    {r['pred_corr_dx_dy']:.3f}   (≈1 => diagonal collapse)")
        print(f"  MAE vs gt      {np.round(r['mae'],4)}")
        print(f"  pred quadrants {r['pred_quadrants']}")
        print(f"  gt   quadrants {r['gt_quadrants']}")
        ent = r["energy_entropy"]["mean"]
        pctl = r["expert_percentile"]["mean"]
        print(f"  energy entropy {ent:.3f}   (≈1.0 => flat/random energy surface)")
        print(f"  expert pctile  {pctl:.3f}   (≈0.5 => EBM can't rank the expert action)")
        if "dfo_repeat_std" in r:
            print(f"  DFO repeat std {np.round(r['dfo_repeat_std']['mean'],4)} "
                  f"over {r['dfo_repeat_std']['repeats']} runs")
        collapse = (min(p["std"]) < 0.05) or (abs(r["pred_corr_dx_dy"]) > 0.95)
        # A flat energy surface is the IBC-specific failure: the action stats
        # can look healthy (uniform samples in, spread-out actions out) while
        # the policy is effectively random.
        unlearned = (ent > 0.99) or (pctl < 0.6)
        if collapse:
            verdict = "LIKELY COLLAPSED (ignores image)"
        elif unlearned:
            verdict = "ENERGY SURFACE NOT LEARNED (actions ~random despite healthy spread)"
        else:
            verdict = "looks responsive"
        print(f"  => {verdict}")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w") as fh:
        json.dump(results, fh, indent=2)
    print(f"\nWrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
