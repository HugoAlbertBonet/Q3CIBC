#!/usr/bin/env python3
"""Offline check: do the Push-T checkpoints collapse to a fixed action?

On the real robot, seed_0011 produced only (-,-) actions (a straight diagonal).
This runs each checkpoint over the TRAINING dataset and compares its predicted
(normalized) actions to the ground-truth teleop actions, to tell whether the
collapse is in the policy itself (reproduces offline) or a deploy-only artifact.

Per seed it reports, for each action dim:
  - predicted vs ground-truth mean/std and fraction negative,
  - corr(pred_dx, pred_dy)   (≈1 => the two dims move together => diagonal),
  - predicted std             (≈0 => mode collapse, ignores the image),
  - MAE(pred, gt),
  - quadrant histogram of predicted actions (how many land in --,-+,+-,++).

Run on the cluster (needs the project env with torch + tf for JPEG decode):

    sbatch scripts/diagnose_pusht_actions.sbatch
    # or directly:
    .venv/bin/python scripts/diagnose_pusht_actions.py \
        --output-root checkpoints/pusht_real_combinedv2 --seeds 11 29 47 83
"""

from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
import sys
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# Reuse the exact model build + weight load from the deploy client so this
# diagnostic and the robot run agree bit-for-bit.
_spec = importlib.util.spec_from_file_location("deploy", ROOT / "scripts" / "deploy_pusht_real.py")
deploy = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(deploy)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--output-root", type=Path,
                   default=ROOT / "checkpoints" / "pusht_real_combinedv2")
    p.add_argument("--seeds", type=int, nargs="+", default=[11, 29, 47, 83])
    p.add_argument("--dataset", type=Path, default=None,
                   help="override the demo archive. Default: the exact archive "
                        "the checkpoint trained on (data_archive in its "
                        "config.json), so bridge_zip vs zarr_video is picked "
                        "automatically per checkpoint.")
    p.add_argument("--idle-filter", default="none",
                   choices=["none", "drop_zero", "drop_static", "subsample"],
                   help="zarr_video only: how to treat idle (~0) transitions "
                        "when scoring. Default 'none' evaluates the FULL action "
                        "distribution so you can see whether the policy still "
                        "emits the zero spike, regardless of how it was trained.")
    p.add_argument("--num-samples", type=int, default=3000,
                   help="random transitions sampled from the dataset")
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--no-ema", action="store_true")
    p.add_argument("--zero-motion", action="store_true",
                   help="replace the real (t-1,t) frame stack with the newest "
                        "frame duplicated across all slots, so the obs carries "
                        "zero inter-frame motion. Tests whether the deploy-only "
                        "(-,-) collapse is caused by out-of-distribution "
                        "static/near-static stacks rather than image content.")
    p.add_argument("--device", default="cuda")
    p.add_argument("--out", type=Path, default=ROOT / "results" / "pusht_action_diagnostic.json")
    p.add_argument("--dump-arrays", type=Path, default=None,
                   help="if set, save raw per-sample pred/gt arrays as "
                        "<dir>/<tag>_seed<seed>.npz (keys: pred, gt) for "
                        "offline histogramming. Uses the same seed-0 sample.")
    return p.parse_args()


def load_run_config(seed_dir: Path) -> dict:
    with (seed_dir / "config.json").open() as fh:
        config = json.load(fh)
    return config["environments"][config["active_env"]]


@torch.no_grad()
def predict_batch(cp_gen, q_net, obs_u8, cond=None):
    """CP-cloud argmax over a batch. obs_u8: (B, C, H, W) uint8 -> (B, 2).

    `cond` is the (B, cond_dim) conditioning batch for EEF-conditioned
    checkpoints; both nets read it off `_cond` exactly as the trainer sets it.
    """
    cp_gen._cond = cond
    q_net._cond = cond
    feats = q_net.encode(obs_u8)
    cps = cp_gen(obs_u8)                                  # (B, P, 2)
    logits = q_net.score(feats, cps).squeeze(-1)         # (B, P)
    idx = logits.argmax(dim=1)                           # (B,)
    return cps[torch.arange(cps.shape[0], device=cps.device), idx]  # (B, 2)


def quadrant_hist(a: np.ndarray) -> dict:
    dx, dy = a[:, 0], a[:, 1]
    return {
        "--": int(np.sum((dx < 0) & (dy < 0))),
        "-+": int(np.sum((dx < 0) & (dy >= 0))),
        "+-": int(np.sum((dx >= 0) & (dy < 0))),
        "++": int(np.sum((dx >= 0) & (dy >= 0))),
    }


def diagnose_seed(seed: int, args, device) -> dict:
    seed_dir = (args.output_root / f"seed_{seed:04d}").resolve()
    env = load_run_config(seed_dir)
    fs = int(env.get("frame_stack", 2))
    hw = (int(env.get("image_height", 240)), int(env.get("image_width", 320)))

    # norm_stats carries the conditioning schema (cond_dim) — needed to rebuild
    # the nets with the right head width and to know whether to feed `cond`.
    norm_stats = torch.load(seed_dir / "norm_stats.pt", map_location="cpu",
                            weights_only=False)
    cond_dim = int(norm_stats.get("cond_dim", 0))

    # Use the archive the checkpoint TRAINED on (per its config), unless the
    # caller overrode it. This is what makes bridge_zip vs zarr_video automatic.
    archive = str(args.dataset) if args.dataset is not None else env["data_archive"]
    data_format = str(env.get("data_format", "bridge_zip"))

    if data_format == "zarr_video":
        from utils.datasets import PushTWidowXVideoDataset
        ds = PushTWidowXVideoDataset(
            archive_path=archive, frame_stack=fs,
            camera=int(env.get("video_camera", 1)), resize_hw=hw,
            normalize_actions=True, action_norm_range=(-1.0, 1.0),
            idle_filter=args.idle_filter,
            idle_eps=float(env.get("idle_eps", 0.0)),
            idle_move_eps=float(env.get("idle_move_eps", 1e-4)),
            idle_keep_frac=float(env.get("idle_keep_frac", 0.25)),
            cache_dir=env.get("frame_cache_dir"),
            cond_eef_xy=bool(env.get("cond_eef_xy", False)),
        )
    elif data_format == "bridge_zip":
        from utils.datasets import PushTRealPixelsDataset
        cams = tuple(env.get("camera_streams", ["images1"]))
        ds = PushTRealPixelsDataset(
            archive_path=archive, frame_stack=fs,
            camera_streams=cams, resize_hw=hw,
            normalize_actions=True, action_norm_range=(-1.0, 1.0),
        )
    else:
        raise ValueError(f"Unknown data_format {data_format!r} (bridge_zip|zarr_video)")

    in_channels = ds.state_shape[0]
    cp_gen, q_net = deploy.build_models(env, in_channels, device, cond_dim=cond_dim)
    suffix = "" if args.no_ema else "_ema"
    deploy.load_weights(cp_gen, seed_dir / f"control_point_generator{suffix}.pt", device)
    deploy.load_weights(q_net, seed_dir / f"q_estimator{suffix}.pt", device)

    n = len(ds)
    k = min(args.num_samples, n)
    rng = np.random.default_rng(0)
    idxs = rng.choice(n, size=k, replace=False)

    preds, gts = [], []
    for start in range(0, k, args.batch_size):
        chunk = idxs[start:start + args.batch_size]
        items = [ds[int(i)] for i in chunk]
        states = np.stack([it["state"] for it in items])                 # (b,C,H,W) uint8
        if args.zero_motion:
            # Channels are oldest->newest (datasets.py __getitem__); the newest
            # frame is the last per_frame channels. Duplicate it across every
            # slot so the stack has no inter-frame motion, mimicking a static
            # deploy observation.
            per_frame = states.shape[1] // fs
            newest = states[:, -per_frame:, :, :]
            states = np.tile(newest, (1, fs, 1, 1))
        gt = np.stack([it["action"] for it in items])                    # (b,2) normalized
        obs_u8 = torch.from_numpy(np.ascontiguousarray(states)).to(device)
        cond = None
        if cond_dim:
            cond = torch.from_numpy(
                np.stack([it["cond"] for it in items]).astype(np.float32)
            ).to(device)
        pred = predict_batch(cp_gen, q_net, obs_u8, cond=cond).cpu().numpy()
        preds.append(pred)
        gts.append(gt)
    pred = np.concatenate(preds).astype(np.float64)
    gt = np.concatenate(gts).astype(np.float64)

    def col_stats(a):
        return {"mean": a.mean(axis=0).tolist(), "std": a.std(axis=0).tolist(),
                "frac_neg": (a < 0).mean(axis=0).tolist()}

    corr = float(np.corrcoef(pred[:, 0], pred[:, 1])[0, 1]) if pred.std() > 0 else float("nan")
    if args.dump_arrays is not None:
        args.dump_arrays.mkdir(parents=True, exist_ok=True)
        # seed_dir may be a symlink to the real tag dir; name the dump after it.
        tag = seed_dir.resolve().name
        np.savez(args.dump_arrays / f"{tag}_seed{seed:04d}.npz", pred=pred, gt=gt)
    return {
        "seed": seed, "samples": int(k),
        "act_min": np.asarray(ds.act_min).tolist(), "act_max": np.asarray(ds.act_max).tolist(),
        "pred": col_stats(pred), "gt": col_stats(gt),
        "pred_corr_dx_dy": corr,
        "mae": np.abs(pred - gt).mean(axis=0).tolist(),
        "pred_quadrants": quadrant_hist(pred),
        "gt_quadrants": quadrant_hist(gt),
    }


def main() -> int:
    args = parse_args()
    device = torch.device(args.device if (torch.cuda.is_available() or args.device == "cpu")
                          else "cpu")
    print(f"device={device}  dataset={args.dataset}  samples={args.num_samples}")
    if args.zero_motion:
        print("ZERO-MOTION mode: frame stack = newest frame duplicated "
              "(no inter-frame motion). If a healthy policy now collapses to "
              "-- / near-fixed actions, the deploy (-,-) runaway is an OOD "
              "static-stack artifact, not a checkpoint or orientation bug.")
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
        print(f"  samples        {r['samples']}")
        print(f"  act range      min={r['act_min']}  max={r['act_max']}")
        print(f"  pred mean/std  mean={np.round(p['mean'],4)} std={np.round(p['std'],4)} "
              f"frac_neg={np.round(p['frac_neg'],3)}")
        print(f"  gt   mean/std  mean={np.round(g['mean'],4)} std={np.round(g['std'],4)} "
              f"frac_neg={np.round(g['frac_neg'],3)}")
        print(f"  corr(dx,dy)    {r['pred_corr_dx_dy']:.3f}   (≈1 => diagonal collapse)")
        print(f"  MAE vs gt      {np.round(r['mae'],4)}")
        print(f"  pred quadrants {r['pred_quadrants']}")
        print(f"  gt   quadrants {r['gt_quadrants']}")
        collapse = (min(p['std']) < 0.05) or (abs(r['pred_corr_dx_dy']) > 0.95)
        print(f"  => {'LIKELY COLLAPSED (ignores image)' if collapse else 'looks responsive'}")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w") as fh:
        json.dump(results, fh, indent=2)
    print(f"\nWrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
