#!/usr/bin/env python3
"""Offline check: do the Push-T Diffusion-Policy checkpoints collapse?

DP counterpart of diagnose_pusht_actions.py (Q3C). Same bisection: run each DP
checkpoint over the TRAINING dataset and compare its sampled (normalized)
actions to the ground-truth teleop actions, to tell whether a robot failure
reproduces offline (data/training bug) or is a deploy-only artifact.

DP is a different model: it SAMPLES a fitted distribution instead of taking the
argmax of a learned energy. So beyond the Q3C metrics we also report, per obs,
the spread across independent samples (--repeats) — a policy that has genuinely
collapsed onto the (0,0) idle spike produces near-zero actions with near-zero
sampling spread, whereas a healthy DP keeps the demo's 2-D spread.

Per seed it reports, for each action dim:
  - sampled vs ground-truth mean/std and fraction negative,
  - corr(pred_dx, pred_dy)   (≈1 => the two dims move together => diagonal),
  - sampled std              (≈0 => mode collapse, ignores the image),
  - mean per-obs sampling std across --repeats draws,
  - MAE(pred, gt),
  - quadrant histogram of sampled actions (how many land in --,-+,+-,++),
  - zero-share: fraction of samples with |a| < idle_eps (the idle spike).

Both DDPM and DDIM are inference-time schedules over the SAME trained denoiser
(utils.diffusion.GaussianDiffusion). Pick with --sampler; DDIM sub-samples the
schedule (--ddim-steps, --ddim-eta), matching the deploy client's two schedules.

Run on the cluster (needs the project env with torch + tf for JPEG decode):

    sbatch scripts/diagnose_pusht_widowx_dp.sbatch
    # or directly:
    .venv/bin/python scripts/diagnose_pusht_actions_dp.py \
        --output-root checkpoints/pusht_real_dp --seeds 1
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

from utils.diffusion import build_diffusion, build_pixel_denoiser, resolve_dp_params


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--output-root", type=Path,
                   default=ROOT / "checkpoints" / "pusht_real_dp")
    p.add_argument("--seeds", type=int, nargs="+", default=[1])
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
    p.add_argument("--no-ema", action="store_true",
                   help="score the raw denoiser instead of the EMA copy "
                        "(deploy default is EMA).")
    p.add_argument("--sampler", default="ddpm", choices=["ddpm", "ddim"],
                   help="inference schedule over the same trained denoiser.")
    p.add_argument("--ddim-steps", type=int, default=None,
                   help="DDIM sub-sampled steps (default: ddim_eval_steps[0] "
                        "from the checkpoint's norm_stats).")
    p.add_argument("--ddim-eta", type=float, default=None,
                   help="DDIM stochasticity (default: ddim_eta from norm_stats, "
                        "usually 0 = deterministic).")
    p.add_argument("--repeats", type=int, default=4,
                   help="independent draws per obs for the sampling-spread "
                        "metric. The reported pred stats use the FIRST draw.")
    p.add_argument("--idle-eps", type=float, default=None,
                   help="|a| threshold for the zero-share metric. Default: "
                        "idle_eps from norm_stats, else 1e-3 (normalized).")
    p.add_argument("--zero-motion", action="store_true",
                   help="replace the real (t-1,t) frame stack with the newest "
                        "frame duplicated across all slots, so the obs carries "
                        "zero inter-frame motion. Tests whether a deploy-only "
                        "collapse is an OOD static-stack artifact rather than "
                        "image content.")
    p.add_argument("--seed", type=int, default=0, help="RNG seed for sampling")
    p.add_argument("--device", default="cuda")
    p.add_argument("--out", type=Path,
                   default=ROOT / "results" / "pusht_action_diagnostic_dp.json")
    return p.parse_args()


def load_run_config(seed_dir: Path) -> dict:
    with (seed_dir / "config.json").open() as fh:
        config = json.load(fh)
    return config["environments"][config["active_env"]]


def quadrant_hist(a: np.ndarray) -> dict:
    dx, dy = a[:, 0], a[:, 1]
    return {
        "--": int(np.sum((dx < 0) & (dy < 0))),
        "-+": int(np.sum((dx < 0) & (dy >= 0))),
        "+-": int(np.sum((dx >= 0) & (dy < 0))),
        "++": int(np.sum((dx >= 0) & (dy >= 0))),
    }


@torch.no_grad()
def sample_batch(diffusion, denoiser, obs_u8, args, ddim_steps, ddim_eta, cond=None):
    """One draw of the action per obs. obs_u8: (B,C,H,W) uint8 -> (B,2)."""
    if cond is not None:
        denoiser._cond = cond
    state = obs_u8.float()
    if args.sampler == "ddim":
        return diffusion.ddim_sample(denoiser, state, action_dim=2,
                                     num_steps=ddim_steps, eta=ddim_eta)
    return diffusion.ddpm_sample(denoiser, state, action_dim=2)


def diagnose_seed(seed: int, args, device) -> dict:
    seed_dir = (args.output_root / f"seed_{seed:04d}").resolve()
    env = load_run_config(seed_dir)
    norm_stats = torch.load(seed_dir / "norm_stats.pt", map_location="cpu",
                            weights_only=False)

    fs = int(norm_stats.get("frame_stack", env.get("frame_stack", 2)))
    hw = tuple(norm_stats.get("image_hw",
               (int(env.get("image_height", 240)), int(env.get("image_width", 320)))))
    in_channels = int(norm_stats["in_channels"])
    cond_dim = int(norm_stats.get("cond_dim", 0))
    enc_h = int(norm_stats.get("encoder_target_height", 180))
    enc_w = int(norm_stats.get("encoder_target_width", 240))

    # DP hyperparameters: resolve from the run config exactly as the trainer /
    # deploy client do, so the rebuilt denoiser matches the weights bit-for-bit.
    dp = resolve_dp_params(env)
    # norm_stats is the authority on what was actually trained; let it win.
    for k in ("num_train_timesteps", "beta_schedule", "prediction_type",
              "time_emb_dim", "denoiser_network_kind", "denoiser_width",
              "denoiser_depth"):
        if k in norm_stats:
            dp[k] = norm_stats[k]

    ddim_steps = args.ddim_steps
    if ddim_steps is None:
        ev = norm_stats.get("ddim_eval_steps", dp.get("ddim_eval_steps", [10]))
        ddim_steps = int(ev[0]) if ev else 10
    ddim_eta = args.ddim_eta
    if ddim_eta is None:
        ddim_eta = float(norm_stats.get("ddim_eta", dp.get("ddim_eta", 0.0)))
    idle_eps = args.idle_eps
    if idle_eps is None:
        idle_eps = float(norm_stats.get("idle_eps", env.get("idle_eps", 1e-3))) or 1e-3

    if cond_dim:
        # The DP batch (d01..d06) is pixels-only; conditioned DP would need the
        # CondPixelDiffusionDenoiser subclass from train_pusht_real_dp.py.
        raise NotImplementedError(
            f"cond_dim={cond_dim}: conditioned DP diagnosis not wired up "
            "(the pushtWidowXdp batch is pixels-only)."
        )
    denoiser = build_pixel_denoiser(
        2, in_channels, dp,
        encoder_target_height=enc_h, encoder_target_width=enc_w, device=device,
    )
    suffix = "" if args.no_ema else "_ema"
    weights = torch.load(seed_dir / f"denoiser{suffix}.pt", map_location=device,
                         weights_only=True)
    denoiser.load_state_dict(weights)
    denoiser.eval()
    diffusion = build_diffusion(dp, device, (-1.0, 1.0))

    # Use the archive the checkpoint TRAINED on (per its config), unless the
    # caller overrode it. Mirrors diagnose_pusht_actions.py.
    archive = str(args.dataset) if args.dataset is not None else env["data_archive"]
    data_format = str(norm_stats.get("data_format", env.get("data_format", "zarr_video")))
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
        cams = tuple(norm_stats.get("camera_streams", env.get("camera_streams", ["images1"])))
        ds = PushTRealPixelsDataset(
            archive_path=archive, frame_stack=fs,
            camera_streams=cams, resize_hw=hw,
            normalize_actions=True, action_norm_range=(-1.0, 1.0),
        )
    else:
        raise ValueError(f"Unknown data_format {data_format!r} (bridge_zip|zarr_video)")

    n = len(ds)
    k = min(args.num_samples, n)
    rng = np.random.default_rng(0)
    idxs = rng.choice(n, size=k, replace=False)
    torch.manual_seed(args.seed)

    preds, gts, spreads = [], [], []
    for start in range(0, k, args.batch_size):
        chunk = idxs[start:start + args.batch_size]
        items = [ds[int(i)] for i in chunk]
        states = np.stack([it["state"] for it in items])                 # (b,C,H,W) uint8
        if args.zero_motion:
            per_frame = states.shape[1] // fs
            newest = states[:, -per_frame:, :, :]
            states = np.tile(newest, (1, fs, 1, 1))
        gt = np.stack([it["action"] for it in items])                    # (b,2) normalized
        obs_u8 = torch.from_numpy(np.ascontiguousarray(states)).to(device)

        draws = []
        for _ in range(max(1, args.repeats)):
            a = sample_batch(diffusion, denoiser, obs_u8, args,
                             ddim_steps, ddim_eta).cpu().numpy()
            draws.append(a)
        draws = np.stack(draws)                                          # (R,b,2)
        preds.append(draws[0])                                           # first draw = the "pred"
        spreads.append(draws.std(axis=0).mean(axis=1))                   # (b,) per-obs spread
        gts.append(gt)
    pred = np.concatenate(preds).astype(np.float64)
    gt = np.concatenate(gts).astype(np.float64)
    spread = np.concatenate(spreads).astype(np.float64)

    def col_stats(a):
        return {"mean": a.mean(axis=0).tolist(), "std": a.std(axis=0).tolist(),
                "frac_neg": (a < 0).mean(axis=0).tolist()}

    corr = float(np.corrcoef(pred[:, 0], pred[:, 1])[0, 1]) if pred.std() > 0 else float("nan")
    zero_share = float((np.abs(pred).max(axis=1) < idle_eps).mean())
    gt_zero_share = float((np.abs(gt).max(axis=1) < idle_eps).mean())
    return {
        "seed": seed, "samples": int(k),
        "sampler": args.sampler, "ddim_steps": int(ddim_steps),
        "ddim_eta": float(ddim_eta), "repeats": int(max(1, args.repeats)),
        "prediction_type": dp.get("prediction_type"),
        "act_min": np.asarray(ds.act_min).tolist(), "act_max": np.asarray(ds.act_max).tolist(),
        "pred": col_stats(pred), "gt": col_stats(gt),
        "pred_corr_dx_dy": corr,
        "mae": np.abs(pred - gt).mean(axis=0).tolist(),
        "mean_sampling_spread": float(spread.mean()),
        "zero_share": zero_share, "gt_zero_share": gt_zero_share,
        "pred_quadrants": quadrant_hist(pred),
        "gt_quadrants": quadrant_hist(gt),
    }


def main() -> int:
    args = parse_args()
    device = torch.device(args.device if (torch.cuda.is_available() or args.device == "cpu")
                          else "cpu")
    print(f"device={device}  dataset={args.dataset}  samples={args.num_samples}  "
          f"sampler={args.sampler}")
    if args.zero_motion:
        print("ZERO-MOTION mode: frame stack = newest frame duplicated "
              "(no inter-frame motion). If a healthy policy now collapses to "
              "near-fixed actions, a deploy collapse is an OOD static-stack "
              "artifact, not a checkpoint bug.")
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
        print(f"  samples        {r['samples']}  ({r['sampler']}"
              f"{f' {r['ddim_steps']}step eta={r['ddim_eta']}' if r['sampler'] == 'ddim' else ''}, "
              f"{r['prediction_type']}-pred, {r['repeats']} draws)")
        print(f"  act range      min={r['act_min']}  max={r['act_max']}")
        print(f"  pred mean/std  mean={np.round(p['mean'],4)} std={np.round(p['std'],4)} "
              f"frac_neg={np.round(p['frac_neg'],3)}")
        print(f"  gt   mean/std  mean={np.round(g['mean'],4)} std={np.round(g['std'],4)} "
              f"frac_neg={np.round(g['frac_neg'],3)}")
        print(f"  corr(dx,dy)    {r['pred_corr_dx_dy']:.3f}   (≈1 => diagonal collapse)")
        print(f"  MAE vs gt      {np.round(r['mae'],4)}")
        print(f"  sampling spread {r['mean_sampling_spread']:.4f}   (≈0 => no diversity)")
        print(f"  zero-share     pred={r['zero_share']:.3f}  gt={r['gt_zero_share']:.3f}")
        print(f"  pred quadrants {r['pred_quadrants']}")
        print(f"  gt   quadrants {r['gt_quadrants']}")
        collapse = ((min(p['std']) < 0.05) or (abs(r['pred_corr_dx_dy']) > 0.95)
                    or (r['mean_sampling_spread'] < 0.02 and r['zero_share'] > 0.5))
        print(f"  => {'LIKELY COLLAPSED (ignores image / onto idle spike)' if collapse else 'looks responsive'}")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w") as fh:
        json.dump(results, fh, indent=2)
    print(f"\nWrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
