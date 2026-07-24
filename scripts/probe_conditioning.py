"""E1-efficacy probe: is the c09/c10 conditioning actually steering the action,
or decorative?

Holds the image observation FIXED and sweeps the conditioning (normalized EEF
x/y) across the whole training workspace, measuring how much the predicted
action moves. If the action barely responds to a full-workspace cond sweep, the
cond head is weak/mis-wired and the "conditioning" is not really driving the
policy -- which would explain why c09/c10 still stall/orbit despite valid,
in-distribution cond inputs (E1 layout was already verified correct).

Runs on the machine that has the checkpoint (e.g. the widowx box), reusing the
exact deploy model build + selection. Sources real observations from a rollout's
raw frames so the images are in-distribution.

Usage:
    python scripts/probe_conditioning.py \
        --seed-dir checkpoints/pusht_new/c09_condxy_dropzero \
        --log-dir results/roll_c09_base --device cpu --frames 0 40 80 160
"""
import argparse
import collections
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))  # find deploy module
import deploy_pusht_real as deploy


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--seed-dir", type=Path, required=True)
    p.add_argument("--log-dir", type=Path, required=True,
                   help="rollout dir with raw/*.npy to source real observations")
    p.add_argument("--device", default="cpu")
    p.add_argument("--frames", type=int, nargs="+", default=[0, 40, 80, 160],
                   help="which raw frame indices to probe")
    p.add_argument("--grid", type=int, default=5,
                   help="cond sweep resolution per axis over [-1,1]")
    p.add_argument("--no-ema", action="store_true")
    args = p.parse_args()
    device = torch.device(args.device)

    seed_dir = args.seed_dir.resolve()
    env = deploy.load_run_config(seed_dir)
    norm_stats = torch.load(seed_dir / "norm_stats.pt", map_location="cpu",
                            weights_only=False)
    cond_dim = int(norm_stats.get("cond_dim", 0))
    if cond_dim != 2:
        raise SystemExit(f"checkpoint cond_dim={cond_dim}, not a conditioned run")
    cmin = np.asarray(norm_stats["cond_min"], np.float64)
    cmax = np.asarray(norm_stats["cond_max"], np.float64)
    act_min = np.asarray(norm_stats["act_min"], np.float64)
    act_max = np.asarray(norm_stats["act_max"], np.float64)
    norm_range = tuple(env.get("action_norm_range", (-1.0, 1.0)))
    fs = int(env.get("frame_stack", 2))
    H = int(env.get("image_height", 240)); W = int(env.get("image_width", 320))

    cp_gen, q_net = deploy.build_models(env, 3 * fs, device, cond_dim=cond_dim)
    suffix = "" if args.no_ema else "_ema"
    deploy.load_weights(cp_gen, seed_dir / f"control_point_generator{suffix}.pt", device)
    deploy.load_weights(q_net, seed_dir / f"q_estimator{suffix}.pt", device)
    cp_sel = str(env.get("cp_selection", "argmax")); cp_temp = float(env.get("cp_temperature", 1.0))

    raws = sorted((args.log_dir / "raw").glob("*.npy"))
    if not raws:
        raise SystemExit(f"no raw frames in {args.log_dir}/raw")

    def obs_at(idx):
        """Build the frame_stack obs the model saw at step idx (preprocess+stack)."""
        buf = collections.deque(maxlen=fs)
        # fill with the fs frames ending at idx (mirrors deploy's rolling buffer)
        for j in range(max(0, idx - fs + 1), idx + 1):
            buf.append(deploy.preprocess(np.load(raws[j]), (H, W)))
        while len(buf) < fs:
            buf.appendleft(buf[0])
        return deploy.stack_to_tensor(buf, device)

    def act_for(obs_u8, cond_norm):
        cond = torch.tensor(cond_norm, dtype=torch.float32, device=device).view(1, 2) \
            if cond_norm is not None else None
        na = deploy.select_action(cp_gen, q_net, obs_u8, cp_sel, cp_temp, cond=cond)
        return deploy.unnormalize(na, act_min, act_max, norm_range).ravel()[:2]

    gs = np.linspace(-1, 1, args.grid)
    print(f"cond workspace: x[{cmin[0]:.3f},{cmax[0]:.3f}]  y[{cmin[1]:.3f},{cmax[1]:.3f}]")
    print(f"action scale (|act_max|): {np.abs(act_max)}  (m)\n")
    ranges = []
    for idx in args.frames:
        if idx >= len(raws):
            continue
        obs = obs_at(idx)
        acts = np.array([[act_for(obs, (cx, cy)) for cx in gs] for cy in gs])  # (g,g,2)
        dx, dy = acts[..., 0], acts[..., 1]
        rng = np.array([dx.max() - dx.min(), dy.max() - dy.min()])
        ranges.append(rng)
        # directional: action along the x-cond axis (y-cond=0)
        mid = args.grid // 2
        xline = np.array([act_for(obs, (cx, 0.0)) for cx in [-1, 0, 1]])
        print(f"frame {idx:4d}: cond-induced action RANGE dx={rng[0]*1000:6.2f}mm "
              f"dy={rng[1]*1000:6.2f}mm  (of ±{act_max[0]*1000:.1f}mm)")
        print(f"           dx vs cond_x(-1,0,+1): {np.round(xline[:,0]*1000,2)} mm   "
              f"dy: {np.round(xline[:,1]*1000,2)} mm")
    R = np.array(ranges)
    med = np.median(R, axis=0)
    scale = np.abs(act_max)
    print(f"\nmedian cond-induced action range: dx={med[0]*1000:.2f}mm "
          f"dy={med[1]*1000:.2f}mm = {100*med/ (2*scale)}% of full action span")
    frac = (med / (2 * scale)).mean()

    # ---- MIRROR PROBE: image sensitivity at FIXED cond -------------------
    # Vary the image across the sampled frames while holding cond constant;
    # if the action barely moves, the policy is visually blind / over-conditioned.
    print("\n=== IMAGE sensitivity (cond held fixed at center 0,0) ===")
    print("    (frames should have DIFFERENT T positions for this to be "
          "meaningful; similar images -> small range is expected, not proof)")
    fidx = [i for i in args.frames if i < len(raws)]
    obses = [obs_at(i) for i in fidx]
    acts = np.array([act_for(o, (0.0, 0.0)) for o in obses])
    img_rng = np.array([acts[:, 0].max() - acts[:, 0].min(),
                        acts[:, 1].max() - acts[:, 1].min()])
    for i, a in zip(fidx, acts):
        print(f"  frame {i:4d} @cond(0,0): action dx={a[0]*1000:+6.2f} dy={a[1]*1000:+6.2f} mm")
    print(f"  image-induced action range: dx={img_rng[0]*1000:.2f}mm "
          f"dy={img_rng[1]*1000:.2f}mm = {np.round(100*img_rng/(2*scale),1)}% of span")
    img_frac = (img_rng / (2 * scale)).mean()
    print(f"\ncond-vs-image dominance: cond {100*frac:.0f}% vs image {100*img_frac:.0f}% "
          f"of action span")
    if img_frac < 0.25 and frac > 2 * img_frac:
        print("  -> OVER-CONDITIONED: action is driven by eef(x,y), nearly blind to "
              "the T. Drives to a position attractor -> stall/orbit regardless of "
              "the T. Fix = rebalance (down-weight cond / augment it out / more "
              "visual capacity), not deploy tweaks.")
    else:
        print("  -> image also matters; not purely over-conditioned.")

    if frac < 0.10:
        print("VERDICT: WEAK — action barely moves across the whole cond workspace. "
              "Conditioning is near-decorative; c09/c10 are effectively pixels-only.")
    elif frac < 0.35:
        print("VERDICT: MODEST — cond shifts the action somewhat but may be too "
              "weak to dominate; check if it points the right way.")
    else:
        print("VERDICT: STRONG — cond materially steers the action. Conditioning "
              "works; the stall is elsewhere (policy quality / OOD at contact).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
