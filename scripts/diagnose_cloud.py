#!/usr/bin/env python3
"""Why does ranking the candidate cloud not help? Three measurements.

Re-scoring 18 dpq3c checkpoints showed that ranking a 64-candidate cloud with
the Q estimator is neutral-to-harmful (paired 0.819 with ranking vs 0.834
without; worse on 12 of 18; -0.027 at p=0.046 on the strongest group). There are
two very different explanations, and they call for opposite fixes:

  A. THE CLOUD HAS NO DIVERSITY. With DDIM at eta=0 the only thing separating
     candidates is the initial noise, and a well-trained denoiser may map most
     of it to nearly the same action. If so, cloud=1 and cloud=64 are the same
     policy by construction, no critic could ever have helped, and the fix is
     sampler diversity (eta > 0, ancestral DDPM, fewer steps) rather than a
     better critic.

  B. THE CRITIC CANNOT RANK. It is trained by InfoNCE to separate the expert
     from negatives, but at deploy every candidate is a plausible sample from
     the actor, so the task is "which plausible action is better" — a question
     it was never trained on. Note `progress_weight` cannot fix this: it
     constrains Q only AT THE EXPERT ACTION, so it sets an absolute scale but
     says nothing about the ordering of two candidates.

And one number that decides whether any of it is worth fixing:

  C. THE ORACLE GAP. Action-MAE when you pick the BEST candidate in the cloud,
     versus the critic's pick, versus a random pick. This is the ceiling on what
     any critic could buy. If the oracle barely beats random, ranking is a dead
     end regardless of the critic, and the honest result is that dpq3c is a
     diffusion policy.

Everything runs on held-out demonstration transitions. No simulator, no rollouts.

    uv run --extra libero python scripts/diagnose_cloud.py \
        --active-env libero_goal_pixels --trials 21 78 79 80
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--active-env", required=True)
    ap.add_argument("--script", default="dpq3c_training.py")
    ap.add_argument("--trials", type=int, nargs="+", required=True,
                    help="trial ids whose checkpoints to diagnose")
    ap.add_argument("--cloud", type=int, default=64,
                    help="candidates drawn per state (default: the trained-against 64)")
    ap.add_argument("--dp-iters", type=int, default=None,
                    help="denoising steps (default: the run's inference_dp_iters)")
    ap.add_argument("--dp-method", default=None, choices=["ddim", "ddpm"])
    ap.add_argument("--dp-eta", type=float, default=None,
                    help="DDIM eta. Raise it to test whether the cloud is too "
                         "homogeneous — that is hypothesis A.")
    ap.add_argument("--batches", type=int, default=20, help="held-out batches to average over")
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--no-ema", action="store_true")
    ap.add_argument("--device", default="cuda")
    return ap.parse_args()


def spearman(a: torch.Tensor, b: torch.Tensor) -> float:
    """Rank correlation, computed per-row then averaged. a, b: (B, N)."""
    def ranks(x):
        return torch.argsort(torch.argsort(x, dim=1), dim=1).float()
    ra, rb = ranks(a), ranks(b)
    ra = ra - ra.mean(dim=1, keepdim=True)
    rb = rb - rb.mean(dim=1, keepdim=True)
    num = (ra * rb).sum(dim=1)
    den = ra.norm(dim=1) * rb.norm(dim=1) + 1e-12
    return float((num / den).mean())


def main() -> int:
    args = parse_args()
    import hyperparam_search as hs

    trials_path = hs._trials_path(args.script, active_env=args.active_env)
    records = {}
    for line in open(trials_path):
        if line.strip():
            r = json.loads(line)
            records[int(r.get("trial_id", -1))] = r

    runs = []
    for tid in args.trials:
        rec = records.get(tid)
        if rec is None:
            print(f"trial #{tid}: not found in {trials_path}")
            continue
        cfg_path = Path(rec["checkpoint_dir"]) / "config.json"
        if not cfg_path.exists():
            print(f"trial #{tid}: per-run config missing at {cfg_path}")
            continue
        runs.append((tid, rec, json.load(open(cfg_path))))
    if not runs:
        raise SystemExit("no usable trials")

    # Dataset is built ONCE from the first run's config. Every run in a batch
    # shares the data pipeline; assert that rather than assume it, because a
    # mismatch would silently compare clouds against the wrong expert actions.
    _, _, cfg0 = runs[0]
    dkeys = ("frame_stack", "image_crop_size", "action_chunk", "max_demos_per_task")
    def dsig(c):
        e = c["environments"][args.active_env]
        t = e.get("training", {})
        return tuple(str(e.get(k, t.get(k))) for k in dkeys)
    for tid, _, c in runs[1:]:
        if dsig(c) != dsig(cfg0):
            raise SystemExit(f"trial #{tid} has a different data pipeline than "
                             f"trial #{runs[0][0]}; diagnose them separately")

    os.environ["Q3C_CONFIG_PATH"] = str(Path(runs[0][1]["checkpoint_dir"]) / "config.json")
    from combinedv2_cpascounter_training import load_dataset
    dataset = load_dataset()
    print(f"dataset: {len(dataset)} transitions, action_shape={dataset.action_shape}")

    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")
    loader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size,
                                         shuffle=True, num_workers=0)
    # Action scale, for normalizing the diversity number into "fraction of the
    # usable action range" — a raw L2 distance means nothing on its own.
    act_range = float(np.mean(np.asarray(dataset.act_max) - np.asarray(dataset.act_min)))

    from utils.models import PixelQEstimator
    print()
    for tid, rec, cfg in runs:
        env_cfg = cfg["environments"][args.active_env]
        em = env_cfg.get("model", {})
        ckpt = Path(rec["checkpoint_dir"])
        ns = torch.load(ckpt / "norm_stats.pt", map_location="cpu", weights_only=False)
        suffix = "" if args.no_ema else "_ema"

        # Eval-time sampler overrides, so hypothesis A can be probed directly.
        et = env_cfg.setdefault("training", {})
        if args.dp_iters is not None:
            et["inference_dp_iters"] = args.dp_iters
        if args.dp_method is not None:
            et["inference_dp_method"] = args.dp_method
        if args.dp_eta is not None:
            et["inference_dp_eta"] = args.dp_eta
        et["inference_control_points"] = args.cloud

        action_dim = int(np.asarray(ns["act_min"]).size)
        cond_dim = int(ns.get("cond_dim", 0))
        in_ch = int(ns["in_channels"])
        cp_gen = hs._build_dpq3c_generator(
            ckpt / f"denoiser{suffix}.pt", env_cfg, ns, action_dim, args.cloud,
            (-1.0, 1.0), device, pixel=True, in_channels=in_ch, cond_dim=cond_dim,
            encoder_target_height=int(env_cfg.get("encoder_target_height", 128)),
            encoder_target_width=int(env_cfg.get("encoder_target_width", 128)),
            encoder_feature_dim=int(ns.get("encoder_feature_dim", 256)),
            encoder_kind=str(ns.get("encoder_kind", "conv_maxpool")),
            encoder_num_kp=int(ns.get("encoder_num_kp", 64)),
            encoder_norm_kind=str(ns.get("encoder_norm_kind", "bn")),
            encoder_per_camera=bool(ns.get("encoder_per_camera", False)))
        q = PixelQEstimator(
            action_dim=action_dim, in_channels=in_ch,
            encoder_target_height=int(env_cfg.get("encoder_target_height", 128)),
            encoder_target_width=int(env_cfg.get("encoder_target_width", 128)),
            value_width=int(em.get("value_width", 1024)),
            value_num_blocks=int(em.get("value_num_blocks", 1)),
            cond_dim=cond_dim,
            encoder_kind=str(ns.get("encoder_kind", "conv_maxpool")),
            encoder_pretrained=False,
            encoder_num_kp=int(ns.get("encoder_num_kp", 64)),
            encoder_norm_kind=str(ns.get("encoder_norm_kind", "bn")),
            encoder_per_camera=bool(ns.get("encoder_per_camera", False)),
            cond_fusion=str(ns.get("cond_fusion", "concat")),
            goal_dim=int(ns.get("goal_emb_dim", 0)))
        q.load_state_dict(torch.load(ckpt / f"q_estimator{suffix}.pt",
                                     map_location=device, weights_only=True))
        q.to(device).eval()

        acc = {k: [] for k in ("spread", "sd", "q_pick_best", "rho",
                               "mae_oracle", "mae_q", "mae_rand", "mae_mean")}
        with torch.no_grad():
            for i, batch in enumerate(loader):
                if i >= args.batches:
                    break
                s = batch["state"].float().to(device)
                a = batch["action"].float().to(device)
                cond = batch["cond"].float().to(device) if "cond" in batch else None
                if cond is not None:
                    cp_gen._cond = cond
                    q._cond = cond
                cloud = cp_gen(s)                               # (B, N, A)
                B, N, A = cloud.shape

                # (A) diversity: mean pairwise distance inside the cloud
                d = torch.cdist(cloud, cloud)                   # (B, N, N)
                off = d.sum(dim=(1, 2)) / (N * (N - 1))
                acc["spread"].append(float(off.mean()))
                acc["sd"].append(float(cloud.std(dim=1).mean()))

                # (B) can the critic rank? compare against distance-to-expert
                feats = q.encode(s)
                qv = q.score(feats, cloud).squeeze(-1)          # (B, N)
                dist = (cloud - a.unsqueeze(1)).norm(dim=-1)    # (B, N)
                best = dist.argmin(dim=1)
                acc["q_pick_best"].append(float((qv.argmax(dim=1) == best).float().mean()))
                acc["rho"].append(spearman(qv, -dist))

                # (C) the oracle gap, in action-MAE
                idx = torch.arange(B, device=device)
                pick = lambda j: (cloud[idx, j] - a).abs().mean()
                acc["mae_oracle"].append(float(pick(best)))
                acc["mae_q"].append(float(pick(qv.argmax(dim=1))))
                acc["mae_rand"].append(float(pick(torch.randint(0, N, (B,), device=device))))
                acc["mae_mean"].append(float((cloud.mean(dim=1) - a).abs().mean()))

        m = {k: float(np.mean(v)) for k, v in acc.items()}
        chance = 1.0 / args.cloud
        print(f"── trial {tid}  (reported success {rec.get('success_rate')})  "
              f"cloud={args.cloud} steps={et.get('inference_dp_iters')} "
              f"method={et.get('inference_dp_method','ddim')} eta={et.get('inference_dp_eta',0.0)}")
        print(f"   A  cloud spread   mean pairwise dist {m['spread']:.4f} "
              f"= {100*m['spread']/act_range:.1f}% of the action range   "
              f"(per-dim sd {m['sd']:.4f})")
        print(f"   B  rank quality   Q picks the closest candidate "
              f"{m['q_pick_best']:.3f} vs chance {chance:.3f}   "
              f"spearman(Q, -dist) = {m['rho']:+.3f}")
        print(f"   C  oracle gap     MAE  oracle {m['mae_oracle']:.4f} | "
              f"Q {m['mae_q']:.4f} | random {m['mae_rand']:.4f} | "
              f"cloud-mean {m['mae_mean']:.4f}")
        head = m["mae_rand"] - m["mae_oracle"]
        got = m["mae_rand"] - m["mae_q"]
        print(f"      headroom (random-oracle) = {head:.4f};  "
              f"critic captures {100*got/head if head > 1e-9 else 0:.1f}% of it\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
