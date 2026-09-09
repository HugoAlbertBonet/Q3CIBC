"""Tier 2 multimodality eval: mode-coverage on IBC's BlockPushMultimodal task.

The task (2 blocks, 2 targets — see `pushing_multi_env.py`) is only
GENUINELY ambiguous on a subset of start configurations: whenever block0 is
roughly equidistant from target0 and target1 (and symmetrically for block1),
either block->target assignment is a similarly short solution, so the expert
demonstrations split ~50/50 between the two pairings for that geometry. Most
random start configs are NOT ambiguous — one assignment is obviously shorter.

This script does NOT need the trained policy to know which configs are
ambiguous: that's a property of the environment's reset geometry alone. So
for each eval seed we:
  1. Reset PushingMultiEnv(seed) and read the INITIAL block/target positions
     straight out of the observation (no policy involved) to compute an
     "ambiguity score" = |cost(pairing A) - cost(pairing B)|, pairing A/B
     being the two ways to assign 2 blocks to 2 targets. Low score = genuinely
     ambiguous start; high score = only one assignment makes sense.
  2. Reuse `hyperparam_search.evaluate_q3c` (already exercises the full
     checkpoint-loading + rollout path for Q3C / BC-MSE / plain-DP
     checkpoints uniformly) to get each seed's success/reward.
  3. Join ambiguity score with per-seed outcome and compare success rate on
     the ambiguous vs unambiguous subset, across one or more checkpoint dirs.

A policy that collapses multimodal demonstrations to their average action
(explicit BC-MSE is the textbook example) should show a MUCH lower success
rate specifically on the ambiguous subset, while a mode-preserving policy
(Q3C, diffusion, IBC) should hold up close to its unambiguous-subset rate.

Usage:
    python -m simulations.run_pushing_multi_multimodality_eval \\
        --checkpoint-dir checkpoints/pushing_multi_q3c --label Q3C \\
        --checkpoint-dir checkpoints/pushing_multi_bcmse --label BC-MSE \\
        --output-dir results/pushing_multi
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

import hyperparam_search as hs
from simulations.pushing_multi_env import PushingMultiEnv

CONFIG_PATH = Path(__file__).parent.parent / "config_json" / "config.json"


def _ambiguity_score(obs: np.ndarray) -> float:
    """|cost(A) - cost(B)| for the two block->target pairings. Lower = more
    ambiguous (both pairings similarly good, i.e. genuinely multimodal)."""
    block0 = obs[0:2]
    block1 = obs[3:5]
    target0 = obs[10:12]
    target1 = obs[13:15]
    cost_a = np.linalg.norm(block0 - target0) + np.linalg.norm(block1 - target1)
    cost_b = np.linalg.norm(block0 - target1) + np.linalg.norm(block1 - target0)
    return float(abs(cost_a - cost_b))


def compute_ambiguity_scores(env_config: dict, num_seeds: int) -> dict[int, float]:
    env = PushingMultiEnv(
        n_steps=env_config.get("max_episode_steps", 200),
        goal_dist_tolerance=env_config.get("goal_dist_tolerance", 0.04),
    )
    scores = {}
    for seed in range(num_seeds):
        obs, _ = env.reset(seed=seed)
        scores[seed] = _ambiguity_score(obs)
    env.close()
    return scores


def evaluate_checkpoint(checkpoint_dir: str, config: dict) -> list[dict]:
    """Return per-seed [{"seed", "success", "reward"}] via evaluate_q3c."""
    result = hs.evaluate_q3c(checkpoint_dir, config)
    if "error" in result and not result.get("per_seed"):
        raise RuntimeError(f"Evaluation failed for {checkpoint_dir}: {result['error']}")
    return result["per_seed"]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint-dir", action="append", required=True,
                         help="Checkpoint directory (repeatable, pair with --label).")
    parser.add_argument("--label", action="append", required=True,
                         help="Display label for the matching --checkpoint-dir.")
    parser.add_argument("--num-seeds", type=int, default=None,
                         help="Override env_config.num_eval_seeds.")
    parser.add_argument("--ambiguous-frac", type=float, default=0.25,
                         help="Fraction of seeds (by lowest ambiguity score) counted "
                              "as the 'ambiguous' subset (default: bottom quartile).")
    parser.add_argument("--output-dir", type=str, default="results/pushing_multi")
    args = parser.parse_args()

    if len(args.checkpoint_dir) != len(args.label):
        parser.error("--checkpoint-dir and --label must be given the same number of times")

    with open(CONFIG_PATH) as f:
        config = json.load(f)
    config["active_env"] = "pushing_multi"
    env_config = config["environments"]["pushing_multi"]

    num_seeds = args.num_seeds or int(env_config.get("num_eval_seeds", 100))
    print(f"Computing ambiguity scores for {num_seeds} seeds (env geometry only, no policy)...")
    ambiguity = compute_ambiguity_scores(env_config, num_seeds)

    scores_sorted = sorted(ambiguity.values())
    cutoff = scores_sorted[max(0, int(num_seeds * args.ambiguous_frac) - 1)]
    ambiguous_seeds = {s for s, v in ambiguity.items() if v <= cutoff}
    print(f"Ambiguous subset: {len(ambiguous_seeds)}/{num_seeds} seeds "
          f"(ambiguity score <= {cutoff:.4f})")

    os.makedirs(args.output_dir, exist_ok=True)
    csv_path = os.path.join(args.output_dir, "mode_coverage.csv")
    rows = []

    summary = []
    for ckpt_dir, label in zip(args.checkpoint_dir, args.label):
        print(f"\nEvaluating {label} ({ckpt_dir})...")
        config_for_eval = dict(config)
        config_for_eval["environments"] = dict(config["environments"])
        config_for_eval["environments"]["pushing_multi"] = dict(env_config)
        config_for_eval["environments"]["pushing_multi"]["num_eval_seeds"] = num_seeds
        per_seed = evaluate_checkpoint(ckpt_dir, config_for_eval)

        amb_success, amb_reward = [], []
        unamb_success, unamb_reward = [], []
        for r in per_seed:
            seed = r["seed"]
            score = ambiguity.get(seed, float("nan"))
            is_ambiguous = seed in ambiguous_seeds
            rows.append({
                "label": label, "seed": seed, "ambiguity_score": score,
                "ambiguous": is_ambiguous, "success": bool(r["success"]),
                "reward": r["reward"],
            })
            (amb_success if is_ambiguous else unamb_success).append(bool(r["success"]))
            (amb_reward if is_ambiguous else unamb_reward).append(r["reward"])

        amb_sr = float(np.mean(amb_success)) if amb_success else float("nan")
        unamb_sr = float(np.mean(unamb_success)) if unamb_success else float("nan")
        print(f"  Ambiguous-subset success rate:   {amb_sr:.1%}  (n={len(amb_success)})")
        print(f"  Unambiguous-subset success rate: {unamb_sr:.1%}  (n={len(unamb_success)})")
        print(f"  Gap (unambiguous - ambiguous):   {unamb_sr - amb_sr:+.1%}")
        summary.append({
            "label": label, "ambiguous_success_rate": amb_sr,
            "unambiguous_success_rate": unamb_sr, "gap": unamb_sr - amb_sr,
            "n_ambiguous": len(amb_success), "n_unambiguous": len(unamb_success),
        })

    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nPer-seed results written to {csv_path}")

    summary_path = os.path.join(args.output_dir, "mode_coverage_summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Summary written to {summary_path}")

    _plot(rows, summary, args.output_dir)


def _plot(rows: list[dict], summary: list[dict], output_dir: str) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    labels = [s["label"] for s in summary]
    amb = [s["ambiguous_success_rate"] for s in summary]
    unamb = [s["unambiguous_success_rate"] for s in summary]

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Left: grouped bar chart, success rate by ambiguity bucket per method.
    ax = axes[0]
    x = np.arange(len(labels))
    width = 0.35
    ax.bar(x - width / 2, amb, width, label="Ambiguous subset", color="#d95f5f")
    ax.bar(x + width / 2, unamb, width, label="Unambiguous subset", color="#5f9ed9")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("Success rate")
    ax.set_ylim(0, 1)
    ax.set_title("Success rate: ambiguous vs unambiguous start configs")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)

    # Right: scatter of ambiguity score vs success, jittered by method.
    ax2 = axes[1]
    colors = plt.cm.tab10(np.linspace(0, 1, len(labels)))
    for i, label in enumerate(labels):
        label_rows = [r for r in rows if r["label"] == label]
        scores = [r["ambiguity_score"] for r in label_rows]
        succ = [1.0 if r["success"] else 0.0 for r in label_rows]
        jitter = (np.random.default_rng(i).random(len(succ)) - 0.5) * 0.06
        ax2.scatter(scores, np.array(succ) + jitter, alpha=0.5, s=25,
                    color=colors[i], label=label)
    ax2.set_xlabel("Ambiguity score (0 = genuinely ambiguous start)")
    ax2.set_ylabel("Success (jittered)")
    ax2.set_title("Per-seed outcome vs start-config ambiguity")
    ax2.legend()
    ax2.grid(alpha=0.3)

    fig.tight_layout()
    plot_path = os.path.join(output_dir, "mode_coverage.png")
    fig.savefig(plot_path, dpi=150)
    plt.close(fig)
    print(f"Plot written to {plot_path}")


if __name__ == "__main__":
    main()
