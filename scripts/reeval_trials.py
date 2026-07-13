"""Re-evaluate saved Q3C checkpoints whose EVAL failed (training succeeded).

Used after eval-side bug fixes (e.g. the Dstandardlibero frame_stack=2 rebuild
bug) so 20h+ trainings don't have to re-run. For each requested trial id:
  1. read its record from the env's trials.jsonl (checkpoint_dir),
  2. load the per-run config saved next to the checkpoint,
  3. call hyperparam_search.evaluate_q3c on it,
  4. append a NEW corrected record (note: "reeval of trial #N") to trials.jsonl.

Run on a GPU compute node (render eval): MUJOCO_GL=egl.

    uv run --extra libero python scripts/reeval_trials.py \
        --active-env libero_goal_pixels --trials 6 9 10
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import hyperparam_search as hs  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--active-env", required=True)
    ap.add_argument("--script", default="combinedv2_cpascounter_training.py")
    ap.add_argument("--trials", type=int, nargs="+", required=True,
                    help="trial ids (from trials.jsonl) to re-evaluate")
    ap.add_argument("--num-eval-seeds", type=int, default=None,
                    help="override eval episode count (e.g. 500 for final paper "
                         "numbers; default = the per-run config's value)")
    args = ap.parse_args()

    trials_path = (
        hs.RESULTS_BASE_DIR / Path(args.script).stem / args.active_env / "trials.jsonl"
    )
    records = {}
    for line in open(trials_path):
        line = line.strip()
        if not line:
            continue
        r = json.loads(line)
        records[int(r.get("trial_id", -1))] = r

    for tid in args.trials:
        rec = records.get(tid)
        if rec is None:
            print(f"trial #{tid}: NOT FOUND in {trials_path}")
            continue
        ckpt_dir = rec.get("checkpoint_dir")
        cfg_path = Path(ckpt_dir) / "config.json"
        if not cfg_path.exists():
            print(f"trial #{tid}: per-run config missing at {cfg_path}; skipping")
            continue
        with open(cfg_path) as f:
            config = json.load(f)
        if args.num_eval_seeds:
            config["environments"][args.active_env]["num_eval_seeds"] = int(args.num_eval_seeds)
        print(f"\n=== re-eval trial #{tid} ({ckpt_dir}) ===")
        try:
            eval_results = hs.evaluate_q3c(ckpt_dir, config)
        except Exception as exc:  # noqa: BLE001
            print(f"  STILL FAILING: {exc}")
            continue
        sr = eval_results.get("success_rate", 0.0)
        print(f"  success_rate={sr*100:.2f}%  avg_reward={eval_results.get('avg_reward')}")

        new_rec = dict(rec)
        new_rec.update(
            success_rate=eval_results.get("success_rate", 0.0),
            avg_reward=eval_results.get("avg_reward", 0.0),
            std_reward=eval_results.get("std_reward"),
            median_reward=eval_results.get("median_reward"),
            num_seeds=eval_results.get("num_seeds"),
            error=None,
            note=(f"reeval of trial #{tid} (same checkpoint"
                  + (f", {args.num_eval_seeds}-episode final eval)" if args.num_eval_seeds else ")")),
            timestamp=datetime.now(timezone.utc).isoformat(),
        )
        new_id = hs.append_trial(args.script, new_rec, active_env=args.active_env)
        print(f"  appended corrected record as trial #{new_id}")


if __name__ == "__main__":
    main()
