"""Re-evaluate saved Q3C checkpoints whose EVAL failed (training succeeded).

Used after eval-side bug fixes (e.g. the Dstandardlibero frame_stack=2 rebuild
bug) so 20h+ trainings don't have to re-run. For each requested trial id:
  1. read its record from the env's trials.jsonl (checkpoint_dir),
  2. load the per-run config saved next to the checkpoint,
  3. apply optional evaluation-only parameter overrides,
  4. call hyperparam_search.evaluate_q3c on it,
  5. append a NEW corrected record (note: "reeval of trial #N") to trials.jsonl.

Run on a GPU compute node (render eval): MUJOCO_GL=egl.

    uv run --extra libero python scripts/reeval_trials.py \
        --active-env libero_goal_pixels --trials 6 9 10
"""

from __future__ import annotations

import argparse
import json
import sys
import time
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
    ap.add_argument(
        "--param-overrides",
        default=None,
        help=(
            "JSON evaluation-only parameter overrides, e.g. "
            "'{\"action_execute_horizon\": 2}'. Training/model overrides are "
            "rejected because they may not match the saved checkpoint."
        ),
    )
    args = ap.parse_args()

    overrides = json.loads(args.param_overrides) if args.param_overrides else {}
    invalid = sorted(set(overrides) - hs.INFERENCE_ONLY_PARAMS)
    if invalid:
        ap.error(
            "--param-overrides accepts evaluation-only parameters; rejected: "
            + ", ".join(invalid)
        )

    trials_path = hs._trials_path(args.script, active_env=args.active_env)
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
        if config.get("active_env") != args.active_env:
            print(
                f"trial #{tid}: config active_env={config.get('active_env')!r}, "
                f"expected {args.active_env!r}; skipping"
            )
            continue
        config = hs.apply_params_to_config(config, overrides)
        if args.num_eval_seeds:
            config["environments"][args.active_env]["num_eval_seeds"] = int(
                args.num_eval_seeds
            )
        override_text = f" overrides={overrides}" if overrides else ""
        print(f"\n=== re-eval trial #{tid} ({ckpt_dir}){override_text} ===")
        eval_started = time.monotonic()
        try:
            eval_results = hs.evaluate_q3c(ckpt_dir, config)
        except Exception as exc:  # noqa: BLE001
            print(f"  STILL FAILING: {exc}")
            continue
        eval_duration = time.monotonic() - eval_started
        sr = eval_results.get("success_rate", 0.0)
        print(
            f"  success_rate={sr*100:.2f}%  "
            f"avg_reward={eval_results.get('avg_reward')}"
        )

        new_rec = dict(rec)
        source_run_id = rec.get("run_id")
        new_params = dict(rec.get("params") or {})
        new_params.update(overrides)
        new_rec.update({k: v for k, v in eval_results.items() if k != "per_seed"})
        new_rec.update(
            run_id=hs._new_run_id(),
            source_run_id=source_run_id,
            params=new_params,
            eval_details=eval_results.get("per_seed", []),
            eval_error=None,
            error=None,
            training_failed=False,
            duration_seconds=round(eval_duration, 1),
            reeval_only=True,
            reeval_of_trial=tid,
            note=(f"reeval of trial #{tid} (same checkpoint"
                  + (f", {args.num_eval_seeds}-episode final eval" if args.num_eval_seeds else "")
                  + (f", overrides={overrides}" if overrides else "")
                  + ")"),
            timestamp=datetime.now(timezone.utc).isoformat(),
        )
        new_id = hs.append_trial(args.script, new_rec, active_env=args.active_env)
        print(f"  appended corrected record as trial #{new_id}")


if __name__ == "__main__":
    main()
