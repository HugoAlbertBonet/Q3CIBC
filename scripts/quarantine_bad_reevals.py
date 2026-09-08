#!/usr/bin/env python3
"""Mark re-score records that never actually evaluated anything.

Until this was fixed, `reeval_trials.py` force-set `error=None` and
`eval_error=None` AFTER merging the evaluator's output, and started each new
record from `dict(source_record)`. A failed re-score therefore produced a
record that looks entirely legitimate: `success_rate 0.0` and `eval_error None`
sitting beside an `avg_min_dist_to_target` copied verbatim from the run that
worked. Averaging those zeros into a table silently invents a result.

The signature of such a record is precise, and this only touches records that
match all four parts:

    * it is a RE-SCORE  (its checkpoint_dir already appeared in an earlier row)
    * success_rate == 0.0
    * eval_details is empty      <- no episode was ever run
    * a later row for the same checkpoint has a non-zero score, proving the
      checkpoint itself is fine and it was the re-score that failed

Matching records get `success_rate: null` so nothing can average them, plus
`quarantined: true` and an explanatory `eval_error`. Nothing is deleted: the row
stays, with its trial_id, so the history reads correctly.

    uv run python scripts/quarantine_bad_reevals.py --dry-run
    uv run python scripts/quarantine_bad_reevals.py --apply
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

REASON = ("re-score never ran: eval_details empty and success_rate 0.0 while a "
          "sibling row for the same checkpoint scored normally. The old "
          "reeval_trials.py masked the evaluator's error and inherited the "
          "source record's metrics, so this row is not a measurement.")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--paths", nargs="*", default=None,
                    help="trials.jsonl files. Default: every one under results/")
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--dry-run", action="store_true")
    g.add_argument("--apply", action="store_true")
    args = ap.parse_args()

    paths = ([Path(p) for p in args.paths] if args.paths
             else sorted((ROOT / "results" / "hyperparam_search").rglob("trials.jsonl")))

    total = 0
    for path in paths:
        rows = [json.loads(l) for l in open(path) if l.strip()]
        by_ckpt = defaultdict(list)
        for i, r in enumerate(rows):
            by_ckpt[r.get("checkpoint_dir")].append(i)

        hits = []
        for cd, idxs in by_ckpt.items():
            if len(idxs) < 2 or cd is None:
                continue
            healthy = any(isinstance(rows[i].get("success_rate"), (int, float))
                          and rows[i]["success_rate"] > 0 for i in idxs)
            if not healthy:
                continue                      # the checkpoint itself may be bad
            for i in idxs[1:]:                # index 0 is the training run
                r = rows[i]
                if r.get("quarantined"):
                    continue
                # An honest failure already SAYS it failed — particle/8 trial 9
                # carries "Evaluation failed: Error(s) in loading state_dict"
                # and needs no marking. Only rows whose error was masked are a
                # problem, so require eval_error to be absent.
                if (r.get("success_rate") == 0.0
                        and not (r.get("eval_details") or [])
                        and not r.get("eval_error") and not r.get("error")):
                    hits.append(i)

        if not hits:
            continue
        total += len(hits)
        rel = path.relative_to(ROOT)
        print(f"{rel}: {len(hits)} record(s)")
        ids = [rows[i].get("trial_id") for i in hits]
        print(f"    trial ids: {ids}")
        if args.dry_run:
            continue

        shutil.copy2(path, path.with_suffix(".jsonl.bak"))
        for i in hits:
            rows[i]["success_rate"] = None
            rows[i]["avg_reward"] = None
            rows[i]["quarantined"] = True
            rows[i]["eval_error"] = REASON
            rows[i]["error"] = REASON
        with open(path, "w") as fh:
            for r in rows:
                fh.write(json.dumps(r) + "\n")
        print(f"    marked; original saved to {rel}.bak")

    print(f"\n{'would mark' if args.dry_run else 'marked'} {total} record(s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
