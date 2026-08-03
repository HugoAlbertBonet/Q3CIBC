#!/usr/bin/env python3
"""Export one episode from a zarr_video archive into a standalone replay bundle.

scripts/replay_pusht_episode.py needs three things from a demonstration: the
expert action sequence, the demo's first frame per camera (for the alignment
gate), and the demo's EEF trace (start pose + the plot's reference path). All
three are a few hundred kB, while the archive they live in is ~2 GB -- so they
get exported ONCE here, on whatever machine holds the archive, and committed.
The robot-side machine then runs the replay with no archive at all.

Bundle layout (one directory per episode):

    data/replay_episodes/ep000/
        meta.json      # episode index, source archive, step count, rate, ...
        actions.npy    # (T, 2) float32 -- planar (dx, dy) deltas in metres
        eef.npy        # (T, 3) float32 -- measured EEF (x, y, z) in metres
        cam0.png       # demo frame 0, D435          (640x480 RGB)
        cam1.png       # demo frame 0, blue scene cam

Usage:

    python scripts/export_replay_episode.py \
        --archive data/pusht_2026_07_zarr.zip --episode 0
    python scripts/export_replay_episode.py \
        --archive data/pusht_2026_07_zarr.zip --episode 0 1 2 --cameras 1
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# The readers live in the replay client; importing them keeps one implementation.
_spec = importlib.util.spec_from_file_location(
    "replay", ROOT / "scripts" / "replay_pusht_episode.py")
replay = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(replay)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--archive", type=Path,
                   default=ROOT / "data" / "pusht_2026_07_zarr.zip")
    p.add_argument("--episode", type=int, nargs="+", default=[0])
    p.add_argument("--cameras", type=int, nargs="+", default=[0, 1],
                   help="cameras whose FIRST frame is exported (alignment gate).")
    p.add_argument("--out-root", type=Path,
                   default=ROOT / "data" / "replay_episodes")
    p.add_argument("--force", action="store_true",
                   help="overwrite an existing bundle directory.")
    return p.parse_args()


def main() -> int:
    import cv2

    args = parse_args()
    if not args.archive.is_file():
        raise SystemExit(f"archive not found: {args.archive}")

    actions, eef, ends = replay.load_lowdim(args.archive)
    move_duration = None
    meta_all = replay.load_archive_metadata(args.archive)
    if meta_all:
        move_duration = (meta_all.get("provenance") or {}).get("move_duration")

    for episode in args.episode:
        if not 0 <= episode < len(ends):
            raise SystemExit(
                f"--episode must be in [0, {len(ends)}); got {episode}")
        out_dir = args.out_root / f"ep{episode:03d}"
        if out_dir.exists() and not args.force:
            raise SystemExit(f"{out_dir} already exists; pass --force to replace")
        out_dir.mkdir(parents=True, exist_ok=True)

        start = int(ends[episode - 1]) if episode > 0 else 0
        end = int(ends[episode])
        ep_actions = np.asarray(actions[start:end, :2], np.float32)
        ep_eef = np.asarray(eef[start:end, :3], np.float32)
        np.save(out_dir / "actions.npy", ep_actions)
        np.save(out_dir / "eef.npy", ep_eef)

        for cam in args.cameras:
            frame = replay.load_episode_frames(args.archive, episode, cam, [0])[0]
            cv2.imwrite(str(out_dir / f"cam{cam}.png"),
                        cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))

        zero_frac = float((np.linalg.norm(ep_actions, axis=1) == 0).mean())
        meta = {
            "episode": int(episode),
            "source_archive": args.archive.name,
            "rows": [start, end],
            "n_steps": int(end - start),
            "cameras": [int(c) for c in args.cameras],
            "move_duration": move_duration,
            "zero_action_share": zero_frac,
            "eef_start": ep_eef[0].tolist(),
            "eef_end": ep_eef[-1].tolist(),
            "action_units": "metres, planar EEF delta (dx, dy) per control step",
        }
        (out_dir / "meta.json").write_text(json.dumps(meta, indent=2) + "\n")
        print(f"ep{episode:03d}: {end - start} steps, zero-action {zero_frac:.1%}, "
              f"cameras {args.cameras} -> {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
