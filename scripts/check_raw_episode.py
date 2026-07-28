#!/usr/bin/env python3
"""Validate raw bridge_data_robot Push-T trajectories before converting them.

Run this on the FIRST episode of a collection session, before recording 150 of
them, and again over the whole session before the zarr conversion. Every check
here corresponds to a property the training pipeline silently assumes; a
violation costs a re-collection if it is found late.

    python scripts/check_raw_episode.py ~/widowx_data/.../raw/traj_group0/traj0
    python scripts/check_raw_episode.py ~/widowx_data/.../raw/traj_group0   # all trajs

Reference values come from the 2026-03 collection (data/03-23-pusht-data.zip)
and the archive built from it (data/pusht_widowx_data.zip):

    dt          0.0503 s        20 Hz control loop (`move_duration: 0.05`)
    |action|max 0.008 m         `vr_xy_step_clip`
    idle frac   0.24            fraction of exactly-(0,0) actions -- BEAT THIS,
                                it is the absorbing state described in
                                PUSHT_DATA_COLLECTION_RUNBOOK.md section 6
    z           0.0197 m held   deploy audit measured a sag to 0.009 at reach
    task_stage  {0, 1}          B pressed once, episode terminated properly
"""

from __future__ import annotations

import argparse
import glob
import os
import pickle
import sys

import numpy as np

# --- expectations -----------------------------------------------------------

DT_TARGET = 0.05
DT_TOL = 0.0025          # 5%: the loop prints its own warning past 1.05x
ACT_CLIP = 0.008
IDLE_REFERENCE = 0.24    # the old collection's idle fraction
IDLE_MAX = 0.20          # fail above this -- the point is to improve on 0.24
Z_TARGET = 0.02
Z_TOL = 0.004            # 4 mm; the old deploy sag was 11 mm


class Report:
    """Accumulates pass/fail lines for one trajectory."""

    def __init__(self, name: str):
        self.name = name
        self.lines: list[tuple[bool, str]] = []

    def check(self, ok: bool, msg: str) -> bool:
        self.lines.append((bool(ok), msg))
        return bool(ok)

    def note(self, msg: str) -> None:
        self.lines.append((None, msg))

    @property
    def failed(self) -> bool:
        return any(ok is False for ok, _ in self.lines)

    def render(self) -> None:
        print(f"\n=== {self.name}")
        for ok, msg in self.lines:
            mark = "    " if ok is None else ("ok  " if ok else "FAIL")
            print(f"  {mark} {msg}")


def check_traj(traj_dir: str) -> Report:
    rep = Report(os.path.basename(traj_dir.rstrip("/")))

    obs_path = os.path.join(traj_dir, "obs_dict.pkl")
    pol_path = os.path.join(traj_dir, "policy_out.pkl")
    if not os.path.isfile(obs_path) or not os.path.isfile(pol_path):
        rep.check(False, f"missing obs_dict.pkl / policy_out.pkl in {traj_dir}")
        return rep

    with open(obs_path, "rb") as f:
        obs = pickle.load(f)
    with open(pol_path, "rb") as f:
        policy_out = pickle.load(f)

    # RawSaver writes the 7-D `actions_save` under the 'actions' key for
    # action_mode '2trans', so this is (T-1, 7) with only dims 0:2 active.
    act = np.array([p["actions"] for p in policy_out], dtype=np.float64)
    n_obs = len(obs["time_stamp"])

    # --- control rate -------------------------------------------------------
    ts = np.asarray(obs["time_stamp"], dtype=np.float64)
    dt = np.diff(ts)
    rep.check(
        abs(dt.mean() - DT_TARGET) <= DT_TOL,
        f"dt mean {dt.mean():.4f}s (want {DT_TARGET:.4f} +/- {DT_TOL:.4f}) "
        f"| median {np.median(dt):.4f} p90 {np.percentile(dt, 90):.4f} "
        f"max {dt.max():.4f}",
    )
    slow = int((dt > DT_TARGET * 1.05).sum())
    rep.check(
        slow < 0.05 * len(dt),
        f"{slow}/{len(dt)} steps overran 1.05x dt ({100 * slow / len(dt):.1f}%)",
    )

    # --- action law ---------------------------------------------------------
    xy = act[:, :2]
    amax = np.abs(xy).max()
    rep.check(
        abs(amax - ACT_CLIP) < 1e-9,
        f"|action|max {amax:.4f} m (want exactly {ACT_CLIP} -- the VR clip "
        f"should be reached at least once)",
    )
    rep.check(
        np.abs(act[:, 2:]).max() == 0.0,
        f"action dims 2:7 all zero (max {np.abs(act[:, 2:]).max():.4g})",
    )

    idle = float((np.abs(xy).max(axis=1) == 0).mean())
    rep.check(
        idle <= IDLE_MAX,
        f"idle fraction {idle:.3f} (old collection {IDLE_REFERENCE:.2f}, "
        f"want <= {IDLE_MAX:.2f})",
    )
    if idle > 0:
        # P(zero | previous zero) -- the absorbing-state signature. Base rate is
        # `idle`; a much higher conditional means the pauses are long runs
        # rather than isolated steps, which is the learnable failure.
        z = np.abs(xy).max(axis=1) == 0
        if z[:-1].sum():
            p_cond = float(z[1:][z[:-1]].mean())
            rep.note(f"P(idle|prev idle) {p_cond:.2f} vs base rate {idle:.2f}")

    # --- z lock -------------------------------------------------------------
    z_traj = np.asarray(obs["full_state"], dtype=np.float64)[:, 2]
    rep.check(
        abs(z_traj.mean() - Z_TARGET) <= Z_TOL and z_traj.std() < Z_TOL,
        f"z mean {z_traj.mean():.4f} min {z_traj.min():.4f} "
        f"max {z_traj.max():.4f} std {z_traj.std():.4f} "
        f"(want ~{Z_TARGET:.4f}; sag at reach shows up as a low min)",
    )

    # --- episode termination ------------------------------------------------
    stages = sorted(set(np.asarray(obs["task_stage"]).ravel().tolist()))
    rep.check(
        stages == [0, 1],
        f"task_stage {stages} (want [0, 1]: B pressed once, env_done fired)",
    )

    # --- frame / row alignment ----------------------------------------------
    # The converter pairs video frame t with row t, so any camera that dropped
    # frames silently misaligns every observation-action pair downstream.
    counts = {}
    for cam_dir in sorted(glob.glob(os.path.join(traj_dir, "images*"))):
        counts[os.path.basename(cam_dir)] = len(
            glob.glob(os.path.join(cam_dir, "im_*.jpg"))
        )
    rep.check(
        len(counts) >= 2,
        f"camera dirs {counts} (want images0 and images1 -- images1 is blue)",
    )
    rep.check(
        all(c == n_obs for c in counts.values()),
        f"frame counts {counts} vs {n_obs} observations",
    )
    rep.check(
        act.shape[0] == n_obs - 1,
        f"{act.shape[0]} actions vs {n_obs} observations (want T-1)",
    )

    rep.note(f"length {n_obs} steps ({n_obs * DT_TARGET:.1f}s)")
    return rep


def main() -> int:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument(
        "path",
        help="a trajN directory, or a traj_group directory containing many",
    )
    args = p.parse_args()

    path = os.path.abspath(os.path.expanduser(args.path))
    if os.path.isfile(os.path.join(path, "obs_dict.pkl")):
        traj_dirs = [path]
    else:
        traj_dirs = sorted(
            glob.glob(os.path.join(path, "traj*")),
            key=lambda s: int("".join(c for c in os.path.basename(s) if c.isdigit()) or 0),
        )
    if not traj_dirs:
        print(f"no trajectories found under {path}", file=sys.stderr)
        return 2

    reports = [check_traj(d) for d in traj_dirs]
    for rep in reports:
        rep.render()

    bad = [r.name for r in reports if r.failed]
    print(f"\n{len(reports) - len(bad)}/{len(reports)} trajectories passed")
    if bad:
        print("failed: " + ", ".join(bad))
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
