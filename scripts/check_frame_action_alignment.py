#!/usr/bin/env python3
"""A1 — action <-> EEF alignment check on the raw WidowX zarr.

Loader-external on purpose: reads replay_buffer.zarr directly (mirroring
PushTWidowXVideoDataset._load_lowdim) instead of going through the dataset class
under suspicion. If the flat action/eef/episode_ends block is coherent, the
command action[t] must drive the measured EEF displacement eef[t+1]-eef[t]:

    action[:, :2]         = commanded planar EEF delta (dx, dy) in metres
    robot_eef_pose[:, :2] = measured EEF (x, y) in metres
    => eef[t+1]-eef[t] should ~equal action[t]  (corr ~ +0.9 on the old data)

Per episode, per axis we report Pearson corr at lag +1, plus a lag scan
(-1/0/+1) to expose off-by-one, plus slope and residual |Δeef - action|.

Usage:
    python scripts/check_frame_action_alignment.py \
        --archive data/pusht_widowx_data.zip
"""
import argparse
import os
import shutil
import tempfile
import zipfile
from pathlib import Path

import numpy as np


def load_lowdim(archive_path: Path):
    import zarr
    with zipfile.ZipFile(archive_path, "r") as ar:
        prefix = None
        for name in ar.namelist():
            idx = name.find("replay_buffer.zarr/")
            if idx != -1:
                prefix = name[: idx + len("replay_buffer.zarr/")]
                break
        if prefix is None:
            raise SystemExit(f"replay_buffer.zarr not found in {archive_path}")
        members = [n for n in ar.namelist() if n.startswith(prefix)]
        tmp = tempfile.mkdtemp(prefix="a1_zarr_")
        try:
            ar.extractall(tmp, members=members)
            root = zarr.open(os.path.join(tmp, prefix.rstrip("/")), mode="r")
            actions = np.asarray(root["data/action"][:], dtype=np.float64)
            eef = np.asarray(root["data/robot_eef_pose"][:], dtype=np.float64)
            ends = np.asarray(root["meta/episode_ends"][:], dtype=np.int64)
        finally:
            shutil.rmtree(tmp, ignore_errors=True)
    return actions, eef, ends


def pearson(a, b):
    if a.size < 3 or a.std() < 1e-12 or b.std() < 1e-12:
        return np.nan
    return float(np.corrcoef(a, b)[0, 1])


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--archive", type=Path, default=Path("data/pusht_widowx_data.zip"))
    p.add_argument("--move-eps", type=float, default=1e-4,
                   help="|action| above this = a moving step (drops teleop zeros)")
    p.add_argument("--flag-below", type=float, default=0.5,
                   help="flag episodes whose lag+1 corr falls below this")
    args = p.parse_args()

    act, eef, ends = load_lowdim(args.archive)
    act = act[:, :2]
    eef = eef[:, :2]
    starts = np.concatenate([[0], ends[:-1]])
    n_ep = len(ends)
    print(f"archive={args.archive}  frames={len(act)}  episodes={n_ep}")
    zero_share = float(np.mean(np.all(act == 0.0, axis=1)))
    print(f"exact-zero action share: {zero_share:.1%}\n")

    axis = ("dx", "dy")
    # Per-episode lag+1 corr per axis, plus pooled moving/all and lag scan.
    per_ep = {0: [], 1: []}
    pooled = {"all": {0: ([], []), 1: ([], [])},
              "move": {0: ([], []), 1: ([], [])}}
    lagscan = {lag: {0: ([], []), 1: ([], [])} for lag in (-1, 0, 1)}
    resid = {0: [], 1: []}
    flagged = []

    for i in range(n_ep):
        s, e = int(starts[i]), int(ends[i])
        a = act[s:e]            # (L, 2)
        d = np.diff(eef[s:e], axis=0)   # (L-1, 2) = eef[t+1]-eef[t]
        L = len(d)
        if L < 3:
            continue
        a_t = a[:L]                     # action[t], t in [0, L)
        ep_corr = []
        for c in range(2):
            cc = pearson(a_t[:, c], d[:, c])
            per_ep[c].append(cc)
            ep_corr.append(cc)
            # pooled all
            pooled["all"][c][0].append(a_t[:, c]); pooled["all"][c][1].append(d[:, c])
            # pooled moving-only
            m = np.abs(a_t[:, c]) > args.move_eps
            pooled["move"][c][0].append(a_t[m, c]); pooled["move"][c][1].append(d[m, c])
            resid[c].append(np.abs(d[:, c] - a_t[:, c]))
            # lag scan: action[t] vs eef[t+lag]-eef[t+lag-1]
            for lag in (-1, 0, 1):
                # displacement realized between step t+lag-1 and t+lag
                jj = np.arange(L)
                kk = jj + lag
                ok = (kk >= 0) & (kk < L)
                lagscan[lag][c][0].append(a_t[jj[ok], c])
                lagscan[lag][c][1].append(d[kk[ok], c])
        if np.nanmin(ep_corr) < args.flag_below:
            flagged.append((i, ep_corr[0], ep_corr[1]))

    def cat(pair):
        return np.concatenate(pair[0]), np.concatenate(pair[1])

    print("=== per-episode lag+1 corr(action, Δeef) ===")
    for c in range(2):
        arr = np.array(per_ep[c])
        print(f"  {axis[c]}: mean={np.nanmean(arr):+.3f}  min={np.nanmin(arr):+.3f}"
              f"  median={np.nanmedian(arr):+.3f}  "
              f"episodes<{args.flag_below}: {int(np.nansum(arr < args.flag_below))}/{len(arr)}")

    print("\n=== pooled corr ===")
    for c in range(2):
        aa, dd = cat(pooled["all"][c])
        am, dm = cat(pooled["move"][c])
        print(f"  {axis[c]}: all={pearson(aa, dd):+.3f}   moving-only={pearson(am, dm):+.3f}")

    # Natural pairing is lag 0: action[t] -> eef[t+1]-eef[t]. Peak at lag 0 or +1
    # (+1 = one step of actuation latency) is benign; a sharp peak elsewhere with
    # lag0 ~ 0 would signal an off-by-one in the flat block.
    print("\n=== lag scan (pooled, all steps) — expect strong lag0/+1 ===")
    for c in range(2):
        row = []
        for lag in (-1, 0, 1):
            aa, dd = cat(lagscan[lag][c])
            row.append(f"lag{lag:+d}={pearson(aa, dd):+.3f}")
        print(f"  {axis[c]}: " + "  ".join(row))

    print("\n=== residual |Δeef - action| (m) — ~0 if command executed ===")
    for c in range(2):
        r = np.concatenate(resid[c])
        print(f"  {axis[c]}: mean={r.mean():.5f}  p95={np.percentile(r, 95):.5f}  "
              f"max={r.max():.5f}")

    print()
    if flagged:
        print(f"!! {len(flagged)} episode(s) with corr < {args.flag_below} "
              f"(possible misalignment):")
        for i, cx, cy in flagged[:20]:
            print(f"   ep {i:3d}: dx={cx:+.3f} dy={cy:+.3f}")
    else:
        print(f"OK: every episode has lag+1 corr >= {args.flag_below} on both axes.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
