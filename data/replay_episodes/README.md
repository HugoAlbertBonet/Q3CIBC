# Replay episode bundles

Everything `scripts/replay_pusht_episode.py` needs from a demonstration, without
the ~2 GB `pusht_2026_07_zarr.zip`. Exported by
`scripts/export_replay_episode.py` on a machine that has the archive; committed
so the robot-side machine needs nothing else.

```
ep<NNN>/
  actions.npy   (T, 2) float32 — expert planar EEF deltas (dx, dy) in metres,
                                 one row per control step. THIS is what the
                                 replay commands open-loop.
  eef.npy       (T, 3) float32 — measured EEF (x, y, z). Row 0 is the start pose
                                 the replay moves to; the whole trace is the
                                 plot's demo reference path.
  cam0.png                     — demo frame 0, D435 (640x480 RGB, lossless)
  cam1.png                     — demo frame 0, blue scene cam
  meta.json                    — episode index, source archive, step count,
                                 recorded move_duration, idle share, EEF ends
```

The PNGs are what the alignment gate blends against the live view, so the T in
`cam*.png` is where the T must be before the replay starts.

## Which episodes are here

Chosen for a clean replay: no all-zero `robot_eef_pose` rows (37 of the 151
episodes have them) and a low idle-action share.

| episode | steps | idle actions | EEF displacement | notes |
|---------|-------|--------------|------------------|-------|
| 70      | 319   | 9.7%         | 14.0 cm          | shortest clean one — start here |
| 112     | 679   | 6.6%         | 15.4 cm          | |
| 140     | 730   | 6.3%         | 14.0 cm          | lowest idle share |

None of the three dips below its own start x, so the deploy client's approach
floor never clips the expert. Episode 0 is deliberately absent: 43% of its
actions are exactly (0,0) and 32 of its EEF rows are all-zero.

## Adding another episode

```bash
python scripts/export_replay_episode.py \
    --archive data/pusht_2026_07_zarr.zip --episode 95 --cameras 0 1
```

At ~500 kB per episode these are meant to be committed. Verified byte-identical
to reading the archive directly (`--archive`), frames included.
