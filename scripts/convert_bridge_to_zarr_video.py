#!/usr/bin/env python3
"""Convert a bridge-format Push-T capture zip into the zarr_video archive that
`PushTWidowXVideoDataset` reads.

Source layout (bridge_zip):
    <root>/<session>/raw/traj_group0/traj<N>/images{0,1}/im_<step>.jpg
    <root>/<session>/raw/traj_group0/traj<N>/policy_out.pkl   # list of step dicts

Each step dict: actions (7,), new_robot_transform (4x4), delta_robot_transform
(4x4), policy_type. actions[:2] are the planar (dx,dy) deltas in metres
(range ~+/-0.008); new_robot_transform[:3,3] is the EEF (x,y,z).

Output layout (zarr_video), zipped:
    <out_root>/replay_buffer.zarr/            (zarr DirectoryStore)
        data/action           (N, 7)  float32   # [:2] planar, matches old data
        data/robot_eef_pose   (N, 7)  float32   # [:3] xyz, rest 0
        meta/episode_ends     (E,)    int64      # cumulative step counts
    <out_root>/videos/<ep>/<cam>.mp4            # one per (episode, camera)

Episodes are ordered deterministically by (session name, traj index). Per
episode we keep L = number of pkl steps; images are numerically sorted and the
first L frames per camera are encoded (trailing frames dropped, mirroring the
loader's align-to-L behaviour).
"""
from __future__ import annotations

import argparse
import io
import os
import pickle
import re
import shutil
import tempfile
import zipfile
from pathlib import Path

import numpy as np

TRAJ_RE = re.compile(
    r"^(?P<root>.*?)/(?P<session>[^/]+)/raw/traj_group0/traj(?P<idx>\d+)/policy_out\.pkl$"
)
IMG_RE = re.compile(r"/im_(\d+)\.jpg$")


def enumerate_episodes(names):
    """Return [(session, idx, pkl_name, img_prefix)] sorted by (session, idx)."""
    eps = []
    for n in names:
        m = TRAJ_RE.match(n)
        if not m:
            continue
        traj_dir = n[: -len("policy_out.pkl")]  # .../traj<N>/
        eps.append(
            (m.group("root"), m.group("session"), int(m.group("idx")), n, traj_dir)
        )
    eps.sort(key=lambda e: (e[1], e[2]))  # (session, idx)
    return eps


def sorted_frames(names, img_dir):
    """Numerically-sorted im_*.jpg entries under img_dir (e.g. .../images0/)."""
    frames = []
    for n in names:
        if n.startswith(img_dir):
            m = IMG_RE.search(n)
            if m:
                frames.append((int(m.group(1)), n))
    frames.sort(key=lambda x: x[0])
    return [n for _, n in frames]


def _render_readme(m: dict) -> str:
    el = m["episode_length"]
    n_ep = m["n_episodes"]
    n_last = n_ep - 1
    cams = "\n".join(f"  - `{c}.mp4` — {desc}" for c, desc in m["cameras"].items())
    prov = m.get("provenance") or {}
    prov_rows = "\n".join(
        f"| {k} | `{v}` |" for k, v in prov.items() if v is not None
    ) or "| (none captured) | |"
    return f"""# {m['name']} — Push-T Real (converted)

Real-world **Push-T** dataset (robot pushing a T-block to a target pose),
converted from the raw bridge capture `{m['source_zip']}` into the
[Diffusion Policy](https://github.com/real-stanford/diffusion_policy) real-robot
**zarr_video** format so it drops directly into `PushTWidowXVideoDataset`.
Low-dim signals live in a single Zarr `ReplayBuffer`; RGB observations are
per-episode MP4s (one per camera).

## Directory Layout

```
{m['name']}/
├── README.md
├── metadata.json           # machine-readable summary + per-episode provenance
├── replay_buffer.zarr/
│   ├── data/
│   │   ├── action           # (N, 7)  see below
│   │   └── robot_eef_pose   # (N, 7)  [:3] = xyz
│   └── meta/episode_ends    # ({n_ep},) episode boundaries
└── videos/
    ├── 0/{{0.mp4, 1.mp4}}     # episode 0, one mp4 per camera
    └── ...                  # episodes 0 .. {n_last}
```

## Dataset Summary

| Property               | Value                                             |
|------------------------|---------------------------------------------------|
| Episodes               | **{m['n_episodes']}**                              |
| Total timesteps        | **{m['total_timesteps']:,}**                       |
| Episode length         | min {el['min']} / max {el['max']} / mean ≈ {el['mean']} |
| Collection sessions    | **{m['n_sessions']}** (see `metadata.json`)        |
| Cameras                | **{len(m['cameras'])}** per episode:               |
{cams}
| Video resolution       | {m['video']['resolution_wxh']}, {m['video']['codec']} crf {m['video']['crf']}, {m['video']['fps']} fps |
| Video frames / episode | exactly that episode's timestep count             |

## Low-Dimensional Data (`replay_buffer.zarr/data`)

All episodes concatenated along axis 0 (`shape[0] == {m['total_timesteps']}`);
slice per-episode with `meta/episode_ends`.

| Key              | Shape        | dtype | Description |
|------------------|--------------|-------|-------------|
| `action`         | (N, 7)       | f4    | Commanded action. Dims 0–1 = (Δx, Δy) EEF deltas in metres, range [{m['action']['planar_range_m'][0]:.3f}, {m['action']['planar_range_m'][1]:.3f}] (teleop VR step clip ±0.008); dims 2–6 = 0 for this planar task. |
| `robot_eef_pose` | (N, 7)       | f4    | End-effector pose; dims 0–2 = (x, y, z) in metres; dims 3–6 = 0 (rotation/gripper not recorded in the source `policy_out.pkl`). |

> Note: unlike the earlier `pusht_widowx_data.zip`, the source capture stored
> only `actions` + robot transforms per step, so `robot_joint`,
> `*_vel`, `stage`, and `timestamp` arrays are **not** present here.

### Meta (`replay_buffer.zarr/meta`)

| Key            | Shape    | dtype | Description |
|----------------|----------|-------|-------------|
| `episode_ends` | ({n_ep},) | i8    | Exclusive end index of each episode; episode `i` spans `[episode_ends[i-1], episode_ends[i])` (0 for `i=0`). |

## RGB Data (`videos/`)

- One folder per integer episode index; inside, one MP4 per camera.
- Frame `t` of every video is time-aligned with row `t` of that episode's Zarr
  slice (image ↔ state correspond one-to-one).

## Collection Provenance (from the source rig config)

| Field | Value |
|-------|-------|
{prov_rows}

## Usage

```python
from utils.datasets import PushTWidowXVideoDataset
ds = PushTWidowXVideoDataset(
    archive_path='{m['name']}.zip', frame_stack=2, cameras=[0, 1],
    resize_hw=(240, 320), idle_filter='drop_zero', action_chunk=1,
)
```
"""


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--src", type=Path, default=Path("data/pusht_2026_07.zip"))
    ap.add_argument("--out", type=Path, default=Path("data/pusht_2026_07_zarr.zip"))
    ap.add_argument("--out-root", default="pusht_2026_07",
                    help="top-level dir inside the output zip")
    ap.add_argument("--cameras", type=int, nargs="+", default=[0, 1])
    ap.add_argument("--fps", type=int, default=10, help="mp4 container fps (arbitrary)")
    ap.add_argument("--crf", type=int, default=18,
                    help="libx264 quality (lower=better/bigger; 18 is high-quality, "
                         "~3x smaller than the jpg source; the old archive used ~23)")
    ap.add_argument("--limit", type=int, default=None,
                    help="convert only the first N episodes (smoke test)")
    args = ap.parse_args()

    import imageio.v2 as imageio  # ffmpeg writer

    if not args.src.is_file():
        raise FileNotFoundError(args.src)
    src = zipfile.ZipFile(args.src, "r")
    names = src.namelist()
    episodes = enumerate_episodes(names)
    if args.limit:
        episodes = episodes[: args.limit]
    if not episodes:
        raise ValueError("no trajectories matched in source zip")
    print(f"episodes: {len(episodes)} (cameras {args.cameras}, crf {args.crf})")

    work = Path(tempfile.mkdtemp(prefix="bridge2zarr_"))
    scratch = work / "scratch"
    scratch.mkdir()
    vid_dir = work / "videos"
    vid_dir.mkdir()

    all_actions = []
    all_eef = []
    ends = []
    ep_meta = []
    total = 0

    # Provenance from a source session config (same rig/teleop for all sessions).
    provenance = {}
    try:
        import json as _json
        cfgs = [n for n in names if n.endswith("config.json")]
        if cfgs:
            c = _json.load(io.BytesIO(src.read(cfgs[0])))
            env = c.get("agent", {}).get("env", [None, {}])[1]
            provenance = {
                "action_mode": env.get("action_mode"),
                "camera_topics": env.get("camera_topics"),
                "fixed_z_height": env.get("fixed_z_height"),
                "vr_xy_step_clip": env.get("vr_xy_step_clip"),
                "move_duration": env.get("move_duration"),
            }
    except Exception as e:  # provenance is best-effort, never fatal
        print(f"  (provenance read skipped: {e})")

    try:
        for ep, (root, session, idx, pkl_name, traj_dir) in enumerate(episodes):
            steps = pickle.load(io.BytesIO(src.read(pkl_name)))
            L = len(steps)
            actions = np.zeros((L, 7), np.float32)
            eef = np.zeros((L, 7), np.float32)
            for t, s in enumerate(steps):
                actions[t] = np.asarray(s["actions"], np.float32)[:7]
                eef[t, :3] = np.asarray(s["new_robot_transform"], np.float32)[:3, 3]

            for cam in args.cameras:
                img_dir = f"{traj_dir}images{cam}/"
                frames = sorted_frames(names, img_dir)
                if len(frames) < L:
                    raise ValueError(
                        f"ep{ep} {session}/traj{idx} cam{cam}: {len(frames)} frames "
                        f"< {L} steps"
                    )
                out_mp4 = vid_dir / str(ep)
                out_mp4.mkdir(exist_ok=True)
                writer = imageio.get_writer(
                    out_mp4 / f"{cam}.mp4", fps=args.fps, codec="libx264",
                    pixelformat="yuv420p",
                    output_params=["-crf", str(args.crf)],
                    macro_block_size=1,
                )
                for name in frames[:L]:
                    writer.append_data(imageio.imread(io.BytesIO(src.read(name))))
                writer.close()

            all_actions.append(actions)
            all_eef.append(eef)
            total += L
            ends.append(total)
            ep_meta.append({"episode": ep, "session": session, "traj": idx,
                            "length": int(L)})
            if (ep + 1) % 10 == 0 or ep == len(episodes) - 1:
                print(f"  [{ep + 1}/{len(episodes)}] {session}/traj{idx} L={L} "
                      f"(total frames {total})")

        actions = np.concatenate(all_actions, 0)
        eef = np.concatenate(all_eef, 0)
        episode_ends = np.asarray(ends, np.int64)
        print(f"action range [:2]: {actions[:, :2].min(0)} -> {actions[:, :2].max(0)}")

        # ── Write the zarr replay buffer (DirectoryStore) ──────────────────
        import zarr
        zroot = work / "replay_buffer.zarr"
        z = zarr.open(str(zroot), mode="w")
        z.create_dataset("data/action", data=actions, chunks=(4096, 7))
        z.create_dataset("data/robot_eef_pose", data=eef, chunks=(4096, 7))
        z.create_dataset("meta/episode_ends", data=episode_ends, chunks=(len(episode_ends),))

        # ── Metadata + README ──────────────────────────────────────────────
        lens = np.array([m["length"] for m in ep_meta])
        sessions = sorted({m["session"] for m in ep_meta})
        metadata = {
            "name": args.out_root,
            "source_zip": args.src.name,
            "format": "zarr_video (Diffusion-Policy real-robot layout)",
            "converted_by": "scripts/convert_bridge_to_zarr_video.py",
            "n_episodes": len(ep_meta),
            "total_timesteps": int(total),
            "episode_length": {"min": int(lens.min()), "max": int(lens.max()),
                               "mean": round(float(lens.mean()), 1)},
            "n_sessions": len(sessions),
            "sessions": sessions,
            "cameras": {str(c): ("D435 (images0)" if c == 0 else
                                 "blue scene cam (images1)" if c == 1 else f"images{c}")
                        for c in args.cameras},
            "video": {"resolution_wxh": "640x480", "codec": "h264",
                      "crf": args.crf, "pix_fmt": "yuv420p", "fps": args.fps},
            "action": {"shape": [int(total), 7],
                       "planar_dims": [0, 1],
                       "planar_range_m": [float(actions[:, :2].min()),
                                          float(actions[:, :2].max())],
                       "note": "dims 0-1 = (dx, dy) EEF deltas in metres; dims 2-6 = 0"},
            "robot_eef_pose": {"shape": [int(total), 7],
                               "note": "dims 0-2 = (x, y, z) in metres; dims 3-6 = 0"},
            "provenance": provenance,
            "episodes": ep_meta,
        }
        import json as _json
        (work / "metadata.json").write_text(_json.dumps(metadata, indent=1))
        (work / "README.md").write_text(_render_readme(metadata))

        # ── Pack everything into the output zip ────────────────────────────
        args.out.parent.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(args.out, "w", zipfile.ZIP_STORED) as zf:
            zf.write(work / "README.md", f"{args.out_root}/README.md")
            zf.write(work / "metadata.json", f"{args.out_root}/metadata.json")
            for base in (zroot, vid_dir):
                top = "replay_buffer.zarr" if base is zroot else "videos"
                for p in sorted(base.rglob("*")):
                    if p.is_file():
                        arc = f"{args.out_root}/{top}/{p.relative_to(base)}"
                        zf.write(p, arc)
        size_gb = args.out.stat().st_size / 1e9
        print(f"\nwrote {args.out} ({size_gb:.2f} GB) — {len(episodes)} episodes, "
              f"{total} frames")
    finally:
        shutil.rmtree(work, ignore_errors=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
