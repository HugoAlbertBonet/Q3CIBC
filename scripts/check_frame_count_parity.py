"""A1 Half 2 — frame<->flat-index parity, closed locally from the source mp4s.

The 17 GB frame cache is a deterministic function of the per-episode videos
(_build_frame_cache: assert len(frames) >= L, then frames[:L] written
sequentially into mm[start:end]). So the cache is correct-by-construction if,
for every episode, the video frame count matches the lowdim episode length
L = episode_ends[ep] - episode_starts[ep]:

  n_video == L        -> no truncation, exact 1:1, alignment closed.
  n_video  > L        -> build keeps frames[:L]; safe ONLY if the extra frame is
                         trailing (build's assumption). A LEADING extra would
                         shift that episode by 1 and be invisible to Half 1.
  n_video  < L        -> build would have raised; cache could not exist.

We read counts straight from the zip's mp4s (present in the 292 MB archive; only
the decoded cache is 17 GB), so this needs no cluster. Also dumps episode-0
frame 0 for the A2 visual spot-check.

Usage:
    python scripts/check_frame_count_parity.py \
        --archive data/pusht_widowx_data.zip --camera 1 --dump results/a1_ep0_f0.png
"""
import argparse
import os
import shutil
import tempfile
import zipfile
from collections import Counter
from pathlib import Path

import numpy as np


def zarr_prefix(ar: zipfile.ZipFile) -> str:
    for name in ar.namelist():
        idx = name.find("replay_buffer.zarr/")
        if idx != -1:
            return name[: idx + len("replay_buffer.zarr/")]
    raise SystemExit("replay_buffer.zarr not found")


def load_ends(archive_path: Path):
    import zarr
    with zipfile.ZipFile(archive_path, "r") as ar:
        prefix = zarr_prefix(ar)
        members = [n for n in ar.namelist() if n.startswith(prefix)]
        tmp = tempfile.mkdtemp(prefix="a1p_zarr_")
        try:
            ar.extractall(tmp, members=members)
            root = zarr.open(os.path.join(tmp, prefix.rstrip("/")), mode="r")
            ends = np.asarray(root["meta/episode_ends"][:], dtype=np.int64)
        finally:
            shutil.rmtree(tmp, ignore_errors=True)
    return ends


def count_frames(video_path: str) -> int:
    import imageio.v3 as iio
    # Try metadata first (no full decode); fall back to a streaming count.
    try:
        props = iio.improps(video_path)
        n = int(props.shape[0])
        if n > 0:
            return n
    except Exception:
        pass
    return sum(1 for _ in iio.imiter(video_path))


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--archive", type=Path, default=Path("data/pusht_widowx_data.zip"))
    p.add_argument("--camera", type=int, default=1)
    p.add_argument("--dump", type=Path, default=None,
                   help="write episode-0 frame 0 here for the A2 visual check")
    args = p.parse_args()

    ends = load_ends(args.archive)
    starts = np.concatenate([[0], ends[:-1]])
    n_ep = len(ends)
    lens = (ends - starts).astype(int)
    print(f"archive={args.archive}  camera={args.camera}  episodes={n_ep}  "
          f"total_frames(lowdim)={int(ends[-1])}\n")

    import imageio.v3 as iio
    with zipfile.ZipFile(args.archive, "r") as ar:
        names = set(ar.namelist())
        root = zarr_prefix(ar).split("replay_buffer.zarr/")[0]
        scratch = tempfile.mkdtemp(prefix="a1p_vid_")
        deltas = np.zeros(n_ep, dtype=int)
        bad = []
        try:
            for ep in range(n_ep):
                member = f"{root}videos/{ep}/{args.camera}.mp4"
                if member not in names:
                    raise SystemExit(f"missing video: {member}")
                ar.extract(member, scratch)
                vp = os.path.join(scratch, member)
                n_video = count_frames(vp)
                d = n_video - int(lens[ep])
                deltas[ep] = d
                if d < 0:
                    bad.append((ep, n_video, int(lens[ep])))
                if args.dump is not None and ep == 0:
                    frame0 = iio.imread(vp, index=0)
                    args.dump.parent.mkdir(parents=True, exist_ok=True)
                    iio.imwrite(args.dump, frame0)
                    print(f"wrote ep0 frame0 -> {args.dump}  shape={frame0.shape}\n")
                os.remove(vp)
                if (ep + 1) % 25 == 0 or ep == n_ep - 1:
                    print(f"  ...{ep + 1}/{n_ep}")
        finally:
            shutil.rmtree(scratch, ignore_errors=True)

    print("\n=== n_video - L distribution ===")
    for d, c in sorted(Counter(deltas.tolist()).items()):
        tag = ""
        if d == 0:
            tag = "  (exact 1:1)"
        elif d < 0:
            tag = "  (!! build would RAISE — impossible if cache exists)"
        else:
            tag = "  (trailing-truncated by build; safe IFF extra is trailing)"
        print(f"  delta={d:+d}: {c} episodes{tag}")

    print()
    if bad:
        print(f"!! {len(bad)} episode(s) with n_video < L (broken):")
        for ep, nv, L in bad[:20]:
            print(f"   ep {ep}: video={nv} lowdim={L}")
        return 1
    if np.all(deltas == 0):
        print("CLOSED: every video has exactly L frames. No truncation, exact "
              "1:1 frame<->flat-index. A1 alignment fully verified locally.")
    else:
        maxd = int(deltas.max())
        print(f"PARTIAL: all videos >= L (build ok), but {int(np.sum(deltas>0))} "
              f"episode(s) carry up to {maxd} extra frame(s). Build assumes these "
              f"are TRAILING. If any is leading, that episode is shifted by up to "
              f"{maxd}. Recommend a content spot-check on those episodes.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
