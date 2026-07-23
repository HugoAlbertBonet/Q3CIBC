"""A4/D2 — train (tf AREA) vs deploy (cv2 INTER_AREA) resize parity.

The frame cache resizes 640x480 -> 240x320 with tf.image.resize(AREA,
antialias=True) then round+clip+uint8 (utils/datasets.py _build_frame_cache).
The deploy client resizes the live frame with cv2.resize(INTER_AREA), uint8
(scripts/deploy_pusht_real.py preprocess). Both are "area" resamplers but are
NOT bit-identical; any systematic gap is a train/deploy pixel shift on EVERY
frame. This measures it on real frames (same RGB input, so channel order — D1 —
is out of scope here).

Force CPU for TF (the resize is tiny): CUDA_VISIBLE_DEVICES="" .

Usage:
    CUDA_VISIBLE_DEVICES="" python scripts/check_resize_parity.py \
        --archive data/pusht_widowx_data.zip --episodes 0 40 80 --per-ep 4
"""
import argparse
import os
import shutil
import tempfile
import zipfile
from pathlib import Path

import numpy as np


def zarr_prefix(ar):
    for name in ar.namelist():
        idx = name.find("replay_buffer.zarr/")
        if idx != -1:
            return name[: idx + len("replay_buffer.zarr/")]
    raise SystemExit("replay_buffer.zarr not found")


def tf_resize(frames_u8, H, W):
    import tensorflow as tf
    res = tf.image.resize(frames_u8.astype(np.float32), (H, W),
                          method=tf.image.ResizeMethod.AREA, antialias=True)
    return tf.cast(tf.clip_by_value(tf.round(res), 0, 255), tf.uint8).numpy()


def cv2_resize(frames_u8, H, W):
    import cv2
    out = np.empty((len(frames_u8), H, W, 3), dtype=np.uint8)
    for i, f in enumerate(frames_u8):
        out[i] = cv2.resize(f, (W, H), interpolation=cv2.INTER_AREA)
    return out


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--archive", type=Path, default=Path("data/pusht_widowx_data.zip"))
    p.add_argument("--camera", type=int, default=1)
    p.add_argument("--episodes", type=int, nargs="+", default=[0, 40, 80, 120])
    p.add_argument("--per-ep", type=int, default=4, help="frames sampled per episode")
    p.add_argument("--hw", type=int, nargs=2, default=[240, 320])
    p.add_argument("--dump", type=Path, default=Path("results/a4_resize_diff.png"))
    args = p.parse_args()
    H, W = args.hw

    import imageio.v3 as iio
    tf_all, cv_all = [], []
    with zipfile.ZipFile(args.archive, "r") as ar:
        names = set(ar.namelist())
        root = zarr_prefix(ar).split("replay_buffer.zarr/")[0]
        scratch = tempfile.mkdtemp(prefix="a4_")
        try:
            for ep in args.episodes:
                member = f"{root}videos/{ep}/{args.camera}.mp4"
                if member not in names:
                    raise SystemExit(f"missing {member}")
                ar.extract(member, scratch)
                vp = os.path.join(scratch, member)
                frames = iio.imread(vp)                      # (L,480,640,3) RGB uint8
                L = len(frames)
                idx = np.linspace(0, L - 1, args.per_ep).astype(int)
                sub = frames[idx]
                tf_all.append(tf_resize(sub, H, W))
                cv_all.append(cv2_resize(sub, H, W))
                os.remove(vp)
        finally:
            shutil.rmtree(scratch, ignore_errors=True)

    a = np.concatenate(tf_all).astype(np.int16)     # train path
    b = np.concatenate(cv_all).astype(np.int16)     # deploy path
    d = np.abs(a - b)
    n = len(a)
    print(f"frames compared: {n}   size {H}x{W}   (0-255 scale)\n")
    print(f"abs diff  mean={d.mean():.3f}  p50={np.percentile(d,50):.1f}  "
          f"p95={np.percentile(d,95):.1f}  p99={np.percentile(d,99):.1f}  "
          f"max={d.max()}")
    print(f"share of pixels differing by >1: {np.mean(d>1):.3%}")
    print(f"share of pixels differing by >5: {np.mean(d>5):.3%}")
    for c, name in enumerate("RGB"):
        dc = d[..., c]
        print(f"  {name}: mean={dc.mean():.3f}  max={dc.max()}")
    # bias: does one method run systematically brighter?
    bias = (a - b).mean()
    print(f"signed mean (tf - cv2): {bias:+.4f}  (systematic brightness bias)")

    # dump a visual for frame 0: tf | cv2 | 10x abs diff
    try:
        import imageio.v3 as iio
        tf0, cv0 = a[0].astype(np.uint8), b[0].astype(np.uint8)
        diff0 = np.clip(d[0] * 10, 0, 255).astype(np.uint8)
        panel = np.concatenate([tf0, cv0, diff0], axis=1)
        args.dump.parent.mkdir(parents=True, exist_ok=True)
        iio.imwrite(args.dump, panel)
        print(f"\nwrote {args.dump}  (tf | cv2 | 10x|diff|)")
    except Exception as e:
        print(f"(dump skipped: {e})")

    print()
    if d.mean() < 0.5 and np.percentile(d, 99) <= 3:
        print("VERDICT: negligible — tf vs cv2 AREA agree to <0.5 mean, p99<=3. "
              "A4/D2 is NOT a meaningful train/deploy shift.")
    else:
        print("VERDICT: non-trivial gap — deploy frames are systematically off "
              "from training. Make deploy use the tf path (or rebuild the cache "
              "with cv2) so both sides match.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
