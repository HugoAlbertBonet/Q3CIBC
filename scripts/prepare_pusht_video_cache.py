#!/usr/bin/env python3
"""Pre-build the decoded frame cache for the Diffusion-Policy Push-T archive.

`PushTWidowXVideoDataset` decodes `videos/<ep>/<cam>.mp4` once into a uint8
memmap (random access into H.264 is seek-bound, so per-sample decoding would
starve the GPU). The build is safe under concurrency — one process builds, the
rest wait — but a batch of GPU jobs waiting on it burns allocation, so run this
once on a CPU node before submitting the batch.

    python scripts/prepare_pusht_video_cache.py            # defaults
    python scripts/prepare_pusht_video_cache.py --cache-dir /scratch/pusht

Cost: ~17 GB and a few minutes for the 150-episode / 73k-frame collection at
240x320. Idempotent: re-running with an existing cache is a no-op.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dataset", type=Path,
                   default=ROOT / "data" / "pusht_widowx_data.zip")
    p.add_argument("--camera", type=int, default=1,
                   help="1 = fixed blue scene camera (the trained/deployed view)")
    p.add_argument("--image-height", type=int, default=240)
    p.add_argument("--image-width", type=int, default=320)
    p.add_argument("--frame-stack", type=int, default=2)
    p.add_argument("--cache-dir", type=Path, default=None,
                   help="default: <dataset dir>/_frame_cache")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    if not args.dataset.is_file():
        raise FileNotFoundError(f"Dataset not found: {args.dataset}")

    from utils.datasets import PushTWidowXVideoDataset

    # Constructing the dataset is what triggers (or reuses) the cache build.
    # idle_filter="none" keeps this independent of any training config.
    ds = PushTWidowXVideoDataset(
        archive_path=str(args.dataset),
        frame_stack=args.frame_stack,
        camera=args.camera,
        resize_hw=(args.image_height, args.image_width),
        idle_filter="none",
        cache_dir=str(args.cache_dir.resolve()) if args.cache_dir else None,
    )
    sample = ds[0]
    print(
        f"\nCache ready: {ds._cache_path}\n"
        f"  frames={ds._cache_len}  episodes={ds.n_episodes}  "
        f"sample state={tuple(sample['state'].shape)} {sample['state'].dtype}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
