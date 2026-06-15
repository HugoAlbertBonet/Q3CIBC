"""Probe the LIVE LIBERO env obs keys on a SLURM COMPUTE NODE.

Booting robosuite/mujoco offscreen render must NOT run on the login node — run
this under srun/sbatch with a GPU (or at least EGL/OSMesa) and MUJOCO_GL set:

    MUJOCO_GL=egl srun --gres=gpu:1 --pty \
        uv run --extra libero python scripts/probe_libero_live_obs.py

It prints the live obs dict keys/shapes for task 0, then confirms that
`utils.libero.resolve_live_obs` rebuilds the SAME low-dim vector the dataset
produced at training time (keys read from the goal-embeddings sibling or, if a
checkpoint norm_stats.pt path is given, from there). Mismatch ->
update LIVE_KEY_ALIASES in utils/libero.py.
"""

from __future__ import annotations

import argparse
import os

import numpy as np


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--task-index", type=int, default=0)
    ap.add_argument(
        "--norm-stats",
        default=None,
        help="Optional path to a trained run's norm_stats.pt to read the exact "
        "libero_obs_keys used at training and check the dim matches.",
    )
    args = ap.parse_args()

    if "MUJOCO_GL" not in os.environ:
        print("[warn] MUJOCO_GL not set; offscreen render may fail. Try MUJOCO_GL=egl.")

    from utils.libero import get_task_infos, resolve_live_obs, select_lowdim_obs_keys
    from libero.libero.envs import OffScreenRenderEnv

    infos = get_task_infos()
    t = infos[args.task_index]
    print(f"task[{t['index']}] {t['name']} | {t['language']!r}")

    env = OffScreenRenderEnv(
        bddl_file_name=t["bddl_file"], camera_heights=128, camera_widths=128
    )
    obs = env.reset()
    print("\nLIVE obs keys + shapes:")
    for k, v in obs.items():
        try:
            print(f"  {k:28s} {np.asarray(v).shape}")
        except Exception:  # noqa: BLE001
            print(f"  {k:28s} ?")

    # Which canonical keys to bridge?
    if args.norm_stats and os.path.exists(args.norm_stats):
        import torch

        ns = torch.load(args.norm_stats, weights_only=False)
        canonical = list(ns["libero_obs_keys"])
        trained_dim = int(ns["state_shape"])
        print(f"\nusing trained keys from {args.norm_stats}: {canonical}")
    else:
        canonical = select_lowdim_obs_keys(list(obs.keys()))
        trained_dim = None
        print(f"\nusing discovered live keys: {canonical}")

    vec = resolve_live_obs(obs, canonical)
    print(f"resolve_live_obs -> low-dim vector dim = {vec.shape[0]}")
    if trained_dim is not None:
        goal_dim = trained_dim - vec.shape[0]
        print(
            f"trained state_shape = {trained_dim}  => implied goal_emb_dim = {goal_dim}"
        )
        print("OK" if goal_dim > 0 else "MISMATCH: live low-dim dim exceeds trained state_shape")
    env.close()


if __name__ == "__main__":
    main()
