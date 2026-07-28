#!/usr/bin/env python3
"""Measure the table contact height in the WidowX arm's base frame.

Push-T locks the end-effector to a fixed z (``FIXED_Z_HEIGHT`` in
``conf_clam_pusht.py``, 0.02 m for the 2026-03 collection). That number is only
meaningful relative to the table: if the arm or the table has moved, 0.02 is no
longer the height at which the closed gripper contacts the surface, and the
demonstrations record an inconsistent contact regime.

This drives the EEF straight down in small increments with the gripper closed,
watching the ACHIEVED z. When commanded descent stops producing achieved
descent, the gripper is on the table -- that achieved z is the contact height.

    # server must be up:
    #   docker compose exec robonet bash -lic "widowx_env_service --server"
    python scripts/measure_floor_height.py --dry_run     # no motion, prints state
    python scripts/measure_floor_height.py

SAFETY: this deliberately drives the arm toward the table. Keep a hand on the
power switch. The descent is capped (``--max_descent``), each step is small
(``--step``), and it stops as soon as the arm stalls. Ctrl-C aborts.

Reads the same ``state`` observation key the deploy client uses
(``scripts/deploy_pusht_real_ibc.py:_extract_eef_xy``); element 2 is z.
"""

from __future__ import annotations

import argparse
import sys
import time

import numpy as np


def load_widowx():
    try:
        from widowx_envs.widowx_env_service import WidowXClient, WidowXConfigs
    except ImportError as exc:  # pragma: no cover - environment dependent
        print(
            "Could not import widowx_envs. Activate the deploy env "
            "(conda activate q3c_deploy) and ensure widowx_envs is installed:\n"
            "  pip install -e ~/bridge_data_robot/widowx_envs",
            file=sys.stderr,
        )
        raise SystemExit(2) from exc
    return WidowXClient, WidowXConfigs


def read_state(client, max_wait_sec: float = 3.0) -> np.ndarray | None:
    """Poll get_observation() until a usable state vector comes back."""
    deadline = time.monotonic() + max_wait_sec
    while time.monotonic() < deadline:
        try:
            obs = client.get_observation()
        except Exception:
            obs = None
        if obs is not None:
            for key in ("eef_pos", "ee_pos", "agent_pos", "state", "proprio"):
                val = obs.get(key)
                if val is None:
                    continue
                arr = np.asarray(val, dtype=np.float64).reshape(-1)
                if arr.size >= 3:
                    return arr
        time.sleep(0.05)
    return None


def main() -> int:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--ip", default="localhost")
    p.add_argument("--port", type=int, default=5556)
    p.add_argument("--step", type=float, default=0.002,
                   help="commanded descent per step, metres (default 2 mm)")
    p.add_argument("--max_descent", type=float, default=0.06,
                   help="hard cap on total commanded descent, metres")
    p.add_argument("--stall_eps", type=float, default=0.0005,
                   help="achieved descent below this counts as stalled, metres")
    p.add_argument("--stall_steps", type=int, default=3,
                   help="consecutive stalled steps that mean contact")
    p.add_argument("--settle_sec", type=float, default=0.4,
                   help="wait after each step before reading z")
    p.add_argument("--start_z", type=float, default=0.06,
                   help="z to rise to before descending, metres")
    p.add_argument("--dry_run", action="store_true",
                   help="connect and print state only; never commands motion")
    args = p.parse_args()

    WidowXClient, WidowXConfigs = load_widowx()

    # 3trans so z is commandable at all: in the '2trans' mode used for
    # collection the z element is forced to zero and lock_z overwrites the
    # target, which is exactly what we are trying to measure around.
    env_params = WidowXConfigs.DefaultEnvParams.copy()
    env_params.update(
        {
            "action_mode": "3trans",
            "lock_z": False,
            "fixed_z_height": None,
            "fix_zangle": 0.1,
            "move_duration": 0.2,
            "adaptive_wait": True,
            "fixed_gripper": 0.0,
            "skip_move_to_neutral": False,
            "move_to_rand_start_freq": -1,
        }
    )

    client = WidowXClient(host=args.ip, port=args.port)
    print("[INFO] init() -- applying fresh env params (action_mode=3trans, lock_z off)")
    client.init(env_params, image_size=256)
    client.reset()

    state = read_state(client)
    if state is None:
        print("[FAIL] no state from the server; is widowx_env_service running?",
              file=sys.stderr)
        return 2
    print(f"[INFO] state after reset: {np.round(state[:6], 4)}")
    print(f"[INFO] starting z = {state[2]:.4f}")

    if args.dry_run:
        print("[INFO] --dry_run: no motion commanded. Done.")
        client.stop()
        return 0

    print()
    print("  The arm will now descend in "
          f"{args.step * 1000:.1f} mm steps, up to {args.max_descent * 1000:.0f} mm total,")
    print("  with the gripper CLOSED, until it stops descending (table contact).")
    print("  Keep a hand on the power switch. Ctrl-C aborts.")
    if input("  Type 'go' to proceed: ").strip().lower() != "go":
        print("[INFO] aborted by user")
        client.stop()
        return 0

    z_hist = [float(state[2])]
    stalled = 0
    contact_z = None
    descended = 0.0

    try:
        while descended < args.max_descent:
            client.step_action(
                np.array([0.0, 0.0, -args.step, 0.0], dtype=np.float32)
            )
            descended += args.step
            time.sleep(args.settle_sec)

            state = read_state(client)
            if state is None:
                print("[WARN] lost state, retrying")
                continue
            z = float(state[2])
            delta = z_hist[-1] - z
            z_hist.append(z)

            stalled = stalled + 1 if delta < args.stall_eps else 0
            flag = "  <- stalled" if delta < args.stall_eps else ""
            print(f"  commanded -{descended * 1000:5.1f} mm   "
                  f"achieved z {z:+.4f}   step delta {delta * 1000:+5.2f} mm{flag}")

            if stalled >= args.stall_steps:
                contact_z = z
                break
    except KeyboardInterrupt:
        print("\n[INFO] interrupted")

    print()
    if contact_z is None:
        print(f"[WARN] no stall detected within {args.max_descent * 1000:.0f} mm of descent.")
        print("       Either the arm never reached the table, or it is drooping as fast")
        print("       as it is commanded. Lowest achieved z was "
              f"{min(z_hist):+.4f}.")
    else:
        print(f"[RESULT] table contact at achieved z = {contact_z:+.4f} m")
        print(f"         2026-03 collection held z = 0.0197 (FIXED_Z_HEIGHT = 0.02)")
        print()
        print("  Set FIXED_Z_HEIGHT in conf_clam_pusht.py to this contact height")
        print("  (plus any pusher offset), and remember the SAME value belongs in")
        print("  the deploy client: scripts/deploy_pusht_real_ibc.py --fixed_z_height.")

    print("[INFO] raising the arm before exit")
    for _ in range(int(args.max_descent / args.step) + 2):
        client.step_action(np.array([0.0, 0.0, args.step, 0.0], dtype=np.float32))
        time.sleep(0.05)
    client.stop()
    return 0


if __name__ == "__main__":
    sys.exit(main())
