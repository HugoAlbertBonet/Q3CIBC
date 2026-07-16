#!/usr/bin/env python3
"""Live camera-alignment helper for the Push-T fixed scene camera.

Overlays the live ``images1`` / ``/blue/image_raw`` stream (``over_shoulder_img``
from the robot server) on top of a reference frame from the training data, so
you can physically nudge the Logitech until the live view matches what seed_00XX
was trained on. Misalignment here is the #1 silent cause of a policy that
"trained fine but does nothing sensible on the robot".

Requires the robot server up:  docker compose exec robonet bash -lic "widowx_env_service --server"

    python scripts/align_pusht_camera.py            # localhost
    python scripts/align_pusht_camera.py --ip 10.0.0.5

Keys (in the OpenCV window):
    b   blend view  (alpha slider)
    e   edge view   (reference Canny edges drawn over the live frame)
    d   diff view   (abs difference; dark = aligned)
    l   live only   /   r   reference only
    [ ] decrease / increase blend alpha
    s   save current composite to --out
    q / ESC  quit

The edge view ('e') is usually the most useful: align live features
(table borders, target outline) to the green reference edges.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_REF = ROOT / "scripts" / "assets" / "pusht_images1_ref.jpg"

# Same env init the deploy client uses. Single camera on the current rig:
# blue (Logitech) == full_image[0] == external_img.
DEPLOY_ENV_PARAMS = {
    "camera_topics": [
        {"name": "/blue/image_raw"},
    ],
    "gripper_attached": "custom",
    "skip_move_to_neutral": True,      # alignment: don't move the arm
    "move_to_rand_start_freq": -1,
    "fix_zangle": 0.1,
    "action_mode": "2trans",
    "adaptive_wait": True,
    "fixed_z_height": 0.02,
    "neutral_z_height": 0.02,
    "lock_z": True,
    "action_clipping": None,
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--ip", default="localhost")
    p.add_argument("--port", type=int, default=5556)
    p.add_argument("--reference", type=Path, default=DEFAULT_REF)
    p.add_argument("--alpha", type=float, default=0.5,
                   help="initial blend weight of the live frame (0..1)")
    p.add_argument("--out", type=Path, default=ROOT / "camera_align.png")
    p.add_argument("--no-init", action="store_true",
                   help="skip client.init (use if the env is already initialized)")
    p.add_argument("--obs-key", default="auto",
                   choices=["auto", "external_img", "over_shoulder_img"],
                   help="which get_observation() field holds the blue frame")
    return p.parse_args()


def pick_blue_frame(obs: dict, obs_key: str):
    """over_shoulder_img if present (dual-cam) else external_img (single-cam)."""
    if obs_key == "auto":
        return obs.get("over_shoulder_img") if obs.get("over_shoulder_img") is not None \
            else obs.get("external_img")
    return obs.get(obs_key)


def main() -> int:
    import cv2

    args = parse_args()
    if not args.reference.is_file():
        raise FileNotFoundError(f"reference frame not found: {args.reference}")

    ref_bgr = cv2.imread(str(args.reference), cv2.IMREAD_COLOR)  # BGR
    H, W = ref_bgr.shape[:2]
    ref_gray = cv2.cvtColor(ref_bgr, cv2.COLOR_BGR2GRAY)
    ref_edges = cv2.Canny(ref_gray, 60, 160)
    print(f"Reference: {args.reference}  ({W}x{H})")

    from widowx_envs.widowx_env_service import WidowXClient

    client = WidowXClient(host=args.ip, port=args.port)
    if not args.no_init:
        client.init(DEPLOY_ENV_PARAMS, image_size=256)

    obs = None
    while obs is None:
        obs = client.get_observation()
        if obs is None or pick_blue_frame(obs, args.obs_key) is None:
            print("Waiting for blue frame (images1)...")
            obs = None
            time.sleep(1.0)

    mode = "e"          # start on edge overlay (most useful for alignment)
    alpha = float(np.clip(args.alpha, 0.0, 1.0))
    win = "pusht camera align  (b/e/d/l/r  [ ]  s  q)"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)
    print("Aligning. Focus the OpenCV window; press 'q' or ESC to quit.")

    try:
        while True:
            obs = client.get_observation()
            live = None if obs is None else pick_blue_frame(obs, args.obs_key)
            if live is None:
                continue
            # BGR (cv2 decode)
            if live.shape[:2] != (H, W):
                live = cv2.resize(live, (W, H), interpolation=cv2.INTER_AREA)

            if mode == "l":
                comp = live.copy()
            elif mode == "r":
                comp = ref_bgr.copy()
            elif mode == "d":
                comp = cv2.absdiff(live, ref_bgr)
            elif mode == "e":
                comp = live.copy()
                comp[ref_edges > 0] = (0, 255, 0)           # green reference edges
            else:  # blend
                comp = cv2.addWeighted(live, alpha, ref_bgr, 1.0 - alpha, 0.0)

            label = {"l": "LIVE", "r": "REFERENCE", "d": "DIFF (dark=aligned)",
                     "e": "EDGES (match to green)",
                     "b": f"BLEND live={alpha:.2f}"}[mode]
            cv2.putText(comp, label, (8, 22), cv2.FONT_HERSHEY_SIMPLEX,
                        0.6, (0, 0, 255), 2, cv2.LINE_AA)
            cv2.imshow(win, comp)

            k = cv2.waitKey(30) & 0xFF
            if k in (ord("q"), 27):
                break
            elif k == ord("b"):
                mode = "b"
            elif k == ord("e"):
                mode = "e"
            elif k == ord("d"):
                mode = "d"
            elif k == ord("l"):
                mode = "l"
            elif k == ord("r"):
                mode = "r"
            elif k == ord("["):
                alpha = max(0.0, alpha - 0.05); mode = "b"
            elif k == ord("]"):
                alpha = min(1.0, alpha + 0.05); mode = "b"
            elif k == ord("s"):
                cv2.imwrite(str(args.out), comp)
                print(f"saved {args.out}")
    except KeyboardInterrupt:
        pass
    finally:
        cv2.destroyAllWindows()
        client.stop()
    print("Alignment window closed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
