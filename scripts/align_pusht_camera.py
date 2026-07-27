#!/usr/bin/env python3
"""Live camera-alignment helper for the Push-T fixed scene cameras.

Overlays the live camera stream on top of a reference frame from the training
data, so you can physically nudge the camera until the live view matches what
seed_00XX was trained on. Misalignment here is the #1 silent cause of a policy
that "trained fine but does nothing sensible on the robot".

By default only the blue Logitech (``images1`` / ``/blue/image_raw``) is
aligned -- that is the stream every checkpoint trained on. With ``--both`` the
excluded D435 (``images0`` / ``/D435/color/image_raw``) is aligned too, shown
side by side; this needs the realsense enabled in bridge_data_robot (see
"Enabling the D435" below).

Requires the robot server up:  docker compose exec robonet bash -lic "widowx_env_service --server"

    python scripts/align_pusht_camera.py            # blue only, localhost
    python scripts/align_pusht_camera.py --both     # blue + D435
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

Enabling the D435 (for --both), in bridge_data_robot:
    widowx_envs/widowx_controller/launch/widowx_rs.launch
        realsense       default -> true
        camera1         default -> "D435"
        serial_no_camera1 -> the actual D435 serial
    widowx_envs/scripts/run.sh
        realsense:=true (both the camera_string on L6 and the roslaunch L17)
    ./generate_usb_config.sh, then check usb_connector_chart.yml has a non-empty
    D435 entry alongside blue.
Then rebuild:  USB_CONNECTOR_CHART=$(pwd)/usb_connector_chart.yml docker compose up --build robonet
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
# References = frames from the trained/deployed view of pusht_widowx_data.zip:
# episode 26, first frame (T at start position, drawn target outline and table
# corner marker clearly visible for alignment). Both cameras are taken from the
# same episode/frame, so they show the exact same physical instant.
#   camera 1 -> blue Logitech  (images1, the stream the policy consumes)
#   camera 0 -> D435 color     (images0, excluded from training)
# Regenerate with the same camera streams the dataset/policy uses so the live
# views match.
DEFAULT_REF_BLUE = ROOT / "scripts" / "assets" / "pusht_widowx_cam1_ref.jpg"
DEFAULT_REF_D435 = ROOT / "scripts" / "assets" / "pusht_widowx_cam0_ref.jpg"

BLUE_TOPIC = "/blue/image_raw"
D435_TOPIC = "/D435/color/image_raw"

# Same env init the deploy client uses. Single camera on the current rig:
# blue (Logitech) == full_image[0] == external_img.
DEPLOY_ENV_PARAMS = {
    "camera_topics": [
        {"name": BLUE_TOPIC},
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

# --both: same topic order the data was collected with (conf_clam_pusht.py),
# so full_image[0] == D435 == images0 and full_image[1] == blue == images1.
BOTH_ENV_PARAMS = {**DEPLOY_ENV_PARAMS,
                   "camera_topics": [{"name": D435_TOPIC},
                                     {"name": BLUE_TOPIC}]}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--ip", default="localhost")
    p.add_argument("--port", type=int, default=5556)
    p.add_argument("--both", action="store_true",
                   help="align both cameras side by side: blue (images1) and D435 (images0)")
    p.add_argument("--reference", type=Path, default=DEFAULT_REF_BLUE,
                   help="reference frame for the blue camera (images1)")
    p.add_argument("--reference-d435", type=Path, default=DEFAULT_REF_D435,
                   help="reference frame for the D435 camera (images0); --both only")
    p.add_argument("--alpha", type=float, default=0.5,
                   help="initial blend weight of the live frame (0..1)")
    p.add_argument("--out", type=Path, default=ROOT / "camera_align.png")
    p.add_argument("--no-init", action="store_true",
                   help="skip client.init (use if the env is already initialized)")
    p.add_argument("--obs-key", default="auto",
                   choices=["auto", "external_img", "over_shoulder_img"],
                   help="which get_observation() field holds the blue frame "
                        "(single-camera mode only; --both resolves both keys itself)")
    return p.parse_args()


def pick_blue_frame(obs: dict, obs_key: str):
    """over_shoulder_img if present (dual-cam) else external_img (single-cam)."""
    if obs_key == "auto":
        return obs.get("over_shoulder_img") if obs.get("over_shoulder_img") is not None \
            else obs.get("external_img")
    return obs.get(obs_key)


class Pane:
    """One camera: its reference frame, precomputed edges, and live extraction."""

    def __init__(self, name: str, ref_path: Path, obs_getter):
        import cv2

        if not ref_path.is_file():
            raise FileNotFoundError(f"reference frame not found: {ref_path}")
        self.name = name
        self.ref_bgr = cv2.imread(str(ref_path), cv2.IMREAD_COLOR)
        if self.ref_bgr is None:
            raise ValueError(f"could not decode reference frame: {ref_path}")
        self.h, self.w = self.ref_bgr.shape[:2]
        self.ref_edges = cv2.Canny(cv2.cvtColor(self.ref_bgr, cv2.COLOR_BGR2GRAY), 60, 160)
        self.obs_getter = obs_getter
        print(f"Reference [{name}]: {ref_path}  ({self.w}x{self.h})")

    def live(self, obs):
        import cv2

        frame = None if obs is None else self.obs_getter(obs)
        if frame is None:
            return None
        if frame.shape[:2] != (self.h, self.w):
            frame = cv2.resize(frame, (self.w, self.h), interpolation=cv2.INTER_AREA)
        return frame

    def compose(self, live, mode: str, alpha: float):
        import cv2

        if mode == "l":
            comp = live.copy()
        elif mode == "r":
            comp = self.ref_bgr.copy()
        elif mode == "d":
            comp = cv2.absdiff(live, self.ref_bgr)
        elif mode == "e":
            comp = live.copy()
            comp[self.ref_edges > 0] = (0, 255, 0)          # green reference edges
        else:  # blend
            comp = cv2.addWeighted(live, alpha, self.ref_bgr, 1.0 - alpha, 0.0)

        label = {"l": "LIVE", "r": "REFERENCE", "d": "DIFF (dark=aligned)",
                 "e": "EDGES (match to green)",
                 "b": f"BLEND live={alpha:.2f}"}[mode]
        cv2.putText(comp, f"{self.name}: {label}", (8, 22), cv2.FONT_HERSHEY_SIMPLEX,
                    0.6, (0, 0, 255), 2, cv2.LINE_AA)
        return comp


def build_panes(args) -> tuple[list[Pane], dict]:
    """Panes (left to right) plus the env params to init the robot with."""
    if not args.both:
        pane = Pane("blue (images1)", args.reference,
                    lambda obs: pick_blue_frame(obs, args.obs_key))
        return [pane], DEPLOY_ENV_PARAMS
    # camera_topics order in BOTH_ENV_PARAMS decides the mapping:
    #   full_image[0] -> external_img      -> D435
    #   full_image[1] -> over_shoulder_img -> blue
    blue = Pane("blue (images1)", args.reference,
                lambda obs: obs.get("over_shoulder_img"))
    d435 = Pane("D435 (images0)", args.reference_d435,
                lambda obs: obs.get("external_img"))
    return [blue, d435], BOTH_ENV_PARAMS


def main() -> int:
    import cv2

    args = parse_args()
    panes, env_params = build_panes(args)

    from widowx_envs.widowx_env_service import WidowXClient

    client = WidowXClient(host=args.ip, port=args.port)
    if not args.no_init:
        client.init(env_params, image_size=256)

    waiting = ", ".join(p.name for p in panes)
    while True:
        obs = client.get_observation()
        if obs is not None and all(p.live(obs) is not None for p in panes):
            break
        print(f"Waiting for frames ({waiting})...")
        time.sleep(1.0)

    mode = "e"          # start on edge overlay (most useful for alignment)
    alpha = float(np.clip(args.alpha, 0.0, 1.0))
    win = "pusht camera align  (b/e/d/l/r  [ ]  s  q)"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)
    print("Aligning. Focus the OpenCV window; press 'q' or ESC to quit.")

    try:
        while True:
            obs = client.get_observation()
            lives = [p.live(obs) for p in panes]
            if any(l is None for l in lives):
                continue

            comps = [p.compose(l, mode, alpha) for p, l in zip(panes, lives)]
            comp = comps[0] if len(comps) == 1 else np.hstack(comps)
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
