"""F — does a +dx / +dy action move things the SAME way in the IMAGE at training
and at inference?

corr(command, eef-delta) only checks the eef/base frame is self-consistent. The
policy reasons in IMAGE space: it maps pixels -> action in the TRAINING camera<->
base geometry. If the deploy camera/base relation differs, a +dx command moves
the arm the same way in eef-space but a DIFFERENT way in the image -> the policy
fights a visually rotated/flipped control frame, and corr(cmd,eef) stays +0.9.

Method: dense optical flow between consecutive frames gives the image-space
motion (Δu right+, Δv down+). Over many moving steps we fit the 2x2 map
  [Δu, Δv]^T = M @ [dx, dy]^T
separately for TRAINING (zarr video + actions) and INFERENCE (rollout raw frames
+ commanded actions), and compare M. Matching M (same orientation/signs) =>
consistent action<->image frame. A flipped/rotated M => the bug.

Also dumps annotated before/after pairs with the flow arrow for a clear +dx and
+dy example from each source, so the movement can be verified by eye.

Runs fully local: training frames from the zip, inference frames from a rollout.

Usage:
    python scripts/check_action_image_frame.py \
        --archive data/pusht_widowx_data.zip \
        --log-dir results/roll_c09_base \
        --train-episodes 0 20 40 --move-eps 0.002 --out-dir results/frame_check
"""
import argparse
import glob
import json
import os
import shutil
import tempfile
import zipfile
from pathlib import Path

import cv2
import numpy as np


def flow_vec(prev_bgr, next_bgr):
    """Mean optical-flow vector over the moving region (Δu right+, Δv down+)."""
    g0 = cv2.cvtColor(prev_bgr, cv2.COLOR_RGB2GRAY)
    g1 = cv2.cvtColor(next_bgr, cv2.COLOR_RGB2GRAY)
    flow = cv2.calcOpticalFlowFarneback(g0, g1, None, 0.5, 3, 21, 3, 5, 1.2, 0)
    mag = np.linalg.norm(flow, axis=2)
    thr = max(0.5, np.percentile(mag, 95))          # the moving pixels
    m = mag >= thr
    if m.sum() < 20:
        return None
    return np.array([flow[..., 0][m].mean(), flow[..., 1][m].mean()])


def fit_map(actions, flows):
    """Least-squares M (2x2): flow = M @ action.  Returns M, R^2 per row."""
    A = np.asarray(actions)      # (n,2)
    F = np.asarray(flows)        # (n,2)
    M, *_ = np.linalg.lstsq(A, F, rcond=None)   # A @ M = F  -> M is (2,2), flow=action@M
    pred = A @ M
    ss = 1 - ((F - pred) ** 2).sum(0) / ((F - F.mean(0)) ** 2).sum(0)
    return M.T, ss              # return so that flow = M.T @ action (col j = dir of +action_j)


def collect_training(archive, episodes, move_eps, max_per_ep=120):
    import imageio.v3 as iio
    import zarr
    with zipfile.ZipFile(archive) as ar:
        pre = [n for n in ar.namelist() if "replay_buffer.zarr/" in n][0]
        pre = pre[:pre.find("replay_buffer.zarr/") + len("replay_buffer.zarr/")]
        root0 = pre.split("replay_buffer.zarr/")[0]
        tmp = tempfile.mkdtemp(prefix="fc_z_")
        ar.extractall(tmp, [n for n in ar.namelist() if n.startswith(pre)])
        root = zarr.open(os.path.join(tmp, pre.rstrip("/")), "r")
        act = np.asarray(root["data/action"][:, :2], np.float64)
        ends = np.asarray(root["meta/episode_ends"][:], np.int64)
        shutil.rmtree(tmp, ignore_errors=True)
        starts = np.concatenate([[0], ends[:-1]])
        actions, flows, samples = [], [], []
        scratch = tempfile.mkdtemp(prefix="fc_v_")
        try:
            for ep in episodes:
                member = f"{root0}videos/{ep}/1.mp4"
                ar.extract(member, scratch)
                frames = iio.imread(os.path.join(scratch, member))   # (L,480,640,3) RGB
                s = int(starts[ep])
                L = min(len(frames) - 1, int(ends[ep]) - s - 1)
                moving = [t for t in range(L)
                          if np.linalg.norm(act[s + t]) > move_eps][:max_per_ep]
                for t in moving:
                    fv = flow_vec(frames[t], frames[t + 1])
                    if fv is None:
                        continue
                    actions.append(act[s + t]); flows.append(fv)
                    samples.append((frames[t], frames[t + 1], act[s + t], fv))
                os.remove(os.path.join(scratch, member))
        finally:
            shutil.rmtree(scratch, ignore_errors=True)
    return np.array(actions), np.array(flows), samples


def collect_inference(log_dir, move_eps):
    raws = sorted(glob.glob(f"{log_dir}/raw/*.npy"))
    S = [json.loads(l) for l in open(f"{log_dir}/steps.jsonl")]
    actions, flows, samples = [], [], []
    for t in range(min(len(raws) - 1, len(S))):
        a = np.asarray(S[t]["action"][:2], np.float64)
        if np.linalg.norm(a) <= move_eps:
            continue
        f0 = np.load(raws[t]); f1 = np.load(raws[t + 1])
        fv = flow_vec(f0, f1)
        if fv is None:
            continue
        actions.append(a); flows.append(fv)
        samples.append((f0, f1, a, fv))
    return np.array(actions), np.array(flows), samples


def annotate(sample, path, label):
    f0, f1, a, fv = sample
    img = f1.copy()[..., ::-1].copy()               # RGB->BGR for cv2 draw/write
    h, w = img.shape[:2]
    c = (w // 2, h // 2)
    tip = (int(c[0] + fv[0] * 8), int(c[1] + fv[1] * 8))
    cv2.arrowedLine(img, c, tip, (0, 0, 255), 2, tipLength=0.3)
    cv2.putText(img, f"{label} act=({a[0]*1000:+.1f},{a[1]*1000:+.1f})mm "
                f"flow=({fv[0]:+.1f},{fv[1]:+.1f})px", (8, 22),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
    cv2.imwrite(str(path), img)


def report(name, M, ss, n):
    print(f"\n=== {name}  (n={n}) ===")
    print(f"  flow = M @ action ;  M (px per unit-norm action):")
    print(f"    +dx -> image (Δu,Δv) = ({M[0,0]:+.0f}, {M[1,0]:+.0f}) px  "
          f"[{'RIGHT' if M[0,0]>0 else 'LEFT'}/{'DOWN' if M[1,0]>0 else 'UP'}]")
    print(f"    +dy -> image (Δu,Δv) = ({M[0,1]:+.0f}, {M[1,1]:+.0f}) px  "
          f"[{'RIGHT' if M[0,1]>0 else 'LEFT'}/{'DOWN' if M[1,1]>0 else 'UP'}]")
    print(f"    fit R^2: u={ss[0]:.2f} v={ss[1]:.2f}")


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--archive", type=Path, default=Path("data/pusht_widowx_data.zip"))
    p.add_argument("--log-dir", type=Path, required=True)
    p.add_argument("--train-episodes", type=int, nargs="+", default=[0, 20, 40])
    p.add_argument("--move-eps", type=float, default=0.002)
    p.add_argument("--out-dir", type=Path, default=Path("results/frame_check"))
    args = p.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    ta, tf, tsamp = collect_training(args.archive, args.train_episodes, args.move_eps)
    ia, ifl, isamp = collect_inference(args.log_dir, args.move_eps)
    if len(ta) < 5 or len(ia) < 5:
        raise SystemExit(f"too few moving samples: train={len(ta)} infer={len(ia)}")
    Mt, sst = fit_map(ta, tf)
    Mi, ssi = fit_map(ia, ifl)
    report("TRAINING", Mt, sst, len(ta))
    report("INFERENCE", Mi, ssi, len(ia))

    # compare orientation: cosine between the +dx image-dir (and +dy) across sources
    def cos(u, v):
        return float(np.dot(u, v) / (np.linalg.norm(u) * np.linalg.norm(v) + 1e-9))
    cdx = cos(Mt[:, 0], Mi[:, 0]); cdy = cos(Mt[:, 1], Mi[:, 1])
    print(f"\n=== FRAME MATCH ===")
    print(f"  +dx image-direction cosine(train, infer) = {cdx:+.2f}")
    print(f"  +dy image-direction cosine(train, infer) = {cdy:+.2f}")
    if cdx > 0.7 and cdy > 0.7:
        print("  VERDICT: CONSISTENT — action↔image frame matches training. Not the bug.")
    elif cdx < 0 or cdy < 0:
        print("  VERDICT: FLIPPED — an axis is inverted in the image at deploy. "
              "The policy is fighting a mirrored control frame. THIS is a bug.")
    else:
        print("  VERDICT: ROTATED/INCONSISTENT — image mapping differs from training.")

    # visual proof: clearest +dx and +dy example from each source
    def pick(samples, axis, sign):
        best, bi = -1, None
        for s in samples:
            a = s[2]
            if np.sign(a[axis]) == sign and abs(a[axis]) > abs(a[1 - axis]):
                if abs(a[axis]) > best:
                    best, bi = abs(a[axis]), s
        return bi
    for src, samples in [("train", tsamp), ("infer", isamp)]:
        for ax, nm in [(0, "posdx"), (1, "posdy")]:
            s = pick(samples, ax, +1)
            if s is not None:
                annotate(s, args.out_dir / f"{src}_{nm}.png", f"{src} +{'dx' if ax==0 else 'dy'}")
    print(f"\nAnnotated flow images -> {args.out_dir}/  "
          f"(compare train_posdx.png vs infer_posdx.png by eye)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
