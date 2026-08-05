"""Measure how much of the drawn target T is *not* covered by the red T block.

The Push-T rig has the goal pose drawn in pencil on the white table. A rollout
"succeeds" when the red block sits inside that drawing; the residual error is
the white paper still visible inside the drawn outline. This script turns that
into a number, from the two fixed cameras (``images0`` == D435, ``images1`` ==
blue scene cam).

Camera pose and lighting on this rig are fixed, so everything static is solved
once and reused forever; only the block moves. Calibration has already been
run, and its result lives in this file (``TARGET_POLYGONS``) plus two 30 KB
reference images in ``scripts/assets/``. **Measuring needs nothing else** -- no
dataset, no mask files, no calibration directory. To deploy, copy this script
and ``assets/pusht_target_bg_images{0,1}.jpg`` next to it.

``calibrate``
    Only needed if a camera is moved, the lighting changes, or the target is
    redrawn -- and then run against a collection session. Produces, per
    camera: a binary mask of the drawn target region, and a median "empty
    table" reference image. Pencil lines are thin, low contrast, and partly
    occluded by the block in any single frame, so the auto mode accumulates a
    "this pixel is a dark line" vote over many frames (skipping pixels that are
    red in that frame) and only then closes / flood-fills the outline. Feed it
    a whole session: the block visits different places, so the union of the
    un-occluded views recovers the complete outline, and the per-pixel median
    over the same frames is a clean view of the table with the block edited
    out. ``--emit-constants`` then prints the replacement ``TARGET_POLYGONS``
    and ``--update-assets`` refreshes the two reference images, which is the
    whole handover back to the deploy side.

``measure``
    For a frame (or a whole trajectory), segment the red block in HSV and
    report the fraction of the target mask it covers, plus the split of the
    uncovered remainder into bare paper vs. other occlusion (gripper, arm).
    With a reference image present, "bare paper" is decided per pixel against
    what that pixel looks like when empty -- so paper in the block's cast
    shadow still counts as uncovered paper instead of falling into "other" --
    and a drift check warns if the live view no longer matches the reference,
    which is the one way the fixed-rig assumption can silently break.

Bias worth knowing before quoting the number: both cameras look at the table
at an angle and the block is tall, so its *silhouette* strictly contains its
footprint on the paper. Overlap measured from a silhouette is therefore an
over-estimate of the true covered area, i.e. ``uncovered_frac`` is a lower
bound on the real uncovered area. The per-camera numbers bracket the truth --
``max(uncovered_frac)`` over cam0/cam1 is the tightest such bound, and that is
what ``combined.uncovered_frac_lower_bound`` reports. Areas are pixel areas in
each camera's own perspective, so the two cameras will not agree exactly.

Examples
--------
On the robot, from a pair of grabbed frames::

    python measure_target_coverage.py measure \
        --image images0=/tmp/cam0.jpg --image images1=/tmp/cam1.jpg

Against the training zip, last frame of a trajectory, with annotated PNGs::

    python scripts/measure_target_coverage.py measure \
        --zip data/pusht_2026_07.zip --session 2026-07-30_10-19-38 \
        --traj traj5 --frame -1 --save-overlay /tmp/cov

Whole trajectory as a CSV time series::

    python scripts/measure_target_coverage.py measure \
        --zip data/pusht_2026_07.zip --session 2026-07-30_10-19-38 \
        --traj traj0 --all-frames --csv /tmp/traj0_coverage.csv

Re-calibrating, if the rig ever changes::

    python scripts/measure_target_coverage.py calibrate \
        --zip data/pusht_2026_07.zip --session 2026-07-30_10-19-38 \
        --stride 4 --out /tmp/calib --emit-constants --update-assets

That does both cameras; cam0 gets ``--roi 220,30,570,310`` by default, because
the robot base and the table edge form their own big closed dark outlines and
would otherwise win the "largest filled region" vote. Check
``/tmp/calib/<camera>_target_overlay.png`` -- the mask is drawn on the
empty-table median, so a mask that clipped part of the T is obvious -- then
paste the printed constants over ``TARGET_POLYGONS`` below.

If the auto mask comes out wrong (cam0 is the hard one -- part of its view of
the drawing is permanently occluded), draw it by hand instead::

    python scripts/measure_target_coverage.py calibrate --mode manual \
        --camera images0 --zip data/pusht_2026_07.zip \
        --session 2026-07-30_10-19-38 --out /tmp/calib --emit-constants

and click the corners of the drawn T (right-click undo, middle-click finish),
or pass the corners headlessly with
``--mode points --points "x0,y0 x1,y1 ..."``. A ``--masks <dir>`` written by
``calibrate`` overrides the built-ins if you want to test one without editing
this file.

Masks calibrated on 2026-07-27 and on 2026-07-30 land on the same drawing, as
the fixed rig implies. For reference, expert demos in ``data/pusht_2026_07.zip``
end at roughly 2-7% uncovered.
"""

from __future__ import annotations

import argparse
import csv
import fnmatch
import json
import os
import re
import sys
import zipfile
from dataclasses import dataclass, asdict
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np

CAMERAS = ("images0", "images1")
# The rig does not move, so the search window for cam0's drawing is a constant.
# cam0 sees the robot base and the table edge, which are closed dark outlines
# far larger than the target; without this they win the fill.
DEFAULT_ROI = {"images0": (220, 30, 570, 310)}

ASSETS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "assets")
BACKGROUND_ASSETS = {cam: os.path.join(ASSETS, f"pusht_target_bg_{cam}.jpg")
                     for cam in CAMERAS}

# The drawn target in 640x480 camera pixels, traced from the 2026-07 rig
# (calibrated on session 2026-07-30_10-19-38, within 2% IoU of the pixel mask
# it was simplified from). Compiled in so the script runs anywhere with no
# dataset and no mask files -- see `calibrate --emit-constants` to regenerate
# these numbers if a camera is ever moved or the target redrawn.
TARGET_POLYGONS: Dict[str, Tuple[Tuple[int, int], ...]] = {
    "images0": (
        (467, 105), (420, 93), (345, 150), (318, 150), (277, 136), (242, 160), (411, 229),
        (416, 234), (426, 234), (455, 200), (407, 183), (406, 158),
    ),
    "images1": (
        (254, 137), (78, 171), (84, 215), (128, 206), (151, 207), (174, 345), (228, 334),
        (231, 330), (223, 300), (223, 277), (227, 274), (219, 274), (206, 218), (206, 192),
        (249, 182), (264, 182),
    ),
}
POLYGON_IMAGE_SIZE = (480, 640)  # (h, w) the polygons were traced in

# Under this rig's fixed lighting nothing belonging to the table is this dark:
# the table sits around V ~ 140 and the pencil strokes around V ~ 100, so V
# below this is the arm, the gripper, or another foreign object.
DARK_V = 70
ZIP_ROOT_RE = re.compile(r"^[^/]+/(?P<session>[^/]+)/raw/traj_group\d+/(?P<traj>traj\d+)/")


# --------------------------------------------------------------------------
# frame sources
# --------------------------------------------------------------------------
class FrameSource:
    """Yields ``(name, bgr)`` pairs for one camera."""

    def __init__(self, entries: List[Tuple[str, Callable[[], Optional[np.ndarray]]]]):
        self._entries = entries

    def __len__(self) -> int:
        return len(self._entries)

    def names(self) -> List[str]:
        return [n for n, _ in self._entries]

    def __iter__(self):
        for name, loader in self._entries:
            img = loader()
            if img is not None:
                yield name, img

    def get(self, index: int) -> Tuple[str, np.ndarray]:
        name, loader = self._entries[index]
        img = loader()
        if img is None:
            raise RuntimeError(f"failed to decode {name}")
        return name, img


def _decode(buf: bytes) -> Optional[np.ndarray]:
    arr = np.frombuffer(buf, np.uint8)
    return cv2.imdecode(arr, cv2.IMREAD_COLOR)


def _frame_index(path: str) -> int:
    m = re.search(r"im_(\d+)\.jpg$", path)
    return int(m.group(1)) if m else 0


def zip_source(
    zip_path: str,
    camera: str,
    session: Optional[str] = None,
    traj: Optional[str] = None,
) -> FrameSource:
    zf = zipfile.ZipFile(zip_path)
    entries = []
    for name in zf.namelist():
        if f"/{camera}/im_" not in name or not name.endswith(".jpg"):
            continue
        m = ZIP_ROOT_RE.match(name)
        if m is None:
            continue
        if session and not fnmatch.fnmatch(m.group("session"), session):
            continue
        if traj and m.group("traj") != traj:
            continue
        entries.append(name)
    entries.sort(key=lambda p: (p.rsplit("/", 3)[0], _frame_index(p)))
    return FrameSource([(n, (lambda n=n: _decode(zf.read(n)))) for n in entries])


def dir_source(root: str, camera: Optional[str] = None) -> FrameSource:
    paths: List[str] = []
    if os.path.isfile(root):
        paths = [root]
    else:
        for dirpath, _, filenames in os.walk(root):
            if camera and os.path.basename(dirpath) != camera:
                continue
            for fn in filenames:
                if fn.lower().endswith((".jpg", ".jpeg", ".png")):
                    paths.append(os.path.join(dirpath, fn))
    paths.sort(key=lambda p: (os.path.dirname(p), _frame_index(p)))
    return FrameSource([(p, (lambda p=p: cv2.imread(p, cv2.IMREAD_COLOR))) for p in paths])


# --------------------------------------------------------------------------
# segmentation primitives
# --------------------------------------------------------------------------
def red_mask(bgr: np.ndarray, sat_min: int = 70, val_min: int = 10,
             hue_tol: int = 12) -> np.ndarray:
    """Binary mask of the red block (its full silhouette, sides included).

    ``val_min`` is deliberately low: the block's shaded side faces sit as low
    as V ~ 10 while still being strongly red (S ~ 240), and dropping them reads
    as "uncovered target". Shadow on bare paper stays out because it is grey
    (S ~ 5-50), under ``sat_min``.
    """
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    h, s, v = hsv[..., 0], hsv[..., 1], hsv[..., 2]
    m = ((h < hue_tol) | (h > 180 - hue_tol)) & (s > sat_min) & (v > val_min)
    m = m.astype(np.uint8)
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
    return m


def paper_mask(bgr: np.ndarray, background: Optional[np.ndarray] = None,
               val_min: int = 120, sat_max: int = 60,
               shadow_ratio: float = 0.45) -> np.ndarray:
    """Pixels showing bare table, shadowed or not.

    Without a reference, this is just "bright and grey", which throws away
    paper lying in the block's cast shadow. With the fixed-rig reference image
    the test becomes per-pixel: still grey, and no darker than ``shadow_ratio``
    of what this very pixel looks like when the table is empty. Cast shadow
    costs roughly half the brightness, the gripper and the arm cost far more.
    """
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    grey = hsv[..., 1] < sat_max
    if background is None:
        return (grey & (hsv[..., 2] > val_min)).astype(np.uint8)
    bg_v = cv2.cvtColor(background, cv2.COLOR_BGR2HSV)[..., 2].astype(np.float32)
    return (grey & (hsv[..., 2] >= shadow_ratio * bg_v)).astype(np.uint8)


def dark_line_mask(bgr: np.ndarray, sigma: float = 9.0, delta: float = 8.0) -> np.ndarray:
    """Pencil strokes: darker than the local background by ``delta`` grey levels."""
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY).astype(np.float32)
    bg = cv2.GaussianBlur(gray, (0, 0), sigma)
    return ((bg - gray) > delta).astype(np.uint8)


# --------------------------------------------------------------------------
# calibration
# --------------------------------------------------------------------------
@dataclass
class CalibParams:
    vote_thresh: float = 0.4
    dilate: int = 3
    close: int = 15
    line_delta: float = 8.0
    red_pad: int = 25
    min_obs: int = 10
    max_area_frac: float = 0.25
    min_area_px: int = 500
    bg_dark_v: int = DARK_V


@dataclass
class Accumulation:
    """What one sweep over a session's frames yields for a fixed camera."""

    ink_frac: np.ndarray          # per-pixel fraction of unoccluded views showing ink
    n_obs: np.ndarray             # unoccluded views per pixel
    background: np.ndarray        # median empty-table view, block edited out
    clearest: np.ndarray          # single frame with the least red in it
    clearest_name: str
    n_frames: int


def _masked_median(stack: np.ndarray, occluded: np.ndarray, rows: int = 60) -> np.ndarray:
    """Per-pixel median over the views where the pixel was not occluded.

    Plain median would bake the block into the target area -- demos park it
    there for a good share of every episode. Row-chunked because the nan-filled
    float copy of the whole stack does not need to exist at once.
    """
    out = np.zeros(stack.shape[1:], np.uint8)
    fallback = np.median(stack, axis=0)
    for r0 in range(0, stack.shape[1], rows):
        r1 = min(r0 + rows, stack.shape[1])
        chunk = stack[:, r0:r1].astype(np.float32)
        chunk[occluded[:, r0:r1]] = np.nan
        with np.errstate(invalid="ignore"):
            med = np.nanmedian(chunk, axis=0)
        blind = ~np.isfinite(med)
        med[blind] = fallback[r0:r1][blind]
        out[r0:r1] = np.clip(med, 0, 255).astype(np.uint8)
    return out


def accumulate_line_votes(
    source: FrameSource, params: CalibParams, stride: int = 1,
    max_frames: int = 4000, bg_frames: int = 150
) -> Accumulation:
    """Sweep the frames once: ink votes, empty-table median, clearest frame.

    A pixel that is red (block) in a frame -- padded, to also drop the block's
    shadow edge -- is not counted as an ink observation, so a stroke the block
    sits on in half the frames still reaches frac ~ 1. Use every frame you
    have: a stroke only resolves where *some* frame saw it, and gaps are what
    break the fill. The background median needs far fewer frames than that, so
    only ``bg_frames`` of them are kept in memory for it.
    """
    names = source.names()
    if stride < 1:
        stride = 1
    kept = names[::stride][:max_frames]
    keep = set(kept)
    bg_keep = set(kept[::max(1, len(kept) // bg_frames)][:bg_frames])

    votes = obs = None
    used = 0
    clearest, clearest_name, fewest_red = None, "", None
    bg_stack: List[np.ndarray] = []
    bg_occ: List[np.ndarray] = []
    for name, bgr in source:
        if name not in keep:
            continue
        dark = dark_line_mask(bgr, delta=params.line_delta)
        red = red_mask(bgr)
        n_red = int(red.sum())
        if fewest_red is None or n_red < fewest_red:
            clearest, clearest_name, fewest_red = bgr.copy(), name, n_red
        if params.red_pad > 1:
            red = cv2.dilate(red, np.ones((params.red_pad, params.red_pad), np.uint8))
        valid = red == 0
        if votes is None:
            votes = np.zeros(dark.shape, np.float32)
            obs = np.zeros(dark.shape, np.float32)
        votes += (dark > 0) & valid
        obs += valid
        used += 1
        if name in bg_keep:
            # For the empty-table median, "occluded" is the block *and* anything
            # near-black: the arm and gripper are not red, and under this rig's
            # fixed lighting nothing belonging to the table is that dark (the
            # pencil strokes sit around V ~ 100, the table around V ~ 140).
            dark_obj = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)[..., 2] < params.bg_dark_v
            bg_stack.append(bgr)
            bg_occ.append((red > 0) | dark_obj)
    if votes is None:
        raise RuntimeError("no frames read for calibration")
    frac = votes / np.maximum(obs, 1.0)
    frac[obs < params.min_obs] = 0.0
    background = _masked_median(np.stack(bg_stack),
                                np.stack(bg_occ)[..., None].repeat(3, axis=3))
    print(f"  accumulated {used} frames ({len(bg_stack)} into the background median)",
          file=sys.stderr)
    return Accumulation(ink_frac=frac, n_obs=obs, background=background,
                        clearest=clearest, clearest_name=clearest_name, n_frames=used)


def fill_outline(frac: np.ndarray, params: CalibParams,
                 roi: Optional[Tuple[int, int, int, int]] = None) -> Optional[np.ndarray]:
    """Close the (possibly gappy) stroke image and fill the enclosed T."""
    work = frac.copy()
    if roi is not None:
        x0, y0, x1, y1 = roi
        keep = np.zeros(work.shape, bool)
        keep[y0:y1, x0:x1] = True
        work[~keep] = 0.0

    lines = (work > params.vote_thresh).astype(np.uint8)
    if params.dilate > 1:
        lines = cv2.dilate(lines, np.ones((params.dilate,) * 2, np.uint8))
    closed = cv2.morphologyEx(lines, cv2.MORPH_CLOSE,
                              np.ones((params.close,) * 2, np.uint8))

    h, w = closed.shape
    flood = closed.copy()
    cv2.floodFill(flood, np.zeros((h + 2, w + 2), np.uint8), (0, 0), 1)
    filled = ((flood == 0) | (closed > 0)).astype(np.uint8)

    # The drawing is two overlapping rectangles, so its interior is several
    # holes; taking the largest connected component of (holes + strokes) glues
    # them back into one T and drops the small registration square.
    n, labels, stats, _ = cv2.connectedComponentsWithStats(filled, 8)
    max_area = params.max_area_frac * h * w
    best, best_area = None, 0
    for i in range(1, n):
        area = int(stats[i, cv2.CC_STAT_AREA])
        if area > max_area or area < params.min_area_px:
            continue
        if area > best_area:
            best, best_area = i, area
    if best is None:
        return None
    mask = (labels == best).astype(np.uint8)
    if params.dilate > 1:  # undo the stroke dilation
        mask = cv2.erode(mask, np.ones((params.dilate,) * 2, np.uint8))
    return cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))


def auto_mask(frac: np.ndarray, params: CalibParams,
              roi: Optional[Tuple[int, int, int, int]],
              sweep: bool) -> Tuple[np.ndarray, CalibParams]:
    """Fill with the given params, or sweep a small grid and keep the best fill.

    "Best" == largest plausible area: an incomplete outline leaks into the
    background and gets rejected by ``max_area_frac``, while a too-aggressive
    close only ever adds area, so the largest surviving candidate is the one
    whose outline actually closed.
    """
    candidates: List[CalibParams] = [params]
    if sweep:
        candidates = []
        for thresh in (0.5, 0.4, 0.3):
            for dil in (1, 3, 5):
                for close in (9, 15, 21):
                    candidates.append(CalibParams(
                        vote_thresh=thresh, dilate=dil, close=close,
                        line_delta=params.line_delta, red_pad=params.red_pad,
                        min_obs=params.min_obs, max_area_frac=params.max_area_frac,
                        min_area_px=params.min_area_px))

    best_mask, best_params, best_area = None, None, 0
    for cand in candidates:
        mask = fill_outline(frac, cand, roi)
        if mask is None:
            continue
        area = int(mask.sum())
        if area > best_area:
            best_mask, best_params, best_area = mask, cand, area
    if best_mask is None:
        raise RuntimeError(
            "auto calibration found no closed outline -- restrict the search with "
            "--roi, lower --vote-thresh, or fall back to --mode manual/points")
    return best_mask, best_params


def polygon_mask(points: Sequence[Tuple[float, float]], shape: Tuple[int, int]) -> np.ndarray:
    mask = np.zeros(shape, np.uint8)
    pts = np.array(points, np.int32).reshape(-1, 1, 2)
    cv2.fillPoly(mask, [pts], 1)
    return mask


def mask_to_polygon(mask: np.ndarray, epsilon: float = 2.0) -> List[Tuple[int, int]]:
    """Outer contour of a mask, simplified enough to paste into source."""
    contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return []
    biggest = max(contours, key=cv2.contourArea)
    approx = cv2.approxPolyDP(biggest, epsilon, True).reshape(-1, 2)
    return [(int(x), int(y)) for x, y in approx]


def format_constants(polygons: Dict[str, List[Tuple[int, int]]],
                     shape: Tuple[int, int]) -> str:
    """The TARGET_POLYGONS literal, ready to replace the one in this file."""
    out = ["TARGET_POLYGONS = {"]
    for camera, pts in polygons.items():
        out.append(f'    "{camera}": (')
        line = "        "
        for x, y in pts:
            piece = f"({x}, {y}), "
            if len(line) + len(piece) > 92:
                out.append(line.rstrip())
                line = "        "
            line += piece
        out.append(line.rstrip().rstrip(",") + ",")
        out.append("    ),")
    out.append("}")
    out.append(f"POLYGON_IMAGE_SIZE = ({shape[0]}, {shape[1]})")
    return "\n".join(out)


def click_polygon(bgr: np.ndarray, camera: str) -> List[Tuple[float, float]]:
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(10, 7.5))
    ax.imshow(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
    ax.set_title(f"{camera}: click the corners of the drawn T\n"
                 "left = add, right = undo, middle (or Enter) = done")
    pts = plt.ginput(n=-1, timeout=0)
    plt.close(fig)
    if len(pts) < 3:
        raise RuntimeError("need at least 3 points")
    return [(float(x), float(y)) for x, y in pts]


def save_masks(out_dir: str, camera: str, mask: np.ndarray, meta: Dict) -> str:
    os.makedirs(out_dir, exist_ok=True)
    np.save(os.path.join(out_dir, f"{camera}_target.npy"), mask.astype(np.uint8))
    cv2.imwrite(os.path.join(out_dir, f"{camera}_target.png"), mask * 255)
    meta_path = os.path.join(out_dir, "targets.json")
    all_meta = {}
    if os.path.exists(meta_path):
        with open(meta_path) as fh:
            all_meta = json.load(fh)
    all_meta[camera] = meta
    with open(meta_path, "w") as fh:
        json.dump(all_meta, fh, indent=2)
    return meta_path


def builtin_mask(camera: str, shape: Tuple[int, int]) -> np.ndarray:
    """The compiled-in target polygon, scaled to this frame size."""
    pts = TARGET_POLYGONS.get(camera)
    if pts is None:
        raise KeyError(f"no built-in target polygon for {camera}")
    sy = shape[0] / POLYGON_IMAGE_SIZE[0]
    sx = shape[1] / POLYGON_IMAGE_SIZE[1]
    scaled = [(x * sx, y * sy) for x, y in pts]
    return polygon_mask(scaled, shape)


def load_mask(masks_dir: Optional[str], camera: str, shape: Tuple[int, int]) -> np.ndarray:
    """A calibration directory if one is around, otherwise the built-in polygon."""
    path = os.path.join(masks_dir, f"{camera}_target.npy") if masks_dir else ""
    if not path or not os.path.exists(path):
        return builtin_mask(camera, shape)
    mask = np.load(path).astype(np.uint8)
    if mask.shape != shape:
        mask = cv2.resize(mask, (shape[1], shape[0]), interpolation=cv2.INTER_NEAREST)
    return mask


def load_background(masks_dir: Optional[str], camera: str,
                    shape: Tuple[int, int]) -> Optional[np.ndarray]:
    """Empty-table reference: calibration directory first, shipped asset second."""
    candidates = []
    if masks_dir:
        candidates.append(os.path.join(masks_dir, f"{camera}_background.png"))
    candidates.append(BACKGROUND_ASSETS.get(camera, ""))
    for path in candidates:
        if not path or not os.path.exists(path):
            continue
        bg = cv2.imread(path, cv2.IMREAD_COLOR)
        if bg is None:
            continue
        if bg.shape[:2] != shape:
            bg = cv2.resize(bg, (shape[1], shape[0]), interpolation=cv2.INTER_AREA)
        return bg
    return None


class ShiftChecker:
    """How far the view has translated since calibration, by phase correlation.

    The whole calibration -- target mask included -- is pinned to the camera
    pose, so this is the check that the fixed-rig assumption still holds.
    Measured on this dataset: ~0.2 px within a session and ~0.1 px across days,
    against 10.6 px for a frame shifted 9,6 px on purpose.

    Three details keep it honest on a table this featureless, where anything
    high-contrast that has moved can steal the correlation peak (leaving the
    block in produced a bogus 21 px reading on a frame the rig had not moved):

    * everything that moves -- the block, and the near-black arm and gripper --
      is painted out with the reference's own pixels,
    * matching runs on gradient magnitude, so the pencil lines and table edges
      carry the signal rather than the flat white,
    * the target area is windowed out -- with the block parked on the drawing
      there is nothing left to match there anyway.

    Over a 618-frame episode that leaves max 0.6 px of noise, against 11 px for
    a frame shifted 9,6 px on purpose. Skipping the arm alone is worth it: with
    only the block painted out, 3 frames of that episode read over 100 px.
    """

    def __init__(self, background: np.ndarray, target: np.ndarray):
        self.background = background
        h, w = background.shape[:2]
        periphery = cv2.dilate((target > 0).astype(np.uint8),
                               np.ones((41, 41), np.uint8)) == 0
        self.window = (cv2.createHanningWindow((w, h), cv2.CV_32F)
                       * periphery.astype(np.float32))
        self.ref_edges = self._edges(background) * self.window

    @staticmethod
    def _edges(bgr: np.ndarray) -> np.ndarray:
        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY).astype(np.float32)
        return cv2.magnitude(cv2.Sobel(gray, cv2.CV_32F, 1, 0, 3),
                             cv2.Sobel(gray, cv2.CV_32F, 0, 1, 3))

    def __call__(self, bgr: np.ndarray, red: Optional[np.ndarray] = None) -> float:
        dark = (cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)[..., 2] < DARK_V).astype(np.uint8)
        moving = dark if red is None else np.maximum(dark, red)
        moving = cv2.dilate(moving, np.ones((9, 9), np.uint8)) > 0
        live = bgr.copy()
        live[moving] = self.background[moving]
        (dx, dy), _ = cv2.phaseCorrelate(self.ref_edges, self._edges(live) * self.window)
        return float(np.hypot(dx, dy))


def overlay(bgr: np.ndarray, target: np.ndarray, red: Optional[np.ndarray] = None) -> np.ndarray:
    """Green = uncovered target, orange = covered target, blue outline = block."""
    out = bgr.copy()
    if red is None:
        tint = out.copy()
        tint[target > 0] = (0, 255, 0)
        return cv2.addWeighted(out, 0.6, tint, 0.4, 0)
    covered = (target > 0) & (red > 0)
    uncovered = (target > 0) & (red == 0)
    tint = out.copy()
    tint[uncovered] = (0, 255, 0)
    tint[covered] = (0, 140, 255)
    out = cv2.addWeighted(out, 0.55, tint, 0.45, 0)
    contours, _ = cv2.findContours(red, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(out, contours, -1, (255, 0, 0), 1)
    return out


# --------------------------------------------------------------------------
# measurement
# --------------------------------------------------------------------------
def measure_frame(bgr: np.ndarray, target: np.ndarray,
                  red_kw: Optional[Dict] = None,
                  background: Optional[np.ndarray] = None) -> Dict:
    red = red_mask(bgr, **(red_kw or {}))
    paper = paper_mask(bgr, background)
    tgt = target > 0
    n_target = int(tgt.sum())
    if n_target == 0:
        raise RuntimeError("empty target mask")
    covered = int((tgt & (red > 0)).sum())
    uncovered = n_target - covered
    bare = int((tgt & (red == 0) & (paper > 0)).sum())
    return {
        "target_px": n_target,
        "covered_px": covered,
        "uncovered_px": uncovered,
        "uncovered_paper_px": bare,
        "uncovered_other_px": uncovered - bare,
        "covered_frac": covered / n_target,
        "uncovered_frac": uncovered / n_target,
        "uncovered_paper_frac": bare / n_target,
    }, red


def _fmt(res: Dict) -> str:
    return (f"uncovered {100 * res['uncovered_frac']:5.1f}%  "
            f"(bare paper {100 * res['uncovered_paper_frac']:5.1f}%, "
            f"other {100 * (res['uncovered_frac'] - res['uncovered_paper_frac']):4.1f}%)  "
            f"covered {100 * res['covered_frac']:5.1f}%  "
            f"[{res['target_px']} target px]")


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------
def parse_roi(text: Optional[str]) -> Optional[Tuple[int, int, int, int]]:
    if not text:
        return None
    vals = [int(v) for v in re.split(r"[,\s]+", text.strip()) if v]
    if len(vals) != 4:
        raise argparse.ArgumentTypeError("--roi wants x0,y0,x1,y1")
    return tuple(vals)  # type: ignore[return-value]


def parse_points(text: str) -> List[Tuple[float, float]]:
    vals = [float(v) for v in re.split(r"[,\s]+", text.strip()) if v]
    if len(vals) < 6 or len(vals) % 2:
        raise argparse.ArgumentTypeError("--points wants an even list of >=3 x,y pairs")
    return list(zip(vals[0::2], vals[1::2]))


def build_source(args, camera: str) -> FrameSource:
    if args.zip:
        return zip_source(args.zip, camera, session=args.session, traj=getattr(args, "traj", None))
    if args.images:
        return dir_source(args.images, camera if os.path.isdir(args.images) else None)
    raise SystemExit("need --zip or --images")


def cmd_calibrate(args) -> int:
    params = CalibParams(vote_thresh=args.vote_thresh, dilate=args.dilate,
                         close=args.close, line_delta=args.line_delta,
                         red_pad=args.red_pad, min_obs=args.min_obs,
                         max_area_frac=args.max_area_frac,
                         min_area_px=args.min_area_px)
    explicit_roi = parse_roi(args.roi)
    cameras = [args.camera] if args.camera else list(CAMERAS)
    polygons: Dict[str, List[Tuple[int, int]]] = {}
    poly_shape = POLYGON_IMAGE_SIZE

    os.makedirs(args.out, exist_ok=True)
    for camera in cameras:
        roi = explicit_roi or DEFAULT_ROI.get(camera)
        source = build_source(args, camera)
        if len(source) == 0:
            print(f"{camera}: no frames found, skipping", file=sys.stderr)
            continue
        print(f"{camera}: {len(source)} frames available", file=sys.stderr)
        ref_name, ref = source.get(len(source) // 2)
        shape = ref.shape[:2]

        meta = {"mode": args.mode, "reference_frame": ref_name,
                "image_size": [int(shape[0]), int(shape[1])],
                "roi": list(roi) if roi else None}

        if args.mode == "auto":
            acc = accumulate_line_votes(source, params, stride=args.stride,
                                        max_frames=args.max_frames,
                                        bg_frames=args.bg_frames)
            mask, used = auto_mask(acc.ink_frac, params, roi, sweep=not args.no_sweep)
            meta["params"] = asdict(used)
            meta["calibration_frames"] = acc.n_frames
            meta["clearest_frame"] = acc.clearest_name
            # The empty-table median is the honest preview: nothing occludes the
            # drawing in it, so a mask that misses part of the T is obvious.
            ref, ref_name = acc.background, "background median"
            meta["reference_frame"] = ref_name
            cv2.imwrite(os.path.join(args.out, f"{camera}_background.png"), acc.background)
            if args.update_assets:
                asset = BACKGROUND_ASSETS.get(camera)
                if asset:
                    os.makedirs(os.path.dirname(asset), exist_ok=True)
                    cv2.imwrite(asset, acc.background, [cv2.IMWRITE_JPEG_QUALITY, 92])
                    print(f"{camera}: reference asset -> {asset}", file=sys.stderr)
            if args.save_votes:
                cv2.imwrite(os.path.join(args.out, f"{camera}_line_votes.png"),
                            (acc.ink_frac * 255).astype(np.uint8))
        elif args.mode == "points":
            if not args.points:
                raise SystemExit("--mode points needs --points")
            pts = parse_points(args.points)
            mask = polygon_mask(pts, shape)
            meta["points"] = pts
        else:  # manual
            pts = click_polygon(ref, camera)
            mask = polygon_mask(pts, shape)
            meta["points"] = pts

        area = int(mask.sum())
        meta["target_px"] = area
        meta["target_frac_of_image"] = area / float(shape[0] * shape[1])
        os.makedirs(args.out, exist_ok=True)
        cv2.imwrite(os.path.join(args.out, f"{camera}_target_overlay.png"),
                    overlay(ref, mask))
        save_masks(args.out, camera, mask, meta)
        polygons[camera] = mask_to_polygon(mask, args.polygon_epsilon)
        poly_shape = (int(shape[0]), int(shape[1]))
        print(f"{camera}: target mask {area} px "
              f"({100 * meta['target_frac_of_image']:.2f}% of image) -> {args.out}",
              file=sys.stderr)
        print(f"{camera}: check {args.out}/{camera}_target_overlay.png before trusting it",
              file=sys.stderr)

    if args.emit_constants and polygons:
        print("\n# paste over TARGET_POLYGONS / POLYGON_IMAGE_SIZE in this script:\n")
        print(format_constants(polygons, poly_shape))
    return 0


def cmd_measure(args) -> int:
    per_camera: Dict[str, List[Dict]] = {}
    red_kw = {"sat_min": args.sat_min, "val_min": args.val_min, "hue_tol": args.hue_tol}

    explicit = {}
    for spec in args.image or []:
        if "=" in spec:
            cam, path = spec.split("=", 1)
        else:
            cam, path = "images0", spec
        explicit[cam] = path

    cameras = [args.camera] if args.camera else (list(explicit) or list(CAMERAS))
    for camera in cameras:
        if camera in explicit:
            source = dir_source(explicit[camera])
        else:
            try:
                source = build_source(args, camera)
            except SystemExit:
                raise
        if len(source) == 0:
            print(f"{camera}: no frames found, skipping", file=sys.stderr)
            continue

        indices = range(len(source)) if args.all_frames else [args.frame % len(source)]
        rows = []
        drift_warned = False
        checker: Optional[ShiftChecker] = None
        for i in indices:
            name, bgr = source.get(i)
            target = load_mask(args.masks, camera, bgr.shape[:2])
            background = None if args.no_background else load_background(
                args.masks, camera, bgr.shape[:2])
            res, red = measure_frame(bgr, target, red_kw, background)
            res["frame"] = name
            res["camera"] = camera
            if background is not None:
                if checker is None:
                    checker = ShiftChecker(background, target)
                res["camera_shift_px"] = checker(bgr, red)
                if res["camera_shift_px"] > args.max_shift and not drift_warned:
                    drift_warned = True
                    print(f"{camera}: WARNING view has moved {res['camera_shift_px']:.1f} px "
                          f"since calibration (limit {args.max_shift}) -- the target mask no "
                          f"longer lines up, recalibrate", file=sys.stderr)
            rows.append(res)
            if args.save_overlay and (not args.all_frames or i == indices[-1]):
                os.makedirs(args.save_overlay, exist_ok=True)
                out_path = os.path.join(args.save_overlay,
                                        f"{camera}_{_frame_index(name):05d}_overlay.png")
                cv2.imwrite(out_path, overlay(bgr, target, red))
        per_camera[camera] = rows
        if args.all_frames:
            last = rows[-1]
            best = min(rows, key=lambda r: r["uncovered_frac"])
            print(f"{camera}: final   {_fmt(last)}", file=sys.stderr)
            print(f"{camera}: best    {_fmt(best)}  @ {best['frame']}", file=sys.stderr)
        else:
            print(f"{camera}: {_fmt(rows[-1])}", file=sys.stderr)

    if not per_camera:
        return 1

    finals = {cam: rows[-1] for cam, rows in per_camera.items()}
    combined = {
        "uncovered_frac_lower_bound": max(r["uncovered_frac"] for r in finals.values()),
        "uncovered_frac_per_camera": {c: r["uncovered_frac"] for c, r in finals.items()},
        "note": ("silhouette overlap over-estimates coverage for a tall block, so "
                 "the per-camera uncovered fractions are lower bounds; the max is "
                 "the tightest one"),
    }
    print(f"combined: uncovered >= {100 * combined['uncovered_frac_lower_bound']:.1f}% "
          f"of the drawn target", file=sys.stderr)

    payload = {"per_camera": per_camera if args.all_frames else finals,
               "combined": combined}
    if args.json:
        with open(args.json, "w") as fh:
            json.dump(payload, fh, indent=2)
        print(f"wrote {args.json}", file=sys.stderr)
    else:
        print(json.dumps(payload["combined"] if args.all_frames else payload, indent=2))

    if args.csv:
        with open(args.csv, "w", newline="") as fh:
            rows = [r for rows in per_camera.values() for r in rows]
            writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)
        print(f"wrote {args.csv}", file=sys.stderr)
    return 0


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="cmd", required=True)

    def add_source_args(p):
        p.add_argument("--zip", help="dataset zip, e.g. data/pusht_2026_07.zip")
        p.add_argument("--session", help="session dir inside the zip (fnmatch pattern)")
        p.add_argument("--images", help="directory tree or single image file")
        p.add_argument("--camera", choices=CAMERAS, help="default: both")

    c = sub.add_parser("calibrate", help="build the drawn-target mask per camera")
    add_source_args(c)
    c.add_argument("--traj", help="restrict to one trajectory (auto mode prefers many)")
    c.add_argument("--mode", choices=("auto", "manual", "points"), default="auto")
    c.add_argument("--points", help="'x0,y0 x1,y1 ...' polygon for --mode points")
    c.add_argument("--roi", help="x0,y0,x1,y1 search window; needed when the scene "
                                 "has other closed dark outlines (cam0)")
    c.add_argument("--out", default="data/target_masks")
    c.add_argument("--stride", type=int, default=1, help="use every Nth frame")
    c.add_argument("--max-frames", type=int, default=4000)
    c.add_argument("--bg-frames", type=int, default=150,
                   help="frames kept in memory for the empty-table median")
    c.add_argument("--min-obs", type=int, default=CalibParams.min_obs,
                   help="unoccluded views a pixel needs before its vote counts")
    c.add_argument("--vote-thresh", type=float, default=CalibParams.vote_thresh)
    c.add_argument("--dilate", type=int, default=CalibParams.dilate)
    c.add_argument("--close", type=int, default=CalibParams.close)
    c.add_argument("--line-delta", type=float, default=CalibParams.line_delta,
                   help="grey levels below local background to call a pixel ink")
    c.add_argument("--red-pad", type=int, default=CalibParams.red_pad,
                   help="dilation of the block mask when rejecting occluded pixels")
    c.add_argument("--max-area-frac", type=float, default=CalibParams.max_area_frac)
    c.add_argument("--min-area-px", type=int, default=CalibParams.min_area_px)
    c.add_argument("--no-sweep", action="store_true",
                   help="use the given params instead of sweeping a small grid")
    c.add_argument("--save-votes", action="store_true",
                   help="also dump the accumulated ink-vote image")
    c.add_argument("--emit-constants", action="store_true",
                   help="print the TARGET_POLYGONS literal to paste back into this "
                        "script, so measuring needs no mask files")
    c.add_argument("--polygon-epsilon", type=float, default=2.0,
                   help="contour simplification for --emit-constants, in px")
    c.add_argument("--update-assets", action="store_true",
                   help=f"also refresh the shipped reference images in {ASSETS}")
    c.set_defaults(func=cmd_calibrate)

    m = sub.add_parser("measure", help="coverage of the target by the block")
    add_source_args(m)
    m.add_argument("--traj", help="trajectory inside the zip session")
    m.add_argument("--masks", default=None,
                   help="calibration directory; without it the built-in target polygon "
                        "and the shipped reference images are used")
    m.add_argument("--image", action="append",
                   help="images0=/path.jpg (repeatable); bypasses --zip/--images")
    m.add_argument("--frame", type=int, default=-1, help="frame index, negative from end")
    m.add_argument("--all-frames", action="store_true", help="measure the whole sequence")
    m.add_argument("--save-overlay", help="directory for annotated PNGs")
    m.add_argument("--sat-min", type=int, default=70, help="block HSV saturation floor")
    m.add_argument("--val-min", type=int, default=10,
                   help="block HSV value floor; low so shaded faces still count as block")
    m.add_argument("--hue-tol", type=int, default=12, help="block hue half-width around 0")
    m.add_argument("--no-background", action="store_true",
                   help="ignore the calibrated reference image (absolute brightness "
                        "test for bare paper, no drift check)")
    m.add_argument("--max-shift", type=float, default=3.0,
                   help="warn above this camera translation (px) since calibration")
    m.add_argument("--json", help="write results here instead of stdout")
    m.add_argument("--csv", help="write per-frame rows here")
    m.set_defaults(func=cmd_measure)

    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
