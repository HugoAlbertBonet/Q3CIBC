#!/usr/bin/env python3
"""Open-loop replay of one recorded Push-T episode, with the policy in SHADOW.

Purpose: split the two failure modes that a live rollout confounds.

  1. Is the RIG still able to reproduce a demonstration? We command the expert's
     recorded (dx, dy) sequence open-loop, through the exact same action pipeline
     the deploy client uses (min-step snap -> approach floor -> 7-D lift ->
     safety clip -> action-mode projection -> step_action). If the T does not end
     where the demo's last frame shows it, the problem is the rig / scene / start
     alignment, not the policy.
  2. Are the POLICY's actions reasonable? At every step the trained checkpoint
     sees the live frame stack and samples an action exactly as its deploy
     client would, but that action is only RECORDED — never sent. Afterwards we
     plot expert vs shadow action per timestep.

--seed-dir is repeatable, so a DP and a Q3C checkpoint can shadow the SAME
replay; --policy defaults to `auto`, which reads each directory's weight files
(denoiser*.pt -> DP, control_point_generator*.pt -> Q3C). Models keep their own
cameras, resize, frame_stack and action normalization -- the buffer holds RAW
camera frames and each model builds the stack its deploy client would.

--drop-zero skips the expert's exactly-(0,0) steps, matching the trainers'
--idle-filter drop_zero (24% of this collection). Those transitions were removed
from training, so a policy fitted on the filtered set never learned to emit 0 and
scores badly on them; keeping them in the comparison measures nothing useful. The
commanded path is identical either way -- a zero delta moves nothing -- only the
pauses disappear, so the replay finishes sooner and the T ends up in the same
place.

--variant is repeatable and runs SEVERAL inference configs per step, all against
those same raw frames, then plots them together. That is the only fair
comparison: a separate replay per config gives each one a different scene
history, so their errors are not comparable. Variants route to a model by
`model=NAME`, or automatically, because the DP and Q3C kind names are disjoint
(ddpm/ddim vs argmax/sample/dfo/langevin).

Everything policy-side is imported from scripts/deploy_pusht_real_dp.py and
scripts/deploy_pusht_real.py (build_dp_policy, dp_sample_action, build_models,
select_action, preprocess, build_stack_frame, stack_to_tensor, unnormalize,
apply_min_step, apply_approach_floor, to_action_7d, safety_clip_action,
project_action_to_env_mode, the init/reset retry helpers). No logic is
re-implemented here, so if the replay works the only remaining difference to a
live rollout is *which* action gets sent.

Cameras: identical to the deploy client. Topics are registered in the
collection's order (D435 then blue), each camera is read from its position in
that list, and the policy stack is built from the checkpoint's camera_streams --
one camera (g01/g02/g03) or two (g04) with no flag change. The alignment gate
below shows whichever cameras --align-cameras names, independently of that.

Deviations from the deploy client, both deliberate and both printed at startup:
  * The start pose is this EPISODE's first robot_eef_pose (translation) instead
    of the mean-of-demos asset, so the arm starts where the demo started.
  * The policy's action is recorded, not sent; the expert's is sent.

The episode comes from a committed BUNDLE, not the 2 GB archive: the robot-side
machine only needs data/replay_episodes/ep<NNN>/ (actions.npy, eef.npy, one
cam<N>.png per camera, meta.json), written by scripts/export_replay_episode.py
wherever the archive lives. --archive still reads the zip directly if you have
it.

Usage (server already up):

    # dry run: no motion, but full alignment + shadow policy + plot
    python scripts/replay_pusht_episode.py \
        --seed-dir checkpoints/pusht_real_dp_2026_07/g01_resnet18_s11_350k \
        --episode 0 --device cpu --sampler ddim \
        --dry-run --log-dir results/replay_ep0

    # live replay
    python scripts/replay_pusht_episode.py \
        --seed-dir checkpoints/pusht_real_dp_2026_07/g01_resnet18_s11_350k \
        --episode 0 --device cpu --sampler ddim --log-dir results/replay_ep0

    # Q3C, four inference configs compared in ONE replay
    python scripts/replay_pusht_episode.py \
        --seed-dir checkpoints/pusht_real_combinedv2/seed_0011 --policy q3c \
        --episode 70 --device cpu --shadow-every 4 \
        --variant argmax \
        --variant dfo:iters=50 \
        --variant dfo:iters=200,label=dfo200 \
        --variant langevin:iters=50 \
        --log-dir results/replay_ep70_q3c

    # DP and Q3C side by side, several variants each, one replay
    python scripts/replay_pusht_episode.py \
        --seed-dir dp=checkpoints/pusht_real_dp_2026_07/g01_resnet18_s11_350k \
        --seed-dir q3c=checkpoints/pusht_real_combinedv2/seed_0011 \
        --episode 70 --device cpu --shadow-every 4 \
        --variant ddim:steps=10 \
        --variant ddpm \
        --variant argmax \
        --variant langevin:iters=50 \
        --log-dir results/replay_ep70_both

    # read the archive directly (only where the zip is present)
    python scripts/replay_pusht_episode.py ... --archive data/pusht_2026_07_zarr.zip
"""

from __future__ import annotations

import argparse
import collections
import importlib.util
import json
import os
import shutil
import sys
import tempfile
import time
import zipfile
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# The DP deploy client is the reference implementation; loading it also loads the
# Q3C client it shares every robot-facing helper with (exposed as `dp.d`).
_spec = importlib.util.spec_from_file_location(
    "deploy_dp", ROOT / "scripts" / "deploy_pusht_real_dp.py")
dp = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(dp)
d = dp.d


# ---------------------------------------------------------------------------
# Archive readers (loader-external on purpose: no frame cache is built)
# ---------------------------------------------------------------------------

def _zarr_prefix(archive: zipfile.ZipFile) -> str:
    for name in archive.namelist():
        idx = name.find("replay_buffer.zarr/")
        if idx != -1:
            return name[: idx + len("replay_buffer.zarr/")]
    raise SystemExit("replay_buffer.zarr not found in the archive")


def load_lowdim(archive_path: Path):
    """Return (actions[N,7], eef[N,7], episode_ends[E]) from the zarr block."""
    import zarr

    with zipfile.ZipFile(archive_path, "r") as ar:
        prefix = _zarr_prefix(ar)
        members = [n for n in ar.namelist() if n.startswith(prefix)]
        tmp = tempfile.mkdtemp(prefix="replay_zarr_")
        try:
            ar.extractall(tmp, members=members)
            root = zarr.open(os.path.join(tmp, prefix.rstrip("/")), mode="r")
            actions = np.asarray(root["data/action"][:], dtype=np.float64)
            eef = np.asarray(root["data/robot_eef_pose"][:], dtype=np.float64)
            ends = np.asarray(root["meta/episode_ends"][:], dtype=np.int64)
        finally:
            shutil.rmtree(tmp, ignore_errors=True)
    return actions, eef, ends


def load_episode_frames(archive_path: Path, episode: int, camera: int,
                        indices) -> dict[int, np.ndarray]:
    """Decode selected frames of videos/<episode>/<camera>.mp4 as (H,W,3) RGB."""
    import imageio.v3 as iio

    wanted = sorted({int(i) for i in indices})
    with zipfile.ZipFile(archive_path, "r") as ar:
        root = _zarr_prefix(ar).split("replay_buffer.zarr/")[0]
        member = f"{root}videos/{episode}/{camera}.mp4"
        if member not in set(ar.namelist()):
            raise SystemExit(f"missing video in archive: {member}")
        scratch = tempfile.mkdtemp(prefix="replay_vid_")
        try:
            ar.extract(member, scratch)
            path = os.path.join(scratch, member)
            out: dict[int, np.ndarray] = {}
            for i, frame in enumerate(iio.imiter(path)):
                if i in wanted:
                    out[i] = np.asarray(frame, dtype=np.uint8)
                if wanted and i >= wanted[-1]:
                    break
        finally:
            shutil.rmtree(scratch, ignore_errors=True)
    missing = [i for i in wanted if i not in out]
    if missing:
        raise SystemExit(
            f"episode {episode} camera {camera}: frames {missing} not in the video")
    return out


def load_archive_metadata(archive_path: Path) -> dict:
    """The archive's metadata.json, or {} if it has none."""
    with zipfile.ZipFile(archive_path, "r") as ar:
        for name in ar.namelist():
            if name.endswith("metadata.json"):
                return json.loads(ar.read(name))
    return {}


def load_episode_from_archive(archive_path: Path, episode: int, cameras):
    """(actions[T,2], eef[T,3], {cam: frame0}, meta) straight from the zip."""
    actions, eef, ends = load_lowdim(archive_path)
    if not 0 <= episode < len(ends):
        raise SystemExit(f"--episode must be in [0, {len(ends)}); got {episode}")
    start = int(ends[episode - 1]) if episode > 0 else 0
    end = int(ends[episode])
    frames = {int(c): load_episode_frames(archive_path, episode, int(c), [0])[0]
              for c in cameras}
    prov = (load_archive_metadata(archive_path).get("provenance") or {})
    meta = {"episode": episode, "source_archive": archive_path.name,
            "rows": [start, end], "n_steps": end - start,
            "move_duration": prov.get("move_duration")}
    return (np.asarray(actions[start:end, :2], np.float64),
            np.asarray(eef[start:end, :3], np.float64), frames, meta)


def load_episode_bundle(bundle_dir: Path, cameras):
    """(actions[T,2], eef[T,3], {cam: frame0}, meta) from an exported bundle.

    Bundles are written by scripts/export_replay_episode.py and committed under
    data/replay_episodes/, so the robot-side machine never needs the ~2 GB
    archive -- only the expert actions, the demo EEF trace and one PNG per
    camera.
    """
    import cv2

    if not bundle_dir.is_dir():
        raise SystemExit(
            f"no episode bundle at {bundle_dir}. Either export it where the "
            f"archive lives (python scripts/export_replay_episode.py --episode "
            f"N) or point --archive at the zip.")
    actions = np.asarray(np.load(bundle_dir / "actions.npy"), np.float64)
    eef = np.asarray(np.load(bundle_dir / "eef.npy"), np.float64)
    meta_path = bundle_dir / "meta.json"
    meta = json.loads(meta_path.read_text()) if meta_path.is_file() else {}

    frames = {}
    for cam in cameras:
        png = bundle_dir / f"cam{int(cam)}.png"
        if not png.is_file():
            raise SystemExit(
                f"{png} missing: the bundle holds cameras "
                f"{meta.get('cameras', 'unknown')}. Re-export with --cameras, or "
                "restrict --align-cameras.")
        img = cv2.imread(str(png), cv2.IMREAD_COLOR)
        if img is None:
            raise SystemExit(f"could not read {png}")
        frames[int(cam)] = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return actions, eef, frames, meta


def resolve_backend(policy: str, seed_dir: Path) -> str:
    """"auto" -> "dp" | "q3c", decided by which weight files the run wrote."""
    if policy != "auto":
        return policy
    has_dp = any((seed_dir / f"denoiser{s}.pt").is_file() for s in ("_ema", ""))
    has_q3c = any((seed_dir / f"control_point_generator{s}.pt").is_file()
                  for s in ("_ema", ""))
    if has_dp and not has_q3c:
        return "dp"
    if has_q3c and not has_dp:
        return "q3c"
    if has_dp and has_q3c:
        raise SystemExit(
            f"{seed_dir} holds BOTH denoiser*.pt and control_point_generator*.pt; "
            "pass --policy dp or --policy q3c.")
    raise SystemExit(
        f"no policy weights in {seed_dir}: expected denoiser{'{_ema,}'}.pt (DP) "
        f"or control_point_generator{'{_ema,}'}.pt + q_estimator*.pt (Q3C).")


# ---------------------------------------------------------------------------
# Shadow variants
# ---------------------------------------------------------------------------
# A variant is one inference configuration of the loaded checkpoint. Several can
# run per step against the SAME frame stack, which is the only way to compare
# them fairly: rerunning the replay gives each config a different scene history.
#
#   Q3C   argmax | sample:temp=1.0
#         dfo:iters=100,noise=0.1,decay=0.8
#         langevin:iters=50,lr0=0.1,lr1=1e-5
#   DP    ddpm | ddim:steps=10,eta=0.0
#
# `label=` renames the series in the plot; everything else defaults to the
# corresponding CLI flag, so `dfo` alone means "dfo at --refine-iters".
_VARIANT_KINDS = {
    "q3c": {"argmax": ("temp",), "sample": ("temp",),
            "dfo": ("iters", "noise", "decay"),
            "langevin": ("iters", "lr0", "lr1")},
    "dp": {"ddpm": (), "ddim": ("steps", "eta")},
}


def load_model(name: str, seed_dir: Path, policy: str, args, device) -> dict:
    """Load one checkpoint (DP or Q3C) and everything needed to run it.

    Each model keeps its OWN preprocessing (cameras, resize, frame_stack) and
    its own action normalization, so a Q3C run trained on `images1` at 240x320
    and a DP run trained on two `video` streams can be compared in the same
    replay: the frame buffer stores raw camera frames and every model builds its
    stack from those.
    """
    backend = resolve_backend(policy, seed_dir)
    env_cfg = d.load_run_config(seed_dir)
    norm_stats = torch.load(seed_dir / "norm_stats.pt", map_location="cpu",
                            weights_only=False)
    m = {"name": name, "seed_dir": seed_dir, "backend": backend,
         "env_cfg": env_cfg, "norm_stats": norm_stats}
    m["act_min"] = np.asarray(norm_stats["act_min"], np.float32)
    m["act_max"] = np.asarray(norm_stats["act_max"], np.float32)
    m["norm_range"] = tuple(norm_stats.get("action_norm_range", (-1.0, 1.0)))
    m["frame_stack"] = int(norm_stats.get("frame_stack",
                                          env_cfg.get("frame_stack", 2)))
    image_hw = norm_stats.get("image_hw",
                              (int(env_cfg.get("image_height", 240)),
                               int(env_cfg.get("image_width", 320))))
    m["image_hw"] = (int(image_hw[0]), int(image_hw[1]))
    # The DP trainer records camera_streams as ("video1",); the Q3C runs carry
    # ("images1",) in their run config. Both index the same dataset cameras.
    cams = tuple(norm_stats.get("camera_streams",
                                env_cfg.get("camera_streams", ["video1"])))
    m["cams"] = cams
    m["cam_ids"] = d.camera_ids_from_streams(cams)
    expected = 3 * len(m["cam_ids"]) * m["frame_stack"]
    m["in_channels"] = int(norm_stats.get("in_channels", expected))
    if expected != m["in_channels"]:
        raise SystemExit(
            f"{name}: checkpoint says in_channels={m['in_channels']} but its "
            f"camera_streams {cams} x frame_stack {m['frame_stack']} imply "
            f"{expected}.")
    # Action chunking (--action-chunk K): the heads are 2*K wide and the trainer
    # sized them from dataset.action_shape. Shadowing one is fine -- nothing is
    # executed -- and we compare its FIRST predicted step against the expert,
    # which is what an open-loop chunk would command at this timestep.
    if m["act_min"].size % 2:
        raise SystemExit(
            f"{name}: act_min has {m['act_min'].size} entries, which is not a "
            "whole number of (dx, dy) pairs.")
    m["action_chunk"] = int(m["act_min"].size // 2)
    if m["action_chunk"] > 1 and backend == "dp":
        raise SystemExit(
            f"{name}: action_chunk={m['action_chunk']}, but the DP sampler "
            "(deploy_pusht_real_dp.dp_sample_action) hardcodes action_dim=2.")

    # EEF (x, y) conditioning: Q3C runs trained with --cond-eef-xy carry it; the
    # DP client has no conditioned path (deploy_pusht_real_dp.build_dp_policy
    # raises), so mirror that limitation instead of inventing one here.
    cond_dim = int(norm_stats.get("cond_dim", 0))
    m["cond_dim"] = cond_dim
    m["cond_min"] = m["cond_max"] = None
    if cond_dim:
        if backend == "dp":
            raise SystemExit(
                f"{name}: conditioned DP checkpoints are not wired up "
                "(same limitation as deploy_pusht_real_dp.py).")
        if str(norm_stats.get("cond_kind", "")) != "eef_xy":
            raise SystemExit(
                f"{name}: cond_dim={cond_dim} "
                f"cond_kind={norm_stats.get('cond_kind')!r}; only eef_xy is known")
        m["cond_min"] = np.asarray(norm_stats["cond_min"], np.float32)
        m["cond_max"] = np.asarray(norm_stats["cond_max"], np.float32)

    # Printed at startup so an encoder mismatch is visible before load_state_dict
    # fails: these are exactly the knobs that must agree with training.
    ek = norm_stats.get("encoder_kind", env_cfg.get("model", {}).get(
        "encoder_kind", "conv_maxpool"))
    m["encoder_desc"] = str(ek)
    if str(ek) == "resnet18":
        mm = env_cfg.get("model", {})
        m["encoder_desc"] += (
            f"(norm={norm_stats.get('encoder_norm_kind', mm.get('encoder_norm_kind', 'bn'))},"
            f"kp={norm_stats.get('encoder_num_kp', mm.get('encoder_num_kp', 64))},"
            f"pretrained={bool(norm_stats.get('encoder_pretrained', mm.get('encoder_pretrained', True)))})")

    suffix = "" if args.no_ema else "_ema"
    m["which"] = "raw" if args.no_ema else "EMA"
    if backend == "dp":
        denoiser, diffusion, dpar = dp.build_dp_policy(
            env_cfg, norm_stats, m["in_channels"], device)
        denoiser.load_state_dict(torch.load(seed_dir / f"denoiser{suffix}.pt",
                                            map_location=device, weights_only=True))
        denoiser.eval()
        ev = norm_stats.get("ddim_eval_steps", dpar.get("ddim_eval_steps", [10]))
        m["denoiser"], m["diffusion"], m["dpar"] = denoiser, diffusion, dpar
        m["ddim_steps"] = args.ddim_steps or (int(ev[0]) if ev else 10)
        m["ddim_eta"] = (args.ddim_eta if args.ddim_eta is not None
                         else float(norm_stats.get("ddim_eta",
                                                   dpar.get("ddim_eta", 0.0))))
        m["detail"] = (f"pred={dpar.get('prediction_type')} "
                       f"T={dpar.get('num_train_timesteps')}")
    else:
        cp_gen, q_net = d.build_models(env_cfg, m["in_channels"], device,
                                       cond_dim=cond_dim, norm_stats=norm_stats)
        d.load_weights(cp_gen, seed_dir / f"control_point_generator{suffix}.pt",
                       device)
        d.load_weights(q_net, seed_dir / f"q_estimator{suffix}.pt", device)
        m["cp_gen"], m["q_net"] = cp_gen, q_net
        m["cp_selection"] = (args.cp_selection
                             or str(norm_stats.get("cp_selection", "argmax")))
        m["cp_temp"] = (args.cp_temperature if args.cp_temperature is not None
                        else float(norm_stats.get("cp_selection_temperature", 1.0)))
        m["detail"] = f"checkpoint cp_selection={m['cp_selection']}"
    return m


def parse_variant(spec: str, models: list, args) -> dict:
    """"langevin:iters=50" -> a fully-resolved variant bound to a model.

    Routing: an explicit `model=NAME` wins; otherwise the KIND decides, because
    the DP and Q3C kind names are disjoint (ddpm/ddim vs argmax/sample/dfo/
    langevin). Only if that still leaves several candidates is `model=` required.
    """
    kind, _, rest = spec.partition(":")
    kind = kind.strip()
    kv = {}
    for part in (p for p in rest.split(",") if p.strip()):
        if "=" not in part:
            raise SystemExit(
                f"--variant {spec!r}: {part!r} is not key=value")
        k, v = part.split("=", 1)
        kv[k.strip()] = v.strip()
    want_model = kv.pop("model", None)
    label = kv.pop("label", spec)

    if want_model is not None:
        cand = [m for m in models if m["name"] == want_model]
        if not cand:
            raise SystemExit(
                f"--variant {spec!r}: model={want_model!r} is not one of "
                f"{[m['name'] for m in models]}")
    else:
        cand = [m for m in models if kind in _VARIANT_KINDS[m["backend"]]]
        if not cand:
            backends = sorted({m["backend"] for m in models})
            raise SystemExit(
                f"--variant {spec!r}: unknown kind {kind!r} for the loaded "
                f"{backends} checkpoint(s); expected one of "
                f"{sorted(k for b in backends for k in _VARIANT_KINDS[b])}")
        if len(cand) > 1:
            raise SystemExit(
                f"--variant {spec!r}: {kind!r} fits several loaded models "
                f"({[m['name'] for m in cand]}); disambiguate with model=NAME")
    model = cand[0]
    backend = model["backend"]
    known = _VARIANT_KINDS[backend]
    if kind not in known:
        raise SystemExit(
            f"--variant {spec!r}: unknown kind {kind!r} for {model['name']} "
            f"(a {backend} checkpoint); expected one of {sorted(known)}")
    unknown = set(kv) - set(known[kind])
    if unknown:
        raise SystemExit(
            f"--variant {spec!r}: {sorted(unknown)} not valid for {kind!r} "
            f"(accepts {list(known[kind])})")

    v = {"label": label, "kind": kind, "model": model["name"], "_model": model}
    if backend == "dp":
        v["steps"] = int(kv.get("steps", args.ddim_steps or 0)) or None
        v["eta"] = (float(kv["eta"]) if "eta" in kv else args.ddim_eta)
    else:
        # inference="argmax"/"sample" ignores the refinement path; cp_selection
        # is what picks argmax-vs-softmax there (see select_action).
        v["inference"] = kind
        v["cp_selection"] = kind if kind in ("argmax", "sample") else "argmax"
        v["temp"] = float(kv.get("temp", args.cp_temperature or 1.0))
        v["iters"] = int(kv.get("iters", args.refine_iters))
        v["noise"] = float(kv.get("noise", args.dfo_noise_init))
        v["decay"] = float(kv.get("decay", args.dfo_noise_decay))
        v["lr0"] = float(kv.get("lr0", args.langevin_lr_init))
        v["lr1"] = float(kv.get("lr1", args.langevin_lr_final))
    return v


def describe_variant(v: dict) -> str:
    m = v["_model"]
    head = f"{v['label']} [{v['model']}]:"
    if m["backend"] == "dp":
        # Unset steps/eta fall back to the checkpoint's own values at sample time.
        steps = v["steps"] if v["steps"] is not None else m["ddim_steps"]
        eta = v["eta"] if v["eta"] is not None else m["ddim_eta"]
        return (f"{head} {v['kind']}"
                + (f" ({steps} steps, eta={eta})" if v["kind"] == "ddim" else ""))
    if v["kind"] in ("argmax", "sample"):
        return (f"{head} CP-cloud {v['kind']}"
                + (f" (temp={v['temp']})" if v["kind"] == "sample" else ""))
    if v["kind"] == "dfo":
        return (f"{head} dfo x{v['iters']} "
                f"(noise {v['noise']}, decay {v['decay']})")
    return (f"{head} langevin x{v['iters']} (lr {v['lr0']} -> {v['lr1']})")


# ---------------------------------------------------------------------------
# Alignment gate
# ---------------------------------------------------------------------------

def alignment_panel(ref_rgb: np.ndarray, live_rgb: np.ndarray, alpha: float,
                    label: str) -> np.ndarray:
    """[ref | live | blend] as one BGR strip, annotated. Shapes are matched."""
    import cv2

    ref = ref_rgb
    if ref.shape[:2] != live_rgb.shape[:2]:
        ref = cv2.resize(ref, (live_rgb.shape[1], live_rgb.shape[0]),
                         interpolation=cv2.INTER_AREA)
    blend = cv2.addWeighted(ref, alpha, live_rgb, 1.0 - alpha, 0.0)
    strip = np.concatenate([ref, live_rgb, blend], axis=1)
    strip = cv2.cvtColor(strip, cv2.COLOR_RGB2BGR)
    w = live_rgb.shape[1]
    for i, text in enumerate((f"{label} DEMO frame 0", f"{label} LIVE",
                              f"{label} BLEND")):
        cv2.putText(strip, text, (10 + i * w, 24), cv2.FONT_HERSHEY_SIMPLEX,
                    0.7, (0, 255, 255), 2, cv2.LINE_AA)
    return strip


def wait_for_alignment(grab_obs, live_frame, ref_frames: dict[int, np.ndarray],
                       alpha: float, use_gui: bool, dump_dir: Path | None) -> None:
    """Block until the operator confirms the T matches the demo's first frame."""
    import cv2

    cams = sorted(ref_frames)
    print("\n=== ALIGNMENT ===")
    print(f"Place the T so the LIVE view matches the DEMO frame 0 on cameras {cams}.")
    if use_gui:
        print("  [Enter]/[space] = confirmed, aligned    [s] = save png    [q] = abort")
    else:
        print(f"  headless: panels are written to {dump_dir}; refresh and inspect.")

    saved = 0
    while True:
        raw_obs = grab_obs()
        strips = [alignment_panel(ref_frames[c], live_frame(raw_obs, c),
                                  alpha, f"cam{c}") for c in cams]
        width = max(s.shape[1] for s in strips)
        strips = [s if s.shape[1] == width else
                  cv2.resize(s, (width, int(s.shape[0] * width / s.shape[1])))
                  for s in strips]
        view = np.concatenate(strips, axis=0)

        if dump_dir is not None:
            cv2.imwrite(str(dump_dir / "alignment.png"), view)

        if not use_gui:
            ans = input("Aligned? [y = start replay / n = re-check / q = abort]: ")
            ans = ans.strip().lower()
            if ans in ("y", "yes", ""):
                return
            if ans in ("q", "quit", "abort"):
                raise SystemExit("aborted at the alignment gate")
            continue

        cv2.imshow("pusht replay alignment", view)
        key = cv2.waitKey(50) & 0xFF
        if key in (13, 10, 32):            # Enter / space
            cv2.destroyAllWindows()
            return
        if key == ord("q"):
            cv2.destroyAllWindows()
            raise SystemExit("aborted at the alignment gate")
        if key == ord("s") and dump_dir is not None:
            out = dump_dir / f"alignment_{saved:02d}.png"
            cv2.imwrite(str(out), view)
            saved += 1
            print(f"  saved {out}")


# ---------------------------------------------------------------------------
# Plot
# ---------------------------------------------------------------------------

def variant_stats(expert, policy_v):
    """Per-axis MAE / corr / sign agreement of one variant against the expert."""
    valid = np.isfinite(policy_v).all(axis=1)
    out = {"n": int(valid.sum()), "mae": np.array([np.nan, np.nan]),
           "corr": np.array([np.nan, np.nan]), "sign": np.array([np.nan, np.nan])}
    if not valid.any():
        return out, valid
    e, p = expert[valid], policy_v[valid]
    out["mae"] = np.abs(p - e).mean(axis=0)
    for i in range(2):
        if e[:, i].std() > 1e-9 and p[:, i].std() > 1e-9:
            out["corr"][i] = np.corrcoef(e[:, i], p[:, i])[0, 1]
        out["sign"][i] = float((np.sign(e[:, i]) == np.sign(p[:, i])).mean())
    return out, valid


def plot_expert_vs_policy(expert, policy, executed, eef_meas, eef_demo,
                          demo_steps, labels, out_path: Path, title: str) -> None:
    """expert/executed: (T,2) metres. policy: (T,V,2). eef_*: (T,2) or None.

    `demo_steps` is each executed step's index in the ORIGINAL episode, so the
    time axis shows the gaps --drop-zero leaves behind.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    T = len(expert)
    t = (np.arange(T) if demo_steps is None
         else np.asarray(demo_steps)[:T].astype(int))
    contiguous = len(t) == T and np.array_equal(t, np.arange(T))
    V = policy.shape[1]
    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    stats = [variant_stats(expert, policy[:, v]) for v in range(V)]

    fig, axes = plt.subplots(2, 2, figsize=(15, 9))

    for ax, ax_i, name in ((axes[0, 0], 0, "dx"), (axes[0, 1], 1, "dy")):
        ax.plot(t, expert[:, ax_i] * 1000, lw=1.4, color="k",
                label="expert (executed)")
        for v in range(V):
            st, valid = stats[v]
            if not valid.any():
                continue
            ax.plot(t[valid], policy[valid, v, ax_i] * 1000, lw=1.0, alpha=0.8,
                    color=colors[v % len(colors)],
                    label=f"{labels[v]} (MAE {st['mae'][ax_i] * 1000:.2f} mm)")
        ax.axhline(0.0, color="k", lw=0.5)
        ax.set_xlabel("timestep" if contiguous else "demo timestep (idle dropped)")
        ax.set_ylabel(f"{name} [mm]")
        ax.set_title(name)
        ax.legend(fontsize=7)
        ax.grid(alpha=0.3)

    ax = axes[1, 0]
    exp_path = np.cumsum(executed, axis=0)
    ax.plot(exp_path[:, 0] * 100, exp_path[:, 1] * 100, lw=1.6, color="k",
            label="expert commands (cumsum)")
    for v in range(V):
        _, valid = stats[v]
        if not valid.any():
            continue
        # Zero-order hold across un-sampled steps (--shadow-every > 1), so the
        # hypothetical path is not artificially shortened by the missing draws.
        idx = np.where(valid, np.arange(T), 0)
        pol = policy[np.maximum.accumulate(idx), v]
        pol[: int(np.argmax(valid))] = 0.0
        pol_path = np.cumsum(pol, axis=0)
        ax.plot(pol_path[:, 0] * 100, pol_path[:, 1] * 100, lw=1.1, alpha=0.8,
                color=colors[v % len(colors)], label=f"{labels[v]} (cumsum)")
    if eef_meas is not None and len(eef_meas):
        m = eef_meas - eef_meas[0]
        ax.plot(m[:, 0] * 100, m[:, 1] * 100, lw=1.2, ls="--", color="dimgray",
                label="measured EEF (live)")
    if eef_demo is not None and len(eef_demo):
        dm = eef_demo - eef_demo[0]
        ax.plot(dm[:, 0] * 100, dm[:, 1] * 100, lw=1.2, ls=":", color="dimgray",
                label="measured EEF (demo)")
    ax.set_xlabel("dx from start [cm]")
    ax.set_ylabel("dy from start [cm]")
    ax.set_title("planar path")
    ax.legend(fontsize=7)
    ax.grid(alpha=0.3)
    ax.set_aspect("equal", adjustable="datalim")

    ax = axes[1, 1]
    if V == 1:
        st, valid = stats[0]
        if valid.any():
            lim = 1000 * max(np.abs(expert).max(),
                             np.abs(policy[valid, 0]).max(), 1e-6)
            for ax_i, name, marker in ((0, "dx", "o"), (1, "dy", "x")):
                ax.scatter(expert[valid, ax_i] * 1000,
                           policy[valid, 0, ax_i] * 1000, s=6, alpha=0.35,
                           marker=marker,
                           label=f"{name}: r={st['corr'][ax_i]:.2f}, "
                                 f"sign match={st['sign'][ax_i]:.0%}")
            ax.plot([-lim, lim], [-lim, lim], "k--", lw=0.8)
            ax.set_xlim(-lim, lim)
            ax.set_ylim(-lim, lim)
            ax.legend(fontsize=8)
        ax.set_xlabel("expert [mm]")
        ax.set_ylabel("policy [mm]")
        ax.set_title("per-step agreement")
    else:
        # Many variants: a scatter per variant is unreadable, so compare them.
        x = np.arange(V)
        for k, (ax_i, name) in enumerate(((0, "dx"), (1, "dy"))):
            ax.bar(x + (k - 0.5) * 0.38,
                   [stats[v][0]["mae"][ax_i] * 1000 for v in range(V)],
                   width=0.38, label=name)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=20, ha="right", fontsize=7)
        ax.set_ylabel("MAE vs expert [mm]")
        ax.set_title("per-variant error (lower = closer to the expert)")
        ax.legend(fontsize=8)
        for v in range(V):
            st = stats[v][0]
            ax.annotate(f"r={np.nanmean(st['corr']):.2f}", (v, 0),
                        textcoords="offset points", xytext=(0, 3),
                        ha="center", fontsize=6)
    ax.grid(alpha=0.3)

    fig.suptitle(title)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=130)
    plt.close(fig)
    print(f"Plot -> {out_path}")


# ---------------------------------------------------------------------------
# CLI (a copy of deploy_pusht_real_dp.parse_args + the replay-specific knobs)
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--seed-dir", action="append", required=True, metavar="[NAME=]PATH",
                   help="repeatable: run SEVERAL checkpoints in the shadow at "
                        "once, e.g. --seed-dir dp=checkpoints/.../g01 "
                        "--seed-dir q3c=checkpoints/.../seed_0011. NAME "
                        "defaults to the directory name and is what "
                        "--variant model=NAME refers to. All of them see the "
                        "same live frames, each with its own preprocessing.")
    p.add_argument("--device", default="cuda")
    p.add_argument("--no-ema", action="store_true",
                   help="use raw denoiser instead of the EMA copy")
    p.add_argument("--ip", default="localhost")
    p.add_argument("--port", type=int, default=5556)
    p.add_argument("--widowx-envs-path", type=Path, default=None)
    p.add_argument("--camera-topics", nargs="+", default=d.CAMERA_TOPICS,
                   help="ROS topics, registered in THIS order. Default = the "
                        "order the training data was collected in "
                        f"({d.DATASET_CAMERA_TOPICS}).")
    p.add_argument("--topic-camera-ids", nargs="+", type=int, default=None,
                   help="dataset camera id of each --camera-topics entry "
                        "(default 0,1,...). Blue-only rig: `--camera-topics "
                        "/blue/image_raw --topic-camera-ids 1 --align-cameras 1`.")

    # --- episode source ------------------------------------------------------
    # Default: the committed bundle, so the robot machine needs no archive.
    p.add_argument("--episode", type=int, default=70,
                   help="archive episode index. The exported bundles are 70 "
                        "(319 steps, shortest clean one), 112 and 140; episode "
                        "0 is NOT a good replay (43%% idle actions and 32 "
                        "all-zero robot_eef_pose rows).")
    p.add_argument("--episode-dir", type=Path, default=None,
                   help="exported bundle (actions.npy, eef.npy, cam*.png, "
                        "meta.json). Default: data/replay_episodes/ep<NNN>. "
                        "Build one with scripts/export_replay_episode.py.")
    p.add_argument("--archive", type=Path, default=None,
                   help="OPTIONAL: read the episode straight from the "
                        "zarr_video zip instead of a bundle. Only works where "
                        "the ~2 GB archive is present.")
    p.add_argument("--max-steps", type=int, default=None,
                   help="truncate the replay (default: the whole episode).")
    p.add_argument("--drop-zero", action="store_true",
                   help="skip the expert's idle steps, matching the trainers' "
                        "--idle-filter drop_zero. Those (0,0) transitions were "
                        "removed from training, so a policy fitted on the "
                        "filtered set never learned to emit 0 and is unfairly "
                        "penalised on them. The commanded path is unchanged "
                        "(a zero delta moves nothing); only the pauses go away.")
    p.add_argument("--idle-eps", type=float, default=0.0,
                   help="|a| <= this counts as idle for --drop-zero. 0.0 = "
                        "exactly (0,0), which is what the trainers use.")
    p.add_argument("--align-cameras", type=int, nargs="+", default=[0, 1],
                   help="camera indices shown at the alignment gate.")
    p.add_argument("--align-alpha", type=float, default=0.5,
                   help="blend weight of the DEMO frame in the blend panel.")
    p.add_argument("--no-gui", action="store_true",
                   help="no cv2 window: write alignment.png and prompt on stdin.")
    p.add_argument("--skip-alignment", action="store_true",
                   help="skip the alignment gate entirely (not recommended).")

    # --- service image geometry ---------------------------------------------
    p.add_argument("--im-size", type=int, default=480)
    p.add_argument("--im-width", type=int, default=640)

    # --- control -------------------------------------------------------------
    p.add_argument("--step-duration", type=float, default=d.STEP_DURATION,
                   help="control period; also the env move_duration. Default is "
                        "the collection's move_duration (20 Hz).")
    p.add_argument("--non-blocking", action="store_true")
    p.add_argument("--action-mode", default="2trans",
                   choices=["2trans", "3trans", "3trans1rot", "3trans3rot"])
    p.add_argument("--safety-max-xy-delta", type=float, default=d.SAFETY_MAX_XY_DELTA)
    p.add_argument("--workspace-xyz", type=float, nargs=6, default=None,
                   metavar=("X0", "Y0", "Z0", "X1", "Y1", "Z1"),
                   help="override the server's workspace box (metres); see "
                        "deploy_pusht_real.py. Only applied on init(), so a "
                        "reused env keeps whatever box it was started with.")
    p.add_argument("--z-drop-abort", type=float, default=0.008,
                   help="metres. Stop the replay if the measured EEF z falls "
                        "this far below --fixed-z-height for --z-drop-steps "
                        "consecutive steps: the pusher is then dragging on the "
                        "table and the rest of the episode is meaningless. "
                        "<=0 disables.")
    p.add_argument("--z-drop-steps", type=int, default=3,
                   help="consecutive out-of-tolerance steps before aborting.")
    p.add_argument("--min-step-xy", type=float, default=0.0)
    p.add_argument("--match-exposure", action="store_true",
                   help="OFF by default, same as the deploy client.")
    p.add_argument("--exposure-gains", type=float, nargs=3, default=[1.22, 1.18, 1.17],
                   metavar=("R", "G", "B"))
    p.add_argument("--lock-z", dest="lock_z", action="store_true", default=True)
    p.add_argument("--no-lock-z", dest="lock_z", action="store_false")
    p.add_argument("--fixed-z-height", type=float, default=d.FIXED_Z_HEIGHT)
    p.add_argument("--neutral-z-height", type=float, default=d.NEUTRAL_Z_HEIGHT)
    p.add_argument("--z-hold", type=float, default=0.0)
    p.add_argument("--z-hold-gain", type=float, default=1.0)
    p.add_argument("--z-hold-max", type=float, default=0.01)
    p.add_argument("--fixed-gripper", type=float, default=d.FIXED_GRIPPER)
    p.add_argument("--gripper-command", type=float, default=0.0)
    p.add_argument("--skip-move-to-neutral", action="store_true")
    p.add_argument("--i-traj", type=int, default=0)

    # --- initial pose --------------------------------------------------------
    p.add_argument("--move-to-demo-start", dest="move_to_demo_start",
                   action="store_true", default=True)
    p.add_argument("--no-move-to-demo-start", dest="move_to_demo_start",
                   action="store_false")
    p.add_argument("--start-eep-npy", type=Path,
                   default=ROOT / "scripts" / "assets" / "pusht_start_eep.npy",
                   help="4x4 EEF transform; its ROTATION is always used, and its "
                        "translation only when --no-start-from-episode.")
    p.add_argument("--start-from-episode", dest="start_from_episode",
                   action="store_true", default=True,
                   help="translate to THIS episode's first robot_eef_pose.")
    p.add_argument("--no-start-from-episode", dest="start_from_episode",
                   action="store_false")
    p.add_argument("--start-move-duration", type=float, default=1.5)
    p.add_argument("--max-initial-move-retries", type=int, default=5)

    # --- HARD approach guard -------------------------------------------------
    p.add_argument("--approach-floor", dest="approach_floor",
                   action="store_true", default=True)
    p.add_argument("--no-approach-floor", dest="approach_floor",
                   action="store_false")
    p.add_argument("--approach-floor-x", type=float, default=None)

    # --- init / reset robustness --------------------------------------------
    p.add_argument("--init-timeout-ms", type=int, default=180_000)
    p.add_argument("--init-retries", type=int, default=8)
    p.add_argument("--init-retry-sleep", type=float, default=2.0)
    p.add_argument("--reset-timeout-ms", type=int, default=60_000)
    p.add_argument("--reset-retries", type=int, default=3)
    p.add_argument("--reset-retry-sleep", type=float, default=1.0)
    p.add_argument("--rpc-timeout-ms", type=int, default=5_000)
    p.add_argument("--force-fresh-init", action="store_true")
    p.add_argument("--no-reuse-existing-env", dest="reuse_existing_env",
                   action="store_false", default=True)

    # --- which policy runs in the shadow -------------------------------------
    p.add_argument("--policy", default="auto", choices=["auto", "dp", "q3c"],
                   help="policy family of every --seed-dir. auto (default) "
                        "infers it per directory from the weight files "
                        "(denoiser*.pt -> dp, control_point_generator*.pt -> "
                        "q3c), which is what a mixed DP+Q3C run needs.")

    p.add_argument("--variant", action="append", default=None, metavar="SPEC",
                   help="repeatable: run SEVERAL inference configs per step, "
                        "against the same raw frames, and plot them together. "
                        "Q3C kinds: argmax | sample:temp=1.0 | "
                        "dfo:iters=100,noise=0.1,decay=0.8 | "
                        "langevin:iters=50,lr0=0.1,lr1=1e-5. DP kinds: ddpm | "
                        "ddim:steps=10,eta=0.0. The kind picks the model when "
                        "it is unambiguous; add model=NAME when several loaded "
                        "models share a backend. label=NAME renames the series. "
                        "Omitted keys fall back to the flags below. Default: "
                        "one variant per --seed-dir, built from those flags.")

    # --- DP sampler knobs (deploy_pusht_real_dp.py) --------------------------
    p.add_argument("--sampler", default="ddpm", choices=["ddpm", "ddim"])
    p.add_argument("--ddim-steps", type=int, default=None)
    p.add_argument("--ddim-eta", type=float, default=None)
    p.add_argument("--sample-seed", type=int, default=None)

    # --- Q3C selection knobs (deploy_pusht_real.py) --------------------------
    p.add_argument("--cp-selection", choices=["argmax", "sample"], default=None,
                   help="default: the checkpoint's cp_selection in norm_stats.")
    p.add_argument("--cp-temperature", type=float, default=None,
                   help="default: the checkpoint's cp_selection_temperature.")
    p.add_argument("--inference", choices=["argmax", "sample", "langevin", "dfo"],
                   default="argmax",
                   help="argmax/sample rank the CP cloud; langevin matches the "
                        "TRAINING sampler; dfo is the cheap derivative-free "
                        "refinement. Same semantics as deploy_pusht_real.py.")
    p.add_argument("--refine-iters", type=int, default=50)
    p.add_argument("--langevin-lr-init", type=float, default=0.1)
    p.add_argument("--langevin-lr-final", type=float, default=1e-5)
    p.add_argument("--dfo-noise-init", type=float, default=0.1)
    p.add_argument("--dfo-noise-decay", type=float, default=0.8)

    # --- shadow / diagnostics ------------------------------------------------
    p.add_argument("--shadow-every", type=int, default=1,
                   help="sample the policy every N steps. DP's DDPM at T=100 and "
                        "Q3C's langevin/dfo refinement can both exceed a 0.05 s "
                        "control period on CPU; raise this (or use --sampler "
                        "ddim / --inference argmax) if the loop cannot keep up.")
    p.add_argument("--no-shadow", action="store_true",
                   help="pure open-loop replay, no policy at all (rig check).")
    p.add_argument("--dry-run", action="store_true",
                   help="everything except step_action: no motion.")
    p.add_argument("--log-dir", type=Path, default=None,
                   help="per-step log: steps.jsonl, replay.npz, plot, frames.")
    p.add_argument("--save-frames", action="store_true",
                   help="also dump the fed frames (needs --log-dir).")
    p.add_argument("--plot-out", type=Path, default=None,
                   help="plot path (default: <log-dir>/expert_vs_policy.png).")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    if args.z_hold > 0 and args.action_mode == "2trans":
        raise SystemExit(
            "--z-hold needs an action_mode that sends z "
            "(3trans/3trans1rot/3trans3rot); got 2trans.")
    seed_dirs = []
    for entry in args.seed_dir:
        name, sep, path = str(entry).partition("=")
        if not sep:
            name, path = None, entry
        p = Path(path).expanduser().resolve()
        seed_dirs.append((name or p.name, p))
    if len({n for n, _ in seed_dirs}) != len(seed_dirs):
        raise SystemExit(
            f"--seed-dir names must be unique, got {[n for n, _ in seed_dirs]}; "
            "prefix them with NAME= to disambiguate.")
    log_dir = args.log_dir
    if log_dir is not None:
        log_dir.mkdir(parents=True, exist_ok=True)

    # --- episode -------------------------------------------------------------
    align_cams = [] if args.skip_alignment else sorted(
        {int(c) for c in args.align_cameras})
    if args.archive is not None:
        ep_actions, ep_eef, ref_frames, ep_meta = load_episode_from_archive(
            args.archive, args.episode, align_cams)
        source = args.archive.name
    else:
        bundle = args.episode_dir or (ROOT / "data" / "replay_episodes"
                                      / f"ep{args.episode:03d}")
        ep_actions, ep_eef, ref_frames, ep_meta = load_episode_bundle(
            bundle, align_cams)
        source = f"{bundle.name} (from {ep_meta.get('source_archive', '?')})"
    n_steps = len(ep_actions)
    if args.max_steps is not None:
        n_steps = min(n_steps, int(args.max_steps))
    zero_frac = float((np.linalg.norm(ep_actions, axis=1) == 0).mean())
    print(f"Episode {args.episode} from {source}: {len(ep_actions)} steps "
          f"(replaying {n_steps}), zero-action share {zero_frac:.1%}")

    # Which demo steps this replay actually commands. With --drop-zero the idle
    # transitions are skipped exactly as the trainers' --idle-filter drop_zero
    # removed them from the training set; `demo_steps` keeps the original index
    # of every executed step so the plot and the log stay aligned to the demo.
    demo_steps = np.arange(n_steps)
    if args.drop_zero:
        idle = np.linalg.norm(ep_actions[:n_steps], axis=1) <= args.idle_eps
        demo_steps = demo_steps[~idle]
        if not len(demo_steps):
            raise SystemExit(
                f"--drop-zero removed every step of episode {args.episode}.")
        print(f"  --drop-zero: {int(idle.sum())} idle steps skipped "
              f"(|a| <= {args.idle_eps}), {len(demo_steps)} commanded")
    n_exec = len(demo_steps)
    print(f"  demo EEF start={np.round(ep_eef[0], 4)} end={np.round(ep_eef[-1], 4)}")
    # 37 of the 151 episodes carry all-zero robot_eef_pose rows (the source
    # capture dropped the transform on those steps). They corrupt the start pose
    # and the plot's demo reference path, not the expert actions themselves.
    n_zero_eef = int((np.abs(ep_eef).sum(axis=1) == 0).sum())
    if n_zero_eef:
        print(f"[WARN] {n_zero_eef}/{len(ep_eef)} robot_eef_pose rows in this "
              "episode are all-zero; the demo EEF reference path is unreliable. "
              "Episodes 70, 112 and 140 have none.")
    rec_dt = ep_meta.get("move_duration")
    if rec_dt is not None and abs(float(rec_dt) - args.step_duration) > 1e-9:
        print(f"[WARN] this episode was recorded at move_duration={rec_dt}s but "
              f"--step-duration is {args.step_duration}s: the replay will not "
              "run at the demonstrated rate.")

    # --- checkpoints (each identical to its own deploy client) ---------------
    topic_camera_ids = d.resolve_topic_camera_ids(args.camera_topics,
                                                  args.topic_camera_ids)
    device = torch.device(
        args.device if (torch.cuda.is_available() or args.device == "cpu") else "cpu")

    models = []
    if args.no_shadow:
        print("[INFO] --no-shadow: no policy loaded; pure open-loop rig check.")
    else:
        for name, path in seed_dirs:
            models.append(load_model(name, path, args.policy, args, device))
        if args.sample_seed is not None:
            torch.manual_seed(args.sample_seed)
        for m in models:
            print(f"Shadow model {m['name']}: "
                  f"{'DP denoiser' if m['backend'] == 'dp' else 'Q3C energy model'}"
                  f" ({m['which']}) from {m['seed_dir']}")
            print(f"    {m['detail']}  frame_stack={m['frame_stack']} "
                  f"cameras={m['cams']} (ids {m['cam_ids']}) "
                  f"model_hw={m['image_hw']} in_channels={m['in_channels']} "
                  f"cond_dim={m['cond_dim']} encoder={m['encoder_desc']}")
            if m["action_chunk"] > 1:
                print(f"    action_chunk={m['action_chunk']} -> comparing the "
                      "FIRST predicted step against the expert")
            print(f"    act_min={m['act_min']} act_max={m['act_max']} "
                  f"norm_range={m['norm_range']}")

    # Every model reads the live cameras it was trained on; they need not agree.
    cam_ids = sorted({c for m in models for c in m["cam_ids"]})
    missing = [c for c in cam_ids if c not in topic_camera_ids]
    if missing:
        raise SystemExit(
            f"the loaded checkpoints need cameras {missing}, which the "
            f"registered topics ({topic_camera_ids}) do not provide.")
    max_stack = max((m["frame_stack"] for m in models), default=1)

    # --- inference variants --------------------------------------------------
    # One draw per variant per step, all against the SAME raw frames.
    variants = []
    if models:
        specs = args.variant
        if not specs:
            # No --variant: one default per model, from the legacy flags.
            specs = []
            for m in models:
                if m["backend"] == "dp":
                    kind = args.sampler
                else:
                    kind = (args.inference if args.inference in ("langevin", "dfo")
                            else m["cp_selection"])
                specs.append(f"{kind}:model={m['name']}"
                             + (f",label={m['name']}" if len(models) > 1 else ""))
        variants = [parse_variant(s, models, args) for s in specs]
        labels = [v["label"] for v in variants]
        if len(set(labels)) != len(labels):
            raise SystemExit(f"--variant labels must be unique, got {labels}")
        print(f"{len(variants)} inference variant(s) on device={device}:")
        for v in variants:
            print(f"  - {describe_variant(v)}")
    n_var = max(1, len(variants))
    labels = [v["label"] for v in variants] or ["(none)"]

    def make_cond(m, raw_obs):
        """Live EEF (x,y) -> (1,2) normalized, mirroring the Q3C deploy client."""
        if not m["cond_dim"]:
            return None
        st = None if raw_obs is None else raw_obs.get("state")
        if st is None:
            raise RuntimeError(
                f"{m['name']} needs EEF conditioning but the observation has no "
                "'state' field")
        xy = np.asarray(st, np.float32).reshape(-1)[:2]
        lo, hi = m["cond_min"], m["cond_max"]
        span = np.where(hi == lo, np.ones_like(hi), hi - lo)
        z = np.clip(-1.0 + 2.0 * (xy - lo) / span, -1.0, 1.0)
        return torch.from_numpy(z.astype(np.float32)).unsqueeze(0).to(device)

    def sample(v, obs_u8, raw_obs):
        """One draw of variant `v` (normalized action), the same call its deploy
        client makes -- only the inference config differs between variants."""
        m = v["_model"]
        if m["backend"] == "dp":
            return dp.dp_sample_action(
                m["diffusion"], m["denoiser"], obs_u8, v["kind"],
                v["steps"] if v["steps"] is not None else m["ddim_steps"],
                v["eta"] if v["eta"] is not None else m["ddim_eta"], cond=None)
        return d.select_action(
            m["cp_gen"], m["q_net"], obs_u8, v["cp_selection"], v["temp"],
            cond=make_cond(m, raw_obs), inference=v["inference"],
            refine_iters=v["iters"],
            langevin_lr=(v["lr0"], v["lr1"]),
            dfo_noise=(v["noise"], v["decay"]))

    # --- reference frames (already loaded with the episode) ------------------
    if align_cams:
        unavailable = [c for c in align_cams if c not in topic_camera_ids]
        if unavailable:
            raise SystemExit(
                f"--align-cameras {unavailable} are not among the registered "
                f"topics (which map to cameras {topic_camera_ids}).")
        print(f"Demo frame 0 loaded for cameras {align_cams} "
              f"({ref_frames[align_cams[0]].shape})")

    # --- connect (verbatim from deploy_pusht_real_dp.main) -------------------
    WidowXClient, WidowXConfigs, WidowXStatus = d.load_widowx_dependencies(
        args.widowx_envs_path)
    env_params = d.build_env_params(args, WidowXConfigs)
    print(f"Camera topics: {args.camera_topics} -> dataset camera ids "
          f"{topic_camera_ids}; policy reads {cam_ids}")

    # Preflight: this episode was recorded inside the COLLECTION's workspace
    # box, which is not necessarily the one we are about to send. Commanding the
    # arm past a boundary pins it against the wall, and the wrist can sag onto
    # the table from there -- the "z suddenly drops and stays down" failure.
    wb = d.workspace_bounds_from_args(args)
    print(f"Workspace box: x [{wb[0][0]}, {wb[1][0]}]  y [{wb[0][1]}, {wb[1][1]}]"
          f"  z [{wb[0][2]}, {wb[1][2]}]")
    demo_xyz = ep_eef[demo_steps]
    live_rows = np.abs(demo_xyz).sum(axis=1) > 0
    for i, ax in enumerate("xyz"):
        col = demo_xyz[live_rows, i]
        if not col.size:
            continue
        outside = int(((col < wb[0][i]) | (col > wb[1][i])).sum())
        if outside:
            print(f"[WARN] this episode's {ax} leaves the workspace box on "
                  f"{outside}/{col.size} steps (demo range "
                  f"[{col.min():.4f}, {col.max():.4f}], box "
                  f"[{wb[0][i]}, {wb[1][i]}]). The server will clip those "
                  f"commands and the replay cannot follow the demo. Widen it "
                  f"with --workspace-xyz.")
    if wb[0][2] < args.fixed_z_height - 0.005:
        print(f"[WARN] the workspace z floor ({wb[0][2]}) is "
              f"{1000 * (args.fixed_z_height - wb[0][2]):.0f} mm below "
              f"--fixed-z-height ({args.fixed_z_height}), so it will not stop "
              "the wrist from sagging onto the table.")
    print(f"action_mode={args.action_mode} lock_z={args.lock_z} "
          f"fixed_z_height={args.fixed_z_height} move_duration={args.step_duration}")

    client = WidowXClient(host=args.ip, port=args.port)

    reuse_existing_env = False
    if args.reuse_existing_env and not args.force_fresh_init:
        reuse_existing_env = d.widowx_server_has_live_env(client, max_wait_sec=1.0)
        if reuse_existing_env:
            print("[INFO] Server already has a live env; reusing it (skipping init()).")
            print("[WARN] The live env keeps its FIRST env_params -- including its "
                  "camera_topics. If it was started with a different topic list, "
                  "the camera indices below are WRONG; restart the server or pass "
                  "--no-reuse-existing-env.")

    if reuse_existing_env:
        d.set_reqrep_timeout_ms(client, max(1, args.rpc_timeout_ms))
    else:
        init_status = d.init_widowx_with_retry(
            client, env_params, args.im_size, WidowXStatus, args)
        if init_status != WidowXStatus.SUCCESS:
            raise RuntimeError(
                f"WidowX init failed after {args.init_retries} attempts with "
                f"status={d.status_name(init_status, WidowXStatus)}.")
    print("WidowX connection established.")

    reset_status = d.reset_widowx_with_retry(client, WidowXStatus, args, args.i_traj)
    if reset_status != WidowXStatus.SUCCESS:
        raise RuntimeError(
            f"WidowX reset failed with status={d.status_name(reset_status, WidowXStatus)}")
    print(f"Reset done (itraj={args.i_traj}).")

    if args.gripper_command >= 0.0 and hasattr(client, "move_gripper"):
        try:
            gstatus = client.move_gripper(float(args.gripper_command))
            print(f"Gripper commanded to {args.gripper_command} "
                  f"(0=closed,1=open); status={d.status_name(gstatus, WidowXStatus)}")
            time.sleep(1.0)
        except Exception as exc:
            print(f"[WARN] move_gripper({args.gripper_command}) failed: {exc}")

    # --- move to the start pose ---------------------------------------------
    start_T = None
    if args.move_to_demo_start:
        start_path = Path(args.start_eep_npy).expanduser()
        if not start_path.is_file():
            raise FileNotFoundError(f"--start-eep-npy not found: {start_path}")
        start_T = np.load(start_path).astype(np.float32)
        if args.start_from_episode:
            xyz = ep_eef[0]
            if not np.all(np.isfinite(xyz)) or np.abs(xyz).sum() == 0:
                print("[WARN] episode's first robot_eef_pose is all-zero "
                      "(2 episodes in the 2026-07 archive are); using the "
                      "mean-of-demos asset instead.")
            else:
                start_T = start_T.copy()
                start_T[:3, 3] = xyz.astype(np.float32)
        print(f"[INFO] Moving EEF to start pose (x={start_T[0,3]:.4f}, "
              f"y={start_T[1,3]:.4f}, z={start_T[2,3]:.4f})...")
        move_status, tries = None, 0
        while move_status != WidowXStatus.SUCCESS and tries < args.max_initial_move_retries:
            move_status = client.move(start_T, duration=args.start_move_duration)
            tries += 1
        if move_status != WidowXStatus.SUCCESS:
            print(f"[WARN] initial move not SUCCESS after {tries} tries "
                  f"(status={d.status_name(move_status, WidowXStatus)}); continuing.")

    # --- approach floor ------------------------------------------------------
    approach_floor_x = None
    if args.approach_floor:
        if args.approach_floor_x is not None:
            approach_floor_x = float(args.approach_floor_x)
        elif start_T is not None:
            approach_floor_x = float(start_T[0, 3])
        else:
            try:
                approach_floor_x = d.eef_x_from_obs(client.get_observation())
            except Exception:
                approach_floor_x = None
        if approach_floor_x is None:
            raise RuntimeError(
                "Approach guard ON but x floor undeterminable. Pass "
                "--approach-floor-x <metres> or --no-approach-floor.")
        demo_min_x = float(ep_eef[:, 0].min())
        print(f"[SAFETY] Approach floor ARMED: EEF x never below "
              f"{approach_floor_x:.4f} m (this demo's min x is {demo_min_x:.4f} m).")
        if demo_min_x < approach_floor_x - 1e-4:
            print("[WARN] the demo itself goes below the floor; those expert "
                  "steps WILL be clipped. Pass --no-approach-floor for a "
                  "faithful replay.")

    def grab_obs(retries: int = 50):
        for _ in range(retries):
            obs = client.get_observation()
            if obs is not None:
                return obs
            time.sleep(0.2)
        raise RuntimeError("no observation from server after retries")

    exposure_gains = args.exposure_gains if args.match_exposure else None
    if exposure_gains is not None:
        print(f"[match-exposure] per-channel gains RGB={exposure_gains}")

    def live_frame(raw_obs, cam_id):
        return d.frame_for_camera(raw_obs, cam_id, topic_camera_ids)

    def raw_frames(raw_obs) -> dict:
        """One timestep of RAW camera frames, {cam_id: (H,W,3) uint8}.

        Stored unresized because models may disagree on image_hw and on which
        cameras they read; each builds its own stack from these.
        """
        return {c: live_frame(raw_obs, c) for c in cam_ids}

    def model_stack(m, frame_buf) -> torch.Tensor:
        """The (1, C, H, W) uint8 tensor `m` expects, from the raw buffer.

        Reproduces deploy_pusht_real.build_stack_frame + stack_to_tensor, and
        therefore PushTWidowXVideoDataset's channel order: cameras iterate
        INSIDE each stack offset, oldest -> newest. Offsets older than the
        buffer are clamped to its first entry, exactly as the deploy clients
        pad the warm-up.
        """
        blocks = []
        for off in range(m["frame_stack"] - 1, -1, -1):
            raws = frame_buf[max(0, len(frame_buf) - 1 - off)]
            blocks.append(np.concatenate(
                [d.preprocess(raws[c], m["image_hw"], gains=exposure_gains)
                 for c in m["cam_ids"]], axis=-1))
        stacked = np.transpose(np.concatenate(blocks, axis=-1), (2, 0, 1))
        return torch.from_numpy(np.ascontiguousarray(stacked)).unsqueeze(0).to(device)

    # --- alignment gate ------------------------------------------------------
    if not args.skip_alignment:
        wait_for_alignment(grab_obs, live_frame, ref_frames,
                           float(args.align_alpha), use_gui=not args.no_gui,
                           dump_dir=log_dir)
        print("Alignment confirmed.")

    # --- warm up the frame buffer (deploy semantics: pad with the first frame)
    frame_buf = collections.deque(maxlen=max_stack)
    first = raw_frames(grab_obs())
    for _ in range(max_stack):
        frame_buf.append(first)
    for m in models:
        print(f"  {m['name']} stack: {tuple(model_stack(m, frame_buf).shape)} "
              f"from cameras {m['cam_ids']} at {m['image_hw']}")

    # --- replay --------------------------------------------------------------
    log_fh = None
    if log_dir is not None:
        log_fh = (log_dir / "steps.jsonl").open("w")
        if args.save_frames:
            (log_dir / "fed").mkdir(parents=True, exist_ok=True)
        print(f"Log -> {log_dir}")

    blocking = not args.non_blocking
    print(f"\nOpen-loop replay of {n_exec} EXPERT steps"
          + (f" ({n_steps - n_exec} idle skipped)" if n_exec != n_steps else "")
          + f", blocking={blocking}, step_duration={args.step_duration}s"
          + ("  [DRY RUN: no motion]" if args.dry_run else "")
          + ("" if not variants else
             f", {len(variants)} shadow variant(s) over {len(models)} model(s) "
             f"every {args.shadow_every} step(s)")
          + ". Keep a hand on the E-stop.")
    input("Press [Enter] to start.")

    expert_log = np.full((n_exec, 2), np.nan)
    executed_log = np.full((n_exec, 2), np.nan)
    policy_log = np.full((n_exec, n_var, 2), np.nan)
    eef_log = np.full((n_exec, 2), np.nan)
    sample_ms = [[] for _ in range(n_var)]
    step = -1
    z_low = 0
    abort_reason = None
    last_exec = time.time()

    try:
        # `step` indexes the executed sequence; `demo_step` the original episode
        # (they differ once --drop-zero skips the idle transitions).
        for step, demo_step in enumerate(demo_steps):
            demo_step = int(demo_step)
            raw_obs = grab_obs()
            frame_buf.append(raw_frames(raw_obs))

            # --- SHADOW: every variant sees exactly what deploy would --------
            na_by_variant = None
            if variants and step % max(1, args.shadow_every) == 0:
                # One stack per MODEL (they may differ in cameras/size/depth),
                # reused by every variant of that model.
                stacks = {m["name"]: model_stack(m, frame_buf) for m in models}
                na_by_variant = {}
                for vi, v in enumerate(variants):
                    m = v["_model"]
                    t0 = time.time()
                    na = sample(v, stacks[m["name"]], raw_obs)
                    sample_ms[vi].append((time.time() - t0) * 1000.0)
                    # Chunked heads denormalize as a whole, then we keep the
                    # first (dx, dy): the step this timestep would execute.
                    policy_log[step, vi] = d.unnormalize(
                        na, m["act_min"], m["act_max"], m["norm_range"])[:2]
                    na_by_variant[v["label"]] = [float(x) for x in np.ravel(na)]

            # --- EXECUTED: the expert action, through the deploy pipeline ----
            act_xy = ep_actions[demo_step].copy()
            expert_log[step] = act_xy

            act_xy, snapped = d.apply_min_step(act_xy, args.min_step_xy)
            cur_x = d.eef_x_from_obs(raw_obs)
            act_xy, floored = d.apply_approach_floor(act_xy, cur_x, approach_floor_x)
            if floored:
                print(f"[SAFETY] approach floor: clipped dx at x={cur_x:.4f} "
                      f"(floor={approach_floor_x:.4f})")

            action_7d = d.to_action_7d(act_xy, args.fixed_gripper)
            action_7d = d.safety_clip_action(action_7d, args.action_mode,
                                             args.safety_max_xy_delta)
            if args.z_hold > 0:
                action_7d[2] = d.z_hold_dz(d.z_from_obs(raw_obs), args.z_hold,
                                           args.z_hold_gain, args.z_hold_max)
            env_action = d.project_action_to_env_mode(action_7d, args.action_mode)
            executed_log[step] = action_7d[:2]

            st = raw_obs.get("state")
            if st is not None:
                sv = np.ravel(np.asarray(st, dtype=np.float64))
                if sv.size >= 2:
                    eef_log[step] = sv[:2]

            # The pusher must stay at working height. If it sinks, it is
            # dragging on the table: everything after that is noise, and the
            # arm is loading the table, so stop rather than finish the episode.
            cur_z = d.z_from_obs(raw_obs)
            if args.z_drop_abort > 0 and cur_z is not None:
                if cur_z < args.fixed_z_height - args.z_drop_abort:
                    z_low += 1
                    if z_low == 1:
                        print(f"[WARN] EEF z = {cur_z:.4f} m, "
                              f"{1000 * (args.fixed_z_height - cur_z):.1f} mm "
                              f"below --fixed-z-height at step {step} "
                              f"(demo {demo_step}), eef x={cur_x:.4f} "
                              f"y={eef_log[step][1]:.4f}")
                    if z_low >= max(1, args.z_drop_steps):
                        abort_reason = (
                            f"EEF z stayed >{1000 * args.z_drop_abort:.0f} mm "
                            f"below {args.fixed_z_height} m for {z_low} steps "
                            f"(now {cur_z:.4f} m) at step {step} / demo "
                            f"{demo_step}. The pusher is on the table.")
                        print(f"[ABORT] {abort_reason}")
                        print("        Check, in order: (1) the workspace box "
                              "-- the arm pins at a boundary and the wrist can "
                              "sag from there, (2) whether the server env was "
                              "re-initialised with THIS client's env_params "
                              "(a reused env keeps its own box and z lock), "
                              "(3) the z lock itself. --z-drop-abort 0 "
                              "disables this guard.")
                        break
                else:
                    z_low = 0

            if not args.dry_run:
                if not blocking:
                    wait_s = (last_exec + args.step_duration) - time.time()
                    if wait_s > 0:
                        time.sleep(wait_s)
                step_status = client.step_action(env_action, blocking=blocking)
                last_exec = time.time()
                if step_status != WidowXStatus.SUCCESS:
                    raise RuntimeError(
                        "WidowX step_action failed: status="
                        f"{d.status_name(step_status, WidowXStatus)}, "
                        f"env_action={np.asarray(env_action).tolist()}")
            else:
                time.sleep(args.step_duration)
                last_exec = time.time()

            if step % 20 == 0 or snapped or floored:
                pol = " ".join(
                    f"{labels[vi]}=" + ("-" if not np.isfinite(policy_log[step, vi]).all()
                                        else str(np.round(policy_log[step, vi] * 1000, 2)))
                    for vi in range(len(variants)))
                print(f"[{step:04d}/{n_exec} demo {demo_step:04d}] expert(mm)="
                      f"{np.round(expert_log[step] * 1000, 2)} "
                      f"policy(mm) {pol} env_action={np.round(env_action, 5)}")

            if log_fh is not None:
                if args.save_frames:
                    # Raw newest frame per camera; the per-model resize is
                    # deterministic from these.
                    newest = frame_buf[-1]
                    d.save_fed_png(
                        log_dir / "fed" / f"{step:04d}",
                        np.concatenate([newest[c] for c in cam_ids], axis=-1),
                        cam_ids)
                log_fh.write(json.dumps({
                    "step": step,
                    "demo_step": demo_step,
                    "t": time.time(),
                    "expert": expert_log[step].tolist(),
                    "executed": executed_log[step].tolist(),
                    "policy_norm": na_by_variant,
                    "policy": ({labels[vi]: policy_log[step, vi].tolist()
                                for vi in range(len(variants))
                                if np.isfinite(policy_log[step, vi]).all()}
                               or None),
                    "env_action": [float(x) for x in np.ravel(env_action)],
                    "state": (np.ravel(np.asarray(st, dtype=np.float64)).tolist()
                              if st is not None else None),
                }) + "\n")
                log_fh.flush()

    except KeyboardInterrupt:
        print("\nInterrupted by user.")
    finally:
        if log_fh is not None:
            log_fh.close()
        try:
            client.stop()
        except Exception:
            pass

    done = step + 1
    print(f"\nReplay stopped after {done}/{n_exec} steps"
          + (f" -- ABORTED: {abort_reason}" if abort_reason else "") + ".")
    if done <= 0:
        return 1

    expert_log = expert_log[:done]
    executed_log = executed_log[:done]
    policy_log = policy_log[:done]
    eef_log = eef_log[:done]
    demo_steps = demo_steps[:done]

    clipped = int(np.sum(np.abs(executed_log - expert_log) > 1e-9))
    print(f"Expert steps altered by the deploy pipeline: {clipped}")

    if variants:
        exp_mag = np.linalg.norm(expert_log, axis=1).mean() * 1000
        idle_share = float((np.linalg.norm(expert_log, axis=1) == 0).mean())
        print(f"\nExpert |a| mean = {exp_mag:.2f} mm over {done} executed steps "
              f"({idle_share:.1%} of them idle"
              + (", --drop-zero on" if args.drop_zero else "")
              + f"). Shadow variants (control period "
              f"{args.step_duration * 1000:.0f} ms):")
        print(f"  {'variant':<22} {'model':<14} {'MAE dx':>7} {'MAE dy':>7} "
              f"{'r dx':>6} {'r dy':>6} {'|a|':>7} {'ms/draw':>10}")
        total_ms = 0.0
        for vi, v in enumerate(variants):
            st, valid = variant_stats(expert_log, policy_log[:, vi])
            arr = np.asarray(sample_ms[vi]) if sample_ms[vi] else np.array([np.nan])
            total_ms += float(np.nanmean(arr))
            mag = (np.linalg.norm(policy_log[valid, vi], axis=1).mean() * 1000
                   if valid.any() else np.nan)
            print(f"  {v['label']:<22} "
                  f"{v['model'] + '/' + v['_model']['backend']:<14} "
                  f"{st['mae'][0] * 1000:7.2f} "
                  f"{st['mae'][1] * 1000:7.2f} {st['corr'][0]:6.2f} "
                  f"{st['corr'][1]:6.2f} {mag:7.2f} "
                  f"{np.nanmean(arr):6.1f}/{np.nanmax(arr):.0f}")
        if total_ms > args.step_duration * 1000:
            print(f"[WARN] the variants cost {total_ms:.0f} ms/step together, "
                  f"more than the {args.step_duration * 1000:.0f} ms control "
                  f"period, so the replay ran slower than the demo. Drop the "
                  f"expensive variants (argmax/ddim are the cheap ones) or "
                  f"raise --shadow-every.")

    eef_meas = eef_log if np.isfinite(eef_log).all(axis=1).any() else None
    if eef_meas is not None:
        eef_meas = eef_meas[np.isfinite(eef_meas).all(axis=1)]
    plot_out = args.plot_out
    if plot_out is None:
        plot_out = ((log_dir or ROOT / "results" / f"replay_ep{args.episode}")
                    / "expert_vs_policy.png")
    plot_expert_vs_policy(
        expert_log, policy_log, executed_log, eef_meas,
        ep_eef[demo_steps, :2], demo_steps, labels, plot_out,
        f"episode {args.episode} of {ep_meta.get('source_archive', source)} — "
        "expert (executed) vs "
        + ", ".join(f"{m['name']} ({m['backend'].upper()})" for m in models)
        + " (shadow)" + (", idle steps dropped" if args.drop_zero else ""))

    if log_dir is not None:
        np.savez(log_dir / "replay.npz", expert=expert_log, executed=executed_log,
                 policy=policy_log, eef_live=eef_log,
                 eef_demo=ep_eef[demo_steps], demo_steps=demo_steps,
                 drop_zero=bool(args.drop_zero), idle_eps=float(args.idle_eps),
                 variant_labels=np.array(labels),
                 variant_models=np.array([v["model"] for v in variants]),
                 variant_backends=np.array([v["_model"]["backend"]
                                            for v in variants]),
                 variant_specs=np.array([
                     json.dumps({k: val for k, val in v.items() if k != "_model"})
                     for v in variants]),
                 episode=args.episode, step_duration=args.step_duration)
        print(f"Arrays -> {log_dir / 'replay.npz'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
