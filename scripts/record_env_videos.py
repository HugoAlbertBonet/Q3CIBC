"""Record square-format example videos of the paper's simulated environments.

One clip per environment, 1:1 aspect, written as both h264 mp4 and animated GIF
so they drop straight into slides or a project page (the paper's environments
strip, `scripts/paper_fig_environments.py`, only has stills).

Rollout driver, per environment. The intent was a WiFI-BC clip everywhere it
has weights, but the checkpoints stored on this machine are migration runs
(2026-09-18), not the paper's cluster-trained ones, and score zero on every one
of these tasks — `envs.evaluate` reports the same zero without any recording
attached, so this is the weights, not the camera. Each env therefore uses the
best driver that actually performs the task here:

  * particle2d       — scripted oracle (drive the position setpoint to the first
                       goal, then to the second). Particle has no WiFI-BC
                       checkpoint at all, only an IBC energy model.
  * pushing_pixels   — checkpoint (`--method`, default wifi_bc)
  * pen              — consistency_policy checkpoint, the only one here that
                       solves the task (20% success, 337 return, against
                       WiFI-BC's 0% and -36)
  * kitchen          — D4RL demonstration replayed in the env (4/4 subtasks)
  * libero_goal      — checkpoint (`--method`, default wifi_bc)

Pass `--method` to film a different method's checkpoint on the policy-driven
envs; swap the drivers in ENVS once real weights land.

`--q3c` ignores all of that and films THIS repo's own Q3C policies instead: it
reads the env's hyperparameter-search trials, takes the best record (or
`--trial <id>` / `--checkpoint <dir>`), rebuilds that trial's config the way
scripts/reeval_trials.py does, and rolls out through
`hyperparam_search.evaluate_q3c` with `simulations/`. That path has no wifi_bc
dependency, because the hpsearch checkpoints live on the cluster — see
batches/q3cEnvVideos.txt, which submits one job per environment.

The policy clips reuse the wifi_bc repo's evaluator (`envs/evaluate.py`) rather
than rebuilding the policy here: it already resolves the per-env architecture,
normalisation stats and inference settings from the checkpoint. The simulation
class is subclassed on the fly so every `env.step` also grabs a frame, and each
episode becomes its own segment; the clip written out is the first SUCCESSFUL
episode (falling back to the longest one if no seed succeeds).

Frames come from the highest-resolution view each env can give without changing
what the policy sees — the pixel policies are fed their native 128/240px
observations while the video is rendered from the same camera at video scale.

    uv run python scripts/record_env_videos.py                       # all five
    uv run python scripts/record_env_videos.py --envs pen kitchen
    uv run python scripts/record_env_videos.py --episodes 8 --size 512
    uv run python scripts/record_env_videos.py --envs pen --method diffusion_policy
    uv run python scripts/record_env_videos.py --envs pen --q3c --trial best   # cluster
"""

from __future__ import annotations

import argparse
import importlib
import json
import os
import sys
from contextlib import contextmanager
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
# The trained checkpoints, the env code they were trained against and the config
# that describes them all live in the split-out method repo.
WIFI_BC_ROOT = Path(os.environ.get("WIFI_BC_ROOT", "/home/hugo/wifi_bc"))

# MuJoCo (pen, kitchen) and robosuite (LIBERO) need an offscreen GL backend;
# EGL is the one that works headless under WSL2. LIBERO prompts on stdin for a
# dataset path when it finds no config, which hangs a non-interactive run, so
# point it at the checked-in one.
os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("LIBERO_CONFIG_PATH", str(ROOT / ".libero"))


# ── environment table ───────────────────────────────────────────────────────

# label       : human-readable name for logging
# config_env  : key into wifi_bc config/config.json -> environments
# checkpoint  : directory under <wifi_bc>/checkpoints
# sim         : (module, class) of the simulation to subclass for recording
# fps         : playback rate of the mp4 (roughly the env's control rate)
ENVS: dict[str, dict] = {
    "particle2d": {
        "label": "Particle 2D",
        "driver": "oracle",
        "fps": 20,
    },
    "particle16": {
        "label": "Particle 16-D",
        # Q3C-only: the search never trained a 2-D particle policy, so
        # particle2d has no checkpoint and this has no non-q3c driver.
        # ParticleEnv renders n>2 as a stack of per-dimension traces, which is
        # a tall plot — letterbox it instead of cropping 14 panels away.
        "driver": "q3c",
        "fit": "pad",
        "fps": 20,
        "q3c": {
            "active_env": "particle",
            "sim": ("simulations.particle_simulation", "ParticleSimulation"),
            "metric": "success_rate",
        },
    },
    "pushing_pixels": {
        "label": "Pushing (pixels)",
        # Every local checkpoint (all five methods) scores 0 here, so the clip
        # comes from IBC's own scripted oracle — the policy the training
        # demonstrations were generated with.
        "driver": "push_oracle",
        "config_env": "pushing_pixels",
        "method": "wifi_bc",
        "sim": ("envs.pushing_pixels_simulation", "PushingPixelsSimulation"),
        "fps": 10,
        "q3c": {
            "active_env": "pushing_pixels",
            "sim": ("simulations.pushing_pixels_simulation", "PushingPixelsSimulation"),
            "metric": "success_rate",
        },
    },
    "pen": {
        "label": "Adroit Pen",
        "driver": "policy",
        "config_env": "pen",
        # None of the locally-stored WiFI-BC weights clear zero on the paper's
        # tasks (they are 2026-09-18 migration runs, not the cluster-trained
        # ones); consistency_policy is the only pen checkpoint on this machine
        # that solves the task at all, so it is what the clip shows.
        "method": "consistency_policy",
        "sim": ("envs.pen_human_v2_simulation", "PenHumanV2Simulation"),
        "fps": 25,
        # Adroit's default camera puts the top third of the frame in empty
        # space above the table; tighten onto the hand and the target pen.
        "crop": (0.06, 0.18, 0.94, 1.0),
        "q3c": {
            "active_env": "pen",
            "sim": ("simulations.pen_human_v2_simulation", "PenHumanV2Simulation"),
            "metric": "success_rate",
        },
    },
    "kitchen": {
        "label": "Franka Kitchen",
        # Replaying a D4RL demo completes all four subtasks every time, which
        # no local checkpoint does, so the kitchen clip is a demonstration.
        "driver": "demo",
        "config_env": "kitchen",
        "dataset": "D4RL/kitchen/complete-v2",
        "fps": 30,
        "q3c": {
            "active_env": "kitchen",
            "sim": ("simulations.kitchen_simulation", "KitchenSimulation"),
            # Kitchen's headline metric is subtasks solved; its success_rate is
            # "all four", which is 0.0 for every trial in the search.
            "metric": "avg_tasks_completed",
        },
    },
    "libero_goal": {
        "label": "LIBERO-Goal",
        # As with kitchen: no local checkpoint (wifi_bc, consistency_policy,
        # bc_mse all 0/3) completes a LIBERO-Goal task, so the clip replays a
        # demonstration. Needs scripts/download_libero_goal.py to have run.
        "driver": "libero_demo",
        "config_env": "libero_goal_pixels",
        "method": "wifi_bc",
        "benchmark": "libero_goal",
        "task_name": "put_the_bowl_on_the_plate",
        "sim": ("envs.libero_goal_pixels_simulation", "LiberoGoalPixelsSimulation"),
        # LIBERO runs at 20Hz, so 20fps is real time for the demo.
        "fps": 20,
        "q3c": {
            "active_env": "libero_goal_pixels",
            "sim": ("simulations.libero_goal_pixels_simulation",
                    "LiberoGoalPixelsSimulation"),
            "metric": "success_rate",
        },
    },
}


# ── frame grabbing ──────────────────────────────────────────────────────────

def _grab_render(sim, env, size: int) -> np.ndarray | None:
    """Plain gymnasium `render()` — pen, kitchen, particle."""
    del sim, size
    frame = env.render()
    return None if frame is None else np.asarray(frame, dtype=np.uint8)


def _grab_pushing_pixels(sim, env, size: int) -> np.ndarray | None:
    """Re-render the IBC BlockPush camera at video scale.

    The policy's observation stays the env's native 240x320; `_render_camera`
    just takes another picture with the same camera parameters, so nothing the
    policy consumes is touched. Keep the 4:3 aspect the camera intrinsics
    assume and let the square crop happen later.
    """
    del sim
    inner = getattr(env, "_env", None)
    render_camera = getattr(inner, "_render_camera", None)
    if render_camera is not None:
        h = size
        w = int(round(size * 4 / 3))
        frame = np.asarray(render_camera(image_size=(h, w)), dtype=np.uint8)
        return frame[..., :3]
    return None


def _grab_libero(sim, env, size: int) -> np.ndarray | None:
    """Re-render LIBERO's agentview at video scale.

    The policy is fed the env's own 128px `agentview_image`; this is an extra
    MuJoCo render off the same camera. robosuite renders bottom-up, hence the
    vertical flip.
    """
    del sim
    mj_sim = getattr(env, "sim", None)
    if mj_sim is None:
        return None
    frame = mj_sim.render(camera_name="agentview", width=size, height=size)
    return np.asarray(frame, dtype=np.uint8)[::-1, :, :3]


GRABBERS = {
    "pushing_pixels": _grab_pushing_pixels,
    "particle16": _grab_render,
    "libero_goal": _grab_libero,
    "pen": _grab_render,
    "kitchen": _grab_render,
    "particle2d": _grab_render,
}


class Recorder:
    """Collects one list of frames per episode."""

    def __init__(self, grab, render_size: int) -> None:
        self._grab = grab
        self._render_size = render_size
        self.segments: list[list[np.ndarray]] = []

    def new_segment(self) -> None:
        self.segments.append([])

    def capture(self, sim, env) -> None:
        if not self.segments:
            self.new_segment()
        try:
            frame = self._grab(sim, env, self._render_size)
        except Exception as exc:  # noqa: BLE001 — a dropped frame must not kill the rollout
            print(f"  [warn] frame grab failed: {type(exc).__name__}: {exc}")
            return
        if frame is not None:
            self.segments[-1].append(frame)

    def attach(self, sim, env) -> None:
        """Wrap this env instance's step/reset so every transition is filmed."""
        if getattr(env, "_recorder_attached", False):
            return
        orig_step, orig_reset = env.step, env.reset

        def step(*args, **kwargs):
            out = orig_step(*args, **kwargs)
            self.capture(sim, env)
            return out

        def reset(*args, **kwargs):
            out = orig_reset(*args, **kwargs)
            self.capture(sim, env)
            return out

        env.step = step
        env.reset = reset
        env._recorder_attached = True


# ── policy-driven clips (reuse the wifi_bc evaluator) ───────────────────────

@contextmanager
def _wifi_bc_importable():
    """Put the method repo on sys.path and make it cwd.

    Its modules (`envs`, `wifi_bc`, `baselines`) import each other by top-level
    name and a few resolve data relative to the repo root.
    """
    cwd = Path.cwd()
    sys.path.insert(0, str(WIFI_BC_ROOT))
    os.chdir(WIFI_BC_ROOT)
    try:
        yield
    finally:
        os.chdir(cwd)
        try:
            sys.path.remove(str(WIFI_BC_ROOT))
        except ValueError:
            pass


# Sims whose create_env forwards self.render_mode into the env constructor.
_RENDER_MODE_AT_CONSTRUCTION = {"pen", "particle16"}


def _make_recording_sim(base_cls, env_key: str, rec: Recorder):
    """Subclass a simulation so it renders, films and segments per episode."""

    class RecordingSimulation(base_cls):  # type: ignore[misc, valid-type]
        def __init__(self, *args, **kwargs):
            if env_key in _RENDER_MODE_AT_CONSTRUCTION:
                # These sims hand render_mode to the env constructor, and both
                # MuJoCo (pen) and ParticleEnv only produce frames when it was
                # set there — the evaluators pass render_mode=None.
                kwargs["render_mode"] = "rgb_array"
            super().__init__(*args, **kwargs)

        def create_env(self, *args, **kwargs):
            env = super().create_env(*args, **kwargs)
            rec.attach(self, env)
            return env

        def run_episode(self, seed=None):
            rec.new_segment()
            return super().run_episode(seed=seed)

    if env_key == "kitchen":
        def create_env(self):
            # KitchenSimulation recovers the env from the Minari dataset so the
            # obs layout matches the demos; render_mode has to go in at
            # construction (setting it afterwards leaves the inner robot env
            # rendering None), so recover it again with rendering on.
            import minari

            ds = minari.load_dataset(self.dataset_name, download=True)
            env = ds.recover_environment(eval_env=True, render_mode="rgb_array")
            rec.attach(self, env)
            return env

        RecordingSimulation.create_env = create_env  # type: ignore[assignment]

    RecordingSimulation.__name__ = f"Recording{base_cls.__name__}"
    return RecordingSimulation


def record_policy_env(env_key: str, spec: dict, args) -> tuple[list[np.ndarray], str]:
    """Run the checkpoint for *env_key* and return (frames, provenance note)."""
    method = args.method or spec["method"]
    ckpt = WIFI_BC_ROOT / "checkpoints" / method / spec["config_env"]
    if not ckpt.is_dir():
        raise FileNotFoundError(f"checkpoint dir not found: {ckpt}")

    with _wifi_bc_importable():
        from envs.evaluate import _load_config, evaluate

        config = _load_config(str(WIFI_BC_ROOT / "config" / "config.json"))
        config["active_env"] = spec["config_env"]
        config["environments"][spec["config_env"]]["num_eval_seeds"] = args.episodes

        rec = Recorder(GRABBERS[env_key], args.render_size)
        module_name, cls_name = spec["sim"]
        module = importlib.import_module(module_name)
        base_cls = getattr(module, cls_name)
        # evaluate() imports the simulation class inside the call, so swapping
        # the module attribute is enough to have it build the recording one.
        setattr(module, cls_name, _make_recording_sim(base_cls, env_key, rec))
        try:
            results = evaluate(str(ckpt), config)
        finally:
            setattr(module, cls_name, base_cls)

    if results.get("error"):
        raise RuntimeError(f"evaluate() failed: {results['error']}")

    per_seed = results.get("per_seed", [])
    segments = [s for s in rec.segments if s]
    if not segments:
        raise RuntimeError("no frames captured")

    successes = [bool(r.get("success", False)) for r in per_seed][: len(segments)]
    pick = next((i for i, ok in enumerate(successes) if ok), None)
    if pick is None:
        # Nothing succeeded: show the longest attempt, and say so.
        pick = max(range(len(segments)), key=lambda i: len(segments[i]))
        note = f"episode {pick} (no success in {len(segments)} episodes)"
    else:
        note = f"episode {pick} (success)"
    detail = per_seed[pick] if pick < len(per_seed) else {}
    extra = {k: detail[k] for k in ("seed", "task_name", "tasks_completed", "reward") if k in detail}
    print(f"  success rate over {len(segments)} episodes: {results.get('success_rate', 0.0):.2f}")
    return segments[pick], f"{note} {extra}".strip()


# ── Q3C checkpoint clips (this repo's own trained policies) ─────────────────

def _q3c_records(hs, active_env: str, script: str) -> list[dict]:
    path = hs._trials_path(script, active_env=active_env)
    if not Path(path).exists():
        raise FileNotFoundError(f"no trials file at {path}")
    return [json.loads(line) for line in open(path) if line.strip()]


def _q3c_pick_record(records: list[dict], metric: str, trial: str | None) -> dict:
    """The record to film: an explicit trial id, or the best one by *metric*."""
    if trial not in (None, "best"):
        wanted = int(trial)
        for rec in records:
            if int(rec.get("trial_id", -1)) == wanted:
                return rec
        raise KeyError(f"trial {wanted} not in trials.jsonl")

    usable = [
        r for r in records
        if r.get("checkpoint_dir")
        and not r.get("error") and not r.get("eval_error")
        and not r.get("training_failed")
        and r.get(metric) is not None
    ]
    if not usable:
        raise RuntimeError(f"no usable record carrying {metric!r}")
    # Re-evaluations are eligible: they are corrected scores for the same
    # weights, and they carry the eval-side params that produced them.
    return max(usable, key=lambda r: r[metric])


def _q3c_config(hs, ckpt_dir: str, rec: dict | None, active_env: str, args) -> dict:
    """Rebuild the config this checkpoint was scored with.

    Same resolution as scripts/reeval_trials.py: prefer the per-run config
    saved beside the checkpoint, and otherwise reconstruct it from the base
    config plus the trial's recorded params. This matters for more than tidiness
    — the inference-side settings (DFO iterations, sample counts, Langevin) live
    in the config, and falling back to the defaults silently scores a different
    policy than the trial did.
    """
    cfg_path = Path(ckpt_dir) / "config.json"
    if cfg_path.exists():
        with open(cfg_path) as f:
            config = json.load(f)
    elif rec is not None:
        base = hs.load_config()
        base["active_env"] = active_env
        config = hs.apply_params_to_config(base, rec.get("params") or {})
        print(f"  no per-run config at {cfg_path}; reconstructed from base "
              f"config + {len(rec.get('params') or {})} recorded params")
    else:
        raise FileNotFoundError(
            f"{cfg_path} missing and no trial record to reconstruct it from"
        )

    if config.get("active_env") != active_env:
        raise ValueError(
            f"config active_env={config.get('active_env')!r}, expected {active_env!r}"
        )
    config["environments"][active_env]["num_eval_seeds"] = args.episodes
    return config


def record_q3c_env(env_key: str, spec: dict, args) -> tuple[list[np.ndarray], str]:
    """Film this repo's Q3C policy, as scored by the hyperparameter search.

    Runs against Q3CIBC's own `hyperparam_search.evaluate_q3c` and
    `simulations/` — no wifi_bc checkout — so it works unchanged on the cluster
    where the hpsearch checkpoints actually live.
    """
    q3c = spec.get("q3c")
    if q3c is None:
        raise RuntimeError(f"{env_key} has no Q3C checkpoint (see the ENVS table)")

    sys.path.insert(0, str(ROOT))
    import hyperparam_search as hs

    active_env = q3c["active_env"]
    metric = q3c.get("metric", "success_rate")
    rec = None
    if args.checkpoint:
        ckpt_dir = args.checkpoint
        source = f"--checkpoint {ckpt_dir}"
    else:
        records = _q3c_records(hs, active_env, args.script)
        rec = _q3c_pick_record(records, metric, args.trial)
        ckpt_dir = rec["checkpoint_dir"]
        source = (f"trial {rec.get('trial_id')} ({metric}={rec[metric]}, "
                  f"run {rec.get('run_id')})")
    if not Path(ckpt_dir).is_dir():
        raise FileNotFoundError(
            f"checkpoint dir {ckpt_dir} not readable from here — the hpsearch "
            f"checkpoints live on the cluster; run this batch there"
        )
    print(f"  {source}")

    config = _q3c_config(hs, ckpt_dir, rec, active_env, args)

    recorder = Recorder(GRABBERS[env_key], args.render_size)
    module_name, cls_name = q3c["sim"]
    module = importlib.import_module(module_name)
    base_cls = getattr(module, cls_name)
    # evaluate_q3c imports the simulation class inside the call, so swapping the
    # module attribute is enough to have it build the recording subclass.
    setattr(module, cls_name, _make_recording_sim(base_cls, env_key, recorder))
    try:
        results = hs.evaluate_q3c(ckpt_dir, config)
    finally:
        setattr(module, cls_name, base_cls)

    if results.get("error") or results.get("eval_error"):
        raise RuntimeError(f"evaluate_q3c failed: {results.get('error') or results.get('eval_error')}")

    segments = [seg for seg in recorder.segments if seg]
    if not segments:
        raise RuntimeError("no frames captured")
    details = results.get("eval_details", results.get("per_seed", [])) or []

    successes = [bool(d.get("success", False)) for d in details][: len(segments)]
    pick = next((i for i, ok in enumerate(successes) if ok), None)
    if pick is None and active_env == "kitchen":
        # Kitchen reports subtasks, not success: film the best episode instead.
        done = [int(d.get("tasks_completed", 0)) for d in details][: len(segments)]
        if done:
            pick = max(range(len(done)), key=lambda i: done[i])
    if pick is None:
        pick = max(range(len(segments)), key=lambda i: len(segments[i]))
        note = f"episode {pick} (no success in {len(segments)} episodes)"
    else:
        note = f"episode {pick}"
    detail = details[pick] if pick < len(details) else {}
    extra = {k: detail[k] for k in ("seed", "success", "tasks_completed", "reward")
             if k in detail}
    headline = results.get(metric)
    print(f"  {metric} over {len(segments)} episodes: {headline}")
    return segments[pick], f"q3c {source}, {note} {extra}"


# ── LIBERO demonstration clip ───────────────────────────────────────────────

def record_libero_demo(env_key: str, spec: dict, args) -> tuple[list[np.ndarray], str]:
    """Film a LIBERO demonstration replayed in its own env.

    LIBERO ships each task's demos with the MuJoCo state they started from, so
    `set_init_state(states[0])` followed by the recorded actions reproduces the
    demonstration exactly — unlike the Adroit demos, which carry no state to
    restore. No local checkpoint solves LIBERO-Goal, so this is what shows the
    task being done.
    """
    with _wifi_bc_importable():
        import h5py
        from libero.libero import benchmark, get_libero_path
        from libero.libero.envs import OffScreenRenderEnv

        bench = benchmark.get_benchmark_dict()[spec["benchmark"]]()
        names = [bench.get_task(i).name for i in range(bench.n_tasks)]
        wanted = args.task or spec["task_name"]
        if wanted not in names:
            raise KeyError(f"task {wanted!r} not in {spec['benchmark']}; have {names}")
        task = bench.get_task(names.index(wanted))

        bddl = os.path.join(get_libero_path("bddl_files"), task.problem_folder, task.bddl_file)
        demo_path = os.path.join(get_libero_path("datasets"), spec["benchmark"],
                                 f"{task.name}_demo.hdf5")
        if not os.path.exists(demo_path):
            raise FileNotFoundError(
                f"{demo_path} missing — run scripts/download_libero_goal.py first."
            )

        # Nothing here consumes observations, so the env can render straight at
        # video scale instead of the policy's 128px.
        env = OffScreenRenderEnv(bddl_file_name=bddl,
                                 camera_heights=args.render_size,
                                 camera_widths=args.render_size)
        rec = Recorder(GRABBERS[env_key], args.render_size)
        best: list[np.ndarray] = []
        note = ""
        try:
            with h5py.File(demo_path, "r") as f:
                demos = sorted(f["data"].keys(), key=lambda k: int(k.split("_")[-1]))
                for i, demo in enumerate(demos[: args.episodes]):
                    actions = f[f"data/{demo}/actions"][()]
                    states = f[f"data/{demo}/states"][()]
                    rec.new_segment()
                    env.reset()
                    env.set_init_state(states[0])
                    rec.capture(None, env)
                    success = False
                    for action in actions:
                        _, _, done, _ = env.step(action)
                        rec.capture(None, env)
                        success = success or bool(env.check_success())
                        if done:
                            break
                    frames = rec.segments[-1]
                    print(f"  demo {i} ({demo}): {len(frames)} frames, success={success}")
                    if success:
                        return frames, f"demo {demo} of {task.name!r} (success)"
                    if len(frames) > len(best):
                        best, note = frames, f"demo {demo} of {task.name!r} (no success)"
        finally:
            env.close()

    if not best:
        raise RuntimeError("no frames captured")
    return best, note


# ── pushing oracle clip ─────────────────────────────────────────────────────

class _OrientedPushOracle:
    """Numpy port of IBC's `oriented_push_oracle.OrientedPushOracle`.

    The vendored original is a tf-agents `py_policy` wired to tf-agents specs
    (`simulations/ibc_block_pushing/oracles/`), and the vendoring deliberately
    skips those modules. The control law itself is plain numpy, so it is
    reproduced here: approach a point 5cm behind the block on the block-target
    line, re-orient the block when its yaw is too far off that line, then push
    through it.
    """

    # Block has 4-way symmetry, so yaw error folds into +-45 degrees.
    THETA_TO_ORIENT = 0.2
    THETA_FLAT_ENOUGH = 0.03
    ORIENT_CIRCLE_DIAMETER = 0.025

    def __init__(self, control_frequency: float) -> None:
        self.control_frequency = control_frequency
        self.phase = "move_to_pre_block"

    def reset(self) -> None:
        self.phase = "move_to_pre_block"

    @staticmethod
    def _rotate(theta: float, vec: np.ndarray) -> np.ndarray:
        c, s = np.cos(theta), np.sin(theta)
        return np.array([[c, -s], [s, c]]) @ vec

    def act(self, obs: dict) -> np.ndarray:
        xy_block = np.asarray(obs["block_translation"], dtype=np.float64)[:2]
        theta_block = float(np.asarray(obs["block_orientation"]).reshape(-1)[0])
        xy_target = np.asarray(obs["target_translation"], dtype=np.float64)[:2]
        xy_ee = np.asarray(obs["effector_target_translation"], dtype=np.float64)[:2]

        to_target = xy_target - xy_block
        dir_to_target = to_target / max(np.linalg.norm(to_target), 1e-9)
        theta_error = np.arctan2(dir_to_target[1], dir_to_target[0]) - theta_block
        while theta_error > np.pi / 4:
            theta_error -= np.pi / 2
        while theta_error < -np.pi / 4:
            theta_error += np.pi / 2

        xy_pre_block = xy_block - dir_to_target * 0.05
        xy_delta_to_nexttoblock = (xy_block - dir_to_target * 0.03) - xy_ee
        xy_delta_to_touchingblock = (xy_block - dir_to_target * 0.01) - xy_ee
        block_to_ee = xy_ee - xy_block
        dir_block_to_ee = block_to_ee / max(np.linalg.norm(block_to_ee), 1e-9)

        max_step_velocity = 0.35
        xy_delta = xy_delta_to_touchingblock

        if self.phase == "move_to_pre_block":
            max_step_velocity = 0.3
            xy_delta = xy_pre_block - xy_ee
            if np.linalg.norm(xy_delta) < 0.001:
                self.phase = "move_to_block"

        if self.phase == "move_to_block":
            xy_delta = xy_delta_to_nexttoblock
            if np.linalg.norm(xy_delta_to_nexttoblock) < 0.001:
                self.phase = "push_block"
            if theta_error > self.THETA_TO_ORIENT:
                self.phase = "orient_block_left"
            elif theta_error < -self.THETA_TO_ORIENT:
                self.phase = "orient_block_right"

        if self.phase == "push_block":
            xy_delta = xy_delta_to_touchingblock
            if abs(theta_error) > self.THETA_TO_ORIENT:
                self.phase = "move_to_pre_block"

        if self.phase in ("orient_block_left", "orient_block_right"):
            max_step_velocity = 0.15
            sign = 1.0 if self.phase == "orient_block_left" else -1.0
            spot = xy_block + self._rotate(sign * 0.2, dir_block_to_ee) * self.ORIENT_CIRCLE_DIAMETER
            xy_delta = spot - xy_ee
            if sign > 0 and theta_error < self.THETA_FLAT_ENOUGH:
                self.phase = "move_to_pre_block"
            if sign < 0 and theta_error > -self.THETA_FLAT_ENOUGH:
                self.phase = "move_to_pre_block"

        # Cap as a velocity so the oracle is independent of control frequency.
        max_step_distance = max_step_velocity / self.control_frequency
        length = float(np.linalg.norm(xy_delta))
        if length > max_step_distance:
            xy_delta = xy_delta / length * max_step_distance
        return np.asarray(xy_delta, dtype=np.float32)


def record_pushing_oracle(env_key: str, spec: dict, args) -> tuple[list[np.ndarray], str]:
    """Film the scripted push oracle solving the block-pushing task."""
    with _wifi_bc_importable():
        from envs.pushing_pixels_env import PushingPixelsEnv

        env = PushingPixelsEnv(n_steps=args.max_steps or 100)
        rec = Recorder(GRABBERS[env_key], args.render_size)
        oracle = _OrientedPushOracle(env._env.get_control_frequency())

        best: tuple[float, list[np.ndarray], int] = (np.inf, [], -1)
        try:
            for ep in range(args.episodes):
                rec.new_segment()
                env.reset(seed=args.seed + ep)
                rec.capture(None, env)
                oracle.reset()
                done = False
                success = False
                min_dist = np.inf
                while not done:
                    action = oracle.act(env._last_obs_dict)
                    _, _, terminated, truncated, info = env.step(action)
                    rec.capture(None, env)
                    min_dist = min(min_dist, float(info["block_to_target_distance"]))
                    success = success or bool(info["success"])
                    done = terminated or truncated
                frames = rec.segments[-1]
                print(f"  oracle episode {ep}: {len(frames)} frames, "
                      f"success={success}, min dist={min_dist:.4f}")
                if success:
                    return frames, f"oracle episode {ep} (success, seed {args.seed + ep})"
                if min_dist < best[0]:
                    best = (min_dist, frames, ep)
        finally:
            env.close()

    min_dist, frames, ep = best
    if not frames:
        raise RuntimeError("no frames captured")
    return frames, f"oracle episode {ep} (no success, closest {min_dist:.4f})"


# ── demonstration clips (replay a dataset episode) ──────────────────────────

def record_demo_env(env_key: str, spec: dict, args) -> tuple[list[np.ndarray], str]:
    """Film a D4RL demonstration replayed in the env.

    FrankaKitchen's reset is deterministic enough that stepping a demo's action
    sequence re-drives the same four subtasks (verified: 4/4 completed on each
    of the first demos), so this gives a clip of the task being solved without
    depending on any checkpoint.
    """
    import minari

    ds = minari.load_dataset(spec["dataset"], download=True)
    env = ds.recover_environment(eval_env=True, render_mode="rgb_array")
    rec = Recorder(GRABBERS[env_key], args.render_size)
    n_targets = len(getattr(env.unwrapped, "goal", ()) or ())

    best: tuple[int, list[np.ndarray], int] = (-1, [], -1)
    try:
        for ep_i, episode in enumerate(ds.iterate_episodes()):
            if ep_i >= args.episodes:
                break
            rec.new_segment()
            env.reset(seed=args.seed)
            rec.capture(None, env)
            done = 0
            for action in episode.actions:
                _, _, terminated, truncated, info = env.step(action)
                rec.capture(None, env)
                done = len(info.get("episode_task_completions", []))
                if terminated or truncated:
                    break
            frames = rec.segments[-1]
            print(f"  demo {ep_i}: {len(frames)} frames, subtasks {done}/{n_targets}")
            if done > best[0]:
                best = (done, frames, ep_i)
            if n_targets and done >= n_targets:
                break
    finally:
        env.close()

    done, frames, ep_i = best
    if not frames:
        raise RuntimeError("no frames captured")
    return frames, f"demo episode {ep_i} ({done}/{n_targets} subtasks)"


# ── particle oracle clip ────────────────────────────────────────────────────

def record_particle(args) -> tuple[list[np.ndarray], str]:
    """Scripted oracle on the 2-D particle env: first goal, then second goal."""
    with _wifi_bc_importable():
        from envs.particle_env import ParticleEnv

        n_dim = 2
        env = ParticleEnv(n_dim=n_dim, render_mode="rgb_array")
        rec = Recorder(GRABBERS["particle2d"], args.render_size)

        best: list[np.ndarray] = []
        note = ""
        for ep in range(args.episodes):
            rec.new_segment()
            obs, _ = env.reset(seed=args.seed + ep)
            rec.capture(None, env)
            # obs = [pos_agent | vel_agent | pos_first_goal | pos_second_goal]
            first_goal = obs[2 * n_dim:3 * n_dim].copy()
            second_goal = obs[3 * n_dim:4 * n_dim].copy()
            reached_first = False
            success = False
            done = False
            while not done:
                pos = obs[:n_dim]
                if not reached_first and np.linalg.norm(pos - first_goal) < env.goal_distance:
                    reached_first = True
                # The action IS a position setpoint; the env's PD controller
                # does the interpolation, so aiming straight at the goal is the
                # oracle.
                target = second_goal if reached_first else first_goal
                obs, _, terminated, truncated, info = env.step(
                    np.clip(target, 0.0, 1.0).astype(np.float32)
                )
                rec.capture(None, env)
                success = success or bool(info.get("success", False))
                done = terminated or truncated
            frames = rec.segments[-1]
            print(f"  episode {ep}: {len(frames)} frames, success={success}")
            if success:
                env.close()
                return frames, f"oracle episode {ep} (success, seed {args.seed + ep})"
            if len(frames) > len(best):
                best, note = frames, f"oracle episode {ep} (no success, seed {args.seed + ep})"
        env.close()
        if not best:
            raise RuntimeError("no frames captured")
        return best, note


# ── video writing ───────────────────────────────────────────────────────────

def _resize(img: np.ndarray, size: int) -> np.ndarray:
    """Resize to size x size. cv2 when present, Pillow otherwise.

    cv2 only ships in the `libero` extra (pyproject overrides robosuite's
    opencv-python with the headless build), so a SLURM job that synced without
    that extra must still be able to write a video. Pillow comes in with
    imageio, which is a core dependency.
    """
    try:
        import cv2
    except ImportError:
        from PIL import Image
        return np.asarray(Image.fromarray(img).resize((size, size), Image.LANCZOS))
    interp = cv2.INTER_AREA if img.shape[0] >= size else cv2.INTER_CUBIC
    return cv2.resize(img, (size, size), interpolation=interp)


def _to_square(frame: np.ndarray, size: int,
               crop: tuple[float, float, float, float] | None = None,
               fit: str = "crop") -> np.ndarray:
    """Make one frame square at size x size.

    `crop` is (x0, y0, x1, y1) in 0..1, applied first. It exists because a
    couple of the envs' default cameras frame the scene loosely (Adroit's puts
    a third of the picture in empty sky), and the tighter box is chosen per env
    in ENVS rather than guessed here.

    `fit` decides what happens to a non-square frame: "crop" centre-crops to
    1:1 (right for a camera view), "pad" letterboxes the whole frame into the
    square (right for a plot, where cropping would throw away panels).
    """
    if crop is not None:
        fh, fw = frame.shape[:2]
        x0, y0, x1, y1 = crop
        frame = frame[int(y0 * fh):int(y1 * fh), int(x0 * fw):int(x1 * fw)]

    h, w = frame.shape[:2]
    if fit == "pad":
        side = max(h, w)
        # Pad with the frame's own corner colour so a matplotlib panel keeps
        # its white surround instead of gaining black bars.
        canvas = np.empty((side, side, frame.shape[2]), dtype=frame.dtype)
        canvas[:] = frame[0, 0]
        top = (side - h) // 2
        left = (side - w) // 2
        canvas[top:top + h, left:left + w] = frame
        return _resize(canvas, size)

    side = min(h, w)
    top = (h - side) // 2
    left = (w - side) // 2
    return _resize(frame[top:top + side, left:left + side], size)


def write_clips(frames: list[np.ndarray], out_base: Path, args, fps: int,
                crop: tuple[float, float, float, float] | None = None,
                fit: str = "crop") -> list[Path]:
    import imageio.v2 as iio

    square = [_to_square(f, args.size, crop, fit) for f in frames]
    out_base.parent.mkdir(parents=True, exist_ok=True)
    written = []

    mp4 = out_base.with_suffix(".mp4")
    # yuv420p (imageio's default for libx264) keeps the file playable in
    # browsers and Keynote; macro_block_size=None stops it padding the square.
    iio.mimwrite(mp4, square, fps=fps, codec="libx264", quality=8,
                 macro_block_size=None)
    written.append(mp4)

    if not args.no_gif:
        stride = max(1, round(fps / args.gif_fps))
        gif_frames = [_to_square(f, args.gif_size, crop, fit) for f in frames[::stride]]
        gif = out_base.with_suffix(".gif")
        iio.mimwrite(gif, gif_frames, duration=1000.0 / max(1.0, fps / stride), loop=0)
        written.append(gif)

    return written


# ── entry point ─────────────────────────────────────────────────────────────

def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--envs", nargs="+", default=list(ENVS), choices=list(ENVS),
                    help="environments to record (default: all)")
    ap.add_argument("--out", type=Path, default=ROOT / "results/env_videos",
                    help="output directory")
    ap.add_argument("--episodes", type=int, default=5,
                    help="episodes to roll out per env; the first successful one is kept")
    ap.add_argument("--size", type=int, default=512, help="mp4 edge length (square)")
    ap.add_argument("--render-size", type=int, default=512,
                    help="render height requested from the env before cropping")
    ap.add_argument("--gif-size", type=int, default=320, help="gif edge length (square)")
    ap.add_argument("--gif-fps", type=float, default=15.0, help="gif frame rate")
    ap.add_argument("--no-gif", action="store_true", help="write only the mp4")
    ap.add_argument("--q3c", action="store_true",
                    help="film this repo's Q3C checkpoints (hyperparam_search trials) "
                         "instead of each env's default driver")
    ap.add_argument("--trial", default="best",
                    help="with --q3c: trial id from trials.jsonl, or 'best' (default)")
    ap.add_argument("--checkpoint", default=None,
                    help="with --q3c: film this checkpoint dir directly, ignoring trials.jsonl")
    ap.add_argument("--script", default="combinedv2_cpascounter_training.py",
                    help="with --q3c: training script whose trials to read")
    ap.add_argument("--task", default=None,
                    help="LIBERO task name to film (default: the env's task_name)")
    ap.add_argument("--max-steps", type=int, default=0,
                    help="episode step cap for the scripted drivers (0 = the env default)")
    ap.add_argument("--method", default=None,
                    choices=["wifi_bc", "ibc", "diffusion_policy", "consistency_policy", "bc_mse"],
                    help="override the checkpoint method for every policy-driven env")
    ap.add_argument("--seed", type=int, default=0, help="first seed for the particle oracle")
    args = ap.parse_args()

    failures: list[str] = []
    for env_key in args.envs:
        spec = ENVS[env_key]
        print(f"\n=== {spec['label']} ({env_key}) ===")
        try:
            driver = "q3c" if args.q3c else spec["driver"]
            if driver == "oracle":
                frames, note = record_particle(args)
            elif driver == "q3c":
                frames, note = record_q3c_env(env_key, spec, args)
            elif driver == "libero_demo":
                frames, note = record_libero_demo(env_key, spec, args)
            elif driver == "push_oracle":
                frames, note = record_pushing_oracle(env_key, spec, args)
            elif driver == "demo":
                frames, note = record_demo_env(env_key, spec, args)
            else:
                frames, note = record_policy_env(env_key, spec, args)
            written = write_clips(frames, args.out / env_key, args, spec["fps"],
                                  spec.get("crop"), spec.get("fit", "crop"))
        except Exception as exc:  # noqa: BLE001 — one env failing must not sink the rest
            print(f"  FAILED: {type(exc).__name__}: {exc}")
            failures.append(f"{env_key}: {type(exc).__name__}: {exc}")
            continue
        dur = len(frames) / spec["fps"]
        print(f"  {note}")
        print(f"  {len(frames)} frames, {dur:.1f}s @ {spec['fps']}fps")
        for path in written:
            print(f"  -> {path}  ({path.stat().st_size / 1e6:.2f} MB)")

    if failures:
        print("\nFailed:")
        for f in failures:
            print(f"  {f}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
