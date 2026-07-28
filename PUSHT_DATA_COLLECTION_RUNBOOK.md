# Push-T (WidowX) — re-collecting the demonstration set

Replaces `data/pusht_widowx_data.zip` (150 eps, 2026-03 collection) with a new
collection on the *current* rig: same task, same control law, same rate, new
lights and **D415 in place of the D435** on camera 0. Camera 1 (blue Logitech,
`/blue/image_raw`) stays exactly what it is — it is the stream every checkpoint
and the deploy client read.

Everything below is derived from the collection stack that produced the old
data (`data/bridge_data_robot_pusht.zip`) and from the old data itself
(`data/03-23-pusht-data.zip` raw, `data/pusht_widowx_data.zip` converted), so
"match the old dataset" means matching numbers that were measured, not guessed.

---

## 0. The spec you are reproducing

Verified against the old archives. Do not drift from this or the new data will
not be comparable to the published curves.

| Property | Value | Where it comes from |
|---|---|---|
| Control / video rate | **20 Hz**, dt measured 0.0503 s | `TimedLoop` step = `env.move_duration` |
| Action | `[dx, dy]` planar EEF delta, metres | `action_mode: '2trans'` |
| Action range | **±0.008 m** exactly | `vr_xy_step_clip: 0.008` |
| Action deadband | 0.0015 m (→ exact zeros) | `vr_xy_step_deadband` |
| Z | locked at **0.02 m** | `FIXED_Z_HEIGHT`, `lock_z: True` |
| Yaw | fixed, `fix_zangle: 0.1` | env params |
| Gripper | fixed closed, `fixed_gripper: 0.0` | env params |
| Cameras | 2, index **0 = realsense**, **1 = blue** | `camera_topics` order |
| Image size saved | 640 × 480 JPEG per frame | `agent.image_height/width` |
| Episode cap | `T = 1000` steps | `agent.T` |
| Episode length seen | min 210 / max 985 / mean ≈ 487 | old `episode_ends` |
| Episodes / steps | **150 eps, 72 988 steps** | old dataset |
| Start pose | fixed, EEF ≈ (0.117, −0.018, 0.018) | old `full_state[0]`, consistent across trajs |
| Idle actions | 24 % of steps are exactly (0,0) | see `idle_filter` in `utils/datasets.py` |

The 24 % idle spike is a known pathology (`PUSHT_DEPLOY_HANDOFF.md`, the
`idle_filter` docstring): a static frame stack paired with a zero action is an
absorbing state the policy learns to sit in. **This collection is your chance to
not create it** — see §6.

---

## 1. Rig: mount the D415, set the lights, then freeze everything

Order matters: the camera-1 view is the one the policy consumes, so it is fixed
first and never touched again.

1. **Camera 1 (blue Logitech).** Put it where you want it for the rest of the
   project. If you want it to keep the old viewpoint, overlay live against the
   old reference:

   ```bash
   cd ~/Q3CIBC && conda activate q3c_deploy
   python scripts/align_pusht_camera.py     # overlays scripts/assets/pusht_widowx_cam1_ref.jpg
   ```

   Needs the robot server up (`widowx_env_service --server`). You are
   retraining from scratch, so matching the old view is **optional**. Matching
   it costs nothing and keeps the reference assets, the deploy crops and every
   parity script valid — recommended.

2. **Camera 0 (D415).** The D415 is *not* a drop-in optical replacement for the
   D435: its colour FOV is ~65° × 41° versus the D435's ~69° × 42°, and it is a
   rolling-shutter sensor with a longer minimum range (~0.45 m vs ~0.28 m).
   Practical consequences:
   - mount it **further back / higher** than the D435 sat, or the workspace
     will not fit in frame;
   - keep it ≥ 0.5 m from the table surface or the near field goes soft;
   - rolling shutter is fine at 20 Hz for a slow push, but do not shake the mount.

3. **Lights.** Set them now, at the intensity you will deploy with. Brightness
   is a real covariate: `scripts/check_brightness_parity.py` exists precisely
   because a lighting mismatch between train and deploy already bit this project
   once.

4. **Freeze the rig.** Tape the mounts, mark the table, do not touch camera
   pose, lens, exposure or lights again — not between collection sessions, not
   before deployment. Anything you move after this point invalidates the data.

5. **Record the new reference frames** once frozen (after §4 brings the streams
   up), and keep them under version control:

   ```bash
   # align_pusht_camera.py 's' key saves the composite; for plain stills use
   # any frame-grabbing client and save one per stream
   scripts/assets/pusht_widowx_cam1_ref_2026-07.jpg   # blue  (camera 1)
   scripts/assets/pusht_widowx_cam0_ref_2026-07.jpg   # D415  (camera 0)
   ```

---

## 2. `bridge_data_robot`: put the realsense back

The rig was stripped down to blue-only when the D435 was removed
(`PUSHT_DEPLOY_HANDOFF.md`). Those three edits must be reverted, with D415
names/serial substituted. All paths are on the Alienware, in
`~/bridge_data_robot`.

1. **Get the serial number** (D415 into a USB 3 port directly, no hub):

   ```bash
   rs-enumerate-devices -s
   rs-enumerate-devices | grep -i "usb type"    # want 3.1 / 3.2, not 2.1
   ```

   `lsusb -t` prints class/driver/speed, not vendor names — grepping it for
   "intel" finds nothing. Use `lsusb -d 8086:` (D415 = `8086:0ad3`) to get the
   bus/device, then read that line's speed in `lsusb -t`. The SDK's own
   "Usb Type Descriptor" line is the authoritative one.

2. **`widowx_envs/widowx_controller/launch/widowx_rs.launch`** — undo the
   no-realsense edits and rename the camera:

   ```xml
   <arg name="realsense"         default="true"/>
   <arg name="serial_no_camera1" default="<D415_SERIAL>"/>
   <arg name="camera1"           default="D415"/>
   ```

   The documented way to disable the realsense (LiraLab manual, p.2) is setting
   **`camera1` to `"blue"`** as well as `realsense` to false, so check L8 for a
   stale `"blue"` there. Also remove any `if="$(arg realsense)"` guard that was
   added around the `<group ns="$(arg camera1)">` block (or keep the guard and
   pass `realsense:=true` — either works, just be consistent).

   `camera1` is the ROS namespace: it is what makes the topics
   `/D415/color/image_raw` and `/D415/depth/image_rect_raw`.

3. **`widowx_envs/scripts/run.sh`** — back to `realsense:=true` on both L6
   (`camera_string`) and L17, and make sure `REALSENSE_SERIAL` is actually set
   (the script interpolates it):

   ```bash
   serial_no_camera1:=${REALSENSE_SERIAL} python_node:=false realsense:=true
   ```

   Export it in `docker-compose.yml` under `robonet.environment:`, e.g.
   `- REALSENSE_SERIAL=<D415_SERIAL>`, so it survives container restarts.

4. **`usb_connector_chart.yml`** — regenerate, then blank the realsense entry.

   The manual says to run `./generate_usb_config.sh` after connecting a new
   camera or changing a USB slot, and you should: it re-derives blue's usb path,
   which is the entry that actually matters. But the generator matches
   `v4l2-ctl --list-devices` on `"Intel(R) RealSense(TM) Depth Ca"` — **which
   your D415 also matches** — and writes it into the `D435:` key. A populated
   entry makes `multicam_server` open the RealSense as a plain UVC device while
   `realsense2_camera` is opening the same device by serial: two drivers, one
   camera.

   The pusht collection ran with that entry **empty** and took its realsense
   frames from `realsense2_camera` (proof: `conf_clam_pusht.py` subscribes to
   `/D435/depth/image_rect_raw`, a topic only `realsense2_camera` publishes).
   Do the same:

   ```bash
   ./generate_usb_config.sh
   ```
   ```yaml
   blue: 'usb-0000:00:14.0-1'   # whatever the generator found — must be non-empty
   yellow: ''
   wrist: ''
   D435: ''                     # blank it by hand even though the generator filled it
   ```

---

## 3. Collection config: `experiments/bridge_data_v2/conf_clam_pusht.py`

This is the file that defined the old collection. Three edits.

```python
env_params = {
    'camera_topics': [IMTopic('/D415/color/image_raw'),   # index 0  <- was /D435/...
                      IMTopic('/blue/image_raw')],        # index 1  <- MUST stay index 1
    'depth_camera_topics': [IMTopic('/D415/depth/image_rect_raw', dtype='16UC1')],
    ...
    'move_duration': 0.05,        # <- 20 Hz. The file in the archive says 0.08;
                                  #    the data it produced measures 0.0503 s/step,
                                  #    so 0.05 is what was actually run.
    ...
}
```

- **Camera order is load-bearing.** `utils/datasets.py` selects camera **1** as
  the blue scene view and discards camera 0. Swap the list and every downstream
  default silently trains on the wrong stream.
- **Depth is optional.** Nothing in Q3C/IBC/DP reads it, and it roughly doubles
  the raw footprint. Drop `depth_camera_topics` entirely unless you want it for
  something else later.
- Leave everything else alone: `action_mode`, the VR deadbands/clip, the z lock,
  `fixed_gripper`, `T`, `image_height/width`.
- `config['end_index']` is 500 — the loop just stops when you stop it, so it
  only needs to exceed your target episode count.

---

## 4. Bring the stack up

```bash
# Terminal 1 — container
cd ~/bridge_data_robot
USB_CONNECTOR_CHART=$(pwd)/usb_connector_chart.yml docker compose up --build robonet
```

Then check **both** streams are actually publishing before you record anything:

```bash
# Terminal 2
cd ~/bridge_data_robot
docker compose exec robonet bash -lic "rostopic list | grep -E 'blue|D415'"
docker compose exec robonet bash -lic "rostopic hz /blue/image_raw"
docker compose exec robonet bash -lic "rostopic hz /D415/color/image_raw"
```

Both must sit at ≥ 20 Hz with no gaps. A realsense that renegotiates to USB 2
will publish at a fraction of that and quietly starve the loop — if `rostopic
hz` shows ~6 Hz, it is the cable/port, not the config.

Grab the two reference stills from §1.5 now.

---

## 5. VR teleop hardware

Collection is VR-driven (`VRTeleopPolicy`, `oculus_reader`), not keyboard.
Procedure from the LiraLab manual, p.1:

1. Connect the **Meta Quest 2** with the USB-C → USB cable and **leave it
   plugged in for the whole session** (it charges while you teleoperate).
2. Turn the headset on, put it on, and click **"Allow USB"** on the in-headset
   notification. That is what authorises adb.
3. On the desktop:

   ```bash
   adb devices            # must say "device", not "unauthorized"
   ```

   `unauthorized` ⇒ repeat step 2.
4. If the headset was flat or restarted, launch the **RAIL Oculus
   Teleoperation** app in-headset and redraw the guardian boundaries, or the
   pose stream will not work.
5. Orient the headset **facing you**, and keep the controller inside the
   headset's field of view while driving — tracking is inside-out, so a
   controller behind the headset stops reporting.

- `oculus_reader` must import inside the container — `VR_WidowX.__init__`
  constructs `OculusReader()` at env creation, so a missing/blocked headset
  fails the run before the first episode.
- Right controller only. Controls:

  | Input | Effect |
  |---|---|
  | **Grip (RG / "handle")** | hold to drive the arm; released ⇒ zero action |
  | **B** | during an episode: ends it (`task_stage` 0→1 ⇒ `env_done`). At reset: sends the arm to neutral |
  | **A** | accept the finished trajectory |
  | **RJ** (right stick click) | discard the finished trajectory |

  The keyboard `y/n` prompt (`ask_traj_ok`) may also fire depending on which
  confirmation path is active — answer both the same way.

---

## 6. Run the collection

```bash
cd ~/bridge_data_robot
docker compose exec robonet bash          # you are now inside the container

python widowx_envs/widowx_envs/run_data_collection.py \
    widowx_envs/experiments/bridge_data_v2/conf_clam_pusht.py \
    --prefix pusht_2026-07
```

If it dies with `module 'numpy' has no attribute 'typeDict'`, run
`pip install --upgrade scipy` **inside the container** and retry — documented
failure mode, hits on first launch after a rebuild.

Data lands under `$DATA/robonetv2/bridge_data_v2/pusht_2026-07/<timestamp>/raw/traj_group0/trajN/`
which is `~/widowx_data/...` on the host (bind-mounted in `docker-compose.yml`).

Per-episode loop:

1. The arm resets to neutral, then to the fixed start pose, and prints
   *"waiting for handle button press to start recording"*.
2. Place the T block and the target outline. **Vary the T's start pose across
   episodes** — that variety is the only source of state coverage you have.
3. Press and **hold grip**; recording starts.
4. Push the T onto the target.
5. Press **B** the moment the T is on target. Do not idle after success.
6. **A** to keep, **RJ** to discard. Discard anything with a collision, a
   dropped frame burst, or a long stall.

### Guardrails while recording

- **Do not release the grip mid-episode.** A released grip emits the zero
  action every step, and that is where most of the old 24 % idle spike came
  from. Keep it held for the whole episode, and end the episode instead of
  pausing.
- **Keep moving above the deadband.** Motions under 0.0015 m/step are clipped
  to exactly zero, which feeds the same absorbing state. Smooth, continuous
  pushes; no micro-corrections held for seconds.
- **Watch for `Warning, loop takes too long`.** That message means the step
  exceeded 0.0525 s and your effective rate has dropped below 20 Hz. If it is
  frequent, something (usually camera bandwidth) is stealing the budget — stop,
  fix it, and discard the affected episodes.
- **Aim for ~490 steps (≈ 25 s) per episode**, matching the old distribution.
  The hard cap is 1000 steps (50 s); an episode that hits it was too slow.
- **Target 150 episodes / ≈ 73 000 steps.** Fewer than ~120 and you are not
  comparable to the published runs. Budget ~15–20 GB of raw JPEGs for 150
  episodes at two cameras (the old 110-episode set was 9.2 GB zipped).
- Collect in sittings, but **never move the rig between them**. Note the wall
  clock of each sitting; if the room light changes through the day, that is a
  covariate you want recorded.

### Shutting down a sitting

Order matters and there is a physical hazard here:

1. Kill the collection process (window 2) first.
2. **Hold the arm by hand**, then kill the container (window 1). Losing the ROS
   node releases the servos and the arm drops onto the table.
3. Return the arm to its resting position and power it off. Do not leave it
   powered between sittings.
4. Charge the Quest.

---

## 7. Verify the raw collection before converting

Run these on the host against `.../raw/traj_group0/`:

```python
import pickle, glob, numpy as np
for d in sorted(glob.glob('raw/traj_group0/traj*')):
    obs = pickle.load(open(f'{d}/obs_dict.pkl','rb'))
    po  = pickle.load(open(f'{d}/policy_out.pkl','rb'))
    a   = np.array([p['actions'] for p in po])          # (T-1, 7)
    dt  = np.diff(np.array(obs['time_stamp']))
    print(d, len(obs['time_stamp']), a.shape,
          'dt=%.4f' % dt.mean(),
          '|a|max=%.4f' % np.abs(a[:, :2]).max(),
          'idle=%.2f' % (np.abs(a[:, :2]).max(1) == 0).mean(),
          'stage', sorted(set(obs['task_stage'])))
```

Accept the collection only if, per episode:

- `dt.mean()` ≈ 0.050 (tolerate ≤ 0.052),
- `|a|max` = 0.008 (the clip is being hit ⇒ teleop scale is right),
- `idle` well **under** 0.24 — that is the number you are trying to beat,
- `task_stage` reaches 1 (B was pressed) and frame counts in `images0/` and
  `images1/` both equal `len(obs['time_stamp'])`,
- open a few `images1/im_*.jpg` and confirm the T renders **red** (they are
  written BGR-swapped by `RawSaver.save_single`, so the file on disk is honest).

`scripts/check_frame_count_parity.py` and `scripts/check_brightness_parity.py`
cover the last two mechanically.

---

## 8. Convert raw → Diffusion-Policy zarr + MP4

The trainer reads `data_format: zarr_video`, i.e. the layout documented in
`pusht_real/README.md` inside the old zip. **No converter for this exists in the
repo** — the old zip arrived already converted. It has to be written; the field
mapping below is reverse-engineered from the two archives and is exact:

| zarr field | shape | source |
|---|---|---|
| `data/action` | (N, 7) f8 | `policy_out[t]['actions']` (already 7-D via `actions_save`); dims 2:7 are 0 |
| `data/robot_eef_pose` | (N, 6) f8 | `obs['full_state'][:, :3]` ⊕ `Rotation.from_matrix(obs['eef_transform'][:, :3, :3]).as_rotvec()` |
| `data/robot_eef_pose_vel` | (N, 6) f8 | forward difference: `(pose[t+1] − pose[t]) / 0.05`, last row repeated/zero |
| `data/robot_joint` | (N, 6) f8 | `obs['qpos']` |
| `data/robot_joint_vel` | (N, 6) f8 | `obs['qvel']` |
| `data/stage` | (N,) i8 | `obs['task_stage']` |
| `data/timestamp` | (N,) f8 | regenerated on an exact 0.05 s grid from the episode start (the old file's diffs are exactly 0.05, not the measured raw times) |
| `meta/episode_ends` | (E,) i8 | cumulative, exclusive |
| `videos/<ep>/0.mp4` | — | `images0/im_*.jpg` → H.264, 640×480, 20 fps |
| `videos/<ep>/1.mp4` | — | `images1/im_*.jpg` → H.264, 640×480, 20 fps |

Two details that will bite:

- **Length**: raw has `T` observations and `T−1` actions. Truncate observations
  to `T−1` so rows, actions and video frames all line up (the old dataset's
  video frame count equals its row count exactly).
- **Frame `t` ↔ row `t` within the episode's slice** — the zarr is globally
  concatenated, the videos are per-episode.

Write the result to `data/pusht_widowx_data_v2.zip` with the same internal
`pusht_real/` prefix, then point training at it (`--dataset`, or
`PUSHT_DATASET=` for `scripts/submit_pusht_real_ibc.sh`).

---

## 9. Then: train

```bash
python scripts/prepare_pusht_video_cache.py --dataset data/pusht_widowx_data_v2.zip
PUSHT_DATASET=$PWD/data/pusht_widowx_data_v2.zip bash scripts/submit_pusht_real_ibc.sh
```

Note that `scripts/train_pusht_real_ibc.py` hardcodes `n_cams = 1` for
`zarr_video` (blue only). `PushTWidowXVideoDataset` already accepts a `cameras`
list and lays channels out interleaved per stack offset, so training on D415 +
blue together needs a small plumbing change in the train script, not new dataset
code.

Re-derive `norm_stats` from the new data — do **not** reuse the old ones. The
action clip is the same ±0.008, but the observation statistics are not.
