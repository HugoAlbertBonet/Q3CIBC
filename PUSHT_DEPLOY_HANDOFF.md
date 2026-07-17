# Push-T Real-Robot Deploy — Handoff

Deploying the Q3CIBC combinedv2 Push-T policy (seeds 11/29/47/83) on the LiraLab
WidowX via the bridge_data_robot server/client architecture.

## Status: pipeline works end-to-end; policy runs away (-,-) on the robot.

Diagnostic (`scripts/diagnose_pusht_actions.py`) proved all 4 checkpoints are
**healthy offline**: pred std ~0.5 (matches GT), corr(dx,dy)~-0.1 (not diagonal),
MAE~0.025 in [-1,1], means match GT, GT leans `++`. So the robot's straight
`(-,-)` diagonal is a **deploy-only** obs→action mismatch, not a bad checkpoint.

Prime suspect: deploy observation is geometrically transformed vs training
(flip / rotation / view-scale). Channel order already verified matching.

## Setup facts (hard-won)
- **Single camera**: D435 removed from rig. Policy trained on `images1` = `/blue/image_raw`
  = Logitech "1080P Pro Stream". Server runs one camera → blue is `full_image[0]`
  → arrives as `external_img` (not `over_shoulder_img`). Client auto-picks.
- **bridge_data_robot** (~/bridge_data_robot on Alienware) needed edits for no-D435:
  - `widowx_rs.launch` L6 `realsense` default `false`; L23 group guarded `if="$(arg realsense)"`.
  - `run.sh` L17 `realsense:=false`.
  - `usb_connector_chart.yml` = only `blue: '<usb-id from v4l2-ctl --list-devices>'`.
- **Client env** (`q3c_deploy` conda on Alienware): CPU torch + GUI opencv, plus
  `pip install -e ~/bridge_data_robot/widowx_envs`, edgeml, funcsigs pyyaml rospkg
  netifaces defusedxml catkin_pkg. Must be **numpy<2** + **opencv-python==4.10.0.84**
  (server numpy is 1.x; numpy 2.x arrays fail to unpickle server-side as numpy._core).
- **Color**: server frame already has red in ch0 (matches tf-decoded training JPEGs).
  Deploy default = NO swap. `imshow` shows the T blue = correct (imshow assumes BGR).
- **step_action**: send numpy array (env needs .shape); needs numpy<2 for pickle.
- **reset() after init** is required or the first step crashes the server.
- Action = 2D planar EEF delta (x,y) metres; act range ±0.008. Step cap 985 (longest ep).

## Scripts (in Q3CIBC repo)
- `scripts/deploy_pusht_real.py` — the client. `--dry-run` dumps `deploy_dryrun/fed_*.png`
  (preprocessed) + `raw_*.npy` (raw blue frame). `--swap-rgb`, `--obs-key`, `--hz`, `--steps`.
- `scripts/align_pusht_camera.py` — live overlay vs `scripts/assets/pusht_images1_ref.jpg`.
- `scripts/diagnose_pusht_actions.py` (+ `.sbatch`) — offline collapse check, all seeds.
- `scripts/check_preproc_parity.py` — sweeps image transforms (flips/rot/channel) on
  captured `raw_*.npy`, reports action+quadrant per transform. **Run this first tomorrow.**

## Tomorrow
1. `cd ~/Q3CIBC && git pull` on Alienware (get raw-dump + parity tool).
2. Bring robot up (docker up robonet blue-only, server, `conda activate q3c_deploy`).
3. `python scripts/deploy_pusht_real.py --seed-dir checkpoints/pusht_real_combinedv2/seed_0011 --device cpu --dry-run --dry-run-steps 40` to capture `raw_*.npy` (also re-confirm red T).
4. `python scripts/check_preproc_parity.py --seed-dir checkpoints/pusht_real_combinedv2/seed_0011`
   → find the transform whose mean/quadrants match the offline `++` reference.
5. If a transform (e.g. flip_vert) fixes it → add that geometric correction to
   `deploy_pusht_real.preprocess`, then short capped rollout with `--steps 20` + lower `--hz`.
6. If NO transform fixes it → the live camera view differs from training (physical
   pose/scale); re-check alignment against the reference frame, or it's covariate
   shift (then add per-step delta clip + reset-and-retry, or accept BC limits).
