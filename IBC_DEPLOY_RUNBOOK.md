# IBC Push-T — Slurm → Alienware deploy runbook

Getting the IBC (EBM + DFO) checkpoints off discovery and running them on the
WidowX. Companion to `PUSHT_DEPLOY_HANDOFF.md`, which covers the rig itself
(camera wiring, `bridge_data_robot` edits, colour/channel findings).

Split to keep in mind: **code travels by git, checkpoints travel by scp.**
`checkpoints/` is gitignored, so `git pull` on the Alienware brings the deploy
script and `utils/ibc_policy.py` but never the weights.

---

## 1. Connect from the Alienware to discovery

Plain `ssh halbertb@discovery.usc.edu` from this machine picks an automatic
method (an offered public key, or GSSAPI/Kerberos) and never gets round to
prompting for the password. Force the interactive methods and turn the
automatic ones off:

```bash
ssh -o PreferredAuthentications=keyboard-interactive,password \
    -o PubkeyAuthentication=no \
    -o GSSAPIAuthentication=no \
    halbertb@discovery.usc.edu
```

If you want to see what it is choosing on its own, `ssh -v` prints each method
as it is attempted — the useful lines are `Authentications that can continue:`
and `Next authentication method: …`.

### Make it permanent, and authenticate only once

Rather than repeating those flags on every `ssh`, `scp` and `rsync`, put them
in `~/.ssh/config` on the Alienware:

```
Host discovery
    HostName discovery.usc.edu
    User halbertb
    PreferredAuthentications keyboard-interactive,password
    PubkeyAuthentication no
    GSSAPIAuthentication no

    # Reuse one authenticated connection for every later ssh/scp/rsync to this
    # host: you type the password (and clear any 2FA prompt) once, and the
    # transfers below ride the same channel for 8 hours.
    ControlMaster auto
    ControlPath ~/.ssh/cm-%r@%h:%p
    ControlPersist 8h

    # Long rsyncs otherwise get dropped by an idle timeout.
    ServerAliveInterval 60
    ServerAliveCountMax 10
```

```bash
chmod 600 ~/.ssh/config          # ssh refuses a group/world-writable config
ssh discovery                    # authenticate once; leave this session open
```

The `ControlMaster` block is the part that matters most here. The rest of this
runbook runs several commands against discovery (a listing, an rsync, an
md5sum), and without multiplexing each one re-authenticates from scratch.

To drop the shared connection early:

```bash
ssh -O exit discovery
```

With that alias in place, `discovery` stands in for
`halbertb@discovery.usc.edu` everywhere below.

### Optional: key-based login

If the cluster permits public keys, this removes the password prompt entirely
(2FA, where enforced, may still apply on first connect):

```bash
ssh-keygen -t ed25519 -C "alienware-q3c"     # only if ~/.ssh/id_ed25519 is absent
ssh-copy-id -o PubkeyAuthentication=no halbertb@discovery.usc.edu
```

Then drop `PubkeyAuthentication no` and the `PreferredAuthentications` line
from the config block above, or the key you just installed will be ignored.

## 2. On discovery: confirm what finished

```bash
ssh discovery
cd ~/Q3CIBC
ls -la checkpoints/pusht_real_ibc/*/
```

Each seed directory holds:

| File | Size | Needed to deploy |
|---|---|---|
| `config.json` | ~1.4 KB | yes — model + DFO settings |
| `norm_stats.pt` | ~2 KB | yes — act_min/act_max, frame stack, cameras |
| `q_estimator.pt` | ~8.8 MB | yes — final weights |
| `q_estimator_step*.pt` | ~8.8 MB each | no — 10k-step snapshots |
| `last_step.json` | 12 B | no |

A finished 100k-step seed has `q_estimator.pt`. If it is missing, the run is
still going (or died) and only snapshots exist — the deploy script and the
diagnostic both fall back to the newest snapshot and say so, so a partial seed
is still usable for a smoke test.

Check progress without waiting for the job:

```bash
tail -3 slurm_jobs/pusht_ibc/pusht_ibc_*.out
squeue -u halbertb
```

## 3. Copy the checkpoints to the Alienware

Only three files per seed, ~9 MB each — skip the snapshots. Run **from the
Alienware** (it can reach discovery; the reverse usually cannot), reusing the
connection from step 1 so this does not re-prompt:

```bash
cd ~/Q3CIBC
mkdir -p checkpoints/pusht_real_ibc

rsync -avP \
  -e 'ssh -o PreferredAuthentications=keyboard-interactive,password -o PubkeyAuthentication=no -o GSSAPIAuthentication=no' \
  --include='seed_*/' \
  --include='seed_*/config.json' \
  --include='seed_*/norm_stats.pt' \
  --include='seed_*/q_estimator.pt' \
  --exclude='*' \
  halbertb@discovery.usc.edu:Q3CIBC/checkpoints/pusht_real_ibc/ \
  checkpoints/pusht_real_ibc/

```

If you skipped the `~/.ssh/config` alias in step 1, `discovery:` will not
resolve and nothing is forcing the password prompt. Pass the ssh options
through with `-e` and spell out the host instead:

```bash
rsync -avP \
  -e 'ssh -o PreferredAuthentications=keyboard-interactive,password -o PubkeyAuthentication=no -o GSSAPIAuthentication=no' \
  --include='seed_*/' \
  --include='seed_*/config.json' \
  --include='seed_*/norm_stats.pt' \
  --include='seed_*/q_estimator.pt' \
  --exclude='*' \
  halbertb@discovery.usc.edu:Q3CIBC/checkpoints/pusht_real_ibc/ \
  checkpoints/pusht_real_ibc/
```

That prompts once for this transfer only — every later `ssh`/`scp` needs the
same treatment, which is the argument for the config block.

Single seed, if you prefer scp:

```bash
mkdir -p checkpoints/pusht_real_ibc/seed_0029
scp 'discovery:Q3CIBC/checkpoints/pusht_real_ibc/seed_0029/{config.json,norm_stats.pt,q_estimator.pt}' \
    checkpoints/pusht_real_ibc/seed_0029/
```

Verify the transfer landed intact — a truncated `q_estimator.pt` fails late,
inside `load_state_dict`, with a confusing error:

```bash
ls -la checkpoints/pusht_real_ibc/*/
# compare against discovery:
ssh discovery 'md5sum ~/Q3CIBC/checkpoints/pusht_real_ibc/seed_0029/q_estimator.pt'
md5sum checkpoints/pusht_real_ibc/seed_0029/q_estimator.pt
```

## 4. On the Alienware: update the code

```bash
cd ~/Q3CIBC
git pull
conda activate q3c_deploy
```

The pull must bring both `scripts/deploy_pusht_real_ibc.py` and
`utils/ibc_policy.py`. The latter is new — the deploy client and the offline
diagnostic share it so their DFO cannot drift apart.

### One new dependency

The rewritten client uses absl flags (the previous Push-T clients used
`argparse`), so the `q3c_deploy` env needs `absl-py`, which it has never had:

```bash
pip install absl-py
```

`absl-py` is pure Python and does not depend on numpy, but re-check the pin
afterwards anyway — **numpy must stay below 2** or `step_action` payloads fail
to unpickle server-side as `numpy._core`:

```bash
python -c "import numpy, cv2, torch, absl; print(numpy.__version__, cv2.__version__, torch.__version__)"
# numpy must be 1.x, opencv 4.10.0.84
```

Video recording is optional and needs an extra backend. Skip it unless you
want rollout videos; the script warns and continues without it:

```bash
pip install 'imageio[ffmpeg]'     # only if you pass --video_save_path
```

Confirm the policy side loads before touching the robot:

```bash
python -c "
import sys; sys.path.insert(0, '.')
import torch; from pathlib import Path
from utils import ibc_policy
p = ibc_policy.load_policy(Path('checkpoints/pusht_real_ibc/seed_0029'), torch.device('cpu'))
print(p.name, p.camera_streams, p.frame_stack, p.act_min, p.act_max, p.dfo)
"
```

## 5. Bring the robot up

Per `PUSHT_DEPLOY_HANDOFF.md`: docker up robonet (blue camera only), then the
action server. The client's TCP preflight will warn if the server is not
listening on `localhost:5556`.

## 6. Dry run first — always

No motion. Captures what the model actually sees and what it would command:

```bash
python scripts/deploy_pusht_real_ibc.py \
    --seed_dir checkpoints/pusht_real_ibc/seed_0029 \
    --device cpu \
    --dry_run --dry_run_steps 40
```

Then check, in order:

1. `deploy_dryrun_ibc/fed_000.png` — **the T must render RED.** (Opening it in
   an `imshow` window instead shows it blue, because imshow assumes BGR; the
   PNG on disk is the honest one.)
2. Printed actions vary across steps and are not pinned to one quadrant. A
   fixed `(-,-)` diagonal is the failure signature from the v1 deploy.
3. `dx, dy` magnitudes sit inside ±0.008 m — that is the full trained range.

If the frames look geometrically wrong (flipped/rotated versus training),
`scripts/check_preproc_parity.py` sweeps transforms over the captured
`raw_*.npy` before you commit to a live run.

Worth running the offline diagnostic on the same checkpoint too, since it
answers a question the dry run cannot — whether the energy surface learned
anything at all:

```bash
python scripts/diagnose_pusht_actions_ibc.py \
    --output-root checkpoints/pusht_real_ibc --seeds 29 47 --device cpu \
    --num-samples 500
```

`energy entropy ≈ 1.0` or `expert percentile ≈ 0.5` means the EBM is
effectively random and no amount of deploy tuning will help. Note this needs
the training zip and TensorFlow for JPEG decode, so it is usually easier to
run on discovery via `sbatch scripts/diagnose_pusht_actions_ibc.sbatch`.

## 7. Live rollout

Hand on the E-stop. Stop any time with `s`.

```bash
python scripts/deploy_pusht_real_ibc.py \
    --seed_dir checkpoints/pusht_real_ibc/seed_0029 \
    --device cpu \
    --step_duration 0.05 \
    --max_duration 120 \
    --num_rollouts 3 \
    --widowx_force_fresh_init \
    --widowx_init_timeout_ms 180000 --widowx_init_retries 8 \
    --forensic_log_dir deploy_logs/ibc_seed0029
```

**Use `--widowx_force_fresh_init` on the first launch after starting the
server.** By default the client reuses a live server-side env and skips
`init()`, which means the env params it wants — `action_mode=2trans` above
all — are never applied and the server keeps whatever it was started with.
In a mode whose last action element is the gripper, a 2-element `[dx, dy]`
command is read as a gripper command: **the claw actuates and the arm never
translates.** The client prints a warning whenever it takes the reuse path.

The script waits for `[Enter]` before each rollout, resets with the rollout
index, moves the EEF to the demo start pose, then runs closed loop. A rollout
ends on `s`, on `--max_duration`, or by dwelling in the termination area
(auto-aligned to the post-reset pose).

Flags worth knowing:

| Flag | Why |
|---|---|
| `--device cpu` | The deploy env has CPU torch. DFO is ~60 ms/step there, fine at 20 Hz. |
| `--dfo_samples`, `--dfo_iterations` | Override the trained 2048×3 to trade latency for search quality. |
| `--nomove_to_demo_start` | Skips the start-pose move. Not recommended — the arm then begins ~17 cm out of distribution. |
| `--norequire_fresh` | Disables the duplicate-frame guard. Only for debugging; a repeated frame gives the stack zero inter-frame motion. |
| `--safety_max_xy_delta` | Backstop clip, default 0.02 m against a trained range of ±0.008. |
| `--robot_exec_hz` | Leave at 0 (one command per inference). Higher splits each delta across substeps rather than repeating it. |
| `--policy_seed` | DFO is stochastic; fix this to make a rollout reproducible. |
| `--widowx_force_fresh_init` | Applies the env params (notably `action_mode`) instead of reusing the server's existing env. |
| `--disable_term_area` | Ends rollouts only on `s` or `--max_duration`, never on returning to the start pose. |
| `--policy` | `auto` (default), `ibc` or `q3c`. See below. |

### Running Q3C through the same client

The client drives both policy families; **only action selection differs**, so
the observation pipeline, robot loop, duplicate-frame guard, safety clip and
every stopping condition are identical. That is what makes a head-to-head
comparison on this rig meaningful — any difference in behaviour is the policy,
not the harness.

Point it at a Q3C seed directory and it detects the family from the files
present (a `control_point_generator*.pt` means Q3C):

```bash
python scripts/deploy_pusht_real_ibc.py \
    --seed_dir checkpoints/pusht_real_combinedv2_v2/seed_0029 \
    --device cpu \
    --step_duration 0.05 \
    --widowx_force_fresh_init \
    --forensic_log_dir deploy_logs/q3c_seed0029
```

Pass `--policy ibc` or `--policy q3c` to assert it explicitly. Q3C-only flags:
`--no_ema` (raw instead of the default EMA weights), `--cp_selection
argmax|sample`, and `--cp_dfo_iterations` (the deployed seeds use 0, i.e. pure
CP-cloud argmax). Flags belonging to the other family are ignored with a
warning rather than silently.

The startup banner reports the cost difference directly, and it is large — on
CPU, Q3C ranks ~20 control points where IBC scores 2048 samples three times
over:

```
Loaded Q3C policy: ...   Q3C CP-cloud: 20 control points, selection=argmax
[NFE] Q3C: 1 scoring pass x 20 candidates = 20 value-head evaluations
Loaded IBC policy: ...   IBC EBM + DFO: 2048 samples x 3 iters, ...
[NFE] IBC EBM + DFO: 3 iterations x 2048 action samples = 6144 value-head evaluations
```
| `--video_save_path` | Needs `imageio[ffmpeg]`. Writes cam0/cam1 mp4s plus a timing JSON. |

## 8. What to keep from a session

`--forensic_log_dir` writes per rollout: `raw/*.npy` (server frames as
received), `fed/*.png` (what the model saw), and `steps.jsonl` with the
normalized action, the metric action, inference latency and the EEF state per
step. That is the input to the same offline analysis as
`deployment_forensics.md`, so record it on any run you might want to explain
later.

Latency statistics print at the end of each rollout and land in the timing
JSON alongside the videos when `--video_save_path` is set.

---

## Troubleshooting

| Symptom | Cause |
|---|---|
| ssh never prompts for a password | A key or GSSAPI is being tried automatically — force the methods as in step 1 |
| `Too many authentication failures` | Several keys offered before the password; `PubkeyAuthentication no` or `IdentitiesOnly yes` stops it |
| `Bad owner or permissions on ~/.ssh/config` | `chmod 600 ~/.ssh/config` |
| ssh config edits appear to do nothing | An earlier `Host` block matched first — ssh takes the *first* value for each option, so put `Host discovery` above any `Host *` |
| Multiplexed session refuses to reconnect | Stale socket: `ssh -O exit discovery`, or delete `~/.ssh/cm-*` |
| Only the gripper opens/closes, the arm never translates | The server env is not in `action_mode=2trans`, so the 2-element `[dx, dy]` is read as a gripper command. Relaunch with `--widowx_force_fresh_init` |
| Rollout ends after a fraction of a second | Older builds armed the return-to-start stop while the arm was still at the start pose. Fixed; `--disable_term_area` rules it out entirely |
| `ModuleNotFoundError: absl` | `pip install absl-py` (step 4) |
| `ModuleNotFoundError: utils` | Run from the repo root, or the pull did not bring `utils/ibc_policy.py` |
| `no q_estimator.pt and no q_estimator_step*.pt` | Only `config.json`/`norm_stats.pt` copied — re-run the rsync |
| Warning about using a snapshot | That seed has not finished training; expected mid-run |
| `numpy._core` unpickle error server-side | numpy 2.x crept into `q3c_deploy`; pin back below 2 |
| Init times out | Server not up, or needs a longer `--widowx_init_timeout_ms` |
| `[WARN] stale frame` repeatedly | Camera stream stalled; check the blue camera before trusting the rollout |
| T renders blue in the dumped PNG | Channel order is off for this rig — try `--swap_rgb` |
