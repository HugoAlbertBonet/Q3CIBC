#!/usr/bin/env bash
# One-time LIBERO-Goal setup — RUN THIS ON THE SLURM SERVER (discovery), from the
# repo root. It creates the gitignored, machine-local pieces that don't sync via
# git: the editable LIBERO clone, the uv env, the demos, and the goal embeddings.
#
# Prereqs on the server:
#   - You have pulled the latest repo (must include the libero entries in
#     pyproject.toml + uv.lock, utils/libero.py, scripts/, simulations/, config).
#   - `uv` and `git` are available.
#   - Network access (login node is fine for clone + download).
#
# Steps that NEED A GPU COMPUTE NODE (mujoco offscreen render) are NOT done here;
# this script prints them at the end. Run those under srun with MUJOCO_GL=egl.
#
# Idempotent: safe to re-run; it skips parts already done.

set -euo pipefail
cd "$(dirname "$0")/.."   # repo root
REPO="$PWD"
echo "repo: $REPO"

# 1. Clone LIBERO into third_party/ (in-repo, gitignored) + add the package
#    marker that upstream is missing (else the editable build ships no package).
if [[ ! -f third_party/LIBERO/setup.py ]]; then
  echo "[1/4] cloning LIBERO into third_party/LIBERO ..."
  mkdir -p third_party
  git clone https://github.com/Lifelong-Robot-Learning/LIBERO.git third_party/LIBERO
else
  echo "[1/4] third_party/LIBERO already present."
fi
touch third_party/LIBERO/libero/__init__.py   # CRITICAL: upstream omits this

# 2. Build the uv env with the libero extra. pyproject pins LIBERO's runtime
#    deps (robosuite==1.4.0, bddl==1.0.1, easydict, gym, numba, opencv-headless,
#    termcolor, pynput, ...) and references the editable clone via
#    [tool.uv.sources]. Do NOT use `uv pip install` for these — `uv run`
#    auto-sync would prune them.
echo "[2/4] uv sync --extra libero ..."
uv sync --extra libero

# Sanity: libero benchmark importable (this works on a login node). Do NOT
# import robosuite here — it initializes an EGL/GL context at import and fails
# on login nodes without a GPU; robosuite is only exercised later on a compute
# node (extract/probe/eval) with MUJOCO_GL set.
uv run --extra libero python -c "import libero; from libero.libero import benchmark; benchmark.get_benchmark_dict(); print('libero OK:', libero.__file__)"

# 3. Download the libero_goal demos (~6 GB) if missing. LIBERO's downloader
#    prompts twice interactively; we feed 'y'. utils/libero forces the config to
#    <repo>/.libero so paths stay in-repo and node-mount-agnostic.
DEMO_DIR="third_party/LIBERO/libero/datasets/libero_goal"
if ! ls "$DEMO_DIR"/*.hdf5 >/dev/null 2>&1; then
  echo "[3/4] downloading libero_goal demos ..."
  printf 'y\ny\n' | uv run --extra libero python \
    third_party/LIBERO/benchmark_scripts/download_libero_datasets.py --datasets libero_goal
else
  echo "[3/4] demos already present ($(ls "$DEMO_DIR"/*.hdf5 | wc -l) files)."
fi

# 4. Precompute goal-language embeddings (CPU; no render needed).
EMB=datasets/libero/libero_goal_goal_embs.npz
if [[ ! -f "$EMB" ]]; then
  echo "[4/4] precomputing goal embeddings ..."
  uv run --extra libero python scripts/precompute_libero_goal_embs.py --out "$EMB" --encoder minilm
else
  echo "[4/4] embeddings already present: $EMB"
fi

cat <<'NEXT'

=============================================================================
Login-node setup done. REMAINING STEPS NEED A GPU COMPUTE NODE (mujoco render):

  srun --gres=gpu:1 --pty bash
  cd ~/Q3CIBC && export MUJOCO_GL=egl     # use osmesa if egl fails

  # a) Solve the object-blind caveat (adds obs/object-state to the demos):
  uv run --extra libero python scripts/extract_libero_object_states.py
  #    -> note the printed state_dim; set it in
  #       config_json/config.json environments.libero_goal.state_dim

  # b) Confirm live obs keys match the demos:
  uv run --extra libero python scripts/probe_libero_live_obs.py

  # c) Smoke-test one short trial end-to-end (train + live eval):
  uv run --extra libero python hyperparam_search.py combinedv2_cpascounter_training.py \
      --run --active-env libero_goal \
      --fixed-params '{"trial_seed":0,"training_steps":2000,"control_points":20,"top_k_control_points":8}'

Then submit the batch (job template already exports MUJOCO_GL=egl):
  ./submit_experiments.sh batches/liberoGoalA.txt
=============================================================================
NEXT
