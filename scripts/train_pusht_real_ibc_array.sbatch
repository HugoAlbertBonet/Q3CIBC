#!/usr/bin/env bash
#SBATCH --account=biyik_1165
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --constraint="a100|l40s|a40|v100"
#SBATCH --mem=48G
#SBATCH --time=48:00:00
#SBATCH --array=0-3
#SBATCH --job-name=pusht-ibc
#SBATCH --output=pusht-ibc_%A_%a.out
#SBATCH --error=pusht-ibc_%A_%a.err

set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
# Slurm executes a spooled copy of this script under /var/spool/slurm.  The
# submit directory, rather than BASH_SOURCE, is therefore the project root.
ROOT_DIR=${SLURM_SUBMIT_DIR:-$(cd -- "${SCRIPT_DIR}/.." && pwd)}
cd "${ROOT_DIR}"

SEEDS=(11 29 47 83)
SEED=${SEEDS[$SLURM_ARRAY_TASK_ID]}
DATASET=${PUSHT_DATASET:-"${ROOT_DIR}/data/pusht_widowx_data.zip"}
# Decoded-frame memmap shared by all array tasks. Build it BEFORE submitting
# (scripts/prepare_pusht_video_cache.py) or N-1 tasks sit on GPUs waiting for
# whichever one wins the build lock.
FRAME_CACHE=${PUSHT_FRAME_CACHE:-""}
IDLE_FILTER=${PUSHT_IDLE_FILTER:-"drop_zero"}
OUTPUT_ROOT=${PUSHT_OUTPUT_ROOT:-"${ROOT_DIR}/checkpoints/pusht_real_ibc"}
PYTHON=${PYTHON:-"${ROOT_DIR}/.venv/bin/python"}
LOG_DIR=${PUSHT_LOG_DIR:-"${ROOT_DIR}/slurm_jobs/pusht_ibc"}
LOG_FILE="${LOG_DIR}/pusht_ibc_$(printf '%03d' "$((SLURM_ARRAY_TASK_ID + 1))").out"

mkdir -p "${LOG_DIR}"
exec >"${LOG_FILE}" 2>&1

if [[ ! -f "${DATASET}" ]]; then
    echo "Dataset not found on the compute node: ${DATASET}" >&2
    exit 2
fi
if [[ ! -x "${PYTHON}" ]]; then
    echo "Project Python not found/executable: ${PYTHON}" >&2
    exit 2
fi

# Never let parallel array tasks mutate the shared environment or lockfile.
export UV_NO_SYNC=1
export UV_FROZEN=1
export PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}

echo "job=${SLURM_JOB_ID} task=${SLURM_ARRAY_TASK_ID} seed=${SEED} node=${SLURM_JOB_NODELIST}"
echo "dataset=${DATASET}  idle_filter=${IDLE_FILTER}  frame_cache=${FRAME_CACHE:-<default>}"
echo "output=${OUTPUT_ROOT}/seed_$(printf '%04d' "${SEED}")"

# IBC best-gin recipe (pixel_ebm_best.gin): 100k steps, batch 128, 256 uniform
# counter-examples — all defaults of train_pusht_real_ibc.py; only the
# machine-specific knobs are passed here.
srun "${PYTHON}" scripts/train_pusht_real_ibc.py \
    --dataset "${DATASET}" \
    --data-format zarr_video \
    --seed "${SEED}" \
    --workers "${SLURM_CPUS_PER_TASK}" \
    --frame-stack 2 \
    --video-camera 1 \
    --idle-filter "${IDLE_FILTER}" \
    ${FRAME_CACHE:+--frame-cache-dir "${FRAME_CACHE}"} \
    --output-root "${OUTPUT_ROOT}"
