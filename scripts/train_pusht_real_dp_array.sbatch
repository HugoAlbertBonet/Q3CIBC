#!/usr/bin/env bash
#SBATCH --account=biyik_1165
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --constraint="a100|l40s|a40"
#SBATCH --mem=48G
#SBATCH --time=48:00:00
#SBATCH --array=0-3
#SBATCH --job-name=pusht-dp
#SBATCH --output=pusht-dp_%A_%a.out
#SBATCH --error=pusht-dp_%A_%a.err

set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
# Slurm executes a spooled copy of this script under /var/spool/slurm.  The
# submit directory, rather than BASH_SOURCE, is therefore the project root.
ROOT_DIR=${SLURM_SUBMIT_DIR:-$(cd -- "${SCRIPT_DIR}/.." && pwd)}
cd "${ROOT_DIR}"

# 2 seeds x 2 prediction types. DDPM and DDIM are inference-time schedules over
# the SAME denoiser, so they are not separate trainings — each of these four
# checkpoints is evaluated under both samplers at deploy time.
SEEDS=(11 29 11 29)
PRED_TYPES=(epsilon epsilon v v)
SEED=${SEEDS[$SLURM_ARRAY_TASK_ID]}
PRED=${PRED_TYPES[$SLURM_ARRAY_TASK_ID]}

DATASET=${PUSHT_DATASET:-"${ROOT_DIR}/data/pusht_widowx_data.zip"}
OUTPUT_ROOT=${PUSHT_OUTPUT_ROOT:-"${ROOT_DIR}/checkpoints/pusht_real_dp"}
PYTHON=${PYTHON:-"${ROOT_DIR}/.venv/bin/python"}
LOG_DIR=${PUSHT_LOG_DIR:-"${ROOT_DIR}/slurm_jobs/pusht_dp"}
LOG_FILE="${LOG_DIR}/pusht_dp_${PRED}_$(printf '%03d' "$((SLURM_ARRAY_TASK_ID + 1))").out"

# The zarr_video archive decodes its MP4s into a ~17 GB uint8 memmap on first
# use. Build it up front (scripts/prepare_pusht_video_cache.sbatch) or the four
# tasks hold GPUs while one of them decodes.
CACHE_DIR=${PUSHT_CACHE_DIR:-}
IDLE_FILTER=${PUSHT_IDLE_FILTER:-drop_zero}

# 750k steps is the pushing_pixels convergence budget (~39h there). If the
# 48h wall clock is tight on a slower node, lower PUSHT_STEPS — the trainer
# checkpoints every --save-interval steps, so a truncated run stays usable.
STEPS=${PUSHT_STEPS:-300000}

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

echo "job=${SLURM_JOB_ID} task=${SLURM_ARRAY_TASK_ID} seed=${SEED} pred=${PRED} node=${SLURM_JOB_NODELIST}"
echo "dataset=${DATASET}"
echo "output=${OUTPUT_ROOT}/${PRED}pred_seed_$(printf '%04d' "${SEED}")"

# Best pushing_pixels DP recipe (batches/pushingPixelsDPD.txt, DPE.txt):
# ConvMaxpoolEncoder + DenseResnet 1024x1, lr 3e-4, batch 128, T=100 cosine,
# EMA 0.999 — all defaults of train_pusht_real_dp.py; only the machine- and
# run-specific knobs are passed here.
CACHE_ARGS=()
if [[ -n "${CACHE_DIR}" ]]; then
    CACHE_ARGS=(--frame-cache-dir "${CACHE_DIR}")
fi

srun "${PYTHON}" scripts/train_pusht_real_dp.py \
    --dataset "${DATASET}" \
    --data-format zarr_video \
    --video-camera 1 \
    --seed "${SEED}" \
    --prediction-type "${PRED}" \
    --steps "${STEPS}" \
    --workers "${SLURM_CPUS_PER_TASK}" \
    --frame-stack 2 \
    --idle-filter "${IDLE_FILTER}" \
    ${CACHE_ARGS[@]+"${CACHE_ARGS[@]}"} \
    --output-root "${OUTPUT_ROOT}"
