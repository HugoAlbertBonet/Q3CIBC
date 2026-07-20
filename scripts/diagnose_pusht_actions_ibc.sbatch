#!/usr/bin/env bash
#SBATCH --account=biyik_1165
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --constraint="a100|l40s|a40|v100"
#SBATCH --mem=48G
#SBATCH --time=01:00:00
#SBATCH --job-name=pusht-diag-ibc
#SBATCH --output=pusht-diag-ibc_%j.out
#SBATCH --error=pusht-diag-ibc_%j.err

set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
ROOT_DIR=${SLURM_SUBMIT_DIR:-$(cd -- "${SCRIPT_DIR}/.." && pwd)}
cd "${ROOT_DIR}"

DATASET=${PUSHT_DATASET:-"${ROOT_DIR}/data/03-23-pusht-data.zip"}
OUTPUT_ROOT=${PUSHT_OUTPUT_ROOT:-"${ROOT_DIR}/checkpoints/pusht_real_ibc"}
PYTHON=${PYTHON:-"${ROOT_DIR}/.venv/bin/python"}
# Only the seeds that have finished training. Override with
# PUSHT_DIAG_SEEDS="11 29 47 83" once the rest land; missing seeds are
# reported as errors in the JSON rather than aborting the job.
SEEDS=${PUSHT_DIAG_SEEDS:-"29 47"}
# Result JSON named after the checkpoint root so runs never overwrite each
# other. PUSHT_DIAG_OUT overrides.
DIAG_OUT=${PUSHT_DIAG_OUT:-"${ROOT_DIR}/results/pusht_action_diagnostic_$(basename "${OUTPUT_ROOT}")${PUSHT_DIAG_ZERO_MOTION:+_zeromotion}.json"}
# PUSHT_DIAG_ZERO_MOTION=1 duplicates the newest frame across the stack, to
# test whether a deploy runaway is an OOD static-stack artifact.
ZERO_MOTION_FLAG=${PUSHT_DIAG_ZERO_MOTION:+--zero-motion}
# PUSHT_DIAG_CKPT_STEP=60000 diagnoses an intermediate snapshot instead of the
# final weights (useful for a seed still training).
CKPT_STEP_FLAG=${PUSHT_DIAG_CKPT_STEP:+--ckpt-step ${PUSHT_DIAG_CKPT_STEP}}

export UV_NO_SYNC=1
export UV_FROZEN=1
export PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}

echo "job=${SLURM_JOB_ID} node=${SLURM_JOB_NODELIST}"
echo "checkpoints=${OUTPUT_ROOT}  seeds=${SEEDS}"
echo "out=${DIAG_OUT}"

# DFO scores batch_size x 2048 candidates per forward pass, so the batch is
# smaller than the q3c diagnostic's. --dfo-repeats 3 measures how much the
# stochastic search moves the selected action on an unchanged observation.
srun "${PYTHON}" scripts/diagnose_pusht_actions_ibc.py \
    --output-root "${OUTPUT_ROOT}" \
    --seeds ${SEEDS} \
    --dataset "${DATASET}" \
    --num-samples 3000 \
    --batch-size 32 \
    --dfo-repeats 3 \
    --out "${DIAG_OUT}" \
    ${ZERO_MOTION_FLAG} \
    ${CKPT_STEP_FLAG}
