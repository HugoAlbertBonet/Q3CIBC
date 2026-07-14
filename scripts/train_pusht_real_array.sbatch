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
#SBATCH --job-name=pusht-q3c
#SBATCH --output=pusht-q3c_%A_%a.out
#SBATCH --error=pusht-q3c_%A_%a.err

set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
ROOT_DIR=$(cd -- "${SCRIPT_DIR}/.." && pwd)
cd "${ROOT_DIR}"

SEEDS=(11 29 47 83)
SEED=${SEEDS[$SLURM_ARRAY_TASK_ID]}
DATASET=${PUSHT_DATASET:-"${ROOT_DIR}/data/03-23-pusht-data.zip"}
OUTPUT_ROOT=${PUSHT_OUTPUT_ROOT:-"${ROOT_DIR}/checkpoints/pusht_real_combinedv2"}
PYTHON=${PYTHON:-"${ROOT_DIR}/.venv/bin/python"}

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
export WANDB_MODE=${WANDB_MODE:-online}

echo "job=${SLURM_JOB_ID} task=${SLURM_ARRAY_TASK_ID} seed=${SEED} node=${SLURM_JOB_NODELIST}"
echo "dataset=${DATASET}"
echo "output=${OUTPUT_ROOT}/seed_$(printf '%04d' "${SEED}")"

srun "${PYTHON}" scripts/train_pusht_real.py \
    --dataset "${DATASET}" \
    --seed "${SEED}" \
    --steps 100000 \
    --batch-size 64 \
    --workers "${SLURM_CPUS_PER_TASK}" \
    --frame-stack 2 \
    --cameras images0 images1 \
    --output-root "${OUTPUT_ROOT}"
