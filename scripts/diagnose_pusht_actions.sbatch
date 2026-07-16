#!/usr/bin/env bash
#SBATCH --account=biyik_1165
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --constraint="a100|l40s|a40|v100"
#SBATCH --mem=48G
#SBATCH --time=00:30:00
#SBATCH --job-name=pusht-diag
#SBATCH --output=pusht-diag_%j.out
#SBATCH --error=pusht-diag_%j.err

set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
ROOT_DIR=${SLURM_SUBMIT_DIR:-$(cd -- "${SCRIPT_DIR}/.." && pwd)}
cd "${ROOT_DIR}"

DATASET=${PUSHT_DATASET:-"${ROOT_DIR}/data/03-23-pusht-data.zip"}
OUTPUT_ROOT=${PUSHT_OUTPUT_ROOT:-"${ROOT_DIR}/checkpoints/pusht_real_combinedv2"}
PYTHON=${PYTHON:-"${ROOT_DIR}/.venv/bin/python"}

export UV_NO_SYNC=1
export UV_FROZEN=1
export PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}

# All four seeds in one job (inference is cheap).
srun "${PYTHON}" scripts/diagnose_pusht_actions.py \
    --output-root "${OUTPUT_ROOT}" \
    --seeds 11 29 47 83 \
    --dataset "${DATASET}" \
    --num-samples 3000
