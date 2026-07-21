#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
ROOT_DIR=$(cd -- "${SCRIPT_DIR}/.." && pwd)
cd "${ROOT_DIR}"

LOG_DIR=${PUSHT_LOG_DIR:-"${ROOT_DIR}/slurm_jobs/pusht_dp"}
mkdir -p "${LOG_DIR}"

sbatch \
    --chdir="${ROOT_DIR}" \
    --output="${LOG_DIR}/slurm_%A_%a.out" \
    --error="${LOG_DIR}/slurm_%A_%a.err" \
    scripts/train_pusht_real_dp_array.sbatch
