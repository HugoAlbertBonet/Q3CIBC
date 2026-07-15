#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
ROOT_DIR=$(cd -- "${SCRIPT_DIR}/.." && pwd)
cd "${ROOT_DIR}"

LOG_DIR=${PUSHT_LOG_DIR:-"${ROOT_DIR}/slurm_jobs/pusht"}
mkdir -p "${LOG_DIR}"

sbatch \
    --output=/dev/null \
    --error=/dev/null \
    scripts/train_pusht_real_array.sbatch
