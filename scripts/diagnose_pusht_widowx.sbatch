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
#SBATCH --job-name=pusht-diag-wx
#SBATCH --output=pusht-diag-wx_%j.out
#SBATCH --error=pusht-diag-wx_%j.err

# Diagnose the 10 WidowX batch checkpoints (c01..c10, all --seed 11 -> tag dirs).
# diagnose_pusht_actions.py addresses runs as output_root/seed_XXXX, but the
# batch wrote tag dirs, so we symlink one seed_0001 slot per tag and reuse it.
# --dataset is OMITTED on purpose: diagnose then reads data_archive/data_format
# from each checkpoint's own config.json (zarr_video), which is the whole point.

set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
ROOT_DIR=${SLURM_SUBMIT_DIR:-$(cd -- "${SCRIPT_DIR}/.." && pwd)}
cd "${ROOT_DIR}"

OUTPUT_ROOT=${PUSHT_OUTPUT_ROOT:-"${ROOT_DIR}/checkpoints/pusht_real_combinedv2_v2"}
PYTHON=${PYTHON:-"${ROOT_DIR}/.venv/bin/python"}
# PUSHT_DIAG_NO_EMA=1 -> raw weights (the faithful search recipe); default EMA.
EMA_FLAG=${PUSHT_DIAG_NO_EMA:+--no-ema}
EMA_SUFFIX=${PUSHT_DIAG_NO_EMA:+_raw}

export UV_NO_SYNC=1
export UV_FROZEN=1
export PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}

TAGS=(
    c01_dropzero_base
    c02_none_control
    c03_dropstatic
    c04_subsample
    c05_libero_recipe
    c06_hard_negatives
    c07_lowmse_cp
    c08_dropzero_aug
    c09_condxy_dropzero
    c10_condxy_nofilter
)

mkdir -p "${ROOT_DIR}/results"

for tag in "${TAGS[@]}"; do
    if [[ ! -d "${OUTPUT_ROOT}/${tag}" ]]; then
        echo "SKIP ${tag}: ${OUTPUT_ROOT}/${tag} missing"
        continue
    fi
    echo "===== diagnose ${tag} ====="
    ln -sfn "${tag}" "${OUTPUT_ROOT}/seed_0001"
    srun "${PYTHON}" scripts/diagnose_pusht_actions.py \
        --output-root "${OUTPUT_ROOT}" \
        --seeds 1 \
        --num-samples 3000 \
        --out "${ROOT_DIR}/results/diag_${tag}${EMA_SUFFIX}.json" \
        ${EMA_FLAG}
done

rm -f "${OUTPUT_ROOT}/seed_0001"
