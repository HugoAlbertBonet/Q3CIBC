#!/usr/bin/env bash
#SBATCH --account=biyik_1165
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --constraint="a100|l40s|a40|v100"
#SBATCH --mem=48G
#SBATCH --time=02:00:00
#SBATCH --job-name=pusht-diag-dp
#SBATCH --output=pusht-diag-dp_%j.out
#SBATCH --error=pusht-diag-dp_%j.err

# Diagnose the 6 WidowX Diffusion-Policy batch checkpoints (d01..d06). Like the
# Q3C sbatch: diagnose_pusht_actions_dp.py addresses runs as output_root/seed_XXXX
# but the batch wrote tag dirs (mixed seeds 11/29), so symlink one seed_0001 slot
# per tag and reuse it. Each checkpoint is scored under BOTH inference schedules
# (DDPM and DDIM) over the SAME denoiser, matching how DP is deployed.
# --dataset OMITTED on purpose: each checkpoint's config.json / norm_stats picks
# its own archive + data_format (zarr_video).

set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
ROOT_DIR=${SLURM_SUBMIT_DIR:-$(cd -- "${SCRIPT_DIR}/.." && pwd)}
cd "${ROOT_DIR}"

OUTPUT_ROOT=${PUSHT_OUTPUT_ROOT:-"${ROOT_DIR}/checkpoints/pusht_real_dp"}
PYTHON=${PYTHON:-"${ROOT_DIR}/.venv/bin/python"}
# PUSHT_DIAG_NO_EMA=1 -> raw denoiser; default EMA (deploy default).
EMA_FLAG=${PUSHT_DIAG_NO_EMA:+--no-ema}
EMA_SUFFIX=${PUSHT_DIAG_NO_EMA:+_raw}
# PUSHT_DIAG_SAMPLERS overrides the schedules to run (space-separated).
SAMPLERS=${PUSHT_DIAG_SAMPLERS:-"ddpm ddim"}

export UV_NO_SYNC=1
export UV_FROZEN=1
export PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}

TAGS=(
    d01_eps_s11_dropzero
    d02_eps_s29_dropzero
    d03_vpred_s11_dropzero
    d04_vpred_s29_dropzero
    d05_vpred_s11_none
    d06_eps_s11_none
)

mkdir -p "${ROOT_DIR}/results"

for tag in "${TAGS[@]}"; do
    if [[ ! -d "${OUTPUT_ROOT}/${tag}" ]]; then
        echo "SKIP ${tag}: ${OUTPUT_ROOT}/${tag} missing"
        continue
    fi
    ln -sfn "${tag}" "${OUTPUT_ROOT}/seed_0001"
    for sampler in ${SAMPLERS}; do
        echo "===== diagnose ${tag} (${sampler}) ====="
        srun "${PYTHON}" scripts/diagnose_pusht_actions_dp.py \
            --output-root "${OUTPUT_ROOT}" \
            --seeds 1 \
            --num-samples 3000 \
            --sampler "${sampler}" \
            --out "${ROOT_DIR}/results/diag_${tag}_${sampler}${EMA_SUFFIX}.json" \
            --dump-arrays "${ROOT_DIR}/results/diag_arrays${EMA_SUFFIX}" \
            ${EMA_FLAG}
    done
done

rm -f "${OUTPUT_ROOT}/seed_0001"
