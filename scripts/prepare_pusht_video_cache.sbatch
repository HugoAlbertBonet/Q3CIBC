#!/usr/bin/env bash
# Build the decoded frame cache for the Diffusion-Policy Push-T archive.
#
# CPU-ONLY on purpose: this is ffmpeg decode + an AREA resize, no GPU involved.
# Requesting a GPU here would hold an accelerator idle for the whole build.
#
# Run this ONCE before submitting batches/pushtWidowX.txt. Without it the first
# training job builds the cache while the other nine sit on GPUs waiting for it.
#
#   sbatch scripts/prepare_pusht_video_cache.sbatch
#   # or onto scratch (recommended -- the cache is ~17 GB and regenerable):
#   PUSHT_CACHE_DIR=/scratch1/$USER/pusht_cache \
#       sbatch scripts/prepare_pusht_video_cache.sbatch
#
# Then pass the same directory to training via --frame-cache-dir.
#
#SBATCH --account=biyik_1165
#SBATCH --partition=main
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=02:00:00
#SBATCH --job-name=pusht-cache
#SBATCH --output=slurm_jobs/pusht_cache_%j.out
#SBATCH --error=slurm_jobs/pusht_cache_%j.err

set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
ROOT_DIR=${SLURM_SUBMIT_DIR:-$(cd -- "${SCRIPT_DIR}/.." && pwd)}
cd "${ROOT_DIR}"
mkdir -p slurm_jobs

DATASET=${PUSHT_DATASET:-"${ROOT_DIR}/data/pusht_widowx_data.zip"}
CAMERA=${PUSHT_CAMERA:-1}
CACHE_DIR=${PUSHT_CACHE_DIR:-}

if [[ ! -f "${DATASET}" ]]; then
    echo "Dataset not found on the compute node: ${DATASET}" >&2
    echo "Copy it over first, e.g.:" >&2
    echo "  rsync -avP data/pusht_widowx_data.zip ${USER}@discovery.usc.edu:${ROOT_DIR}/data/" >&2
    exit 2
fi

# Read-only against the shared .venv, exactly like the training jobs: never
# sync, never touch the lockfile, so a concurrent job can't race on it.
export UV_NO_SYNC=1
export UV_FROZEN=1
export PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-4}

echo "job=${SLURM_JOB_ID:-none} node=${SLURMD_NODENAME:-local}"
echo "dataset=${DATASET}  camera=${CAMERA}  cache_dir=${CACHE_DIR:-<default: data/_frame_cache>}"
echo "start: $(date)"

ARGS=(--dataset "${DATASET}" --camera "${CAMERA}")
if [[ -n "${CACHE_DIR}" ]]; then
    mkdir -p "${CACHE_DIR}"
    ARGS+=(--cache-dir "${CACHE_DIR}")
fi

srun uv run python scripts/prepare_pusht_video_cache.py "${ARGS[@]}"

echo "done: $(date)"
