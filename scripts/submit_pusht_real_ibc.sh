#!/usr/bin/env bash
# Submit the IBC Push-T training array on discovery.
#
# The zarr_video dataset decodes its MP4s into a ~17 GB uint8 memmap on first
# use. That build is concurrency-safe (one process builds, the rest poll), but
# if the GPU array starts cold then N-1 tasks sit on allocated GPUs waiting for
# whichever task won the lock. So this submits a CPU-only cache-prep job first
# and makes the training array depend on it; when the cache already exists the
# prep job is skipped entirely.
#
#   bash scripts/submit_pusht_real_ibc.sh
#   PUSHT_DRY_RUN=1 bash scripts/submit_pusht_real_ibc.sh    # print, don't submit
#
# Env overrides:
#   PUSHT_DATASET           archive path (default data/pusht_widowx_data.zip)
#   PUSHT_FRAME_CACHE       cache dir (default <dataset dir>/_frame_cache).
#                           Point at scratch if home quota cannot take ~17 GB.
#   PUSHT_IDLE_FILTER       none|drop_zero|drop_static|subsample (default drop_zero)
#   PUSHT_OUTPUT_ROOT       checkpoint root (default checkpoints/pusht_real_ibc)
#   PUSHT_LOG_DIR           slurm logs (default slurm_jobs/pusht_ibc)
#   PUSHT_ARRAY             array spec (default 0-3 = seeds 11 29 47 83)
#   PUSHT_CACHE_PARTITION   CPU partition for the prep job (default main)
#   PUSHT_ACCOUNT           slurm account (default biyik_1165)
#   PUSHT_SKIP_CACHE=1      submit the array without the cache dependency
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
ROOT_DIR=$(cd -- "${SCRIPT_DIR}/.." && pwd)
cd "${ROOT_DIR}"

DATASET=${PUSHT_DATASET:-"${ROOT_DIR}/data/pusht_widowx_data.zip"}
FRAME_CACHE=${PUSHT_FRAME_CACHE:-"$(dirname "${DATASET}")/_frame_cache"}
IDLE_FILTER=${PUSHT_IDLE_FILTER:-"drop_zero"}
OUTPUT_ROOT=${PUSHT_OUTPUT_ROOT:-"${ROOT_DIR}/checkpoints/pusht_real_ibc"}
LOG_DIR=${PUSHT_LOG_DIR:-"${ROOT_DIR}/slurm_jobs/pusht_ibc"}
ARRAY=${PUSHT_ARRAY:-"0-3"}
CACHE_PARTITION=${PUSHT_CACHE_PARTITION:-"main"}
ACCOUNT=${PUSHT_ACCOUNT:-"biyik_1165"}
PYTHON=${PYTHON:-"${ROOT_DIR}/.venv/bin/python"}
DRY_RUN=${PUSHT_DRY_RUN:-0}

VIDEO_CAMERA=1
IMAGE_H=240
IMAGE_W=320

if [[ ! -f "${DATASET}" ]]; then
    echo "Dataset not found: ${DATASET}" >&2
    exit 2
fi
if [[ ! -x "${PYTHON}" ]]; then
    echo "Project Python not found/executable: ${PYTHON}" >&2
    echo "Run 'uv sync' in ${ROOT_DIR} first." >&2
    exit 2
fi

mkdir -p "${LOG_DIR}"

# Mirrors PushTWidowXVideoDataset._ensure_frame_cache: the cache is complete
# only when BOTH the memmap and its .json sidecar exist.
CACHE_TAG="$(basename "${DATASET}" .zip)_cam${VIDEO_CAMERA}_${IMAGE_H}x${IMAGE_W}"
CACHE_FILE="${FRAME_CACHE}/${CACHE_TAG}.u8"

# Command tracing goes to stderr so that `$(run sbatch --parsable ...)` captures
# only the job id, in dry-run as well as for real.
run() {
    if [[ "${DRY_RUN}" != "0" ]]; then
        { printf '[dry-run]'; printf ' %q' "$@"; echo; } >&2
        echo "DRYRUN_JOBID"
        return 0
    fi
    "$@"
}

echo "dataset      ${DATASET}"
echo "frame cache  ${CACHE_FILE}"
echo "idle filter  ${IDLE_FILTER}"
echo "output root  ${OUTPUT_ROOT}"
echo "array        ${ARRAY}"
echo

DEPENDENCY=()
if [[ -f "${CACHE_FILE}" && -f "${CACHE_FILE}.json" ]]; then
    echo "Frame cache already built; submitting the training array directly."
elif [[ "${PUSHT_SKIP_CACHE:-0}" != "0" ]]; then
    echo "PUSHT_SKIP_CACHE set: skipping the prep job. The first array task will"
    echo "build the cache while the others idle on their GPUs."
else
    echo "Frame cache missing — submitting a CPU-only prep job first."
    echo "(partition '${CACHE_PARTITION}'; override with PUSHT_CACHE_PARTITION)"
    CACHE_JOB=$(run sbatch --parsable \
        --account="${ACCOUNT}" \
        --partition="${CACHE_PARTITION}" \
        --job-name=pusht-ibc-cache \
        --nodes=1 --ntasks=1 --cpus-per-task=8 --mem=16G --time=02:00:00 \
        --chdir="${ROOT_DIR}" \
        --output="${LOG_DIR}/cache_%j.out" \
        --error="${LOG_DIR}/cache_%j.err" \
        --wrap "UV_NO_SYNC=1 UV_FROZEN=1 '${PYTHON}' scripts/prepare_pusht_video_cache.py --dataset '${DATASET}' --camera ${VIDEO_CAMERA} --image-height ${IMAGE_H} --image-width ${IMAGE_W} --cache-dir '${FRAME_CACHE}'")
    echo "  cache job: ${CACHE_JOB}"
    # afterok, not afterany: if the decode fails there is nothing to train on.
    DEPENDENCY=(--dependency="afterok:${CACHE_JOB}")
fi

export PUSHT_DATASET="${DATASET}"
export PUSHT_FRAME_CACHE="${FRAME_CACHE}"
export PUSHT_IDLE_FILTER="${IDLE_FILTER}"
export PUSHT_OUTPUT_ROOT="${OUTPUT_ROOT}"
export PUSHT_LOG_DIR="${LOG_DIR}"

TRAIN_JOB=$(run sbatch --parsable \
    --array="${ARRAY}" \
    ${DEPENDENCY[@]+"${DEPENDENCY[@]}"} \
    --chdir="${ROOT_DIR}" \
    --output="${LOG_DIR}/slurm_%A_%a.out" \
    --error="${LOG_DIR}/slurm_%A_%a.err" \
    scripts/train_pusht_real_ibc_array.sbatch)
echo "  train job: ${TRAIN_JOB}"

echo
echo "Watch:    squeue -u \$USER"
echo "Progress: tail -f ${LOG_DIR}/pusht_ibc_001.out"
