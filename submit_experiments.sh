#!/usr/bin/env bash
# Submit each experiment in an input file as a SEPARATE SLURM job.
#
# Usage:
#   ./submit_experiments.sh <experiments_file> [batch_name]
#   cat commands.txt | ./submit_experiments.sh - [batch_name]
#
# experiments_file:
#   plain text, one shell command per line. Blank lines and lines starting
#   with '#' are ignored (so you can keep the same commented-template style
#   you already use in hyperparams_dfo.sh).
#
# batch_name:
#   optional. Used as the job-name prefix and the per-job script/log
#   subdirectory. Defaults to the file's basename.
#
# For each command, this script generates slurm_jobs/<batch>/<batch>_NNN.sh
# (one self-contained .sh per experiment) and submits it via sbatch. Logs
# land next to the per-job .sh as <batch>_NNN.out.

set -euo pipefail

if [[ $# -lt 1 ]]; then
  echo "Usage: $0 <experiments_file|-> [batch_name]" >&2
  echo "       (use '-' to read commands from stdin)" >&2
  exit 1
fi

exp_file=$1
batch_name=${2:-}
if [[ "$exp_file" == "-" ]]; then
  : "${batch_name:=stdin_$(date -u +%Y%m%dT%H%M%S)}"
  mapfile -t raw_lines
else
  if [[ ! -f "$exp_file" ]]; then
    echo "Error: $exp_file not found" >&2
    exit 1
  fi
  : "${batch_name:=$(basename "$exp_file")}"
  batch_name=${batch_name%.txt}
  batch_name=${batch_name%.sh}
  mapfile -t raw_lines < "$exp_file"
fi

# Filter blanks and '#'-comments (keep commands verbatim otherwise).
commands=()
for line in "${raw_lines[@]}"; do
  trimmed=${line#"${line%%[![:space:]]*}"}
  [[ -z "$trimmed" || "$trimmed" == \#* ]] && continue
  commands+=("$line")
done

if [[ ${#commands[@]} -eq 0 ]]; then
  echo "No commands found." >&2
  exit 1
fi

out_dir="slurm_jobs/${batch_name}"
mkdir -p "$out_dir"

# NOTE: the jobs run read-only against the shared .venv (UV_NO_SYNC / UV_FROZEN
# in each job script) so concurrent jobs never race on syncing it. That means
# the .venv must already be built BEFORE submitting, with the extras your batch
# needs, e.g.:
#   uv sync --extra libero
# (No pre-sync is done here on purpose — a bare `uv sync` would prune the extra
# packages the jobs depend on.)

# Cluster profile, picked from the submitting host (override with Q3C_CLUSTER).
#   CARC (default): whole GPU on a100|l40s|a40, 48 h wall.
#   snoopy1 (USC lab node, 8x RTX A6000, each also split into 48 shards): the
#   lab's `shared` QOS caps the whole `lab` account at gres/gpu=2 and, as a
#   SEPARATE limit, gres/shard=96, plus 32 CPUs, 250G and 36 h per job. Jobs
#   past the cap wait in the queue (QOSGrpGRES).
#   Default is one whole GPU per job, so the QOS itself keeps us at <= 2 GPUs.
#   Measured 2026-09-14: pen q3c on a whole idle A6000 runs 0.38 s/step (82%
#   util, GPU-bound: ~10.6 h per 100k-step run); shard jobs are placed on GPUs
#   that already hold shards, and particle-16D on a shard of a GPU another user
#   saturated ran 2.3x slower than on a whole GPU. SHARDS=N switches to
#   --gres=shard:N. Do not mix the two in one push: the caps are separate, so
#   2 GPUs + 96 shards would be 4 GPUs' worth.
#   uv, its cache and its managed Python live under /scr/$USER (home is small).
cluster=${Q3C_CLUSTER:-$(hostname -s)}
case "$cluster" in
  snoopy*)
    if [[ -n "${SHARDS:-}" ]]; then snoopy_gres="shard:${SHARDS}"; else snoopy_gres="gpu:1"; fi
    sbatch_resources="#SBATCH --account=lab
#SBATCH --partition=partition-1
#SBATCH --qos=shared
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=${CPUS:-4}
#SBATCH --gres=${snoopy_gres}
#SBATCH --mem=${MEM:-32G}
#SBATCH --time=36:00:00"
    env_setup="export PATH=/scr/\$USER/uv/bin:\$PATH
export UV_CACHE_DIR=/scr/\$USER/uv/cache
export UV_PYTHON_INSTALL_DIR=/scr/\$USER/uv/python"
    ;;
  *)
    sbatch_resources="#SBATCH --account=biyik_1165
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=${CPUS:-4}
#SBATCH --gres=gpu:1
#SBATCH --constraint=\"a100|l40s|a40\"
#SBATCH --mem=32G
#SBATCH --time=48:00:00"
    env_setup=""
    ;;
esac

echo "Batch: ${batch_name}  (${#commands[@]} jobs)  cluster profile: ${cluster}"
echo "Per-job scripts and logs: ${out_dir}/"
echo

submitted_ids=()
i=0
for cmd in "${commands[@]}"; do
  i=$((i + 1))
  tag=$(printf '%03d' "$i")
  job_script="${out_dir}/${batch_name}_${tag}.sh"
  cat > "$job_script" <<EOF
#!/usr/bin/env bash
${sbatch_resources}
#SBATCH --job-name=${batch_name}_${tag}
#SBATCH --output=${out_dir}/${batch_name}_${tag}.out
#SBATCH --error=${out_dir}/${batch_name}_${tag}.err

set -euo pipefail
cd "${PWD}"
${env_setup}

# The shared .venv is built ONCE manually before submitting (see the NOTE near
# the top of this script). Force every uv run in this job to be READ-ONLY
# against it: never sync, never touch the lockfile. This both prevents
# concurrent jobs from racing on .venv (the remove-.venv-bin /
# numpy.random-missing bug) AND stops any job from pruning the 'libero' extra
# that liberoGoal runs depend on.
export UV_NO_SYNC=1
export UV_FROZEN=1
export PYTHONDONTWRITEBYTECODE=1
# Reclaim fragmented CUDA memory (helps the 2-camera pusht runs, which sit near
# the 44-48 GB card limit). Harmless for jobs that fit comfortably. Set both the
# new and legacy names so it works across torch versions.
export PYTORCH_ALLOC_CONF=\${PYTORCH_ALLOC_CONF:-expandable_segments:True}
export PYTORCH_CUDA_ALLOC_CONF=\${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}

# Headless mujoco offscreen render (LIBERO eval). Harmless for non-render envs.
export MUJOCO_GL=\${MUJOCO_GL:-egl}

echo "=========================================="
echo "Batch:   ${batch_name}"
echo "Tag:     ${tag}"
echo "Node:    \$SLURM_JOB_NODELIST"
echo "JobId:   \$SLURM_JOB_ID"
echo "Start:   \$(date)"
echo "=========================================="

# set -x makes bash echo each command BEFORE executing it, so we see the
# fully-expanded form in the log without needing to quote-escape it for echo.
set -x
${cmd}
set +x

echo "=========================================="
echo "Done:    \$(date)"
EOF
  chmod +x "$job_script"

  # Submit and capture the job id ("Submitted batch job 12345")
  # DEPENDENCY=afterany:<id>:<id> makes every job of this batch wait for those
  # jobs (sbatch has no environment variable for --dependency).
  submit_out=$(sbatch ${DEPENDENCY:+--dependency="$DEPENDENCY"} "$job_script")
  job_id=$(awk '{print $NF}' <<<"$submit_out")
  submitted_ids+=("$job_id")
  printf '  [%s] %s  →  job %s\n' "$tag" "$(basename "$job_script")" "$job_id"
done

echo
echo "Submitted ${#submitted_ids[@]} jobs: ${submitted_ids[*]}"
echo
echo "Useful follow-ups:"
echo "  squeue -u \$USER -o '%.12i %.20j %.8T %.10M %.6D %R'   # see queue"
echo "  tail -F ${out_dir}/${batch_name}_001.out                # follow first job"
echo "  scancel ${submitted_ids[*]}                              # cancel them all"
