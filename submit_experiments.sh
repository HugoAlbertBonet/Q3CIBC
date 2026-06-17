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

# Pre-build the shared .venv ONCE here on the submit node. Every SLURM job
# below runs `uv run` against this same .venv. If the env is missing/stale when
# the jobs fire, each `uv run` would try to (re)sync it concurrently and they
# race -- one job removing .venv/bin while another reads it, or numpy half
# reinstalled mid-import. Symptoms seen on the cluster:
#   error: failed to remove file `.venv/bin`: No such file or directory
#   ModuleNotFoundError: No module named 'numpy.random'
# Syncing once up front, then forcing every job to read-only (UV_NO_SYNC /
# UV_FROZEN in the job script), removes the race. Concurrent reads are safe.
echo "Pre-syncing shared .venv (so jobs never sync concurrently)..."
uv sync --frozen
echo

echo "Batch: ${batch_name}  (${#commands[@]} jobs)"
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
#SBATCH --account=biyik_1165
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --constraint="a100|l40s|a40|v100"
#SBATCH --mem=32G
#SBATCH --time=30:00:00
#SBATCH --job-name=${batch_name}_${tag}
#SBATCH --output=${out_dir}/${batch_name}_${tag}.out
#SBATCH --error=${out_dir}/${batch_name}_${tag}.err

set -euo pipefail
cd "${PWD}"

# The shared .venv was already built once by the submit script. Force every
# `uv run` in this job to be READ-ONLY against it -- never sync, never touch
# the lockfile. This is what prevents concurrent jobs from racing on .venv
# (the "failed to remove .venv/bin" / "No module named 'numpy.random'" bug).
export UV_NO_SYNC=1
export UV_FROZEN=1
export PYTHONDONTWRITEBYTECODE=1

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
  submit_out=$(sbatch "$job_script")
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
