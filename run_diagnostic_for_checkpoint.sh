#!/usr/bin/env bash
# Run the particle diagnostic on a checkpoint dir using THAT run's own config.
# Usage:
#   ./run_diagnostic_for_checkpoint.sh <checkpoint_dir> [seeds...]
#
# The script picks up config.json from the checkpoint dir and exposes it via
# Q3C_CONFIG_PATH so run_diagnostic_particle.py uses the exact arch / Langevin
# settings the run was trained and evaluated with.
set -euo pipefail

if [[ $# -lt 1 ]]; then
  echo "Usage: $0 <checkpoint_dir> [seeds...]" >&2
  exit 1
fi

ckpt_dir=$1
shift
seeds=("$@")
if [[ ${#seeds[@]} -eq 0 ]]; then
  seeds=(0 1 2)
fi

cfg="${ckpt_dir}/config.json"
ckpt="${ckpt_dir}/control_point_generator.pt"
if [[ ! -f "$cfg" ]]; then
  echo "Missing $cfg" >&2; exit 1
fi
if [[ ! -f "$ckpt" ]]; then
  echo "Missing $ckpt" >&2; exit 1
fi

run_id=$(basename "$ckpt_dir")
out_dir="plots/particle_diagnostic_n16/${run_id}"

Q3C_CONFIG_PATH="$cfg" uv run python -m simulations.run_diagnostic_particle \
  --checkpoint "$ckpt" \
  --seeds "${seeds[@]}" \
  --output-dir "$out_dir" \
  --no-show-langevin-trajectories --no-show-langevin-final-positions
