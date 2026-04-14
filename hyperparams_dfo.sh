#!/bin/bash
#SBATCH --account=biyik_1165
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:v100:1
#SBATCH --mem=16G
#SBATCH --time=24:00:00

export WANDB_API_KEY=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
set -e
srun uv run python hyperparam_search.py combinedv2_cpascounter_training.py --run --params '{"control_points": 30, "top_k_control_points": 30, "counter_examples": 8, "mse_weight": 20.0, "separation_weight": 0.1, "separation_loss": "entropy", "entropy_bandwidth": 0.1, "generator_infonce_weight": 0.05, "info_nce_weight": 1.0, "infonce_logit_clamp": 20.0, "learning_rate": 0.001, "training_steps": 200000}'
srun uv run python hyperparam_search.py combinedv2_cpascounter_training.py --run --params '{"control_points": 30, "top_k_control_points": 30, "counter_examples": 16, "mse_weight": 20.0, "separation_weight": 0.1, "separation_loss": "entropy", "entropy_bandwidth": 0.1, "generator_infonce_weight": 0.05, "info_nce_weight": 1.0, "infonce_logit_clamp": 20.0, "learning_rate": 0.001, "training_steps": 200000, "cosine_t_max": 400000}'
srun uv run python hyperparam_search.py combinedv2_cpascounter_training.py --run --params '{"control_points": 30, "top_k_control_points": 30, "counter_examples": 8, "mse_weight": 20.0, "separation_weight": 0.1, "separation_loss": "entropy", "entropy_bandwidth": 0.1, "generator_infonce_weight": 0.05, "info_nce_weight": 1.0, "infonce_logit_clamp": 20.0, "learning_rate": 0.001, "training_steps": 200000, "cosine_t_max": 400000}'
