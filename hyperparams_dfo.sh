#!/bin/bash
#SBATCH --account=biyik_1165
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --constraint="a100|l40s|a40|v100"
#SBATCH --mem=8G
#SBATCH --time=25:00:00

export WANDB_API_KEY=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
# IBC + CP 1
# 1. Baseline (current particle config, Langevin off everywhere)
#uv run hyperparam_search.py combinedv2_cpascounter_training.py --run --fixed-params '{"langevin_num_iterations": 0, "inference_langevin_iterations": 0, "num_hidden_layers": 2, "num_neurons": 256}'

# 2. More control points — does action-space coverage matter?
#uv run hyperparam_search.py combinedv2_cpascounter_training.py --run --params '{"control_points": 50}' --fixed-params '{"langevin_num_iterations": 0, "inference_langevin_iterations": 0, "num_hidden_layers": 2, "num_neurons": 256}'

# 3. Heavier regression signal — mse_weight 5 -> 10
#uv run hyperparam_search.py combinedv2_cpascounter_training.py --run --params '{"mse_weight": 10.0}' --fixed-params '{"langevin_num_iterations": 0, "inference_langevin_iterations": 0, "num_hidden_layers": 2, "num_neurons": 256}'

# 4. Lower LR (both optimizers) — is 1e-3 destabilizing?
#uv run hyperparam_search.py combinedv2_cpascounter_training.py --run --params '{"learning_rate": 5e-4, "estimator_learning_rate": 5e-4}' --fixed-params '{"langevin_num_iterations": 0, "inference_langevin_iterations": 0, "num_hidden_layers": 2, "num_neurons": 256}'

# 5. Fewer counter-examples — does restricting top-k focus training?
uv run hyperparam_search.py combinedv2_cpascounter_training.py --run --params '{"top_k_control_points": 10}' --fixed-params '{"langevin_num_iterations": 0, "inference_langevin_iterations": 0, "num_hidden_layers": 2, "num_neurons": 256}'


# Completely implicit
#uv run python hyperparam_search.py ibc_with_cpsv2_training.py --run --params '{"control_points": 20, "num_hidden_layers": 2, "num_neurons": 256, "batch_size": 512, "learning_rate": 1e-3, "estimator_learning_rate": 1e-3, "counter_examples": 64, "mse_weight": 5, "separation_loss": "entropy", "entropy_bandwidth": 0.1, "separation_weight": 0.1, "target_update_interval": 2000, "training_steps": 150000}'

