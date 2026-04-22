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
# IBC + CP 4
#uv run python hyperparam_search.py ibc_with_cps_training.py --run --params '{"control_points": 30, "num_hidden_layers": 2, "num_neurons": 256, "batch_size": 512, "learning_rate": 1e-3, "estimator_learning_rate": 1e-3, "counter_examples": 32, "mse_weight": 20, "separation_loss": "entropy", "entropy_bandwidth": 0.1, "separation_weight": 0.1, "training_steps": 150000}'
#uv run python hyperparam_search.py ibc_with_cps_training.py --run --params '{"control_points": 30, "num_hidden_layers": 4, "num_neurons": 512, "batch_size": 512, "learning_rate": 1e-3, "estimator_learning_rate": 1e-3, "counter_examples": 32, "mse_weight": 20, "separation_loss": "entropy", "entropy_bandwidth": 0.1, "separation_weight": 0.1, "training_steps": 150000}'
#uv run python hyperparam_search.py ibc_with_cps_training.py --run --params '{"control_points": 20, "num_hidden_layers": 2, "num_neurons": 256, "batch_size": 512, "learning_rate": 1e-3, "estimator_learning_rate": 1e-3, "counter_examples": 64, "mse_weight": 20, "separation_loss": "entropy", "entropy_bandwidth": 0.1, "separation_weight": 0.1, "training_steps": 150000}'
#uv run python hyperparam_search.py ibc_with_cps_training.py --run --params '{"control_points": 30, "num_hidden_layers": 2, "num_neurons": 256, "batch_size": 512, "learning_rate": 1e-3, "estimator_learning_rate": 1e-3, "counter_examples": 32, "mse_weight": 50, "separation_loss": "entropy", "entropy_bandwidth": 0.05, "separation_weight": 0.1, "training_steps": 150000}'

# Completely implicit 2
# Extend the sweep upward to bracket
#uv run python hyperparam_search.py ibc_with_cpsv2_training.py --run --params '{"control_points": 20, "num_hidden_layers": 2, "num_neurons": 256, "batch_size": 512, "learning_rate": 1e-3, "estimator_learning_rate": 1e-3, "counter_examples": 64, "mse_weight": 5, "separation_loss": "entropy", "entropy_bandwidth": 0.1, "separation_weight": 0.1, "target_update_interval": 10000, "training_steps": 150000}'

# Noise check: replicate the best (B.3) — accuracy delta within 1-2pp means the whole trend is noise
uv run python hyperparam_search.py ibc_with_cpsv2_training.py --run --params '{"control_points": 20, "num_hidden_layers": 2, "num_neurons": 256, "batch_size": 512, "learning_rate": 1e-3, "estimator_learning_rate": 1e-3, "counter_examples": 64, "mse_weight": 5, "separation_loss": "entropy", "entropy_bandwidth": 0.1, "separation_weight": 0.1, "target_update_interval": 5000, "training_steps": 150000}'

