#!/bin/bash
#SBATCH --account=biyik_1165
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --constraint="a100|l40s|a40|v100"
#SBATCH --mem=8G
#SBATCH --time=8:00:00

export WANDB_API_KEY=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
# IBC + CP 4
#uv run python hyperparam_search.py ibc_with_cps_training.py --run --params '{"control_points": 50, "mse_weight": 50.0, "separation_weight": 0.05, "separation_loss": "separation", "separation_epsilon": 1.0, "learning_rate": 1e-3, "scheduler_type": "cosine", "training_steps": 150000}'
#uv run python hyperparam_search.py ibc_with_cps_training.py --run --params '{"control_points": 30, "mse_weight": 30.0, "separation_weight": 0.01, "separation_loss": "entropy", "entropy_bandwidth": 0.1, "learning_rate": 5e-4, "scheduler_type": "cosine", "training_steps": 150000}'
#uv run python hyperparam_search.py ibc_with_cps_training.py --run --params '{"control_points": 100, "mse_weight": 20.0, "separation_weight": 0.1, "separation_loss": "separation", "separation_epsilon": 1.0, "learning_rate": 1e-3, "scheduler_type": "cosine_warm_restarts", "cosine_t0": 50000, "training_steps": 200000}'
#uv run python hyperparam_search.py ibc_with_cps_training.py --run --params '{"control_points": 75, "mse_weight": 30.0, "separation_weight": 0.05, "separation_loss": "entropy", "entropy_bandwidth": 0.05, "learning_rate": 1e-3, "scheduler_type": "cosine", "training_steps": 150000}'
uv run python hyperparam_search.py ibc_with_cps_training.py --run --params '{"control_points": 30, "mse_weight": 100.0, "separation_weight": 0.01, "separation_loss": "separation", "separation_epsilon": 1.0, "learning_rate": 1e-3, "scheduler_type": "cosine", "training_steps": 100000}'

# Q3C 6
#uv run python hyperparam_search.py combinedv2_cpascounter_training.py --run --params '{"control_points": 30, "counter_examples": 0, "entropy_bandwidth": 0.1, "generator_infonce_weight": 0.05, "info_nce_weight": 1.0, "infonce_logit_clamp": 20.0, "learning_rate": 1e-3, "mse_weight": 20.0, "num_hidden_layers": 2, "num_neurons": 256, "separation_loss": "entropy", "separation_weight": 0.1, "top_k_control_points": 30, "training_steps": 200000, "batch_size": 512}'
#uv run python hyperparam_search.py combinedv2_cpascounter_training.py --run --params '{"control_points": 30, "counter_examples": 0, "entropy_bandwidth": 0.1, "generator_infonce_weight": 0.05, "info_nce_weight": 1.0, "infonce_logit_clamp": 20.0, "learning_rate": 1e-3, "mse_weight": 20.0, "num_hidden_layers": 4, "num_neurons": 256, "separation_loss": "entropy", "separation_weight": 0.1, "top_k_control_points": 30, "training_steps": 200000, "batch_size": 512}'
#uv run python hyperparam_search.py combinedv2_cpascounter_training.py --run --params '{"control_points": 30, "counter_examples": 0, "entropy_bandwidth": 0.1, "generator_infonce_weight": 0.05, "info_nce_weight": 1.0, "infonce_logit_clamp": 20.0, "learning_rate": 1e-3, "mse_weight": 20.0, "num_hidden_layers": 2, "num_neurons": 512, "separation_loss": "entropy", "separation_weight": 0.1, "top_k_control_points": 30, "training_steps": 200000, "batch_size": 512}'
#uv run python hyperparam_search.py combinedv2_cpascounter_training.py --run --params '{"control_points": 30, "counter_examples": 0, "entropy_bandwidth": 0.1, "generator_infonce_weight": 0.1, "info_nce_weight": 1.0, "infonce_logit_clamp": 20.0, "learning_rate": 1e-3, "mse_weight": 15.0, "num_hidden_layers": 2, "num_neurons": 256, "separation_loss": "entropy", "separation_weight": 0.1, "top_k_control_points": 30, "training_steps": 200000, "batch_size": 512}'
#uv run python hyperparam_search.py combinedv2_cpascounter_training.py --run --params '{"control_points": 30, "counter_examples": 0, "entropy_bandwidth": 0.08, "generator_infonce_weight": 0.05, "info_nce_weight": 1.0, "infonce_logit_clamp": 20.0, "learning_rate": 1e-3, "mse_weight": 25.0, "num_hidden_layers": 2, "num_neurons": 256, "separation_loss": "entropy", "separation_weight": 0.1, "top_k_control_points": 30, "training_steps": 200000, "batch_size": 512}'
#uv run python hyperparam_search.py combinedv2_cpascounter_training.py --run --params '{"control_points": 30, "counter_examples": 0, "entropy_bandwidth": 0.1, "generator_infonce_weight": 0.05, "info_nce_weight": 1.0, "infonce_logit_clamp": 20.0, "learning_rate": 1e-3, "mse_weight": 20.0, "num_hidden_layers": 2, "num_neurons": 256, "separation_loss": "entropy", "separation_weight": 0.12, "top_k_control_points": 30, "training_steps": 150000, "batch_size": 512}'
