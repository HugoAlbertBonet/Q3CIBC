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

# Q3C-IBC batch 12 — fill in seeds 2/3/4 for I.c@cp=30 and seeds 1/2/3/4 for I.c++.
# I.c@cp=30   = E + cp=30 + top_k=15 + inf_lv=50 (lr=0.01, noise=0.05, clip=0.01)
# I.c++       = E + cp=20 + top_k=10 + inf_lv=75 (lr=0.015, noise=0.05, clip=0.015)

# === I.c @ cp=30 — seeds 2, 3, 4 === DONE
#uv run hyperparam_search.py combinedv2_cpascounter_training.py --run --fixed-params '{"control_points": 30, "learning_rate": 1e-3, "batch_size": 512, "top_k_control_points": 15, "separation_weight": 0.1, "mse_weight": 10.0, "info_nce_weight": 0.5, "generator_infonce_weight": 0.05, "training_steps": 150000, "separation_loss": "entropy", "entropy_bandwidth": 0.1, "num_hidden_layers": 2, "num_neurons": 256, "estimator_learning_rate": 1e-3, "scheduler_type": "cosine", "cosine_t0": 50000, "infonce_logit_clamp": 20.0, "num_uniform_negatives": 0, "num_langevin_negatives": 0, "langevin_num_iterations": 0, "gradient_penalty_weight": 0.0, "inference_langevin_iterations": 50, "langevin_lr_init": 0.01, "langevin_noise_scale": 0.05, "langevin_delta_clip": 0.01, "trial_seed": 2}'

#uv run hyperparam_search.py combinedv2_cpascounter_training.py --run --fixed-params '{"control_points": 30, "learning_rate": 1e-3, "batch_size": 512, "top_k_control_points": 15, "separation_weight": 0.1, "mse_weight": 10.0, "info_nce_weight": 0.5, "generator_infonce_weight": 0.05, "training_steps": 150000, "separation_loss": "entropy", "entropy_bandwidth": 0.1, "num_hidden_layers": 2, "num_neurons": 256, "estimator_learning_rate": 1e-3, "scheduler_type": "cosine", "cosine_t0": 50000, "infonce_logit_clamp": 20.0, "num_uniform_negatives": 0, "num_langevin_negatives": 0, "langevin_num_iterations": 0, "gradient_penalty_weight": 0.0, "inference_langevin_iterations": 50, "langevin_lr_init": 0.01, "langevin_noise_scale": 0.05, "langevin_delta_clip": 0.01, "trial_seed": 3}'

#uv run hyperparam_search.py combinedv2_cpascounter_training.py --run --fixed-params '{"control_points": 30, "learning_rate": 1e-3, "batch_size": 512, "top_k_control_points": 15, "separation_weight": 0.1, "mse_weight": 10.0, "info_nce_weight": 0.5, "generator_infonce_weight": 0.05, "training_steps": 150000, "separation_loss": "entropy", "entropy_bandwidth": 0.1, "num_hidden_layers": 2, "num_neurons": 256, "estimator_learning_rate": 1e-3, "scheduler_type": "cosine", "cosine_t0": 50000, "infonce_logit_clamp": 20.0, "num_uniform_negatives": 0, "num_langevin_negatives": 0, "langevin_num_iterations": 0, "gradient_penalty_weight": 0.0, "inference_langevin_iterations": 50, "langevin_lr_init": 0.01, "langevin_noise_scale": 0.05, "langevin_delta_clip": 0.01, "trial_seed": 4}'

# === I.c++ (cp=20, inf_lv=75, lr=0.015, clip=0.015) — seeds 1, 2, 3, 4 === DONE
#uv run hyperparam_search.py combinedv2_cpascounter_training.py --run --fixed-params '{"control_points": 20, "learning_rate": 1e-3, "batch_size": 512, "top_k_control_points": 10, "separation_weight": 0.1, "mse_weight": 10.0, "info_nce_weight": 0.5, "generator_infonce_weight": 0.05, "training_steps": 150000, "separation_loss": "entropy", "entropy_bandwidth": 0.1, "num_hidden_layers": 2, "num_neurons": 256, "estimator_learning_rate": 1e-3, "scheduler_type": "cosine", "cosine_t0": 50000, "infonce_logit_clamp": 20.0, "num_uniform_negatives": 0, "num_langevin_negatives": 0, "langevin_num_iterations": 0, "gradient_penalty_weight": 0.0, "inference_langevin_iterations": 75, "langevin_lr_init": 0.015, "langevin_noise_scale": 0.05, "langevin_delta_clip": 0.015, "trial_seed": 1}'

#uv run hyperparam_search.py combinedv2_cpascounter_training.py --run --fixed-params '{"control_points": 20, "learning_rate": 1e-3, "batch_size": 512, "top_k_control_points": 10, "separation_weight": 0.1, "mse_weight": 10.0, "info_nce_weight": 0.5, "generator_infonce_weight": 0.05, "training_steps": 150000, "separation_loss": "entropy", "entropy_bandwidth": 0.1, "num_hidden_layers": 2, "num_neurons": 256, "estimator_learning_rate": 1e-3, "scheduler_type": "cosine", "cosine_t0": 50000, "infonce_logit_clamp": 20.0, "num_uniform_negatives": 0, "num_langevin_negatives": 0, "langevin_num_iterations": 0, "gradient_penalty_weight": 0.0, "inference_langevin_iterations": 75, "langevin_lr_init": 0.015, "langevin_noise_scale": 0.05, "langevin_delta_clip": 0.015, "trial_seed": 2}'

#uv run hyperparam_search.py combinedv2_cpascounter_training.py --run --fixed-params '{"control_points": 20, "learning_rate": 1e-3, "batch_size": 512, "top_k_control_points": 10, "separation_weight": 0.1, "mse_weight": 10.0, "info_nce_weight": 0.5, "generator_infonce_weight": 0.05, "training_steps": 150000, "separation_loss": "entropy", "entropy_bandwidth": 0.1, "num_hidden_layers": 2, "num_neurons": 256, "estimator_learning_rate": 1e-3, "scheduler_type": "cosine", "cosine_t0": 50000, "infonce_logit_clamp": 20.0, "num_uniform_negatives": 0, "num_langevin_negatives": 0, "langevin_num_iterations": 0, "gradient_penalty_weight": 0.0, "inference_langevin_iterations": 75, "langevin_lr_init": 0.015, "langevin_noise_scale": 0.05, "langevin_delta_clip": 0.015, "trial_seed": 3}'

uv run hyperparam_search.py combinedv2_cpascounter_training.py --run --fixed-params '{"control_points": 20, "learning_rate": 1e-3, "batch_size": 512, "top_k_control_points": 10, "separation_weight": 0.1, "mse_weight": 10.0, "info_nce_weight": 0.5, "generator_infonce_weight": 0.05, "training_steps": 150000, "separation_loss": "entropy", "entropy_bandwidth": 0.1, "num_hidden_layers": 2, "num_neurons": 256, "estimator_learning_rate": 1e-3, "scheduler_type": "cosine", "cosine_t0": 50000, "infonce_logit_clamp": 20.0, "num_uniform_negatives": 0, "num_langevin_negatives": 0, "langevin_num_iterations": 0, "gradient_penalty_weight": 0.0, "inference_langevin_iterations": 75, "langevin_lr_init": 0.015, "langevin_noise_scale": 0.05, "langevin_delta_clip": 0.015, "trial_seed": 4}'


# 4
#uv run python hyperparam_search.py ibc_with_cpsv2_training.py --run --params '{"control_points": 20, "num_hidden_layers": 2, "num_neurons": 256, "batch_size": 512, "learning_rate": 1e-3, "estimator_learning_rate": 1e-3, "counter_examples": 64, "mse_weight": 5, "separation_loss": "entropy", "entropy_bandwidth": 0.1, "separation_weight": 0.1, "target_update_interval": 2000, "exclude_top_from_separation": false, "training_steps": 150000}'

#uv run python hyperparam_search.py ibc_with_cpsv2_training.py --run --params '{"control_points": 20, "num_hidden_layers": 2, "num_neurons": 256, "batch_size": 512, "learning_rate": 1e-3, "estimator_learning_rate": 1e-3, "counter_examples": 64, "mse_weight": 5, "separation_loss": "entropy", "entropy_bandwidth": 0.1, "separation_weight": 0.1, "target_update_interval": 2000, "exclude_top_from_separation": true, "scheduler_type": "cosine_warm_restarts", "training_steps": 150000}'

#uv run python hyperparam_search.py ibc_with_cpsv2_training.py --run --params '{"control_points": 20, "num_hidden_layers": 2, "num_neurons": 256, "batch_size": 512, "learning_rate": 1e-3, "estimator_learning_rate": 1e-3, "counter_examples": 64, "mse_weight": 5, "separation_loss": "entropy", "entropy_bandwidth": 0.1, "separation_weight": 0.05, "target_update_interval": 2000, "exclude_top_from_separation": true, "scheduler_type": "cosine_warm_restarts", "training_steps": 150000}'

#uv run python hyperparam_search.py ibc_with_cpsv2_training.py --run --params '{"control_points": 20, "num_hidden_layers": 2, "num_neurons": 256, "batch_size": 512, "learning_rate": 1e-3, "estimator_learning_rate": 1e-3, "counter_examples": 64, "mse_weight": 5, "separation_loss": "entropy", "entropy_bandwidth": 0.05, "separation_weight": 0.1, "target_update_interval": 2000, "exclude_top_from_separation": true, "scheduler_type": "cosine_warm_restarts", "training_steps": 150000}'

# Experiment 1 with warm cosine scheduler
#uv run python hyperparam_search.py ibc_with_cpsv2_training.py --run --params '{"control_points": 20, "num_hidden_layers": 2, "num_neurons": 256, "batch_size": 512, "learning_rate": 1e-3, "estimator_learning_rate": 1e-3, "counter_examples": 64, "mse_weight": 5, "separation_loss": "entropy", "entropy_bandwidth": 0.1, "separation_weight": 0.1, "target_update_interval": 2000, "exclude_top_from_separation": false, "scheduler_type": "cosine_warm_restarts", "training_steps": 150000}'
