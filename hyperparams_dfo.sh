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


# combinedv2 @ n_dim=32 — batch 1 — n_dim=8 winning recipe verbatim + cp probe.
# Architecture & loss settings exactly as the n_dim=8 winner (I.c++).
# Inference Langevin: gentle (lr=0.015, noise=0.05, clip=0.015, 75 iters) — calibrated to
# goal_distance/3 = 0.05/3 ≈ 0.015, which is unchanged at n_dim=32.

# === Exp A — winning recipe verbatim (cp=20, top_k=10), seeds 1 & 2 === DONE
#uv run hyperparam_search.py combinedv2_cpascounter_training.py --run --fixed-params '{"control_points": 20, "learning_rate": 1e-3, "batch_size": 512, "top_k_control_points": 10, "separation_weight": 0.1, "mse_weight": 10.0, "info_nce_weight": 0.5, "generator_infonce_weight": 0.05, "training_steps": 150000, "separation_loss": "entropy", "entropy_bandwidth": 0.1, "num_hidden_layers": 2, "num_neurons": 256, "estimator_learning_rate": 1e-3, "scheduler_type": "cosine", "cosine_t0": 50000, "infonce_logit_clamp": 20.0, "num_uniform_negatives": 0, "num_langevin_negatives": 0, "langevin_num_iterations": 0, "gradient_penalty_weight": 0.0, "inference_langevin_iterations": 75, "langevin_lr_init": 0.015, "langevin_noise_scale": 0.05, "langevin_delta_clip": 0.015, "trial_seed": 1}'

#uv run hyperparam_search.py combinedv2_cpascounter_training.py --run --fixed-params '{"control_points": 20, "learning_rate": 1e-3, "batch_size": 512, "top_k_control_points": 10, "separation_weight": 0.1, "mse_weight": 10.0, "info_nce_weight": 0.5, "generator_infonce_weight": 0.05, "training_steps": 150000, "separation_loss": "entropy", "entropy_bandwidth": 0.1, "num_hidden_layers": 2, "num_neurons": 256, "estimator_learning_rate": 1e-3, "scheduler_type": "cosine", "cosine_t0": 50000, "infonce_logit_clamp": 20.0, "num_uniform_negatives": 0, "num_langevin_negatives": 0, "langevin_num_iterations": 0, "gradient_penalty_weight": 0.0, "inference_langevin_iterations": 75, "langevin_lr_init": 0.015, "langevin_noise_scale": 0.05, "langevin_delta_clip": 0.015, "trial_seed": 2}'

# === Exp B — cp=30 + top_k=15 (50%-ratio preserved), seeds 1 & 2 ===DONE
#uv run hyperparam_search.py combinedv2_cpascounter_training.py --run --fixed-params '{"control_points": 30, "learning_rate": 1e-3, "batch_size": 512, "top_k_control_points": 15, "separation_weight": 0.1, "mse_weight": 10.0, "info_nce_weight": 0.5, "generator_infonce_weight": 0.05, "training_steps": 150000, "separation_loss": "entropy", "entropy_bandwidth": 0.1, "num_hidden_layers": 2, "num_neurons": 256, "estimator_learning_rate": 1e-3, "scheduler_type": "cosine", "cosine_t0": 50000, "infonce_logit_clamp": 20.0, "num_uniform_negatives": 0, "num_langevin_negatives": 0, "langevin_num_iterations": 0, "gradient_penalty_weight": 0.0, "inference_langevin_iterations": 75, "langevin_lr_init": 0.015, "langevin_noise_scale": 0.05, "langevin_delta_clip": 0.015, "trial_seed": 1}'

uv run hyperparam_search.py combinedv2_cpascounter_training.py --run --fixed-params '{"control_points": 30, "learning_rate": 1e-3, "batch_size": 512, "top_k_control_points": 15, "separation_weight": 0.1, "mse_weight": 10.0, "info_nce_weight": 0.5, "generator_infonce_weight": 0.05, "training_steps": 150000, "separation_loss": "entropy", "entropy_bandwidth": 0.1, "num_hidden_layers": 2, "num_neurons": 256, "estimator_learning_rate": 1e-3, "scheduler_type": "cosine", "cosine_t0": 50000, "infonce_logit_clamp": 20.0, "num_uniform_negatives": 0, "num_langevin_negatives": 0, "langevin_num_iterations": 0, "gradient_penalty_weight": 0.0, "inference_langevin_iterations": 75, "langevin_lr_init": 0.015, "langevin_noise_scale": 0.05, "langevin_delta_clip": 0.015, "trial_seed": 2}'

