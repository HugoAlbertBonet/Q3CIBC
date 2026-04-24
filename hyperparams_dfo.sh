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

# Q3C-IBC batch 7
#   (A) Reproducibility: run trial 21 3x to measure seed variance on the same config.
#   (B) Gentle inference Langevin: knobs from trajectory analysis (lr=0.01, noise=0.1,
#       clip=0.02) — confirmed to raise Q smoothly with drift ~0.15.
#   (C) Two landscape probes: pure-grad at same iters, even gentler, more aggressive.
#   (D) One cross-config transfer: gentle Langevin on cp=40 (trial 29 = 52% solo best).
# Concurrency-safe: each trial gets its own run_<id> config/checkpoint, trial_ids assigned
# atomically via flock, so all 10 can run as parallel sbatch jobs.

# === A. Reproducibility: trial 21 baseline × 3 ===

# 1. Trial 21 rep 1
#uv run hyperparam_search.py combinedv2_cpascounter_training.py --run --fixed-params '{"control_points": 20, "learning_rate": 1e-3, "batch_size": 512, "top_k_control_points": 10, "separation_weight": 0.1, "mse_weight": 10.0, "info_nce_weight": 0.5, "generator_infonce_weight": 0.05, "training_steps": 150000, "separation_loss": "entropy", "entropy_bandwidth": 0.1, "num_hidden_layers": 2, "num_neurons": 256, "estimator_learning_rate": 1e-3, "scheduler_type": "cosine", "cosine_t0": 50000, "infonce_logit_clamp": 20.0, "langevin_num_iterations": 0, "inference_langevin_iterations": 0}'

# 2. Trial 21 rep 2
#uv run hyperparam_search.py combinedv2_cpascounter_training.py --run --fixed-params '{"control_points": 20, "learning_rate": 1e-3, "batch_size": 512, "top_k_control_points": 10, "separation_weight": 0.1, "mse_weight": 10.0, "info_nce_weight": 0.5, "generator_infonce_weight": 0.05, "training_steps": 150000, "separation_loss": "entropy", "entropy_bandwidth": 0.1, "num_hidden_layers": 2, "num_neurons": 256, "estimator_learning_rate": 1e-3, "scheduler_type": "cosine", "cosine_t0": 50000, "infonce_logit_clamp": 20.0, "langevin_num_iterations": 0, "inference_langevin_iterations": 0}'

# 3. Trial 21 rep 3
#uv run hyperparam_search.py combinedv2_cpascounter_training.py --run --fixed-params '{"control_points": 20, "learning_rate": 1e-3, "batch_size": 512, "top_k_control_points": 10, "separation_weight": 0.1, "mse_weight": 10.0, "info_nce_weight": 0.5, "generator_infonce_weight": 0.05, "training_steps": 150000, "separation_loss": "entropy", "entropy_bandwidth": 0.1, "num_hidden_layers": 2, "num_neurons": 256, "estimator_learning_rate": 1e-3, "scheduler_type": "cosine", "cosine_t0": 50000, "infonce_logit_clamp": 20.0, "langevin_num_iterations": 0, "inference_langevin_iterations": 0}'

# === B. Gentle inference Langevin (lr=0.01, noise=0.1, clip=0.02) — iter sweep ===

# 4. inf_lv=25
#uv run hyperparam_search.py combinedv2_cpascounter_training.py --run --fixed-params '{"control_points": 20, "learning_rate": 1e-3, "batch_size": 512, "top_k_control_points": 10, "separation_weight": 0.1, "mse_weight": 10.0, "info_nce_weight": 0.5, "generator_infonce_weight": 0.05, "training_steps": 150000, "separation_loss": "entropy", "entropy_bandwidth": 0.1, "num_hidden_layers": 2, "num_neurons": 256, "estimator_learning_rate": 1e-3, "scheduler_type": "cosine", "cosine_t0": 50000, "infonce_logit_clamp": 20.0, "langevin_num_iterations": 0, "inference_langevin_iterations": 25, "langevin_lr_init": 0.01, "langevin_noise_scale": 0.1, "langevin_delta_clip": 0.02}'

# 5. inf_lv=50
#uv run hyperparam_search.py combinedv2_cpascounter_training.py --run --fixed-params '{"control_points": 20, "learning_rate": 1e-3, "batch_size": 512, "top_k_control_points": 10, "separation_weight": 0.1, "mse_weight": 10.0, "info_nce_weight": 0.5, "generator_infonce_weight": 0.05, "training_steps": 150000, "separation_loss": "entropy", "entropy_bandwidth": 0.1, "num_hidden_layers": 2, "num_neurons": 256, "estimator_learning_rate": 1e-3, "scheduler_type": "cosine", "cosine_t0": 50000, "infonce_logit_clamp": 20.0, "langevin_num_iterations": 0, "inference_langevin_iterations": 50, "langevin_lr_init": 0.01, "langevin_noise_scale": 0.1, "langevin_delta_clip": 0.02}'

# 6. inf_lv=100
#uv run hyperparam_search.py combinedv2_cpascounter_training.py --run --fixed-params '{"control_points": 20, "learning_rate": 1e-3, "batch_size": 512, "top_k_control_points": 10, "separation_weight": 0.1, "mse_weight": 10.0, "info_nce_weight": 0.5, "generator_infonce_weight": 0.05, "training_steps": 150000, "separation_loss": "entropy", "entropy_bandwidth": 0.1, "num_hidden_layers": 2, "num_neurons": 256, "estimator_learning_rate": 1e-3, "scheduler_type": "cosine", "cosine_t0": 50000, "infonce_logit_clamp": 20.0, "langevin_num_iterations": 0, "inference_langevin_iterations": 100, "langevin_lr_init": 0.01, "langevin_noise_scale": 0.1, "langevin_delta_clip": 0.02}'

# === C. Landscape probes ===

# 7. Pure gradient ascent (noise=0), inf_lv=50 — isolate deterministic refinement
#uv run hyperparam_search.py combinedv2_cpascounter_training.py --run --fixed-params '{"control_points": 20, "learning_rate": 1e-3, "batch_size": 512, "top_k_control_points": 10, "separation_weight": 0.1, "mse_weight": 10.0, "info_nce_weight": 0.5, "generator_infonce_weight": 0.05, "training_steps": 150000, "separation_loss": "entropy", "entropy_bandwidth": 0.1, "num_hidden_layers": 2, "num_neurons": 256, "estimator_learning_rate": 1e-3, "scheduler_type": "cosine", "cosine_t0": 50000, "infonce_logit_clamp": 20.0, "langevin_num_iterations": 0, "inference_langevin_iterations": 50, "langevin_lr_init": 0.01, "langevin_noise_scale": 0.0, "langevin_delta_clip": 0.02}'

# 8. Even gentler (lr=0.005, clip=0.01), inf_lv=50 — ultra-fine refinement
#uv run hyperparam_search.py combinedv2_cpascounter_training.py --run --fixed-params '{"control_points": 20, "learning_rate": 1e-3, "batch_size": 512, "top_k_control_points": 10, "separation_weight": 0.1, "mse_weight": 10.0, "info_nce_weight": 0.5, "generator_infonce_weight": 0.05, "training_steps": 150000, "separation_loss": "entropy", "entropy_bandwidth": 0.1, "num_hidden_layers": 2, "num_neurons": 256, "estimator_learning_rate": 1e-3, "scheduler_type": "cosine", "cosine_t0": 50000, "infonce_logit_clamp": 20.0, "langevin_num_iterations": 0, "inference_langevin_iterations": 50, "langevin_lr_init": 0.005, "langevin_noise_scale": 0.05, "langevin_delta_clip": 0.01}'

# 9. Slightly more aggressive grad (lr=0.02, clip=0.02), inf_lv=50 — larger steps but bounded
#uv run hyperparam_search.py combinedv2_cpascounter_training.py --run --fixed-params '{"control_points": 20, "learning_rate": 1e-3, "batch_size": 512, "top_k_control_points": 10, "separation_weight": 0.1, "mse_weight": 10.0, "info_nce_weight": 0.5, "generator_infonce_weight": 0.05, "training_steps": 150000, "separation_loss": "entropy", "entropy_bandwidth": 0.1, "num_hidden_layers": 2, "num_neurons": 256, "estimator_learning_rate": 1e-3, "scheduler_type": "cosine", "cosine_t0": 50000, "infonce_logit_clamp": 20.0, "langevin_num_iterations": 0, "inference_langevin_iterations": 50, "langevin_lr_init": 0.02, "langevin_noise_scale": 0.05, "langevin_delta_clip": 0.02}'

# === D. Cross-config transfer: gentle Langevin on cp=40 ===

# 10. cp=40 (trial 29 = 52%) + gentle inf_lv=50
#uv run hyperparam_search.py combinedv2_cpascounter_training.py --run --fixed-params '{"control_points": 40, "learning_rate": 1e-3, "batch_size": 512, "top_k_control_points": 30, "separation_weight": 0.1, "mse_weight": 10.0, "info_nce_weight": 1.0, "generator_infonce_weight": 0.05, "training_steps": 150000, "separation_loss": "entropy", "entropy_bandwidth": 0.1, "num_hidden_layers": 2, "num_neurons": 256, "estimator_learning_rate": 1e-3, "scheduler_type": "cosine", "cosine_t0": 50000, "infonce_logit_clamp": 20.0, "langevin_num_iterations": 0, "inference_langevin_iterations": 50, "langevin_lr_init": 0.01, "langevin_noise_scale": 0.1, "langevin_delta_clip": 0.02}'



uv run python hyperparam_search.py ibc_with_cpsv2_training.py --run --params '{"control_points": 20, "num_hidden_layers": 2, "num_neurons": 256, "batch_size": 512, "learning_rate": 1e-3, "estimator_learning_rate": 1e-3, "counter_examples": 64, "mse_weight": 5, "separation_loss": "entropy", "entropy_bandwidth": 0.1, "separation_weight": 0.1, "target_update_interval": 2000, "exclude_top_from_separation": false, "training_steps": 150000}'

#uv run python hyperparam_search.py ibc_with_cpsv2_training.py --run --params '{"control_points": 20, "num_hidden_layers": 2, "num_neurons": 256, "batch_size": 512, "learning_rate": 1e-3, "estimator_learning_rate": 1e-3, "counter_examples": 64, "mse_weight": 5, "separation_loss": "entropy", "entropy_bandwidth": 0.1, "separation_weight": 0.1, "target_update_interval": 2000, "exclude_top_from_separation": true, "training_steps": 150000}'

#uv run python hyperparam_search.py ibc_with_cpsv2_training.py --run --params '{"control_points": 20, "num_hidden_layers": 2, "num_neurons": 256, "batch_size": 512, "learning_rate": 1e-3, "estimator_learning_rate": 1e-3, "counter_examples": 64, "mse_weight": 5, "separation_loss": "entropy", "entropy_bandwidth": 0.1, "separation_weight": 0.05, "target_update_interval": 2000, "exclude_top_from_separation": true, "training_steps": 150000}'

#uv run python hyperparam_search.py ibc_with_cpsv2_training.py --run --params '{"control_points": 20, "num_hidden_layers": 2, "num_neurons": 256, "batch_size": 512, "learning_rate": 1e-3, "estimator_learning_rate": 1e-3, "counter_examples": 64, "mse_weight": 5, "separation_loss": "entropy", "entropy_bandwidth": 0.05, "separation_weight": 0.1, "target_update_interval": 2000, "exclude_top_from_separation": true, "training_steps": 150000}'
