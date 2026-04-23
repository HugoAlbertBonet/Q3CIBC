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

# Q3C-IBC no-Langevin batch 5 — SOTA: trial 21 = 54% (cp=20, info_nce=0.5, top_k=10, 50% ratio)
# Strategy: push cp while preserving the 50% top_k ratio; probe untouched axes (entropy_bandwidth,
# cosine_t_max, narrow gen_infonce sweep). Avoid stacking on trial 21 (all prior stacks hurt).

# 1. cp=40 + info_nce=0.5 (stack two best solo wins, but keep top_k=30 = 75% ratio)
#uv run hyperparam_search.py combinedv2_cpascounter_training.py --run --fixed-params '{"control_points": 40, "learning_rate": 1e-3, "batch_size": 512, "top_k_control_points": 30, "separation_weight": 0.1, "mse_weight": 10.0, "info_nce_weight": 0.5, "generator_infonce_weight": 0.05, "training_steps": 150000, "separation_loss": "entropy", "entropy_bandwidth": 0.1, "num_hidden_layers": 2, "num_neurons": 256, "estimator_learning_rate": 1e-3, "scheduler_type": "cosine", "cosine_t0": 50000, "infonce_logit_clamp": 20.0, "langevin_num_iterations": 0, "inference_langevin_iterations": 0}'

# 2. cp=40 + top_k=20 + info_nce=0.5 (preserve trial-21's 50% top_k ratio)
#uv run hyperparam_search.py combinedv2_cpascounter_training.py --run --fixed-params '{"control_points": 40, "learning_rate": 1e-3, "batch_size": 512, "top_k_control_points": 20, "separation_weight": 0.1, "mse_weight": 10.0, "info_nce_weight": 0.5, "generator_infonce_weight": 0.05, "training_steps": 150000, "separation_loss": "entropy", "entropy_bandwidth": 0.1, "num_hidden_layers": 2, "num_neurons": 256, "estimator_learning_rate": 1e-3, "scheduler_type": "cosine", "cosine_t0": 50000, "infonce_logit_clamp": 20.0, "langevin_num_iterations": 0, "inference_langevin_iterations": 0}'

# 3. cp=50, info_nce=1.0 (does cp scaling keep paying off past 40?)
#uv run hyperparam_search.py combinedv2_cpascounter_training.py --run --fixed-params '{"control_points": 50, "learning_rate": 1e-3, "batch_size": 512, "top_k_control_points": 30, "separation_weight": 0.1, "mse_weight": 10.0, "info_nce_weight": 1.0, "generator_infonce_weight": 0.05, "training_steps": 150000, "separation_loss": "entropy", "entropy_bandwidth": 0.1, "num_hidden_layers": 2, "num_neurons": 256, "estimator_learning_rate": 1e-3, "scheduler_type": "cosine", "cosine_t0": 50000, "infonce_logit_clamp": 20.0, "langevin_num_iterations": 0, "inference_langevin_iterations": 0}'

# 4. cp=50 + top_k=25 + info_nce=0.5 (50% ratio at cp=50)
#uv run hyperparam_search.py combinedv2_cpascounter_training.py --run --fixed-params '{"control_points": 50, "learning_rate": 1e-3, "batch_size": 512, "top_k_control_points": 25, "separation_weight": 0.1, "mse_weight": 10.0, "info_nce_weight": 0.5, "generator_infonce_weight": 0.05, "training_steps": 150000, "separation_loss": "entropy", "entropy_bandwidth": 0.1, "num_hidden_layers": 2, "num_neurons": 256, "estimator_learning_rate": 1e-3, "scheduler_type": "cosine", "cosine_t0": 50000, "infonce_logit_clamp": 20.0, "langevin_num_iterations": 0, "inference_langevin_iterations": 0}'

# 5. Trial 21 + gen_infonce=0.03 (narrow the gen_infonce peak on the low side of 0.05)
#uv run hyperparam_search.py combinedv2_cpascounter_training.py --run --fixed-params '{"control_points": 20, "learning_rate": 1e-3, "batch_size": 512, "top_k_control_points": 10, "separation_weight": 0.1, "mse_weight": 10.0, "info_nce_weight": 0.5, "generator_infonce_weight": 0.03, "training_steps": 150000, "separation_loss": "entropy", "entropy_bandwidth": 0.1, "num_hidden_layers": 2, "num_neurons": 256, "estimator_learning_rate": 1e-3, "scheduler_type": "cosine", "cosine_t0": 50000, "infonce_logit_clamp": 20.0, "langevin_num_iterations": 0, "inference_langevin_iterations": 0}'

# 6. Trial 21 + gen_infonce=0.1 (narrow the gen_infonce peak on the high side of 0.05)
#uv run hyperparam_search.py combinedv2_cpascounter_training.py --run --fixed-params '{"control_points": 20, "learning_rate": 1e-3, "batch_size": 512, "top_k_control_points": 10, "separation_weight": 0.1, "mse_weight": 10.0, "info_nce_weight": 0.5, "generator_infonce_weight": 0.1, "training_steps": 150000, "separation_loss": "entropy", "entropy_bandwidth": 0.1, "num_hidden_layers": 2, "num_neurons": 256, "estimator_learning_rate": 1e-3, "scheduler_type": "cosine", "cosine_t0": 50000, "infonce_logit_clamp": 20.0, "langevin_num_iterations": 0, "inference_langevin_iterations": 0}'

# 7. Trial 21 + entropy_bandwidth=0.05 (tighter entropy kernel — penalize closer CPs harder)
#uv run hyperparam_search.py combinedv2_cpascounter_training.py --run --fixed-params '{"control_points": 20, "learning_rate": 1e-3, "batch_size": 512, "top_k_control_points": 10, "separation_weight": 0.1, "mse_weight": 10.0, "info_nce_weight": 0.5, "generator_infonce_weight": 0.05, "training_steps": 150000, "separation_loss": "entropy", "entropy_bandwidth": 0.05, "num_hidden_layers": 2, "num_neurons": 256, "estimator_learning_rate": 1e-3, "scheduler_type": "cosine", "cosine_t0": 50000, "infonce_logit_clamp": 20.0, "langevin_num_iterations": 0, "inference_langevin_iterations": 0}'

# 8. Trial 21 + entropy_bandwidth=0.2 (looser entropy kernel — smoother density)
#uv run hyperparam_search.py combinedv2_cpascounter_training.py --run --fixed-params '{"control_points": 20, "learning_rate": 1e-3, "batch_size": 512, "top_k_control_points": 10, "separation_weight": 0.1, "mse_weight": 10.0, "info_nce_weight": 0.5, "generator_infonce_weight": 0.05, "training_steps": 150000, "separation_loss": "entropy", "entropy_bandwidth": 0.2, "num_hidden_layers": 2, "num_neurons": 256, "estimator_learning_rate": 1e-3, "scheduler_type": "cosine", "cosine_t0": 50000, "infonce_logit_clamp": 20.0, "langevin_num_iterations": 0, "inference_langevin_iterations": 0}'

# 9. Trial 21 + cosine_t_max=100000 (decay LR to 1e-6 by step 100k, then hold at eta_min for 50k)
#uv run hyperparam_search.py combinedv2_cpascounter_training.py --run --fixed-params '{"control_points": 20, "learning_rate": 1e-3, "batch_size": 512, "top_k_control_points": 10, "separation_weight": 0.1, "mse_weight": 10.0, "info_nce_weight": 0.5, "generator_infonce_weight": 0.05, "training_steps": 150000, "separation_loss": "entropy", "entropy_bandwidth": 0.1, "num_hidden_layers": 2, "num_neurons": 256, "estimator_learning_rate": 1e-3, "scheduler_type": "cosine", "cosine_t0": 50000, "cosine_t_max": 100000, "infonce_logit_clamp": 20.0, "langevin_num_iterations": 0, "inference_langevin_iterations": 0}'

# 10. Trial 21 + steps=175000 with cosine_t_max=150000 (longer training, low-LR fine-tune tail;
#     avoids trial 20's full-150k decay blowup by freezing LR early).
#uv run hyperparam_search.py combinedv2_cpascounter_training.py --run --fixed-params '{"control_points": 20, "learning_rate": 1e-3, "batch_size": 512, "top_k_control_points": 10, "separation_weight": 0.1, "mse_weight": 10.0, "info_nce_weight": 0.5, "generator_infonce_weight": 0.05, "training_steps": 175000, "separation_loss": "entropy", "entropy_bandwidth": 0.1, "num_hidden_layers": 2, "num_neurons": 256, "estimator_learning_rate": 1e-3, "scheduler_type": "cosine", "cosine_t0": 50000, "cosine_t_max": 150000, "infonce_logit_clamp": 20.0, "langevin_num_iterations": 0, "inference_langevin_iterations": 0}'

# 1. Plateau check — is the mse sweet spot wide or narrow?
#uv run python hyperparam_search.py ibc_with_cpsv2_training.py --run --params '{"control_points": 20, "num_hidden_layers": 2, "num_neurons": 256, "batch_size": 512, "learning_rate": 1e-3, "estimator_learning_rate": 1e-3, "counter_examples": 64, "mse_weight": 10, "separation_loss": "entropy", "entropy_bandwidth": 0.1, "separation_weight": 0.1, "target_update_interval": 2000, "training_steps": 150000}'

# 2. Lower separation — does InfoNCE provide enough spread on its own?
#uv run python hyperparam_search.py ibc_with_cpsv2_training.py --run --params '{"control_points": 20, "num_hidden_layers": 2, "num_neurons": 256, "batch_size": 512, "learning_rate": 1e-3, "estimator_learning_rate": 1e-3, "counter_examples": 64, "mse_weight": 5, "separation_loss": "entropy", "entropy_bandwidth": 0.1, "separation_weight": 0.05, "target_update_interval": 2000, "training_steps": 150000}'

# 3. Higher separation — does forcing more diversity hurt or help?
uv run python hyperparam_search.py ibc_with_cpsv2_training.py --run --params '{"control_points": 20, "num_hidden_layers": 2, "num_neurons": 256, "batch_size": 512, "learning_rate": 1e-3, "estimator_learning_rate": 1e-3, "counter_examples": 64, "mse_weight": 5, "separation_loss": "entropy", "entropy_bandwidth": 0.1, "separation_weight": 0.2, "target_update_interval": 2000, "training_steps": 150000}'

