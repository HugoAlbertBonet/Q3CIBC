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

# Q3C-IBC no-Langevin batch 4 — anchor on trial 21 (54% success)
# Trial 21 = cosine + mse=10 + info_nce=0.5 + top_k=10
# Also probing cp=30 (50% solo) and two new axes (sep_loss type, asymmetric LRs).

# 1. Combine the two best solo wins: cp=30 + info_nce=0.5
#uv run hyperparam_search.py combinedv2_cpascounter_training.py --run --fixed-params '{"control_points": 30, "learning_rate": 1e-3, "batch_size": 512, "top_k_control_points": 30, "separation_weight": 0.1, "mse_weight": 10.0, "info_nce_weight": 0.5, "generator_infonce_weight": 0.05, "training_steps": 150000, "separation_loss": "entropy", "entropy_bandwidth": 0.1, "num_hidden_layers": 2, "num_neurons": 256, "estimator_learning_rate": 1e-3, "scheduler_type": "cosine", "cosine_t0": 50000, "infonce_logit_clamp": 20.0, "langevin_num_iterations": 0, "inference_langevin_iterations": 0}'

# 2. Quadruple stack: trial 21 + cp=30
#uv run hyperparam_search.py combinedv2_cpascounter_training.py --run --fixed-params '{"control_points": 30, "learning_rate": 1e-3, "batch_size": 512, "top_k_control_points": 10, "separation_weight": 0.1, "mse_weight": 10.0, "info_nce_weight": 0.5, "generator_infonce_weight": 0.05, "training_steps": 150000, "separation_loss": "entropy", "entropy_bandwidth": 0.1, "num_hidden_layers": 2, "num_neurons": 256, "estimator_learning_rate": 1e-3, "scheduler_type": "cosine", "cosine_t0": 50000, "infonce_logit_clamp": 20.0, "langevin_num_iterations": 0, "inference_langevin_iterations": 0}'

# 3. Push info_nce_weight even lower: 0.5 -> 0.25
#uv run hyperparam_search.py combinedv2_cpascounter_training.py --run --fixed-params '{"control_points": 20, "learning_rate": 1e-3, "batch_size": 512, "top_k_control_points": 30, "separation_weight": 0.1, "mse_weight": 10.0, "info_nce_weight": 0.25, "generator_infonce_weight": 0.05, "training_steps": 150000, "separation_loss": "entropy", "entropy_bandwidth": 0.1, "num_hidden_layers": 2, "num_neurons": 256, "estimator_learning_rate": 1e-3, "scheduler_type": "cosine", "cosine_t0": 50000, "infonce_logit_clamp": 20.0, "langevin_num_iterations": 0, "inference_langevin_iterations": 0}'

# 4. More CPs: cp=40 (cp=30 helped at 50%; does it keep scaling?)
#uv run hyperparam_search.py combinedv2_cpascounter_training.py --run --fixed-params '{"control_points": 40, "learning_rate": 1e-3, "batch_size": 512, "top_k_control_points": 30, "separation_weight": 0.1, "mse_weight": 10.0, "info_nce_weight": 1.0, "generator_infonce_weight": 0.05, "training_steps": 150000, "separation_loss": "entropy", "entropy_bandwidth": 0.1, "num_hidden_layers": 2, "num_neurons": 256, "estimator_learning_rate": 1e-3, "scheduler_type": "cosine", "cosine_t0": 50000, "infonce_logit_clamp": 20.0, "langevin_num_iterations": 0, "inference_langevin_iterations": 0}'

# 5. New axis: separation_loss "entropy" -> "separation" with matching weight
#uv run hyperparam_search.py combinedv2_cpascounter_training.py --run --fixed-params '{"control_points": 20, "learning_rate": 1e-3, "batch_size": 512, "top_k_control_points": 30, "separation_weight": 0.1, "mse_weight": 10.0, "info_nce_weight": 1.0, "generator_infonce_weight": 0.05, "training_steps": 150000, "separation_loss": "separation", "entropy_bandwidth": 0.1, "num_hidden_layers": 2, "num_neurons": 256, "estimator_learning_rate": 1e-3, "scheduler_type": "cosine", "cosine_t0": 50000, "infonce_logit_clamp": 20.0, "langevin_num_iterations": 0, "inference_langevin_iterations": 0}'

# 6. New axis: asymmetric LRs — slower estimator (5e-4) while generator stays at 1e-3
#uv run hyperparam_search.py combinedv2_cpascounter_training.py --run --fixed-params '{"control_points": 20, "learning_rate": 1e-3, "batch_size": 512, "top_k_control_points": 30, "separation_weight": 0.1, "mse_weight": 10.0, "info_nce_weight": 1.0, "generator_infonce_weight": 0.05, "training_steps": 150000, "separation_loss": "entropy", "entropy_bandwidth": 0.1, "num_hidden_layers": 2, "num_neurons": 256, "estimator_learning_rate": 5e-4, "scheduler_type": "cosine", "cosine_t0": 50000, "infonce_logit_clamp": 20.0, "langevin_num_iterations": 0, "inference_langevin_iterations": 0}'

# 7. Trial 21 + clamp=50 — loose clamp stacked with winning triple
#uv run hyperparam_search.py combinedv2_cpascounter_training.py --run --fixed-params '{"control_points": 20, "learning_rate": 1e-3, "batch_size": 512, "top_k_control_points": 10, "separation_weight": 0.1, "mse_weight": 10.0, "info_nce_weight": 0.5, "generator_infonce_weight": 0.05, "training_steps": 150000, "separation_loss": "entropy", "entropy_bandwidth": 0.1, "num_hidden_layers": 2, "num_neurons": 256, "estimator_learning_rate": 1e-3, "scheduler_type": "cosine", "cosine_t0": 50000, "infonce_logit_clamp": 50.0, "langevin_num_iterations": 0, "inference_langevin_iterations": 0}'

# 8. Trial 21 + sep_weight=0.05 — less separation stacked with winning triple
#uv run hyperparam_search.py combinedv2_cpascounter_training.py --run --fixed-params '{"control_points": 20, "learning_rate": 1e-3, "batch_size": 512, "top_k_control_points": 10, "separation_weight": 0.05, "mse_weight": 10.0, "info_nce_weight": 0.5, "generator_infonce_weight": 0.05, "training_steps": 150000, "separation_loss": "entropy", "entropy_bandwidth": 0.1, "num_hidden_layers": 2, "num_neurons": 256, "estimator_learning_rate": 1e-3, "scheduler_type": "cosine", "cosine_t0": 50000, "infonce_logit_clamp": 20.0, "langevin_num_iterations": 0, "inference_langevin_iterations": 0}'

# 9. Two-lever combo not yet tested: cp=30 + clamp=50
#uv run hyperparam_search.py combinedv2_cpascounter_training.py --run --fixed-params '{"control_points": 30, "learning_rate": 1e-3, "batch_size": 512, "top_k_control_points": 30, "separation_weight": 0.1, "mse_weight": 10.0, "info_nce_weight": 1.0, "generator_infonce_weight": 0.05, "training_steps": 150000, "separation_loss": "entropy", "entropy_bandwidth": 0.1, "num_hidden_layers": 2, "num_neurons": 256, "estimator_learning_rate": 1e-3, "scheduler_type": "cosine", "cosine_t0": 50000, "infonce_logit_clamp": 50.0, "langevin_num_iterations": 0, "inference_langevin_iterations": 0}'

# 10. Kitchen sink: cp=30 + info_nce=0.5 + top_k=10 + clamp=50 (stack everything that helped)
uv run hyperparam_search.py combinedv2_cpascounter_training.py --run --fixed-params '{"control_points": 30, "learning_rate": 1e-3, "batch_size": 512, "top_k_control_points": 10, "separation_weight": 0.1, "mse_weight": 10.0, "info_nce_weight": 0.5, "generator_infonce_weight": 0.05, "training_steps": 150000, "separation_loss": "entropy", "entropy_bandwidth": 0.1, "num_hidden_layers": 2, "num_neurons": 256, "estimator_learning_rate": 1e-3, "scheduler_type": "cosine", "cosine_t0": 50000, "infonce_logit_clamp": 50.0, "langevin_num_iterations": 0, "inference_langevin_iterations": 0}'
