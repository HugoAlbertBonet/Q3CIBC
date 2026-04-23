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

export WANDB_API_KEY="${WANDB_API_KEY:?set WANDB_API_KEY before sbatch}"

# Q3C-IBC no-Langevin baseline search — batch of 10
# Anchor: trial 1 config (20 CPs, 2x256, lr 1e-3, mse 5, top_k 30, entropy, warm-restarts)
# All params passed via --fixed-params so parallel runs cannot pollute each other.

# 1. Plain cosine scheduler — remove periodic LR disruption from warm restarts
#uv run hyperparam_search.py combinedv2_cpascounter_training.py --run --fixed-params '{"control_points": 20, "learning_rate": 1e-3, "batch_size": 512, "top_k_control_points": 30, "separation_weight": 0.1, "mse_weight": 5.0, "info_nce_weight": 1.0, "generator_infonce_weight": 0.05, "training_steps": 150000, "separation_loss": "entropy", "entropy_bandwidth": 0.1, "num_hidden_layers": 2, "num_neurons": 256, "estimator_learning_rate": 1e-3, "scheduler_type": "cosine", "cosine_t0": 50000, "infonce_logit_clamp": 20.0, "langevin_num_iterations": 0, "inference_langevin_iterations": 0}'

# 2. Lower LR on both optimizers (1e-3 -> 5e-4)
#uv run hyperparam_search.py combinedv2_cpascounter_training.py --run --fixed-params '{"control_points": 20, "learning_rate": 5e-4, "batch_size": 512, "top_k_control_points": 30, "separation_weight": 0.1, "mse_weight": 5.0, "info_nce_weight": 1.0, "generator_infonce_weight": 0.05, "training_steps": 150000, "separation_loss": "entropy", "entropy_bandwidth": 0.1, "num_hidden_layers": 2, "num_neurons": 256, "estimator_learning_rate": 5e-4, "scheduler_type": "cosine_warm_restarts", "cosine_t0": 50000, "infonce_logit_clamp": 20.0, "langevin_num_iterations": 0, "inference_langevin_iterations": 0}'

# 3. Higher MSE weight (5 -> 10) — push generator CPs closer to expert
#uv run hyperparam_search.py combinedv2_cpascounter_training.py --run --fixed-params '{"control_points": 20, "learning_rate": 1e-3, "batch_size": 512, "top_k_control_points": 30, "separation_weight": 0.1, "mse_weight": 10.0, "info_nce_weight": 1.0, "generator_infonce_weight": 0.05, "training_steps": 150000, "separation_loss": "entropy", "entropy_bandwidth": 0.1, "num_hidden_layers": 2, "num_neurons": 256, "estimator_learning_rate": 1e-3, "scheduler_type": "cosine_warm_restarts", "cosine_t0": 50000, "infonce_logit_clamp": 20.0, "langevin_num_iterations": 0, "inference_langevin_iterations": 0}'

# 4. Halve info_nce_weight — shift balance toward MSE/separation
#uv run hyperparam_search.py combinedv2_cpascounter_training.py --run --fixed-params '{"control_points": 20, "learning_rate": 1e-3, "batch_size": 512, "top_k_control_points": 30, "separation_weight": 0.1, "mse_weight": 5.0, "info_nce_weight": 0.5, "generator_infonce_weight": 0.05, "training_steps": 150000, "separation_loss": "entropy", "entropy_bandwidth": 0.1, "num_hidden_layers": 2, "num_neurons": 256, "estimator_learning_rate": 1e-3, "scheduler_type": "cosine_warm_restarts", "cosine_t0": 50000, "infonce_logit_clamp": 20.0, "langevin_num_iterations": 0, "inference_langevin_iterations": 0}'

# 5. Weaker generator_infonce (0.05 -> 0.01) — let generator train almost purely by MSE+separation
#uv run hyperparam_search.py combinedv2_cpascounter_training.py --run --fixed-params '{"control_points": 20, "learning_rate": 1e-3, "batch_size": 512, "top_k_control_points": 30, "separation_weight": 0.1, "mse_weight": 5.0, "info_nce_weight": 1.0, "generator_infonce_weight": 0.01, "training_steps": 150000, "separation_loss": "entropy", "entropy_bandwidth": 0.1, "num_hidden_layers": 2, "num_neurons": 256, "estimator_learning_rate": 1e-3, "scheduler_type": "cosine_warm_restarts", "cosine_t0": 50000, "infonce_logit_clamp": 20.0, "langevin_num_iterations": 0, "inference_langevin_iterations": 0}'

# 6. Stronger generator_infonce (0.05 -> 0.2) — opposite direction, estimator shapes generator more
#uv run hyperparam_search.py combinedv2_cpascounter_training.py --run --fixed-params '{"control_points": 20, "learning_rate": 1e-3, "batch_size": 512, "top_k_control_points": 30, "separation_weight": 0.1, "mse_weight": 5.0, "info_nce_weight": 1.0, "generator_infonce_weight": 0.2, "training_steps": 150000, "separation_loss": "entropy", "entropy_bandwidth": 0.1, "num_hidden_layers": 2, "num_neurons": 256, "estimator_learning_rate": 1e-3, "scheduler_type": "cosine_warm_restarts", "cosine_t0": 50000, "infonce_logit_clamp": 20.0, "langevin_num_iterations": 0, "inference_langevin_iterations": 0}'

# 7. top_k_control_points 30 -> 10 — only the hardest CPs as counter-examples
#uv run hyperparam_search.py combinedv2_cpascounter_training.py --run --fixed-params '{"control_points": 20, "learning_rate": 1e-3, "batch_size": 512, "top_k_control_points": 10, "separation_weight": 0.1, "mse_weight": 5.0, "info_nce_weight": 1.0, "generator_infonce_weight": 0.05, "training_steps": 150000, "separation_loss": "entropy", "entropy_bandwidth": 0.1, "num_hidden_layers": 2, "num_neurons": 256, "estimator_learning_rate": 1e-3, "scheduler_type": "cosine_warm_restarts", "cosine_t0": 50000, "infonce_logit_clamp": 20.0, "langevin_num_iterations": 0, "inference_langevin_iterations": 0}'

# 8. Smaller batch (512 -> 256) — more gradient steps per epoch
#uv run hyperparam_search.py combinedv2_cpascounter_training.py --run --fixed-params '{"control_points": 20, "learning_rate": 1e-3, "batch_size": 256, "top_k_control_points": 30, "separation_weight": 0.1, "mse_weight": 5.0, "info_nce_weight": 1.0, "generator_infonce_weight": 0.05, "training_steps": 150000, "separation_loss": "entropy", "entropy_bandwidth": 0.1, "num_hidden_layers": 2, "num_neurons": 256, "estimator_learning_rate": 1e-3, "scheduler_type": "cosine_warm_restarts", "cosine_t0": 50000, "infonce_logit_clamp": 20.0, "langevin_num_iterations": 0, "inference_langevin_iterations": 0}'

# 9. Looser logit clamp (20 -> 50) — let InfoNCE produce bigger score gaps
#uv run hyperparam_search.py combinedv2_cpascounter_training.py --run --fixed-params '{"control_points": 20, "learning_rate": 1e-3, "batch_size": 512, "top_k_control_points": 30, "separation_weight": 0.1, "mse_weight": 5.0, "info_nce_weight": 1.0, "generator_infonce_weight": 0.05, "training_steps": 150000, "separation_loss": "entropy", "entropy_bandwidth": 0.1, "num_hidden_layers": 2, "num_neurons": 256, "estimator_learning_rate": 1e-3, "scheduler_type": "cosine_warm_restarts", "cosine_t0": 50000, "infonce_logit_clamp": 50.0, "langevin_num_iterations": 0, "inference_langevin_iterations": 0}'

# 10. Combo — plain cosine + higher MSE (the two most promising levers stacked)
uv run hyperparam_search.py combinedv2_cpascounter_training.py --run --fixed-params '{"control_points": 20, "learning_rate": 1e-3, "batch_size": 512, "top_k_control_points": 30, "separation_weight": 0.1, "mse_weight": 10.0, "info_nce_weight": 1.0, "generator_infonce_weight": 0.05, "training_steps": 150000, "separation_loss": "entropy", "entropy_bandwidth": 0.1, "num_hidden_layers": 2, "num_neurons": 256, "estimator_learning_rate": 1e-3, "scheduler_type": "cosine", "cosine_t0": 50000, "infonce_logit_clamp": 20.0, "langevin_num_iterations": 0, "inference_langevin_iterations": 0}'
