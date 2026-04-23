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

# Q3C-IBC no-Langevin batch 3 — anchor on cosine + mse=10 (42% success from previous batch)
# Fixed across all: cp=20, 2x256 net, plain cosine, langevin off, lr=1e-3, batch=512 unless varied.

# 1. Stack the two best levers: cosine + mse=10 + top_k=10
#uv run hyperparam_search.py combinedv2_cpascounter_training.py --run --fixed-params '{"control_points": 20, "learning_rate": 1e-3, "batch_size": 512, "top_k_control_points": 10, "separation_weight": 0.1, "mse_weight": 10.0, "info_nce_weight": 1.0, "generator_infonce_weight": 0.05, "training_steps": 150000, "separation_loss": "entropy", "entropy_bandwidth": 0.1, "num_hidden_layers": 2, "num_neurons": 256, "estimator_learning_rate": 1e-3, "scheduler_type": "cosine", "cosine_t0": 50000, "infonce_logit_clamp": 20.0, "langevin_num_iterations": 0, "inference_langevin_iterations": 0}'

# 2. Anchor + halve info_nce_weight — shift balance further toward MSE
#uv run hyperparam_search.py combinedv2_cpascounter_training.py --run --fixed-params '{"control_points": 20, "learning_rate": 1e-3, "batch_size": 512, "top_k_control_points": 30, "separation_weight": 0.1, "mse_weight": 10.0, "info_nce_weight": 0.5, "generator_infonce_weight": 0.05, "training_steps": 150000, "separation_loss": "entropy", "entropy_bandwidth": 0.1, "num_hidden_layers": 2, "num_neurons": 256, "estimator_learning_rate": 1e-3, "scheduler_type": "cosine", "cosine_t0": 50000, "infonce_logit_clamp": 20.0, "langevin_num_iterations": 0, "inference_langevin_iterations": 0}'

# 3. Push MSE further: mse_weight 10 -> 20
#uv run hyperparam_search.py combinedv2_cpascounter_training.py --run --fixed-params '{"control_points": 20, "learning_rate": 1e-3, "batch_size": 512, "top_k_control_points": 30, "separation_weight": 0.1, "mse_weight": 20.0, "info_nce_weight": 1.0, "generator_infonce_weight": 0.05, "training_steps": 150000, "separation_loss": "entropy", "entropy_bandwidth": 0.1, "num_hidden_layers": 2, "num_neurons": 256, "estimator_learning_rate": 1e-3, "scheduler_type": "cosine", "cosine_t0": 50000, "infonce_logit_clamp": 20.0, "langevin_num_iterations": 0, "inference_langevin_iterations": 0}'

# 4. Longer training on anchor — does the d2 tail close with more steps?
#uv run hyperparam_search.py combinedv2_cpascounter_training.py --run --fixed-params '{"control_points": 20, "learning_rate": 1e-3, "batch_size": 512, "top_k_control_points": 30, "separation_weight": 0.1, "mse_weight": 10.0, "info_nce_weight": 1.0, "generator_infonce_weight": 0.05, "training_steps": 200000, "separation_loss": "entropy", "entropy_bandwidth": 0.1, "num_hidden_layers": 2, "num_neurons": 256, "estimator_learning_rate": 1e-3, "scheduler_type": "cosine", "cosine_t0": 50000, "infonce_logit_clamp": 20.0, "langevin_num_iterations": 0, "inference_langevin_iterations": 0}'

# 5. Anchor + smaller batch (512 -> 256) — more gradient updates per step budget
#uv run hyperparam_search.py combinedv2_cpascounter_training.py --run --fixed-params '{"control_points": 20, "learning_rate": 1e-3, "batch_size": 256, "top_k_control_points": 30, "separation_weight": 0.1, "mse_weight": 10.0, "info_nce_weight": 1.0, "generator_infonce_weight": 0.05, "training_steps": 150000, "separation_loss": "entropy", "entropy_bandwidth": 0.1, "num_hidden_layers": 2, "num_neurons": 256, "estimator_learning_rate": 1e-3, "scheduler_type": "cosine", "cosine_t0": 50000, "infonce_logit_clamp": 20.0, "langevin_num_iterations": 0, "inference_langevin_iterations": 0}'

# 6. Anchor + looser logit clamp (20 -> 50) — bigger InfoNCE score gaps
uv run hyperparam_search.py combinedv2_cpascounter_training.py --run --fixed-params '{"control_points": 20, "learning_rate": 1e-3, "batch_size": 512, "top_k_control_points": 30, "separation_weight": 0.1, "mse_weight": 10.0, "info_nce_weight": 1.0, "generator_infonce_weight": 0.05, "training_steps": 150000, "separation_loss": "entropy", "entropy_bandwidth": 0.1, "num_hidden_layers": 2, "num_neurons": 256, "estimator_learning_rate": 1e-3, "scheduler_type": "cosine", "cosine_t0": 50000, "infonce_logit_clamp": 50.0, "langevin_num_iterations": 0, "inference_langevin_iterations": 0}'

# 7. Anchor + less separation (0.1 -> 0.05) — trade CP spread for regression fidelity
#uv run hyperparam_search.py combinedv2_cpascounter_training.py --run --fixed-params '{"control_points": 20, "learning_rate": 1e-3, "batch_size": 512, "top_k_control_points": 30, "separation_weight": 0.05, "mse_weight": 10.0, "info_nce_weight": 1.0, "generator_infonce_weight": 0.05, "training_steps": 150000, "separation_loss": "entropy", "entropy_bandwidth": 0.1, "num_hidden_layers": 2, "num_neurons": 256, "estimator_learning_rate": 1e-3, "scheduler_type": "cosine", "cosine_t0": 50000, "infonce_logit_clamp": 20.0, "langevin_num_iterations": 0, "inference_langevin_iterations": 0}'

# 8. Anchor + more control points (20 -> 30) — does anchor config tolerate more CPs now?
#uv run hyperparam_search.py combinedv2_cpascounter_training.py --run --fixed-params '{"control_points": 30, "learning_rate": 1e-3, "batch_size": 512, "top_k_control_points": 30, "separation_weight": 0.1, "mse_weight": 10.0, "info_nce_weight": 1.0, "generator_infonce_weight": 0.05, "training_steps": 150000, "separation_loss": "entropy", "entropy_bandwidth": 0.1, "num_hidden_layers": 2, "num_neurons": 256, "estimator_learning_rate": 1e-3, "scheduler_type": "cosine", "cosine_t0": 50000, "infonce_logit_clamp": 20.0, "langevin_num_iterations": 0, "inference_langevin_iterations": 0}'

# 9. Anchor + fewer control points (20 -> 15) — force tighter coverage of expert action
#uv run hyperparam_search.py combinedv2_cpascounter_training.py --run --fixed-params '{"control_points": 15, "learning_rate": 1e-3, "batch_size": 512, "top_k_control_points": 15, "separation_weight": 0.1, "mse_weight": 10.0, "info_nce_weight": 1.0, "generator_infonce_weight": 0.05, "training_steps": 150000, "separation_loss": "entropy", "entropy_bandwidth": 0.1, "num_hidden_layers": 2, "num_neurons": 256, "estimator_learning_rate": 1e-3, "scheduler_type": "cosine", "cosine_t0": 50000, "infonce_logit_clamp": 20.0, "langevin_num_iterations": 0, "inference_langevin_iterations": 0}'

# 10. Triple-stack: cosine + mse=10 + top_k=10 + info_nce=0.5 (all three top individual wins)
#uv run hyperparam_search.py combinedv2_cpascounter_training.py --run --fixed-params '{"control_points": 20, "learning_rate": 1e-3, "batch_size": 512, "top_k_control_points": 10, "separation_weight": 0.1, "mse_weight": 10.0, "info_nce_weight": 0.5, "generator_infonce_weight": 0.05, "training_steps": 150000, "separation_loss": "entropy", "entropy_bandwidth": 0.1, "num_hidden_layers": 2, "num_neurons": 256, "estimator_learning_rate": 1e-3, "scheduler_type": "cosine", "cosine_t0": 50000, "infonce_logit_clamp": 20.0, "langevin_num_iterations": 0, "inference_langevin_iterations": 0}'
