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
# Trial 45: CE=32 + higher info_nce_weight (push estimator harder with CEs)
#uv run python hyperparam_search.py combinedv2_cpascounter_training.py --run --params '{"control_points": 30, "top_k_control_points": 30, "counter_examples": 32, "mse_weight": 20.0, "separation_weight": 0.1, "separation_loss": "entropy", "entropy_bandwidth": 0.1, "generator_infonce_weight": 0.05, "info_nce_weight": 2.0, "infonce_logit_clamp": 20.0, "learning_rate": 0.001, "training_steps": 200000}'
# Trial 46: CE=32 + weaker generator adversarial signal (avoid destabilizing generator)
#uv run python hyperparam_search.py combinedv2_cpascounter_training.py --run --params '{"control_points": 30, "top_k_control_points": 30, "counter_examples": 32, "mse_weight": 20.0, "separation_weight": 0.1, "separation_loss": "entropy", "entropy_bandwidth": 0.1, "generator_infonce_weight": 0.01, "info_nce_weight": 1.0, "infonce_logit_clamp": 20.0, "learning_rate": 0.001, "training_steps": 200000}'
# Trial 47: CE=8 + fewer topk CP (reduce total negatives: 15+8=23 instead of 30+8=38)
#uv run python hyperparam_search.py combinedv2_cpascounter_training.py --run --params '{"control_points": 30, "top_k_control_points": 15, "counter_examples": 8, "mse_weight": 20.0, "separation_weight": 0.1, "separation_loss": "entropy", "entropy_bandwidth": 0.1, "generator_infonce_weight": 0.05, "info_nce_weight": 1.0, "infonce_logit_clamp": 20.0, "learning_rate": 0.001, "training_steps": 200000}'
# Trial 48: CE=0 + batch_size=256 (more gradient noise = implicit regularization against collapse)
uv run python hyperparam_search.py combinedv2_cpascounter_training.py --run --params '{"control_points": 30, "top_k_control_points": 30, "counter_examples": 0, "batch_size": 256, "mse_weight": 20.0, "separation_weight": 0.1, "separation_loss": "entropy", "entropy_bandwidth": 0.1, "generator_infonce_weight": 0.05, "info_nce_weight": 1.0, "infonce_logit_clamp": 20.0, "learning_rate": 0.001, "training_steps": 200000}'

# Trial 1: Baseline — CP=30, mse_w=20, sep_w=0.1, entropy, 200k steps (best combinedv2 generator config)
#uv run python hyperparam_search.py ibc_with_cps_training.py --run --fixed-params '{"counter_examples": 32, "estimator_learning_rate": 0.001, "num_hidden_layers": 2, "num_neurons": 256}' --params '{"control_points": 30, "top_k_control_points": 30, "mse_weight": 20.0, "separation_weight": 0.1, "separation_loss": "entropy", "entropy_bandwidth": 0.1, "learning_rate": 0.001, "training_steps": 200000, "batch_size": 512}'
# Trial 2: CP=50 — n_dim=6 liked CP=50, and this script has a better estimator now
#uv run python hyperparam_search.py ibc_with_cps_training.py --run --fixed-params '{"counter_examples": 32, "estimator_learning_rate": 0.001, "num_hidden_layers": 2, "num_neurons": 256}' --params '{"control_points": 50, "top_k_control_points": 30, "mse_weight": 20.0, "separation_weight": 0.1, "separation_loss": "entropy", "entropy_bandwidth": 0.1, "learning_rate": 0.001, "training_steps": 200000, "batch_size": 512}'
# Trial 3: Higher mse_weight=30 (stronger generator supervision)
#uv run python hyperparam_search.py ibc_with_cps_training.py --run --fixed-params '{"counter_examples": 32, "estimator_learning_rate": 0.001, "num_hidden_layers": 2, "num_neurons": 256}' --params '{"control_points": 30, "top_k_control_points": 30, "mse_weight": 30.0, "separation_weight": 0.1, "separation_loss": "entropy", "entropy_bandwidth": 0.1, "learning_rate": 0.001, "training_steps": 200000, "batch_size": 512}'
# Trial 4: CP=20 + 100k steps (lighter generator, rely on strong DFO estimator)
#uv run python hyperparam_search.py ibc_with_cps_training.py --run --fixed-params '{"counter_examples": 32, "estimator_learning_rate": 0.001, "num_hidden_layers": 2, "num_neurons": 256}' --params '{"control_points": 20, "top_k_control_points": 20, "mse_weight": 20.0, "separation_weight": 0.1, "separation_loss": "entropy", "entropy_bandwidth": 0.1, "learning_rate": 0.001, "training_steps": 100000, "batch_size": 512}'

