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


# K1 — paper-faithful Q (ResNet 128×16 + spectral norm)
#uv run python hyperparam_search.py combinedv2_cpascounter_training.py --run --fixed-params '{"trial_seed":1,"control_points":30,"top_k_control_points":15,"q_network_kind":"resnet","q_width":128,"q_depth":16,"q_use_spectral_norm":true,"cp_network_kind":"mlp","cp_width":256,"cp_depth":2,"num_uniform_negatives":32,"num_langevin_negatives":0,"langevin_num_iterations":100,"langevin_lr_init":0.1,"langevin_noise_scale":1.0,"langevin_delta_clip":0.1,"langevin_decay_power":2.0,"inference_langevin_iterations":100}'
#uv run python hyperparam_search.py combinedv2_cpascounter_training.py --run --fixed-params '{"trial_seed":2,"control_points":30,"top_k_control_points":15,"q_network_kind":"resnet","q_width":128,"q_depth":16,"q_use_spectral_norm":true,"cp_network_kind":"mlp","cp_width":256,"cp_depth":2,"num_uniform_negatives":32,"num_langevin_negatives":0,"langevin_num_iterations":100,"langevin_lr_init":0.1,"langevin_noise_scale":1.0,"langevin_delta_clip":0.1,"langevin_decay_power":2.0,"inference_langevin_iterations":100}'

# K2 — paper-D4RL Q (ResNet 512×8 + spectral norm)
#uv run python hyperparam_search.py combinedv2_cpascounter_training.py --run --fixed-params '{"trial_seed":1,"control_points":30,"top_k_control_points":15,"q_network_kind":"resnet","q_width":512,"q_depth":8,"q_use_spectral_norm":true,"cp_network_kind":"mlp","cp_width":256,"cp_depth":2,"num_uniform_negatives":32,"num_langevin_negatives":0,"langevin_num_iterations":100,"langevin_lr_init":0.1,"langevin_noise_scale":1.0,"langevin_delta_clip":0.1,"langevin_decay_power":2.0,"inference_langevin_iterations":100}'
#uv run python hyperparam_search.py combinedv2_cpascounter_training.py --run --fixed-params '{"trial_seed":2,"control_points":30,"top_k_control_points":15,"q_network_kind":"resnet","q_width":512,"q_depth":8,"q_use_spectral_norm":true,"cp_network_kind":"mlp","cp_width":256,"cp_depth":2,"num_uniform_negatives":32,"num_langevin_negatives":0,"langevin_num_iterations":100,"langevin_lr_init":0.1,"langevin_noise_scale":1.0,"langevin_delta_clip":0.1,"langevin_decay_power":2.0,"inference_langevin_iterations":100}'

# K3 — paper-particle Q (ResNet 2048×2, no SN)
#uv run python hyperparam_search.py combinedv2_cpascounter_training.py --run --fixed-params '{"trial_seed":1,"control_points":30,"top_k_control_points":15,"q_network_kind":"resnet","q_width":2048,"q_depth":2,"q_use_spectral_norm":false,"cp_network_kind":"mlp","cp_width":256,"cp_depth":2,"num_uniform_negatives":32,"num_langevin_negatives":0,"langevin_num_iterations":100,"langevin_lr_init":0.1,"langevin_noise_scale":1.0,"langevin_delta_clip":0.1,"langevin_decay_power":2.0,"inference_langevin_iterations":100}'
#uv run python hyperparam_search.py combinedv2_cpascounter_training.py --run --fixed-params '{"trial_seed":2,"control_points":30,"top_k_control_points":15,"q_network_kind":"resnet","q_width":2048,"q_depth":2,"q_use_spectral_norm":false,"cp_network_kind":"mlp","cp_width":256,"cp_depth":2,"num_uniform_negatives":32,"num_langevin_negatives":0,"langevin_num_iterations":100,"langevin_lr_init":0.1,"langevin_noise_scale":1.0,"langevin_delta_clip":0.1,"langevin_decay_power":2.0,"inference_langevin_iterations":100}'

# K4 — current-Q + better negatives + paper Langevin (isolates architecture-vs-rest)
#uv run python hyperparam_search.py combinedv2_cpascounter_training.py --run --fixed-params '{"trial_seed":1,"control_points":30,"top_k_control_points":15,"q_network_kind":"mlp","q_width":256,"q_depth":2,"q_use_spectral_norm":false,"cp_network_kind":"mlp","cp_width":256,"cp_depth":2,"num_uniform_negatives":32,"num_langevin_negatives":0,"langevin_num_iterations":100,"langevin_lr_init":0.1,"langevin_noise_scale":1.0,"langevin_delta_clip":0.1,"langevin_decay_power":2.0,"inference_langevin_iterations":100}'
#uv run python hyperparam_search.py combinedv2_cpascounter_training.py --run --fixed-params '{"trial_seed":2,"control_points":30,"top_k_control_points":15,"q_network_kind":"mlp","q_width":256,"q_depth":2,"q_use_spectral_norm":false,"cp_network_kind":"mlp","cp_width":256,"cp_depth":2,"num_uniform_negatives":32,"num_langevin_negatives":0,"langevin_num_iterations":100,"langevin_lr_init":0.1,"langevin_noise_scale":1.0,"langevin_delta_clip":0.1,"langevin_decay_power":2.0,"inference_langevin_iterations":100}'

# K5 — current-Q with paper Langevin only (no uniform negatives)
#uv run python hyperparam_search.py combinedv2_cpascounter_training.py --run --fixed-params '{"trial_seed":1,"control_points":30,"top_k_control_points":15,"q_network_kind":"mlp","q_width":256,"q_depth":2,"q_use_spectral_norm":false,"cp_network_kind":"mlp","cp_width":256,"cp_depth":2,"num_uniform_negatives":0,"num_langevin_negatives":0,"langevin_num_iterations":100,"langevin_lr_init":0.1,"langevin_noise_scale":1.0,"langevin_delta_clip":0.1,"langevin_decay_power":2.0,"inference_langevin_iterations":100}'
uv run python hyperparam_search.py combinedv2_cpascounter_training.py --run --fixed-params '{"trial_seed":2,"control_points":30,"top_k_control_points":15,"q_network_kind":"mlp","q_width":256,"q_depth":2,"q_use_spectral_norm":false,"cp_network_kind":"mlp","cp_width":256,"cp_depth":2,"num_uniform_negatives":0,"num_langevin_negatives":0,"langevin_num_iterations":100,"langevin_lr_init":0.1,"langevin_noise_scale":1.0,"langevin_delta_clip":0.1,"langevin_decay_power":2.0,"inference_langevin_iterations":100}'

