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

# Exp A — verbatim winner (cp=20, top_k=10, lv=75 @ lr=0.015 noise=0.05 clip=0.015)
#uv run python hyperparam_search.py combinedv2_cpascounter_training.py --run --fixed-params '{"trial_seed":1,"control_points":20,"top_k_control_points":10,"inference_langevin_iterations":75,"langevin_lr_init":0.015,"langevin_noise_scale":0.05,"langevin_delta_clip":0.015}'
#uv run python hyperparam_search.py combinedv2_cpascounter_training.py --run --fixed-params '{"trial_seed":2,"control_points":20,"top_k_control_points":10,"inference_langevin_iterations":75,"langevin_lr_init":0.015,"langevin_noise_scale":0.05,"langevin_delta_clip":0.015}'

# Exp B — more CPs (cp=30, top_k=15)
#uv run python hyperparam_search.py combinedv2_cpascounter_training.py --run --fixed-params '{"trial_seed":1,"control_points":30,"top_k_control_points":15,"inference_langevin_iterations":75,"langevin_lr_init":0.015,"langevin_noise_scale":0.05,"langevin_delta_clip":0.015}'
#uv run python hyperparam_search.py combinedv2_cpascounter_training.py --run --fixed-params '{"trial_seed":2,"control_points":30,"top_k_control_points":15,"inference_langevin_iterations":75,"langevin_lr_init":0.015,"langevin_noise_scale":0.05,"langevin_delta_clip":0.015}'

# Exp C — refine inference (lv=120, lr=0.010, noise=0.04, clip=0.012)
#uv run python hyperparam_search.py combinedv2_cpascounter_training.py --run --fixed-params '{"trial_seed":1,"control_points":20,"top_k_control_points":10,"inference_langevin_iterations":120,"langevin_lr_init":0.010,"langevin_noise_scale":0.04,"langevin_delta_clip":0.012}'
#uv run python hyperparam_search.py combinedv2_cpascounter_training.py --run --fixed-params '{"trial_seed":2,"control_points":20,"top_k_control_points":10,"inference_langevin_iterations":120,"langevin_lr_init":0.010,"langevin_noise_scale":0.04,"langevin_delta_clip":0.012}'

# Exp D — capacity + refine (cp=30, top_k=15, lv=120, lr=0.010, noise=0.04, clip=0.012)
#uv run python hyperparam_search.py combinedv2_cpascounter_training.py --run --fixed-params '{"trial_seed":1,"control_points":30,"top_k_control_points":15,"inference_langevin_iterations":120,"langevin_lr_init":0.010,"langevin_noise_scale":0.04,"langevin_delta_clip":0.012}'
#uv run python hyperparam_search.py combinedv2_cpascounter_training.py --run --fixed-params '{"trial_seed":2,"control_points":30,"top_k_control_points":15,"inference_langevin_iterations":120,"langevin_lr_init":0.010,"langevin_noise_scale":0.04,"langevin_delta_clip":0.012}'

# Exp E — longer training (training_steps=200000)
#uv run python hyperparam_search.py combinedv2_cpascounter_training.py --run --fixed-params '{"trial_seed":1,"control_points":20,"top_k_control_points":10,"training_steps":200000,"inference_langevin_iterations":75,"langevin_lr_init":0.015,"langevin_noise_scale":0.05,"langevin_delta_clip":0.015}'
uv run python hyperparam_search.py combinedv2_cpascounter_training.py --run --fixed-params '{"trial_seed":2,"control_points":20,"top_k_control_points":10,"training_steps":200000,"inference_langevin_iterations":75,"langevin_lr_init":0.015,"langevin_noise_scale":0.05,"langevin_delta_clip":0.015}'

