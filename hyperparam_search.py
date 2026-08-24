"""Agent-assisted hyperparameter search for Q3C-IBC training scripts.

Runs training trials with different hyperparameter configurations, evaluates
success rates, and supports iterative refinement guided by AI analysis.

Modes:
    --run               Run a single trial (with --params or auto-suggested)
    --auto              Run multiple trials with adaptive exploration
    --analyze           Print summary table of all past trials

Usage:
    python hyperparam_search.py combinedv2_cpascounter_training.py --run
    python hyperparam_search.py combinedv2_cpascounter_training.py --run --params '{"learning_rate": 5e-4}'
    python hyperparam_search.py combinedv2_cpascounter_training.py --auto --max-trials 5
    python hyperparam_search.py combinedv2_cpascounter_training.py --analyze
    python hyperparam_search.py combinedv2_cpascounter_training.py --auto --max-trials 3 --reduced-steps 20000
"""

from __future__ import annotations

import argparse
import fcntl
import json
import os
import random
import re
import secrets
import subprocess
import sys
import time
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch

# torch>=2.6 defaults torch.load to weights_only=True, which rejects the numpy
# arrays inside our norm_stats.pt (obs mean/std, action min/max, and the LIBERO
# goal-embedding matrix). add_safe_globals doesn't reliably fix it under numpy
# 2.x: the pickle stores the global as `numpy.core.multiarray._reconstruct`,
# but on numpy 2.x the real callable's module is `numpy._core.multiarray`, so
# torch's allowlist match misses. We trust our own checkpoints, so force every
# torch.load in this process to weights_only=False.
_ORIG_TORCH_LOAD = torch.load


def _trusted_torch_load(*args, **kwargs):
    kwargs["weights_only"] = False
    return _ORIG_TORCH_LOAD(*args, **kwargs)


torch.load = _trusted_torch_load

ROOT_DIR = Path(__file__).parent
sys.path.insert(0, str(ROOT_DIR))

CONFIG_PATH = ROOT_DIR / "config_json" / "config.json"
RESULTS_BASE_DIR = ROOT_DIR / "results" / "hyperparam_search"
CHECKPOINTS_BASE_DIR = ROOT_DIR / "checkpoints" / "hpsearch"


def _new_run_id() -> str:
    """Unique identifier for a trial: timestamp + random suffix. Safe under concurrency."""
    return datetime.now().strftime("%Y%m%dT%H%M%S") + "_" + secrets.token_hex(4)

# ─── Search space: param_name -> {values, type, location} ────────────────────
# location: "env_training" = environments.<active_env>.training
#            "training_shared" = training_shared
#            "env_model"       = environments.<active_env>.model
SEARCH_SPACE: dict[str, dict] = {
    "control_points": {
        "values": [20, 30, 50, 75, 100],
        "type": "int",
        "location": "env_model",
    },
    "learning_rate": {
        "values": [1e-4, 3e-4, 5e-4, 1e-3, 2e-3, 5e-3],
        "type": "float",
        "location": "env_training",
    },
    "batch_size": {
        "values": [128, 256, 512],
        "type": "int",
        "location": "env_training",
    },
    "counter_examples": {
        "values": [8, 16, 32, 64],
        "type": "int",
        "location": "env_training",
    },
    "top_k_control_points": {
        "values": [10, 20, 50, 70],
        "type": "int",
        "location": "env_training",
    },
    "separation_weight": {
        "values": [0.01, 0.05, 0.1, 0.2],
        "type": "float",
        "location": "training_shared",
    },
    "mse_weight": {
        "values": [1.0, 3.0, 5.0, 10.0],
        "type": "float",
        "location": "training_shared",
    },
    "info_nce_weight": {
        "values": [0.5, 1.0, 2.0],
        "type": "float",
        "location": "training_shared",
    },
    "generator_infonce_weight": {
        "values": [0.01, 0.05, 0.1, 0.2],
        "type": "float",
        "location": "training_shared",
    },
    "training_steps": {
        "values": [50000, 100000, 150000, 200000, 250000, 300000, 350000, 400000, 500000],
        "type": "int",
        "location": "env_training",
    },
    "separation_loss": {
        "values": ["separation", "entropy", "chamfer"],
        "type": "str",
        "location": "env_training",
    },
    "exclude_top_from_separation": {
        "values": [False, True],
        "type": "bool",
        "location": "training_shared",
    },
    "noisy_expert_count": {
        "values": [0, 4, 8, 16],
        "type": "int",
        "location": "training_shared",
    },
    "noisy_expert_std": {
        "values": [0.02, 0.05, 0.1, 0.2],
        "type": "float",
        "location": "training_shared",
    },
    "entropy_bandwidth": {
        "values": [0.05, 0.1, 0.2],
        "type": "float",
        "location": "env_training",
    },
    "num_hidden_layers": {
        "values": [2, 4, 8],
        "type": "int",
        "location": "env_model",
    },
    "num_neurons": {
        "values": [128, 256, 512],
        "type": "int",
        "location": "env_model",
    },
    "estimator_learning_rate": {
        "values": [1e-4, 3e-4, 5e-4, 1e-3],
        "type": "float",
        "location": "env_training",
    },
    "scheduler_type": {
        "values": ["cosine", "cosine_warm_restarts"],
        "type": "str",
        "location": "env_training",
    },
    "cosine_t0": {
        "values": [25000, 50000, 100000],
        "type": "int",
        "location": "env_training",
    },
    "infonce_logit_clamp": {
        "values": [10.0, 20.0, 30.0, 50.0],
        "type": "float",
        "location": "env_training",
    },
    "use_spectral_norm": {
        "values": [True, False],
        "type": "bool",
        "location": "env_model",
    },
    # Per-net architecture (added 2026-05-07 — Q-estimator capacity probe).
    # Mirrors IBC paper's ResNetPreActivation. "mlp" preserves legacy behavior.
    "q_network_kind": {
        "values": ["mlp", "resnet"],
        "type": "str",
        "location": "env_model",
    },
    # Pixel-env image encoder (pushing_pixels / libero_goal_pixels).
    "encoder_kind": {
        "values": ["conv_maxpool", "resnet18"],
        "type": "str",
        "location": "env_model",
    },
    "encoder_pretrained": {
        # true/"imagenet" = torchvision ImageNet weights; "r3m" = local R3M
        # ResNet-18 weights (Ego4D); false = scratch.
        "values": [True, False, "imagenet", "r3m"],
        "type": "str",
        "location": "env_model",
    },
    "encoder_per_camera": {
        "values": [True, False],
        "type": "bool",
        "location": "env_model",
    },
    # Language-goal fusion: concat at head (default) or FiLM into the ResNet
    # stages (DP-LIBERO style; proprio stays concat at head).
    "cond_fusion": {
        "values": ["concat", "film"],
        "type": "str",
        "location": "env_model",
    },
    "frame_stack": {
        "values": [1, 2],
        "type": "int",
        "location": "env",
    },
    # Predict a chunk of K consecutive actions per CP (executed open-loop at
    # eval). 1 = single-step (legacy). DP predicts 16, executes 8.
    "action_chunk": {
        "values": [1, 2, 4, 8, 16],
        "type": "int",
        "location": "env_training",
    },
    # Receding horizon: execute only the first R steps of the K-step chunk,
    # then replan. 0 (default) = execute the whole chunk (pure chunking).
    # Eval-time only — does not change training. DP-style: K=16, R=8.
    "action_execute_horizon": {
        "values": [0, 1, 2, 4, 8],
        "type": "int",
        "location": "env_training",
    },
    "encoder_num_kp": {
        "values": [32, 64, 128],
        "type": "int",
        "location": "env_model",
    },
    # ResNet norm strategy: raw BN's train/eval stat mismatch is hostile to
    # EBM training (Bstandardlibero); gn = DP recipe, bn_frozen keeps ImageNet
    # running stats locked.
    "encoder_norm_kind": {
        "values": ["bn", "gn", "bn_frozen"],
        "type": "str",
        "location": "env_model",
    },
    # Train-time random-crop size (0 = off); eval center-crops to match.
    "image_crop_size": {
        "values": [0, 108, 116],
        "type": "int",
        "location": "env_training",
    },
    # Encoder LR = learning_rate * scale (pretrained trunks want ~0.1x).
    "encoder_lr_scale": {
        "values": [0.05, 0.1, 0.5, 1.0],
        "type": "float",
        "location": "env_training",
    },
    "q_width": {
        "values": [128, 256, 512, 1024, 2048],
        "type": "int",
        "location": "env_model",
    },
    "q_depth": {
        "values": [2, 4, 8, 16],
        "type": "int",
        "location": "env_model",
    },
    "q_use_spectral_norm": {
        "values": [True, False],
        "type": "bool",
        "location": "env_model",
    },
    # IBC-faithful resnet head: official MLPEBM has NO trailing activation
    # before the energy projection. False = faithful; True = legacy (what
    # kcD-era resnet checkpoints trained with). Changes state_dict indices —
    # train and eval must agree (both read this same config key).
    "q_resnet_final_activation": {
        "values": [True, False],
        "type": "bool",
        "location": "env_model",
    },
    # Kitchen only: train/eval on qpos dims [0:9]+[18:39] (30-D), dropping the
    # 29 velocity dims the gymnasium port added. Matches the IBC paper's
    # legacy-d4rl kitchen input content (qpos + constant goal).
    "kitchen_qpos_only": {
        "values": [True, False],
        "type": "bool",
        "location": "env_training",
    },
    "cp_network_kind": {
        "values": ["mlp", "resnet"],
        "type": "str",
        "location": "env_model",
    },
    "cp_width": {
        "values": [128, 256, 512, 1024],
        "type": "int",
        "location": "env_model",
    },
    "cp_depth": {
        "values": [2, 4, 8],
        "type": "int",
        "location": "env_model",
    },
    "cp_use_spectral_norm": {
        "values": [True, False],
        "type": "bool",
        "location": "env_model",
    },
    "cosine_t_max": {
        "values": [
            100000,
            150000,
            200000,
            250000,
            300000,
            350000,
            400000,
            500000,
        ],
        "type": "int",
        "location": "env_training",
    },
    "target_update_interval": {
        "values": [200, 500, 1000, 2000, 5000],
        "type": "int",
        "location": "training_shared",
    },
    "inference_langevin_iterations": {
        "values": [0, 10, 25, 50, 100, 150, 200, 250, 300],
        "type": "int",
        "location": "env_training",
    },
    # Inference-time Langevin hyperparam overrides. When set, these REPLACE
    # the corresponding training-Langevin values (langevin_lr_init, etc.)
    # ONLY for the eval-time refinement chain — training Langevin negs still
    # use the paper-faithful aggressive values. Use to test gentle inference
    # refinement on Q3C's narrow-trained Q surface.
    "inference_langevin_lr_init": {
        "values": [0.005, 0.01, 0.05, 0.1, 0.5],
        "type": "float",
        "location": "env_training",
    },
    "inference_langevin_lr_final": {
        "values": [1e-7, 1e-6, 1e-5, 1e-4],
        "type": "float",
        "location": "env_training",
    },
    "inference_langevin_decay_power": {
        "values": [1.0, 2.0, 4.0],
        "type": "float",
        "location": "env_training",
    },
    "inference_langevin_delta_clip": {
        "values": [0.005, 0.01, 0.02, 0.05, 0.1, 0.5],
        "type": "float",
        "location": "env_training",
    },
    "inference_langevin_noise_scale": {
        "values": [0.0, 0.05, 0.1, 0.3, 1.0],
        "type": "float",
        "location": "env_training",
    },
    # Official-IBC-faithful inference chain options (see memory ibc-repro-fixes;
    # these tripled our in-env IBC's kitchen score). noise_via_stepsize: noise
    # shrinks linearly with stepsize (chain end = pure polish). again_iterations
    # > 0 runs a second polish chain at constant 1e-5 stepsize.
    "inference_langevin_noise_via_stepsize": {
        "values": [True, False],
        "type": "bool",
        "location": "env_training",
    },
    "inference_langevin_again_iterations": {
        "values": [0, 10, 20, 50, 100],
        "type": "int",
        "location": "env_training",
    },
    "inference_langevin_again_noise_scale": {
        "values": [0.0, 0.5, 1.0],
        "type": "float",
        "location": "env_training",
    },
    # 0 = refine the whole CP cloud; k>0 = refine only the top-k CPs by
    # initial Q (k=1 -> single chain from the argmax CP, cheapest Langevin).
    "inference_langevin_top_k": {
        "values": [0, 1, 8, 32],
        "type": "int",
        "location": "env_training",
    },
    # ── CP-DFO refinement at inference (Q3CIBC-specific) ────────────────────
    # When > 0, replaces inference-time Langevin with a DFO-style iterative
    # refinement starting from the CP cloud (optionally with a few extra
    # uniform samples for safety). Cheaper than Langevin (~5 forward passes
    # vs ~100, no autograd) and matches DFO's quality whenever the CP cloud
    # already covers the right action mode — which pushingA showed it does on
    # Pushing. If both `inference_dfo_iterations > 0` and
    # `inference_langevin_iterations > 0` are set, DFO takes precedence.
    "inference_dfo_iterations": {
        "values": [0, 3, 5, 10, 15],
        "type": "int",
        "location": "env_training",
    },
    "inference_dfo_iteration_std": {
        "values": [0.005, 0.01, 0.015, 0.03, 0.05, 0.1, 0.2],
        "type": "float",
        "location": "env_training",
    },
    "inference_dfo_iteration_std_decay": {
        "values": [0.5, 0.7, 0.9],
        "type": "float",
        "location": "env_training",
    },
    "inference_dfo_num_uniform": {
        "values": [0, 16, 32, 64],
        "type": "int",
        "location": "env_training",
    },
    "langevin_num_iterations": {
        "values": [0, 10, 25, 50, 100],
        "type": "int",
        "location": "env_training",
    },
    # Langevin refinement hyperparameters. These override the per-env
    # env_model.langevin_config defaults at both training and inference time.
    "langevin_lr_init": {
        "values": [0.005, 0.01, 0.02, 0.05, 0.1],
        "type": "float",
        "location": "env_training",
    },
    "langevin_lr_final": {
        "values": [1e-6, 1e-5, 1e-4],
        "type": "float",
        "location": "env_training",
    },
    "langevin_noise_scale": {
        "values": [0.0, 0.05, 0.1, 0.3, 1.0],
        "type": "float",
        "location": "env_training",
    },
    "langevin_delta_clip": {
        "values": [0.01, 0.02, 0.05, 0.1],
        "type": "float",
        "location": "env_training",
    },
    "langevin_decay_power": {
        "values": [1.0, 2.0, 4.0],
        "type": "float",
        "location": "env_training",
    },
    # Training-chain fidelity switches. False/0 preserve legacy Q3C; True
    # plus a 0.05 buffer matches the official IBC update mechanics.
    "langevin_noise_via_stepsize": {
        "values": [True, False],
        "type": "bool",
        "location": "env_training",
    },
    "langevin_boundary_buffer": {
        "values": [0.0, 0.05],
        "type": "float",
        "location": "env_training",
    },
    # IBC negative mixture (Florence et al., 2021).
    "num_uniform_negatives": {
        "values": [0, 16, 32, 64, 128],
        "type": "int",
        "location": "env_training",
    },
    "num_langevin_negatives": {
        "values": [0, 16, 32, 64],
        "type": "int",
        "location": "env_training",
    },
    # Langevin negative starting distribution. "uniform" = paper-faithful;
    # "cps" = start from CP cloud, find Q-peaks in CP neighbourhoods.
    "langevin_init_kind": {
        "values": ["uniform", "cps"],
        "type": "str",
        "location": "env_training",
    },
    "langevin_init_jitter": {
        "values": [0.0, 0.01, 0.03, 0.05, 0.1],
        "type": "float",
        "location": "env_training",
    },
    # Noisy-expert hard negatives (estimator-only). σ linearly interpolates
    # from sigma_start to sigma_final over training.
    "noisy_expert_count": {
        "values": [0, 8, 16, 32, 64],
        "type": "int",
        "location": "training_shared",
    },
    "noisy_expert_sigma_start": {
        "values": [0.05, 0.1, 0.2, 0.3, 0.5],
        "type": "float",
        "location": "training_shared",
    },
    "noisy_expert_sigma_final": {
        "values": [0.005, 0.01, 0.02, 0.05],
        "type": "float",
        "location": "training_shared",
    },
    # IBC gradient penalty (Florence et al., 2021, App. B).
    "gradient_penalty_weight": {
        "values": [0.0, 0.1, 1.0, 10.0],
        "type": "float",
        "location": "training_shared",
    },
    "gradient_penalty_margin": {
        # 0.05–0.2 is the firing range for our 2x256 MLP on [0,1]^8;
        # 0.5–2.0 stays in line with the IBC paper at larger scales.
        "values": [0.05, 0.1, 0.2, 0.5, 1.0, 2.0],
        "type": "float",
        "location": "training_shared",
    },
    "gradient_penalty_form": {
        # "hinge"  = IBC-faithful one-sided: penalty = max(0, |grad|-margin)^2
        # "target" = WGAN-GP two-sided:      penalty = (|grad|-margin)^2
        "values": ["hinge", "target"],
        "type": "str",
        "location": "training_shared",
    },
    "gradient_penalty_norm": {
        # Official IBC uses linf; l2 is retained for old Q3C compatibility.
        "values": ["l2", "linf"],
        "type": "str",
        "location": "training_shared",
    },
    "ema_decay": {
        "values": [0.0, 0.99, 0.999],
        "type": "float",
        "location": "env_training",
    },
    # Best-checkpoint selection: keep max-reward weights via periodic env-eval.
    "best_ckpt": {
        "values": [False, True],
        "type": "bool",
        "location": "env_training",
    },
    "best_ckpt_eval_interval": {
        "values": [10000, 20000],
        "type": "int",
        "location": "env_training",
    },
    "best_ckpt_eval_seeds": {
        "values": [20, 50],
        "type": "int",
        "location": "env_training",
    },
    # Stochastic CP selection at eval: "argmax" or "sample" from softmax(Q/temp).
    "cp_selection": {
        "values": ["argmax", "sample"],
        "type": "str",
        "location": "env_training",
    },
    "cp_selection_temperature": {
        "values": [0.5, 1.0, 2.0],
        "type": "float",
        "location": "env_training",
    },
    # Rescale Q within the cloud before the softmax. Raw Q magnitude drifts with
    # the state, so one temperature is near-greedy on some frames and
    # near-uniform on others; monotone, so it cannot change an argmax.
    "cp_score_norm": {
        "values": ["none", "zscore", "rank"],
        "type": "str",
        "location": "env_training",
    },
    # Deterministic seed — searchable so we can run reps with seed=0,1,2,...
    "trial_seed": {
        "values": [0, 1, 2, 3, 4],
        "type": "int",
        "location": "env_training",
    },
    # ── dpq3c (dpq3c_training.py) ────────────────────────────────────────────
    # apply_params_to_config SILENTLY DROPS any key missing from SEARCH_SPACE,
    # so every knob a dpq3c batch passes through --fixed-params must be declared
    # here or it is ignored and the trial trains the default recipe.
    #
    # Actor — the diffusion process.
    "num_train_timesteps": {
        "values": [50, 100, 200],
        "type": "int",
        "location": "env_training",
    },
    "beta_schedule": {
        "values": ["cosine", "linear"],
        "type": "str",
        "location": "env_training",
    },
    "prediction_type": {
        "values": ["epsilon", "v"],
        "type": "str",
        "location": "env_training",
    },
    "time_emb_dim": {
        "values": [64, 128],
        "type": "int",
        "location": "env_training",
    },
    "denoiser_network_kind": {
        "values": ["mlp", "dense_resnet"],
        "type": "str",
        "location": "env_training",
    },
    "denoiser_width": {
        "values": [256, 512, 1024],
        "type": "int",
        "location": "env_training",
    },
    "denoiser_depth": {
        "values": [1, 2, 4],
        "type": "int",
        "location": "env_training",
    },
    "denoiser_use_spectral_norm": {
        "values": [False, True],
        "type": "bool",
        "location": "env_training",
    },
    # Critic — negatives drawn from the diffusion policy itself. This is the
    # alignment a separately-trained DP + Q3C pair cannot have; 0 disables it
    # and gives the "trained apart" control.
    "dp_negatives": {
        "values": [0, 16, 64],
        "type": "int",
        "location": "env_training",
    },
    "dp_negative_iters": {
        "values": [2, 4, 8],
        "type": "int",
        "location": "env_training",
    },
    "dp_negative_method": {
        "values": ["ddim", "ddpm"],
        "type": "str",
        "location": "env_training",
    },
    "dp_negative_warmup_steps": {
        "values": [0, 5000, 20000],
        "type": "int",
        "location": "env_training",
    },
    # Critic — objective terms. progress_weight is the reward-free absolute-scale
    # anchor (Monte-Carlo time-to-go); without it InfoNCE and the margin only
    # constrain score differences and Q stays a ranker.
    "progress_weight": {
        "values": [0.0, 0.1, 0.5],
        "type": "float",
        "location": "env_training",
    },
    "margin_weight": {
        "values": [0.0, 0.5, 1.0],
        "type": "float",
        "location": "env_training",
    },
    "margin": {
        "values": [0.05, 0.1, 0.5],
        "type": "float",
        "location": "env_training",
    },
    # Actor <- critic feedback (training-time analogue of deploy --q-guidance).
    "q_actor_weight": {
        "values": [0.0, 0.01, 0.05],
        "type": "float",
        "location": "env_training",
    },
    # Eval-time sampler for the DiffusionControlPointGenerator. The cloud SIZE
    # at eval is `control_points` (env_model), shared with q3c.
    "inference_dp_iters": {
        "values": [5, 10, 25],
        "type": "int",
        "location": "env_training",
    },
    "inference_dp_method": {
        "values": ["ddim", "ddpm"],
        "type": "str",
        "location": "env_training",
    },
    "inference_dp_eta": {
        "values": [0.0, 0.5, 1.0],
        "type": "float",
        "location": "env_training",
    },
    # Evaluation episodes per trial. A top-level env key (evaluate_q3c reads
    # env_config["num_eval_seeds"]). Searchable so a batch can buy tighter
    # error bars when the effect being measured is smaller than the noise at
    # the env's default count.
    "num_eval_seeds": {
        "values": [50, 100, 200, 500],
        "type": "int",
        "location": "env",
    },
}


def effective_langevin_config(env_config: dict) -> dict:
    """Merge env_training langevin_* overrides onto env_model.langevin_config defaults.

    Returns a dict keyed by the native sample_langevin arg names (lr_init, etc.),
    so callers in both training and evaluation share one source of truth.
    """
    base = dict(env_config.get("model", {}).get("langevin_config", {}))
    training = env_config.get("training", {})
    overrides = {
        "num_iterations": "langevin_num_iterations",
        "lr_init": "langevin_lr_init",
        "lr_final": "langevin_lr_final",
        "noise_scale": "langevin_noise_scale",
        "delta_action_clip": "langevin_delta_clip",
        "polynomial_decay_power": "langevin_decay_power",
    }
    for native_key, training_key in overrides.items():
        if training_key in training:
            base[native_key] = training[training_key]
    return base


def effective_inference_langevin_config(env_config: dict) -> dict:
    """Inference-time Langevin config: training defaults overridden by inference_* keys.

    Lets callers run aggressive paper-faithful Langevin during training
    (for hard negatives) while using a gentler inference chain to refine
    actions on Q3C's narrow-trained Q surface. Falls back to training values
    for any key not overridden, so an empty inference_* set = same as training.
    """
    cfg = effective_langevin_config(env_config)
    training = env_config.get("training", {})
    overrides = {
        "lr_init": "inference_langevin_lr_init",
        "lr_final": "inference_langevin_lr_final",
        "noise_scale": "inference_langevin_noise_scale",
        "delta_action_clip": "inference_langevin_delta_clip",
        "polynomial_decay_power": "inference_langevin_decay_power",
    }
    for native_key, inf_key in overrides.items():
        if inf_key in training:
            cfg[native_key] = training[inf_key]
    return cfg


# ─── Config I/O ──────────────────────────────────────────────────────────────

def load_config() -> dict:
    """Read the on-disk default config. Never mutated by parallel trials."""
    with open(CONFIG_PATH, "r") as f:
        return json.load(f)


# ─── Trials I/O ──────────────────────────────────────────────────────────────

def _results_dir(script_name: str, active_env: str | None = None) -> Path:
    """Resolve the trials directory for a (script, env) pair.

    When `active_env` is provided (the CLI override path) we use it directly
    and DO NOT touch config.json. This avoids the long-standing race where
    config.json was flipped between trial submit and trial record, which
    caused pushing trials to be logged into the particle folder and vice
    versa. The disk-read fallback is kept so old call paths (`--analyze`
    without --active-env) still work.
    """
    if active_env is None:
        config = load_config()
        active_env = config.get("active_env", "particle")
        env_cfg = config.get("environments", {}).get(active_env, {})
    else:
        # Still need env_cfg to detect particle n_dim partitioning; read from
        # disk but use the override for the env selection.
        config = load_config()
        env_cfg = config.get("environments", {}).get(active_env, {})

    # Env → results-subpath mapping. D4RL-family envs go under d4rl/<env>
    # so they can be grouped together as the codebase grows (kitchen, hammer,
    # door, etc. all share the AdroitHand-like protocol).
    _ENV_PATH_MAP: dict[str, str] = {
        "pen": "d4rl/pen",
        "door": "d4rl/door",
        "kitchen": "d4rl/kitchen",
    }
    env_subpath = _ENV_PATH_MAP.get(active_env, active_env)
    results_dir = RESULTS_BASE_DIR / Path(script_name).stem / env_subpath

    # For particle experiments, partition trials by n_dim to avoid mixing runs.
    if "n_dim" in env_cfg:
        results_dir = results_dir / str(env_cfg["n_dim"])

    return results_dir


def _trials_path(script_name: str, active_env: str | None = None) -> Path:
    return _results_dir(script_name, active_env=active_env) / "trials.jsonl"


def load_trials(script_name: str, active_env: str | None = None) -> list[dict]:
    path = _trials_path(script_name, active_env=active_env)
    if not path.exists():
        return []
    trials = []
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if line:
                trials.append(json.loads(line))
    return trials


def append_trial(script_name: str, record: dict, active_env: str | None = None) -> int:
    """Atomically assign a monotonically-increasing trial_id and append the record.

    Uses fcntl.flock for an exclusive lock over the jsonl for the short read-max +
    write-line section. Safe under parallel sbatch submissions. Returns the id.
    """
    path = _trials_path(script_name, active_env=active_env)
    path.parent.mkdir(parents=True, exist_ok=True)

    with open(path, "a") as f:
        fcntl.flock(f.fileno(), fcntl.LOCK_EX)
        try:
            max_id = 0
            try:
                with open(path, "r") as rf:
                    for line in rf:
                        line = line.strip()
                        if not line:
                            continue
                        try:
                            max_id = max(max_id, int(json.loads(line).get("trial_id", 0)))
                        except (json.JSONDecodeError, ValueError):
                            continue
            except FileNotFoundError:
                pass
            trial_id = max_id + 1
            record["trial_id"] = trial_id
            f.write(json.dumps(record, sort_keys=True, default=str) + "\n")
            f.flush()
            os.fsync(f.fileno())
        finally:
            fcntl.flock(f.fileno(), fcntl.LOCK_UN)
    return trial_id


# ─── Hyperparameter detection ────────────────────────────────────────────────

# Params consumed at evaluation time only (never referenced by training scripts).
# They must still appear in the search so they get tuned.
INFERENCE_ONLY_PARAMS: set[str] = {
    "action_execute_horizon",
    "inference_langevin_iterations",
    "inference_langevin_lr_init",
    "inference_langevin_lr_final",
    "inference_langevin_decay_power",
    "inference_langevin_delta_clip",
    "inference_langevin_noise_scale",
    "inference_dfo_iterations",
    "inference_dfo_iteration_std",
    "inference_dfo_iteration_std_decay",
    "inference_dfo_num_uniform",
    # Evaluation-only for BOTH algorithms: these change how a saved checkpoint
    # is queried, never what was trained, so reeval_trials.py may override them.
    "cp_selection",
    "cp_selection_temperature",
    "cp_score_norm",
    # dpq3c only (q3c ignores them): the sampler that draws the candidate cloud.
    "inference_dp_iters",
    "inference_dp_method",
    "inference_dp_eta",
}


def detect_script_params(script_path: Path) -> list[str]:
    """Scan training script source to find which search-space params it reads."""
    with open(script_path, "r") as f:
        source = f.read()
    detected = []
    for param_name in SEARCH_SPACE:
        if param_name in INFERENCE_ONLY_PARAMS:
            detected.append(param_name)
            continue
        if re.search(rf'["\']({re.escape(param_name)})["\']', source):
            detected.append(param_name)
    return detected


def get_baseline_params(config: dict, detected_params: list[str]) -> dict:
    """Read current config values for the detected params."""
    active_env = config.get("active_env", "particle")
    env_training = config["environments"][active_env].get("training", {})
    env_model = config["environments"][active_env].get("model", {})
    training_shared = config.get("training_shared", {})

    baseline: dict = {}
    for param in detected_params:
        space = SEARCH_SPACE[param]
        if space["location"] == "env_model":
            val = env_model.get(param)
        elif space["location"] == "env_training":
            val = env_training.get(param, training_shared.get(param))
        elif space["location"] == "env":
            val = config["environments"][active_env].get(param)
        else:
            val = training_shared.get(param)
        if val is not None:
            baseline[param] = val
    return baseline


def apply_params_to_config(config: dict, params: dict) -> dict:
    """Return a deep copy of *config* with hyperparameter overrides applied."""
    config = deepcopy(config)
    active_env = config.get("active_env", "particle")
    for param, value in params.items():
        if param not in SEARCH_SPACE:
            continue
        space = SEARCH_SPACE[param]
        if space["location"] == "env_model":
            config["environments"][active_env].setdefault("model", {})[param] = value
        elif space["location"] == "env_training":
            config["environments"][active_env].setdefault("training", {})[param] = value
        elif space["location"] == "env":
            # Top-level env key (e.g. frame_stack) — read by dataset, model
            # build, and eval alike via env_config.
            config["environments"][active_env][param] = value
        else:
            config.setdefault("training_shared", {})[param] = value
    return config


def set_run_checkpoint_dir(config: dict, run_id: str) -> str:
    """Point model_save_dir to a per-run directory (unique even under parallel runs)."""
    run_dir = str(CHECKPOINTS_BASE_DIR / f"run_{run_id}")
    config.setdefault("training_shared", {})["model_save_dir"] = run_dir
    return run_dir


# ─── Training subprocess ─────────────────────────────────────────────────────

def run_training(
    script_path: Path,
    timeout: int | None = None,
    env_extras: dict[str, str] | None = None,
) -> tuple[bool, str, float]:
    """Run a training script as a subprocess, streaming output live.

    Returns (success, captured_stdout, duration_seconds).
    `env_extras` is layered on top of the inherited env (e.g., Q3C_CONFIG_PATH).
    """
    start = time.time()
    stdout_lines: list[str] = []
    env = {**os.environ, "WANDB_MODE": "disabled", "PYTHONUNBUFFERED": "1"}
    if env_extras:
        env.update(env_extras)

    try:
        proc = subprocess.Popen(
            [sys.executable, str(script_path)],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            cwd=str(ROOT_DIR),
            env=env,
        )
        assert proc.stdout is not None
        for line in proc.stdout:
            sys.stdout.write(f"  | {line}")
            sys.stdout.flush()
            stdout_lines.append(line)
        proc.wait(timeout=timeout)
        duration = time.time() - start
        return proc.returncode == 0, "".join(stdout_lines), duration
    except subprocess.TimeoutExpired:
        proc.kill()  # type: ignore[possibly-undefined]
        duration = time.time() - start
        return False, "".join(stdout_lines) + "\n[TIMED OUT]", duration
    except Exception as exc:
        duration = time.time() - start
        return False, "".join(stdout_lines) + f"\n[ERROR: {exc}]", duration


def extract_final_metrics(stdout: str) -> dict:
    """Parse the last log line of training stdout for loss/accuracy."""
    metrics: dict = {}
    for line in reversed(stdout.strip().splitlines()):
        m_total = re.search(r"Total:\s*([\d.]+)", line)
        m_loss = re.search(r"Loss:\s*([\d.]+)", line)
        m_acc = re.search(r"Acc:\s*([\d.]+)", line)
        if m_total:
            metrics["final_train_loss"] = float(m_total.group(1))
        elif m_loss:
            metrics["final_train_loss"] = float(m_loss.group(1))
        if m_acc:
            metrics["final_train_accuracy"] = float(m_acc.group(1))
        if metrics:
            break
    return metrics


# ─── Evaluation ──────────────────────────────────────────────────────────────

def _build_dpq3c_generator(weights_path, env_config, norm_stats, action_dim,
                           control_points, action_bounds, device, *, pixel,
                           in_channels=None, cond_dim=0, state_dim=None,
                           encoder_target_height=180, encoder_target_width=240,
                           encoder_feature_dim=256, encoder_kind="conv_maxpool",
                           encoder_num_kp=64, encoder_norm_kind="bn",
                           encoder_per_camera=False):
    """Load a dpq3c denoiser and expose it as a control-point generator.

    dpq3c swaps only WHERE the candidate cloud comes from, so the diffusion
    policy is wrapped in the `cp_gen(state) -> (B, N, A)` signature the whole
    evaluation stack already speaks. Every simulation class, plus the CP-DFO and
    Langevin refinement wrappers below, then works on a dpq3c checkpoint
    unchanged.

    norm_stats is the authority on the sampler that was trained (it is what the
    trainer actually used), falling back to the config's training block.
    """
    from utils.diffusion import (build_denoiser, build_diffusion,
                                 build_dpq3c_denoiser, resolve_dp_params,
                                 DiffusionControlPointGenerator)

    dp = resolve_dp_params(env_config)
    for key in ("num_train_timesteps", "beta_schedule", "prediction_type",
                "time_emb_dim", "denoiser_network_kind", "denoiser_width",
                "denoiser_depth", "denoiser_use_spectral_norm"):
        if key in norm_stats:
            dp[key] = norm_stats[key]

    if pixel:
        denoiser = build_dpq3c_denoiser(
            action_dim, in_channels, dp, cond_dim=cond_dim,
            encoder_target_height=encoder_target_height,
            encoder_target_width=encoder_target_width,
            encoder_feature_dim=encoder_feature_dim,
            encoder_kind=encoder_kind,
            # Weights come from the state_dict; never fetch ImageNet on a
            # compute node that may have no network.
            encoder_pretrained=False,
            encoder_num_kp=encoder_num_kp,
            encoder_norm_kind=encoder_norm_kind,
            encoder_per_camera=encoder_per_camera,
            device=device)
    else:
        denoiser = build_denoiser(state_dim, action_dim, dp, device=device)
    denoiser.load_state_dict(
        torch.load(weights_path, map_location=device, weights_only=True))
    denoiser.to(device).eval()

    et = env_config.get("training", {})
    # Eval sampler knobs. Default the step count to the first entry the trainer
    # recorded in ddim_eval_steps, so a trial evaluates at the schedule it was
    # set up for rather than a hardcoded guess.
    ddim_default = norm_stats.get("ddim_eval_steps", dp.get("ddim_eval_steps", [10]))
    num_steps = int(et.get("inference_dp_iters",
                           (ddim_default[0] if ddim_default else 10)))
    method = str(et.get("inference_dp_method", "ddim"))
    eta = float(et.get("inference_dp_eta",
                       norm_stats.get("ddim_eta", dp.get("ddim_eta", 0.0))))
    print(f"dpq3c: diffusion CP generator — {control_points} candidates via "
          f"{method} x{num_steps} (eta={eta}), action_dim={action_dim}")
    gen = DiffusionControlPointGenerator(
        denoiser, build_diffusion(dp, device, action_bounds),
        control_points, action_dim, num_steps=num_steps, eta=eta, method=method)
    return gen.to(device).eval()


def evaluate_q3c(checkpoint_dir: str, config: dict) -> dict:
    """Load Q3C models from *checkpoint_dir* and measure success rate."""
    from utils.models import ControlPointGenerator, QEstimator
    from utils.sampling import sample_langevin

    active_env = config.get("active_env", "particle")
    env_config = config["environments"][active_env]
    sim_config = config.get("simulation", {})

    # Pick the right simulation class. `pushing` uses the vendored IBC env
    # (PyBullet + gym), which lives behind the `pushing` optional-extras and
    # is NOT installed on SLURM nodes running particle/dummy/pen trials. Keep
    # the import lazy so a particle SLURM job never touches pushing deps.
    if active_env == "pushing":
        from simulations.pushing_simulation import PushingSimulation
        SimulationCls = PushingSimulation
    elif active_env == "pushing_multi":
        from simulations.pushing_multi_simulation import PushingMultiSimulation
        SimulationCls = PushingMultiSimulation
    elif active_env == "pushing_pixels":
        from simulations.pushing_pixels_simulation import PushingPixelsSimulation
        SimulationCls = PushingPixelsSimulation
    elif active_env == "pen":
        from simulations.pen_human_v2_simulation import PenHumanV2Simulation
        SimulationCls = PenHumanV2Simulation
    elif active_env == "door":
        from simulations.door_human_v2_simulation import DoorHumanV2Simulation
        SimulationCls = DoorHumanV2Simulation
    elif active_env == "kitchen":
        from simulations.kitchen_simulation import KitchenSimulation
        SimulationCls = KitchenSimulation
    elif active_env == "libero_goal":
        from simulations.libero_goal_simulation import LiberoGoalSimulation
        SimulationCls = LiberoGoalSimulation
    elif active_env == "libero_goal_pixels":
        from simulations.libero_goal_pixels_simulation import LiberoGoalPixelsSimulation
        SimulationCls = LiberoGoalPixelsSimulation
    else:
        from simulations.particle_simulation import ParticleSimulation
        SimulationCls = ParticleSimulation

    state_dim = env_config["state_dim"]
    action_dim = env_config["action_dim"]
    frame_stack = env_config.get("frame_stack", 1)
    action_bounds = tuple(env_config.get("action_bounds", [0, 1]))
    n_dim = env_config.get("n_dim", 2)
    em = env_config["model"]
    control_points = em["control_points"]
    num_hidden_layers = em["num_hidden_layers"]
    num_neurons = em["num_neurons"]
    use_spectral_norm = em.get("use_spectral_norm", False)
    hidden_dims = [num_neurons] * num_hidden_layers
    # Per-net architecture (mirrors combinedv2_cpascounter_training.py).
    q_network_kind = em.get("q_network_kind", "mlp")
    q_width = em.get("q_width", num_neurons)
    q_depth = em.get("q_depth", num_hidden_layers)
    q_use_spectral_norm = em.get("q_use_spectral_norm", use_spectral_norm)
    cp_network_kind = em.get("cp_network_kind", "mlp")
    cp_width = em.get("cp_width", num_neurons)
    cp_depth = em.get("cp_depth", num_hidden_layers)
    cp_use_spectral_norm = em.get("cp_use_spectral_norm", False)
    # Per-env override wins over the shared simulation.max_episode_steps.
    # Pushing needs 100 (IBC paper BlockPush-v0); particle uses the global 50.
    max_episode_steps = env_config.get(
        "max_episode_steps", sim_config.get("max_episode_steps", 50)
    )
    # IBC Table 3 reports simulated pushing over 100 evaluation episodes per
    # training seed. Keep this env-scoped so particle's established eval count
    # does not change.
    num_seeds = int(
        env_config.get(
            "num_eval_seeds",
            sim_config.get("num_seeds", len(sim_config.get("default_seeds", [0]))),
        )
    )
    if num_seeds <= 0:
        raise ValueError("simulation.num_seeds must be >= 1")
    seeds = list(range(num_seeds))

    inference_langevin_iterations = int(
        env_config.get("training", {}).get("inference_langevin_iterations", 0)
    )
    # CP-DFO refinement (Q3CIBC-specific, no IBC analog). Takes precedence
    # over inference Langevin when > 0, so a trial can opt into either path
    # without changing the rest of the recipe.
    inference_dfo_iterations = int(
        env_config.get("training", {}).get("inference_dfo_iterations", 0)
    )
    inference_dfo_iteration_std = float(
        env_config.get("training", {}).get("inference_dfo_iteration_std", 0.1)
    )
    inference_dfo_iteration_std_decay = float(
        env_config.get("training", {}).get("inference_dfo_iteration_std_decay", 0.7)
    )
    inference_dfo_num_uniform = int(
        env_config.get("training", {}).get("inference_dfo_num_uniform", 0)
    )
    # Effective langevin hyperparams for INFERENCE chain. Starts from training
    # Langevin config (env_model.langevin_config + langevin_* training overrides),
    # then applies any inference_langevin_* overrides on top. Lets eval use
    # gentler step sizes than training while keeping training paper-faithful.
    langevin_cfg = effective_inference_langevin_config(env_config)
    # Official-IBC-faithful chain extras (audit fixes; see ibc-repro-fixes).
    inference_langevin_noise_via_stepsize = bool(
        env_config.get("training", {}).get("inference_langevin_noise_via_stepsize", False)
    )
    inference_langevin_again_iterations = int(
        env_config.get("training", {}).get("inference_langevin_again_iterations", 0)
    )
    inference_langevin_again_noise_scale = float(
        env_config.get("training", {}).get("inference_langevin_again_noise_scale", 0.5)
    )
    inference_langevin_top_k = int(
        env_config.get("training", {}).get("inference_langevin_top_k", 0)
    )

    # dpq3c (dpq3c_training.py) writes a diffusion denoiser where q3c writes a
    # control-point generator. Everything else about the evaluation is
    # unchanged, so detect it by which proposal file is on disk and swap only
    # the generator build below.
    dpq3c_raw_path = os.path.join(checkpoint_dir, "denoiser.pt")
    dpq3c_ema_path = os.path.join(checkpoint_dir, "denoiser_ema.pt")
    is_dpq3c = os.path.exists(dpq3c_raw_path) or os.path.exists(dpq3c_ema_path)

    if is_dpq3c:
        cp_raw_path, cp_ema_path = dpq3c_raw_path, dpq3c_ema_path
    else:
        cp_raw_path = os.path.join(checkpoint_dir, "control_point_generator.pt")
        cp_ema_path = os.path.join(checkpoint_dir, "control_point_generator_ema.pt")
    q_raw_path = os.path.join(checkpoint_dir, "q_estimator.pt")
    q_ema_path = os.path.join(checkpoint_dir, "q_estimator_ema.pt")
    eval_ema_decay = float(env_config.get("training", {}).get("ema_decay", 0.0))
    use_ema = (
        eval_ema_decay > 0.0
        and os.path.exists(cp_ema_path)
        and os.path.exists(q_ema_path)
    )
    cp_path = cp_ema_path if use_ema else cp_raw_path
    q_path = q_ema_path if use_ema else q_raw_path
    norm_stats_path = os.path.join(checkpoint_dir, "norm_stats.pt")

    if not os.path.exists(cp_path) or not os.path.exists(q_path):
        return {
            "success_rate": 0.0,
            "avg_reward": 0.0,
            "error": f"Checkpoints not found in {checkpoint_dir}",
        }
    if eval_ema_decay > 0.0 and not use_ema:
        print(
            f"Warning: EMA requested (decay={eval_ema_decay}) but paired EMA "
            "checkpoints are missing; evaluating raw weights."
        )
    elif use_ema:
        print(f"Evaluating Q3C EMA weights (decay={eval_ema_decay}).")

    # Presence of norm_stats.pt = ibc_with_cps (actions normalized to [0,1]
    # before the Q estimator sees them).
    norm_stats = None
    if os.path.exists(norm_stats_path):
        norm_stats = torch.load(norm_stats_path, weights_only=False)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    if active_env in ("pushing_pixels", "libero_goal_pixels"):
        from utils.models import PixelControlPointGenerator, PixelQEstimator
        # in_channels MUST come from the checkpoint's norm_stats when present:
        # the config's static state_dim assumes frame_stack=1 (6 channels), but
        # fs=2 trials trained with 12 — rebuilding from config broke every
        # frame_stack=2 eval in Dstandardlibero (head 1432 vs 920 mismatch).
        in_channels = int((norm_stats or {}).get("in_channels", state_dim[0]))
        enc_h = int(env_config.get("encoder_target_height", 180))
        enc_w = int(env_config.get("encoder_target_width", 240))
        value_width = int(em.get("value_width", 1024))
        value_num_blocks = int(em.get("value_num_blocks", 1))
        # libero_goal_pixels conditions on proprio+goal (cond_dim from norm_stats).
        cond_dim = int(norm_stats["cond_dim"]) if (norm_stats and "cond_dim" in norm_stats) else 0
        # Encoder architecture: read from norm_stats (what training used), falling
        # back to the config model block for older checkpoints.
        _ns = norm_stats or {}
        encoder_kind = _ns.get("encoder_kind", em.get("encoder_kind", "conv_maxpool"))
        # Always rebuild with pretrained=False: the trained weights come from the
        # checkpoint's state_dict anyway, and compute nodes may have no network
        # to download ImageNet weights (they'd be overwritten regardless).
        encoder_pretrained = False
        encoder_num_kp = int(_ns.get("encoder_num_kp", em.get("encoder_num_kp", 64)))
        encoder_norm_kind = _ns.get("encoder_norm_kind", em.get("encoder_norm_kind", "bn"))
        encoder_per_camera = bool(_ns.get("encoder_per_camera", em.get("encoder_per_camera", False)))
        cond_fusion = _ns.get("cond_fusion", em.get("cond_fusion", "concat"))
        goal_dim = int(_ns.get("goal_emb_dim", 0))
        # Action chunking: model output = action_dim * K per CP.
        action_chunk = int(_ns.get("action_chunk", 1) or 1)
        action_dim_eff = action_dim * action_chunk
        if is_dpq3c:
            cp_gen = _build_dpq3c_generator(
                cp_path, env_config, norm_stats or {}, action_dim_eff,
                control_points, action_bounds, device,
                pixel=True, in_channels=in_channels, cond_dim=cond_dim,
                encoder_target_height=enc_h, encoder_target_width=enc_w,
                encoder_feature_dim=int(_ns.get("encoder_feature_dim", 256)),
                encoder_kind=encoder_kind, encoder_num_kp=encoder_num_kp,
                encoder_norm_kind=encoder_norm_kind,
                encoder_per_camera=encoder_per_camera,
            )
        else:
            cp_gen = PixelControlPointGenerator(
                output_dim=action_dim_eff,
                control_points=control_points,
                hidden_dims=[cp_width] * cp_depth,
                action_bounds=action_bounds,
                network_kind=cp_network_kind,
                width=cp_width,
                depth=cp_depth,
                use_spectral_norm=cp_use_spectral_norm,
                in_channels=in_channels,
                encoder_target_height=enc_h,
                encoder_target_width=enc_w,
                cond_dim=cond_dim,
                encoder_kind=encoder_kind,
                encoder_pretrained=encoder_pretrained,
                encoder_num_kp=encoder_num_kp,
                encoder_norm_kind=encoder_norm_kind,
                encoder_per_camera=encoder_per_camera,
                cond_fusion=cond_fusion,
                goal_dim=goal_dim,
            )
            cp_gen.load_state_dict(torch.load(cp_path, map_location=device, weights_only=True))
            cp_gen.to(device).eval()

        q_est = PixelQEstimator(
            action_dim=action_dim_eff,
            in_channels=in_channels,
            encoder_target_height=enc_h,
            encoder_target_width=enc_w,
            value_width=value_width,
            value_num_blocks=value_num_blocks,
            cond_dim=cond_dim,
            encoder_kind=encoder_kind,
            encoder_pretrained=encoder_pretrained,
            encoder_num_kp=encoder_num_kp,
            encoder_norm_kind=encoder_norm_kind,
            encoder_per_camera=encoder_per_camera,
            cond_fusion=cond_fusion,
            goal_dim=goal_dim,
        )
        q_est.load_state_dict(torch.load(q_path, map_location=device, weights_only=True))
        q_est.to(device).eval()
    else:
        # libero_goal bakes the goal embedding into the state AFTER frame-stacking,
        # so its input dim is NOT state_dim*frame_stack. Same for kitchen when
        # kitchen_qpos_only shrank the obs below config's state_dim. Read the
        # exact length the dataset used straight from norm_stats.
        if active_env in ("libero_goal", "kitchen") and norm_stats is not None and "state_shape" in norm_stats:
            flat_input_dim = int(norm_stats["state_shape"])
        else:
            flat_input_dim = state_dim * frame_stack
        # Action chunking: the model was trained on K-step chunk targets, so
        # its output/action dim is action_dim*K (norm_stats carries K).
        flat_action_chunk = int((norm_stats or {}).get("action_chunk", 1) or 1)
        flat_action_dim = action_dim * flat_action_chunk
        if is_dpq3c:
            cp_gen = _build_dpq3c_generator(
                cp_path, env_config, norm_stats or {}, flat_action_dim,
                control_points, action_bounds, device,
                pixel=False, state_dim=flat_input_dim,
            )
        else:
            cp_gen = ControlPointGenerator(
                input_dim=flat_input_dim,
                output_dim=flat_action_dim,
                control_points=control_points,
                hidden_dims=[cp_width] * cp_depth,
                action_bounds=action_bounds,
                network_kind=cp_network_kind,
                width=cp_width,
                depth=cp_depth,
                use_spectral_norm=cp_use_spectral_norm,
            )
            cp_gen.load_state_dict(
                torch.load(cp_path, map_location=device, weights_only=True)
            )
            cp_gen.to(device).eval()

        q_est = QEstimator(
            state_dim=flat_input_dim,
            action_dim=flat_action_dim,
            hidden_dims=[q_width] * q_depth,
            use_spectral_norm=q_use_spectral_norm,
            network_kind=q_network_kind,
            width=q_width,
            depth=q_depth,
            resnet_final_activation=bool(em.get("q_resnet_final_activation", True)),
        )
        q_est.load_state_dict(
            torch.load(q_path, map_location=device, weights_only=True)
        )
        q_est.to(device).eval()

    # ── Pixel envs: dedicated late-fused DFO / Langevin refinement ────────
    # The flat-state wrappers below assume `obs.unsqueeze(1).expand(-1, N, -1)`
    # is cheap — that's true for vector obs, but for images it would re-encode
    # the (1, C, H, W) tensor N (DFO) or 100 (Langevin) times PER ENV STEP.
    # Instead we encode ONCE per step, cache the 256-D features, and run the
    # refinement inner loop against PixelQEstimator.score(features, actions).
    # This is what IBC's `late_fusion = True` config flag does upstream.
    if active_env == "pushing_pixels":
        if inference_dfo_iterations > 0:
            _dfo_iters = inference_dfo_iterations
            _dfo_std0 = inference_dfo_iteration_std
            _dfo_decay = inference_dfo_iteration_std_decay
            _dfo_n_uniform = inference_dfo_num_uniform

            class PixelDFORefinedSimulation(SimulationCls):
                """Pixel-aware CP-DFO refinement (encode once per env step).

                Same algorithm as DFORefinedSimulation below — initial pop = CP
                cloud (+ optional uniform safety samples); each iter resamples
                via category-ordered softmax(Q) and jitters — but the Q forward
                calls run against cached image features instead of re-encoding.
                """

                def select_action(self, observation, return_q_range: bool = False):
                    obs_tensor = self._obs_to_tensor(observation)  # (1, C, H, W) uint8

                    with torch.no_grad():
                        features = self.q_estimator.encode(obs_tensor)  # (1, F)
                        cps = self.control_point_generator(obs_tensor)  # (1, N_cp, A)

                        if _dfo_n_uniform > 0:
                            unif = torch.empty(
                                1, _dfo_n_uniform, cps.shape[-1], device=self.device
                            ).uniform_(float(action_bounds[0]), float(action_bounds[1]))
                            candidates = torch.cat([cps, unif], dim=1)
                        else:
                            candidates = cps.clone()

                        N = candidates.shape[1]
                        std = float(_dfo_std0)
                        for it in range(_dfo_iters):
                            log_probs = self.q_estimator.score(features, candidates).squeeze(-1)  # (1, N)
                            probs = torch.softmax(log_probs.squeeze(0), dim=-1)
                            idx = torch.multinomial(probs, N, replacement=True)
                            counts = torch.bincount(idx, minlength=N)
                            repeat_idx = torch.repeat_interleave(
                                torch.arange(N, device=self.device), counts
                            )
                            candidates = candidates[:, repeat_idx, :]
                            if it < _dfo_iters - 1:
                                candidates = candidates + torch.randn_like(candidates) * std
                                candidates = candidates.clamp(
                                    float(action_bounds[0]), float(action_bounds[1])
                                )
                                std *= _dfo_decay
                        # Re-score after the final reorder so argmax index aligns
                        # with the (reordered) candidates tensor — same fix as
                        # the flat-state DFORefinedSimulation.
                        final_log_probs = self.q_estimator.score(features, candidates).squeeze(-1)
                        sel = final_log_probs.argmax(dim=1)
                        action_normalized = candidates[0, sel[0], :].cpu().numpy()
                        q_range = (final_log_probs.min().item(), final_log_probs.max().item())

                    action = np.clip(action_normalized, action_bounds[0], action_bounds[1])
                    action = self._denormalize_action(action)
                    if return_q_range:
                        return action, q_range
                    return action

            sim_cls = PixelDFORefinedSimulation

        elif inference_langevin_iterations > 0:
            class PixelLangevinRefinedSimulation(SimulationCls):
                """Pixel-aware Langevin refinement (encode once per env step).

                Encodes the (1, C, H, W) image once into 256-D features, picks
                the argmax-Q CP as the starting action, then runs Langevin MCMC
                on actions against the cached features. The energy_function
                ignores `sample_langevin`'s expanded-obs argument and uses the
                closed-over `features` tensor instead — that's how we get the
                speedup vs the flat-state wrapper.
                """

                def select_action(self, observation, return_q_range: bool = False):
                    obs_tensor = self._obs_to_tensor(observation)  # (1, C, H, W) uint8

                    with torch.no_grad():
                        features = self.q_estimator.encode(obs_tensor)  # (1, F)
                        cps = self.control_point_generator(obs_tensor)  # (1, N_cp, A)
                        q_values = self.q_estimator.score(features, cps).squeeze(-1)  # (1, N)
                        best_idx = q_values.argmax(dim=1)
                        q_range = (q_values.min().item(), q_values.max().item())
                        best_cp = cps[0, best_idx[0], :].view(1, 1, -1).clone()  # (1, 1, A)

                    act_min_t = torch.full(
                        (cps.shape[-1],), float(action_bounds[0]), device=self.device
                    )
                    act_max_t = torch.full(
                        (cps.shape[-1],), float(action_bounds[1]), device=self.device
                    )

                    for p in self.q_estimator.parameters():
                        p.requires_grad_(False)

                    # Closed over `features` — the loop uses the cached encoding,
                    # not sample_langevin's expanded `obs_lv` arg (we just need
                    # to accept its signature).
                    def _neg_energy_fn(obs_lv, actions_lv):
                        return -self.q_estimator.score(features, actions_lv).squeeze(-1)

                    refined = sample_langevin(
                        energy_function=_neg_energy_fn,
                        observations=features,  # (1, F) — expanded internally, ignored by our fn
                        num_samples=1,
                        action_min=act_min_t,
                        action_max=act_max_t,
                        num_iterations=inference_langevin_iterations,
                        lr_init=float(langevin_cfg.get("lr_init", 0.1)),
                        lr_final=float(langevin_cfg.get("lr_final", 1e-5)),
                        polynomial_decay_power=float(
                            langevin_cfg.get("polynomial_decay_power", 2.0)
                        ),
                        delta_action_clip=float(
                            langevin_cfg.get("delta_action_clip", 0.1)
                        ),
                        noise_scale=float(langevin_cfg.get("noise_scale", 1.0)),
                        initial_actions=best_cp,
                        device=self.device,
                    )

                    for p in self.q_estimator.parameters():
                        p.requires_grad_(True)

                    action = refined[0, 0, :].cpu().numpy()
                    action = np.clip(action, action_bounds[0], action_bounds[1])
                    action = self._denormalize_action(action)
                    if return_q_range:
                        return action, q_range
                    return action

            sim_cls = PixelLangevinRefinedSimulation

        else:
            sim_cls = SimulationCls

    elif inference_dfo_iterations > 0:
        # ── CP-DFO refinement (Q3CIBC inference). Cheaper than Langevin: no
        # autograd, only N small-batch forward passes through the Q-net.
        # Initial population = CP cloud (+ optional N_uniform random
        # samples). Each iter: score → category-ordered resample with
        # softmax(Q) → small Gaussian jitter → clip. Mirrors IBC's
        # `iterative_dfo` mechanics (see `bench_inference.iterative_dfo_pass`)
        # but with a model-trained initial population.
        _dfo_iters = inference_dfo_iterations
        _dfo_std0 = inference_dfo_iteration_std
        _dfo_decay = inference_dfo_iteration_std_decay
        _dfo_n_uniform = inference_dfo_num_uniform

        class DFORefinedSimulation(SimulationCls):
            """Refines the CP cloud with iterative DFO before acting."""

            def select_action(self, observation, return_q_range: bool = False):
                obs_tensor = (
                    torch.tensor(observation, dtype=torch.float32)
                    .unsqueeze(0)
                    .to(self.device)
                )
                obs_tensor = self.obs_normalizer.normalize(obs_tensor)

                with torch.no_grad():
                    cps = self.control_point_generator(obs_tensor)  # (1, N_cp, D)

                    # Action normalization helper (matches the Langevin path).
                    def _norm(a):
                        if self._act_min_t is not None:
                            return (a - self._act_min_t) / self._act_rng_t
                        return a

                    # Mix in uniform safety samples if requested.
                    if _dfo_n_uniform > 0:
                        unif = torch.empty(
                            1, _dfo_n_uniform, cps.shape[-1], device=self.device
                        ).uniform_(float(action_bounds[0]), float(action_bounds[1]))
                        candidates = torch.cat([cps, unif], dim=1)
                    else:
                        candidates = cps.clone()

                    N = candidates.shape[1]
                    obs_expanded = obs_tensor.unsqueeze(1).expand(-1, N, -1)
                    std = float(_dfo_std0)
                    for it in range(_dfo_iters):
                        log_probs = self.q_estimator(obs_expanded, _norm(candidates)).squeeze(-1)
                        probs = torch.softmax(log_probs.squeeze(0), dim=-1)
                        # IBC-style category-ordered resample.
                        idx = torch.multinomial(probs, N, replacement=True)
                        counts = torch.bincount(idx, minlength=N)
                        repeat_idx = torch.repeat_interleave(
                            torch.arange(N, device=self.device), counts
                        )
                        candidates = candidates[:, repeat_idx, :]
                        if it < _dfo_iters - 1:
                            candidates = candidates + torch.randn_like(candidates) * std
                            candidates = candidates.clamp(
                                float(action_bounds[0]), float(action_bounds[1])
                            )
                            std *= _dfo_decay
                    # FIX: re-score AFTER the final reorder so argmax index lines
                    # up with the (now reordered) candidates tensor. The previous
                    # version used log_probs from the iteration's pre-reorder
                    # scoring and indexed into the reordered candidates, picking
                    # the wrong action when softmax mass was spread (the bug was
                    # masked on pushing where Q is sharply peaked).
                    final_log_probs = self.q_estimator(obs_expanded, _norm(candidates)).squeeze(-1)
                    sel = final_log_probs.argmax(dim=1)
                    action_normalized = candidates[0, sel[0], :].cpu().numpy()
                    q_range = (final_log_probs.min().item(), final_log_probs.max().item())

                action = np.clip(action_normalized, action_bounds[0], action_bounds[1])
                # _denormalize_action maps model-space (e.g., [-1, 1] for pushing)
                # back to env-action space. It's a no-op for particle where
                # _raw_act_min is None, but REQUIRED for pushing.
                action = self._denormalize_action(action)
                if return_q_range:
                    return action, q_range
                return action

        sim_cls = DFORefinedSimulation
    elif inference_langevin_iterations > 0:
        class LangevinRefinedParticleSimulation(SimulationCls):
            """Refines the CP CLOUD with official-IBC-faithful Langevin MCMC.

            Upgraded after the IBC audit (memory: ibc-repro-fixes; these chain
            details tripled our in-env IBC's kitchen score):
              - Chains start from ALL control points (model-trained proposals),
                not just the argmax CP — Q3C's analog of IBC's 512 uniform
                inits, but ~5x fewer and already near-modal, so short chains
                suffice (efficiency is the point of Q3C).
              - Optional noise_via_stepsize (official langevin_step): noise
                shrinks linearly with stepsize -> chain end is a pure polish.
              - Optional second chain at constant 1e-5 stepsize (official
                IbcPolicy.optimize_again).
              - Final action = argmax Q over the REFINED cloud (greedy, same
                as official GreedyPolicy mode).
            """

            def select_action(self, observation, return_q_range: bool = False):
                obs_tensor = (
                    torch.tensor(observation, dtype=torch.float32)
                    .unsqueeze(0)
                    .to(self.device)
                )
                obs_tensor = self.obs_normalizer.normalize(obs_tensor)

                with torch.no_grad():
                    cps = self.control_point_generator(obs_tensor)  # (1, N, D)
                    # top_k > 0: refine only the k best CPs by initial Q
                    # (k=1 = single chain from the argmax CP). 0 = whole cloud.
                    if inference_langevin_top_k > 0:
                        if self._act_min_t is not None:
                            cp_q_in = (cps - self._act_min_t) / self._act_rng_t
                        else:
                            cp_q_in = cps
                        obs_exp0 = obs_tensor.unsqueeze(1).expand(-1, cps.shape[1], -1)
                        q0 = self.q_estimator(obs_exp0, cp_q_in).squeeze(-1)  # (1, N)
                        k = min(inference_langevin_top_k, cps.shape[1])
                        top_idx = q0.topk(k, dim=1).indices  # (1, k)
                        cps = torch.gather(
                            cps, 1, top_idx.unsqueeze(-1).expand(-1, -1, cps.shape[-1])
                        )

                act_min_t = torch.full(
                    (cps.shape[-1],), float(action_bounds[0]), device=self.device
                )
                act_max_t = torch.full(
                    (cps.shape[-1],), float(action_bounds[1]), device=self.device
                )

                for p in self.q_estimator.parameters():
                    p.requires_grad_(False)

                _norm_min = self._act_min_t
                _norm_rng = self._act_rng_t

                def _neg_energy_fn(obs_lv, actions_lv):
                    if _norm_min is not None:
                        a_in = (actions_lv - _norm_min) / _norm_rng
                    else:
                        a_in = actions_lv
                    return -self.q_estimator(obs_lv, a_in).squeeze(-1)

                refined = sample_langevin(
                    energy_function=_neg_energy_fn,
                    observations=obs_tensor,
                    num_samples=cps.shape[1],
                    action_min=act_min_t,
                    action_max=act_max_t,
                    num_iterations=inference_langevin_iterations,
                    lr_init=float(langevin_cfg.get("lr_init", 0.1)),
                    lr_final=float(langevin_cfg.get("lr_final", 1e-5)),
                    polynomial_decay_power=float(
                        langevin_cfg.get("polynomial_decay_power", 2.0)
                    ),
                    delta_action_clip=float(
                        langevin_cfg.get("delta_action_clip", 0.1)
                    ),
                    noise_scale=float(langevin_cfg.get("noise_scale", 1.0)),
                    initial_actions=cps.clone(),
                    device=self.device,
                    noise_via_stepsize=inference_langevin_noise_via_stepsize,
                )
                if inference_langevin_again_iterations > 0:
                    refined = sample_langevin(
                        energy_function=_neg_energy_fn,
                        observations=obs_tensor,
                        num_samples=cps.shape[1],
                        action_min=act_min_t,
                        action_max=act_max_t,
                        num_iterations=inference_langevin_again_iterations,
                        lr_init=1e-5,
                        lr_final=1e-5,
                        polynomial_decay_power=float(
                            langevin_cfg.get("polynomial_decay_power", 2.0)
                        ),
                        delta_action_clip=float(
                            langevin_cfg.get("delta_action_clip", 0.1)
                        ),
                        noise_scale=inference_langevin_again_noise_scale,
                        initial_actions=refined,
                        device=self.device,
                        noise_via_stepsize=inference_langevin_noise_via_stepsize,
                    )

                for p in self.q_estimator.parameters():
                    p.requires_grad_(True)

                # Greedy over the refined cloud (re-scored post-refinement so
                # the argmax indexes the actions actually being returned).
                with torch.no_grad():
                    obs_expanded = obs_tensor.unsqueeze(1).expand(-1, refined.shape[1], -1)
                    if _norm_min is not None:
                        ref_for_q = (refined - _norm_min) / _norm_rng
                    else:
                        ref_for_q = refined
                    q_values = self.q_estimator(obs_expanded, ref_for_q).squeeze(-1)
                    best_idx = q_values.argmax(dim=1)
                    q_range = (q_values.min().item(), q_values.max().item())
                    action = refined[0, best_idx[0], :].cpu().numpy()

                action = np.clip(action, action_bounds[0], action_bounds[1])
                # Denormalize to the env's native action box when the
                # simulation declares a non-identity inverse (Pushing). For
                # ParticleSimulation this is a no-op (action_bounds = [0, 1]
                # is already the env action box).
                action = self._denormalize_action(action)
                if return_q_range:
                    return action, q_range
                return action

        sim_cls = LangevinRefinedParticleSimulation
    else:
        sim_cls = SimulationCls

    # PushingSimulation has no n_dim arg (1-block/1-target, fixed schema)
    # but has its own goal_dist_tolerance knob (IBC paper used 0.02).
    sim_kwargs: dict = dict(
        control_point_generator=cp_gen,
        q_estimator=q_est,
        device=device,
        max_episode_steps=max_episode_steps,
        render_mode=None,
        frame_stack=frame_stack,
        norm_stats=norm_stats,
    )
    if active_env == "pushing":
        sim_kwargs["goal_dist_tolerance"] = float(
            env_config.get("goal_dist_tolerance", 0.02)
        )
    elif active_env == "pushing_multi":
        # IBC class default for the multimodal variant is 0.04 (looser than
        # single-target because both blocks must satisfy the criterion).
        sim_kwargs["goal_dist_tolerance"] = float(
            env_config.get("goal_dist_tolerance", 0.04)
        )
    elif active_env == "pushing_pixels":
        # Single-target physics — same 0.02 tolerance as states variant.
        sim_kwargs["goal_dist_tolerance"] = float(
            env_config.get("goal_dist_tolerance", 0.02)
        )
    elif active_env in ("pen", "door", "kitchen"):
        # Adroit D4RL + FrankaKitchen — no goal_dist_tolerance / n_dim knobs.
        if active_env in ("pen", "kitchen"):
            # Receding horizon (execute R of the K-step chunk then replan).
            # 0 = execute all K (pure chunking). Eval-time-only knob.
            sim_kwargs["execute_horizon"] = int(
                env_config.get("training", {}).get("action_execute_horizon", 0)
            )
    elif active_env == "libero_goal":
        # Multi-task language-conditioned eval — obs schema + goal embeddings
        # come from norm_stats; no n_dim / tolerance knobs.
        pass
    elif active_env == "libero_goal_pixels":
        # Render eval grouped by task needs the eval-episode count to map seeds
        # task-major (avoids per-episode EGL env churn).
        sim_kwargs["num_eval_seeds"] = int(
            env_config.get("num_eval_seeds", len(seeds))
        )
    else:
        sim_kwargs["n_dim"] = n_dim
    sim = sim_cls(**sim_kwargs)

    all_results = []
    for seed in seeds:
        result = sim.run_episode(seed=seed)
        all_results.append(result)
    sim.close()

    def _finite(x: float) -> float | None:
        """JSON-safe: inf/nan → None so trials.jsonl stays strictly valid JSON."""
        xf = float(x)
        return xf if np.isfinite(xf) else None

    successes = [bool(r.get("success", False)) for r in all_results]
    rewards = [float(r.get("total_reward", 0.0)) for r in all_results]
    ep_lengths = [int(r.get("episode_length", 0)) for r in all_results]
    terminated_flags = [bool(r.get("terminated", False)) for r in all_results]

    if active_env == "kitchen":
        # FrankaKitchen headline metric = avg_tasks_completed (0..N), matching
        # IBC Table 2 (kitchen-complete = 3.37/4). success = solved ALL tasks.
        tasks_done = [int(r.get("tasks_completed", 0)) for r in all_results]
        return {
            "success_rate": float(np.mean(successes)),
            "success_rate_std": float(np.std(successes)),
            "avg_tasks_completed": float(np.mean(tasks_done)),
            "std_tasks_completed": float(np.std(tasks_done)),
            "median_tasks_completed": float(np.median(tasks_done)),
            "avg_reward": float(np.mean(rewards)),
            "std_reward": float(np.std(rewards)),
            "median_reward": float(np.median(rewards)),
            "avg_episode_length": float(np.mean(ep_lengths)),
            "num_seeds": len(seeds),
            "per_seed": [
                {
                    "seed": seeds[i],
                    "success": successes[i],
                    "tasks_completed": tasks_done[i],
                    "reward": rewards[i],
                    "episode_length": ep_lengths[i],
                    "terminated": terminated_flags[i],
                }
                for i in range(len(seeds))
            ],
        }

    if active_env in ("pen", "door", "libero_goal", "libero_goal_pixels"):
        # Adroit D4RL human tasks AND LIBERO-Goal report success_rate as the
        # headline metric (LIBERO's canonical number is per-suite success rate;
        # the env emits a binary success info bit). avg_reward is logged too but
        # is secondary for libero_goal. per_seed here is per-eval-episode; for
        # libero_goal the sim cycles tasks across episodes (see LiberoGoalSimulation).
        return {
            "success_rate": float(np.mean(successes)),
            "success_rate_std": float(np.std(successes)),
            "avg_reward": float(np.mean(rewards)),
            "std_reward": float(np.std(rewards)),
            "median_reward": float(np.median(rewards)),
            "avg_episode_length": float(np.mean(ep_lengths)),
            "num_seeds": len(seeds),
            "per_seed": [
                {
                    "seed": seeds[i],
                    "success": successes[i],
                    "reward": rewards[i],
                    "episode_length": ep_lengths[i],
                    "terminated": terminated_flags[i],
                }
                for i in range(len(seeds))
            ],
        }

    if active_env in ("pushing", "pushing_pixels"):
        # Single-target pushing (states OR pixels) — single goal, same metric
        # layout so the trial logs / analyzer queries stay uniform across the
        # two observation modalities.
        dists_target = [float(r.get("min_dist_to_target", np.inf)) for r in all_results]
        finite_target = [d for d in dists_target if np.isfinite(d)]
        return {
            "success_rate": float(np.mean(successes)),
            "success_rate_std": float(np.std(successes)),
            "avg_reward": float(np.mean(rewards)),
            "std_reward": float(np.std(rewards)),
            "median_reward": float(np.median(rewards)),
            "avg_min_dist_to_target": float(np.mean(finite_target)) if finite_target else None,
            "std_min_dist_to_target": float(np.std(finite_target)) if finite_target else None,
            "median_min_dist_to_target": float(np.median(finite_target)) if finite_target else None,
            "avg_episode_length": float(np.mean(ep_lengths)),
            "num_seeds": len(seeds),
            "per_seed": [
                {
                    "seed": seeds[i],
                    "success": successes[i],
                    "reward": rewards[i],
                    "min_dist_to_target": _finite(dists_target[i]),
                    "episode_length": ep_lengths[i],
                    "terminated": terminated_flags[i],
                }
                for i in range(len(seeds))
            ],
        }

    if active_env == "pushing_multi":
        # Multimodal pushing: 2 blocks, 2 targets. Each block is independently
        # assigned to its closest target (mirrors IBC's _get_reward). We log
        # per-block min distances + the mean so trial logs surface partial
        # progress when only one block lands.
        d_mean = [float(r.get("min_mean_dist_to_target", np.inf)) for r in all_results]
        d_b0 = [float(r.get("min_block0_dist_to_target", np.inf)) for r in all_results]
        d_b1 = [float(r.get("min_block1_dist_to_target", np.inf)) for r in all_results]
        finite_mean = [d for d in d_mean if np.isfinite(d)]
        finite_b0 = [d for d in d_b0 if np.isfinite(d)]
        finite_b1 = [d for d in d_b1 if np.isfinite(d)]
        return {
            "success_rate": float(np.mean(successes)),
            "success_rate_std": float(np.std(successes)),
            "avg_reward": float(np.mean(rewards)),
            "std_reward": float(np.std(rewards)),
            "median_reward": float(np.median(rewards)),
            "avg_min_mean_dist_to_target": float(np.mean(finite_mean)) if finite_mean else None,
            "std_min_mean_dist_to_target": float(np.std(finite_mean)) if finite_mean else None,
            "median_min_mean_dist_to_target": float(np.median(finite_mean)) if finite_mean else None,
            "avg_min_block0_dist_to_target": float(np.mean(finite_b0)) if finite_b0 else None,
            "avg_min_block1_dist_to_target": float(np.mean(finite_b1)) if finite_b1 else None,
            "avg_episode_length": float(np.mean(ep_lengths)),
            "num_seeds": len(seeds),
            "per_seed": [
                {
                    "seed": seeds[i],
                    "success": successes[i],
                    "reward": rewards[i],
                    "min_mean_dist_to_target": _finite(d_mean[i]),
                    "min_block0_dist_to_target": _finite(d_b0[i]),
                    "min_block1_dist_to_target": _finite(d_b1[i]),
                    "episode_length": ep_lengths[i],
                    "terminated": terminated_flags[i],
                }
                for i in range(len(seeds))
            ],
        }

    dists_first = [float(r.get("min_dist_to_first_goal", np.inf)) for r in all_results]
    dists_second = [float(r.get("min_dist_to_second_goal", np.inf)) for r in all_results]
    finite_first = [d for d in dists_first if np.isfinite(d)]
    finite_second = [d for d in dists_second if np.isfinite(d)]

    return {
        "success_rate": float(np.mean(successes)),
        "avg_reward": float(np.mean(rewards)),
        "std_reward": float(np.std(rewards)),
        "median_reward": float(np.median(rewards)),
        "avg_min_dist_first_goal": float(np.mean(finite_first)) if finite_first else None,
        "avg_min_dist_second_goal": float(np.mean(finite_second)) if finite_second else None,
        "median_min_dist_first_goal": float(np.median(finite_first)) if finite_first else None,
        "median_min_dist_second_goal": float(np.median(finite_second)) if finite_second else None,
        "avg_episode_length": float(np.mean(ep_lengths)),
        "num_seeds": len(seeds),
        "per_seed": [
            {
                "seed": seeds[i],
                "success": successes[i],
                "reward": rewards[i],
                "min_dist_first_goal": _finite(dists_first[i]),
                "min_dist_second_goal": _finite(dists_second[i]),
                "episode_length": ep_lengths[i],
                "terminated": terminated_flags[i],
            }
            for i in range(len(seeds))
        ],
    }


# ─── Auto-suggest strategy ───────────────────────────────────────────────────

def suggest_next_params(
    trials: list[dict],
    detected_params: list[str],
    baseline: dict,
) -> tuple[dict, str]:
    """Adaptively pick the next hyperparameter configuration to try.

    Strategy:
      Phase 1 (trial 1): Run the baseline.
      Phase 2 (trials 2..len(detected)+1): Vary one parameter at a time from
              the current best, choosing the parameter not yet explored.
      Phase 3 (afterwards): Combine best-per-param values with one random
              perturbation. Avoid exact duplicate configs.
    """
    if not trials:
        return baseline.copy(), "baseline"

    best_trial = max(trials, key=lambda t: t.get("success_rate", -1))
    best_params: dict = best_trial["params"]
    tried_signatures = {
        json.dumps(t["params"], sort_keys=True) for t in trials
    }

    # Phase 2: one-at-a-time exploration from best-so-far
    explored_params = set()
    for t in trials:
        for p in detected_params:
            if t["params"].get(p) != baseline.get(p):
                explored_params.add(p)

    unexplored = [p for p in detected_params if p not in explored_params and p in SEARCH_SPACE]
    if unexplored:
        param = unexplored[0]
        space = SEARCH_SPACE[param]
        current_val = best_params.get(param, baseline.get(param))
        candidates = [v for v in space["values"] if v != current_val]
        if candidates:
            new_val = random.choice(candidates)
            suggested = best_params.copy()
            suggested[param] = new_val
            return suggested, f"varying {param}={new_val} (from {current_val})"

    # Phase 3: combine best-per-param + perturbation
    combined: dict = {}
    for param in detected_params:
        if param not in SEARCH_SPACE:
            continue
        param_scores: dict = {}
        for t in trials:
            val = t["params"].get(param)
            if val is not None:
                sr = t.get("success_rate", 0)
                if val not in param_scores or sr > param_scores[val]:
                    param_scores[val] = sr
        if param_scores:
            combined[param] = max(param_scores, key=param_scores.get)  # type: ignore[arg-type]
        elif param in baseline:
            combined[param] = baseline[param]

    # Perturb one random parameter to avoid stagnation
    perturb_param = random.choice(detected_params)
    if perturb_param in SEARCH_SPACE:
        combined[perturb_param] = random.choice(SEARCH_SPACE[perturb_param]["values"])

    sig = json.dumps(combined, sort_keys=True)
    if sig in tried_signatures:
        for _ in range(20):
            p = random.choice(detected_params)
            if p in SEARCH_SPACE:
                combined[p] = random.choice(SEARCH_SPACE[p]["values"])
            sig = json.dumps(combined, sort_keys=True)
            if sig not in tried_signatures:
                break

    return combined, "combining best-per-param + perturbation"


# ─── Trial runner ─────────────────────────────────────────────────────────────

def run_single_trial(
    script_path: Path,
    params: dict,
    training_steps_override: int | None = None,
    timeout: int | None = None,
    active_env_override: str | None = None,
) -> dict:
    """Write a per-run config, train, evaluate, and atomically append the trial record.

    Every trial gets a unique run_id. Its config is written to a unique path and the
    training subprocess reads it via Q3C_CONFIG_PATH. The checkpoint directory is also
    run-id-scoped. No shared config.json mutation occurs, so parallel sbatch jobs do
    not collide on either config state or checkpoint files.

    `active_env_override`, when set, pins the env for this trial across training,
    evaluation, AND result-logging. This is the supported way to dispatch envs
    from the CLI — flipping config.json's active_env is racy.
    """
    script_name = script_path.name
    run_id = _new_run_id()

    print(f"\n{'=' * 80}")
    print(f"RUN {run_id} — {script_name}")
    print(f"{'=' * 80}")
    print(f"Parameters:\n{json.dumps(params, indent=2)}")

    config = load_config()
    if active_env_override is not None:
        if active_env_override not in config.get("environments", {}):
            raise ValueError(
                f"--active-env {active_env_override!r} is not in config.json's "
                f"environments. Known: {list(config.get('environments', {}).keys())}"
            )
        config["active_env"] = active_env_override
        print(f"  active_env override → {active_env_override}")
    config = apply_params_to_config(config, params)
    checkpoint_dir = set_run_checkpoint_dir(config, run_id)

    active_env = config.get("active_env", "particle")
    if training_steps_override is not None:
        config["environments"][active_env].setdefault("training", {})[
            "training_steps"
        ] = training_steps_override

    actual_steps = (
        config["environments"][active_env]
        .get("training", {})
        .get(
            "training_steps",
            config.get("training_shared", {}).get("training_steps", 100000),
        )
    )

    # Per-run config lives next to the checkpoints so you can always reconstruct a run.
    os.makedirs(checkpoint_dir, exist_ok=True)
    trial_config_path = Path(checkpoint_dir) / "config.json"
    with open(trial_config_path, "w") as f:
        json.dump(config, f, indent=4)

    print(f"\n  Training ({actual_steps} steps) — config at {trial_config_path}")
    success, stdout, duration = run_training(
        script_path,
        timeout=timeout,
        env_extras={"Q3C_CONFIG_PATH": str(trial_config_path)},
    )

    timestamp = datetime.now(timezone.utc).isoformat()

    if not success:
        print(f"\n  Training FAILED after {duration:.0f}s")
        last_lines = "\n".join(stdout.strip().splitlines()[-5:])
        record = {
            "run_id": run_id,
            "script": script_name,
            "active_env": active_env,
            "params": params,
            "training_steps": actual_steps,
            "duration_seconds": round(duration, 1),
            "success_rate": 0.0,
            "avg_reward": 0.0,
            "training_failed": True,
            "error": last_lines[-300:],
            "checkpoint_dir": checkpoint_dir,
            "timestamp": timestamp,
        }
        trial_id = append_trial(script_name, record, active_env=active_env)
        print(f"  Recorded as trial #{trial_id}")
        return record

    print(f"\n  Training completed in {duration:.0f}s")
    train_metrics = extract_final_metrics(stdout)

    print("  Evaluating...")
    eval_results: dict
    try:
        eval_results = evaluate_q3c(checkpoint_dir, config)
    except Exception as exc:
        eval_results = {
            "success_rate": 0.0,
            "avg_reward": 0.0,
            "error": f"Evaluation failed: {exc}",
            "per_seed": [],
        }
        print(f"  Evaluation failed: {exc}")

    # Env-agnostic record schema. Particle eval returns `*_first_goal` /
    # `*_second_goal` keys; pushing returns `*_to_target` keys. We spread
    # ALL non-private eval scalars into the top level so analyzers don't
    # silently drop env-specific metrics. Particle-style keys are kept as
    # explicit fields for backward compatibility with the legacy analyzer.
    env_specific = {
        k: v for k, v in eval_results.items()
        if k not in ("per_seed", "error", "success_rate", "avg_reward")
    }
    record = {
        "run_id": run_id,
        "script": script_name,
        "active_env": active_env,
        "params": params,
        "training_steps": actual_steps,
        "duration_seconds": round(duration, 1),
        "success_rate": eval_results.get("success_rate", 0.0),
        "avg_reward": eval_results.get("avg_reward", 0.0),
        **env_specific,
        **train_metrics,
        "eval_details": eval_results.get("per_seed", []),
        "eval_error": eval_results.get("error"),
        "checkpoint_dir": checkpoint_dir,
        "timestamp": timestamp,
    }
    trial_id = append_trial(script_name, record, active_env=active_env)

    import math as _math
    sr = record["success_rate"]
    rw = record["avg_reward"]
    rw_std = record.get("std_reward")
    n_eval = int(record.get("num_seeds") or 1)
    # Print SEM = σ_ep / √n_eval — standard error of THIS trial's mean over
    # its n_eval episodes. Cross-seed SEM (over multiple training seeds) is
    # what print_analysis aggregates; this single-trial SEM is the
    # within-trial counterpart.
    if rw_std is not None and n_eval > 0:
        sem = rw_std / _math.sqrt(n_eval)
        rw_str = f"{rw:.3f} ± {sem:.3f} (SEM, n={n_eval}; σ_ep={rw_std:.1f})"
    else:
        rw_str = f"{rw:.3f}"
    tc = record.get("avg_tasks_completed")
    tc_str = f", avg_tasks_completed={tc:.3f}" if tc is not None else ""
    print(
        f"\n  Result (trial #{trial_id}): success_rate={sr:.2%}{tc_str}, avg_reward={rw_str}"
    )
    return record


# ─── Analyze / summary ───────────────────────────────────────────────────────

def print_analysis(
    script_name: str,
    active_env: str | None = None,
    min_trial_id: int = 0,
) -> None:
    """Print a formatted results table sorted by success rate.

    `min_trial_id`: skip trials with id below this value — useful when env
    config has changed (e.g. pen `max_episode_steps` 200→100) and earlier
    trials have the same `params` dict but were trained under a different
    protocol. Defaults to 0 (no filter).
    """
    trials = load_trials(script_name, active_env=active_env)
    if min_trial_id > 0:
        trials = [t for t in trials if int(t.get("trial_id", 0)) >= min_trial_id]
    if not trials:
        print(f"No trials found for {script_name}.")
        return

    all_param_names: list[str] = []
    seen: set[str] = set()
    for t in trials:
        for p in t.get("params", {}):
            if p not in seen:
                all_param_names.append(p)
                seen.add(p)

    # Column definitions: (header, width)
    cols: list[tuple[str, int]] = [("Trial", 6)]
    for p in all_param_names:
        cols.append((p[:12], max(len(p[:12]), 10)))
    cols += [("Steps", 8), ("Success", 8), ("Reward", 18), ("Loss", 8), ("Time", 7)]

    header = " | ".join(f"{name:>{w}}" for name, w in cols)
    separator = "-+-".join("-" * w for _, w in cols)

    print(f"\n{'=' * len(header)}")
    print(f"  Hyperparameter search results: {script_name}")
    print(f"{'=' * len(header)}")
    print(header)
    print(separator)

    env_names_for_sort = {t.get("active_env") for t in trials}
    if env_names_for_sort <= {"pen", "door"}:
        sorted_trials = sorted(
            trials, key=lambda t: t.get("avg_reward", float("-inf")), reverse=True
        )
    else:
        sorted_trials = sorted(
            trials, key=lambda t: t.get("success_rate", -1), reverse=True
        )
    for t in sorted_trials:
        row_vals: list[str] = [f"{t['trial_id']:>6}"]
        for p in all_param_names:
            val = t.get("params", {}).get(p, "")
            if isinstance(val, float):
                row_vals.append(f"{val:>{cols[len(row_vals)][1]}.4g}")
            else:
                row_vals.append(f"{str(val):>{cols[len(row_vals)][1]}}")

        steps = t.get("training_steps", "?")
        row_vals.append(f"{steps:>8}")

        failed = t.get("training_failed", False)
        sr = t.get("success_rate", 0)
        rw = t.get("avg_reward", 0)
        rw_std = t.get("std_reward")
        loss = t.get("final_train_loss")
        dur = t.get("duration_seconds", 0)

        row_vals.append("  FAILED" if failed else f"{sr:>7.0%}")
        rw_cell = (
            f"{rw:.3f} ± {rw_std:.3f}" if rw_std is not None else f"{rw:.3f}"
        )
        row_vals.append(f"{rw_cell:>18}")
        row_vals.append(f"{loss:>8.4f}" if loss is not None else f"{'—':>8}")
        row_vals.append(f"{dur / 60:>6.1f}m")

        print(" | ".join(row_vals))

    print(f"{'=' * len(header)}")

    valid = [t for t in trials if not t.get("training_failed")]
    if valid:
        # D4RL Adroit objectives are reward (IBC paper Table 2 reports raw
        # returns for pen/door). Other envs still rank by success_rate.
        env_names = {t.get("active_env") for t in valid}
        if env_names <= {"pen", "door"}:
            best = max(valid, key=lambda t: t.get("avg_reward", float("-inf")))
        else:
            best = max(valid, key=lambda t: t.get("success_rate", 0))
        best_std = best.get("std_reward")
        rw_str = (
            f"{best['avg_reward']:.3f} ± {best_std:.3f}"
            if best_std is not None
            else f"{best['avg_reward']:.3f}"
        )
        print(f"\nBest trial: #{best['trial_id']}  "
              f"success_rate={best['success_rate']:.2%}  "
              f"avg_reward={rw_str}")
        print(f"  Params: {json.dumps(best['params'], indent=4)}")

    print(f"\nTotal trials: {len(trials)} ({len(valid)} completed, "
          f"{len(trials) - len(valid)} failed)\n")

    # ── Cross-seed aggregation table ──────────────────────────────────────
    # Groups trials with identical config but different `trial_seed`. For
    # each group computes:
    #   - mean of per-seed means
    #   - σ_ep_avg : average per-episode std across the group's seeds
    #     (env-intrinsic spread of episode rewards — bimodal on pen)
    #   - cross_std : sample stdev (ddof=1) of per-seed means
    #     (cross-seed variability — comparable to IBC paper's "± std")
    #   - cross_sem : cross_std / √n  (standard error of the mean across
    #     seeds — what IBC paper Table 2's ±65 most likely reports)
    import math
    from collections import defaultdict

    # Skip trials with eval errors (those have avg_reward=0 and would distort
    # cross-seed means if mixed with successful trials of the same config).
    eval_ok = [t for t in valid if not t.get("eval_error")]

    sig_groups: dict[str, list[dict]] = defaultdict(list)
    for t in eval_ok:
        p = dict(t.get("params") or {})
        p.pop("trial_seed", None)
        sig = json.dumps(p, sort_keys=True, default=str)
        sig_groups[sig].append(t)

    multi = [
        (sig, ts) for sig, ts in sig_groups.items() if len(ts) >= 2
    ]
    if multi:
        rows = []
        for sig, ts in multi:
            means = [float(t.get("avg_reward", 0)) for t in ts]
            per_ep = [float(t.get("std_reward") or 0) for t in ts]
            srs = [float(t.get("success_rate", 0)) for t in ts]
            n = len(means)
            mean_ = sum(means) / n
            var = sum((m - mean_) ** 2 for m in means) / (n - 1)
            cross_std = math.sqrt(var)
            cross_sem = cross_std / math.sqrt(n)
            avg_per_ep = sum(per_ep) / n
            avg_sr = sum(srs) / n
            seeds = sorted(
                {(t.get("params") or {}).get("trial_seed") for t in ts}
            )
            tid_list = sorted(t.get("trial_id", 0) for t in ts)
            rows.append((mean_, cross_std, cross_sem, avg_per_ep, avg_sr, n, seeds, tid_list))
        rows.sort(key=lambda r: -r[0])

        print("=" * 110)
        print("Cross-seed aggregates (groups with ≥2 trials of same config, different trial_seed)")
        print("=" * 110)
        print(
            f"{'n':>3} {'seeds':<14} {'trial_ids':<22} {'mean_R':>10} "
            f"{'cross_std':>10} {'SEM':>8} {'σ_ep(avg)':>11} {'SR(avg)':>8}"
        )
        print("-" * 110)
        for mean_, cstd, csem, pep, sr, n, seeds, tids in rows[:25]:
            seed_str = ",".join(str(s) for s in seeds)
            tid_str = ",".join(str(t) for t in tids[:6]) + ("…" if len(tids) > 6 else "")
            print(
                f"{n:>3} {seed_str:<14} {tid_str:<22} {mean_:>10.1f} "
                f"{cstd:>10.2f} {csem:>8.2f} {pep:>11.1f} {sr*100:>7.1f}%"
            )
        print("=" * 110)
        print(
            "  σ_ep(avg)  : intrinsic per-episode reward spread, averaged across the group's seeds.\n"
            "  cross_std  : sample stdev of per-seed means (sometimes printed as ± in papers).\n"
            "  SEM        : cross_std / √n  — IBC paper Table 2 ±65 best matches this interpretation.\n"
        )


# ─── CLI ──────────────────────────────────────────────────────────────────────

# ─── Completion notification (Telegram) ─────────────────────────────────────

def _notify_completion(status: str) -> None:
    """Send a Telegram message when the search finishes: which job ended + squeue --me.

    Opt-in: does nothing unless TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID are set
    in the environment (export them in ~/.bashrc or the sbatch script). Never
    raises — a notification failure must not turn a successful run into a
    failed job. Uses stdlib urllib so SLURM nodes need no extra packages.
    """
    token = os.environ.get("TELEGRAM_BOT_TOKEN")
    chat_id = os.environ.get("TELEGRAM_CHAT_ID")
    if not token or not chat_id:
        # Fallback: ~/.telegram.env (KEY=VALUE lines). SLURM's --export=ALL
        # snapshots the env at *submission* time, so jobs queued before the
        # exports existed have no TELEGRAM_* vars — but they read this file
        # at completion time, so notifications work without resubmitting.
        env_file = Path.home() / ".telegram.env"
        try:
            for line in env_file.read_text().splitlines():
                line = line.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                key, _, value = line.partition("=")
                key, value = key.strip(), value.strip().strip("'\"")
                if key == "TELEGRAM_BOT_TOKEN" and not token:
                    token = value
                elif key == "TELEGRAM_CHAT_ID" and not chat_id:
                    chat_id = value
        except OSError:
            pass
    if not token or not chat_id:
        return

    import html
    import urllib.parse
    import urllib.request

    job_id = os.environ.get("SLURM_JOB_ID", "local")
    job_name = os.environ.get("SLURM_JOB_NAME", "")
    try:
        squeue_out = subprocess.run(
            ["squeue", "--me"], capture_output=True, text=True, timeout=30
        ).stdout.strip()
    except Exception as exc:
        squeue_out = f"(squeue failed: {exc})"

    # <pre> keeps squeue's column alignment in the Telegram client.
    # Telegram rejects messages over 4096 chars, so clamp the table.
    squeue_block = html.escape(squeue_out or "(no jobs in queue)")
    if len(squeue_block) > 3200:
        squeue_block = squeue_block[:3200] + "\n… (truncated)"
    text = (
        f"<b>hyperparam_search {html.escape(status)}</b>\n"
        f"job {html.escape(str(job_id))} {html.escape(job_name)}\n"
        f"<code>{html.escape(' '.join(sys.argv[1:]))}</code>\n\n"
        f"<b>squeue --me</b>\n<pre>{squeue_block}</pre>"
    )

    data = urllib.parse.urlencode(
        {"chat_id": chat_id, "text": text, "parse_mode": "HTML"}
    ).encode()
    try:
        urllib.request.urlopen(
            urllib.request.Request(
                f"https://api.telegram.org/bot{token}/sendMessage", data=data
            ),
            timeout=15,
        )
    except Exception as exc:
        print(f"[notify] Telegram send failed: {exc}", file=sys.stderr)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Agent-assisted hyperparameter search for Q3C-IBC training scripts.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  python hyperparam_search.py combinedv2_cpascounter_training.py --run\n"
            "  python hyperparam_search.py combinedv2_cpascounter_training.py --run "
            "--params '{\"learning_rate\": 5e-4}'\n"
            "  python hyperparam_search.py combinedv2_cpascounter_training.py --auto "
            "--max-trials 5 --reduced-steps 20000\n"
            "  python hyperparam_search.py combinedv2_cpascounter_training.py --analyze\n"
        ),
    )
    parser.add_argument(
        "script",
        type=str,
        help="Training script to optimize (e.g. combinedv2_cpascounter_training.py)",
    )

    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--run", action="store_true", help="Run a single trial")
    mode.add_argument(
        "--auto",
        action="store_true",
        help="Run multiple trials with adaptive exploration",
    )
    mode.add_argument(
        "--analyze", action="store_true", help="Print summary of past trials"
    )
    parser.add_argument(
        "--min-trial-id", type=int, default=0,
        help="When analyzing, skip trials with id below this value. Useful "
             "to scope cross-seed aggregation to a recent batch when env "
             "protocol has changed (e.g. pen max_episode_steps 200→100).",
    )

    parser.add_argument(
        "--params",
        type=str,
        default=None,
        help='JSON string of param overrides for --run (e.g. \'{"learning_rate": 5e-4}\')',
    )
    parser.add_argument(
        "--fixed-params",
        type=str,
        default=None,
        help=(
            "JSON string of params to lock in all trials (works with --run and --auto), "
            "e.g. '{\"counter_examples\": 0}'"
        ),
    )
    parser.add_argument(
        "--max-trials",
        type=int,
        default=5,
        help="Number of trials for --auto mode (default: 5)",
    )
    parser.add_argument(
        "--reduced-steps",
        type=int,
        default=None,
        help="Override training_steps for faster exploration",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=None,
        help="Per-trial timeout in seconds",
    )
    parser.add_argument(
        "--num-reps",
        type=int,
        default=1,
        help=(
            "Number of repetitions per config (each rep gets trial_seed=0,1,...)."
            " Use this to measure variance honestly (default: 1)."
        ),
    )
    parser.add_argument(
        "--active-env",
        type=str,
        default=None,
        help=(
            "Override `active_env` from config.json for this invocation only. "
            "When set, the trial trains, evaluates, AND is logged under this "
            "env — no race with concurrent config.json edits. Recommended for "
            "all SLURM batches. Choices: any key under environments.* in config.json."
        ),
    )
    args = parser.parse_args()

    fixed_params: dict = {}
    if args.fixed_params:
        fixed_params = json.loads(args.fixed_params)

    script_path = ROOT_DIR / args.script
    if not script_path.exists():
        print(f"Error: script not found at {script_path}")
        sys.exit(1)

    script_name = Path(args.script).name

    active_env_cli = args.active_env

    # ── Analyze mode ──────────────────────────────────────────────────────
    if args.analyze:
        print_analysis(
            script_name, active_env=active_env_cli,
            min_trial_id=args.min_trial_id,
        )
        return

    # ── Detect params and baseline ────────────────────────────────────────
    detected_params = detect_script_params(script_path)
    if not detected_params:
        print(f"Warning: no tunable hyperparameters detected in {script_name}.")
        print("The script may use hardcoded values instead of config.json.")
    else:
        print(f"Detected tunable params in {script_name}:")
        for p in detected_params:
            print(f"  - {p}  (search space: {SEARCH_SPACE[p]['values']})")

    config = load_config()
    if active_env_cli is not None:
        # Make baseline reflect the override so the baseline params come from
        # the right env's training/model blocks.
        if active_env_cli not in config.get("environments", {}):
            print(
                f"Error: --active-env {active_env_cli!r} not found in config.json. "
                f"Known: {list(config.get('environments', {}).keys())}"
            )
            sys.exit(1)
        config["active_env"] = active_env_cli
        print(f"Active env override (CLI): {active_env_cli}")
    baseline = get_baseline_params(config, detected_params)
    print(f"\nBaseline (current config):")
    for k, v in baseline.items():
        print(f"  {k} = {v}")
    print()

    # ── Run mode ──────────────────────────────────────────────────────────
    if args.run:
        if args.params:
            user_params = json.loads(args.params)
            params = baseline.copy()
            params.update(user_params)
        else:
            trials = load_trials(script_name, active_env=active_env_cli)
            params, reason = suggest_next_params(trials, detected_params, baseline)
            print(f"Auto-suggested ({reason})")

        if fixed_params:
            params.update(fixed_params)
            print(f"Applied fixed params: {json.dumps(fixed_params)}")

        seed_pinned = "trial_seed" in params
        for rep in range(max(1, args.num_reps)):
            rep_params = dict(params)
            if not seed_pinned:
                rep_params["trial_seed"] = rep
            if args.num_reps > 1:
                print(f"\n[rep {rep + 1}/{args.num_reps}] trial_seed={rep_params['trial_seed']}")
            run_single_trial(
                script_path=script_path,
                params=rep_params,
                training_steps_override=args.reduced_steps,
                timeout=args.timeout,
                active_env_override=active_env_cli,
            )
        return

    # ── Auto mode ─────────────────────────────────────────────────────────
    if args.auto:
        for i in range(args.max_trials):
            trials = load_trials(script_name, active_env=active_env_cli)
            params, reason = suggest_next_params(trials, detected_params, baseline)
            print(f"\n[Auto {i + 1}/{args.max_trials}] Strategy: {reason}")

            if fixed_params:
                params.update(fixed_params)
                print(f"[Auto {i + 1}/{args.max_trials}] Fixed params: {json.dumps(fixed_params)}")

            seed_pinned = "trial_seed" in params
            for rep in range(max(1, args.num_reps)):
                rep_params = dict(params)
                if not seed_pinned:
                    rep_params["trial_seed"] = rep
                if args.num_reps > 1:
                    print(f"  [rep {rep + 1}/{args.num_reps}] trial_seed={rep_params['trial_seed']}")
                run_single_trial(
                    script_path=script_path,
                    params=rep_params,
                    training_steps_override=args.reduced_steps,
                    timeout=args.timeout,
                    active_env_override=active_env_cli,
                )

        print("\n\nAuto-exploration complete. Full results:")
        print_analysis(script_name, active_env=active_env_cli)


if __name__ == "__main__":
    _status = "completed"
    try:
        main()
    except BaseException as exc:  # noqa: BLE001 — report crashes/cancellations too
        _status = f"FAILED ({type(exc).__name__}: {exc})"
        raise
    finally:
        # --analyze is a quick local read; only notify for actual runs.
        if "--analyze" not in sys.argv:
            _notify_completion(_status)
