"""DP+Q3C combined training: diffusion-policy proposals, Q-estimator selection.

Drop-in sibling of ``combinedv2_cpascounter_training.py``. Same contract in every
way that matters to the surrounding tooling:

  * reads its config from ``Q3C_CONFIG_PATH`` (falling back to
    ``config_json/config.json``), with the identical
    ``environments / <active_env> / {model, training}`` + ``training_shared``
    schema, so ``hyperparam_search``-style per-trial configs work unchanged;
  * loads datasets through ``combinedv2_cpascounter_training.load_dataset``, so
    every env, idle filter, chunking and val-split behaviour is byte-identical
    rather than re-implemented here;
  * writes ``norm_stats.pt`` plus raw and EMA weights into ``model_save_dir``;
  * logs the SAME wandb metric keys (``metric/accuracy``,
    ``metric/cp_to_expert_min``, ``metric/cp_ranking_gap``,
    ``metric/q_pick_closest_frac``, ``val/action_mae`` ...) so existing plotting
    and analysis scripts read a dpq3c run the way they read a q3c run.

The ONE substitution: the control-point cloud is produced by a diffusion policy
instead of ``PixelControlPointGenerator``.

    q3c:   cp_gen(s)      -> (B, N, A)  -> Q ranks
    dpq3c: DP sampler(s)  -> (B, N, A)  -> Q ranks

Checkpoints are therefore ``denoiser[_ema].pt`` + ``q_estimator[_ema].pt`` —
exactly the two files ``scripts/deploy_pusht_real_dpq3c.py`` expects. Because one
run now produces both, deploy it with ``--dp-dir RUN --q-dir RUN``.

── What is trained ─────────────────────────────────────────────────────────────

ACTOR (denoiser), by the ordinary diffusion denoising loss on expert action
chunks. Optionally, ``q_actor_weight > 0`` adds a Q term that pulls the
denoiser's predicted CLEAN sample toward high Q — the training-time form of the
deploy client's ``--q-guidance``, at the cost of one extra Q pass per step and
with the estimator frozen for that term. Off by default.

CRITIC (Q estimator), by InfoNCE with the expert chunk as the positive and
negatives drawn from, in order:

    (a) the DIFFUSION POLICY itself (``dp_negatives``) — the alignment that
        matters. A critic whose negatives came from a different proposal
        distribution than the one it will rank at deploy is being asked to
        extrapolate at exactly the moment it decides the robot's command. Q3C's
        estimator learns on its own generator's control points; this learns on
        the samples it will actually see. Before ``dp_negative_warmup_steps`` the
        denoiser is still noise, so uniform draws stand in.
    (b) uniform draws over the action box, (c) Langevin hard negatives,
    (d) noisy-expert copies — all three exactly as combinedv2 uses them.

Two optional critic terms, both off by default, both discussed in the training
plan:

  ``margin_weight``   a DQfD-style large-margin hinge, ``max(0, m + max_i
                      Q(s, a_i) - Q(s, a_expert))`` over the DP candidates. Same
                      information as InfoNCE in a one-sided form.
  ``progress_weight`` the REWARD-FREE value anchor. Regresses Q at the expert
                      action toward the Monte-Carlo time-to-go return
                      ``-remaining_steps / scale``, a target every demonstration
                      dataset already contains and which needs no reward
                      labelling at all. This is what stops the critic from being
                      a pure ranker: InfoNCE and the margin are both invariant to
                      adding a per-state constant to every score, so neither can
                      fix an absolute scale. Turn this on and Q values become
                      comparable across states; leave it off and the critic is a
                      density model with a Q-shaped API.
"""

import copy
import json
import os
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import wandb

from utils.diffusion import (build_denoiser, build_diffusion, build_dpq3c_denoiser,
                             resolve_dp_params)
from utils.loss import lossInfoNCE
from utils.models import PixelQEstimator, QEstimator
from utils.normalizations import ObservationNormalizer
from utils.sampling import sample_langevin

# Same resolution order as combinedv2: a per-trial config wins, else the shared
# default. hyperparam_search drives both scripts through this env var.
config_path = Path(
    os.environ.get("Q3C_CONFIG_PATH")
    or (Path(__file__).parent / "config_json" / "config.json")
)
with open(config_path, "r") as f:
    config = json.load(f)

active_env = config.get("active_env", "pusht_real_pixels")
env_config = config["environments"][active_env]
training_shared = config.get("training_shared", {})
env_training = env_config.get("training", {})
env_model = env_config.get("model", {})

# Dataset construction is IMPORTED, not copied: every env branch, idle filter,
# action-chunking and val-split rule must stay bit-identical to the q3c trainer
# or a dpq3c-vs-q3c comparison is confounded by the data pipeline. The import
# re-reads the same Q3C_CONFIG_PATH, so it sees this run's config.
from combinedv2_cpascounter_training import load_dataset  # noqa: E402

PIXEL_ENVS = ("pushing_pixels", "pusht_real_pixels", "libero_goal_pixels")

# ── Training parameters (identical names/defaults to combinedv2) ─────────────
training_steps = env_training.get("training_steps", training_shared.get("training_steps", 100000))
batch_size = env_training.get("batch_size", training_shared.get("batch_size", 128))
learning_rate = env_training.get("learning_rate", training_shared.get("learning_rate", 1e-3))
estimator_learning_rate = env_training.get(
    "estimator_learning_rate", training_shared.get("estimator_learning_rate", learning_rate)
)

info_nce_weight = training_shared.get("info_nce_weight", 1.0)
MODEL_SAVE_DIR = training_shared.get("model_save_dir", "checkpoints")
log_interval = training_shared.get("log_interval", 1000)
save_interval = training_shared.get("save_interval", 10000)

scheduler_type = env_training.get("scheduler_type", training_shared.get("scheduler_type", "cosine"))
cosine_t0 = env_training.get("cosine_t0", training_shared.get("cosine_t0", 50000))
cosine_t_max = env_training.get("cosine_t_max", training_shared.get("cosine_t_max", None))
infonce_logit_clamp = env_training.get(
    "infonce_logit_clamp", training_shared.get("infonce_logit_clamp", 50.0)
)

trial_seed = env_training.get("trial_seed", training_shared.get("trial_seed", 0))
nan_abort_threshold = env_training.get(
    "nan_abort_threshold", training_shared.get("nan_abort_threshold", 50)
)
ema_decay = float(env_training.get("ema_decay", training_shared.get("ema_decay", 0.0)))
if not 0.0 <= ema_decay < 1.0:
    raise ValueError("ema_decay must satisfy 0 <= ema_decay < 1")

# ── Negatives (same knobs as combinedv2, plus the DP source) ─────────────────
num_uniform_negatives = env_training.get(
    "num_uniform_negatives", training_shared.get("num_uniform_negatives", 32)
)
num_langevin_negatives = env_training.get(
    "num_langevin_negatives", training_shared.get("num_langevin_negatives", 32)
)
noisy_expert_count = int(
    training_shared.get("noisy_expert_count", env_training.get("noisy_expert_count", 0))
)
noisy_expert_sigma_start = float(
    training_shared.get(
        "noisy_expert_sigma_start", env_training.get("noisy_expert_sigma_start", 0.1)
    )
)
noisy_expert_sigma_final = float(
    training_shared.get(
        "noisy_expert_sigma_final", env_training.get("noisy_expert_sigma_final", 0.02)
    )
)

langevin_config = env_model.get("langevin_config", {})
langevin_num_iterations = env_training.get(
    "langevin_num_iterations", langevin_config.get("num_iterations", 50)
)
langevin_lr_init = env_training.get("langevin_lr_init", langevin_config.get("lr_init", 0.1))
langevin_lr_final = env_training.get("langevin_lr_final", langevin_config.get("lr_final", 1e-5))
langevin_decay_power = env_training.get(
    "langevin_decay_power", langevin_config.get("polynomial_decay_power", 2.0)
)
langevin_delta_clip = env_training.get(
    "langevin_delta_clip", langevin_config.get("delta_action_clip", 0.1)
)
langevin_noise_scale = env_training.get(
    "langevin_noise_scale", langevin_config.get("noise_scale", 1.0)
)

gradient_penalty_weight = env_training.get(
    "gradient_penalty_weight", training_shared.get("gradient_penalty_weight", 0.0)
)
gradient_penalty_margin = env_training.get(
    "gradient_penalty_margin", training_shared.get("gradient_penalty_margin", 1.0)
)
gradient_penalty_form = env_training.get(
    "gradient_penalty_form", training_shared.get("gradient_penalty_form", "hinge")
)
if gradient_penalty_form not in ("hinge", "target"):
    raise ValueError(f"gradient_penalty_form must be hinge|target, got {gradient_penalty_form!r}")

# ── dpq3c-specific ───────────────────────────────────────────────────────────
# How many diffusion samples the critic ranks per state during training. Match
# this to the deploy --cp so the critic is calibrated on the cloud size it will
# actually face; the score gaps inside a cloud grow with N, so a critic (and a
# selection temperature) tuned at one N is not the same at another.
dp_negatives = int(env_training.get("dp_negatives", training_shared.get("dp_negatives", 16)))
# Denoising steps used to DRAW those negatives. They do not have to be great
# samples, they have to be the kind of thing that gets proposed — so a short
# DDIM chain is the right trade. This is the dominant per-step cost knob.
dp_negative_iters = int(
    env_training.get("dp_negative_iters", training_shared.get("dp_negative_iters", 4))
)
dp_negative_method = str(
    env_training.get("dp_negative_method", training_shared.get("dp_negative_method", "ddim"))
)
if dp_negative_method not in ("ddim", "ddpm"):
    raise ValueError(f"dp_negative_method must be ddim|ddpm, got {dp_negative_method!r}")
# Until the denoiser has learned anything its samples are indistinguishable from
# noise, so they are wasted compute as "hard" negatives. Uniform draws stand in.
dp_negative_warmup_steps = int(
    env_training.get(
        "dp_negative_warmup_steps", training_shared.get("dp_negative_warmup_steps", 0)
    )
)
# Actor <- critic feedback (training-time analogue of deploy --q-guidance).
q_actor_weight = float(
    env_training.get("q_actor_weight", training_shared.get("q_actor_weight", 0.0))
)
# DQfD-style large-margin term on the expert chunk.
margin_weight = float(env_training.get("margin_weight", training_shared.get("margin_weight", 0.0)))
margin_value = float(env_training.get("margin", training_shared.get("margin", 0.1)))
# Reward-free absolute-scale anchor: Monte-Carlo time-to-go return.
progress_weight = float(
    env_training.get("progress_weight", training_shared.get("progress_weight", 0.0))
)

env_id = env_config["env_id"]
action_bounds = env_config.get("action_bounds", [-1, 1])
frame_stack = env_config.get("frame_stack", 1)

dp = resolve_dp_params(env_config, training_shared)


def denoiser_head(model: nn.Module) -> nn.Module:
    """The flat epsilon head — what the sampler loop should call.

    Pixel denoisers wrap (encoder, head); running the sampler against the whole
    module would re-run the conv tower on every denoising iteration, which is
    what made the Q3C Langevin chain blow the wall clock before it was switched
    to late fusion. Flat denoisers ARE the head.
    """
    return getattr(model, "denoiser", model)


def denoiser_features(model: nn.Module, states: torch.Tensor) -> torch.Tensor:
    """Encode once per batch (pixels) or pass the state through (flat)."""
    return model.encode(states) if hasattr(model, "encode") else states


@torch.no_grad()
def sample_dp_actions(diffusion, model, feats, num, action_dim, steps, method="ddim",
                      eta=0.0):
    """Draw `num` diffusion samples per state. feats: (B, F) -> (B, num, A).

    One flattened sampler call over B*num rows: the candidate count is width, not
    sequential depth, so this costs `steps` head passes regardless of `num`.
    """
    B, F = feats.shape
    flat = feats.unsqueeze(1).expand(B, num, F).reshape(B * num, F)
    head = denoiser_head(model)
    if method == "ddim":
        x = diffusion.ddim_sample(head, flat, action_dim=action_dim,
                                  num_steps=int(steps), eta=float(eta))
    else:
        x = diffusion.ddpm_sample(head, flat, action_dim=action_dim)
    return x.view(B, num, action_dim)


def build_progress_targets(dataset) -> tuple[np.ndarray, float]:
    """Per-sample steps-remaining-in-episode, and the scale to divide it by.

    The reward-free anchor. Episode boundaries live on the RAW contiguous
    timeline while ``_samples`` is a filtered index list into it, so the
    remaining-step count must be computed against the raw indices — the same
    discipline ``build_chunked_actions`` uses. Getting this wrong silently
    produces targets that are off by however many transitions the idle filter
    removed.

    Datasets describe their episode structure two different ways, so both are
    accepted: PushTWidowXVideoDataset carries ``_episode_ends`` (exclusive
    cumulative ends) plus a filtered ``_samples`` index list, while the sim
    datasets (PushingPixels, LiberoGoalPixels, Particle) carry only an
    ``_episode_starts`` boolean mask and do no filtering. Ends are derived from
    starts in the second case.
    """
    samples = getattr(dataset, "_samples", None)
    ends = getattr(dataset, "_episode_ends", None)

    if ends is None:
        starts = getattr(dataset, "_episode_starts", None)
        if starts is None:
            raise SystemExit(
                "progress_weight > 0 needs a dataset exposing episode "
                "boundaries (`_episode_ends` or `_episode_starts`); "
                f"{type(dataset).__name__} exposes neither. Use "
                "progress_weight 0, or add the attribute."
            )
        starts = np.asarray(starts)
        if starts.dtype == bool:
            # Boolean mask over the raw timeline (the sim datasets' form).
            n_total = int(starts.size)
            start_idx = np.flatnonzero(starts).astype(np.int64)
        else:
            start_idx = np.asarray(starts, dtype=np.int64)
            n_total = (int(np.asarray(samples).max()) + 1 if samples is not None
                       else len(dataset))
        if start_idx.size == 0 or start_idx[0] != 0:
            start_idx = np.concatenate([[0], start_idx])
        ends = np.append(start_idx[1:], n_total)

    if samples is None:
        # No idle filtering: the sample list IS the raw timeline.
        samples = np.arange(len(dataset), dtype=np.int64)

    samples = np.asarray(samples, dtype=np.int64)
    ends = np.asarray(ends, dtype=np.int64)             # exclusive episode ends
    ep = np.clip(np.searchsorted(ends, samples, side="right"), 0, len(ends) - 1)
    starts = np.concatenate([[0], ends[:-1]])
    remaining = (ends[ep] - 1 - samples).astype(np.float32)
    lengths = (ends[ep] - starts[ep]).astype(np.float32)
    return remaining, float(max(1.0, lengths.mean()))


class _WithIndex(torch.utils.data.Dataset):
    """Adds the sample index to each item, so the loop can look up its target."""

    def __init__(self, base):
        self.base = base

    def __len__(self):
        return len(self.base)

    def __getitem__(self, i):
        item = self.base[i]
        item["idx"] = i
        return item


def main():
    random.seed(trial_seed)
    np.random.seed(trial_seed)
    torch.manual_seed(trial_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(trial_seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"trial_seed={trial_seed} (deterministic seeding applied)")
    print(f"Using device: {device}")
    print(f"Active environment: {active_env}")
    print(f"Training steps: {training_steps}   batch: {batch_size}")
    print(f"LR: denoiser={learning_rate}  estimator={estimator_learning_rate}  "
          f"scheduler={scheduler_type}")
    print(f"Diffusion: T={dp['num_train_timesteps']} schedule={dp['beta_schedule']} "
          f"pred={dp['prediction_type']} head={dp['denoiser_network_kind']}"
          f"({dp['denoiser_width']}x{dp['denoiser_depth']})")
    print(f"DP negatives: {dp_negatives} via {dp_negative_method} x{dp_negative_iters} "
          f"(warmup {dp_negative_warmup_steps} steps -> uniform stand-in)")
    print(f"Critic terms: info_nce={info_nce_weight} margin={margin_weight}"
          f"(m={margin_value}) progress={progress_weight} gp={gradient_penalty_weight}")
    print(f"Other negatives: uniform={num_uniform_negatives} "
          f"langevin={num_langevin_negatives}x{langevin_num_iterations} "
          f"noisy_expert={noisy_expert_count}")
    print(f"Actor<-Q weight: {q_actor_weight} "
          f"({'ON' if q_actor_weight > 0 else 'off'})")
    print(f"EMA decay: {ema_decay}   frame_stack: {frame_stack}")

    wandb_run_name = f"{active_env}_dpq3c_n{dp_negatives}_lr{learning_rate}_seed{trial_seed}"
    wandb.init(
        project="Q3CIBC",
        config={
            "algorithm": "dpq3c",
            "active_env": active_env,
            "env_config": env_config,
            "training_shared": training_shared,
        },
        name=wandb_run_name,
    )

    print(f"Loading {active_env} dataset...")
    dataset = load_dataset()
    print(f"Dataset size: {len(dataset)}")
    action_dim = int(dataset.action_shape)

    is_pixels = active_env in PIXEL_ENVS
    cond_dim = int(getattr(dataset, "cond_dim", 0))
    goal_dim = int(getattr(dataset, "goal_emb_dim", 0))

    # ── Models ───────────────────────────────────────────────────────────────
    diffusion = build_diffusion(dp, device, (action_bounds[0], action_bounds[1]))

    if is_pixels:
        in_channels = dataset.state_shape[0]
        enc_h = env_config.get("encoder_target_height", 180)
        enc_w = env_config.get("encoder_target_width", 240)
        enc_feat = int(env_model.get("encoder_feature_dim", 256))
        encoder_kind = env_model.get("encoder_kind", "conv_maxpool")
        encoder_pretrained = bool(env_model.get("encoder_pretrained", True))
        encoder_num_kp = int(env_model.get("encoder_num_kp", 64))
        encoder_norm_kind = env_model.get("encoder_norm_kind", "bn")
        encoder_per_camera = bool(env_model.get("encoder_per_camera", False))
        cond_fusion = env_model.get("cond_fusion", "concat")

        if cond_dim:
            print(f"Denoiser: PIXEL(cond={cond_dim}) enc={encoder_kind} {enc_h}x{enc_w}")
            denoiser = build_dpq3c_denoiser(
                action_dim, in_channels, dp, cond_dim=cond_dim,
                encoder_target_height=enc_h, encoder_target_width=enc_w,
                encoder_feature_dim=enc_feat, encoder_kind=encoder_kind,
                encoder_pretrained=encoder_pretrained, encoder_num_kp=encoder_num_kp,
                encoder_norm_kind=encoder_norm_kind,
                encoder_per_camera=encoder_per_camera, device=device)
            print("[WARN] cond_dim > 0: scripts/deploy_pusht_real_dpq3c.py cannot "
                  "yet rebuild a CONDITIONED denoiser (build_dp_denoiser raises). "
                  "Train it if you like, but wire up the deploy side before "
                  "expecting to run it on the arm.")
        else:
            print(f"Denoiser: PIXEL enc={encoder_kind} {enc_h}x{enc_w} "
                  f"in_ch={in_channels}")
            denoiser = build_dpq3c_denoiser(
                action_dim, in_channels, dp, cond_dim=0,
                encoder_target_height=enc_h, encoder_target_width=enc_w,
                encoder_feature_dim=enc_feat, encoder_kind=encoder_kind,
                encoder_pretrained=encoder_pretrained, encoder_num_kp=encoder_num_kp,
                encoder_norm_kind=encoder_norm_kind,
                encoder_per_camera=encoder_per_camera, device=device,
            )

        value_width = int(env_model.get("value_width", 1024))
        value_num_blocks = int(env_model.get("value_num_blocks", 1))
        print(f"Q estimator: PIXEL DenseResnetValue(w={value_width}, "
              f"blocks={value_num_blocks}) cond={cond_dim}")
        estimator = PixelQEstimator(
            action_dim=action_dim, in_channels=in_channels,
            encoder_target_height=enc_h, encoder_target_width=enc_w,
            encoder_feature_dim=enc_feat,
            value_width=value_width, value_num_blocks=value_num_blocks,
            cond_dim=cond_dim, encoder_kind=encoder_kind,
            encoder_pretrained=encoder_pretrained, encoder_num_kp=encoder_num_kp,
            encoder_norm_kind=encoder_norm_kind,
            encoder_per_camera=encoder_per_camera, cond_fusion=cond_fusion,
            goal_dim=goal_dim,
        ).to(device)
    else:
        state_dim = int(dataset.state_shape)
        print(f"Denoiser: FLAT state_dim={state_dim}")
        denoiser = build_denoiser(state_dim, action_dim, dp, device=device)
        q_width = int(env_model.get("q_width", env_model.get("num_neurons", 512)))
        q_depth = int(env_model.get("q_depth", env_model.get("num_hidden_layers", 8)))
        print(f"Q estimator: FLAT kind={env_model.get('q_network_kind', 'mlp')} "
              f"width={q_width} depth={q_depth}")
        estimator = QEstimator(
            state_dim=state_dim, action_dim=action_dim,
            hidden_dims=[q_width for _ in range(q_depth)],
            use_spectral_norm=bool(env_model.get("q_use_spectral_norm", False)),
            network_kind=env_model.get("q_network_kind", "mlp"),
            width=q_width, depth=q_depth,
        ).to(device)

    ema_denoiser = copy.deepcopy(denoiser) if ema_decay > 0.0 else None
    ema_estimator = copy.deepcopy(estimator) if ema_decay > 0.0 else None
    for m in (ema_denoiser, ema_estimator):
        if m is not None:
            m.eval()
            for p in m.parameters():
                p.requires_grad_(False)

    @torch.no_grad()
    def update_ema(ema_model: nn.Module, source_model: nn.Module) -> None:
        src_params = dict(source_model.named_parameters())
        for name, p in ema_model.named_parameters():
            p.mul_(ema_decay).add_(src_params[name].detach(), alpha=1.0 - ema_decay)
        src_buffers = dict(source_model.named_buffers())
        for name, b in ema_model.named_buffers():
            src = src_buffers[name].detach()
            # SpatialSoftmax coordinate grids materialize lazily on the first
            # forward, i.e. after this deepcopy; adopt the source shape once.
            if b.shape != src.shape:
                b.resize_(src.shape)
            b.copy_(src)

    def save_checkpoints() -> None:
        os.makedirs(MODEL_SAVE_DIR, exist_ok=True)
        torch.save(denoiser.state_dict(), os.path.join(MODEL_SAVE_DIR, "denoiser.pt"))
        torch.save(estimator.state_dict(), os.path.join(MODEL_SAVE_DIR, "q_estimator.pt"))
        if ema_denoiser is not None and ema_estimator is not None:
            torch.save(ema_denoiser.state_dict(),
                       os.path.join(MODEL_SAVE_DIR, "denoiser_ema.pt"))
            torch.save(ema_estimator.state_dict(),
                       os.path.join(MODEL_SAVE_DIR, "q_estimator_ema.pt"))

    def q_score_candidates(state: torch.Tensor, actions_bna: torch.Tensor) -> torch.Tensor:
        """Q for (B, N, A) candidates; encoder runs ONCE per state for pixels."""
        if state.ndim == 4:
            return estimator(state, actions_bna)
        states_expanded = state.unsqueeze(1).expand(-1, actions_bna.shape[1], -1)
        return estimator(states_expanded, actions_bna)

    # ── Optimizers / schedulers (same structure as combinedv2) ───────────────
    encoder_lr_scale = float(env_training.get("encoder_lr_scale", 1.0))
    if encoder_lr_scale != 1.0 and hasattr(denoiser, "encoder"):
        def _split_groups(model, base_lr):
            enc_ids = {id(p) for p in model.encoder.parameters()}
            enc = [p for p in model.parameters() if id(p) in enc_ids]
            rest = [p for p in model.parameters() if id(p) not in enc_ids]
            return [{"params": rest, "lr": base_lr},
                    {"params": enc, "lr": base_lr * encoder_lr_scale}]
        print(f"Split LR: encoder x{encoder_lr_scale}")
        optimizer_denoiser = torch.optim.AdamW(_split_groups(denoiser, learning_rate))
        optimizer_estimator = torch.optim.AdamW(
            _split_groups(estimator, estimator_learning_rate))
    else:
        optimizer_denoiser = torch.optim.AdamW(denoiser.parameters(), lr=learning_rate)
        optimizer_estimator = torch.optim.AdamW(estimator.parameters(),
                                                lr=estimator_learning_rate)

    effective_t_max = cosine_t_max if cosine_t_max is not None else training_steps
    if scheduler_type == "cosine_warm_restarts":
        make_sched = lambda o: torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(  # noqa: E731
            o, T_0=cosine_t0, eta_min=1e-6)
    else:
        make_sched = lambda o: torch.optim.lr_scheduler.CosineAnnealingLR(  # noqa: E731
            o, T_max=effective_t_max, eta_min=1e-6)
    scheduler_denoiser = make_sched(optimizer_denoiser)
    scheduler_estimator = make_sched(optimizer_estimator)

    # ── Data ─────────────────────────────────────────────────────────────────
    progress_remaining = progress_scale = None
    train_dataset = dataset
    if progress_weight > 0.0:
        progress_remaining, progress_scale = build_progress_targets(dataset)
        progress_remaining_t = torch.from_numpy(progress_remaining).to(device)
        train_dataset = _WithIndex(dataset)
        print(f"Progress anchor: ON (scale={progress_scale:.1f} steps, target = "
              f"-remaining/scale at the expert action)")

    num_workers = env_config.get(
        "dataloader_num_workers", 4 if is_pixels else 0)
    dataloader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, persistent_workers=num_workers > 0,
        timeout=600 if num_workers > 0 else 0,
    )

    val_loader = None
    val_interval = int(env_training.get("val_interval", save_interval))
    if (str(env_config.get("data_format", "")) == "zarr_video"
            and float(env_config.get("val_frac", 0.0)) > 0.0):
        val_dataset = load_dataset(split="val")
        val_loader = torch.utils.data.DataLoader(
            val_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
        print(f"Validation: {len(val_dataset)} held-out transitions; val "
              f"action-MAE logged every {val_interval} steps.")

    if is_pixels:
        obs_normalizer = None
        print("Observation normalizer: NONE (pixel encoder handles preprocessing)")
    elif hasattr(dataset, "obs_mean"):
        obs_normalizer = ObservationNormalizer(
            env_id=env_id, device=device, frame_stack=frame_stack,
            obs_mean=dataset.obs_mean, obs_std=dataset.obs_std)
        print("Observation normalizer: standardize")
    else:
        obs_normalizer = ObservationNormalizer(
            env_id=env_id, device=device, frame_stack=frame_stack)
        print("Observation normalizer: minmax")

    action_min_tensor = torch.full((action_dim,), action_bounds[0], device=device)
    action_max_tensor = torch.full((action_dim,), action_bounds[1], device=device)
    action_range_tensor = action_max_tensor - action_min_tensor

    # ── norm_stats: one file serving BOTH halves of the deploy client ────────
    def persist_norm_stats() -> None:
        # combinedv2 gates this on an env whitelist; keying on the attribute is
        # equivalent and does not need updating when an env is added. Datasets
        # without action stats (dummy) have nothing for eval to denormalize.
        if not hasattr(dataset, "act_min"):
            return
        norm_stats = {
            "act_min": dataset.act_min,
            "act_max": dataset.act_max,
            "action_norm_range": getattr(dataset, "action_norm_range", (-1.0, 1.0)),
            "frame_stack": frame_stack,
            "env_id": env_id,
            "action_chunk": int(getattr(dataset, "action_chunk", 1)),
            "algorithm": "dpq3c",
            # Deploy-time CP-cloud selection defaults.
            "cp_selection": str(env_training.get("cp_selection", "argmax")),
            "cp_selection_temperature": float(
                env_training.get("cp_selection_temperature", 1.0)),
            # The cloud size this critic was CALIBRATED on. Deploy should use the
            # same --cp unless it also retunes the temperature.
            "dp_negatives": dp_negatives,
            # Sampler reconstruction (deploy's build_dp_denoiser reads these).
            "num_train_timesteps": dp["num_train_timesteps"],
            "beta_schedule": dp["beta_schedule"],
            "prediction_type": dp["prediction_type"],
            "time_emb_dim": dp["time_emb_dim"],
            "denoiser_network_kind": dp["denoiser_network_kind"],
            "denoiser_width": dp["denoiser_width"],
            "denoiser_depth": dp["denoiser_depth"],
            "ddim_eval_steps": dp["ddim_eval_steps"],
            "ddim_eta": dp["ddim_eta"],
            "cond_dim": cond_dim,
        }
        if is_pixels:
            # Source these defensively: the pixel datasets do NOT share one
            # attribute set. PushingPixelsDataset has neither `_H`/`_W` nor
            # `in_channels`, and LiberoGoalPixelsDataset has `_H` but no `_W`.
            # `state_shape` is (C, H, W) on all three, so it is the one reliable
            # source; the config is the last resort.
            _ss = list(getattr(dataset, "state_shape", []) or [])
            if len(_ss) == 3:
                _in_ch, _img_h, _img_w = int(_ss[0]), int(_ss[1]), int(_ss[2])
            else:
                _in_ch = int(getattr(dataset, "in_channels", 0))
                _img_h = int(env_config.get("image_height", 240))
                _img_w = int(env_config.get("image_width", 320))
            norm_stats.update({
                "in_channels": _in_ch,
                "image_hw": [_img_h, _img_w],
                "state_shape": _ss,
                "encoder_target_height": env_config.get("encoder_target_height", 180),
                "encoder_target_width": env_config.get("encoder_target_width", 240),
                "encoder_feature_dim": int(env_model.get("encoder_feature_dim", 256)),
                "encoder_kind": env_model.get("encoder_kind", "conv_maxpool"),
                "encoder_pretrained": bool(env_model.get("encoder_pretrained", True)),
                "encoder_num_kp": int(env_model.get("encoder_num_kp", 64)),
                "encoder_norm_kind": env_model.get("encoder_norm_kind", "bn"),
                "encoder_per_camera": bool(env_model.get("encoder_per_camera", False)),
                "cond_fusion": env_model.get("cond_fusion", "concat"),
            })
        if active_env == "pusht_real_pixels":
            norm_stats["camera_streams"] = list(dataset.camera_streams)
            norm_stats["action_dims"] = list(dataset.action_dims)
            norm_stats["action_semantics"] = dataset.action_semantics
            norm_stats["idle_filter"] = str(env_config.get("idle_filter", "none"))
            norm_stats["data_format"] = str(env_config.get("data_format", ""))
            if cond_dim > 0:
                norm_stats["cond_kind"] = "eef_xy"
                norm_stats["cond_min"] = dataset.cond_min
                norm_stats["cond_max"] = dataset.cond_max
        if active_env == "libero_goal_pixels":
            norm_stats["libero_obs_keys"] = dataset.libero_obs_keys
            norm_stats["goal_embeddings"] = dataset.goal_embeddings
            norm_stats["goal_task_names"] = dataset.goal_task_names
            norm_stats["goal_emb_dim"] = dataset.goal_emb_dim
            norm_stats["proprio_dim"] = dataset.proprio_dim
            norm_stats["image_crop_size"] = int(env_training.get("image_crop_size", 0))
        if hasattr(dataset, "obs_mean"):
            norm_stats["obs_mean"] = dataset.obs_mean
            norm_stats["obs_std"] = dataset.obs_std
        os.makedirs(MODEL_SAVE_DIR, exist_ok=True)
        torch.save(norm_stats, os.path.join(MODEL_SAVE_DIR, "norm_stats.pt"))
        print(f"norm_stats.pt saved (act range {dataset.act_min} → {dataset.act_max})")

    # Written up front so a wall-clock-killed run is still evaluable from its
    # periodic checkpoints (same rationale as combinedv2).
    persist_norm_stats()

    # ── Validation: deploy-matching DP-cloud + argmax-Q selection ────────────
    @torch.no_grad()
    def _cloud_action_mae(dn, qn, states_t, actions_t, cond_t=None):
        if cond_t is not None:
            if getattr(dn, "cond_dim", 0):
                dn._cond = cond_t
            qn._cond = cond_t
        feats = denoiser_features(dn, states_t)
        cloud = sample_dp_actions(diffusion, dn, feats, dp_negatives, action_dim,
                                  dp_negative_iters, dp_negative_method,
                                  dp.get("ddim_eta", 0.0))
        if states_t.ndim == 4:
            qv = qn(states_t, cloud).squeeze(-1)
        else:
            qv = qn(states_t.unsqueeze(1).expand(-1, cloud.shape[1], -1), cloud).squeeze(-1)
        best = cloud[torch.arange(states_t.shape[0], device=states_t.device),
                     qv.argmax(dim=1)]
        return (best - actions_t).abs().mean().item()

    @torch.no_grad()
    def _val_action_mae(dn, qn):
        dn.eval(); qn.eval()
        tot, n = 0.0, 0
        for vb in val_loader:
            vs = vb["state"].float().to(device)
            va = vb["action"].float().to(device)
            vc = vb["cond"].float().to(device) if "cond" in vb else None
            bs = vs.shape[0]
            tot += _cloud_action_mae(dn, qn, vs, va, vc) * bs
            n += bs
        dn.train(); qn.train()
        return tot / max(n, 1)

    # ── Train ────────────────────────────────────────────────────────────────
    start_time = time.time()
    step = 0
    consecutive_nan_batches = 0
    best_val_mae = float("inf")
    T = int(dp["num_train_timesteps"])
    prediction_type = str(dp["prediction_type"])

    while step < training_steps:
        for batch in dataloader:
            if step >= training_steps:
                break

            states = batch["state"].float().to(device)
            if obs_normalizer is not None:
                states = obs_normalizer.normalize(states)
            actions = batch["action"].float().to(device)
            B = states.shape[0]

            cond = None
            if "cond" in batch:
                cond = batch["cond"].float().to(device)
                estimator._cond = cond
                if getattr(denoiser, "cond_dim", 0):
                    denoiser._cond = cond

            # ══ Actor: denoising loss ═══════════════════════════════════════
            # Reproduces GaussianDiffusion.training_loss exactly; written out so
            # the same (x_t, model output) can feed the optional Q term below
            # instead of costing a second forward pass.
            t_idx = torch.randint(0, T, (B,), device=device)
            noise = torch.randn_like(actions)
            x_t = diffusion.q_sample(actions, t_idx, noise)
            model_out = denoiser(states, x_t, t_idx.float())
            sqrt_acp = diffusion.sqrt_acp[t_idx].unsqueeze(-1)
            sqrt_omacp = diffusion.sqrt_one_minus_acp[t_idx].unsqueeze(-1)
            if prediction_type == "v":
                target = sqrt_acp * noise - sqrt_omacp * actions
            else:
                target = noise
            loss_denoise = torch.mean((model_out - target) ** 2)

            # Optional: pull the predicted CLEAN sample toward high Q. The Q net
            # has only ever seen clean actions, so the term is applied to x0-hat,
            # never to the noisy iterate.
            if q_actor_weight > 0.0:
                if prediction_type == "v":
                    x0_hat = sqrt_acp * x_t - sqrt_omacp * model_out
                else:
                    x0_hat = (x_t - sqrt_omacp * model_out) / sqrt_acp
                x0_hat = x0_hat.clamp(action_bounds[0], action_bounds[1])
                for p in estimator.parameters():
                    p.requires_grad_(False)
                if states.ndim == 4:
                    q_of_x0 = estimator(states, x0_hat.unsqueeze(1)).squeeze(-1).squeeze(-1)
                else:
                    q_of_x0 = estimator(states, x0_hat).squeeze(-1)
                for p in estimator.parameters():
                    p.requires_grad_(True)
                loss_q_actor = -q_actor_weight * q_of_x0.mean()
            else:
                loss_q_actor = torch.zeros((), device=device)

            loss_actor_total = loss_denoise + loss_q_actor

            # ══ Critic: negatives ═══════════════════════════════════════════
            with torch.no_grad():
                feats_for_neg = denoiser_features(denoiser, states)
            use_dp_neg = dp_negatives > 0 and step >= dp_negative_warmup_steps
            if use_dp_neg:
                dp_neg = sample_dp_actions(
                    diffusion, denoiser, feats_for_neg, dp_negatives, action_dim,
                    dp_negative_iters, dp_negative_method, dp.get("ddim_eta", 0.0))
            elif dp_negatives > 0:
                # Warmup stand-in: the denoiser is still noise, so its samples
                # carry no information the uniform draws don't already give.
                dp_neg = (torch.rand(B, dp_negatives, action_dim, device=device)
                          * action_range_tensor + action_min_tensor)
            else:
                dp_neg = torch.zeros(B, 0, action_dim, device=device)

            neg_chunks: list[torch.Tensor] = [dp_neg]
            if num_uniform_negatives > 0:
                neg_chunks.append(
                    torch.rand(B, num_uniform_negatives, action_dim, device=device)
                    * action_range_tensor + action_min_tensor)

            if num_langevin_negatives > 0 and langevin_num_iterations > 0:
                for p in estimator.parameters():
                    p.requires_grad_(False)
                if states.ndim == 4:
                    with torch.no_grad():
                        _lv_feats = estimator.encode(states)

                    def _neg_energy_fn(_obs, actions_batch):
                        return -estimator.score(_lv_feats, actions_batch).squeeze(-1)
                else:
                    def _neg_energy_fn(obs_expanded, actions_batch):
                        return -estimator(obs_expanded, actions_batch).squeeze(-1)

                langevin_neg = sample_langevin(
                    energy_function=_neg_energy_fn, observations=states,
                    num_samples=num_langevin_negatives,
                    action_min=action_min_tensor, action_max=action_max_tensor,
                    num_iterations=langevin_num_iterations,
                    lr_init=langevin_lr_init, lr_final=langevin_lr_final,
                    polynomial_decay_power=langevin_decay_power,
                    delta_action_clip=langevin_delta_clip,
                    noise_scale=langevin_noise_scale, device=device,
                )
                for p in estimator.parameters():
                    p.requires_grad_(True)
                neg_chunks.append(langevin_neg.detach())

            if noisy_expert_count > 0:
                progress_frac = min(1.0, max(0.0, step / max(1, training_steps - 1)))
                sigma = (noisy_expert_sigma_start
                         + progress_frac * (noisy_expert_sigma_final - noisy_expert_sigma_start))
                exp_exp = actions.unsqueeze(1).expand(-1, noisy_expert_count, -1)
                neg_chunks.append(torch.clamp(
                    exp_exp + torch.randn_like(exp_exp) * sigma,
                    action_bounds[0], action_bounds[1]))

            counter_samples = torch.cat([c for c in neg_chunks if c.shape[1] > 0], dim=1)
            all_actions = torch.cat([actions.unsqueeze(1), counter_samples], dim=1)

            energies = q_score_candidates(states, all_actions).squeeze(-1)
            loss_infonce = lossInfoNCE(energies, logit_clamp=infonce_logit_clamp)

            # DQfD-style hinge: the expert chunk must beat the best DP proposal
            # by `margin`. Same information as InfoNCE, one-sided.
            if margin_weight > 0.0 and dp_neg.shape[1] > 0:
                q_expert = energies[:, 0]
                q_dp_best = energies[:, 1:1 + dp_neg.shape[1]].max(dim=1).values
                loss_margin = margin_weight * torch.clamp(
                    margin_value + q_dp_best - q_expert, min=0.0).mean()
            else:
                loss_margin = torch.zeros((), device=device)

            # Reward-free absolute-scale anchor. InfoNCE and the margin only ever
            # constrain score DIFFERENCES, so without this the critic is
            # identified only up to a per-state shift.
            if progress_weight > 0.0:
                rem = progress_remaining_t[batch["idx"].to(device)]
                target_v = -(rem / progress_scale)
                loss_progress = progress_weight * torch.mean(
                    (energies[:, 0] - target_v) ** 2)
            else:
                loss_progress = torch.zeros((), device=device)

            if gradient_penalty_weight > 0.0:
                gp_actions = all_actions.detach().clone().requires_grad_(True)
                gp_energies = q_score_candidates(states, gp_actions).squeeze(-1)
                gp_grad = torch.autograd.grad(gp_energies.sum(), gp_actions,
                                              create_graph=True)[0]
                grad_norms = gp_grad.flatten(start_dim=2).norm(dim=-1)
                if gradient_penalty_form == "hinge":
                    penalty = torch.clamp(grad_norms - gradient_penalty_margin,
                                          min=0.0).pow(2).mean()
                else:
                    penalty = (grad_norms - gradient_penalty_margin).pow(2).mean()
                loss_gp = gradient_penalty_weight * penalty
            else:
                loss_gp = torch.zeros((), device=device)

            loss_critic_total = (info_nce_weight * loss_infonce + loss_margin
                                 + loss_progress + loss_gp)

            if (torch.isnan(loss_actor_total) or torch.isnan(loss_critic_total)):
                consecutive_nan_batches += 1
                optimizer_denoiser.state.clear()
                optimizer_estimator.state.clear()
                if consecutive_nan_batches >= nan_abort_threshold:
                    raise RuntimeError(
                        f"Training diverged: {consecutive_nan_batches} consecutive "
                        f"NaN batches")
                if consecutive_nan_batches % 10 == 1:
                    print(f"NaN loss detected (run {consecutive_nan_batches}); "
                          f"cleared optimizer state, continuing.")
                continue
            consecutive_nan_batches = 0

            optimizer_denoiser.zero_grad()
            optimizer_estimator.zero_grad()
            total_loss = loss_actor_total + loss_critic_total
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(denoiser.parameters(), 1.0)
            torch.nn.utils.clip_grad_norm_(estimator.parameters(), 1.0)
            optimizer_denoiser.step()
            optimizer_estimator.step()
            if ema_denoiser is not None and ema_estimator is not None:
                update_ema(ema_denoiser, denoiser)
                update_ema(ema_estimator, estimator)
            scheduler_denoiser.step()
            scheduler_estimator.step()
            step += 1

            # ── Held-out validation ─────────────────────────────────────────
            if val_loader is not None and step % val_interval == 0:
                use_ema = ema_denoiser is not None and ema_estimator is not None
                ev_d = ema_denoiser if use_ema else denoiser
                ev_q = ema_estimator if use_ema else estimator
                val_mae = _val_action_mae(ev_d, ev_q)
                if not use_ema:
                    denoiser.eval(); estimator.eval()
                train_mae = _cloud_action_mae(ev_d, ev_q, states, actions, cond)
                if not use_ema:
                    denoiser.train(); estimator.train()
                best_val_mae = min(best_val_mae, val_mae)
                print(f"[val] step {step}: action_MAE train={train_mae:.4f} "
                      f"val={val_mae:.4f} gap={val_mae - train_mae:+.4f} "
                      f"best_val={best_val_mae:.4f}")
                wandb.log({
                    "step": step,
                    "val/action_mae": val_mae,
                    "val/action_mae_train": train_mae,
                    "val/action_mae_gap": val_mae - train_mae,
                    "val/action_mae_best": best_val_mae,
                })

            # ── Logging (same keys as combinedv2 so plots keep working) ─────
            if step % log_interval == 0:
                with torch.no_grad():
                    accuracy = (energies.argmax(dim=1) == 0).float().mean().item()
                    if dp_neg.shape[1] > 0:
                        cloud = dp_neg
                        q_cloud = energies[:, 1:1 + cloud.shape[1]]
                        to_expert = (cloud - actions.unsqueeze(1)).norm(dim=-1)
                        closest_idx = to_expert.argmin(dim=1)
                        closest = to_expert.min(dim=1).values.mean().item()
                        q_arg = q_cloud.argmax(dim=1)
                        qbest = to_expert.gather(1, q_arg.unsqueeze(-1)).squeeze(-1).mean().item()
                        pick = (q_arg == closest_idx).float().mean().item()
                    else:
                        closest = qbest = pick = float("nan")

                current_lr = scheduler_denoiser.get_last_lr()[0]
                elapsed = time.time() - start_time
                print(f"Step {step}/{training_steps} | Total: {total_loss.item():.4f} "
                      f"(Denoise: {loss_denoise.item():.4f}, "
                      f"EST: {loss_infonce.item():.4f}, "
                      f"Margin: {loss_margin.item():.4f}, "
                      f"Prog: {loss_progress.item():.4f}, "
                      f"GP: {loss_gp.item():.4f}, "
                      f"Qact: {loss_q_actor.item():.4f}, "
                      f"Acc: {accuracy:.3f}) | "
                      f"cp→a*: closest={closest:.4f} qbest={qbest:.4f} "
                      f"pick={pick:.3f} | LR: {current_lr:.2e} | {elapsed:.1f}s")
                wandb.log({
                    "step": step,
                    "loss/total": total_loss.item(),
                    "loss/denoise": loss_denoise.item(),
                    "loss/generator": loss_actor_total.item(),
                    "loss/estimator": loss_infonce.item(),
                    "loss/margin": loss_margin.item(),
                    "loss/progress": loss_progress.item(),
                    "loss/gradient_penalty": loss_gp.item(),
                    "loss/q_actor": loss_q_actor.item(),
                    "metric/accuracy": accuracy,
                    "metric/cp_to_expert_min": closest,
                    "metric/cp_to_expert_qbest": qbest,
                    "metric/cp_ranking_gap": qbest - closest,
                    "metric/q_pick_closest_frac": pick,
                    "learning_rate": current_lr,
                })

            if step % save_interval == 0:
                save_checkpoints()

    total_time = time.time() - start_time
    print(f"\nTraining completed in {total_time:.1f}s ({total_time/60:.2f} min)")
    save_checkpoints()
    persist_norm_stats()
    print(f"Models saved to {MODEL_SAVE_DIR}/")

    artifact = wandb.Artifact("model-checkpoints", type="model")
    artifact.add_file(os.path.join(MODEL_SAVE_DIR, "denoiser.pt"))
    artifact.add_file(os.path.join(MODEL_SAVE_DIR, "q_estimator.pt"))
    if ema_denoiser is not None and ema_estimator is not None:
        artifact.add_file(os.path.join(MODEL_SAVE_DIR, "denoiser_ema.pt"))
        artifact.add_file(os.path.join(MODEL_SAVE_DIR, "q_estimator_ema.pt"))
    wandb.log_artifact(artifact)
    wandb.summary["total_training_time_min"] = total_time / 60
    wandb.finish()


if __name__ == "__main__":
    main()
