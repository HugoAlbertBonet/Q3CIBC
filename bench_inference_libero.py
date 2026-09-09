"""Inference-time benchmark for dpq3c vs Q3CIBC vs IBC on **LIBERO-Goal, pixels**.

Sibling of bench_inference_pixels.py — same comparison principle (random
weights, forward-pass cost only, architectures sized to the configs we actually
trained), for the libero_goal_pixels observation: two camera streams at
128x128, frame_stack 1 (in_channels 6), centre-cropped to 116, plus a
proprio+goal conditioning vector, predicting a 16-step chunk of 7-DoF actions
(action_dim 112).

Why this exists
---------------
The libero success table has no cost column, and on this benchmark the spread
between families is enormous — the same gap that makes the pushing_pixels table
readable (IBC scores 100.0 there while spending 12288 energy evaluations per
action against q3c's 30). It matters here specifically because the best libero
result we have is dpq3c with the critic SWITCHED OFF (cloud=1, 91.3 vs q3c's
published 86.7), and the argument for dropping the critic is much stronger if
doing so is also cheaper.

Methods benchmarked
-------------------
The actor is a PixelDiffusionDenoiser (ResNet-18 + GroupNorm + 128
SpatialSoftmax keypoints, DenseResnet head 1024x1, time_emb 128, T=100) and the
critic a PixelQEstimator (DenseResnetValue 1024x1). Six rows:

1. **DP DDIM-10 (cloud=1)** — one sample from a 10-step sub-sampled chain.
   Nothing to rank, so the critic is never called: this is plain diffusion
   policy, and the configuration that scores best on this env.

2. **DP DDPM-100 (cloud=1)** — the same actor walking all 100 training
   timesteps ancestrally. Same weights, same output shape, TEN TIMES the
   sequential denoiser passes. Isolating that cost is the point of this pair:
   the sampler is a free runtime choice, so it should be chosen on the
   speed/quality trade-off rather than by default.

3-4. **dpq3c DDIM-10 / DDPM-100 (cloud=64 + Q)** — the same two samplers
   drawing 64 candidates, then the critic scoring all 64 in ONE batched pass.
   The conv tower runs once for the denoiser and once for the critic, and the
   candidates are width rather than sequential depth, so the gap between these
   and rows 1-2 is what the ranking machinery actually costs.

5. **Q3CIBC** — PixelControlPointGenerator (same encoder, MLP head 512x4) ->
   100 control points, ranked by PixelQEstimator in one batched pass. No
   iterative refinement, matching the deployed recipe.

6. **IBC (DFO)** — PixelQEstimator only, with derivative-free optimisation over
   2048 samples for 100 iterations on cached encoder features (late fusion), per
   the best libero IBC config we ran.

7-9. **Q3CIBC + Langevin-K**, K in {10, 25, 100} — row 5 followed by Langevin
   MCMC on the single argmax control point, against the cached features, which
   is exactly what `inference_langevin_iterations` does on this env (see
   hyperparam_search.LiberoRefinedSimulation). This is the only refinement that
   has ever moved the libero number: at K=25 with lr_init 0.1 it is worth
   +2.9 +/- 3.6 points paired over 9 checkpoints (t=+2.39, p=0.017), while
   K=100 gives it all back (-0.7). Each iteration is a forward AND a backward
   through the value head (Langevin needs dE/da), so unlike DFO the cost per
   iteration cannot be amortised across candidates — which is why K belongs in
   a cost table next to the 2.9 points it buys.

Random weights note
-------------------
Inference wall-clock depends on tensor shapes and the algorithm graph, not on
weight values: random and trained weights give identical FLOPs/second. The only
ways that could break are data-dependent early exits (none of these have any)
or numerical degeneracy stalling cuDNN's algorithm choice, which warm-up
removes. Same reasoning, and same convention, as bench_inference_pixels.py.

Success rates quoted in the CSV are measured elsewhere, NOT here: they are the
seed-averaged numbers from results/hyperparam_search/*/libero_goal_pixels/
trials.jsonl and from libero_goal/standard_results.csv.

CSV output
----------
results/hyperparam_search/combinedv2_cpascounter_training/libero_goal/
inference_time_libero.csv

Usage
-----
    uv run --managed-python --extra libero python bench_inference_libero.py \
        --num-steps 50 --warmup 5 --device auto
"""

from __future__ import annotations

import argparse
import csv
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent
OUT = (ROOT / "results" / "hyperparam_search" / "combinedv2_cpascounter_training"
       / "libero_goal" / "inference_time_libero.csv")

# ── the libero_goal_pixels observation/action contract we actually trained ──
IN_CHANNELS = 6          # 2 cameras x 3 x frame_stack 1
CROP = 116               # image_crop_size; the encoder resizes to 128x128
ENC_HW = 128
ACTION_DIM = 112         # 7 DoF x action_chunk 16
COND_DIM = 81            # proprio + goal embedding (libero schema)
GOAL_DIM = 72
ENC_KIND, ENC_NORM, ENC_KP = "resnet18", "gn", 128
VALUE_WIDTH, VALUE_BLOCKS = 1024, 1
DENOISER_WIDTH, DENOISER_DEPTH, TIME_EMB, TIMESTEPS = 1024, 1, 128, 100
DDIM_STEPS = 10
DPQ3C_CLOUD = 64
Q3C_CONTROL_POINTS = 100
Q3C_CP_WIDTH, Q3C_CP_DEPTH = 512, 4
IBC_SAMPLES, IBC_ITERS = 2048, 100
# Inference-time Langevin refinement of the chosen control point. The lr/clip/
# noise triple is the one the winning reeval used (q3cLangevinV2, lr_init 0.1);
# the iteration counts are the three that were actually scored.
LANGEVIN_ITERS = (10, 25, 100)
LANGEVIN_LR_INIT, LANGEVIN_LR_FINAL = 0.1, 1e-5
LANGEVIN_DECAY_POWER, LANGEVIN_DELTA_CLIP, LANGEVIN_NOISE = 2.0, 0.1, 0.1

# Seed-averaged success rates, measured elsewhere (see the module docstring).
# Seed-averaged success rates, measured elsewhere (see the module docstring).
# None = that cell has never been evaluated, and is left blank rather than
# filled with a number from a different encoder.
#   DDIM rows      pretrained ImageNet trunk, cloud=1 / cloud=64, n=50 eval eps
#   DDPM rows      never run on the pretrained checkpoints. On the SCRATCH
#                  checkpoints DDPM at cloud=1 scored 75.2 +/- 7.3 (5 runs)
#                  against DDIM-10's 81.2 +/- 4.1, so DDIM is ~6 points better
#                  there — but a scratch number does not belong in a column
#                  whose other rows are pretrained.
#   Q3CIBC         film fusion + scratch trunk, the ablation reference (n=3).
#                  NOT the published 86.7, which was a different (ImageNet,
#                  chunk 8) configuration.
SUCCESS = {
    "DP DDIM-10 (cloud=1)": (91.3, 3.1, 3),
    "DP DDPM-100 (cloud=1)": (None, None, None),
    "dpq3c DDIM-10 (cloud=64 + Q)": (88.7, 4.2, 3),
    "dpq3c DDPM-100 (cloud=64 + Q)": (None, None, None),
    "Q3CIBC (film, scratch)": (92.7, 3.1, 3),
    "IBC (DFO 2048x100)": (42.0, 8.8, 3),
    # Langevin rows: the q3cLangevin/q3cLangevinV2 re-evaluations, mean +/- std
    # over the SAME 9 q3c checkpoints (so they are paired with each other and
    # with their own 0-iteration baseline of 89.6 +/- 5.4). These 9 span several
    # q3c configurations, which is why the baseline here is not the 92.7 of the
    # single ablation-reference row above.
    "Q3CIBC + Langevin-10": (90.2, 4.4, 9),
    "Q3CIBC + Langevin-25": (92.4, 2.8, 9),
    "Q3CIBC + Langevin-100": (88.9, 4.8, 9),
}
# The paired baseline for the three Langevin rows (same 9 checkpoints, 0 iters).
LANGEVIN_BASELINE = (89.6, 5.4, 9)


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--num-steps", type=int, default=50, help="timed calls per method")
    ap.add_argument("--warmup", type=int, default=5)
    ap.add_argument("--device", default="auto", choices=["auto", "cuda", "cpu"])
    ap.add_argument("--csv", type=Path, default=OUT)
    return ap.parse_args()


def timeit(fn, device: torch.device, steps: int, warmup: int) -> tuple[float, float]:
    for _ in range(warmup):
        fn()
    if device.type == "cuda":
        torch.cuda.synchronize()
    samples = []
    for _ in range(steps):
        if device.type == "cuda":
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        fn()
        if device.type == "cuda":
            torch.cuda.synchronize()
        samples.append((time.perf_counter() - t0) * 1000.0)
    s = sorted(samples)
    return float(np.mean(s)), s[len(s) // 2]


def params_m(*mods) -> float:
    seen, tot = set(), 0
    for m in mods:
        for p in m.parameters():
            if id(p) not in seen:
                seen.add(id(p)); tot += p.numel()
    return tot / 1e6


def main() -> int:
    args = parse_args()
    device = torch.device("cuda" if (args.device in ("auto", "cuda") and torch.cuda.is_available())
                          else "cpu")
    torch.manual_seed(0)

    from utils.models import PixelQEstimator, PixelControlPointGenerator
    from utils.diffusion import build_dpq3c_denoiser, build_diffusion, resolve_dp_params
    from utils.sampling import sample_langevin

    enc = dict(encoder_target_height=ENC_HW, encoder_target_width=ENC_HW,
               encoder_kind=ENC_KIND, encoder_pretrained=False,
               encoder_num_kp=ENC_KP, encoder_norm_kind=ENC_NORM,
               encoder_per_camera=False)

    obs = torch.randint(0, 255, (1, IN_CHANNELS, CROP, CROP), dtype=torch.uint8, device=device)
    cond = torch.randn(1, COND_DIM, device=device)

    q = PixelQEstimator(action_dim=ACTION_DIM, in_channels=IN_CHANNELS,
                        value_width=VALUE_WIDTH, value_num_blocks=VALUE_BLOCKS,
                        cond_dim=COND_DIM, cond_fusion="concat", goal_dim=GOAL_DIM,
                        **enc).to(device).eval()
    q._cond = cond

    dp = resolve_dp_params({})
    dp.update(num_train_timesteps=TIMESTEPS, beta_schedule="cosine", prediction_type="v",
              time_emb_dim=TIME_EMB, denoiser_network_kind="dense_resnet",
              denoiser_width=DENOISER_WIDTH, denoiser_depth=DENOISER_DEPTH,
              denoiser_use_spectral_norm=False)
    den = build_dpq3c_denoiser(ACTION_DIM, IN_CHANNELS, dp, cond_dim=COND_DIM,
                               encoder_feature_dim=256, device=device, **enc).eval()
    den._cond = cond
    diffusion = build_diffusion(dp, device, (-1.0, 1.0))
    head = den.denoiser

    cp = PixelControlPointGenerator(
        output_dim=ACTION_DIM, control_points=Q3C_CONTROL_POINTS,
        hidden_dims=[Q3C_CP_WIDTH] * Q3C_CP_DEPTH, action_bounds=(-1.0, 1.0),
        network_kind="mlp", width=Q3C_CP_WIDTH, depth=Q3C_CP_DEPTH,
        use_spectral_norm=False, in_channels=IN_CHANNELS, cond_dim=COND_DIM,
        cond_fusion="concat", goal_dim=GOAL_DIM, **enc).to(device).eval()
    cp._cond = cond

    @torch.no_grad()
    def dpq3c(cloud: int, sampler: str = "ddim"):
        # Encoder ONCE, broadcast over the cloud; only the small head runs in
        # the sampling loop. A cloud of one never calls the critic.
        #
        # DDIM-10 sub-samples the chain to 10 steps; DDPM walks all
        # num_train_timesteps (100) ancestrally, so it is 10x the sequential
        # denoiser passes for the same candidate count. That ratio is the whole
        # reason the sampler belongs in a cost table.
        f = den.encode(obs).expand(cloud, -1)
        if sampler == "ddim":
            cps = diffusion.ddim_sample(head, f, action_dim=ACTION_DIM,
                                        num_steps=DDIM_STEPS, eta=0.0)
        else:
            cps = diffusion.ddpm_sample(head, f, action_dim=ACTION_DIM)
        cps = cps.unsqueeze(0)
        if cloud > 1:
            qf = q.encode(obs)
            return q.score(qf, cps).squeeze(-1).argmax(dim=1)
        return cps

    @torch.no_grad()
    def q3c():
        cps = cp(obs)
        qf = q.encode(obs)
        return q.score(qf, cps).squeeze(-1).argmax(dim=1)

    @torch.no_grad()
    def ibc_dfo():
        # Late fusion: encode once, then run the whole DFO loop against the
        # cached features. This is the cheapest faithful form of IBC inference.
        f = q.encode(obs)
        x = torch.rand(1, IBC_SAMPLES, ACTION_DIM, device=device) * 2 - 1
        std = 0.33
        for i in range(IBC_ITERS):
            s = q.score(f, x).squeeze(-1)
            idx = torch.multinomial(torch.softmax(s.squeeze(0), -1), IBC_SAMPLES,
                                    replacement=True)
            x = x[:, idx, :]
            if i < IBC_ITERS - 1:
                x = (x + torch.randn_like(x) * std).clamp(-1, 1)
                std *= 0.5
        return q.score(f, x).squeeze(-1).argmax(dim=1)

    act_lo = torch.full((ACTION_DIM,), -1.0, device=device)
    act_hi = torch.full((ACTION_DIM,), 1.0, device=device)

    def q3c_langevin(iters: int):
        """Row 5, then Langevin on the winning control point.

        Mirrors LiberoRefinedSimulation: encode once, rank the cloud under
        no_grad, then run the chain against the cached features. The chain is
        NOT under no_grad — sample_langevin differentiates the energy w.r.t.
        the action — so this row pays a backward pass per iteration.
        """
        with torch.no_grad():
            cps = cp(obs)
            f = q.encode(obs)
            best = cps[0, q.score(f, cps).squeeze(-1).argmax(dim=1)[0], :].view(1, 1, -1)

        def neg_energy(_obs_lv, actions_lv):
            return -q.score(f, actions_lv).squeeze(-1)

        return sample_langevin(
            energy_function=neg_energy, observations=f, num_samples=1,
            action_min=act_lo, action_max=act_hi, num_iterations=iters,
            lr_init=LANGEVIN_LR_INIT, lr_final=LANGEVIN_LR_FINAL,
            polynomial_decay_power=LANGEVIN_DECAY_POWER,
            delta_action_clip=LANGEVIN_DELTA_CLIP, noise_scale=LANGEVIN_NOISE,
            initial_actions=best, device=device)

    # sample_langevin flips requires_grad on the action, not on the model, but
    # the real path freezes the critic around the chain; do the same so the
    # timed graph matches (no parameter grads accumulated).
    for _p in q.parameters():
        _p.requires_grad_(False)

    methods = [
        ("DP DDIM-10 (cloud=1)", lambda: dpq3c(1, "ddim"), DDIM_STEPS, 0,
         params_m(den)),
        ("DP DDPM-100 (cloud=1)", lambda: dpq3c(1, "ddpm"), TIMESTEPS, 0,
         params_m(den)),
        ("dpq3c DDIM-10 (cloud=64 + Q)", lambda: dpq3c(DPQ3C_CLOUD, "ddim"),
         DDIM_STEPS, DPQ3C_CLOUD, params_m(den, q)),
        ("dpq3c DDPM-100 (cloud=64 + Q)", lambda: dpq3c(DPQ3C_CLOUD, "ddpm"),
         TIMESTEPS, DPQ3C_CLOUD, params_m(den, q)),
        ("Q3CIBC (film, scratch)", q3c, 0, Q3C_CONTROL_POINTS, params_m(cp, q)),
        ("IBC (DFO 2048x100)", ibc_dfo, 0, IBC_SAMPLES * IBC_ITERS, params_m(q)),
    ] + [
        # scoring_evals counts Q calls: the 100-CP ranking pass plus one per
        # Langevin iteration (each of which is also a backward).
        (f"Q3CIBC + Langevin-{k}", (lambda k=k: q3c_langevin(k)), 0,
         Q3C_CONTROL_POINTS + k, params_m(cp, q))
        for k in LANGEVIN_ITERS
    ]

    print(f"device={device}  obs={tuple(obs.shape)}  action_dim={ACTION_DIM}  "
          f"cond_dim={COND_DIM}\nRandom weights: wall-clock depends on shapes and "
          f"the graph, not weight values (see module docstring).\n")
    rows = []
    for name, fn, denoiser_passes, scoring_evals, pm in methods:
        mean_ms, med_ms = timeit(fn, device, args.num_steps, args.warmup)
        sr, sd, ns = SUCCESS.get(name, (None, None, None))
        rows.append(dict(method=name,
                         success_rate_pct=sr, success_rate_std_pct=sd, num_seeds=ns,
                         denoiser_passes=denoiser_passes, scoring_evals=scoring_evals,
                         params_m=round(pm, 2),
                         inference_time_ms=round(mean_ms, 3),
                         inference_time_median_ms=round(med_ms, 3)))
        srtxt = "n/a" if sr is None else (f"{sr:.1f}" + (f" +/- {sd:.1f}" if sd is not None else ""))
        print(f"  {name:32s} {mean_ms:8.2f} ms   scoring_evals={scoring_evals:>6}  "
              f"params={pm:6.2f}M   SR={srtxt}")

    args.csv.parent.mkdir(parents=True, exist_ok=True)
    with args.csv.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)
    print(f"\n-> {args.csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
