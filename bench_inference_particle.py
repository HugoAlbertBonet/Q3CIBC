"""Inference-time benchmark: explicit BC (MSE) vs Diffusion Policy on **particle**.

Sibling of bench_inference_particle16.py (which covers Q3CIBC and IBC+Langevin)
and of bench_inference.py / _pixels / _pen / _kitchen. Same convention: random
weights, forward-pass cost only, architectures sized to the published configs.
Wall-clock depends on tensor shapes and the algorithm graph, not on weight
values, so random and trained weights give identical timings — the only ways
that could break are data-dependent early exits (neither method has any) or
numerical degeneracy stalling cuDNN's kernel choice, which warm-up removes.

Run at two dimensionalities, because the interesting thing about this task is
what happens BETWEEN them:

    n_dim = 2    every method solves it
    n_dim = 16   explicit BC collapses; the implicit and diffusion policies do not

Methods
-------
1. **BC-MSE** — one forward pass, observation to action. No search, no sampling,
   no refinement. This is the cheapest possible policy and the reference point
   the other two have to justify their cost against.

2. **DP DDIM-10** — the denoiser recipe from batches/dpParticleV2.txt
   (dense_resnet head, width 1024, depth 1, time_emb 128, T=100,
   v-prediction), sampled with a 10-step sub-chain. Ten SEQUENTIAL denoiser
   passes per action, so it cannot be batched away.

3. **DP DDPM-100** — the SAME trained denoiser, sampled with the full
   stochastic chain instead: all 100 training timesteps, one denoiser pass
   each. This is the sampler axis in isolation — identical weights, identical
   architecture, 10x the sequential passes — which is why the two DP rows are
   directly comparable and the cost ratio is the whole story.

Success rates quoted below are NOT measured here.

  BC-MSE       IBC paper Figure 6 (digitized), stored in
               results/particle/success_rates.csv as success_rate_mse_paper:
               0.99 at n_dim=2, and 0.03 at n_dim=16 — the collapse that motivates
               implicit policies in the first place.
  DP DDIM-10   our own dpParticleV2 runs (2 seeds, cloud=1), mean +/- std:
               100.0 +/- 0.0 at n_dim=2, 71.0 +/- 7.1 at n_dim=16.
  DP DDPM-100  the dpParticleDDPM re-evaluations of those same checkpoints:
               98.0 +/- 0.0 at n_dim=2, 67.0 +/- 7.1 at n_dim=16.

ARCHITECTURE CAVEAT for BC-MSE
------------------------------
The IBC paper's particle EBM config is documented in this repo as 256x2
(ibc/ibc_dfo_particle_training.py, "Table 7 / mlp_ebm_langevin.gin"). Its
particle *MSE* config is NOT vendored here — the only explicit-BC configs the
repo records are for other tasks (D4RL: ResNetPreAct 2048 wide, 8 dense layers,
per configs/d4rl/mlp_mse_best.gin; pixels: 512x4 MLP + tanh, Appendix D.2).
So the MSE trunk here DEFAULTS to the particle EBM's 256x2 rather than to a
number invented for it, and is exposed as --mse-width / --mse-depth. If you
find the paper's particle MSE config, pass it — the timing scales with it and
the default should not be quoted as "the paper's MSE architecture".

Usage
-----
    uv run --managed-python python bench_inference_particle.py \
        --n-dims 2 16 --num-steps 50 --warmup 5 --device auto
"""

from __future__ import annotations

import argparse
import csv
import statistics
import time
from pathlib import Path

import torch
import torch.nn as nn

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "results" / "particle" / "inference_time_bc_vs_dp.csv"

FRAME_STACK = 2               # particle config
OBS_PER_DIM = 4               # obs is 4*n_dim per frame; stacked -> 8*n_dim
TIMESTEPS = 100               # dpParticleV2: num_train_timesteps
DDIM_STEPS = 10               # the only DDIM setting particle has been scored at

# Success rates measured elsewhere; see the module docstring for provenance.
# (mean, std, n_seeds, provenance). std is over TRAINING seeds, not episodes.
SUCCESS = {
    (2,  "BC-MSE"):       (99.0,  None, None, "IBC paper Fig 6 (digitized)"),
    (16, "BC-MSE"):       (3.0,   None, None, "IBC paper Fig 6 (digitized)"),
    (2,  "DP DDIM-10"):   (100.0, 0.0,  2,    "dpParticleV2, 2 seeds x 50 eps"),
    (16, "DP DDIM-10"):   (71.0,  7.1,  2,    "dpParticleV2, 2 seeds x 50 eps"),
    (2,  "DP DDPM-100"):  (98.0,  0.0,  2,    "dpParticleDDPM reeval, 2 seeds x 50 eps"),
    (16, "DP DDPM-100"):  (67.0,  7.1,  2,    "dpParticleDDPM reeval, 2 seeds x 50 eps"),
}


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--n-dims", type=int, nargs="+", default=[2, 16])
    ap.add_argument("--num-steps", type=int, default=50)
    ap.add_argument("--warmup", type=int, default=5)
    ap.add_argument("--device", default="auto", choices=["auto", "cuda", "cpu"])
    ap.add_argument("--mse-width", type=int, default=256,
                    help="explicit-BC trunk width (default 256, the particle "
                         "config this repo documents; see the docstring)")
    ap.add_argument("--mse-depth", type=int, default=2)
    ap.add_argument("--ddim-steps", type=int, default=DDIM_STEPS)
    ap.add_argument("--ddpm-steps", type=int, default=TIMESTEPS,
                    help="full-chain length for the DDPM row; must equal "
                         "num_train_timesteps (default 100)")
    ap.add_argument("--csv", type=Path, default=OUT)
    return ap.parse_args()


def timeit(fn, device, steps, warmup):
    for _ in range(warmup):
        fn()
    if device.type == "cuda":
        torch.cuda.synchronize()
    s = []
    for _ in range(steps):
        if device.type == "cuda":
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        fn()
        if device.type == "cuda":
            torch.cuda.synchronize()
        s.append((time.perf_counter() - t0) * 1000.0)
    s.sort()
    return statistics.mean(s), s[len(s) // 2], (statistics.stdev(s) if len(s) > 1 else 0.0)


def main() -> int:
    args = parse_args()
    device = torch.device("cuda" if (args.device in ("auto", "cuda")
                                     and torch.cuda.is_available()) else "cpu")
    torch.manual_seed(0)
    from utils.diffusion import build_denoiser, build_diffusion, resolve_dp_params
    from utils.models import _build_backbone

    dp = resolve_dp_params({})
    dp.update(num_train_timesteps=TIMESTEPS, beta_schedule="cosine",
              prediction_type="v", time_emb_dim=128,
              denoiser_network_kind="dense_resnet", denoiser_width=1024,
              denoiser_depth=1, denoiser_use_spectral_norm=False)

    print(f"device={device}   frame_stack={FRAME_STACK}   "
          f"obs = {OBS_PER_DIM}*n_dim*frame_stack\n"
          f"Random weights: timing depends on shapes and the graph, not on "
          f"weight values (see docstring).\n")
    rows = []
    for nd in args.n_dims:
        obs_dim = OBS_PER_DIM * nd * FRAME_STACK
        act_dim = nd
        obs = torch.randn(1, obs_dim, device=device)

        # 1. explicit BC: one forward pass, obs -> action.
        mse_net = _build_backbone(
            input_dim=obs_dim, output_dim=act_dim, network_kind="mlp",
            hidden_dims=[args.mse_width] * args.mse_depth,
            width=args.mse_width, depth=args.mse_depth,
            activation=nn.ReLU, use_spectral_norm=False).to(device).eval()

        # 2. diffusion policy: DDIM-10 over the dpParticleV2 denoiser.
        den = build_denoiser(obs_dim, act_dim, dp, device=device).eval()
        diffusion = build_diffusion(dp, device, (0.0, 1.0))

        @torch.no_grad()
        def bc_mse(net=mse_net):
            return net(obs)

        @torch.no_grad()
        def dp_ddim(den=den, act_dim=act_dim):
            return diffusion.ddim_sample(den, obs, action_dim=act_dim,
                                         num_steps=args.ddim_steps, eta=0.0)

        # Full stochastic chain. ddpm_sample always walks num_train_timesteps,
        # so the chain length is fixed by the diffusion object, not an argument
        # — assert rather than silently time a different chain than the label.
        assert diffusion.num_timesteps == args.ddpm_steps, (
            f"ddpm_sample walks {diffusion.num_timesteps} steps but the DDPM "
            f"row is labelled {args.ddpm_steps}")

        @torch.no_grad()
        def dp_ddpm(den=den, act_dim=act_dim):
            return diffusion.ddpm_sample(den, obs, action_dim=act_dim)

        pm = lambda m: sum(p.numel() for p in m.parameters()) / 1e6
        for name, fn, evals, mod in (("BC-MSE", bc_mse, 1, mse_net),
                                     (f"DP DDIM-{args.ddim_steps}", dp_ddim,
                                      args.ddim_steps, den),
                                     (f"DP DDPM-{args.ddpm_steps}", dp_ddpm,
                                      args.ddpm_steps, den)):
            mean, med, sd = timeit(fn, device, args.num_steps, args.warmup)
            key = (nd, name if name.startswith("BC") else
                   ("DP DDIM-10" if "DDIM" in name else "DP DDPM-100"))
            sr, sr_std, sr_n, src = SUCCESS.get(key, (None, None, None, ""))
            rows.append(dict(n_dim=nd, method=name, obs_dim=obs_dim,
                             action_dim=act_dim, forward_passes=evals,
                             params_m=round(pm(mod), 3),
                             inference_time_ms=round(mean, 4),
                             inference_median_ms=round(med, 4),
                             inference_std_ms=round(sd, 4),
                             success_rate_pct=sr, success_std_pct=sr_std,
                             success_seeds=sr_n, success_source=src))
            srtxt = ("n/a" if sr is None else
                     (f"{sr:.1f}%" if sr_std is None else f"{sr:.1f} +/- {sr_std:.1f}%"))
            print(f"  n_dim={nd:>2}  {name:<13} {mean:8.3f} ms  "
                  f"(median {med:7.3f}, sd {sd:6.3f})   passes={evals:>3}  "
                  f"params={pm(mod):6.3f}M   SR={srtxt}")
        print()

    args.csv.parent.mkdir(parents=True, exist_ok=True)
    with args.csv.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)
    print(f"-> {args.csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
