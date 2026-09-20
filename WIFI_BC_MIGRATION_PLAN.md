# WiFI-BC Public Repo Plan (v2 — lean release)

## 0. Scope change from v1

The first draft of this plan treated the move as an archival reorganization — carry over everything salvageable, curate the rest. That was the wrong target. This repo's job is to let someone else **use the WiFI-BC algorithm** and **reproduce the 7 headline experiments if they want to**, nothing more. Months of hyperparameter-search infrastructure, diagnostic scripts, camera-ready figure regenerators, and rig-specific real-robot code are the authors' process, not the product. v2 cuts hard.

Decisions locked in after discussion:
- No `hyperparam_search*.py` anywhere. Every training script runs standalone from a config file; the shipped config holds the actual best-found hyperparameters, and the README gives one exact command per method × env.
- No real-robot code at all (no `scripts/deploy_pusht_real*.py`, no WidowX training/eval, no calibration assets). The Push-T-real result is reported in the README as a number/video link, nothing runnable.
- No `tests/`, `results/`, LaTeX `paper/`, or `docs/`.
- Envs: the 6 headline simulated/offline tasks (particle, pushing-states, pushing-pixels, pen, kitchen, libero-goal-pixels) plus `point_maze_pillar` (backs a multimodality figure in the main paper body, not an appendix). Everything else (`dummy`, `dummy_bimodal`, `two_choice`, `pushing_multi`, `door`) is dropped.
- `dpq3c_training.py` and `wirefit_q3c_training.py` are dropped — treated as abandoned parallel experiments, not reported ablations.
- `ibc_with_cpsv2_training.py` is dropped too. It was never the source of the paper's IBC baseline anyway (see v1's §0b — it only ever ran on the now-dropped `dummy_bimodal`/`pushing_multi`). The real IBC baseline logic lives inside `hyperparam_search_dfo.py`, which also gets dropped — so a new, plain `baselines/ibc.py` + `training/ibc_training.py` needs to be **extracted** from it (its core loop only imports `QEstimator`, `ObservationNormalizer`, `sample_langevin` — the search/JSONL/argparse machinery around that is what's being stripped). This is real code work, not a file move — tracked as a to-do in §7.

## 1. Final repo structure

```
wifi-bc/
├── README.md                      # install, algorithm summary, one command per method x env, plug-and-play usage
├── LICENSE
├── CITATION.cff
├── pyproject.toml                 # package renamed wifi_bc
├── .gitignore
│
├── config/
│   ├── config.json                        # curated: real best-found hyperparams per env, not defaults
│   └── observation_bounds.json
│
├── wifi_bc/                        # the algorithm itself
│   ├── __init__.py
│   ├── models.py                          # ControlPointGenerator, QEstimator, PixelQEstimator + shared blocks
│   │                                       # (_build_backbone, _DenseResnetBlock, _build_pixel_encoder — reused by baselines/)
│   ├── loss.py                            # InfoNCE / MSE / separation / entropy-KDE
│   ├── normalizations.py                  # ObservationNormalizer, wire-fitting Q-value norm
│   ├── sampling.py                        # uniform + Langevin MCMC (shared with baselines/ibc.py)
│   └── policy.py                          # NEW — plug-and-play WiFIBC class, see §3
│
├── baselines/                      # reference implementations WiFI-BC is compared against
│   ├── diffusion.py                       # ← utils/diffusion.py, DDPM/DDIM backbone (imports wifi_bc.models)
│   ├── consistency.py                     # ← utils/consistency.py, Consistency Policy (imports wifi_bc.models)
│   └── ibc.py                             # NEW — extracted from hyperparam_search_dfo.py, see §0/§7
│
├── envs/                            # ← simulations/, trimmed to the 7 kept tasks
│   ├── base_simulation.py
│   ├── datasets.py                        # ← utils/datasets.py, D4RL/Particle/Pushing/LIBERO/Kitchen loaders
│   ├── libero.py                          # ← utils/libero.py
│   ├── particle_env.py / particle_simulation.py
│   ├── pushing_env.py / pushing_simulation.py
│   ├── pushing_pixels_env.py / pushing_pixels_simulation.py
│   ├── pen_human_v2_simulation.py
│   ├── kitchen_simulation.py
│   ├── libero_goal_pixels_simulation.py           # dropping state-based libero_goal_simulation.py — pixels is the paper's reported variant
│   ├── point_maze_pillar_env.py / point_maze_pillar_simulation.py
│   ├── vis_point_maze_pillar.py                   # ← utils/vis_point_maze_pillar.py
│   ├── plot_style.py                              # ← utils/plot_style.py, kept only because point_maze_pillar's eval plot needs it
│   ├── cp_selection.py / plots.py / run_simulation.py   # generic multi-seed eval entry point
│   ├── run_point_maze_pillar_softmax_eval.py
│   └── ibc_block_pushing/                          # vendored Google IBC pushing env — Apache-2.0, see §6
│
├── training/                       # one standalone script per method, no search wrapper
│   ├── wifi_bc_training.py                # ← combinedv2_cpascounter_training.py (the paper's method)
│   ├── ibc_training.py                    # NEW — plain script around baselines/ibc.py
│   ├── diffusion_policy_training.py
│   ├── consistency_policy_training.py
│   └── bc_mse_training.py
│
├── scripts/
│   ├── download_libero_goal.py            # LIBERO-Goal dataset download
│   ├── extract_libero_object_states.py    # LIBERO-Goal state-based obs prep
│   ├── precompute_libero_goal_embs.py     # precomputes language-goal embeddings LIBERO needs
│   └── setup_libero.sh                    # clones/patches third_party/LIBERO (touch libero/__init__.py fix)
│
└── third_party/LIBERO/              # NOT vendored — setup_libero.sh clones it; documented, gitignored
```

Everything not listed above is **not** part of the release: `ibc/`, `main.py`, `test.py`, `inspect_env.py`, all `*_training.py` variants beyond the 5 listed, `hyperparam_search*.py`, `wirefit_q3c_training.py`, `dpq3c_training.py`, `dummy*`/`two_choice`/`pushing_multi`/`door_human_v2` envs, all of `scripts/` beyond the 4 LIBERO helpers (no real-robot, no diagnostics, no figure regenerators), `analyze_langevin_*.py`, `diagnostic_q_importance.py`, `eval_dfo_sweep.py`, `run_diagnostic_for_checkpoint.sh`, `hyperparams_dfo.sh`, `run_dfo_quick_experiments.py`, `vis_dummy.py`, `vis_two_choice.py`, `d4rl.py` (root and `utils/` copies — dead scratch, not a helper, see §2), `tests/`, `results/`, the LaTeX `paper/` dir, `docs/`, `data/`, `checkpoints/`, `plots/`, `wandb/`, `deploy_logs/`, `deploy_dryrun/`, `batches/`, `slurm_jobs/`, `assets/fonts/` (only needed by dropped figure scripts).

## 2. Answers to specific questions raised

- **`d4rl.py`**: not a helper module. Read its full contents — it's 27 lines with import-time side effects (calls `minari.load_dataset(...)` and prints at module scope, no functions or classes). Root and `utils/` copies are byte-identical. Pure dead scratch — dropped, not moved anywhere.
- **`ibc_policy.py` / `q3c_policy.py` "real-robot checkpoint"**: `ibc_policy.py`'s own docstring says it's the loader for a *trained model's saved weights* (a `.pt` file produced by `scripts/train_pusht_real_ibc.py`) for the physical WidowX arm — shared so the live robot control loop and an offline diagnostic script rebuild the exact same network and DFO inference bit-for-bit. Useless without that specific rig, hence dropped along with the rest of the real-robot stack.
- **`ibc_block_pushing/`**: the vendored *environment* (PyBullet physics sim, robot URDF/OBJ assets, oracle policy) for the "Pushing" task, copied from Google's original IBC repo — it's simulation code, unrelated to the IBC algorithm baseline itself.
- **`scripts/plot_*.py` / `paper_fig_*.py`**: camera-ready figure regenerators hardcoded to this paper's exact fonts/palette/panel layout, parsing `trials.jsonl` and `results/paper_figures/`. Zero reuse value outside reproducing the exact PDF figures — dropped along with `results/`.

## 3. `wifi_bc/policy.py` — the plug-and-play class (new code)

`wifi_bc_training.py` is a config.json-driven training loop (data loading, optimizer, wandb, checkpointing) — not designed to be imported. `policy.py` is a separate, new piece of code: a `WiFIBC` class that wraps a trained `ControlPointGenerator` + `QEstimator` behind a `.act(state) -> action` interface, so someone can drop it into their own robot stack without touching the training script. It should expose the paper's three inference variants:

```python
policy = WiFIBC.from_checkpoint(path, inference_mode="argmax")  # or "dfo" / "langevin"
action = policy.act(state)
```

Implementation-wise this is mostly a thin wrapper around logic that already exists scattered across `combinedv2_cpascounter_training.py`'s eval path and `simulations/run_simulation.py` — the work is extracting and packaging it cleanly, not inventing new math.

## 4. `baselines/ibc.py` + `training/ibc_training.py` — extraction plan (new code)

`hyperparam_search_dfo.py` (1568 lines) is the only place the paper's actual IBC baseline (energy-model + Langevin-train / DFO-inference) lives. Its core algorithm only depends on `wifi_bc.models.QEstimator`, `wifi_bc.normalizations.ObservationNormalizer`, and `wifi_bc.sampling.sample_langevin` — everything else in the file (argparse modes `--run`/`--auto`/`--analyze`, `fcntl`-based JSONL trial locking, per-trial config isolation) is search-infrastructure to strip. Plan:

1. Pull the energy-model definition + Langevin training step + DFO inference routine into `baselines/ibc.py` as plain functions/classes (mirroring how `diffusion.py`/`consistency.py` are already structured as self-contained baseline modules).
2. Write a new `training/ibc_training.py` in the same style as `wifi_bc_training.py` — reads `config/config.json`, no CLI trial machinery, one straight training loop.
3. Verify against the existing `results/hyperparam_search/ibc_dfo_<env>/trials.jsonl` numbers (kept locally for verification, not shipped) that the extracted version reproduces the same success rates before calling this done.

## 5. `config/config.json` — needs real content curation, not just copying

Right now `config_json/config.json` mostly holds defaults; the actual winning hyperparameters per env live in `--fixed-params`/`--params` JSON blobs inside `batches/*.txt`, which aren't being shipped. Before release, pull the best-found values for each of the 6 kept envs × 4 kept methods (WiFI-BC, IBC, DP, BC; Consistency Policy where run) out of the batch files / local trial logs and bake them directly into `config/config.json`'s per-env blocks, so `uv run python training/wifi_bc_training.py` with the shipped config reproduces the reported numbers out of the box. This is the single most important content task for making the "reproduce experiments" claim true.

## 6. Licensing

- `envs/ibc_block_pushing/`: independently verified Apache-2.0 (`block_pushing.py` header: "Copyright 2024 The Reach ML Authors"). No `LICENSE`/`NOTICE` file exists in the directory today — add one.
- `third_party/LIBERO`: not vendored — `scripts/setup_libero.sh` clones it fresh; license stays with upstream, just cite it in the README.
- Top-level project license: none currently exists anywhere in the repo — needs to be added (pick MIT/Apache/BSD).

## 7. Outstanding to-dos before this repo is real

1. Write `wifi_bc/policy.py` (§3).
2. Extract `baselines/ibc.py` + `training/ibc_training.py` from `hyperparam_search_dfo.py` and verify parity (§4).
3. Curate `config/config.json` with real best-found hyperparameters per env (§5).
4. Write the README: install steps, one command per method × env, a short usage example for `WiFIBC.from_checkpoint(...)`, and a results table (numbers only, no CSVs shipped).
5. Rename package `q3cibc` → `wifi_bc` in `pyproject.toml`, update its placeholder description.
6. Add `LICENSE`, `CITATION.cff`, and the `ibc_block_pushing` Apache-2.0 notice.
7. Migration itself is a plain local file copy from this repo into the new folder — no git involved (no init, no history carry-over, no commits as part of the move).
