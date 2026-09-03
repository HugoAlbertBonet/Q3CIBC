The ablation in `sections/experiments.tex` (paper repo at
`6a8ca3b24200ec26f0fd2d36`) is not an ablation: its points come from a
hyperparameter SEARCH in `results/hyperparam_search/.../trials.jsonl`, so every
other parameter varies with the one being plotted, per-point trial counts range
from 1 to 89, training budgets differ, and means are taken over converged runs
only — a selection effect, given particle-16 diverges in 59% of trials. Replace
it with a controlled study: freeze one reference config per environment (pushing
states = the 3-iteration arm of `batches/pushingH.txt`; particle-16 = trial 231
in its `trials.jsonl`), change one parameter at a time, hold the seed set and
training length fixed, and report every run including divergences as their own
row. Six axes, each a one-at-a-time change from the reference:
`control_points` {1,5,10,20,40,80}; `inference_dfo_iterations` {0,1,3,5,10};
`inference_uniform_proposal` {false,true}; `generator_infonce_weight`
{0,0.05,0.2}; `top_k_control_points` {1,4,8,20}; and `separation_loss`
{separation,entropy} x `separation_weight` {0,0.1}. The last four have never
been ablated and are exactly the contributions the Method section claims;
`inference_uniform_proposal` (already in `hyperparam_search.py`, syntax-checked
but never run — smoke-test it) is the control a reviewer asks for first, since
if uniform candidates match the learned proposal at equal N the method reduces
to "IBC with fewer samples".

Run all six axes on particle-16 (0.7 h/run, 5 seeds, ~115 runs) and
pushing-states (2.3 h/run, 3 seeds, ~69 runs); add axes 1-3 coarsely on
pushing-pixels (12.3 h/run) if budget allows; skip kitchen, pen and LIBERO at
15-39 h/run. First add an eval-only path to `hyperparam_search.py` that
re-evaluates an existing `checkpoint_dir` under new inference parameters — axes
2 and 3 change inference only, so this turns 21 retrains into 3 runs plus cheap
re-evaluations per environment. Deliver: the eval-only path with a smoke test;
one batch file per axis per environment in `batches/` named
`ablN_<env>_<axis>.txt`, in the existing one-command-per-line format runnable by
`./submit_experiments.sh`, each with a header saying what it tests and how to
read the result; then, once results land, one figure per axis plus a table of
every cell with mean, standard deviation, seed count and divergence count.
Traps: commit but never push; `test -e` before writing any file, as
`batches/pushingI.txt` already exists; the benchmark scripts overwrite
hand-curated CSVs (`pen_inference_results.csv`, `kitchen_inference_results.csv`,
`single_target_pixels.csv`, `inference_time_libero.csv`) with rows they cannot
regenerate, so `git checkout -- results/` afterwards and take numbers from
stdout; pushing-DFO and kitchen are compute-bound, so keep every timing on the
one RTX 5070 and run benchmarks serially; the paper builds with tectonic, so
verify it compiles before committing LaTeX.
