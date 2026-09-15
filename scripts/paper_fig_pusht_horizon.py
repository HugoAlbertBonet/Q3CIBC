"""Paper figure: Push-T IoU against per-step inference cost, across receding horizons.

Recovered from the session that first drew figures/pusht_horizon_sweep (paper repo
commits f2f2a3f, 35b5514) and kept here so the figure can be regenerated. Colour
is the method (the paper palette in utils/plot_style.py, identical in every figure);
dash pattern and marker separate the variants of a method. Set in TeX Gyre Pagella
(assets/fonts), as the published figure. No background grid; the legend sits to the
right of the axes in a single column, sized for a full column width (\linewidth).

    uv run python scripts/paper_fig_pusht_horizon.py --out results/paper_figures/pusht_horizon_sweep
"""
import argparse
import collections
import csv
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.ticker import NullFormatter, ScalarFormatter

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from utils.plot_style import C_BC, C_DP, C_IBC, C_WIFI  # noqa: E402

FONTS = ROOT / "assets/fonts"
SWEEP = ROOT / "results/pusht/plots_normalized/coverage_avg_vs_inference_time_horizon.csv"
BEST = ROOT / "results/pusht/plots_normalized/coverage_avg_best_iters_position_balanced.csv"

# IBC DFO-10 is absent from the sweep file because at h >= 2 it covers 5 start
# positions rather than 8, so it fails that file's balance rule. Timings are the
# per-horizon means from inference_speed.csv (cuda rows).
IBC10_MS = {1: 11.15, 2: 5.60, 4: 2.81, 8: 1.41}

STYLE = {  # key -> label, colour, linestyle, marker, linewidth, z
    ("bc",  "deterministic",   "0"):   ("BC",             C_BC,   "-",                        "o", 1.1, 2),
    ("dp",  "ddim",            "5"):   ("DP DDIM-5",      C_DP,   ":",                        "s", 1.1, 2),
    ("dp",  "ddim",            "10"):  ("DP DDIM-10",     C_DP,   "--",                       "D", 1.1, 2),
    ("dp",  "ddim",            "25"):  ("DP DDIM-25",     C_DP,   "-.",                       "p", 1.1, 2),
    ("dp",  "ddpm",            "100"): ("DP DDPM-100",    C_DP,   (0, (3, 1, 1, 1, 1, 1)),    "h", 1.1, 2),
    ("ibc", "dfo",             "5"):   ("IBC DFO-5",      C_IBC,  "--",                       "v", 1.3, 3),
    ("ibc", "dfo",             "10"):  ("IBC DFO-10",     C_IBC,  "-",                        "^", 1.3, 3),
    ("q3c", "argmax_fallback", "5"):   ("WiFI-BC argmax", C_WIFI, "--",                       "P", 1.7, 4),
    ("q3c", "dfo",             "5"):   ("WiFI-BC DFO-5",  C_WIFI, "-",                        "*", 1.8, 5),
}
ORDER = [("bc", "deterministic", "0"),
         ("dp", "ddim", "5"), ("dp", "ddim", "10"), ("dp", "ddim", "25"), ("dp", "ddpm", "100"),
         ("ibc", "dfo", "5"), ("ibc", "dfo", "10"),
         ("q3c", "argmax_fallback", "5"), ("q3c", "dfo", "5")]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", type=Path, default=ROOT / "results/paper_figures/pusht_horizon_sweep",
                    help="output path without extension (.pdf and .png are written)")
    args = ap.parse_args()

    for f in ("regular", "italic", "bold"):
        font_manager.fontManager.addfont(str(FONTS / f"texgyrepagella-{f}.otf"))
    fam = font_manager.FontProperties(fname=str(FONTS / "texgyrepagella-regular.otf")).get_name()

    data = collections.defaultdict(list)
    for r in csv.DictReader(open(SWEEP)):
        # "yes (spliced)" is a plotted row too -- an exact-match test on "yes" drops
        # the IBC h=1 point, which is the only spliced entry in the file.
        if not r["plotted"].strip().startswith("yes"):
            continue
        k = (r["algorithm"], r["inference"], r["refine_iters"])
        if k in STYLE:
            data[k].append((float(r["ms_per_step"]), float(r["mean"]), float(r["sem"])))
    for r in csv.DictReader(open(BEST)):
        if (r["algorithm"], r["inference"], r["refine_iters"]) == ("ibc", "dfo", "10"):
            h = int(r["exec_horizon"])
            data[("ibc", "dfo", "10")].append((IBC10_MS[h], float(r["mean"]), float(r["sem"])))

    plt.rcParams.update({
        "font.family": fam, "font.size": 8,
        "axes.edgecolor": "#444444", "axes.linewidth": 0.7,
        "xtick.color": "#333333", "ytick.color": "#333333",
        "axes.labelcolor": "#000000", "pdf.fonttype": 42,
    })

    # Column-wide (3.5 in) with the legend to the right of the axes, one entry per row.
    fig, ax = plt.subplots(figsize=(3.5, 2.5))
    for k in ORDER:
        lab, col, ls, mk, lw, z = STYLE[k]
        pts = sorted(data[k])
        if not pts:
            continue
        xs, ys, es = zip(*pts)
        ax.errorbar(xs, ys, yerr=es, color=col, ecolor=col, linestyle=ls, marker=mk,
                    markersize=7.5 if mk == "*" else 4.6, linewidth=lw, elinewidth=0.6,
                    capsize=1.5, markeredgecolor="white", markeredgewidth=0.4,
                    label=lab, zorder=z, alpha=0.97)

    ax.set_xscale("log")
    ax.set_xlabel("Inference time per step (ms)")
    ax.set_ylabel("IoU")
    ax.set_xticks([1, 2, 5, 10, 20])
    ax.xaxis.set_major_formatter(ScalarFormatter())
    ax.xaxis.set_minor_formatter(NullFormatter())
    ax.tick_params(axis="x", which="minor", bottom=False)
    ax.grid(False)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.tick_params(labelsize=7.5)
    ax.legend(loc="center left", bbox_to_anchor=(1.02, 0.5), ncol=1,
              fontsize=6.3, frameon=False, handlelength=2.2,
              handletextpad=0.45, borderaxespad=0.0, labelspacing=0.55)

    fig.tight_layout(pad=0.3)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(f"{args.out}.{ext}", bbox_inches="tight", dpi=300)
    for k in ORDER:
        print(f"  {STYLE[k][0]:<16} {len(data[k])} pts  {[(round(a, 2), round(b, 3)) for a, b, _ in sorted(data[k])]}")
    print(f"-> {args.out}.{{pdf,png}}  font: {fam}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
