"""Paper figure: the task-settings strip (one frame per environment, labelled beneath).

The original generator (paper repo commit b9dc8f9) rendered each frame from its
environment into a scratch directory that no longer exists. To relabel the figure
without changing a single pixel of the frames, the six panels were extracted from
the published figures/environments.pdf into results/paper_figures/environments_panels/
(panel_0.png .. panel_5.png, left to right, already square-cropped) and are laid out
here with the original layout, label size and font (TeX Gyre Pagella, assets/fonts).

    uv run python scripts/paper_fig_environments.py --out results/paper_figures/environments
"""
import argparse
from pathlib import Path

import imageio.v2 as iio
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager

ROOT = Path(__file__).resolve().parents[1]
FONTS = ROOT / "assets/fonts"
PANELS_DIR = ROOT / "results/paper_figures/environments_panels"
LABELS = ["Particle ($n$-D)", "Simulated Pushing", "Adroit Pen", "Franka Kitchen", "LIBERO-Goal", "Push-T (real)"]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", type=Path, default=ROOT / "results/paper_figures/environments",
                    help="output path without extension (.pdf and .png are written)")
    args = ap.parse_args()

    for f in ("regular", "italic", "bold"):
        font_manager.fontManager.addfont(str(FONTS / f"texgyrepagella-{f}.otf"))
    fam = font_manager.FontProperties(fname=str(FONTS / "texgyrepagella-regular.otf")).get_name()

    panels = [iio.imread(PANELS_DIR / f"panel_{i}.png") for i in range(len(LABELS))]
    plt.rcParams.update({"font.family": fam, "pdf.fonttype": 42})
    fig, axes = plt.subplots(1, len(panels), figsize=(7.0, 1.42))
    for ax, img, label in zip(axes, panels, LABELS):
        ax.imshow(img)
        ax.set_xticks([]); ax.set_yticks([])
        for sp in ax.spines.values():
            sp.set_color("#999999"); sp.set_linewidth(0.5)
        ax.set_xlabel(label, fontsize=7.2, color="#000000", labelpad=3)
    fig.subplots_adjust(left=0.003, right=0.997, top=0.99, bottom=0.13, wspace=0.05)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(f"{args.out}.{ext}", bbox_inches="tight", dpi=400)
    print(f"-> {args.out}.{{pdf,png}}  font: {fam}  panels: {[p.shape for p in panels]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
