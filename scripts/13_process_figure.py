"""Draw the deployment loop used in report section 4.5 (no data; a schematic of the pipeline in
src/camo/drift and src/camo/selftrain). Writes outputs/figures/fig_process.png."""
from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from _common import ROOT
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

STEPS = [
    ("Score new reviews", "deployed model on the\ncurrent graph"),
    ("Monitor (label-free)", "PSI / KS of text\nconsistency vs calibrated\nthresholds"),
    ("Alarm", "2 flagged windows\nin a row"),
    ("Analyst labels", "100 probe reviews\nevery window"),
    ("Challenger", "retrain with quota-\nbased pseudo-labels;\nbias audit"),
    ("Promotion guard", "promote only if it wins\nin ≥ 90% of bootstrap\nresamples of the probe"),
]


def main():
    # two rows read as a cycle: top row left to right, bottom row right to left, back up to the start
    fig, ax = plt.subplots(figsize=(7.2, 3.3))
    ax.set_xlim(0, 3); ax.set_ylim(0, 2.05); ax.axis("off")
    w, h = 0.8, 0.72
    pos = [(0, 1), (1, 1), (2, 1), (2, 0), (1, 0), (0, 0)]   # (column, row) for each step
    col = "#33507a"
    for i, ((title, body), (c, r)) in enumerate(zip(STEPS, pos)):
        x, y = c + 0.1, r + 0.18
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02",
                                    fc="#fdf1e6" if i == 3 else "#eef3fb", ec=col, lw=1.2))
        ax.text(x + w / 2, y + h - 0.16, f"{i + 1}. {title}", ha="center", va="center", fontsize=9, weight="bold")
        ax.text(x + w / 2, y + 0.26, body, ha="center", va="center", fontsize=7.8)
    arrow = dict(arrowstyle="-|>", mutation_scale=12, color=col, lw=1.2)
    for c in (0, 1):                                   # top row, left to right
        ax.add_patch(FancyArrowPatch((c + 0.9, 1.54), (c + 1.1, 1.54), **arrow))
    ax.add_patch(FancyArrowPatch((2.5, 1.16), (2.5, 0.92), **arrow))   # alarm -> analyst labels
    for c in (2, 1):                                   # bottom row, right to left
        ax.add_patch(FancyArrowPatch((c + 0.1, 0.54), (c - 0.1, 0.54), **arrow))
    ax.add_patch(FancyArrowPatch((0.5, 0.92), (0.5, 1.16), **arrow))   # promotion -> scoring
    ax.text(0.53, 1.04, "promoted model replaces the incumbent", ha="left", va="center", fontsize=7.3, style="italic")
    ax.text(2.47, 1.04, "alarm uses the labels so far", ha="right", va="center", fontsize=7.3, style="italic")
    fig.tight_layout()
    out = ROOT / "outputs" / "figures" / "fig_process.png"
    fig.savefig(out, dpi=200, bbox_inches="tight"); plt.close(fig)
    print(out)


if __name__ == "__main__":
    main()
