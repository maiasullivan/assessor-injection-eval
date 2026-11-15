"""Headline figure: attack success on essays vs CVs, every model x attack x defence.

    uv run python scripts/fig_overview.py   # needs results/asr.csv and results/cv/asr.csv

Writes results/fig0_overview.png. One row per model x attack; hollow dot = essays, filled = CVs,
a grey line joins the two domains so the change between them is the thing you read.
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D

from analyse import ATTACKS, COLORS, DEFENCES, MODELS, OUT, style


def main() -> None:
    both = pd.concat([pd.read_csv(OUT / "asr.csv").assign(domain="essay"),
                      pd.read_csv(OUT / "cv" / "asr.csv").assign(domain="cv")])
    rows = [(m, a) for m in MODELS for a in ATTACKS]
    y = np.arange(len(rows))[::-1].astype(float)
    y -= np.repeat(np.arange(len(MODELS)), len(ATTACKS)) * 0.6  # gap between model groups
    fig, axes = plt.subplots(1, 2, figsize=(10, 5.6), sharey=True)
    for ax, d in zip(axes, DEFENCES):
        style(ax)
        ax.yaxis.grid(False)
        ax.xaxis.grid(True, color="#e6e6e1", linewidth=0.8)
        ax.axvline(0, color="#55554f", lw=0.8)
        for yi, (m, a) in zip(y, rows):
            g = both[(both.model == m) & (both.attack == a) & (both.defence == d)].set_index("domain")
            e, c = g.loc["essay"], g.loc["cv"]
            ax.plot([e.asr, c.asr], [yi, yi], color="#c9c9c4", lw=2, zorder=1, solid_capstyle="round")
            for r, filled, dy in [(e, False, 0.12), (c, True, -0.12)]:
                ax.plot([r.ci_low, r.ci_high], [yi + dy] * 2, color=COLORS[m], lw=1, alpha=0.5, zorder=2)
            ax.scatter(e.asr, yi + 0.12, s=60, facecolor="#fcfcfb", edgecolor=COLORS[m], linewidth=1.8, zorder=3)
            ax.scatter(c.asr, yi - 0.12, s=60, color=COLORS[m], edgecolor="#fcfcfb", linewidth=1.2, zorder=3)
        ax.set_xlim(-0.15, 1.02)
        ax.set_ylim(y.min() - 0.6, y.max() + 0.95)
        ax.set_xlabel("Attack success rate (weak items rated pass/good fit, attacked − clean)", fontsize=9, color="#55554f")
        ax.set_title("No defence" if d == "none" else "Spotlighting", fontsize=10, color="#22221f", loc="left", pad=10)
    axes[0].set_yticks(y, [a for _, a in rows])
    for k, m in enumerate(MODELS):  # model name above its group, in ink (the dots carry the colour)
        axes[0].text(-0.15, y[k * len(ATTACKS)] + 0.6, MODELS[m], fontsize=9, color="#22221f", fontweight="bold",
                     ha="left", va="center", transform=axes[0].transData)
    handles = [Line2D([], [], ls="", marker="o", ms=7, mfc="#fcfcfb", mec="#55554f", mew=1.6),
               Line2D([], [], ls="", marker="o", ms=7, mfc="#55554f", mec="#fcfcfb")]
    fig.legend(handles, ["essays (ASAP, pass = 4+ of 6)", "CVs (Djinni, good fit = 4+ of 5)"], frameon=False, fontsize=9,
               ncol=2, loc="upper left", bbox_to_anchor=(0.005, 0.96))
    fig.suptitle("Same attacks, two domains: how often a weak item is pushed over the line (95% CI)",
                 fontsize=11, color="#22221f", x=0.01, y=0.995, ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(OUT / "fig0_overview.png", dpi=200, facecolor="#fcfcfb")
    plt.close(fig)
    print(f"wrote {OUT / 'fig0_overview.png'}")


if __name__ == "__main__":
    main()
