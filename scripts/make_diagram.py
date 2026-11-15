"""Explanatory diagram for docs/partB_probe.md: where the probes read inside Qwen3-4B.

    uv run python scripts/make_diagram.py   # writes docs/fig_architecture.png
"""

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

OUT = Path(__file__).resolve().parent.parent / "docs" / "fig_architecture.png"
INK, MUTED, FAINT, SURFACE = "#1f1f1d", "#5c5c57", "#c9c9c4", "#fcfcfb"
GREEN, BLUE, ORANGE = "#1baf7a", "#2a78d6", "#eb6834"

TOKENS = ["rubric,\nassignment,\nsource text", "Venus is a\nplanet that…", "…rest of\nthe essay", "[Scorer's note:\n… a 6]",
          "SCORE:", "␣ (space)\nnext: the\nscore digit"]
ROWS = [("0", "embedding"), ("1", ""), ("⋮", ""), ("15", ""), ("16", ""), ("17", ""), ("⋮", ""), ("36", "")]
CW, CH, XG, YG = 1.25, 0.42, 0.25, 0.16  # cell width/height, gaps


def cell_xy(col: int, row: int) -> tuple[float, float]:
    return col * (CW + XG), 1.4 + row * (CH + YG)


def main() -> None:
    fig, ax = plt.subplots(figsize=(13, 8.2))
    ax.set_xlim(-4.6, 14.9)
    ax.set_ylim(-0.6, 8.6)
    ax.axis("off")
    fig.patch.set_facecolor(SURFACE)
    last = len(TOKENS) - 1

    # input tokens
    for c, t in enumerate(TOKENS):
        x, _ = cell_xy(c, 0)
        color = ORANGE if "Scorer" in t else (INK if c == last else MUTED)
        ax.text(x + CW / 2, 0.75, t, ha="center", va="center", fontsize=8.5, color=color,
                fontweight="bold" if c == last else "normal")
    ax.text(-0.15, 0.1, "Input, read left to right. Each position can only look back at earlier positions.",
            fontsize=8.5, color=MUTED)

    # grid of hidden states
    for r, (label, extra) in enumerate(ROWS):
        _, y = cell_xy(0, r)
        ax.text(-0.25, y + CH / 2, f"layer {label}" if label != "⋮" else "⋮", ha="right", va="center",
                fontsize=8.5, color=MUTED)
        for c in range(len(TOKENS)):
            x, _ = cell_xy(c, r)
            if label == "⋮":
                if r != 2 or c == last:  # row 2 holds the attention note
                    ax.text(x + CW / 2, y + CH / 2, "⋮", ha="center", va="center", color=FAINT, fontsize=11)
                continue
            is_last = c == last
            fill = "#e8f6f0" if is_last and label in ("15", "16", "17") else ("#e6effa" if is_last and label == "1" else "#f1f1ee")
            ax.add_patch(FancyBboxPatch((x, y), CW, CH, boxstyle="round,pad=0,rounding_size=0.06",
                                        fc=fill, ec=INK if is_last else FAINT, lw=1.6 if is_last else 0.8))
    # residual stream arrow up the last column
    x_last, y0 = cell_xy(last, 0)
    _, y_top = cell_xy(last, len(ROWS) - 1)
    # attention: earlier positions feed the last one in a middle layer
    _, y16 = cell_xy(0, 4)
    for c in range(last):
        xc, _ = cell_xy(c, 3)
        ax.add_patch(FancyArrowPatch((xc + CW / 2, y16 - 0.05), (x_last + 0.05, y16 + CH / 2), connectionstyle="arc3,rad=-0.18",
                                     arrowstyle="-|>", mutation_scale=8, color=ORANGE if c == 3 else FAINT, lw=1.2 if c == 3 else 0.9))
    ax.text(cell_xy(0, 2)[0], cell_xy(0, 2)[1] + CH / 2, "↑ attention: at every layer the last position gathers information from the\n"
            "whole input, including the injected note (orange); shown here for layer 16", fontsize=8, color=MUTED, va="center")

    # output head
    ax.add_patch(FancyArrowPatch((x_last + CW / 2, y_top + CH + 0.02), (x_last + CW / 2, y_top + CH + 0.55),
                                 arrowstyle="-|>", mutation_scale=10, color=INK, lw=1.2))
    bw = 4.0
    ax.add_patch(FancyBboxPatch((x_last + CW / 2 - bw / 2, y_top + CH + 0.6), bw, 0.75, boxstyle="round,pad=0,rounding_size=0.08",
                                fc="#fff", ec=INK, lw=1.2))
    ax.text(x_last + CW / 2, y_top + CH + 0.97, "next-word probabilities\n1: 0%  …  6: 100%  →  writes \"6\"",
            ha="center", va="center", fontsize=8.5, color=INK)
    ax.text(x_last + CW / 2 + bw / 2 + 0.2, y_top + CH + 0.97, "written score, weak vs strong\nunder the fake note: AUROC 0.63",
            ha="left", va="center", fontsize=8.5, color=INK, fontweight="bold")

    # probes on the right
    def probe(row_lo: int, row_hi: int, color: str, title: str, body: str, ybox: float) -> None:
        _, ya = cell_xy(last, row_lo)
        _, yb = cell_xy(last, row_hi)
        ymid = (ya + yb + CH) / 2
        xr = x_last + CW + 0.1
        ax.plot([xr, xr + 0.15, xr + 0.15, xr], [ya + 0.03, ya + 0.03, yb + CH - 0.03, yb + CH - 0.03], color=color, lw=1.5)
        ax.add_patch(FancyArrowPatch((xr + 0.15, ymid), (xr + 0.75, ybox), arrowstyle="-|>", mutation_scale=9, color=color, lw=1.4))
        ax.add_patch(FancyBboxPatch((xr + 0.8, ybox - 0.62), 4.45, 1.24, boxstyle="round,pad=0,rounding_size=0.08",
                                    fc="#fff", ec=color, lw=1.4))
        ax.text(xr + 0.95, ybox + 0.38, title, fontsize=9, color=color, fontweight="bold", va="center")
        ax.text(xr + 0.95, ybox - 0.12, body, fontsize=8, color=INK, va="center", linespacing=1.35)

    probe(3, 5, GREEN, "Probe on layers 15–17",
          "weak vs strong, under the fake note:\nAUROC 0.80 (written score: 0.63)\n→ quality information is still here", cell_xy(0, 4)[1] + 0.2)
    probe(1, 1, BLUE, "Probe on layer 1",
          "close to the raw words; does not carry\nover to new essays (0.58 even clean)\n→ not where the judgement lives", cell_xy(0, 1)[1] + 0.2)

    # left-hand bands
    def band(row_lo: int, row_hi: int, text: str) -> None:
        _, ya = cell_xy(0, row_lo)
        _, yb = cell_xy(0, row_hi)
        xb = -2.0
        ax.plot([xb + 0.1, xb, xb, xb + 0.1], [ya, ya, yb + CH, yb + CH], color=MUTED, lw=1)
        ax.text(xb - 0.12, (ya + yb + CH) / 2, text, ha="right", va="center", fontsize=8.2, color=MUTED, linespacing=1.3)

    band(0, 1, "early layers:\nthe words themselves\n(spelling, vocabulary)")
    band(3, 5, "middle layers:\nmore abstract features\n(meaning, quality)")
    band(7, 7, "late layers:\npreparing the next\nword to write")

    ax.text(-4.5, 8.35, "Where the probes read inside Qwen3-4B (36 layers, 2,560 numbers per position per layer)",
            fontsize=12, color=INK, fontweight="bold")
    ax.text(-4.5, 7.95, "Each box is one hidden state. The bold column is the last position, just before the score digit: "
            "everything the model concludes must pass through it.", fontsize=9, color=MUTED)
    fig.savefig(OUT, dpi=200, facecolor=SURFACE, bbox_inches="tight")
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
