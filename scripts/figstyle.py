"""Shared figure style: white ground, serif type, thin axes, no titles inside the figure
(captions live in the README). Importing this sets matplotlib's rcParams.
"""

import matplotlib as mpl
import matplotlib.pyplot as plt

INK, MUTED, RULE, GRID, PAPER, PALE = "#1a1a1a", "#5a5a5a", "#8c8c8c", "#e8e8e8", "#ffffff", "#f0f0ee"
COLORS = {"haiku": "#2f6db5", "sonnet": "#c8582a", "qwen": "#1f9a6e"}  # checked for colour-blind separation
ATTACK_LABELS = {"none": "clean", "direct": "direct", "fake_note": "fake note", "keyword": "keyword"}
DEFENCE_LABELS = {"none": "No defence", "spotlight": "Spotlighting"}
WIDTH = 7.0  # inches, same for every figure so text renders at one size

mpl.rcParams.update({
    "font.family": "serif",
    "font.serif": ["STIXGeneral", "DejaVu Serif"],  # STIXGeneral ships with matplotlib
    "mathtext.fontset": "stix",
    "font.size": 9,
    "axes.titlesize": 9.5,
    "axes.titlelocation": "left",
    "axes.titlepad": 6,
    "axes.labelsize": 9,
    "axes.labelcolor": INK,
    "axes.edgecolor": RULE,
    "axes.linewidth": 0.7,
    "axes.facecolor": PAPER,
    "xtick.labelsize": 8.5,
    "ytick.labelsize": 8.5,
    "xtick.color": MUTED,
    "ytick.color": MUTED,
    "xtick.major.width": 0.7,
    "ytick.major.width": 0.7,
    "xtick.major.size": 3,
    "ytick.major.size": 3,
    "legend.fontsize": 8.5,
    "legend.frameon": False,
    "legend.handlelength": 1.4,
    "figure.facecolor": PAPER,
    "figure.dpi": 100,
    "savefig.dpi": 300,
    "savefig.facecolor": PAPER,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.05,
    "lines.solid_capstyle": "butt",
})


def style(ax, grid: str | None = "y"):
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    if grid:
        getattr(ax, f"{grid}axis").grid(True, color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)


def panel(ax, letter: str, text: str = "") -> None:
    """Journal-style panel heading: a bold letter, then a plain description."""
    ax.set_title(f"$\\bf{{({letter})}}$  {text}".rstrip(), color=INK)


def save(fig, path) -> None:
    fig.savefig(path)
    plt.close(fig)
