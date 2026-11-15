"""Analyse the wording follow-up (scripts/run_variants.py): ASR per wording on all 20 weak items.

    uv run python scripts/analyse_variants.py              # essays: logs/variants/ -> results/
    uv run python scripts/analyse_variants.py --domain cv  # CVs: logs/variants_cv/ -> results/cv/

Writes variants_asr.csv (and prints it). The clean pass rate of each item comes from the main
matrix (same model, defence, items and epochs), so ASR = attacked - clean as in asr.csv.
Figures: fig_variants.png (ASR per wording, 95% CI) and fig_variant_grid.png (item x wording:
is vulnerability a property of the wording or of the item?).
"""

import argparse

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap

from analyse import COLORS, DISCLOSED, LOGS, MODELS, OUT, PASS_SCORE, boot_ci, load, per_essay_means, style

DOMAINS = {  # variant logs, clean-baseline logs, output dir, noun
    "essay": (LOGS.parent / "variants", LOGS, OUT, "essays"),
    "cv": (LOGS.parent / "variants_cv", LOGS.parent / "matrix_cv", OUT / "cv", "candidates"),
}


def per_item(att: pd.DataFrame, clean: pd.DataFrame) -> pd.DataFrame:
    e = (att.groupby(["model", "defence", "attack", "variant", "essay_id"], as_index=False)
            .agg(passed=("pred", lambda x: (x.dropna() >= PASS_SCORE).mean()), pred=("pred", "mean"),
                 disclosed=("disclosed", "sum"), hits=("pred", lambda x: int((x.dropna() >= PASS_SCORE).sum())),
                 years=("experience_years", "first")))
    return e.merge(clean, on=["model", "defence", "essay_id"], suffixes=("", "_clean"))


def asr_table(e: pd.DataFrame) -> pd.DataFrame:
    out = []
    for (m, d, a, v), g in e.groupby(["model", "defence", "attack", "variant"]):
        diff = (g.passed - g.passed_clean).to_numpy(float)
        lo, hi = boot_ci(diff)
        out.append({"model": m, "defence": d, "attack": a, "variant": v, "n_weak": len(g),
                    "attacked_rate": g.passed.mean(), "clean_rate": g.passed_clean.mean(), "asr": diff.mean(),
                    "ci_low": lo, "ci_high": hi, "essays_flipped": int((g.passed > 0).sum()),
                    "disclosed": f"{int(g.disclosed.sum())}/{int(g.hits.sum())}"})
    return pd.DataFrame(out)


def fig_variants(t: pd.DataFrame, path, noun: str) -> None:
    """ASR per wording: one panel per (defence, attack), models side by side within each wording."""
    attacks = [a for a in ["direct", "fake_note", "keyword"] if a in set(t.attack)]
    defences = [d for d in ["none", "spotlight"] if d in set(t.defence)]
    models = [m for m in MODELS if m in set(t.model)]
    fig, axes = plt.subplots(len(defences), len(attacks), figsize=(3.4 * len(attacks), 2.9 * len(defences) + 0.5),
                             sharey=True, squeeze=False)
    off = np.linspace(-0.22, 0.22, len(models))
    for r, d in enumerate(defences):
        for c, a in enumerate(attacks):
            ax = axes[r, c]
            style(ax)
            ax.axhline(0, color="#55554f", lw=0.8)
            for i, m in enumerate(models):
                g = t[(t.model == m) & (t.defence == d) & (t.attack == a)].sort_values("variant")
                if g.empty:
                    continue
                ax.errorbar(g.variant + off[i], g.asr, yerr=[g.asr - g.ci_low, g.ci_high - g.asr], fmt="none",
                            ecolor=COLORS[m], elinewidth=1.4, capsize=0, alpha=0.6)
                ax.scatter(g.variant + off[i], g.asr, s=46, color=COLORS[m], edgecolor="#fcfcfb", linewidth=1.5,
                           zorder=3, label=MODELS[m])
            ax.set_xticks(range(3), ["wording 0", "wording 1", "wording 2"])
            ax.set_xlim(-0.5, 2.5)
            ax.set_ylim(-0.3, 1.05)
            ax.set_title(f"{a} · defence: {d}", fontsize=10, color="#22221f", loc="left")
            if c == 0:
                ax.set_ylabel(f"ASR (weak {noun}\nrated ≥ 4, attacked − clean)", fontsize=9, color="#55554f")
    handles, labels = [], []
    for ax in axes.flat:  # collect one handle per model (some panels lack a model)
        for h, l in zip(*ax.get_legend_handles_labels()):
            if l not in labels:
                handles.append(h)
                labels.append(l)
    fig.suptitle(f"Every wording on all 20 weak {noun}: attack success per wording (95% CI)",
                 fontsize=11, color="#22221f", x=0.01, y=0.995, ha="left")
    fig.legend(handles, labels, frameon=False, fontsize=9, ncol=len(labels), loc="upper left",
               bbox_to_anchor=(0.005, 0.965), handletextpad=0.2, columnspacing=1.2)
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    fig.savefig(path, dpi=200, facecolor="#fcfcfb")
    plt.close(fig)


def fig_grid(e: pd.DataFrame, path, noun: str, min_flips: int = 3) -> None:
    """Item x wording pass rates for every (model, defence, attack) with at least `min_flips` flipped cells."""
    cells = e.groupby(["model", "defence", "attack"]).filter(lambda g: (g.passed > 0).sum() >= min_flips)
    panels = list(cells.groupby(["model", "defence", "attack"], sort=False).groups)
    order = {m: i for i, m in enumerate(MODELS)}
    panels.sort(key=lambda k: (order[k[0]], k[1], k[2]))
    if not panels:
        return
    sort_key = "years" if e.years.notna().any() else "essay_id"
    items = e.drop_duplicates("essay_id").sort_values([sort_key, "essay_id"]).essay_id.tolist()
    fig, axes = plt.subplots(1, len(panels), figsize=(1.55 * len(panels) + 1.2, 6.2), sharey=True, squeeze=False)
    for ax, (m, d, a) in zip(axes[0], panels):
        g = e[(e.model == m) & (e.defence == d) & (e.attack == a)]
        grid = g.pivot(index="essay_id", columns="variant", values="passed").reindex(items)
        cmap = LinearSegmentedColormap.from_list(m, ["#f1f1ee", COLORS[m]])
        ax.imshow(grid.to_numpy(float), cmap=cmap, vmin=0, vmax=1, aspect="auto")
        ax.set_xticks(range(3), ["0", "1", "2"])
        ax.set_xticks(np.arange(-0.5, 3), minor=True)
        ax.set_yticks(np.arange(-0.5, len(items)), minor=True)
        ax.grid(which="minor", color="#fcfcfb", lw=2)  # surface gap between cells
        ax.tick_params(which="minor", length=0)
        ax.tick_params(colors="#55554f", labelsize=8, length=0)
        for s in ax.spines.values():
            s.set_visible(False)
        ax.set_xlabel("wording", fontsize=8, color="#55554f")
        ax.set_title(f"{MODELS[m]}\n{a}\n{'no defence' if d == 'none' else 'spotlight'}", fontsize=9, color="#22221f")
    if sort_key == "years":
        yrs = e.drop_duplicates("essay_id").set_index("essay_id").years.reindex(items)
        axes[0, 0].set_yticks(range(len(items)), [f"{y:g} yrs" for y in yrs])
    else:
        axes[0, 0].set_yticks(range(len(items)), [""] * len(items))
    axes[0, 0].set_ylabel(f"weak {noun} (one row each)", fontsize=9, color="#55554f")
    fig.suptitle(f"Which weak {noun} were rated ≥ 4 by which wording (dark = every epoch, pale = never)",
                 fontsize=10, color="#22221f", x=0.01, ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(path, dpi=200, facecolor="#fcfcfb")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--domain", choices=DOMAINS, default="essay")
    args = parser.parse_args()
    vlogs, clogs, out, noun = DOMAINS[args.domain]
    models = [m for m in MODELS if (vlogs / m).exists()]
    att, base = load(models, vlogs), load(models, clogs)
    if args.domain == "cv":
        att["true"] = base["true"] = -1  # no human score; groupby would otherwise drop the rows
    clean = per_essay_means(base).query("attack == 'none' and tier == 'weak'")[["model", "defence", "essay_id", "passed"]]
    att["disclosed"] = att.reply.str.contains(DISCLOSED, case=False) & (att.pred >= PASS_SCORE)

    e = per_item(att, clean)
    table = asr_table(e)
    table.round(3).to_csv(out / "variants_asr.csv", index=False)
    fig_variants(table, out / "fig_variants.png", noun)
    fig_grid(e, out / "fig_variant_grid.png", noun)
    pd.set_option("display.width", 160)
    print(table.round(2).to_string(index=False))

    # Are the same items vulnerable across wordings? Count, per item, how many wordings flipped it.
    per = e[e.passed > 0].groupby(["model", "defence", "attack"]).essay_id.value_counts()
    print(f"\n{noun.capitalize()} flipped by k of 3 wordings\n",
          per.groupby(level=[0, 1, 2]).value_counts().sort_index().to_string())


if __name__ == "__main__":
    main()
